"""Build per-image manifests for the ProGAN train / val and CNN synth test splits.

Every image is read once in full: SHA-256 of the file bytes, then a full PIL decode
(size, mode, format, decode errors). The 2026-10-31 train manifest was built by reading
progan_train.zip in place from the 7z volumes; the volumes have since been replaced by the
extracted folder (decision D-14), which is what this script reads now. Work is done in resumable chunks; finished chunks
are skipped on restart.

    python scripts/build_dataset_manifests.py [--splits train val test] [--workers 12]

Outputs (configs/paths.toml -> datasets.manifests):
    <split>_manifest.csv, <split>_summary.json, parts/<split>_NNNN.csv (resume state)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path, PurePosixPath

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import WORKSPACE_ROOT, section  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_DIRS = {"0_real": 0, "1_fake": 1}
FIELDS = ["split", "relpath", "label", "generator", "category", "size_bytes", "sha256",
          "width", "height", "mode", "format", "decode_ok", "error"]
CHUNK = 10_000

def parse_labels(rel: str, split: str) -> tuple[int, str, str]:
    """Label comes from the 0_real / 1_fake directory; never from anything else."""
    parts = PurePosixPath(rel).parts
    label_idx = [i for i, p in enumerate(parts) if p in LABEL_DIRS]
    if len(label_idx) != 1:
        raise ValueError(f"cannot determine label for {rel}")
    i = label_idx[0]
    if split == "test":
        generator, category = parts[0], "/".join(parts[1:i])
    else:
        generator, category = "progan", "/".join(parts[:i])
    return LABEL_DIRS[parts[i]], generator, category


def inspect_bytes(data: bytes) -> dict:
    rec = {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
           "width": "", "height": "", "mode": "", "format": "", "decode_ok": 0, "error": ""}
    try:
        with Image.open(io.BytesIO(data)) as im:
            rec.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.load()
        rec["decode_ok"] = 1
    except Exception as e:  # recorded, never raised: the manifest must list every file
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    return rec


def _work_file(args: tuple[str, str]) -> dict:
    root, rel = args
    try:
        data = (Path(root) / rel).read_bytes()
    except Exception as e:
        return {"relpath": rel, "size_bytes": "", "sha256": "", "width": "", "height": "", "mode": "",
                "format": "", "decode_ok": 0, "error": f"read: {type(e).__name__}: {e}"[:200]}
    return {"relpath": rel, **inspect_bytes(data)}


def list_split(split: str, ds: dict) -> tuple[list, str]:
    root = {"train": ds["progan_train"], "val": ds["progan_val"], "test": ds["cnn_synth_test"]}[split]
    rels = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return rels, str(root)


def build(split: str, ds: dict, out_dir: Path, workers: int) -> None:
    items, source = list_split(split, ds)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = (len(items) + CHUNK - 1) // CHUNK
    print(f"[{split}] {len(items):,} images from {source} -> {n_chunks} chunks", flush=True)

    pool = Pool(workers)
    fn, wrap = _work_file, (lambda x: (source, x))

    t0 = time.time()
    with pool:
        for c in range(n_chunks):
            part = parts_dir / f"{split}_{c:04d}.csv"
            if part.exists():
                continue
            chunk = items[c * CHUNK:(c + 1) * CHUNK]
            rows = pool.map(fn, [wrap(x) for x in chunk], chunksize=64)
            tmp = part.with_suffix(".tmp")
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for r in rows:
                    label, gen, cat = parse_labels(r["relpath"], split)
                    w.writerow({"split": split, "label": label, "generator": gen, "category": cat, **r})
            os.replace(tmp, part)  # a chunk is either complete or absent
            done = (c + 1) * CHUNK
            rate = min(done, len(items)) / max(time.time() - t0, 1e-9)
            print(f"  [{split}] chunk {c + 1}/{n_chunks}  ~{rate:,.0f} img/s", flush=True)

    # merge parts -> manifest + summary
    manifest = out_dir / f"{split}_manifest.csv"
    counts, exts, sizes, modes, fails = Counter(), Counter(), Counter(), Counter(), []
    total_bytes = 0
    with open(manifest, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=FIELDS)
        w.writeheader()
        for c in range(n_chunks):
            with open(parts_dir / f"{split}_{c:04d}.csv", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    w.writerow(r)
                    counts[(r["generator"], r["category"], r["label"])] += 1
                    exts[PurePosixPath(r["relpath"]).suffix.lower()] += 1
                    if r["decode_ok"] == "1":
                        sizes[f'{r["width"]}x{r["height"]}'] += 1
                        modes[r["mode"]] += 1
                    else:
                        fails.append({"relpath": r["relpath"], "error": r["error"]})
                    total_bytes += int(r["size_bytes"] or 0)
    n = sum(counts.values())
    if n != len(items):
        raise RuntimeError(f"[{split}] manifest has {n} rows but {len(items)} images were listed")

    by_label = Counter()
    by_group = {}
    for (g, cat, lab), k in counts.items():
        by_label[lab] += k
        key = g if split == "test" else cat
        by_group.setdefault(key, Counter())[("real" if lab == "0" else "fake")] += k
    summary = {
        "split": split,
        "source": str(Path(source).relative_to(WORKSPACE_ROOT)) if Path(source).is_relative_to(WORKSPACE_ROOT) else source,
        "n_images": n,
        "n_real": by_label["0"], "n_fake": by_label["1"],
        "per_group": {k: dict(v) for k, v in sorted(by_group.items())},
        "extensions": dict(exts),
        "top_resolutions": dict(sizes.most_common(10)),
        "n_distinct_resolutions": len(sizes),
        "modes": dict(modes),
        "n_decode_failures": len(fails),
        "decode_failures": fails[:200],
        "total_bytes": total_bytes,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / f"{split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{split}] done: {n:,} images, real {by_label['0']:,} / fake {by_label['1']:,}, "
          f"decode failures {len(fails)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"], choices=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = ap.parse_args()
    ds = section("datasets")
    out_dir = ds["manifests"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.splits:
        build(s, ds, out_dir, args.workers)


if __name__ == "__main__":
    main()
