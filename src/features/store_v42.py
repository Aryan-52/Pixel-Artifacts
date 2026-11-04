"""fp-v4.2 feature store: fp-v4.0 store with two pillars re-extracted (decision D-22).

* spatial  : the fp-v3 1952-d descriptor (extract_spatial_features) from the EfficientNet-B0
             fine-tuned with fresh degradations (src/training/spatial_finetune_v2.py);
* semantic : OpenCLIP ViT-L/14 (OpenAI weights, frozen), L2-normalised image embedding (768-d),
             the backbone of UniversalFakeDetect (Ojha et al., CVPR 2023), replacing ViT-B/32 (512-d).

Rows, labels, copies (clean / fixed blur-JPEG view), splits and the frequency, colour and
physical columns are taken unchanged from the fp-v4.0 store. Each new pillar is extracted on its
own, resumably, so the semantic pass can run while the spatial network is still training:

  1. raw features per split and pillar -> <work>/<split>_<pillar>_raw.npy (fp16, resumable by chunk)
  2. train-only mean/std of the new columns
  3. <out>/ = the store with the new columns, same standardisation and clipping rule.

    python scripts/build_store_v42.py --pillars semantic          # now
    python scripts/build_store_v42.py --pillars spatial --assemble # after the fine-tune
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.features.extraction import split_root, threaded_batches
from src.features.spatial_ft import _sha256, load_finetuned
from src.models import five_pillar_v3 as v3
from src.paths import section
from src.training.head import Store
from src.training.spatial_finetune import _Views

SPLITS = ("train", "val", "test")
CHUNK = 16_384                       # rows per resumable unit
CLIP_MODEL, CLIP_WEIGHTS = "ViT-L-14", "openai"
DIMS = {"spatial": 1952, "semantic": 768}


def load_clip():
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_WEIGHTS, cache_dir=str(section("outputs")["pretrained"] / "open_clip"))
    model.to(v3.DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def semantic_features(clip, x01, mean, std):
    """Same rule as fp-v3 extract_batch: CLIP-normalised 224 crop -> encode_image -> L2 norm."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=v3.DEVICE.type == "cuda"):
        sem = clip.encode_image((x01 - mean) / std)
    return F.normalize(sem.float(), dim=-1)


def spatial_features(net, x01, mean, std):
    with torch.autocast("cuda", dtype=torch.float16, enabled=v3.DEVICE.type == "cuda"):
        return v3.extract_spatial_features(net, (x01 - mean) / std)


def encoder(pillar: str, ckpt: Path):
    """-> (identity string, fn(uint8 batch on any device) -> float features)."""
    v = lambda t: torch.tensor(t, device=v3.DEVICE).view(1, 3, 1, 1)
    if pillar == "semantic":
        clip, mean, std = load_clip(), v(v3.CLIP_MEAN), v(v3.CLIP_STD)
        return f"{CLIP_MODEL}/{CLIP_WEIGHTS}", lambda x: semantic_features(clip, x, mean, std)
    net, mean, std = load_finetuned(ckpt), v(v3.IMAGENET_MEAN), v(v3.IMAGENET_STD)
    return _sha256(ckpt), lambda x: spatial_features(net, x, mean, std)


@torch.inference_mode()
def extract_raw(split: str, old: Store, pillar: str, ident: str, fn, work: Path, threads: int) -> Path:
    idx = old.index(split)
    root = split_root(split)
    ds = _Views([(str(root / r["relpath"]), int(r["label"]), int(r["copy"])) for r in idx])
    n, dim = len(ds), DIMS[pillar]
    work.mkdir(parents=True, exist_ok=True)
    marker = work / f"{pillar}_identity.txt"
    if marker.exists() and marker.read_text() != ident:
        raise RuntimeError(f"{work} holds {pillar} features of {marker.read_text()}, not {ident}")
    v3.atomic_write_text(ident, marker)
    out, mark = work / f"{split}_{pillar}_raw.npy", work / f"{split}_{pillar}_raw.rows"
    done = int(mark.read_text()) if mark.exists() and out.exists() else 0
    if done >= n:
        return out
    raw = np.lib.format.open_memmap(out, mode="r+" if done else "w+", dtype=np.float16, shape=(n, dim))
    bs, t0, start = v3.BATCH_SIZE_FEATURE, time.time(), done
    print(f"{split} {pillar}: {n} rows, resuming at {done}", flush=True)
    for a in range(done, n, CHUNK):
        b = min(a + CHUNK, n)
        batches = [list(range(i, min(i + bs, b))) for i in range(a, b, bs)]
        for imgs, _, ok, i, _ in threaded_batches(ds, batches, threads):
            if not bool(ok.all()):
                raise RuntimeError(f"{split}: image decode failed at rows {i.tolist()}")
            f = fn(imgs.to(v3.DEVICE, non_blocking=True).float() / 255.0).half()
            if not bool(torch.isfinite(f).all()):
                raise RuntimeError(f"{split}: non-finite {pillar} features at rows {i.tolist()}")
            raw[int(i[0]):int(i[-1]) + 1] = f.cpu().numpy()
        raw.flush()
        v3.atomic_write_text(str(b), mark)
        rate = (b - start) / max(time.time() - t0, 1e-9)
        print(f"[{datetime.now():%H:%M:%S}] {split} {pillar} rows {b}/{n} | {rate:6.0f} img/s | "
              f"ETA {(n - b) / max(rate, 1e-9) / 3600:5.2f} h", flush=True)
    del raw
    return out


def raw_done(work: Path, split: str, pillar: str, n: int) -> bool:
    mark = work / f"{split}_{pillar}_raw.rows"
    return mark.exists() and int(mark.read_text()) >= n


def train_stats(raw_path: Path) -> dict:
    raw = np.load(raw_path, mmap_mode="r")
    dim = raw.shape[1]
    s1, s2 = torch.zeros(dim, dtype=torch.float64), torch.zeros(dim, dtype=torch.float64)
    for a in range(0, len(raw), CHUNK * 4):
        x = torch.from_numpy(np.array(raw[a:a + CHUNK * 4])).double()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    n = len(raw)
    mean = s1 / n
    var = (s2 - n * mean * mean) / max(n - 1, 1)
    return {"mean": mean.float(), "std": var.clamp_min(0).sqrt().float().clamp_min(1e-6)}


def assemble(old: Store, work: Path, out: Path, ckpt: Path) -> dict:
    new = list(DIMS)
    dims = {k: (DIMS[k] if k in new else old.meta["dims"][k]) for k in v3.ALL_PILLARS}
    offsets, o = {}, 0
    for k in v3.ALL_PILLARS:
        offsets[k] = [o, o + dims[k]]
        o += dims[k]
    stats = dict(old.stats)
    print("train-only standardisation stats for the new columns...", flush=True)
    for k in new:
        stats[k] = train_stats(work / f"train_{k}_raw.npy")
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        mark = out / f"{split}.written"
        if mark.exists():
            print(f"{split}: already written")
            continue
        Xo, _, _ = old.split(split)
        raws = {k: np.load(work / f"{split}_{k}_raw.npy", mmap_mode="r") for k in new}
        assert all(len(r) == len(Xo) for r in raws.values())
        Xn = np.lib.format.open_memmap(out / f"{split}_X.npy", mode="w+", dtype=np.float16, shape=(len(Xo), o))
        step = CHUNK * 4
        for a in range(0, len(Xo), step):
            blk_o = np.array(Xo[a:a + step])
            blk = np.empty((len(blk_o), o), np.float16)
            for k in v3.ALL_PILLARS:
                lo, hi = offsets[k]
                if k in new:
                    r = torch.from_numpy(np.array(raws[k][a:a + step])).float()
                    blk[:, lo:hi] = ((r - stats[k]["mean"]) / stats[k]["std"]).clamp_(
                        -v3.FEATURE_CLIP, v3.FEATURE_CLIP).half().numpy()
                else:
                    blk[:, lo:hi] = blk_o[:, slice(*old.meta["offsets"][k])]
            Xn[a:a + len(blk)] = blk
        Xn.flush()
        del Xn
        for f in (f"{split}_y.npy", f"{split}_copy.npy", f"{split}_index.csv"):
            shutil.copy2(old.root / f, out / f)
        v3.atomic_write_text(datetime.now().isoformat(), mark)
        print(f"{split}: written", flush=True)
    if (old.root / "leakage_pairs.csv").exists():
        shutil.copy2(old.root / "leakage_pairs.csv", out / "leakage_pairs.csv")
    sha = (work / "spatial_identity.txt").read_text()
    old_stats = torch.load(old.root / "stats.pt")
    v3.atomic_torch_save({**old_stats, "stats": stats, "spatial_weights_sha256": sha}, out / "stats.pt")
    meta = dict(old.meta)
    meta.update({"model_version": "fp-v4.2", "dims": dims, "offsets": offsets,
                 "feature_code_hash": f"{old.meta['feature_code_hash']}+spatial_ft:{sha[:12]}"
                                      f"+semantic:{CLIP_MODEL}/{CLIP_WEIGHTS}",
                 "spatial_encoder": {"finetuned_checkpoint": str(ckpt), "sha256": sha, "base_store": str(old.root)},
                 "semantic_encoder": {"library": "open_clip", "model": CLIP_MODEL, "pretrained": CLIP_WEIGHTS,
                                      "frozen": True, "dim": DIMS["semantic"]}})
    v3.atomic_write_text(json.dumps(meta, indent=1), out / "store_meta.json")
    return meta


def main(argv=None):
    import argparse
    o = section("outputs")
    ap = argparse.ArgumentParser(description="Build the fp-v4.2 store (augmented spatial + ViT-L/14 semantic)")
    ap.add_argument("--pillars", nargs="*", default=list(DIMS), choices=list(DIMS))
    ap.add_argument("--assemble", action="store_true", help="write the store once every raw split is done")
    ap.add_argument("--ckpt", type=Path, default=o["results"] / "training" / "spatial_ft_v2" / "best.pth")
    ap.add_argument("--old", type=Path, default=o["feature_store"])
    ap.add_argument("--out", type=Path, default=o["results"] / "training" / "feature_store_v42")
    ap.add_argument("--work", type=Path, default=o["results"] / "training" / "store_v42_cache")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    old = Store(a.old)
    if "spatial" in a.pillars and not (a.ckpt.parent / "DONE.json").exists():
        raise SystemExit(f"{a.ckpt.parent} has no DONE.json: finish the spatial fine-tune first")
    for pillar in a.pillars:
        ident, fn = encoder(pillar, a.ckpt)
        for s in SPLITS:
            extract_raw(s, old, pillar, ident, fn, a.work, a.threads)
        del fn
        torch.cuda.empty_cache()
    if a.assemble:
        missing = [(s, k) for s in SPLITS for k in DIMS if not raw_done(a.work, s, k, len(old.index(s)))]
        if missing:
            raise SystemExit(f"raw features missing: {missing}")
        meta = assemble(old, a.work, a.out, a.ckpt)
        print(json.dumps({k: meta[k] for k in ("model_version", "feature_code_hash", "dims")}, indent=1))
