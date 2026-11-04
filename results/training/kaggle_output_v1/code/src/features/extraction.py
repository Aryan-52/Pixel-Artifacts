"""Resumable five-pillar feature extraction for fp-v4.0.

Pixel preprocessing and all five extractors are the fp-v3.0 code, imported unchanged from
src/models/five_pillar_v3.py (feature-code hash must stay 369cfa92c5a260ea under Python 3.12).
What differs from fp-v3 is only the driver:

* samples come from the dataset manifests (labels = 0_real / 1_fake directory), minus the
  cross-split overlaps excluded by decision D-13;
* images are decoded by a thread pool, not DataLoader worker processes: on Windows each
  spawned worker re-imports torch/open_clip (~0.75 GB RAM, ~45 s start-up), and PIL decode,
  JPEG and SHA-256 release the GIL anyway (measured: 12 workers exhausted 16 GB RAM; 8 threads
  load ~1,900 img/s vs ~580 img/s GPU throughput);
* every clean read is checked against the manifest SHA-256; mismatching files are dropped;
* shards are never concatenated in RAM (build_feature_store.py streams them).

Shard file: <cache>/<split>/<split>_<sig>_s<k:05d>.pt, same keys as fp-v3 shards plus
"sha_ok". Written atomically, so a crash or power loss loses at most the shard in progress.
Image paths are relative to the split root (the process chdirs there), which keeps the
path-seeded blur/JPEG augmentation identical on any machine.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import Dataset

from src.models import five_pillar_v3 as v3
from src.paths import section

V3_FEATURE_CODE_HASH = "369cfa92c5a260ea"
SPLIT_ROOT_KEY = {"train": "progan_train", "val": "progan_val", "test": "cnn_synth_test"}
SHARD_SIZE = v3.SHARD_SIZE            # 2048 = 32 batches of 64


def load_samples(split: str) -> list[tuple[str, int, str]]:
    """(relpath, label, sha256) in manifest order, minus D-13 exclusions for train/val."""
    man = section("datasets")["manifests"]
    excluded = set()
    if split in ("train", "val"):
        with open(man / "leakage_pairs.csv", newline="") as f:
            excluded = {r["relpath"] for r in csv.DictReader(f) if r["split"] == split}
    with open(man / f"{split}_manifest.csv", newline="") as f:
        rows = [(r["relpath"], int(r["label"]), r["sha256"]) for r in csv.DictReader(f)]
    return [r for r in rows if r[0] not in excluded]


def split_root(split: str) -> Path:
    return section("datasets")[SPLIT_ROOT_KEY[split]]


class VerifiedImageDataset(Dataset):
    """fp-v3 ImageEntryDataset plus a SHA-256 check of the clean (copy 0) read."""

    def __init__(self, entries, shas):
        self.base = v3.ImageEntryDataset(entries)
        self.shas = shas

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        t, label, ok, _ = self.base[i]
        path, _, copy = self.base.entries[i]
        sha_ok = 1
        if copy == 0:
            try:
                with open(path, "rb") as f:
                    sha_ok = int(hashlib.sha256(f.read()).hexdigest() == self.shas[i])
            except OSError:
                sha_ok = 0
        return t, label, ok, i, sha_ok


def _load_batch(ds, idxs):
    items = [ds[i] for i in idxs]
    return (torch.stack([it[0] for it in items]), torch.tensor([it[1] for it in items]),
            torch.tensor([it[2] for it in items]), torch.tensor([it[3] for it in items]),
            torch.tensor([it[4] for it in items]))


def threaded_batches(ds, batches, threads):
    """Yield collated batches in order, keeping 2*threads batches in flight."""
    with ThreadPoolExecutor(threads) as ex:
        it = iter(batches)
        q = deque(ex.submit(_load_batch, ds, b) for b in [next(it, None) for _ in range(2 * threads)] if b)
        while q:
            out = q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(_load_batch, ds, nxt))
            yield out


def _new_acc():
    return {"feats": {k: [] for k in v3.ALL_PILLARS}, "labels": [], "valid": [], "sha_ok": [],
            "n_decode_fail": 0, "n_nonfinite_any": 0, "n_sha_mismatch": 0,
            "nf_rows": {k: 0 for k in v3.ALL_PILLARS}, "nf_vals": {k: 0 for k in v3.ALL_PILLARS},
            "t0": time.time()}


def _add_batch(acc, imgs, lab, ok, sha_ok, enc):
    """Same validity bookkeeping as fp-v3 _extract_shard, plus the SHA check."""
    feats = v3.extract_batch(imgs, enc)
    ok = ok.bool() & sha_ok.bool()
    ok_dev = ok.to(v3.DEVICE)
    finite_rows = torch.ones(imgs.shape[0], dtype=torch.bool, device=v3.DEVICE)
    for k in v3.ALL_PILLARS:
        s = feats[k].to(v3.STORE_DTYPE[k])
        fin = torch.isfinite(s)
        row_ok = fin.all(1)
        acc["nf_rows"][k] += int(((~row_ok) & ok_dev).sum())
        acc["nf_vals"][k] += int(((~fin).sum(1) * ok_dev).sum())
        finite_rows &= row_ok
        acc["feats"][k].append(s.cpu())
    acc["n_decode_fail"] += int((~ok).sum())
    acc["n_sha_mismatch"] += int((~sha_ok.bool()).sum())
    acc["n_nonfinite_any"] += int(((~finite_rows) & ok_dev).sum())
    acc["labels"].append(lab.float())
    acc["valid"].append(ok & finite_rows.cpu())
    acc["sha_ok"].append(sha_ok.bool())


def _finish_shard(acc, start, n):
    secs = time.time() - acc["t0"]
    shard = {k: torch.cat(v) for k, v in acc["feats"].items()}
    shard["labels"] = torch.cat(acc["labels"])
    shard["valid"] = torch.cat(acc["valid"])
    shard["sha_ok"] = torch.cat(acc["sha_ok"])
    shard["start"] = start
    shard["stats"] = {
        "n_entries": n, "n_decode_fail": acc["n_decode_fail"],
        "n_sha_mismatch": acc["n_sha_mismatch"],
        "n_nonfinite_rows_any": acc["n_nonfinite_any"],
        "nonfinite_rows_by_pillar": acc["nf_rows"], "nonfinite_values_by_pillar": acc["nf_vals"],
        "n_kept": int(shard["valid"].sum()), "seconds": secs,
        "images_per_second": n / max(secs, 1e-9), "created": datetime.now().isoformat(),
    }
    return shard


def _shard_ok(path, lo, hi):
    try:
        s = torch.load(path, map_location="cpu", weights_only=False)
        return (s["start"] == lo and len(s["labels"]) == hi - lo and len(s["valid"]) == hi - lo
                and all(len(s[p]) == hi - lo for p in v3.ALL_PILLARS))
    except Exception:
        return False


def shard_dir(split: str, out_root: Path | None = None) -> Path:
    return (out_root or section("outputs")["feature_cache"]) / split


def extract_split(split: str, workers: int = 8, limit: int | None = None,
                  out_root: Path | None = None) -> dict:
    v3.seed_everything(v3.SEED)
    out_dir = shard_dir(split, out_root).resolve()      # absolute: we chdir below
    out_dir.mkdir(parents=True, exist_ok=True)
    v3.PRETRAINED_DIR = section("outputs")["pretrained"]

    samples = load_samples(split)
    if limit:                                       # evenly spread subset, for benchmarking
        step = max(1, len(samples) // limit)
        samples = samples[::step][:limit]
    n_copies = 1 + (v3.TRAIN_AUG_COPIES if split == "train" else 0)

    code_hash = v3._feature_code_hash()
    sig = v3.split_signature(split, [(p, l) for p, l, _ in samples], n_copies)
    entries = [(p, l, c) for c in range(n_copies) for (p, l, _) in samples]
    shas = [h for _ in range(n_copies) for (_, _, h) in samples]
    n_shards = math.ceil(len(entries) / SHARD_SIZE)
    meta = {"split": split, "sig": sig, "feature_code_hash": code_hash,
            "matches_fp_v3_hash": code_hash == V3_FEATURE_CODE_HASH,
            "n_samples": len(samples), "n_copies": n_copies, "n_entries": len(entries),
            "n_shards": n_shards, "shard_size": SHARD_SIZE, "limit": limit,
            "batch_size": v3.BATCH_SIZE_FEATURE, "library_versions": v3.library_versions()}
    print(json.dumps(meta, indent=1))
    if not meta["matches_fp_v3_hash"]:
        print(f"WARNING: feature-code hash {code_hash} != fp-v3 {V3_FEATURE_CODE_HASH}")
    v3.atomic_write_text(json.dumps(meta, indent=1), out_dir / f"{split}_{sig}_meta.json")
    with open(out_dir / f"{split}_{sig}_entries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "relpath", "label", "copy"])
        w.writerows((i, p, l, c) for i, (p, l, c) in enumerate(entries))

    def path_of(k):
        return out_dir / f"{split}_{sig}_s{k:05d}.pt"

    todo = []
    for k in range(n_shards):
        lo, hi = k * SHARD_SIZE, min((k + 1) * SHARD_SIZE, len(entries))
        if not (path_of(k).exists() and _shard_ok(path_of(k), lo, hi)):
            todo.append(k)
    print(f"{n_shards - len(todo)}/{n_shards} shards already done; {len(todo)} to extract")
    if not todo:
        return meta

    bs = v3.BATCH_SIZE_FEATURE
    batches = [list(range(lo, min(lo + bs, hi)))
               for k in todo
               for lo0, hi in [(k * SHARD_SIZE, min((k + 1) * SHARD_SIZE, len(entries)))]
               for lo in range(lo0, hi, bs)]

    os.chdir(split_root(split))                     # entries are relative to the split root
    enc = v3.get_encoders()
    loader = threaded_batches(VerifiedImageDataset(entries, shas), batches, workers)

    t_start, done, cur, acc = time.time(), 0, None, None
    for imgs, lab, ok, idx, sha_ok in loader:
        k = int(idx[0]) // SHARD_SIZE
        if cur is None:
            cur, acc = k, _new_acc()
        _add_batch(acc, imgs, lab, ok, sha_ok, enc)
        lo, hi = k * SHARD_SIZE, min((k + 1) * SHARD_SIZE, len(entries))
        if int(idx[-1]) == hi - 1:
            v3.atomic_torch_save(_finish_shard(acc, lo, hi - lo), path_of(k))
            done += 1
            el = time.time() - t_start
            rate = done * SHARD_SIZE / el
            eta = (len(todo) - done) * SHARD_SIZE / rate
            print(f"[{datetime.now():%H:%M:%S}] {split} shard {k + 1}/{n_shards} saved | "
                  f"{done}/{len(todo)} this run | {rate:6.1f} img/s | "
                  f"ETA {eta / 3600:5.2f} h | sha mismatches {acc['n_sha_mismatch']}", flush=True)
            cur = None
    meta["seconds_this_run"] = time.time() - t_start
    return meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--workers", type=int, default=8, help="image-loading threads")
    ap.add_argument("--limit", type=int, default=None, help="evenly spaced sample subset")
    ap.add_argument("--out", type=Path, default=None, help="cache root (default: config)")
    a = ap.parse_args(argv)
    meta = extract_split(a.split, a.workers, a.limit, a.out)
    print(json.dumps({k: meta[k] for k in meta if k != "library_versions"}, indent=1))
    sys.stdout.flush()
