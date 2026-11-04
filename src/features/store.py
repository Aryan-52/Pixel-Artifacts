"""Standardised feature store: shards -> one fp16 matrix per split (fp-v4.0, decision D-2).

fp-v3 concatenated the whole cache in RAM and standardised it on the GPU, which does not scale
to 1.44M rows. Here the shards are streamed twice:

  1. train shards only -> per-feature mean / std (float64 accumulation, std unbiased like
     torch.std, clamped at 1e-6), over all valid train rows (clean + augmented, as in fp-v3);
  2. every split -> ((x - mean) / std).clamp(+-10) written as fp16 rows to <split>_X.npy.

Store layout (<feature_store>/):
  stats.pt                      {pillar: {"mean", "std"}} fp32, TRAIN ONLY
  store_meta.json               dims, column offsets, per-split counts and signatures
  <split>_X.npy                 [N, 3094] float16, columns in ALL_PILLARS order
  <split>_y.npy  <split>_copy.npy   int8
  <split>_index.csv             row, entry, relpath, label, copy  (valid rows only)

The store is all the head trainer needs, so it is what gets uploaded to Kaggle.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.features.extraction import shard_dir
from src.models import five_pillar_v3 as v3
from src.paths import section

SPLITS = ("train", "val", "test")


def _split_files(split: str, cache_root: Path | None):
    d = shard_dir(split, cache_root)
    metas = sorted(d.glob(f"{split}_*_meta.json"))
    full = [m for m in metas if json.loads(m.read_text())["limit"] is None]
    if len(full) != 1:
        raise RuntimeError(f"{split}: expected exactly one full extraction in {d}, found {len(full)}")
    meta = json.loads(full[0].read_text())
    shards = [d / f"{split}_{meta['sig']}_s{k:05d}.pt" for k in range(meta["n_shards"])]
    missing = [s.name for s in shards if not s.exists()]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} shards missing (first: {missing[0]}); "
                           f"finish extraction first")
    return meta, shards, d / f"{split}_{meta['sig']}_entries.csv"


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_train_stats(shards) -> dict:
    s1, s2, n = {}, {}, 0
    for i, p in enumerate(shards):
        sh = _load(p)
        v = sh["valid"]
        for k in v3.ALL_PILLARS:
            x = sh[k][v].double()
            s1[k] = s1.get(k, 0) + x.sum(0)
            s2[k] = s2.get(k, 0) + (x * x).sum(0)
        n += int(v.sum())
        if (i + 1) % 50 == 0:
            print(f"  stats: {i + 1}/{len(shards)} shards", flush=True)
    stats = {}
    for k in v3.ALL_PILLARS:
        mean = s1[k] / n
        var = (s2[k] - n * mean * mean) / max(n - 1, 1)
        stats[k] = {"mean": mean.float(), "std": var.clamp_min(0).sqrt().float().clamp_min(1e-6)}
    return stats


def write_split(split, meta, shards, entries_csv, stats, out_dir: Path) -> dict:
    with open(entries_csv, newline="") as f:
        entries = [(r["relpath"], int(r["label"]), int(r["copy"])) for r in csv.DictReader(f)]
    valid = torch.cat([_load(p)["valid"] for p in shards])
    n = int(valid.sum())
    dim = sum(int(stats[k]["mean"].numel()) for k in v3.ALL_PILLARS)
    X = np.lib.format.open_memmap(out_dir / f"{split}_X.npy", mode="w+", dtype=np.float16, shape=(n, dim))
    y = np.empty(n, np.int8)
    cp = np.empty(n, np.int8)
    rows, r = [], 0
    for i, p in enumerate(shards):
        sh = _load(p)
        v = sh["valid"]
        parts = []
        for k in v3.ALL_PILLARS:
            x = sh[k][v].float()
            parts.append(((x - stats[k]["mean"]) / stats[k]["std"]).clamp_(-v3.FEATURE_CLIP, v3.FEATURE_CLIP))
        block = torch.cat(parts, 1).half().numpy()
        m = len(block)
        X[r:r + m] = block
        y[r:r + m] = sh["labels"][v].numpy().astype(np.int8)
        for e in (sh["start"] + torch.nonzero(v).flatten()).tolist():
            cp[len(rows)] = entries[e][2]
            rows.append((len(rows), e, *entries[e]))
        r += m
        if (i + 1) % 50 == 0:
            print(f"  {split}: {i + 1}/{len(shards)} shards written", flush=True)
    assert r == n
    X.flush()
    del X
    np.save(out_dir / f"{split}_y.npy", y)
    np.save(out_dir / f"{split}_copy.npy", cp)
    with open(out_dir / f"{split}_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "entry", "relpath", "label", "copy"])
        w.writerows(rows)
    dropped = [entries[e] for e in torch.nonzero(~valid).flatten().tolist()]
    return {"sig": meta["sig"], "n_entries": meta["n_entries"], "n_rows": n,
            "n_dropped": len(dropped), "dropped": dropped[:100],
            "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}


def build_store(splits=SPLITS, cache_root: Path | None = None, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or section("outputs")["feature_store"]
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {s: _split_files(s, cache_root) for s in splits}
    stats_path = out_dir / "stats.pt"
    train_meta = _split_files("train", cache_root)[0]
    if stats_path.exists() and torch.load(stats_path)["train_sig"] == train_meta["sig"]:
        stats = torch.load(stats_path)["stats"]
        print("train stats loaded")
    else:
        print("computing train-only standardisation stats...")
        stats = compute_train_stats(_split_files("train", cache_root)[1])
        v3.atomic_torch_save({"train_sig": train_meta["sig"], "stats": stats}, stats_path)
    dims = {k: int(stats[k]["mean"].numel()) for k in v3.ALL_PILLARS}
    offsets, o = {}, 0
    for k in v3.ALL_PILLARS:
        offsets[k] = [o, o + dims[k]]
        o += dims[k]
    meta_path = out_dir / "store_meta.json"
    store_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    store_meta.update({"pillars": v3.ALL_PILLARS, "dims": dims, "offsets": offsets,
                       "feature_clip": v3.FEATURE_CLIP, "dtype": "float16",
                       "feature_code_hash": train_meta["feature_code_hash"],
                       "train_sig": train_meta["sig"]})
    store_meta.setdefault("splits", {})
    for s in splits:
        if store_meta["splits"].get(s, {}).get("sig") == files[s][0]["sig"] and (out_dir / f"{s}_X.npy").exists():
            print(f"{s}: already written")
            continue
        print(f"writing {s}...")
        store_meta["splits"][s] = write_split(s, *files[s], stats, out_dir)
        v3.atomic_write_text(json.dumps(store_meta, indent=1), meta_path)
    v3.atomic_write_text(json.dumps(store_meta, indent=1), meta_path)
    return store_meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build the standardised fp16 feature store")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    m = build_store(a.splits, a.cache, a.out)
    print(json.dumps({s: {k: v for k, v in d.items() if k != "dropped"} for s, d in m["splits"].items()}, indent=1))
