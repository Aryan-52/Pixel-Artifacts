"""fp-v4.1 feature store: the fp-v4.0 store with the spatial pillar re-extracted (decision D-20).

The spatial descriptor is the same fp-v3 function (extract_spatial_features: mean/std of stages
2, 3, 4, 6 plus the 1280-d pooled output = 1952 dims), run on the fine-tuned EfficientNet-B0
from src/training/spatial_finetune.py instead of the frozen ImageNet one. Rows, labels, copies
(clean / blur-JPEG view), splits and the four other pillars are taken unchanged from the fp-v4.0
store, so fp-v4.0 and fp-v4.1 differ only in the spatial columns.

  1. raw spatial features per split -> <work>/<split>_spatial_raw.npy (fp16, resumable by chunk)
  2. train-only mean/std of the new spatial columns
  3. <out>/ = copy of the fp-v4.0 store with the spatial columns replaced, same standardisation
     and clipping rule; store_meta records the fine-tuned weights' SHA-256.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.features.extraction import split_root, threaded_batches
from src.models import five_pillar_v3 as v3
from src.paths import section
from src.training.head import Store
from src.training.spatial_finetune import _Views, build_effnet

SPLITS = ("train", "val", "test")
CHUNK = 16_384                       # rows per resumable unit
DIM = v3.EXPECTED_DIMS["spatial"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_finetuned(path: Path):
    net = build_effnet()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["effnet"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.inference_mode()
def extract_raw(split: str, old: Store, net, work: Path, threads: int) -> Path:
    idx = old.index(split)
    root = split_root(split)
    ds = _Views([(str(root / r["relpath"]), int(r["label"]), int(r["copy"])) for r in idx])
    n = len(ds)
    work.mkdir(parents=True, exist_ok=True)
    out, mark = work / f"{split}_spatial_raw.npy", work / f"{split}_spatial_raw.rows"
    done = int(mark.read_text()) if mark.exists() and out.exists() else 0
    raw = np.lib.format.open_memmap(out, mode="r+" if done else "w+", dtype=np.float16, shape=(n, DIM))
    if done >= n:
        return out
    mean = torch.tensor(v3.IMAGENET_MEAN, device=v3.DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(v3.IMAGENET_STD, device=v3.DEVICE).view(1, 3, 1, 1)
    bs, t0, start = v3.BATCH_SIZE_FEATURE, time.time(), done
    print(f"{split}: {n} rows, resuming at {done}", flush=True)
    for a in range(done, n, CHUNK):
        b = min(a + CHUNK, n)
        batches = [list(range(i, min(i + bs, b))) for i in range(a, b, bs)]
        for imgs, lab, ok, i, _ in threaded_batches(ds, batches, threads):
            if not bool(ok.all()):
                raise RuntimeError(f"{split}: image decode failed at rows {i.tolist()}")
            x = (imgs.to(v3.DEVICE, non_blocking=True).float() / 255.0 - mean) / std
            with torch.autocast("cuda", dtype=torch.float16, enabled=v3.DEVICE.type == "cuda"):
                f = v3.extract_spatial_features(net, x)
            f = f.half()
            if not bool(torch.isfinite(f).all()):
                raise RuntimeError(f"{split}: non-finite spatial features at rows {i.tolist()}")
            raw[int(i[0]):int(i[-1]) + 1] = f.cpu().numpy()
        raw.flush()
        v3.atomic_write_text(str(b), mark)
        rate = (b - start) / max(time.time() - t0, 1e-9)
        print(f"[{datetime.now():%H:%M:%S}] {split} spatial rows {b}/{n} | {rate:6.0f} img/s | "
              f"ETA {(n - b) / max(rate, 1e-9) / 3600:5.2f} h", flush=True)
    del raw
    return out


def train_stats(raw_path: Path) -> dict:
    raw = np.load(raw_path, mmap_mode="r")
    s1 = torch.zeros(DIM, dtype=torch.float64)
    s2 = torch.zeros(DIM, dtype=torch.float64)
    for a in range(0, len(raw), CHUNK * 4):
        x = torch.from_numpy(np.array(raw[a:a + CHUNK * 4])).double()
        s1 += x.sum(0)
        s2 += (x * x).sum(0)
    n = len(raw)
    mean = s1 / n
    var = (s2 - n * mean * mean) / max(n - 1, 1)
    return {"mean": mean.float(), "std": var.clamp_min(0).sqrt().float().clamp_min(1e-6)}


def write_store(old: Store, raws: dict, st: dict, out: Path, weights_sha: str, ckpt: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = old.meta["offsets"]["spatial"]
    for split in SPLITS:
        mark = out / f"{split}.written"
        if mark.exists():
            print(f"{split}: already written")
            continue
        Xo, _, _ = old.split(split)
        raw = np.load(raws[split], mmap_mode="r")
        assert len(raw) == len(Xo) and raw.shape[1] == hi - lo
        Xn = np.lib.format.open_memmap(out / f"{split}_X.npy", mode="w+", dtype=np.float16, shape=Xo.shape)
        for a in range(0, len(Xo), CHUNK * 4):
            blk = np.array(Xo[a:a + CHUNK * 4])
            r = torch.from_numpy(np.array(raw[a:a + CHUNK * 4])).float()
            blk[:, lo:hi] = ((r - st["mean"]) / st["std"]).clamp_(-v3.FEATURE_CLIP, v3.FEATURE_CLIP).half().numpy()
            Xn[a:a + len(blk)] = blk
        Xn.flush()
        del Xn
        for f in (f"{split}_y.npy", f"{split}_copy.npy", f"{split}_index.csv"):
            shutil.copy2(old.root / f, out / f)
        v3.atomic_write_text(datetime.now().isoformat(), mark)
        print(f"{split}: written", flush=True)
    if (old.root / "leakage_pairs.csv").exists():
        shutil.copy2(old.root / "leakage_pairs.csv", out / "leakage_pairs.csv")
    stats = {k: v for k, v in old.stats.items()}
    stats["spatial"] = st
    old_stats = torch.load(old.root / "stats.pt")
    v3.atomic_torch_save({**old_stats, "stats": stats, "spatial_weights_sha256": weights_sha}, out / "stats.pt")
    meta = dict(old.meta)
    meta.update({"model_version": "fp-v4.1",
                 "feature_code_hash": f"{old.meta['feature_code_hash']}+spatial_ft:{weights_sha[:12]}",
                 "spatial_encoder": {"finetuned_checkpoint": str(ckpt), "sha256": weights_sha,
                                     "base_store": str(old.root)}})
    v3.atomic_write_text(json.dumps(meta, indent=1), out / "store_meta.json")
    return meta


def build(ckpt: Path, old_root: Path, out: Path, work: Path, threads: int) -> dict:
    old = Store(old_root)
    work.mkdir(parents=True, exist_ok=True)
    sha = _sha256(ckpt)
    marker = work / "weights_sha256.txt"
    if marker.exists() and marker.read_text() != sha:
        raise RuntimeError(f"{work} holds features of other fine-tuned weights; use a new --work dir")
    v3.atomic_write_text(sha, marker)
    net = load_finetuned(ckpt)
    raws = {s: extract_raw(s, old, net, work, threads) for s in SPLITS}
    del net
    torch.cuda.empty_cache()
    print("train-only standardisation stats for the new spatial columns...", flush=True)
    return write_store(old, raws, train_stats(raws["train"]), out, sha, ckpt)


def main(argv=None):
    import argparse
    o = section("outputs")
    ap = argparse.ArgumentParser(description="Build the fp-v4.1 store (fine-tuned spatial pillar)")
    ap.add_argument("--ckpt", type=Path, default=o["results"] / "training" / "spatial_ft" / "best.pth")
    ap.add_argument("--old", type=Path, default=o["feature_store"])
    ap.add_argument("--out", type=Path, default=o["results"] / "training" / "feature_store_ft")
    ap.add_argument("--work", type=Path, default=o["results"] / "training" / "spatial_ft_cache")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args(argv)
    if not (a.ckpt.parent / "DONE.json").exists():
        raise SystemExit(f"{a.ckpt.parent} has no DONE.json: finish the spatial fine-tune first")
    meta = build(a.ckpt, a.old, a.out, a.work, a.threads)
    print(json.dumps({k: meta[k] for k in ("model_version", "feature_code_hash", "spatial_encoder")}, indent=1))
