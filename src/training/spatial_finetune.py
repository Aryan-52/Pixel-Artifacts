"""Fine-tune the spatial pillar's EfficientNet-B0 end to end on ProGAN (fp-v4.1, decision D-20).

fp-v4.0 reads the spatial pillar from a frozen ImageNet EfficientNet-B0. Here the whole network
(no frozen blocks, as in Wang et al. and in the team's stage1 model) is trained as a real/fake
classifier, and fp-v4.1 then re-extracts the same 1952-d spatial descriptor from it
(src/features/spatial_ft.py). The other four pillars are unchanged.

Same rules as the head trainer: batch 64, at most 50 epochs, early stopping on clean-validation
BCE, test data never touched. Images, centre crop and the two training views per image (clean,
and the path-seeded blur/JPEG copy) are exactly the fp-v3 pipeline used for the feature store.
One "epoch" is a fresh seeded draw of EPOCH_SAMPLES of those 1.44M training views, so that
early stopping is checked often enough to fit the time budget (a full pass would take ~40 min).

Run directory (results/training/spatial_ft/):
  latest.pth, latest.prev.pth   full training state, saved every ckpt_minutes and every epoch
  best.pth                      {"effnet": state_dict, "meta": ...} of the best epoch
  history.csv, DONE.json
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset

from src.features.extraction import load_samples, split_root, threaded_batches
from src.models import five_pillar_v3 as v3
from src.paths import section
from src.training.head import _load_state, _rng_state, _save_state, _set_rng_state

BATCH_SIZE = 64            # D-5
MAX_EPOCHS = 50            # D-5
EPOCH_SAMPLES = 200_000    # training views per epoch (seeded draw without replacement)
PATIENCE = 3
MIN_DELTA = 1e-4
LR = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = v3.DEVICE


class _Views(Dataset):
    """fp-v3 ImageEntryDataset in the 5-tuple shape threaded_batches expects (no SHA re-check:
    every file was verified against the manifest during feature extraction)."""

    def __init__(self, entries):
        self.base = v3.ImageEntryDataset(entries)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        t, label, ok, _ = self.base[i]
        return t, label, ok, i, 1


def build_effnet():
    from torchvision import models
    torch.hub.set_dir(str(section("outputs")["pretrained"] / "torch_hub"))
    net = models.efficientnet_b0(weights=getattr(models.EfficientNet_B0_Weights, v3.EFFNET_WEIGHTS_NAME))
    net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 1))
    # NCHW, not channels_last: measured 640 vs 97 img/s for B0 training on the RTX 5050
    torch.backends.cudnn.benchmark = True
    return net.to(DEVICE)


def _normalise(x_uint8, mean, std):
    return ((x_uint8.to(DEVICE, non_blocking=True).float() / 255.0) - mean) / std


def _abs_entries(split, entries):
    root = split_root(split)
    return [(str(root / p), l, c) for p, l, c in entries]


@torch.no_grad()
def evaluate(net, ds, threads, mean, std):
    net.eval()
    batches = [list(range(i, min(i + 256, len(ds)))) for i in range(0, len(ds), 256)]
    logits, labels = [], []
    for imgs, lab, ok, _, _ in threaded_batches(ds, batches, threads):
        if not bool(ok.all()):
            raise RuntimeError("validation image failed to decode")
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            lg = net(_normalise(imgs, mean, std))
        logits.append(lg.float().flatten().cpu())
        labels.append(lab.float())
    lg, y = torch.cat(logits), torch.cat(labels)
    return {"val_loss": float(F.binary_cross_entropy_with_logits(lg, y)),
            "val_acc": float(((lg > 0).float() == y).float().mean()),
            "val_auc": float(roc_auc_score(y.numpy(), lg.numpy()))}


def finetune(run_dir: Path, threads=8, ckpt_minutes=5.0, deadline=None, limit=None) -> str:
    """Returns "done" or "paused" (deadline reached; rerun to resume)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE.json").exists():
        print("spatial fine-tune already done")
        return "done"
    v3.seed_everything(v3.SEED)
    n_copies = 1 + v3.TRAIN_AUG_COPIES
    tr = load_samples("train")
    train = _Views(_abs_entries("train", [(p, l, c) for c in range(n_copies) for p, l, _ in tr]))
    val = _Views(_abs_entries("val", [(p, l, 0) for p, l, _ in load_samples("val")]))
    epoch_samples = min(limit or EPOCH_SAMPLES, len(train))
    steps = epoch_samples // BATCH_SIZE

    net = build_effnet()
    decay = [p for p in net.parameters() if p.ndim >= 2]
    no_decay = [p for p in net.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=LR,
                            fused=DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=v3.USE_AMP)
    mean = torch.tensor(v3.IMAGENET_MEAN, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(v3.IMAGENET_STD, device=DEVICE).view(1, 3, 1, 1)
    meta = {"model": "efficientnet_b0", "weights": v3.EFFNET_WEIGHTS_NAME, "frozen_blocks": 0,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "epoch_samples": epoch_samples,
            "n_train_views": len(train), "n_val": len(val), "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "seed": v3.SEED, "augment": "fp-v3 clean + blur/JPEG copy, hflip"}
    st = {"epoch": 0, "batch": 0, "loss_sum": 0.0, "n_seen": 0, "best_loss": float("inf"),
          "best_epoch": -1, "patience": 0, "history": []}
    ck = _load_state(run_dir)
    if ck is not None:
        if ck["meta"] != meta:
            raise RuntimeError(f"checkpoint in {run_dir} has a different configuration")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        st = ck["st"]
        print(f"resumed at epoch {st['epoch'] + 1}, batch {st['batch']}/{steps}")

    def save():
        _save_state({"meta": meta, "model": net.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict(), "rng": _rng_state(), "st": st,
                     "timestamp": datetime.now().isoformat()}, run_dir)

    print(json.dumps(meta, indent=1), flush=True)
    last_ck = time.time()
    while st["epoch"] < MAX_EPOCHS:
        ep = st["epoch"]
        order = np.random.default_rng(v3.SEED + ep).permutation(len(train))[:steps * BATCH_SIZE]
        batches = [order[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].tolist() for i in range(st["batch"], steps)]
        net.train()
        t_ep = time.time()
        for imgs, lab, ok, _, _ in threaded_batches(train, batches, threads):
            x = _normalise(imgs[ok.bool()], mean, std)
            y = lab[ok.bool()].float().to(DEVICE)
            flip = torch.rand(len(x), device=DEVICE) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(net(x).float().flatten(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), v3.MAX_GRAD_NORM)
            scaler.step(opt)
            scaler.update()
            st["batch"] += 1
            st["loss_sum"] += float(loss) * len(y)
            st["n_seen"] += len(y)
            if time.time() - last_ck > ckpt_minutes * 60:
                save()
                last_ck = time.time()
                rate = st["batch"] * BATCH_SIZE / max(time.time() - t_ep, 1e-9)
                print(f"[spatial_ft] epoch {ep + 1} batch {st['batch']}/{steps} | loss "
                      f"{st['loss_sum'] / max(st['n_seen'], 1):.4f} | {rate:.0f} img/s | checkpoint "
                      f"{datetime.now():%H:%M:%S}", flush=True)
            if deadline and time.time() > deadline:
                save()
                print("[spatial_ft] time budget reached; paused cleanly", flush=True)
                return "paused"

        rep = evaluate(net, val, threads, mean, std)
        row = {"epoch": ep + 1, "train_loss": st["loss_sum"] / max(st["n_seen"], 1), **rep,
               "epoch_seconds": time.time() - t_ep}
        improved = rep["val_loss"] < st["best_loss"] - MIN_DELTA
        if improved:
            st.update(best_loss=rep["val_loss"], best_epoch=ep + 1, patience=0)
            v3.atomic_torch_save({"effnet": net.state_dict(), "meta": {**meta, "epoch": ep + 1, **rep}},
                                 run_dir / "best.pth")
        else:
            st["patience"] += 1
        st["history"].append(row)
        st.update(epoch=ep + 1, batch=0, loss_sum=0.0, n_seen=0)
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(st["history"])
        print(f"[spatial_ft] epoch {ep + 1}/{MAX_EPOCHS} | train loss {row['train_loss']:.4f} | val loss "
              f"{rep['val_loss']:.5f} | acc {rep['val_acc']:.4f} | AUC {rep['val_auc']:.4f} | "
              f"{row['epoch_seconds']:.0f}s | patience {st['patience']}{' *best*' if improved else ''}",
              flush=True)
        save()
        last_ck = time.time()
        if st["patience"] >= PATIENCE:
            break
        if deadline and time.time() > deadline:
            print("[spatial_ft] time budget reached; paused cleanly", flush=True)
            return "paused"

    done = {"best_epoch": st["best_epoch"], "best_val_loss": st["best_loss"], "epochs_run": st["epoch"],
            "finished": datetime.now().isoformat()}
    v3.atomic_write_text(json.dumps(done, indent=1), run_dir / "DONE.json")
    print(f"[spatial_ft] done: {done}", flush=True)
    return "done"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Fine-tune the spatial EfficientNet-B0 (fp-v4.1)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=12)   # 8 threads decode ~790 img/s, 12 ~1,700
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None, help="views per epoch (smoke test)")
    a = ap.parse_args(argv)
    out = a.out or section("outputs")["results"] / "training" / "spatial_ft"
    deadline = time.time() + a.hours * 3600 if a.hours else None
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if finetune(out, a.threads, a.ckpt_minutes, deadline, a.limit) == "paused":
        print("Paused. Rerun the same command to continue.")
