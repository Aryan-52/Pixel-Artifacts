"""fp-v4.2 spatial pillar: fine-tune EfficientNet-B0 with fresh degradations every epoch (decision D-22).

fp-v4.1 (src/training/spatial_finetune.py) trained on two FIXED views per image (clean + one
path-seeded blur/JPEG copy) and selected on clean ProGAN validation, which saturated at
acc 0.9997 / AUC 1.0000 by epoch 3, so early stopping could not tell a ProGAN-memorising epoch
from one that generalises. The comparison with the team's stage1 model (83.2 % vs 70.8 % test
accuracy for the texture branch alone) points at the augmentation. Changes, everything else as
fp-v4.1 (batch 64, at most 50 epochs, AdamW 1e-4, early stopping, ProGAN only, test untouched):

* every training image is re-degraded with a new random draw each epoch (seeded by epoch and
  row, so a resumed run sees the same images): downscale p 0.3 (factor 0.5-0.9, then back to the
  original size), Gaussian blur p 0.5 (sigma 0-3), JPEG p 0.8 (quality 30-100), random 224 crop
  instead of the centre crop, horizontal flip p 0.5;
* the train split is the 720k clean originals (copy 0); the augmentation replaces the fixed copy;
* selection on the mean of two ProGAN-validation BCEs: clean, and a fixed degraded copy (same
  degradations, path-seeded, centre crop). Both are in-distribution ProGAN images.

Run directory (results/training/spatial_ft_v2/): same files as fp-v4.1.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import Dataset

from src.features.extraction import load_samples, threaded_batches
from src.models import five_pillar_v3 as v3
from src.paths import section
from src.training.head import _load_state, _rng_state, _save_state, _set_rng_state
from src.training.spatial_finetune import _abs_entries, _normalise, build_effnet, evaluate

BATCH_SIZE = 64            # D-5
MAX_EPOCHS = 50            # D-5
EPOCH_SAMPLES = 200_000    # images per epoch (seeded draw without replacement), as fp-v4.1
PATIENCE = 3
MIN_DELTA = 1e-4
LR = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = v3.DEVICE

P_DOWNSCALE, DOWNSCALE = 0.3, (0.5, 0.9)
P_BLUR, BLUR_SIGMA = 0.5, (0.0, 3.0)
P_JPEG, JPEG_Q = 0.8, (30, 100)
_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Downscale-and-restore, blur, JPEG: what images meet on the way to the web."""
    w, h = img.size
    if rng.random() < P_DOWNSCALE:
        s = rng.uniform(*DOWNSCALE)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), _BICUBIC).resize((w, h), _BICUBIC)
    if rng.random() < P_BLUR:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(*BLUR_SIGMA)))
    if rng.random() < P_JPEG:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=rng.randint(*JPEG_Q))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def _crop(img, rng):
    size = v3.IMAGE_SIZE
    w, h = img.size
    if rng is None or min(w, h) < size:
        return v3.center_crop_or_upscale(img, size)
    x, y = rng.randint(0, w - size), rng.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if rng.random() < 0.5 else img


class _Degraded(Dataset):
    """(path, label) -> degraded uint8 224 crop, in the 5-tuple shape threaded_batches expects.
    train=True: random crop + flip, draw seeded by (epoch, row); False: centre crop, seeded by path."""

    def __init__(self, entries, epoch=None):
        self.entries, self.epoch = entries, epoch

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, label, _ = self.entries[i]
        train = self.epoch is not None
        rng = random.Random(f"{v3.SEED}|{self.epoch}|{i}" if train else f"{v3.SEED}|valdeg|{path}")
        try:
            img = Image.open(path)
            img.load()
            img = _crop(degrade(img.convert("RGB"), rng), rng if train else None)
            t = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()
            return t, float(label), 1, i, 1
        except Exception:
            return torch.zeros(3, v3.IMAGE_SIZE, v3.IMAGE_SIZE, dtype=torch.uint8), float(label), 0, i, 1


def finetune(run_dir: Path, threads=12, ckpt_minutes=5.0, deadline=None, limit=None) -> str:
    """Returns "done" or "paused" (deadline reached; rerun to resume)."""
    from src.training.spatial_finetune import _Views

    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE.json").exists():
        print("spatial fine-tune v2 already done")
        return "done"
    v3.seed_everything(v3.SEED)
    train_entries = _abs_entries("train", [(p, l, 0) for p, l, _ in load_samples("train")])
    val_entries = _abs_entries("val", [(p, l, 0) for p, l, _ in load_samples("val")])
    val_clean, val_deg = _Views(val_entries), _Degraded(val_entries)
    epoch_samples = min(limit or EPOCH_SAMPLES, len(train_entries))
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
            "n_train_images": len(train_entries), "n_val": len(val_entries), "lr": LR,
            "weight_decay": WEIGHT_DECAY, "patience": PATIENCE, "seed": v3.SEED,
            "augment": {"downscale": [P_DOWNSCALE, *DOWNSCALE], "blur": [P_BLUR, *BLUR_SIGMA],
                        "jpeg": [P_JPEG, *JPEG_Q], "crop": "random 224", "hflip": 0.5,
                        "fresh_every_epoch": True},
            "selection": "mean of clean and degraded ProGAN-validation BCE"}
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
        train = _Degraded(train_entries, epoch=ep)
        order = np.random.default_rng(v3.SEED + ep).permutation(len(train))[:steps * BATCH_SIZE]
        batches = [order[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].tolist() for i in range(st["batch"], steps)]
        net.train()
        t_ep = time.time()
        for imgs, lab, ok, _, _ in threaded_batches(train, batches, threads):
            x = _normalise(imgs[ok.bool()], mean, std)
            y = lab[ok.bool()].float().to(DEVICE)
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
                rate = (st["batch"] * BATCH_SIZE) / max(time.time() - t_ep, 1e-9)
                print(f"[spatial_ft_v2] epoch {ep + 1} batch {st['batch']}/{steps} | loss "
                      f"{st['loss_sum'] / max(st['n_seen'], 1):.4f} | {rate:.0f} img/s | checkpoint "
                      f"{datetime.now():%H:%M:%S}", flush=True)
            if deadline and time.time() > deadline:
                save()
                print("[spatial_ft_v2] time budget reached; paused cleanly", flush=True)
                return "paused"

        clean = evaluate(net, val_clean, threads, mean, std)
        deg = evaluate(net, val_deg, threads, mean, std)
        sel = 0.5 * (clean["val_loss"] + deg["val_loss"])
        row = {"epoch": ep + 1, "train_loss": st["loss_sum"] / max(st["n_seen"], 1), "val_loss": sel,
               **{f"clean_{k}": v for k, v in clean.items()}, **{f"degraded_{k}": v for k, v in deg.items()},
               "epoch_seconds": time.time() - t_ep}
        improved = sel < st["best_loss"] - MIN_DELTA
        if improved:
            st.update(best_loss=sel, best_epoch=ep + 1, patience=0)
            v3.atomic_torch_save({"effnet": net.state_dict(), "meta": {**meta, "epoch": ep + 1, **row}},
                                 run_dir / "best.pth")
        else:
            st["patience"] += 1
        st["history"].append(row)
        st.update(epoch=ep + 1, batch=0, loss_sum=0.0, n_seen=0)
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(st["history"])
        print(f"[spatial_ft_v2] epoch {ep + 1}/{MAX_EPOCHS} | train loss {row['train_loss']:.4f} | val loss "
              f"{sel:.5f} (clean {clean['val_loss']:.5f} acc {clean['val_acc']:.4f}, degraded "
              f"{deg['val_loss']:.5f} acc {deg['val_acc']:.4f} AUC {deg['val_auc']:.4f}) | "
              f"{row['epoch_seconds']:.0f}s | patience {st['patience']}{' *best*' if improved else ''}", flush=True)
        save()
        last_ck = time.time()
        if st["patience"] >= PATIENCE:
            break
        if deadline and time.time() > deadline:
            print("[spatial_ft_v2] time budget reached; paused cleanly", flush=True)
            return "paused"

    done = {"best_epoch": st["best_epoch"], "best_val_loss": st["best_loss"], "epochs_run": st["epoch"],
            "finished": datetime.now().isoformat()}
    v3.atomic_write_text(json.dumps(done, indent=1), run_dir / "DONE.json")
    print(f"[spatial_ft_v2] done: {done}", flush=True)
    return "done"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Fine-tune the spatial EfficientNet-B0 with fresh degradations (fp-v4.2)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None, help="images per epoch (smoke test)")
    a = ap.parse_args(argv)
    out = a.out or section("outputs")["results"] / "training" / "spatial_ft_v2"
    deadline = time.time() + a.hours * 3600 if a.hours else None
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if finetune(out, a.threads, a.ckpt_minutes, deadline, a.limit) == "paused":
        print("Paused. Rerun the same command to continue.")
