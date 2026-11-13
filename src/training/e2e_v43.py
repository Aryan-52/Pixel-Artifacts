"""fp-v4.3: the spatial EfficientNet-B0 trained end to end inside the five-pillar model (decision D-23).

fp-v4.1 fine-tuned the EfficientNet on its own, froze it, and trained the fusion head on stored
features of one fixed blur/JPEG copy. Here the network runs live inside the fusion model, so the
head's loss trains it, and every image gets a fresh degradation each epoch (the v4.2 recipe in
src/training/spatial_finetune_v2.py). The other four pillars are read from the fp-v4.1 store for
the same image (its clean or blur/JPEG row, chosen at random per epoch), as the team's v3 hybrid
caches its non-deep branches.

The live 1952-d spatial descriptor is the fp-v3 extract_spatial_features output, standardised
with the FIXED fp-v4.1 train stats and clipped to +-10, so the packaged model loads in the app
exactly like fp-v4.1 (spatial_effnet.pth + head.pth).

Protocol (D-5, D-23): batch 64, at most 50 epochs of 200,000 images, early stopping (patience 3)
on the mean of clean and degraded ProGAN-validation BCE; ProGAN data only; temperature on clean
validation (T >= 1); threshold 0.5; the test split is scored once, in finalize().

    python scripts/train_e2e_v43.py [--hours H] [--smoke]

Run directory (results/training/e2e_v43/): latest.pth / latest.prev.pth (resumable state),
best_effnet.pth + best_model.pth (selected epoch), history.csv, val_degraded_X.npy (the four
cached pillars of the degraded validation copy), test_report.json, val_report.json,
test_predictions.npz, DONE.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.features.extraction import split_root, threaded_batches
from src.features.spatial_ft import extract_raw
from src.models import five_pillar_v3 as v3
from src.paths import section
from src.training.head import (Store, _load_state, _metrics, _overlap_rows, _report, _rng_state,
                               _save_state, _set_rng_state)
from src.training.spatial_finetune import _normalise, _Views, build_effnet
from src.training.spatial_finetune_v2 import _Degraded

MODEL_VERSION = "fp-v4.3"
BATCH_SIZE = 64            # D-5
MAX_EPOCHS = 50            # D-5
EPOCH_SAMPLES = 200_000
PATIENCE = 3
MIN_DELTA = 1e-4
LR = 1e-4
WEIGHT_DECAY = 1e-2
DEVICE = v3.DEVICE
EVAL_BS = 256

TRAINING = section("outputs")["results"] / "training"
STORE = TRAINING / "feature_store_ft"                       # fp-v4.1 store
INIT_EFFNET = TRAINING / "spatial_ft" / "best.pth"          # fp-v4.1 fine-tuned EfficientNet
INIT_HEAD = TRAINING / "runs_ft" / "full" / "best_model.pth"  # fp-v4.1 head
RUN_DIR = TRAINING / "e2e_v43"


class E2E(nn.Module):
    """Live EfficientNet spatial pillar + cached four pillars -> the unchanged FivePillarDetector."""

    def __init__(self, effnet, head, spatial_stats, cols):
        super().__init__()
        self.effnet, self.head, self.cols = effnet, head, cols
        self.register_buffer("s_mean", spatial_stats["mean"].float().clone())
        self.register_buffer("s_std", spatial_stats["std"].float().clone())
        self.register_buffer("im_mean", torch.tensor(v3.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor(v3.IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, imgs_uint8, cached, mask=None):
        x = _normalise(imgs_uint8, self.im_mean, self.im_std)
        sp = v3.extract_spatial_features(self.effnet, x)
        feats = {"spatial": ((sp - self.s_mean) / self.s_std).clamp(-v3.FEATURE_CLIP, v3.FEATURE_CLIP)}
        for k, s in self.cols.items():
            if k != "spatial":
                feats[k] = cached[:, s].float()
        return self.head(feats, mask)


def _entries(store: Store, split: str, copy: int = 0):
    """(absolute path, label, copy) per entry, and that entry's store row for the given copy."""
    root, rows, ents = split_root(split), [], []
    for r in store.index(split):
        if int(r["copy"]) == copy:
            ents.append((str(root / r["relpath"]), int(r["label"]), copy))
            rows.append(int(r["row"]))
    return ents, np.array(rows)


def _train_rows(store: Store):
    """one entry per image (its clean row's order) and, per image, its clean and blur/JPEG store rows.
    Copies of one image share its relpath (copy 0 in the first half of the store, copy 1 in the second)."""
    by = {}
    for r in store.index("train"):
        by.setdefault(r["relpath"], {})[int(r["copy"])] = (int(r["row"]), int(r["label"]))
    root, ents, r0, r1 = split_root("train"), [], [], []
    for rel, c in sorted(by.items(), key=lambda kv: kv[1][0][0]):
        ents.append((str(root / rel), c[0][1], 0))
        r0.append(c[0][0])
        r1.append(c.get(1, c[0])[0])
    return ents, np.array(r0), np.array(r1)


def _cached_block(X, rows):
    order = np.argsort(rows)
    out = np.empty((len(rows), X.shape[1]), np.float16)
    out[order] = X[rows[order]]
    return torch.from_numpy(out).to(DEVICE, non_blocking=True)


@torch.inference_mode()
def degraded_val_cache(store: Store, out: Path, threads: int, limit=None) -> np.ndarray:
    """Four cached pillars of the fixed degraded validation copy, standardised with store stats."""
    if out.exists():
        return np.load(out, mmap_mode="r")
    ents, _ = _entries(store, "val")
    ents = ents[:limit] if limit else ents
    ds = _Degraded(ents)                                   # path-seeded degradation, centre crop
    v3.PRETRAINED_DIR = section("outputs")["pretrained"]  # the shared weight cache, as extraction.py
    enc = v3.get_encoders()
    X = np.zeros((len(ents), store.meta["offsets"]["semantic"][1]), np.float16)
    keys = [k for k in v3.ALL_PILLARS if k != "spatial"]
    batches = [list(range(i, min(i + v3.BATCH_SIZE_FEATURE, len(ds)))) for i in range(0, len(ds), v3.BATCH_SIZE_FEATURE)]
    for imgs, _, ok, i, _ in threaded_batches(ds, batches, threads):
        if not bool(ok.all()):
            raise RuntimeError(f"degraded validation decode failed at {i.tolist()}")
        raw = v3.extract_batch(imgs, enc)
        std = v3.standardize(raw, store.stats, keys)
        for k in keys:
            lo, hi = store.meta["offsets"][k]
            X[int(i[0]):int(i[-1]) + 1, lo:hi] = std[k].half().cpu().numpy()
    v3.release_encoders()
    np.save(out, X)
    print(f"degraded validation cache: {X.shape} -> {out}", flush=True)
    return X


@torch.inference_mode()
def predict(model, ds, X, rows, threads):
    """logits, gates over ds (image i pairs with cached row rows[i])."""
    model.eval()
    batches = [list(range(i, min(i + EVAL_BS, len(ds)))) for i in range(0, len(ds), EVAL_BS)]
    L, G = [], []
    for imgs, _, ok, i, _ in threaded_batches(ds, batches, threads):
        if not bool(ok.all()):
            raise RuntimeError(f"image decode failed at {i.tolist()}")
        cb = _cached_block(X, rows[i.numpy()])
        with torch.autocast("cuda", dtype=torch.float16, enabled=v3.USE_AMP):
            lg, g = model(imgs, cb)
        L.append(lg.float().cpu())
        G.append(g.float().cpu())
    return torch.cat(L), torch.cat(G)


def _build(store: Store):
    head_b = torch.load(INIT_HEAD, map_location="cpu", weights_only=False)
    head = v3.FivePillarDetector(head_b["dims"], head_b["active"], head_b["model_cfg"])
    head.load_state_dict(head_b["model_state_dict"])
    effnet = build_effnet()
    effnet.load_state_dict(torch.load(INIT_EFFNET, map_location="cpu", weights_only=False)["effnet"])
    for p in effnet.parameters():
        p.requires_grad_(True)
    cols = store.cols(v3.ALL_PILLARS)
    return E2E(effnet, head, store.stats["spatial"], cols).to(DEVICE), head_b


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train(run_dir: Path = RUN_DIR, threads=12, ckpt_minutes=5.0, deadline=None, smoke=False) -> str:
    """Returns "done" or "paused" (deadline reached; rerun to resume)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE.json").exists():
        print("fp-v4.3 already done")
        return "done"
    v3.seed_everything(v3.SEED)
    store = Store(STORE)
    Xtr, ytr, _ = store.split("train")
    Xva, yva, _ = store.split("val")
    tr_ents, r0, r1 = _train_rows(store)
    va_ents, va_rows = _entries(store, "val")
    vlim = 512 if smoke else None
    va_ents, va_rows = (va_ents[:vlim], va_rows[:vlim]) if vlim else (va_ents, va_rows)
    yv = np.array([l for _, l, _ in va_ents], np.float32)
    Xvd = degraded_val_cache(store, run_dir / "val_degraded_X.npy", threads, vlim)
    vd_rows = np.arange(len(va_ents))
    val_clean, val_deg = _Views(va_ents), _Degraded(va_ents)

    epoch_samples = 2048 if smoke else EPOCH_SAMPLES
    max_epochs = 2 if smoke else MAX_EPOCHS
    steps = epoch_samples // BATCH_SIZE
    pos_weight = float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)
    loss_fn = v3.make_loss_fn(pos_weight)

    model, head_b = _build(store)
    params = [p for n, p in model.named_parameters() if not n.startswith("effnet.classifier")]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=LR, fused=DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=v3.USE_AMP)
    meta = {"model_version": MODEL_VERSION, "init_effnet": str(INIT_EFFNET), "init_head": str(INIT_HEAD),
            "store": str(STORE), "store_hash": store.meta["feature_code_hash"], "batch_size": BATCH_SIZE,
            "max_epochs": max_epochs, "epoch_samples": epoch_samples, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "seed": v3.SEED, "n_train_images": len(tr_ents), "n_val": len(va_ents),
            "pillar_dropout": v3.PILLAR_DROPOUT, "smoke": smoke,
            "selection": "mean of clean and degraded ProGAN-validation BCE (full model)"}
    st = {"epoch": 0, "batch": 0, "loss_sum": 0.0, "n_seen": 0, "best_loss": float("inf"),
          "best_epoch": -1, "patience": 0, "history": []}
    ck = _load_state(run_dir)
    if ck is not None:
        if ck["meta"] != meta:
            raise RuntimeError(f"checkpoint in {run_dir} has a different configuration")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        st = ck["st"]
        print(f"resumed at epoch {st['epoch'] + 1}, batch {st['batch']}/{steps}", flush=True)

    def save():
        _save_state({"meta": meta, "model": model.state_dict(), "opt": opt.state_dict(),
                     "scaler": scaler.state_dict(), "rng": _rng_state(), "st": st,
                     "timestamp": datetime.now().isoformat()}, run_dir)

    def val_loss(ds, X, rows):
        lg, g = predict(model, ds, X, rows, threads)
        return float(F.binary_cross_entropy_with_logits(lg, torch.from_numpy(yv))), lg, g

    if st["epoch"] == 0 and st["batch"] == 0 and not st["history"]:
        c0, _, _ = val_loss(val_clean, Xva, va_rows)
        d0, _, _ = val_loss(val_deg, Xvd, vd_rows)
        print(f"[fp-v4.3] start (fp-v4.1 weights): clean val BCE {c0:.5f}, degraded val BCE {d0:.5f}", flush=True)

    print(json.dumps(meta, indent=1), flush=True)
    last_ck = time.time()
    while st["epoch"] < max_epochs:
        ep = st["epoch"]
        train_ds = _Degraded(tr_ents, epoch=ep)
        order = np.random.default_rng(v3.SEED + ep).permutation(len(train_ds))[:steps * BATCH_SIZE]
        pick = np.random.default_rng(v3.SEED * 7919 + ep).random(len(train_ds)) < 0.5   # cached copy per image
        batches = [order[i * BATCH_SIZE:(i + 1) * BATCH_SIZE].tolist() for i in range(st["batch"], steps)]
        model.train()
        t_ep = time.time()
        for imgs, lab, ok, idx, _ in threaded_batches(train_ds, batches, threads):
            keep = ok.bool()
            idx = idx[keep].numpy()
            rows = np.where(pick[idx], r1[idx], r0[idx])
            cb = _cached_block(Xtr, rows)
            y = lab[keep].float().to(DEVICE)
            mask = v3.sample_pillar_mask(len(y), len(v3.ALL_PILLARS), v3.PILLAR_DROPOUT, DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=v3.USE_AMP):
                logits, gates = model(imgs[keep], cb, mask)
            loss, _, _ = loss_fn(logits, y, gates)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, v3.MAX_GRAD_NORM)
            scaler.step(opt)
            scaler.update()
            st["batch"] += 1
            st["loss_sum"] += float(loss) * len(y)
            st["n_seen"] += len(y)
            if time.time() - last_ck > ckpt_minutes * 60:
                save()
                last_ck = time.time()
                rate = st["n_seen"] / max(time.time() - t_ep, 1e-9)
                print(f"[fp-v4.3] epoch {ep + 1} batch {st['batch']}/{steps} | loss "
                      f"{st['loss_sum'] / max(st['n_seen'], 1):.4f} | {rate:.0f} img/s | checkpoint "
                      f"{datetime.now():%H:%M:%S}", flush=True)
            if deadline and time.time() > deadline:
                save()
                print("[fp-v4.3] time budget reached; paused cleanly", flush=True)
                return "paused"

        clean, lc, gc = val_loss(val_clean, Xva, va_rows)
        deg, ld, _ = val_loss(val_deg, Xvd, vd_rows)
        sel = 0.5 * (clean + deg)
        mc, md = v3.metrics_from(yv, lc), v3.metrics_from(yv, ld)
        row = {"epoch": ep + 1, "train_loss": st["loss_sum"] / max(st["n_seen"], 1), "val_loss": sel,
               "clean_val_loss": clean, "clean_val_acc": mc["accuracy"], "clean_val_auc": mc["roc_auc"],
               "degraded_val_loss": deg, "degraded_val_acc": md["accuracy"], "degraded_val_auc": md["roc_auc"],
               "epoch_seconds": time.time() - t_ep,
               **{f"gate_{k}": g for k, g in zip(v3.ALL_PILLARS, gc.mean(0).tolist())}}
        improved = sel < st["best_loss"] - MIN_DELTA
        if improved:
            st.update(best_loss=sel, best_epoch=ep + 1, patience=0)
            v3.atomic_torch_save({"effnet": model.effnet.state_dict(), "meta": {**meta, **row}},
                                 run_dir / "best_effnet.pth")
            v3.atomic_torch_save({**head_b, "model_state_dict": model.head.state_dict(), "model_version": MODEL_VERSION,
                                  "epoch": ep + 1, "val_loss": sel, "selection_metric": meta["selection"],
                                  "finalized": False, "created": datetime.now().isoformat()},
                                 run_dir / "best_model.pth")
        else:
            st["patience"] += 1
        st["history"].append(row)
        st.update(epoch=ep + 1, batch=0, loss_sum=0.0, n_seen=0)
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(st["history"])
        print(f"[fp-v4.3] epoch {ep + 1}/{max_epochs} | train loss {row['train_loss']:.4f} | val loss {sel:.5f} "
              f"(clean {clean:.5f} acc {mc['accuracy']:.4f}; degraded {deg:.5f} acc {md['accuracy']:.4f} "
              f"AUC {md['roc_auc']:.4f}) | {row['epoch_seconds']:.0f}s | patience {st['patience']}"
              f"{' *best*' if improved else ''}", flush=True)
        save()
        last_ck = time.time()
        if st["patience"] >= PATIENCE:
            break
        if deadline and time.time() > deadline:
            print("[fp-v4.3] time budget reached; paused cleanly", flush=True)
            return "paused"

    if smoke:
        print("[fp-v4.3] smoke run finished (no test scoring)", flush=True)
        return "done"
    finalize(run_dir, threads)
    return "done"


def finalize(run_dir: Path = RUN_DIR, threads=12):
    """Temperature on clean val (T >= 1), then the single scoring pass over the test split."""
    store = Store(STORE)
    model, _ = _build(store)
    model.effnet.load_state_dict(torch.load(run_dir / "best_effnet.pth", map_location="cpu", weights_only=False)["effnet"])
    best = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=False)
    model.head.load_state_dict(best["model_state_dict"])
    model.eval()
    sha = _sha256(run_dir / "best_effnet.pth")

    Xva, _, _ = store.split("val")
    va_ents, va_rows = _entries(store, "val")
    yv = np.array([l for _, l, _ in va_ents], np.float32)
    lv, gv = predict(model, _Views(va_ents), Xva, va_rows, threads)
    T, t_unc = v3.fit_temperature(lv, yv)
    vidx = store.index("val")
    vcat = np.array([r["relpath"].split("/")[0] for r in vidx if int(r["copy"]) == 0])
    val_rep = _report(yv, lv.numpy(), vcat)
    val_rep.update(best_epoch=best["epoch"], temperature=T, temperature_unconstrained=t_unc,
                   mean_fusion_weights=dict(zip(v3.ALL_PILLARS, gv.mean(0).tolist())),
                   note="in-distribution ProGAN validation; used for selection; not zero-shot")
    v3.atomic_write_text(json.dumps(val_rep, indent=1), run_dir / "val_report.json")

    # test: live spatial pillar of the selected network on the unchanged centre crop, scored once
    Xte, yte, _ = store.split("test")
    tidx = store.index("test")
    root = split_root("test")
    te_ents = [(str(root / r["relpath"]), int(r["label"]), int(r["copy"])) for r in tidx]
    lt, gt = predict(model, _Views(te_ents), Xte, np.arange(len(te_ents)), threads)
    lt = lt.numpy()
    np.savez_compressed(run_dir / "test_predictions.npz", logits=lt, gates=gt.numpy(), y=yte, row=np.arange(len(yte)))
    gen = np.array([r["relpath"].split("/")[0] for r in tidx])
    rep = _report(yte, lt, gen)
    per = rep["per_group"]
    rep["mean_generator_acc"] = float(np.mean([m["accuracy"] for m in per.values()]))
    rep["worst_generator_acc"] = float(min(m["accuracy"] for m in per.values()))
    rep["mean_generator_auc"] = float(np.mean([m["roc_auc"] for m in per.values()]))
    keep = np.setdiff1d(np.arange(len(yte)), _overlap_rows(store, tidx))
    rep["sensitivity_without_train_val_overlap"] = {"n_removed": int(len(yte) - len(keep)), **_metrics(yte[keep], lt[keep])}
    rep.update(threshold=v3.CLASSIFICATION_THRESHOLD, temperature_used_for_threshold=False,
               note="zero-shot: test generators never used for training, selection or threshold")
    v3.atomic_write_text(json.dumps(rep, indent=1), run_dir / "test_report.json")

    best.update(temperature=T, temperature_unconstrained=t_unc, finalized=True,
                feature_code_hash=f"{store.meta['feature_code_hash'].split('+')[0]}+spatial_ft:{sha[:12]}")
    v3.atomic_torch_save(best, run_dir / "best_model.pth")
    o = rep["overall"]
    done = {"model_version": MODEL_VERSION, "best_epoch": best["epoch"], "test_acc": o["accuracy"],
            "test_auc": o["roc_auc"], "effnet_sha256": sha, "finished": datetime.now().isoformat()}
    v3.atomic_write_text(json.dumps(done, indent=1), run_dir / "DONE.json")
    print(f"[fp-v4.3] TEST acc {o['accuracy']:.4f} | AUC {o['roc_auc']:.4f} | AP {o['average_precision']:.4f} | "
          f"real {o['real_acc']:.3f} fake {o['fake_acc']:.3f} | mean gen acc {rep['mean_generator_acc']:.4f}", flush=True)
