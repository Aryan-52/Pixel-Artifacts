"""Resumable fusion-head training on the feature store (fp-v4.0 and its pillar ablations).

Model, loss, optimiser, warmup + cosine schedule, pillar dropout, selection on clean validation
BCE, temperature rule and threshold are the fp-v3.0 code/constants (src/models/five_pillar_v3.py).
Changes (DECISION_LOG): batch 64 and max 50 epochs (D-5); rows streamed from the fp16 store
instead of an in-VRAM tensor (D-2); checkpoints also mid-epoch, so an interruption costs at most
`ckpt_minutes` of work; the test split is scored once, after training, and never used for choices.

Run directory <runs>/<tag>/:
  latest.pth, latest.prev.pth   full training state (atomic; prev = fallback)
  best_model.pth                 deployable bundle (fp-v3 bundle format + "model_version")
  history.csv, val_report.json, test_report.json, test_predictions.npz, DONE.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models import five_pillar_v3 as v3

MODEL_VERSION = "fp-v4.0"
BATCH_SIZE = 64            # D-5 (guide's instruction)
MAX_EPOCHS = 50            # D-5
SUPER_BATCHES = 512        # minibatches gathered per host->GPU transfer (same order as a plain perm)
DEVICE = v3.DEVICE


def run_tag(active) -> str:
    return "full" if list(active) == v3.ALL_PILLARS else "+".join(active)


class Store:
    """Read-only view of the standardised store; rows are memory-mapped, never fully loaded."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "store_meta.json").read_text())
        self.stats = torch.load(self.root / "stats.pt")["stats"]

    def split(self, name):
        X = np.load(self.root / f"{name}_X.npy", mmap_mode="r")
        y = np.load(self.root / f"{name}_y.npy")
        cp = np.load(self.root / f"{name}_copy.npy")
        return X, y, cp

    def index(self, name):
        with open(self.root / f"{name}_index.csv", newline="") as f:
            return list(csv.DictReader(f))

    def cols(self, active):
        return {k: slice(*self.meta["offsets"][k]) for k in active}


def _to_feats(block, cols):
    return {k: block[:, s] for k, s in cols.items()}


@torch.no_grad()
def predict_rows(model, X, cols, rows=None, bs=8192):
    """Logits and gates for X[rows] (all rows if None), read in contiguous chunks."""
    model.eval()
    n = len(X) if rows is None else len(rows)
    L, G = [], []
    for i in range(0, n, bs):
        sel = slice(i, i + bs) if rows is None else np.sort(rows[i:i + bs])
        block = torch.from_numpy(np.array(X[sel])).to(DEVICE).float()
        lg, g = model(_to_feats(block, cols))
        L.append(lg.float().cpu())
        G.append(g.float().cpu())
    return torch.cat(L), torch.cat(G)


def _rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(s):
    torch.set_rng_state(s["torch"].cpu())
    torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _save_state(state, run_dir):
    latest, prev = run_dir / "latest.pth", run_dir / "latest.prev.pth"
    tmp = latest.with_name("latest.pth.tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    if latest.exists():
        latest.replace(prev)
    tmp.replace(latest)


def _load_state(run_dir):
    for p in (run_dir / "latest.pth", run_dir / "latest.prev.pth"):
        if p.exists():
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"{p.name} unreadable ({e}); trying fallback")
    return None


def train_run(store: Store, active, runs_root: Path, ckpt_minutes=5.0, deadline=None,
              max_epochs=MAX_EPOCHS) -> str:
    """Train one head. Returns "done" or "paused" (deadline reached; rerun to resume)."""
    active = v3.validate_active_pillars(list(active))
    tag = run_tag(active)
    run_dir = Path(runs_root) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "DONE.json").exists():
        print(f"[{tag}] already done")
        return "done"

    v3.seed_everything(v3.SEED)
    cols = store.cols(active)
    dims = {k: store.meta["dims"][k] for k in active}
    Xtr, ytr, _ = store.split("train")
    Xva, yva, cpva = store.split("val")
    clean_val = np.where(cpva == 0)[0]
    N = len(Xtr)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    pos_weight = float((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1)

    cfg = v3.model_cfg()
    model = v3.FivePillarDetector(dims, active, cfg).to(DEVICE)
    loss_fn = v3.make_loss_fn(pos_weight)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": v3.WEIGHT_DECAY},
                             {"params": no_decay, "weight_decay": 0.0}], lr=v3.LEARNING_RATE,
                            fused=DEVICE.type == "cuda")     # same update rule, fewer kernel launches

    def lr_lambda(ep):
        if ep < v3.WARMUP_EPOCHS:
            return (ep + 1) / float(v3.WARMUP_EPOCHS)
        prog = (ep - v3.WARMUP_EPOCHS) / float(max(1, max_epochs - v3.WARMUP_EPOCHS))
        return v3.MIN_LR_FRAC + (1 - v3.MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=v3.USE_AMP)
    model_version = store.meta.get("model_version", MODEL_VERSION)   # fp-v4.1 stores say so
    meta = {"model_version": model_version, "active": active, "dims": dims,
            "feature_code_hash": store.meta["feature_code_hash"], "train_sig": store.meta["train_sig"],
            "batch_size": BATCH_SIZE, "max_epochs": max_epochs, "n_train": N, "model_cfg": cfg}

    st = {"epoch": 0, "batch": 0, "acc": [0.0, 0.0, 0.0, 0], "skipped": 0, "best_loss": float("inf"),
          "best_epoch": -1, "patience": 0, "history": [], "finished": False, "epoch_t": 0.0}
    ck = _load_state(run_dir)
    if ck is not None:
        if ck["meta"] != meta:
            raise RuntimeError(f"[{tag}] checkpoint belongs to a different configuration; "
                               f"move {run_dir} away to start fresh")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        st = ck["st"]
        print(f"[{tag}] resumed at epoch {st['epoch'] + 1}, batch {st['batch']}")

    def state_dict():
        return {"meta": meta, "model": model.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(),
                "st": st, "timestamp": datetime.now().isoformat()}

    def bundle(epoch, val_loss):
        return {"bundle_format": 3, "code_version": v3.CODE_VERSION, "model_version": model_version,
                "feature_code_hash": store.meta["feature_code_hash"],
                "model_state_dict": model.state_dict(), "model_cfg": cfg,
                "active": active, "pillar_order": active, "all_pillars": v3.ALL_PILLARS,
                "dims": dims, "raw_concat_dim": sum(dims.values()), "projected_dim": v3.PILLAR_DIM,
                "stats": {k: {"mean": store.stats[k]["mean"], "std": store.stats[k]["std"]}
                          for k in v3.ALL_PILLARS},
                "feature_clip": v3.FEATURE_CLIP, "preprocessing": v3.preprocessing_spec(),
                "pretrained": v3.pretrained_spec(), "library_versions": v3.library_versions(),
                "threshold": v3.CLASSIFICATION_THRESHOLD, "temperature": 1.0,
                "temperature_unconstrained": None, "meta": meta, "epoch": epoch,
                "val_loss": val_loss, "selection_metric": "clean_val_bce_loss",
                "finalized": False, "created": datetime.now().isoformat()}

    n_batches = math.ceil(N / BATCH_SIZE)
    n_act = len(active)
    col_idx = np.concatenate([np.arange(s.start, s.stop) for s in cols.values()])
    n_cols, local_cols, o = len(col_idx), {}, 0
    for k, s in cols.items():
        local_cols[k] = slice(o, o + s.stop - s.start)
        o += s.stop - s.start
    gacc = torch.zeros(3, device=DEVICE)
    last_ck = time.time()
    print(f"[{tag}] {N:,} rows | {n_batches:,} batches/epoch | dims {dims}")

    while not st["finished"] and st["epoch"] < max_epochs:
        ep = st["epoch"]
        t0 = time.time()
        model.train()
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(v3.SEED * 1000 + ep)).numpy()
        b = st["batch"]
        while b < n_batches:
            hi_b = min(b + SUPER_BATCHES, n_batches)
            rows = perm[b * BATCH_SIZE:hi_b * BATCH_SIZE]
            order = np.argsort(rows)                      # sorted read, then restore perm order
            block = np.empty((len(rows), n_cols), np.float16)
            block[order] = Xtr[rows[order]][:, col_idx]     # only the active pillars' columns
            gx = torch.from_numpy(block).to(DEVICE, non_blocking=True)
            gy = ytr_t[torch.from_numpy(rows)].to(DEVICE)
            for j in range(hi_b - b):
                sl = slice(j * BATCH_SIZE, (j + 1) * BATCH_SIZE)
                xb = gx[sl].float()
                fb = _to_feats(xb, local_cols)
                yb = gy[sl]
                mask = (v3.sample_pillar_mask(len(yb), n_act, v3.PILLAR_DROPOUT, DEVICE)
                        if n_act > 1 else None)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=v3.USE_AMP):
                    logits, gates = model(fb, mask)
                loss, bce, ent = loss_fn(logits, yb, gates)
                if not torch.isfinite(loss):
                    st["skipped"] += 1
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), v3.MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                gacc += torch.stack([loss.detach(), bce.detach(), ent.detach()]).float() * len(yb)
                st["acc"][3] += len(yb)
            for i, v in enumerate(gacc.tolist()):          # one host sync per super-batch
                st["acc"][i] += v
            gacc.zero_()
            b = hi_b
            st["batch"] = b
            if time.time() - last_ck > ckpt_minutes * 60 or (deadline and time.time() > deadline):
                st["epoch_t"] += time.time() - t0
                t0 = time.time()
                _save_state(state_dict(), run_dir)
                last_ck = time.time()
                print(f"[{tag}] epoch {ep + 1} batch {b}/{n_batches} | loss {st['acc'][0] / max(st['acc'][3], 1):.4f}"
                      f" | checkpoint {datetime.now():%H:%M:%S}", flush=True)
                if deadline and time.time() > deadline:
                    print(f"[{tag}] time budget reached; paused cleanly")
                    return "paused"
        if st["skipped"] > 0.1 * n_batches:
            raise RuntimeError(f"[{tag}] {st['skipped']}/{n_batches} batches had non-finite loss")

        lv, gv = predict_rows(model, Xva, cols, clean_val)
        yv = yva[clean_val]
        val_loss = float(F.binary_cross_entropy_with_logits(lv, torch.from_numpy(yv.astype(np.float32))))
        mc = v3.metrics_from(yv, lv)
        a = st["acc"]
        rec = {"epoch": ep + 1, "train_loss": a[0] / max(a[3], 1), "train_bce": a[1] / max(a[3], 1),
               "gate_entropy": a[2] / max(a[3], 1), "val_loss_clean": val_loss,
               **{k: mc[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")},
               "lr": opt.param_groups[0]["lr"], "skipped_batches": st["skipped"],
               "epoch_seconds": st["epoch_t"] + time.time() - t0}
        for k, g in zip(active, gv.mean(0).tolist()):
            rec[f"gate_{k}"] = g
        st["history"].append(rec)
        if val_loss < st["best_loss"] - v3.MIN_DELTA:
            st["best_loss"], st["best_epoch"], st["patience"] = val_loss, ep + 1, 0
            v3.atomic_torch_save(bundle(ep + 1, val_loss), run_dir / "best_model.pth")
            star = " *best*"
        else:
            st["patience"] += 1
            star = ""
        sched.step()
        st["finished"] = st["patience"] >= v3.EARLY_STOPPING_PATIENCE or ep + 1 >= max_epochs
        st.update(epoch=ep + 1, batch=0, acc=[0.0, 0.0, 0.0, 0], skipped=0, epoch_t=0.0)
        _save_state(state_dict(), run_dir)
        last_ck = time.time()
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(st["history"][0]))
            w.writeheader()
            w.writerows(st["history"])
        print(f"[{tag}] epoch {ep + 1}/{max_epochs} | val loss {val_loss:.5f} | acc {mc['accuracy']:.4f} | "
              f"AUC {mc['roc_auc']:.4f} | {rec['epoch_seconds']:.0f}s | patience {st['patience']}{star}",
              flush=True)

    finalize(store, active, run_dir)
    return "done"


def _report(y, logits, groups=None):
    out = {"overall": _metrics(y, logits)}
    if groups is not None:
        out["per_group"] = {g: _metrics(y[groups == g], logits[groups == g]) for g in sorted(set(groups))}
    return out


def _metrics(y, logits):
    m = v3.metrics_from(y, logits)
    pred = (torch.sigmoid(torch.as_tensor(logits)).numpy() >= v3.CLASSIFICATION_THRESHOLD).astype(int)
    m["real_acc"] = float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan")
    m["fake_acc"] = float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan")
    m["n"] = int(len(y))
    return m


def finalize(store: Store, active, run_dir: Path):
    """Temperature on clean val (T >= 1), then one scoring pass over the test split."""
    cols = store.cols(active)
    best = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=False)
    model = v3.FivePillarDetector(best["dims"], active, best["model_cfg"]).to(DEVICE)
    model.load_state_dict(best["model_state_dict"])

    Xva, yva, cpva = store.split("val")
    clean = np.where(cpva == 0)[0]
    lv, gv = predict_rows(model, Xva, cols, clean)
    T, t_unc = v3.fit_temperature(lv, yva[clean])
    best.update(temperature=T, temperature_unconstrained=t_unc, finalized=True)
    v3.atomic_torch_save(best, run_dir / "best_model.pth")
    vidx = store.index("val")
    vcat = np.array([vidx[i]["relpath"].split("/")[0] for i in clean])
    val_rep = _report(yva[clean], lv.numpy(), vcat)
    val_rep.update(best_epoch=best["epoch"], temperature=T, temperature_unconstrained=t_unc,
                   mean_fusion_weights=dict(zip(active, gv.mean(0).tolist())),
                   note="in-distribution ProGAN validation; used for selection; not zero-shot")
    v3.atomic_write_text(json.dumps(val_rep, indent=1), run_dir / "val_report.json")

    Xte, yte, _ = store.split("test")
    tidx = store.index("test")
    gen = np.array([r["relpath"].split("/")[0] for r in tidx])
    lt, gt = predict_rows(model, Xte, cols)
    lt = lt.numpy()
    np.savez_compressed(run_dir / "test_predictions.npz", logits=lt, gates=gt.numpy(), y=yte,
                        row=np.arange(len(yte)))
    rep = _report(yte, lt, gen)
    per = rep["per_group"]
    rep["mean_generator_acc"] = float(np.mean([m["accuracy"] for m in per.values()]))
    rep["worst_generator_acc"] = float(min(m["accuracy"] for m in per.values()))
    rep["mean_generator_auc"] = float(np.mean([m["roc_auc"] for m in per.values()]))
    overlap = _overlap_rows(store, tidx)
    keep = np.setdiff1d(np.arange(len(yte)), overlap)
    rep["sensitivity_without_train_val_overlap"] = {"n_removed": int(len(overlap)),
                                                    **_metrics(yte[keep], lt[keep])}
    rep.update(threshold=v3.CLASSIFICATION_THRESHOLD, temperature_used_for_threshold=False,
               note="zero-shot: test generators never used for training, selection or threshold")
    v3.atomic_write_text(json.dumps(rep, indent=1), run_dir / "test_report.json")
    v3.atomic_write_text(json.dumps({"active": active, "best_epoch": best["epoch"],
                                     "test_acc": rep["overall"]["accuracy"],
                                     "test_auc": rep["overall"]["roc_auc"],
                                     "finished": datetime.now().isoformat()}, indent=1),
                         run_dir / "DONE.json")
    print(f"[{run_tag(active)}] TEST acc {rep['overall']['accuracy']:.4f} | AUC {rep['overall']['roc_auc']:.4f} "
          f"| real {rep['overall']['real_acc']:.3f} fake {rep['overall']['fake_acc']:.3f}")


def _overlap_rows(store: Store, tidx):
    from src.paths import section
    p = section("datasets")["manifests"] / "leakage_pairs.csv"
    if not p.exists():
        p = store.root / "leakage_pairs.csv"            # Kaggle: shipped with the store
    with open(p, newline="") as f:
        rel = {r["relpath"] for r in csv.DictReader(f) if r["split"] == "test"}
    return np.array([i for i, r in enumerate(tidx) if r["relpath"] in rel], dtype=int)
