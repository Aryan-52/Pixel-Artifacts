"""Train the source-attribution head (which generator made an AI image).

    python scripts/train_attribution.py                                  # fp-v4.1 -> attribution-v1
    PA_MODEL_VERSION=fp-v4.2 python scripts/train_attribution.py         # fp-v4.2 -> attribution-v2

Data (never the zero-shot CNN_synth test set):
  - Tiny-GenImage (DataSet/Tiny-GenImage, Hugging Face TheKernel01/Tiny-GenImage @ 89c4fe9):
    its train split -> 90% train / 10% early-stopping validation (seed 7); its validation split ->
    held-out attribution test.
  - ProGAN fake images: 2,000 train + 250 validation from progan_train, 500 test from progan_val.

Every image goes through the app's own fp-v4.1 pipeline (Engine.prepare -> Engine._features), so
the head sees exactly the features it will see in the app. The detector's own score is stored too,
which gives a zero-shot detector result on Tiny-GenImage (the detector is never trained on it).

Training: AdamW, batch 64, max 50 epochs, early stopping on validation loss (patience 6). The
"unknown" cut-off is the 5th percentile of the top probability of correctly attributed validation
images. Outputs: models/attribution-v1/{attribution.pth, report.json, CHECKSUMS.json}.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.engine import Engine, ImageError, decode  # noqa: E402
from src.features.extraction import split_root  # noqa: E402
from src.models import attribution as attr  # noqa: E402
from src.paths import section  # noqa: E402

SEED = 7
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 64, 50, 6
GENIMAGE = section("datasets")["tiny_genimage"]
SETTINGS = Settings()                     # PA_MODEL_VERSION picks the detector (default fp-v4.1)
CACHE = REPO / "results" / "attribution" / ("cache" if SETTINGS.model_version == "fp-v4.1"
                                             else f"cache_{SETTINGS.model_version}")
OUT = SETTINGS.attribution_dir
PROGAN_N = {"train": 2000, "val": 250, "test": 500}


def featurize(eng, imgs):
    """PIL images -> (standardised feature rows fp16 [n,D], detector p_ai [n])."""
    x = torch.stack([eng.to_tensor(eng.prepare(im)[1]) for im in imgs])
    with torch.inference_mode():
        feats = eng._features(x)
        logits, _ = eng.v3.predict(eng.model, feats)
    p = torch.sigmoid(logits.float() / eng.temperature)
    return attr.concat(feats).cpu().numpy().astype(np.float16), p.cpu().numpy()


def generator_names(pf):
    hf = json.loads(pf.schema_arrow.metadata[b"huggingface"])
    return hf["info"]["features"]["generator"]["names"]


def cache_parquet(eng, path, split_of):
    """One parquet shard -> cache npz (resumable at shard level)."""
    out = CACHE / f"{path.stem}.npz"
    if out.exists():
        return out
    pf = pq.ParquetFile(path)
    names = generator_names(pf)
    X, P, G, FAKE, SPLIT, bad = [], [], [], [], [], 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["image", "label", "generator"]):
        imgs, rows = [], []
        for im, lab, gen in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist(),
                                batch.column("generator").to_pylist()):
            try:
                imgs.append(decode(im["bytes"], 10**9))
            except ImageError:
                bad += 1
                continue
            rows.append((names[gen], int(lab)))
        if not imgs:
            continue
        xf, p = featurize(eng, imgs)
        X.append(xf), P.append(p)
        for g, lab in rows:
            G.append(g), FAKE.append(lab), SPLIT.append(split_of())
    np.savez(out, X=np.concatenate(X), p_ai=np.concatenate(P), gen=np.array(G), fake=np.array(FAKE, np.int8),
             split=np.array(SPLIT), bad=bad)
    return out


def cache_progan(eng, rng):
    out = CACHE / "progan.npz"
    if out.exists():
        return out
    pick = lambda root, n: rng.sample(sorted(p for p in root.glob("*/1_fake/*") if p.is_file()), n)
    tr = pick(split_root("train"), PROGAN_N["train"] + PROGAN_N["val"])
    files = [(p, "train") for p in tr[:PROGAN_N["train"]]] + [(p, "val") for p in tr[PROGAN_N["train"]:]]
    files += [(p, "test") for p in pick(split_root("val"), PROGAN_N["test"])]
    X, P = [], []
    for i in range(0, len(files), BATCH_SIZE):
        xf, p = featurize(eng, [decode(f.read_bytes(), 10**9) for f, _ in files[i:i + BATCH_SIZE]])
        X.append(xf), P.append(p)
    np.savez(out, X=np.concatenate(X), p_ai=np.concatenate(P), gen=np.array(["ProGAN"] * len(files)),
             fake=np.ones(len(files), np.int8), split=np.array([s for _, s in files]), bad=0)
    return out


def load_all(paths):
    parts = [np.load(p) for p in paths]
    return {k: np.concatenate([d[k] for d in parts]) for k in ("X", "p_ai", "gen", "fake", "split")}, \
        sum(int(d["bad"]) for d in parts)


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def train(d, classes, dev):
    idx = {c: i for i, c in enumerate(classes)}
    fake = d["fake"] == 1
    sel = lambda s: fake & (d["split"] == s)
    t = lambda m: (torch.from_numpy(d["X"][m].astype(np.float32)).to(dev),
                   torch.tensor([idx[g] for g in d["gen"][m]], device=dev))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = t(sel("train")), t(sel("val")), t(sel("test"))
    torch.manual_seed(SEED)
    model = attr.AttributionHead(Xtr.shape[1], len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, wait, hist = 1e9, None, 0, []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)
        tot = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            loss = F.cross_entropy(model(Xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            lv = model(Xva)
            vl = F.cross_entropy(lv, yva).item()
            va = (lv.argmax(1) == yva).float().mean().item()
        hist.append({"epoch": ep, "train_loss": tot / len(Xtr), "val_loss": vl, "val_acc": va})
        print(f"epoch {ep:2d} train {tot / len(Xtr):.4f} val {vl:.4f} acc {va:.4f}")
        if vl < best - 1e-4:
            best, best_state, wait = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv, pt = model(Xva).softmax(1), model(Xte).softmax(1)
    correct = pv.argmax(1) == yva
    tau = float(torch.quantile(pv.max(1).values[correct], 0.05))
    return model, tau, hist, (yva, pv), (yte, pt)


def attribution_report(classes, y, prob, tau):
    y, prob = y.cpu().numpy(), prob.cpu().numpy()
    pred = prob.argmax(1)
    top2 = (np.argsort(-prob, 1)[:, :2] == y[:, None]).any(1)
    cm = np.zeros((len(classes), len(classes)), int)
    np.add.at(cm, (y, pred), 1)
    recall = cm.diagonal() / cm.sum(1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + recall > 0, 2 * prec * recall / np.maximum(prec + recall, 1e-12), 0)
    known = prob.max(1) >= tau
    return {"images": int(len(y)), "accuracy": float((pred == y).mean()), "top2_accuracy": float(top2.mean()),
            "macro_f1": float(f1.mean()), "per_class": {c: {"images": int(cm[i].sum()), "recall": float(recall[i]),
                                                              "precision": float(prec[i])} for i, c in enumerate(classes)},
            "confusion": cm.tolist(), "unknown_below": tau, "answered_share": float(known.mean()),
            "accuracy_when_answered": float((pred[known] == y[known]).mean())}


def detector_report(d, thr):
    gm = d["gen"] != "ProGAN"
    y, p = d["fake"][gm].astype(int), d["p_ai"][gm]
    gens = sorted(set(d["gen"][gm & (d["fake"] == 1)]))
    return {"images": int(gm.sum()), "accuracy": float(((p >= thr) == y).mean()), "auc": auc(y, p),
            "real_acc": float((p[y == 0] < thr).mean()), "fake_acc": float((p[y == 1] >= thr).mean()),
            "ai_caught": {g: float((d["p_ai"][gm & (d["gen"] == g)] >= thr).mean()) for g in gens}}


def main():
    t0 = time.time()
    eng = Engine(SETTINGS).load()
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    paths = []
    for f in sorted((GENIMAGE / "data").glob("*.parquet")):
        split_rng = random.Random(f"{SEED}-{f.stem}")      # per shard, so caching order cannot change splits
        if f.name.startswith("validation"):
            split_of = lambda: "test"
        else:
            split_of = lambda r=split_rng: "val" if r.random() < 0.1 else "train"
        print("features:", f.name, flush=True)
        paths.append(cache_parquet(eng, f, split_of))
    paths.append(cache_progan(eng, rng))
    d, bad = load_all(paths)
    classes = sorted(set(d["gen"][d["fake"] == 1]) - {"ProGAN"}) + ["ProGAN"]
    print("classes:", classes, "undecodable:", bad)

    model, tau, hist, (yva, pv), (yte, pt) = train(d, classes, eng.device)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {s: {c: int(((d["split"] == s) & (d["gen"] == c) & (d["fake"] == 1)).sum()) for c in classes}
              for s in ("train", "val", "test")}
    meta = {"detector": eng.version, "feature_code_hash": eng.bundle["feature_code_hash"],
            "dataset": "TheKernel01/Tiny-GenImage@89c4fe9 + ProGAN (progan_train / progan_val)", "seed": SEED,
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "epochs_run": len(hist), "best_epoch": int(np.argmin([h["val_loss"] for h in hist]) + 1)}
    attr.save(OUT / "attribution.pth", model.cpu(), classes, tau, meta)
    report = {**meta, "classes": classes, "images_per_split": counts, "undecodable": bad,
              "validation": attribution_report(classes, yva, pv, tau),
              "test": attribution_report(classes, yte, pt, tau),
              "detector_zero_shot_genimage": detector_report(d, eng.threshold), "history": hist,
              "minutes": (time.time() - t0) / 60}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    sha = hashlib.sha256((OUT / "attribution.pth").read_bytes()).hexdigest()
    (OUT / "CHECKSUMS.json").write_text(json.dumps({"files": {"attribution.pth": {"sha256": sha}}}, indent=1))
    t = report["test"]
    print(f"test acc {t['accuracy']:.4f} top2 {t['top2_accuracy']:.4f} macroF1 {t['macro_f1']:.4f} tau {tau:.3f}")
    print("detector on Tiny-GenImage:", json.dumps(report["detector_zero_shot_genimage"]))


if __name__ == "__main__":
    main()
