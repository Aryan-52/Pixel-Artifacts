"""Evaluate the app's full-image scan (DECISION_LOG D-21) once on the zero-shot test set.

    python scripts/eval_fullscan.py

Rule (fixed before this script was run): every image is covered by the Engine.window_boxes grid of
224 px windows at full resolution; each window is scored by the unchanged detector; image
p(AI) = mean of window probabilities; threshold 0.5. Nothing is tuned here: the result is only
reported. Images are loaded by a thread pool (D-17) and scored with Engine.score, the same path
Engine.scan uses.

Outputs in models/<version>/ (PA_MODEL_VERSION, default fp-v4.1): test_predictions_fullscan.npz,
test_report_fullscan.json.
"""
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from backend.app.config import Settings  # noqa: E402
from backend.app.engine import Engine, decode, window_boxes  # noqa: E402
from src.features.extraction import split_root  # noqa: E402
from src.paths import section  # noqa: E402

WINDOW_BATCH = 128                # windows per GPU call
THREADS = 8
IMAGE_CHUNK = 32               # at most this many images decoded at once ...
CHUNK_WINDOWS = 2048           # ... and at most this many windows (~300 MB of uint8 crops): 32 large photos
                               # of up to 486 windows each ran out of the 16 GB RAM


def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp = int(((yhat == 0) & (y == 0)).sum()), int(((yhat == 1) & (y == 0)).sum())
    fn, tp = int(((yhat == 0) & (y == 1)).sum()), int(((yhat == 1) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    both = 0 < y.sum() < len(y)
    return {"accuracy": (tp + tn) / len(y), "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "confusion_matrix": [[tn, fp], [fn, tp]], "real_acc": tn / (tn + fp) if tn + fp else None,
            "fake_acc": rec if tp + fn else None, "n": int(len(y))}


def main():
    t0 = time.time()
    eng = Engine(Settings()).load()
    size = eng.v3.IMAGE_SIZE
    root = split_root("test")
    rows = list(csv.DictReader(open(section("datasets")["manifests"] / "test_manifest.csv", newline="")))

    def load(r):
        full = eng.prepare(decode((root / r["relpath"]).read_bytes(), 10**9))[0]
        boxes = window_boxes(*full.size, size)
        crops = ([eng.v3.center_crop_or_upscale(full, size)] if boxes == [None] else
                 [full.crop((x, y, x + size, y + size)) for x, y in boxes])
        return torch.stack([eng.to_tensor(c) for c in crops])

    n = len(rows)
    ckpt = REPO / "results" / "testing" / f"eval_fullscan_ckpt_{eng.version}.npz"
    if ckpt.exists():                                   # resume after an interruption
        c = np.load(ckpt)
        sums, maxes, counts, start = c["sums"], c["maxes"], c["counts"], int(c["next"])
        print(f"resuming at image {start}", flush=True)
    else:
        sums, maxes, counts, start = np.zeros(n), np.zeros(n), np.zeros(n, int), 0
    buf, owner = [], []

    def flush():
        x, o = torch.cat(buf), np.concatenate(owner)
        for s in range(0, len(x), WINDOW_BATCH):         # fixed GPU batch, however large one image is
            p, _ = eng.score(x[s:s + WINDOW_BATCH])
            oo = o[s:s + WINDOW_BATCH]
            np.add.at(sums, oo, p)
            np.maximum.at(maxes, oo, p)
            np.add.at(counts, oo, 1)
        buf.clear(), owner.clear()

    def n_windows(r):                                   # from the manifest, before decoding
        b = window_boxes(int(r["width"]), int(r["height"]), size)
        return 1 if b == [None] else len(b)

    chunks, a = [], start                               # bounded by image count and by window count
    while a < n:
        b, w = a, 0
        while b < n and b - a < IMAGE_CHUNK and (b == a or w + n_windows(rows[b]) <= CHUNK_WINDOWS):
            w += n_windows(rows[b])
            b += 1
        chunks.append((a, b))
        a = b

    last_save = start
    with ThreadPoolExecutor(THREADS) as pool:           # bounded: Executor.map would queue every image
        for a, b in chunks:
            for i, x in enumerate(pool.map(load, rows[a:b]), start=a):
                buf.append(x)
                owner.append(np.full(len(x), i))
            flush()                                     # every image of the chunk is complete
            if b - last_save >= 5120 or b == n:
                np.savez(ckpt, sums=sums, maxes=maxes, counts=counts, next=b)
                last_save = b
                print(f"{b}/{n} images, {time.time() - t0:.0f}s", flush=True)

    p = sums / counts
    y = np.array([int(r["label"]) for r in rows])
    gen = np.array([r["generator"] for r in rows])
    out = eng.s.model_dir
    np.savez(out / "test_predictions_fullscan.npz", p=p, y=y.astype(np.int8), generator=gen,
             relpath=np.array([r["relpath"] for r in rows]), windows=counts, p_max=maxes)
    thr = eng.threshold
    per = {g: metrics(y[gen == g], p[gen == g], thr) for g in sorted(set(gen))}
    centre = json.loads((out / "test_report.json").read_text())
    report = {"protocol": "full-image scan (DECISION_LOG D-21): mean AI probability of 224 px windows covering every pixel",
              "threshold": thr, "windows_total": int(counts.sum()), "overall": metrics(y, p, thr), "per_group": per,
              "mean_generator_acc": float(np.mean([v["accuracy"] for v in per.values()])),
              "worst_generator_acc": float(min(v["accuracy"] for v in per.values())),
              "centre_crop_overall": centre["overall"], "minutes": (time.time() - t0) / 60}
    (out / "test_report_fullscan.json").write_text(json.dumps(report, indent=1))
    ckpt.unlink(missing_ok=True)
    o, c = report["overall"], centre["overall"]
    print(f"full scan : acc {o['accuracy']:.4f} auc {o['roc_auc']:.4f} real {o['real_acc']:.4f} fake {o['fake_acc']:.4f}")
    print(f"centre    : acc {c['accuracy']:.4f} auc {c['roc_auc']:.4f} real {c['real_acc']:.4f} fake {c['fake_acc']:.4f}")
    for g, v in per.items():
        print(f"  {g:16s} acc {v['accuracy']:.4f} (centre {centre['per_group'][g]['accuracy']:.4f})")


if __name__ == "__main__":
    main()
