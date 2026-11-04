"""How much to trust a score: the model's measured behaviour on the zero-shot benchmark.

Source: models/<version>/test_predictions_fullscan.npz (scripts/eval_fullscan.py), the fp-v4.1
whole-image-scan scores the app produces, for all 90,329 CNN_synth_testset images (13 generators never
seen in training). Falls back to the centre-crop test_predictions.npz if the scan was not evaluated. These are used here only to DESCRIBE the fixed
model; nothing in the model, threshold or temperature was chosen with them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BINS = 20


def roc_points(y: np.ndarray, p: np.ndarray, n: int = 120) -> dict:
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    tpr = np.concatenate([[0], tps / max(tps[-1], 1)])
    fpr = np.concatenate([[0], fps / max(fps[-1], 1)])
    auc = float(np.trapezoid(tpr, fpr))
    keep = np.unique(np.linspace(0, len(tpr) - 1, n).astype(int))
    return {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist(), "auc": auc}


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"n": n, "accuracy": (tp + tn) / n if n else 0.0, "precision": prec, "recall": rec,
           "specificity": tn / (tn + fp) if tn + fp else 0.0,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
    if 0 < y.sum() < n:
        out["roc"] = roc_points(y, p)
        out["auc"] = out["roc"]["auc"]
    return out


class Evidence:
    def __init__(self, model_dir: Path, threshold: float):
        if (model_dir / "test_predictions_fullscan.npz").exists():
            d = np.load(model_dir / "test_predictions_fullscan.npz")
            self.p = d["p"].astype(np.float64)
            self.report = json.loads((model_dir / "test_report_fullscan.json").read_text())
            self.protocol = "full-image scan"
        else:
            d = np.load(model_dir / "test_predictions.npz")
            self.p = 1 / (1 + np.exp(-d["logits"].astype(np.float64)))
            self.report = json.loads((model_dir / "test_report.json").read_text())
            self.protocol = "centre crop"
        self.y = d["y"].astype(int)
        self.thr = threshold
        self.val = json.loads((model_dir / "val_report.json").read_text())
        edges = np.linspace(0, 1, BINS + 1)
        idx = np.clip(np.digitize(self.p, edges) - 1, 0, BINS - 1)
        self.bin_n = np.bincount(idx, minlength=BINS)
        self.bin_ai = np.bincount(idx, weights=self.y, minlength=BINS)
        self.hist_real = np.bincount(idx[self.y == 0], minlength=BINS)
        self.hist_ai = np.bincount(idx[self.y == 1], minlength=BINS)
        self.sorted_real = np.sort(self.p[self.y == 0])
        self.sorted_ai = np.sort(self.p[self.y == 1])
        self.benchmark = self._benchmark()

    def _benchmark(self) -> dict:
        o = self.report["overall"]
        m = binary_metrics(self.y, self.p, self.thr)
        return {
            "dataset": "CNN_synth_testset (Wang et al., CVPR 2020)", "images": int(o["n"]),
            "generators": len(self.report["per_group"]), "zero_shot": True, "protocol": self.protocol,
            "accuracy": o["accuracy"], "auc": o["roc_auc"], "ap": o["average_precision"],
            "precision": o["precision"], "recall": o["recall"], "f1": o["f1"],
            "real_acc": o["real_acc"], "fake_acc": o["fake_acc"],
            "confusion": o["confusion_matrix"], "roc": m["roc"],
            "validation": {"accuracy": self.val["overall"]["accuracy"], "auc": self.val["overall"]["roc_auc"],
                           "images": self.val["overall"]["n"]},
            "per_generator": {g: {"accuracy": r["accuracy"], "auc": r["roc_auc"], "n": r["n"],
                                  "real_acc": r["real_acc"], "fake_acc": r["fake_acc"]}
                              for g, r in self.report["per_group"].items()},
            "histogram": {"edges": np.linspace(0, 1, BINS + 1).round(3).tolist(),
                          "real": self.hist_real.tolist(), "ai": self.hist_ai.tolist()},
            "operating_point": {"threshold": self.thr, "fpr": 1 - o["real_acc"], "tpr": o["fake_acc"]},
        }

    def for_score(self, p_ai: float) -> dict:
        """What the benchmark says about images that received a similar score."""
        b = int(min(max(p_ai, 0.0), 1.0) * BINS)
        b = min(b, BINS - 1)
        n = int(self.bin_n[b])
        frac_ai = float(self.bin_ai[b] / n) if n else None
        return {
            "bin": [b / BINS, (b + 1) / BINS], "bin_images": n,
            "bin_share_ai": frac_ai,
            "real_scored_lower": float(np.searchsorted(self.sorted_real, p_ai) / len(self.sorted_real)),
            "ai_scored_lower": float(np.searchsorted(self.sorted_ai, p_ai) / len(self.sorted_ai)),
        }
