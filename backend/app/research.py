"""Numbers for the Research pages, read from the result files (nothing typed in by hand except the
earlier paper's published figures, which have no result file in this repository)."""
from __future__ import annotations

import csv
import json

from backend.app.config import REPO

MODELS = REPO / "models"
ABLATION = REPO / "results" / "ablation" / "ablation_results.csv"
GENERATORS = ["progan", "biggan", "crn", "cyclegan", "deepfake", "gaugan", "imle", "san", "seeingdark",
              "stargan", "stylegan", "stylegan2", "whichfaceisreal"]


def _json(path):
    return json.loads(path.read_text())


def _csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _v4(version: str, notes: str) -> dict:
    t = _json(MODELS / version / "test_report.json")
    v = _json(MODELS / version / "val_report.json")
    o = t["overall"]
    return {"id": version, "notes": notes, "test_images": o["n"], "generators": len(t["per_group"]),
            "accuracy": o["accuracy"], "auc": o["roc_auc"], "ap": o["average_precision"], "f1": o["f1"],
            "real_acc": o["real_acc"], "fake_acc": o["fake_acc"], "val_accuracy": v["overall"]["accuracy"],
            "per_generator": {g: {"accuracy": r["accuracy"], "auc": r["roc_auc"], "fake_acc": r["fake_acc"]}
                              for g, r in t["per_group"].items()}}


def versions() -> list[dict]:
    v3 = _json(MODELS / "fp-v3.0" / "zero_shot_summary.json")
    o3 = v3["overall_clean"]
    return [
        {"id": "Hybrid-v1", "notes": "Earlier submitted paper: 4-branch hybrid (EfficientNet-B0, 22 physics "
                                     "features, 9 noise features, ViT-S/16), late fusion.",
         "test_images": 90245, "generators": 13, "accuracy": 0.663, "auc": 0.891, "ap": 0.906,
         "source": "Research Paper.pdf (team's earlier submission)", "per_generator": {}},
        {"id": "fp-v3.0", "notes": "Five pillars + gated fusion, trained on a 4-category ProGAN subset. "
                                   "Tested on a balanced 8,480-image sample of 12 unseen generators.",
         "test_images": o3["n"], "generators": 12, "accuracy": o3["accuracy"], "auc": o3["roc_auc"],
         "ap": o3["average_precision"], "f1": o3["f1"], "real_acc": o3["specificity"], "fake_acc": o3["recall"],
         "per_generator": {r["generator"]: {"accuracy": r["accuracy"], "auc": r["roc_auc"], "fake_acc": r["recall"]}
                           for r in v3["by_generator_clean"]}},
        _v4("fp-v4.0", "Same five pillars, trained on the full ProGAN set (720k images, 20 categories) "
                       "with blur/JPEG copies. Frozen ImageNet EfficientNet."),
        _v4("fp-v4.1", "fp-v4.0 with the spatial EfficientNet-B0 fine-tuned as a real/fake detector "
                       "(batch 64, early stopping). Shipped in this app."),
    ]


def ablation() -> list[dict]:
    rows = []
    for r in _csv(ABLATION):
        rows.append({k: (_num(v) if k not in ("subset",) else v) for k, v in r.items()})
    return rows


def training() -> dict:
    head = [{k: _num(v) for k, v in r.items()} for r in _csv(MODELS / "fp-v4.1" / "head_history.csv")]
    spatial = [{k: _num(v) for k, v in r.items()} for r in _csv(MODELS / "fp-v4.1" / "spatial_history.csv")]
    return {
        "data": {"train_images": 720030, "train_views": 1440060, "val_images": 7569, "test_images": 90329,
                 "train_generator": "ProGAN (20 LSUN categories)", "test_generators": 13},
        "stages": [
            {"name": "Stage 1: fine-tune spatial EfficientNet-B0", "batch_size": 64, "max_epochs": 50,
             "epoch_size": "200,000 views", "optimizer": "AdamW, lr 1e-4, weight decay 1e-2",
             "early_stopping": "patience 3 on clean validation loss", "augmentation":
             "blur (p 0.5, sigma 0-3) + JPEG (p 0.5, quality 30-100) copy, horizontal flip",
             "epochs_run": len(spatial), "best_epoch": min(spatial, key=lambda r: r["val_loss"])["epoch"]},
            {"name": "Stage 2: five-pillar gated-fusion head", "batch_size": 64, "max_epochs": 50,
             "epoch_size": "1,440,060 views (all)", "optimizer": "AdamW, lr 5e-4, 2 warm-up epochs, cosine decay",
             "early_stopping": "patience 8 on clean validation loss",
             "augmentation": "pillar dropout 0.15, label smoothing 0.02",
             "epochs_run": len(head), "best_epoch": min(head, key=lambda r: r["val_loss_clean"])["epoch"]},
        ],
        "spatial_history": spatial, "head_history": head,
    }


def attribution() -> dict | None:
    path = MODELS / "attribution-v1" / "report.json"
    if not path.exists():
        return None
    r = _json(path)
    return {k: r[k] for k in ("dataset", "classes", "images_per_split", "batch_size", "max_epochs", "epochs_run",
                              "best_epoch", "test", "detector_zero_shot_genimage")}


def protocols() -> dict | None:
    """fp-v4.1 centre crop (training / paper protocol) vs the app's whole-image scan (D-21)."""
    path = MODELS / "fp-v4.1" / "test_report_fullscan.json"
    if not path.exists():
        return None
    f, c = _json(path), _json(MODELS / "fp-v4.1" / "test_report.json")
    pick = lambda o: {k: o[k] for k in ("accuracy", "roc_auc", "average_precision", "real_acc", "fake_acc", "n")}
    return {"windows_total": f["windows_total"], "centre": pick(c["overall"]), "fullscan": pick(f["overall"]),
            "per_generator": {g: {"centre": c["per_group"][g]["accuracy"], "fullscan": f["per_group"][g]["accuracy"]}
                              for g in f["per_group"]}}


def build() -> dict:
    return {"generators": GENERATORS, "versions": versions(), "ablation": ablation(), "training": training(),
            "attribution": attribution(), "protocols": protocols()}
