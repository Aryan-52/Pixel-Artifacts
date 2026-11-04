"""App engine parity: scoring raw test images must reproduce the logged fp-v4.1 test predictions.

Needs the GPU-free model files in models/fp-v4.1 and the test images; skipped when they are absent.
"""
import csv
import random
from pathlib import Path

import numpy as np
import pytest

from backend.app.config import REPO, Settings

MODEL = REPO / "models" / "fp-v4.1"
INDEX = REPO / "results" / "training" / "feature_store_ft" / "test_index.csv"


@pytest.mark.skipif(not (MODEL / "head.pth").exists() or not INDEX.exists(), reason="model or test index missing")
def test_engine_matches_logged_test_predictions():
    import torch

    from backend.app.engine import Engine, decode
    from src.features.extraction import split_root

    eng = Engine(Settings()).load()
    rows = list(csv.DictReader(open(INDEX, newline="")))
    pred = np.load(MODEL / "test_predictions.npz")
    logged = dict(zip(pred["row"].tolist(), pred["logits"].tolist()))
    random.seed(0)
    picks = [i for i in random.sample(range(len(rows)), 40) if int(rows[i]["copy"]) == 0][:16]
    root = split_root("test")
    xs = []
    for i in picks:
        _, crop, _ = eng.prepare(decode(Path(root / rows[i]["relpath"]).read_bytes(), 10**9))
        xs.append(eng.to_tensor(crop))
    p, _ = eng.score(torch.stack(xs))
    want = 1 / (1 + np.exp(-np.array([logged[i] for i in picks])))
    assert np.abs(p - want).max() < 2e-3, (p, want)


@pytest.mark.skipif(not (MODEL / "head.pth").exists(), reason="model missing")
def test_scan_covers_every_pixel_and_averages_windows():
    from PIL import Image

    from backend.app.engine import Engine, window_boxes

    for w, h in [(224, 224), (256, 256), (1000, 700), (6030, 4032)]:
        boxes = window_boxes(w, h)
        cov = np.zeros((h, w), bool)
        for x, y in boxes:
            assert 0 <= x <= w - 224 and 0 <= y <= h - 224
            cov[y:y + 224, x:x + 224] = True
        assert cov.all() and len(boxes) == -(-w // 224) * -(-h // 224)
    assert window_boxes(200, 500) == [None]

    eng = Engine(Settings()).load()
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (300, 500, 3), dtype=np.uint8))
    sc = eng.scan(img)
    xs = [eng.to_tensor(img.crop((w["x"], w["y"], w["x"] + 224, w["y"] + 224))) for w in sc["windows"]]
    import torch
    p, _ = eng.score(torch.stack(xs))
    assert len(sc["windows"]) == 6
    assert abs(sc["p_ai"] - float(p.mean())) < 1e-5
