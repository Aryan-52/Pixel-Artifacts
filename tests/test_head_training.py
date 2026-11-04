"""Head trainer on a small synthetic store: trains, pauses on a deadline, resumes, finalises."""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import five_pillar_v3 as v3  # noqa: E402
from src.training import head  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DIMS = {"spatial": 1952, "frequency": 408, "color": 133, "physical": 89, "semantic": 512}


def make_store(root: Path, n_train=6400, n_val=640, n_test=640):
    rng = np.random.default_rng(0)
    offsets, o = {}, 0
    for k in v3.ALL_PILLARS:
        offsets[k] = [o, o + DIMS[k]]
        o += DIMS[k]
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        y = rng.integers(0, 2, n).astype(np.int8)
        X = rng.standard_normal((n, o)).astype(np.float16)
        X[:, :8] += y[:, None] * 1.5                       # learnable signal in "spatial"
        np.save(root / f"{split}_X.npy", X)
        np.save(root / f"{split}_y.npy", y)
        np.save(root / f"{split}_copy.npy", np.zeros(n, np.int8))
        with open(root / f"{split}_index.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "entry", "relpath", "label", "copy"])
            gens = ["gan_a", "gan_b"] if split == "test" else ["cat"]
            w.writerows((i, i, f"{gens[i % len(gens)]}/{'01'[y[i]]}_x/{i}.png", y[i], 0) for i in range(n))
    stats = {k: {"mean": torch.zeros(DIMS[k]), "std": torch.ones(DIMS[k])} for k in v3.ALL_PILLARS}
    torch.save({"train_sig": "synthetic", "stats": stats}, root / "stats.pt")
    (root / "store_meta.json").write_text(json.dumps({
        "pillars": v3.ALL_PILLARS, "dims": DIMS, "offsets": offsets, "feature_clip": 10.0,
        "feature_code_hash": "synthetic", "train_sig": "synthetic"}))
    with open(root / "leakage_pairs.csv", "w", newline="") as f:
        f.write("sha256,split,relpath,label,generator\n")


def test_train_pause_resume_finalize(tmp_path):
    store_dir, runs = tmp_path / "store", tmp_path / "runs"
    store_dir.mkdir()
    make_store(store_dir)
    store = head.Store(store_dir)
    old = head.SUPER_BATCHES
    head.SUPER_BATCHES = 20
    try:
        # deadline already passed -> pauses after the first super-batch with a checkpoint
        assert head.train_run(store, ["spatial", "color"], runs, deadline=time.time() - 1, max_epochs=3) == "paused"
        ck = torch.load(runs / "spatial+color" / "latest.pth", weights_only=False)
        assert ck["st"]["epoch"] == 0 and ck["st"]["batch"] == 20
        assert head.train_run(store, ["spatial", "color"], runs, max_epochs=3) == "done"
    finally:
        head.SUPER_BATCHES = old
    run = runs / "spatial+color"
    hist = list(csv.DictReader(open(run / "history.csv")))
    assert len(hist) == 3
    rep = json.loads((run / "test_report.json").read_text())
    assert rep["overall"]["accuracy"] > 0.9                 # signal was learnable
    assert set(rep["per_group"]) == {"gan_a", "gan_b"}
    b = torch.load(run / "best_model.pth", weights_only=False)
    assert b["finalized"] and b["temperature"] >= 1.0 and b["active"] == ["spatial", "color"]
    model = v3.FivePillarDetector(b["dims"], b["active"], b["model_cfg"])
    model.load_state_dict(b["model_state_dict"])            # loadable like an fp-v3 bundle
    assert head.train_run(store, ["spatial", "color"], runs, max_epochs=3) == "done"   # skip


def test_single_pillar_and_full(tmp_path):
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    make_store(store_dir, n_train=1280)
    store = head.Store(store_dir)
    for active in (["physical"], v3.ALL_PILLARS):
        assert head.train_run(store, active, tmp_path / "runs", max_epochs=1) == "done"
    assert (tmp_path / "runs" / "full" / "DONE.json").exists()
    assert (tmp_path / "runs" / "physical" / "DONE.json").exists()
