"""Source attribution in the app must match training: the app's live probabilities for raw held-out
Tiny-GenImage images equal the head applied to the cached training-time features.

Skipped when the attribution model, its feature cache or the dataset are absent.
"""
import numpy as np
import pytest

from backend.app.config import REPO, Settings

ATTR = REPO / "models" / "attribution-v1" / "attribution.pth"
CACHE = REPO / "results" / "attribution" / "cache" / "validation-00000-of-00004.npz"


def _shard():
    from src.paths import section
    return section("datasets")["tiny_genimage"] / "data" / "validation-00000-of-00004.parquet"


@pytest.mark.skipif(not ATTR.exists() or not CACHE.exists(), reason="attribution model or cache missing")
def test_app_attribution_matches_training_features():
    import pyarrow.parquet as pq
    import torch

    from backend.app.engine import Engine, decode

    shard = _shard()
    if not shard.exists():
        pytest.skip("Tiny-GenImage not downloaded")
    eng = Engine(Settings()).load()
    assert eng.attr is not None
    cached = np.load(CACHE)
    rows = pq.ParquetFile(shard).read_row_group(0, columns=["image"]).column("image").to_pylist()
    picks = [i for i in range(len(rows)) if cached["fake"][i] == 1][:8]
    with torch.no_grad():
        want = eng.attr(torch.from_numpy(cached["X"][picks].astype(np.float32)).to(eng.device)).softmax(1).cpu().numpy()
    classes = eng.attr_info["classes"]
    for i, w in zip(picks, want):
        _, crop, _ = eng.prepare(decode(rows[i]["bytes"], 10**9))
        prob = eng.explain_batch(eng.to_tensor(crop).unsqueeze(0))["source"][0]
        got = dict(zip(classes, map(float, prob)))
        assert abs(sum(got.values()) - 1) < 1e-4
        assert max(abs(got[c] - w[k]) for k, c in enumerate(classes)) < 2e-3
