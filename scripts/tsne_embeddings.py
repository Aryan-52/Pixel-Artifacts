"""t-SNE of real vs. AI test images for chosen generators, in two feature spaces (fp-v4.1).

    .venv\\Scripts\\python.exe scripts\\tsne_embeddings.py [--gens progan stylegan imle] [--n 600]

Row 1: the frozen CLIP ViT-B/32 pillar alone (512-d, standardised).
Row 2: the fused 256-d embedding sum_k w_k z_k that the fp-v4.1 head classifies.
Visualisation only: features of the locked model on a seeded sample of test images; nothing
is trained, selected or tuned. Writes the 2-D coordinates to results/figures/tsne_<gens>.npz
(keys <gen>_clip, <gen>_fused, <gen>_y). The plot is drawn by BlackBook/scripts/make_figures.py,
so matplotlib is not needed in this environment.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sklearn.manifold import TSNE  # noqa: E402

from src.models import five_pillar_v3 as v3  # noqa: E402
from src.training.head import Store, _to_feats  # noqa: E402

STORE = REPO / "results" / "training" / "feature_store_ft"
HEAD = REPO / "models" / "fp-v4.1" / "head.pth"


@torch.no_grad()
def fused(model, block, cols):
    feats = _to_feats(block, cols)
    z = torch.stack([model.proj[k](feats[k]) for k in model.active], dim=1)
    f, _ = model.fusion(z)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", default=["progan", "stylegan", "imle"])
    ap.add_argument("--n", type=int, default=600, help="images per class per generator")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    store = Store(STORE)
    X, y, _ = store.split("test")
    gen = np.array([r["relpath"].split("/")[0] for r in store.index("test")])
    best = torch.load(HEAD, map_location="cpu", weights_only=False)
    model = v3.FivePillarDetector(best["dims"], best["active"] if "active" in best else v3.ALL_PILLARS,
                                  best["model_cfg"]).to(v3.DEVICE).eval()
    model.load_state_dict(best["model_state_dict"])
    cols = store.cols(model.active)

    rng = np.random.default_rng(a.seed)
    out = {}
    for g in a.gens:
        rows = np.concatenate([rng.choice(np.where((gen == g) & (y == c))[0], a.n, replace=False) for c in (0, 1)])
        rows.sort()
        block = torch.from_numpy(np.array(X[rows])).to(v3.DEVICE).float()
        clip = block[:, cols["semantic"]].cpu().numpy()
        fz = fused(model, block, cols).float().cpu().numpy()
        for space, Z in (("clip", clip), ("fused", fz)):
            out[f"{g}_{space}"] = TSNE(2, perplexity=30, init="pca", random_state=a.seed).fit_transform(Z)
        out[f"{g}_y"] = y[rows]
        print(g, "done", flush=True)

    dest = REPO / "results" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.gens)
    np.savez(dest / f"tsne_{tag}.npz", **out)
    print("wrote", dest / f"tsne_{tag}.npz")


if __name__ == "__main__":
    main()
