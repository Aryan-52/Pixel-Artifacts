"""Copy a seeded random set of benchmark images into frontend/public/samples and score them.

    python scripts/make_app_samples.py

One real and one AI image per generator, chosen at random (seed 7), never hand-picked, so the
landing-page game shows the detector's honest hits and misses. Writes samples.json with the ground
truth, the generator and the fp-v4.1 score.
"""
import json
import random
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.app.config import Settings  # noqa: E402
from backend.app.engine import Engine, decode  # noqa: E402
from src.features.extraction import split_root  # noqa: E402

GENERATORS = ["progan", "stylegan", "stylegan2", "biggan", "cyclegan", "stargan", "gaugan", "whichfaceisreal",
              "crn", "imle", "deepfake", "seeingdark", "san"]
OUT = REPO / "frontend" / "public" / "samples"


def main():
    root = split_root("test")
    rng = random.Random(7)
    eng = Engine(Settings()).load()
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for g in GENERATORS:
        for label, sub in ((0, "0_real"), (1, "1_fake")):
            files = sorted(p for p in (root / g).rglob("*") if p.is_file() and sub in p.parts
                           and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            src = rng.choice(files)
            name = f"{g}_{'ai' if label else 'real'}{src.suffix.lower()}"
            shutil.copy2(src, OUT / name)
            img = decode(src.read_bytes(), 10**9)
            _, crop, _ = eng.prepare(img)
            p, _ = eng.score(eng.to_tensor(crop).unsqueeze(0))
            items.append({"file": f"samples/{name}", "generator": g, "label": "ai" if label else "real",
                          "p_ai": round(float(p[0]), 4), "width": img.size[0], "height": img.size[1]})
            print(f"{name:28s} p_ai={float(p[0]):.3f}")
    (OUT / "samples.json").write_text(json.dumps(items, indent=1))


if __name__ == "__main__":
    main()
