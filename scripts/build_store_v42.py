"""Build the fp-v4.2 feature store (augmented spatial pillar + CLIP ViT-L/14 semantic). Resumable.

    python scripts/build_store_v42.py --pillars semantic           # can run during the fine-tune
    python scripts/build_store_v42.py --pillars spatial --assemble
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.store_v42 import main  # noqa: E402

if __name__ == "__main__":
    main()
