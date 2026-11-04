"""Build the fp-v4.1 feature store (spatial pillar from the fine-tuned EfficientNet). Resumable.

    python scripts/build_ft_store.py [--ckpt results/training/spatial_ft/best.pth]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.spatial_ft import main  # noqa: E402

if __name__ == "__main__":
    main()
