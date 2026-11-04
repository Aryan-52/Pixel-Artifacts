"""Fine-tune the spatial EfficientNet-B0 with fresh degradations every epoch (fp-v4.2). Resumable.

    python scripts/finetune_spatial_v2.py [--hours 4] [--limit 2048]   # --limit = smoke test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.spatial_finetune_v2 import main  # noqa: E402

if __name__ == "__main__":
    main()
