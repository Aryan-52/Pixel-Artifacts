"""Fine-tune the spatial EfficientNet-B0 (fp-v4.1). Resumable: rerun the same command.

    python scripts/finetune_spatial.py [--hours 4] [--limit 2048]   # --limit = smoke test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.spatial_finetune import main  # noqa: E402

if __name__ == "__main__":
    main()
