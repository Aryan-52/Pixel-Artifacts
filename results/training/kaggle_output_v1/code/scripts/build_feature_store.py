"""Build the standardised fp16 feature store from extracted shards (see src/features/store.py).

    python scripts/build_feature_store.py [--splits train val test]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.store import main  # noqa: E402

if __name__ == "__main__":
    main()
