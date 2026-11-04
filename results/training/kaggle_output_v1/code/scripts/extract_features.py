"""Extract five-pillar features into resumable shards (see src/features/extraction.py).

    python scripts/extract_features.py --split train|val|test [--workers 8] [--limit N] [--out DIR]

Safe to stop at any time (Ctrl+C, crash, power loss); rerun the same command to resume.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.extraction import main  # noqa: E402

if __name__ == "__main__":
    main()
