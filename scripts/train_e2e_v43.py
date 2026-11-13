"""Train fp-v4.3 (end-to-end spatial pillar, decision D-23). Resumable: rerun the same command.

    python scripts/train_e2e_v43.py [--hours 6] [--threads 12]
    python scripts/train_e2e_v43.py --smoke      # 2 tiny epochs in results/training/e2e_v43_smoke, no test
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import e2e_v43  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=5.0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run_dir = e2e_v43.RUN_DIR.with_name("e2e_v43_smoke") if a.smoke else e2e_v43.RUN_DIR
    deadline = time.time() + a.hours * 3600 if a.hours else None
    if e2e_v43.train(run_dir, a.threads, a.ckpt_minutes, deadline, a.smoke) == "paused":
        print("Paused. Rerun the same command to continue.")


if __name__ == "__main__":
    main()
