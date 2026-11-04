"""Train the fp-v4.0 head and its pillar-subset ablations, resumably, in priority order.

    python scripts/train_heads.py --group p1            # full 5-pillar model
    python scripts/train_heads.py --group p2            # 5 single pillars + 5 leave-one-out
    python scripts/train_heads.py --group p3 --part 1/2 # pairs + triples, half of them
    python scripts/train_heads.py --group all --hours 11 # stop cleanly before a time limit
    python scripts/train_heads.py --group p3 --part 1/2 --slot 2/3   # 3 parallel processes

Every run is skipped if its DONE.json exists and resumed from its last checkpoint otherwise,
so the same command can simply be rerun after any interruption. --part k/n splits a group
between machines (local laptop / Kaggle) without overlap; --slot s/S splits one machine's
share between S parallel processes (head training is overhead-bound, so they overlap well).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.five_pillar_v3 import ALL_PILLARS  # noqa: E402
from src.paths import section  # noqa: E402
from src.training.head import Store, train_run  # noqa: E402


def subsets(r):
    return [list(c) for c in itertools.combinations(ALL_PILLARS, r)]


GROUPS = {
    "p1": [ALL_PILLARS],
    "p2": subsets(1) + subsets(4),
    "p3": subsets(2) + subsets(3),
}
GROUPS["all"] = GROUPS["p1"] + GROUPS["p2"] + GROUPS["p3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), default="all")
    ap.add_argument("--part", default="1/1", help="k/n: take every n-th run starting at k")
    ap.add_argument("--slot", default="1/1", help="s/S: this process's share of the machine's runs")
    ap.add_argument("--reverse", action="store_true", help="take this share's runs last-first "
                    "(a fallback machine then meets the primary one in the middle)")
    ap.add_argument("--hours", type=float, default=None, help="pause cleanly after this long")
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--runs", type=Path, default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=5.0)
    a = ap.parse_args()
    k, n = map(int, a.part.split("/"))
    s, S = map(int, a.slot.split("/"))
    runs = GROUPS[a.group][k - 1::n][s - 1::S]
    if a.reverse:
        runs = runs[::-1]
    store = Store(a.store or section("outputs")["feature_store"])
    runs_root = a.runs or section("outputs")["results"] / "training" / "runs"
    deadline = time.time() + a.hours * 3600 if a.hours else None
    print(f"{len(runs)} runs: {['+'.join(r) for r in runs]}")
    for active in runs:
        if train_run(store, active, runs_root, a.ckpt_minutes, deadline) == "paused":
            print("Paused. Rerun the same command to continue.")
            return
    print("All requested runs done.")


if __name__ == "__main__":
    main()
