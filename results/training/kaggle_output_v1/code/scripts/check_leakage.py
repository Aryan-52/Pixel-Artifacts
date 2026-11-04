"""Exact-duplicate and cross-split leakage report from the SHA-256 manifests.

    python scripts/check_leakage.py

Reads <manifests>/{train,val,test}_manifest.csv and writes
<manifests>/leakage_report.json plus leakage_pairs.csv (every shared hash, all paths).
Exact byte duplicates only; near-duplicates (re-encodes, resizes) are out of scope.
Nothing is deleted or modified.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import section  # noqa: E402

SPLITS = ["train", "val", "test"]


def main():
    out_dir = section("datasets")["manifests"]
    by_hash = defaultdict(list)  # sha -> [(split, relpath, label, generator)]
    for s in SPLITS:
        with open(out_dir / f"{s}_manifest.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["sha256"]:
                    by_hash[r["sha256"]].append((s, r["relpath"], r["label"], r["generator"]))

    report = {"method": "exact SHA-256 of file bytes", "within_split": {}, "cross_split": {}}
    pairs = []
    for s in SPLITS:
        groups = [v for v in by_hash.values() if sum(1 for e in v if e[0] == s) > 1]
        conflicting = [v for v in groups if len({e[2] for e in v if e[0] == s}) > 1]
        report["within_split"][s] = {
            "duplicate_groups": len(groups),
            "extra_copies": sum(sum(1 for e in v if e[0] == s) - 1 for v in groups),
            "groups_with_conflicting_labels": len(conflicting),
        }
    for a, b in combinations(SPLITS, 2):
        shared = [h for h, v in by_hash.items() if {a, b} <= {e[0] for e in v}]
        by_gen = defaultdict(int)
        for h in shared:
            for e in by_hash[h]:
                if e[0] in (a, b):
                    pairs.append({"sha256": h, "split": e[0], "relpath": e[1], "label": e[2], "generator": e[3]})
                if e[0] == "test":
                    by_gen[e[3]] += 1
        report["cross_split"][f"{a}<->{b}"] = {"shared_hashes": len(shared),
                                              "test_files_by_generator": dict(by_gen)}

    with open(out_dir / "leakage_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sha256", "split", "relpath", "label", "generator"])
        w.writeheader()
        w.writerows(pairs)
    (out_dir / "leakage_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
