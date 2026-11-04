"""Re-report finished runs with one test generator left out, from the saved test predictions.

    .venv\\Scripts\\python.exe scripts\\report_without_generator.py [--exclude whichfaceisreal] [RUN_DIR ...]

No model is re-run and nothing is re-scored: each run's test_predictions.npz (written by the single
test pass) is read, the excluded generator's rows are dropped, and the same metrics as
test_report.json are recomputed. Writes test_report_no_<generator>.json next to test_report.json;
the original reports are not changed. Default runs: the 31 ablation runs, fp-v4.0 and fp-v4.1.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.paths import section  # noqa: E402
from src.training.head import Store, _metrics, _overlap_rows, _report  # noqa: E402

RUNS = REPO / "results" / "training"


def report(store, gen, run_dir: Path, exclude: str) -> dict:
    p = np.load(run_dir / "test_predictions.npz")
    y, logits = p["y"], p["logits"]
    full = json.loads((run_dir / "test_report.json").read_text())
    # the group mapping must reproduce the logged report before anything is left out
    for g, m in full["per_group"].items():
        assert abs(_metrics(y[gen == g], logits[gen == g])["accuracy"] - m["accuracy"]) < 1e-9, (run_dir, g)
    keep = gen != exclude
    rep = _report(y[keep], logits[keep], gen[keep])
    per = rep["per_group"]
    rep["mean_generator_acc"] = float(np.mean([m["accuracy"] for m in per.values()]))
    rep["worst_generator_acc"] = float(min(m["accuracy"] for m in per.values()))
    rep["mean_generator_auc"] = float(np.mean([m["roc_auc"] for m in per.values()]))
    overlap = set(_overlap_rows(store, store.index("test")).tolist())
    rows = np.array([i for i in np.where(keep)[0] if i not in overlap])
    rep["sensitivity_without_train_val_overlap"] = {"n_removed": int(keep.sum() - len(rows)),
                                                    **_metrics(y[rows], logits[rows])}
    rep.update(excluded_generator=exclude, threshold=full["threshold"],
               source="test_predictions.npz of the single test pass; nothing re-scored",
               note=full["note"])
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="whichfaceisreal")
    ap.add_argument("runs", nargs="*", type=Path)
    a = ap.parse_args()
    runs = a.runs or sorted(d for d in (RUNS / "runs").iterdir() if (d / "test_predictions.npz").exists()) \
        + [RUNS / "runs_ft" / "full"]
    store = Store(section("outputs")["feature_store"])
    gen = np.array([r["relpath"].split("/")[0] for r in store.index("test")])
    assert a.exclude in set(gen), a.exclude
    for d in runs:
        rep = report(store, gen, d, a.exclude)
        (d / f"test_report_no_{a.exclude}.json").write_text(json.dumps(rep, indent=1))
        o = rep["overall"]
        print(f"{d.relative_to(REPO)}: n {o['n']} acc {o['accuracy']:.4f} auc {o['roc_auc']:.4f} "
              f"gen-mean {rep['mean_generator_acc']:.4f}")


if __name__ == "__main__":
    main()
