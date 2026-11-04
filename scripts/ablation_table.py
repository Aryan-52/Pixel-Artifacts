"""Collect the 31 pillar-subset runs into the ablation tables (reads finished runs only).

    .venv\\Scripts\\python.exe scripts\\ablation_table.py [--runs DIR] [--out DIR] [--without GENERATOR]

Writes ablation_results.csv / .md (one row per subset) and per_generator_acc.csv / per_generator_auc.csv.
--without whichfaceisreal reads test_report_no_whichfaceisreal.json (scripts/report_without_generator.py)
and writes the same files with a _no_whichfaceisreal suffix.
Rows are ordered by number of pillars, then by the fixed pillar order; nothing is chosen by test score.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PILLARS = ["spatial", "frequency", "color", "physical", "semantic"]


def subsets():
    for k in range(1, 6):
        for c in itertools.combinations(PILLARS, k):
            yield list(c)


def tag(active):
    return "full" if len(active) == 5 else "+".join(active)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=REPO / "results" / "training" / "runs", type=Path)
    ap.add_argument("--out", default=REPO / "results" / "ablation", type=Path)
    ap.add_argument("--without", default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = f"test_report_no_{a.without}.json" if a.without else "test_report.json"
    suffix = f"_no_{a.without}" if a.without else ""

    rows, acc_rows, auc_rows, missing = [], [], [], []
    for active in subsets():
        d = a.runs / tag(active)
        if not (d / "DONE.json").exists():
            missing.append(tag(active))
            continue
        done = json.loads((d / "DONE.json").read_text())
        val = json.loads((d / "val_report.json").read_text())["overall"]
        test = json.loads((d / report).read_text())
        t = test["overall"]
        sens = test["sensitivity_without_train_val_overlap"]
        rows.append({
            "pillars": len(active), "subset": " + ".join(active),
            **{p: int(p in active) for p in PILLARS},
            "best_epoch": done["best_epoch"],
            "val_acc": val["accuracy"], "val_auc": val["roc_auc"],
            "test_acc": t["accuracy"], "test_auc": t["roc_auc"], "test_ap": t["average_precision"],
            "test_f1": t["f1"], "real_acc": t["real_acc"], "fake_acc": t["fake_acc"],
            "mean_gen_acc": test["mean_generator_acc"], "worst_gen_acc": test["worst_generator_acc"],
            "mean_gen_auc": test["mean_generator_auc"],
            "test_acc_no_overlap": sens["accuracy"], "test_auc_no_overlap": sens["roc_auc"],
        })
        gens = sorted(test["per_group"])
        acc_rows.append({"subset": tag(active), **{g: test["per_group"][g]["accuracy"] for g in gens}})
        auc_rows.append({"subset": tag(active), **{g: test["per_group"][g]["roc_auc"] for g in gens}})

    for name, data in [("ablation_results", rows), ("per_generator_acc", acc_rows), ("per_generator_auc", auc_rows)]:
        if data:
            with open(a.out / f"{name}{suffix}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    full_acc = next((r["test_acc"] for r in rows if r["pillars"] == 5), None)
    scope = f"12 generators, without {a.without}" if a.without else "13 generators"
    lines = [f"# fp-v4.0 pillar ablation (zero-shot test: CNN_synth_testset, {scope}, threshold 0.5)", "",
             "| # | Subset | Val acc | Test acc | Δ vs full | Test AUC | Test AP | Real acc | Fake acc | Mean gen acc | Worst gen acc |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        delta = "" if full_acc is None else f"{100 * (r['test_acc'] - full_acc):+.1f}"
        lines.append(f"| {r['pillars']} | {r['subset']} | {r['val_acc']:.4f} | {r['test_acc']:.4f} | {delta} | "
                     f"{r['test_auc']:.4f} | {r['test_ap']:.4f} | {r['real_acc']:.3f} | {r['fake_acc']:.3f} | "
                     f"{r['mean_gen_acc']:.4f} | {r['worst_gen_acc']:.4f} |")
    if missing:
        lines += ["", "Not finished: " + ", ".join(missing)]
    (a.out / f"ablation_results{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} runs -> {a.out}" + (f" | missing: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
