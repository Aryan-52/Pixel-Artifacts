"""Copy a finished fp-v4.x run into models/<version>/ for the app, with SHA-256 checksums.

    python scripts/package_model.py            # fp-v4.1 (fine-tuned spatial pillar)
    python scripts/package_model.py fp-v4.2    # augmented spatial pillar + ViT-L/14 semantic

The app loads only from models/<version>/, never from results/training/. Source files are copied,
not moved, so the training outputs stay as they are.
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "training"

SOURCES = {
    "fp-v4.1": {
        "head.pth": RUNS / "runs_ft" / "full" / "best_model.pth",
        "spatial_effnet.pth": RUNS / "spatial_ft" / "best.pth",
        "test_report.json": RUNS / "runs_ft" / "full" / "test_report.json",
        "val_report.json": RUNS / "runs_ft" / "full" / "val_report.json",
        "test_predictions.npz": RUNS / "runs_ft" / "full" / "test_predictions.npz",
        "head_history.csv": RUNS / "runs_ft" / "full" / "history.csv",
        "spatial_history.csv": RUNS / "spatial_ft" / "history.csv",
    },
    "fp-v4.2": {
        "head.pth": RUNS / "runs_v42" / "full" / "best_model.pth",
        "spatial_effnet.pth": RUNS / "spatial_ft_v2" / "best.pth",
        "store_meta.json": RUNS / "feature_store_v42" / "store_meta.json",
        "test_report.json": RUNS / "runs_v42" / "full" / "test_report.json",
        "val_report.json": RUNS / "runs_v42" / "full" / "val_report.json",
        "test_predictions.npz": RUNS / "runs_v42" / "full" / "test_predictions.npz",
        "head_history.csv": RUNS / "runs_v42" / "full" / "history.csv",
        "spatial_history.csv": RUNS / "spatial_ft_v2" / "history.csv",
    },
    "fp-v4.3": {                                   # end-to-end spatial pillar (D-23)
        "head.pth": RUNS / "e2e_v43" / "best_model.pth",
        "spatial_effnet.pth": RUNS / "e2e_v43" / "best_effnet.pth",
        "test_report.json": RUNS / "e2e_v43" / "test_report.json",
        "val_report.json": RUNS / "e2e_v43" / "val_report.json",
        "test_predictions.npz": RUNS / "e2e_v43" / "test_predictions.npz",
        "head_history.csv": RUNS / "e2e_v43" / "history.csv",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(version: str = "fp-v4.1") -> None:
    out = REPO / "models" / version
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, src in SOURCES[version].items():
        if not src.exists():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, out / name)
        files[name] = {"sha256": sha256(out / name), "source": str(src.relative_to(REPO))}
        print(f"{name:24s} {files[name]['sha256'][:16]}")
    (out / "CHECKSUMS.json").write_text(json.dumps(
        {"model_version": version, "packaged": datetime.now().isoformat(timespec="seconds"),
         "files": files}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
