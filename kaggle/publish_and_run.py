"""Unattended Kaggle hand-off: upload the feature store and code as PRIVATE datasets, wait until
Kaggle has processed them, then push the training notebook and start it on GPU T4 x2.

    .venv\\Scripts\\python.exe kaggle\\publish_and_run.py --user <kaggle-username>

Needs a logged-in Kaggle CLI (`.venv\\Scripts\\kaggle.exe auth login`). Safe to rerun: an
existing dataset gets a new version instead of failing. Log: results/training/logs/kaggle_publish.log
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO / ".venv" / "Scripts" / "kaggle.exe")
STORE = REPO / "results" / "training" / "feature_store"
CODE = REPO.parent / "Backups" / "kaggle_upload" / "code"
KERNEL = REPO.parent / "Backups" / "kaggle_upload" / "kernel"
LOG = REPO / "results" / "training" / "logs" / "kaggle_publish.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def kaggle(*args, check=True):
    log("kaggle " + " ".join(args))
    r = subprocess.run([KAGGLE, *args], capture_output=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    out = (r.stdout + r.stderr).strip()
    log(out[-2000:])
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {args[0]} {args[1]} failed ({r.returncode})")
    return r.returncode, out


def dataset(folder: Path, user: str, slug: str, title: str):
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    code, out = kaggle("datasets", "create", "-p", str(folder), check=False)
    if code != 0 and "already" in out.lower():
        kaggle("datasets", "version", "-p", str(folder), "-m", f"update {datetime.now():%Y-%m-%d %H:%M}")
    elif code != 0:
        raise RuntimeError(f"dataset create failed for {slug}")
    for _ in range(240):                              # up to 2 h for Kaggle to process 9 GB
        _, st = kaggle("datasets", "status", f"{user}/{slug}", check=False)
        if "ready" in st.lower():
            return
        if "error" in st.lower():
            raise RuntimeError(f"dataset {slug} processing error: {st}")
        time.sleep(30)
    raise RuntimeError(f"dataset {slug} not ready after 2 h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    a = ap.parse_args()
    log("=== Kaggle hand-off started ===")
    dataset(CODE, a.user, "pixel-artifacts-code", "Pixel-Artifacts code")
    dataset(STORE, a.user, "pixel-artifacts-feature-store", "Pixel-Artifacts feature store")
    KERNEL.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "kaggle" / "pixel_artifacts_fp_v4_heads.ipynb", KERNEL / "pixel_artifacts_fp_v4_heads.ipynb")
    (KERNEL / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{a.user}/pixel-artifacts-fp-v4-heads", "title": "Pixel-Artifacts fp-v4 heads",
        "code_file": "pixel_artifacts_fp_v4_heads.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{a.user}/pixel-artifacts-feature-store", f"{a.user}/pixel-artifacts-code"],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    kaggle("kernels", "push", "-p", str(KERNEL), "--accelerator", "NvidiaTeslaT4")
    log("=== notebook pushed and running; check kaggle.com/code ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FAILED: {e}")
        sys.exit(1)
