"""Verify that every configured dataset and model path resolves from the workspace root.

    python scripts/check_workspace.py [--hash-bundle]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import WORKSPACE_ROOT, section  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-bundle", action="store_true", help="also verify the fp-v3.0 bundle SHA-256")
    args = ap.parse_args()

    ds, models = section("datasets"), section("models")
    print(f"workspace root: {WORKSPACE_ROOT}")
    ok = True

    def check(label, path: Path, must_exist=True):
        nonlocal ok
        exists = path.exists()
        ok &= exists or not must_exist
        print(f"  [{'ok' if exists else ('MISSING' if must_exist else 'absent')}] {label}: {path}")

    check("CNN dataset root", ds["cnn_root"])
    check("progan_train (extracted)", ds["progan_train"])
    check("progan_val", ds["progan_val"])
    check("CNN synth test", ds["cnn_synth_test"])
    check("manifests dir", ds["manifests"], must_exist=False)
    check("fp-v3.0 bundle", models["fp_v3_bundle"])
    check("fp-v4.0 dir", models["fp_v4_dir"], must_exist=False)

    if args.hash_bundle and models["fp_v3_bundle"].exists():
        h = hashlib.sha256(models["fp_v3_bundle"].read_bytes()).hexdigest()
        match = h == models["fp_v3_bundle_sha256"]
        ok &= match
        print(f"  [{'ok' if match else 'MISMATCH'}] fp-v3.0 bundle sha256 {h}")

    print("RESULT:", "all required paths resolve" if ok else "problems found")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
