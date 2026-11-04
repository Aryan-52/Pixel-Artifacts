# Workspace Organization (single root)

**Rule:** every project resource lives under `D:\ARYAN\Projects\Major Project\Implementation` (the *workspace root*). Code resolves paths from `configs/paths.toml`, relative to that root. `PIXEL_ARTIFACTS_WORKSPACE` can override the root.

Inventory date: 2026-10-31. **Update 2026-11-02:** M4–M7 were applied, the root zips and the four duplicate test-set copies (§5, SHA-256 verified) were deleted with the user's approval; see `CHANGELOG.md`. M1–M3 and M9 were not applied. The tables below describe the state before that cleanup.

## 1. Inventory (sizes measured)

| Item | Size | Notes |
|---|---:|---|
| `DataSet/Our Dataset/CNN/` | 91.0 GB | 7 train volumes (74.9 GB), `progan_val/` + zip, `CNN_synth_testset_1/` (18.7 GB) |
| `DataSet/CNNDetection/` | 20.1 GB | Official Wang et al. repo clone. `dataset/test` = **copy** of the test set (18.7 GB); `weights/` 0.53 GB; `venv/` 0.9 GB |
| `AIGeneratedImageForensics/` | 56.7 GB | `datasets/original` = **copy** of the test set; `unified` and `final` = the same 90,329 files renamed / re-split (18.7 GB each) |
| `five-pillar-detector/` | 1.8 GB | fp-v3.0 system (backend, frontend, tests, bundle, results) |
| `ai-image-detector/` | 1.6 GB | Earlier app (venv 0.96 GB, weights 0.58 GB) |
| Root files | 0.9 GB | `five-pillar-detector.zip`, `five-pillar-detector-training-source.zip`, `ai-image-detector.zip`, `FIVE_PILLAR_FINAL_HANDOFF.zip`, `five_pillar_detector_final.pth`, `multipillar_progan_detector.py` |
| `Project Context/` | <1 MB | Handoff / direction texts |
| `Backups/` | empty | |
| Outside the root | none found | Desktop, Downloads, Documents and `C:\` searched by name |

## 2. Duplicates found

| Duplicate | Evidence | Size |
|---|---|---:|
| CNN synth test set × 5 | `CNNDetection/dataset/test` and `AIGIF/datasets/original/CNN_synth_testset_1`: identical relative paths and sizes to the canonical copy (90,329 files; CNNDetection has one extra `download_testset.sh`). `AIGIF/unified` and `AIGIF/final`: same 90,329 files, renamed. | ~74.7 GB redundant |
| fp-v3.0 bundle × 2 | Root `five_pillar_detector_final.pth` and `five-pillar-detector/model/…`: both SHA-256 `9cd68987…2430` | 6.4 MB |
| Source snapshots | Root `.zip` files are snapshots of the folders beside them | 0.9 GB |

**Legacy-data warning.** `AIGeneratedImageForensics/datasets/final/{train,val,test}` is a re-split of the CNN synth **test** set. Any legacy model trained on it saw the test generators, so it must never be reported as zero-shot.

## 3. Dependencies that a move would break

| Reference | Points to | Effect of moving |
|---|---|---|
| `five-pillar-detector/backend/scripts/run_parallel_robustness.py:20` | absolute path to `AIGeneratedImageForensics/datasets/original/CNN_synth_testset_1` | Breaks if AIGIF moves or its copy is removed. The fp-v3 zero-shot results were produced from this copy. |
| `five-pillar-detector/backend/.venv` | created with absolute path to this folder | venv launchers break after moving the folder; recreate the venv |
| `ai-image-detector/venv`, `DataSet/CNNDetection/venv` | same | same |
| `AIGeneratedImageForensics/scripts/utils/*.py` | `D:\ARYAN\MCA\2. SEM 2\…` (old location) | already broken; no change |
| `five-pillar-detector/backend/app/config.py` | repo-relative (`<repo>/model`) | moves cleanly with its folder |
| `Pixel-Artifacts/configs/paths.toml` | workspace-relative | update one line per moved item |

## 4. Proposed moves (awaiting approval)

All of these are same-volume renames, so they are instant and use no extra disk. None runs while a manifest build is reading the files.

| # | From | To | Follow-up |
|---|---|---|---|
| M1 | `DataSet/Our Dataset/CNN/progan_train.7z.001–007` | `…/CNN/ProGAN_Train_Archives/` | update `paths.toml`; re-run `check_workspace.py` |
| M2 | `…/CNN/progan_val/`, `progan_val.zip` | `…/CNN/ProGAN_Validation/progan_val/`, `…/ProGAN_Validation/progan_val.zip` | same |
| M3 | `…/CNN/CNN_synth_testset_1/` | `…/CNN/CNN_Synth_Test/CNN_synth_testset_1/` | same |
| M4 | root `five_pillar_detector_final.pth` | `Pixel-Artifacts/models/fp-v3.0/` (gitignored) + copy `CHECKSUMS.json` and `handoff/*.json,csv,txt` | verify SHA-256 after the move |
| M5 | `five-pillar-detector/`, `ai-image-detector/`, `AIGeneratedImageForensics/` | `Legacy_Implementations/` | recreate venvs; patch the `run_parallel_robustness.py` path (recorded) |
| M6 | `multipillar_progan_detector.py` | `Legacy_Implementations/hpc-multipillar/` | none |
| M7 | `DataSet/CNNDetection/` (official repo) | `Legacy_Implementations/CNNDetection-official/` | its `weights/` stay with it (reference baseline) |
| M8 | root `*.zip` snapshots (4 files) | `Backups/` | none |
| M9 | `Project Context/` | `Project_Context/` | none |
| M10 | — | create `Shared_Resources/{model_weights,research,technical_handoffs}` | populate later only with non-duplicated material |

`five-pillar-detector-source` does not exist as a folder. Its content is `five-pillar-detector-training-source.zip`, which goes to `Backups/` under M8.

## 5. Redundant copies (optional, approval required separately)

The four extra test-set copies (~74.7 GB) would be removed only after:

1. a full SHA-256 comparison of each copy against `manifests/test_manifest.csv` (renamed copies are matched by hash, not by name); and
2. M5 is done and the robustness script points at the canonical copy.

Until then they stay untouched.
