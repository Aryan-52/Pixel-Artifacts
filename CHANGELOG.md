# CHANGELOG

## 2026-11-04 — Published to GitHub; t-SNE analysis

### Added
- Repository published at https://github.com/Aryan-52/Pixel-Artifacts, on top of its existing history. Includes:
  - the code, docs, result reports and logs;
  - the packaged model weights (`models/**`, each under 30 MB, with SHA-256 checksums).
- Kept out of the published repository (`.gitignore`):
  - `results/app_jobs/` (images uploaded to the app);
  - `frontend/node_modules/` and `frontend/dist/`;
  - `docs/reports/` (internal team comparison).
- `README.md` rewritten for the actual project; `requirements.txt` added (the versions all results were produced with).
- `scripts/tsne_embeddings.py`: t-SNE of real vs. AI test images (ProGAN, StyleGAN, IMLE; 600 + 600 each, seed 42).
  - It compares two feature spaces: the frozen CLIP pillar, and fp-v4.1's fused embedding.
  - It is a visualisation only: nothing is trained or tuned.
  - Output: `results/figures/tsne_progan_stylegan_imle.npz`. The plot is drawn by the blackbook figure script.

## 2026-11-04 — Results without WhichFaceIsReal (guide's instruction); full-scan eval memory fix

### Added
- `scripts/report_without_generator.py`: recomputes test reports with one generator left out, from each run's saved `test_predictions.npz`. It first checks that the generator mapping reproduces every logged per-generator accuracy. Nothing is re-scored; the test set is not used again. Output: `test_report_no_whichfaceisreal.json` next to each `test_report.json` (31 ablation runs, `runs_ft/full`).
- `scripts/ablation_table.py --without GENERATOR` → `results/ablation/*_no_whichfaceisreal.{csv,md}`. The original tables are unchanged.

### Results (12 generators, 88,329 images)
- fp-v4.1: accuracy 83.50%, AUC 0.943, AP 0.952, real kept real 96.9%, AI caught 70.1%, mean generator accuracy 80.9%.
- fp-v4.0: accuracy 82.23%, AUC 0.933.

### Full-image scan evaluated (D-21; rule fixed before any test result; scored once, nothing tuned)
- `models/fp-v4.1/test_report_fullscan.json` and `test_predictions_fullscan.npz`: 603,172 windows in total.
- All 13 generators: accuracy 86.03%, AUC 0.947, AP 0.956, real kept real 98.0%, AI caught 74.1%.
- Without WhichFaceIsReal (12 generators): accuracy 85.86%, AUC 0.948 (centre crop 83.50%, 0.943). Largest gains are CRN +9.7, SeeingDark +8.6 and IMLE +3.9; DeepFake −3.1 and SAN −0.2.
- The run was split over three sessions and resumed from the checkpoint each time (memory fix, then a session end). Its final cleanup could not delete `results/testing/eval_fullscan_ckpt_fp-v4.1.npz`, which was locked by another process. That file is left in place; the outputs were already written.

### Changed
- `scripts/eval_fullscan.py`: each image chunk is now also capped at 2,048 windows (planned from the manifest's width/height). The first run hit a MemoryError at image ~51k, when 32 large photos were held at once; its log is kept as `results/testing/eval_fullscan_log_run1_memoryerror.txt`. The protocol and the outputs are unchanged, and the run resumes from its checkpoint.

## 2026-11-02 — Source attribution (attribution-v1)

### Added
- `DataSet/Tiny-GenImage/` (user-approved download, 8.36 GB): Hugging Face `TheKernel01/Tiny-GenImage` @ 89c4fe9, CC BY-NC-SA 4.0. 35,000 images; real + ADM, BigGAN, GLIDE, Midjourney, SD 1.5, VQDM, Wukong. `configs/paths.toml` key `tiny_genimage`.
- `src/models/attribution.py`, `scripts/train_attribution.py`: attribution head on the fp-v4.1 standardised features (MLP 3094→512→8), batch 64, max 50 epochs, early stopping (patience 6). Classes: the 7 Tiny-GenImage generators + ProGAN (2,000 train / 250 val from progan_train, 500 test from progan_val). Tiny-GenImage train split → 90/10 train/val (seed 7); its validation split = held-out test. The zero-shot CNN_synth test set is not used.
- `models/attribution-v1/{attribution.pth, report.json, CHECKSUMS.json}`; features cached in `results/attribution/cache/`, log in `results/attribution/train_log.txt`.
- App: `Engine` loads the head (checksum + feature-hash check) and returns `source` per image; "Likely source" tab in advanced results; "Source attribution" tab on Research → Results.
- `tests/test_attribution.py`: app probabilities = head on training-time features.
- Dependency: `pyarrow` 26.0.0 (reads the Parquet dataset).

### Results (models/attribution-v1/report.json)
- Attribution, held-out test (4,000 AI images): accuracy 90.9%, top-2 98.4%, macro F1 0.908. Unknown cut-off 0.786 (5th percentile of correct validation confidences): 91.8% answered, 94.4% correct when answered. Weakest: SD 1.5 71.0% (confused with Wukong), Wukong 80.2%.
- fp-v4.1 detector zero-shot on all 35,000 Tiny-GenImage images: accuracy 59.8%, AUC 0.801, real kept real 96.6%, AI caught 22.9% (BigGAN 64.0%, Midjourney 7.4%). Confirms the GAN-only limitation.

### Changed
- `backend/app/main.py`: `index.html` is served with `Cache-Control: no-cache` (a cached page pointed at deleted bundles after a rebuild).
- Research → Publications: Future scope back to five items; the dataset table is removed (it belongs in the blackbook).

## 2026-11-02 — Pixel-Artifacts app (fp-v4.1) and workspace cleanup

### Added
- `backend/app/`: FastAPI server for fp-v4.1 (`engine.py` = training code + fine-tuned EfficientNet; `analysis.py`, `visuals.py` Grad-CAM / spectrum / noise residual / ELA, `metadata.py`, `evidence.py` benchmark reliability, `jobs.py` folder/ZIP batches with CSV, `research.py`, `main.py`). Serves `frontend/dist`.
- `frontend/`: multi-page React + Vite + three.js app (landing, detector, batch report, history, guide, research ×4, team).
- `models/fp-v4.1/` (packaged by `scripts/package_model.py`, SHA-256 in `CHECKSUMS.json`); `models/fp-v4.0/{test,val}_report.json`; `models/fp-v3.0/` bundle, checksums, handoff and zero-shot summary.
- `scripts/make_app_samples.py` (seeded random benchmark samples, scored by the model); `tests/test_app_parity.py` (app scores = logged test predictions).

### Changed
- `configs/paths.toml`: fp-v3.0 now resolves to `Pixel-Artifacts/models/fp-v3.0`.

### Moved (user-approved)
- `five-pillar-detector/`, `ai-image-detector/`, `AIGeneratedImageForensics/` → `Legacy_Implementations/`; `DataSet/CNNDetection/` → `Legacy_Implementations/CNNDetection-official/`; `multipillar_progan_detector.py` → `Legacy_Implementations/hpc-multipillar/`; `Major Project/stage1_best.pt` → `Sahil research paper - Copy/`.

### Deleted (user-approved)
- Root snapshot zips (4), root duplicate `five_pillar_detector_final.pth` (SHA-256 equal to the kept copy), `Backups/kaggle_upload`, `Research Papers/drive-download-*.zip` (all 36 PDFs present in `papers/`).
- Four duplicate CNN_synth_testset copies (`CNNDetection/dataset/test`, `AIGIF/datasets/{original,unified,final}`), each verified by SHA-256 to contain exactly the 90,329 canonical images.
- `Shared_Resources/uv_cache` (project `.venv` files are hard links, unaffected) and the venvs of the legacy apps.
- Kept: training caches, feature stores, all datasets' canonical copies, all results.

## 2026-10-31 — fp-v4.0 pipeline: extraction, feature store, head trainer

### Added
- `.venv` (Python 3.12.13, torch 2.10.0+cu128, open_clip 3.3.0) — same versions as the fp-v3 Kaggle run.
- `src/models/five_pillar_v3.py`: fp-v3.0 training script copied unchanged (SHA-256 3b77983f…), imported as a library.
- `src/features/extraction.py`, `scripts/extract_features.py`: resumable 2,048-row shards, SHA-256 check against the manifests, D-13 exclusions.
- `src/features/store.py`, `scripts/build_feature_store.py`: train-only standardisation, fp16 matrices per split.
- `src/training/head.py`, `scripts/train_heads.py`: resumable head training (batch 64, max 50 epochs), 31-subset ablation runner, test scoring with per-generator tables.
- `tests/test_head_training.py`: train / pause / resume / finalise on a synthetic store.

### Changed
- `configs/paths.toml`: train data = extracted `progan_train/` (D-14); added `feature_store`, `pretrained`.
- `scripts/check_workspace.py`, `scripts/build_dataset_manifests.py`: read the extracted folder; the 7z in-place reader moved to `Legacy_Implementations/pixel-artifacts-archive-reader/`.

### Verified
- Feature-code hash under the new environment = fp-v3's `369cfa92c5a260ea`.
- val 7,569 / test 90,329 images extracted, 0 decode failures, 0 SHA-256 mismatches.
- train 720,030 images (720,119 minus 89 D-13 overlaps) x 2 copies = 1,440,060 rows extracted in 57.7 min, 0 decode failures, 0 SHA-256 mismatches, 0 non-finite rows.
- Feature store: train_X 8.9 GB, val_X 47 MB, test_X 559 MB (fp16, 3,094 columns), 0 rows dropped.

## 2026-10-31 — Single-root workspace, dataset manifests, leakage check

### Added
- `configs/paths.toml`, `src/paths.py`: all paths resolved relative to the Implementation root.
- `src/data/split_archive.py`: reads `progan_train.zip` in place from the 7 volumes (no extraction).
- `scripts/check_workspace.py`, `scripts/build_dataset_manifests.py`, `scripts/check_leakage.py`.
- `docs/architecture/WORKSPACE_ORGANIZATION.md` (inventory, duplicates, dependencies, proposed moves M1–M10).
- `docs/datasets/DATASET_AUDIT_RESULTS.md`; repository scaffolding (`src/*`, `models/*`, `results/*`), `.gitignore`.
- Manifests in `DataSet/Our Dataset/CNN/manifests/` (outside the repo).

### Results
- train 720,119 / val 8,000 / test 90,329 images, 0 decode failures.
- Cross-split overlaps are real images only (train 90, val 432 rows); excluded from fp-v4 train/val.

### Not changed
- No existing file moved, renamed or deleted. Archives were read only.

## 2026-10-31 — Phase 0 audit and Phase 1 dataset verification

### Added
- `docs/architecture/SOURCE_IMPLEMENTATION_COMPARISON.md`
- `docs/architecture/RECOMMENDED_ARCHITECTURE.md`
- `docs/architecture/REPOSITORY_STRUCTURE.md`
- `docs/architecture/DECISION_LOG.md`
- `docs/datasets/DATASET_AUDIT_PLAN.md`
- `docs/experiments/FP_V4_EXPERIMENT_PROTOCOL.md`

### Verified
- `progan_train.7z.001–007`: complete, 7-Zip CRC test OK. Stored wrapper around a single `progan_train.zip` holding 720,119 PNG (360,059 real / 360,060 fake, 20 categories).
- `progan_val`: 8,000 PNG. `CNN_synth_testset_1`: 90,329 images, 13 generators.

### Not changed
- fp-v3.0 artifacts, code, datasets and archives: untouched.

### Experiments / results affected
- None. No training or evaluation was run.
