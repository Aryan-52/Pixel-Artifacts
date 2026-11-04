# fp-v4.0 Experiment Protocol (draft — open decisions marked)

## 1. Hypothesis

Training the unchanged fp-v3.0 five-pillar head on the full ProGAN training set (720,119 images, 20 categories) improves zero-shot cross-generator performance over fp-v3.0 (24,000 images, 4 categories).

## 2. Data

| Split | Source | Use |
|---|---|---|
| Train | `progan_train` (720,119; read in place from the split archive, never extracted) | fit head; fit normalisation stats |
| Val | `progan_val` (8,000) | epoch selection, temperature fit, LR schedule. **Clean only.** |
| Test | `CNN_synth_testset_1` (90,329, 13 generators) | **once**, after the bundle is locked |

Before feature extraction, cross-split exact duplicates (from `leakage_report.json`) are excluded from train (fp-v3 rule), and the exclusion list is stored with the run.

All locations come from `configs/paths.toml`, relative to the workspace root. Feature caches go to `[outputs].feature_cache` (`Pixel-Artifacts/results/training/feature_cache`), bundles to `Pixel-Artifacts/models/fp-v4.0/`, and outputs to `Pixel-Artifacts/results/<phase>/`. Each run records the SHA-256 of the dataset manifests it used.

## 3. Fixed configuration (inherited from fp-v3.0, verified in source)

| Item | Value |
|---|---|
| Preprocess | RGB, center crop 224, upscale only if min side < 224, no downscale |
| Train augmentation | 1 extra cached copy: Gaussian blur p=0.5 σ~U(0,3), then JPEG p=0.5 q~U(30,100) |
| Extractors | fp-v3 code, feature-code hash `369cfa92c5a260ea` (Python 3.12) |
| Normalisation | per-feature mean/std on train rows only (clean + aug), clip ±10 |
| Head | pillar_dim 256, hidden 512, dropouts 0.3/0.2/0.1/0.15/0.1, pillar dropout 0.15 |
| Loss | BCEWithLogits, label smoothing 0.02, pos_weight = n_real/n_fake, gate-entropy 0.01 |
| Optimiser | AdamW lr 5e-4, wd 1e-2 (no decay on biases/norms), grad clip 1.0, AMP |
| Schedule | linear warmup 2 epochs → cosine to 0.02 × lr |
| Selection | min clean-val BCE (min_delta 1e-4) |
| Calibration | temperature on clean val logits, T = max(1, fit); threshold fixed 0.5 |
| Seed | 42 |

## 4. Open decisions (need sign-off before Phase 3)

| ID | Question | Recommendation |
|---|---|---|
| D-1 | ~~Read in place or extract?~~ | **Decided: read in place** (Aryan, 2026-10-31) |
| D-11 | Training environment: the fp-v3 backend venv has CPU-only torch (2.10.0+cpu) | Separate CUDA-enabled venv under `Pixel-Artifacts/.venv` (gitignored), Python 3.12 (combines with D-7) |
| D-5 | Spec says "batch 64, 50 epochs". fp-v3 used head batch 256 / max 40 / early stop 8. | Head batch 64, max 50 epochs, **keep early stopping** (patience 8) and report the stop epoch. 64 is also the v3 extraction batch. |
| D-6 | Augmented copies per training image (fixed per cache, not re-randomised per epoch) | 1 copy (= v3). More copies multiply extraction time linearly. |
| D-7 | Python version for extraction (the feature-code hash depends on it; local default is 3.11.9) | Python 3.12 venv to reproduce the v3 hash, so v3 and v4 features are byte-comparable |

## 5. Run order

1. Benchmark extraction on 2,048 train images; record img/s, VRAM and RAM; estimate full time.
2. Build train/val caches (resumable shards).
3. Build standardised memmaps; record stats checksum.
4. Train; log loss, acc, P, R, F1, AUC, AP, lr, epoch time, GPU memory and throughput per epoch.
5. Lock bundle (weights, config, stats, temperature, threshold, hashes, manifests).
6. Test once: overall + per generator (counts, acc, P, R, F1, AUC, AP, confusion matrix). Report ProGAN-test (seen generator) **separately** from the 12 unseen generators.
7. Robustness (JPEG 95/75/50/30; blur σ 0.5/1/2/3; resize 0.5/0.75/1.5 bicubic; noise σ 2/5/10 on 0–255) and inference-time pillar masking. Labelled as such, never as retrained ablation.

## 6. v3 ↔ v4 comparison rule

Both bundles are scored on the identical test file list. Because v3 was previously reported on an 8,480-image subsample, v3 is re-run on the full 90,329 list. The old number is not reused for comparison.

## 7. Threshold reporting

Every source audited so far ranks well but misses most unseen fakes at 0.5 (see SOURCE_IMPLEMENTATION_COMPARISON §3). Headline metrics use the fixed 0.5. Threshold-free AUC/AP and real-vs-fake accuracy are always reported alongside. No threshold is tuned on test data.
