# Dataset Audit Results (2026-10-31)

Manifests: `DataSet/Our Dataset/CNN/manifests/`, built by `scripts/build_dataset_manifests.py`. Every image was read in full, SHA-256 hashed and fully decoded with PIL. Train images were read in place from the split archive.

## Inventory

| Split | Images | Real | Fake | Decode failures | Format | Resolution | Size |
|---|---:|---:|---:|---:|---|---|---:|
| train (`progan_train`) | 720,119 | 360,059 | 360,060 | 0 | PNG, RGB | all 256×256 | 69.75 GiB |
| val (`progan_val`) | 8,000 | 4,000 | 4,000 | 0 | PNG, RGB | all 256×256 | 0.77 GiB |
| test (`CNN_synth_testset_1`) | 90,329 | 45,169 | 45,160 | 0 | 88,310 PNG, 2,000 JPEG, 19 BMP; 10 grayscale (`L`) | 332 distinct (256×256 most common: 48,711; up to 6030×4032) | 18.68 GiB |

Per-category and per-generator counts are in `{split}_summary.json`.

## Exact-duplicate / leakage check (`scripts/check_leakage.py`)

| Check | Result |
|---|---|
| Within train | 1,413 duplicate groups (1,468 extra copies); 0 with conflicting labels |
| Within val | none |
| Within test | 10,652 groups (10,878 extra copies); 0 with conflicting labels |
| train ↔ val | 32 shared hashes |
| train ↔ test | 55 shared hashes (test files: progan 33, stylegan 17, stylegan2 23) |
| val ↔ test | 400 shared hashes (test files: stylegan 200, stylegan2 400) |

**Every cross-split overlap is a real (label 0) image.** No fake image is shared between splits. The overlaps are LSUN source photos reused across the ProGAN, StyleGAN and StyleGAN2 test folders. Rows in `leakage_pairs.csv`: train 90, val 432, test 673 (both sides of every pair listed).

## Policy for fp-v4.0

- Test set: unchanged (canonical benchmark; keeps comparability with published numbers).
- **Train:** the 90 real images whose bytes also occur in val or test are excluded from fp-v4 training. The list is saved with the run.
- **Val:** the 432 real images whose bytes also occur in train or test are excluded from model selection and temperature fitting.
- Within-split duplicates are kept; they carry no label conflict and are a property of the source data.
- Test results also report metrics with the 673 overlapping real test files removed, as a sensitivity check.
- Limitation: exact hashing does not detect re-encoded or resized near-duplicates.
