# Dataset Audit Plan

Location: `DataSet/Our Dataset/CNN/` under the workspace root, resolved through `configs/paths.toml` (`[datasets]`). Facts marked ✔ were verified on 2026-10-31.

**Decision (D-1, accepted by Aryan 2026-10-31):** train images are read **in place** from the split archive by `src/data/split_archive.py`. No extraction anywhere, nothing written to C:. The archive volumes are opened read-only.

## 1. Training archive — `progan_train.7z.001…007`

| Check | Result |
|---|---|
| ✔ All 7 volumes present, no `.crdownload` | 6 × 10,737,418,240 B + 10,499,493,436 B = 74,924,002,876 B |
| ✔ 7-Zip 26.01 `t` (CRC over all volumes) | `Everything is Ok`, exit 0 |
| ✔ Archive structure | One 7z, **Method = Copy** (stored), containing a **single file `progan_train.zip`** (74,924,002,746 B, CRC `155BF391`) |
| ✔ Inner zip readable in place | Starts at byte offset 32 of the concatenated volumes (`PK\x03\x04`). Python `zipfile` lists it without extraction. |
| ✔ Inner zip contents | 720,119 files, all `.png`; 20 categories × {`0_real`,`1_fake`}; 18,003 per class-folder except `boat/0_real` = 18,002 → **360,059 real / 360,060 fake** |
| ✔ Uncompressed size | 69.75 GiB (74.9 GB) |
| ✔ Spot reads | 6 members read with CRC check; all valid PNG |

**Why the earlier extraction failed (likely).** The naive route writes the 75 GB zip and then 75 GB of images, so it needs about 150 GB free. Only 72 GB is free on D: (97 GB on C:, on the same physical NVMe).

### Access method (decided: read in place)

`open_inner_zip(first_volume)` takes the zip span from the 7z start header (`NextHeaderOffset`). It checks the 7z signature and the `PK\x03\x04` zip signature, then returns a standard `zipfile.ZipFile`. Each DataLoader / pool worker opens its own handle. Every member read is CRC-checked. Extra disk used: 0 bytes.

Rejected: extracting to C: (against the single-root rule) and extracting to D: (not enough space without deleting the volumes).

## 2. Validation — `progan_val/` (from `progan_val.zip`, 830,792,545 B)

✔ 8,000 PNG: 20 categories × 200 real + 200 fake. Matches the HPC record (zip tested OK there).

## 3. Test — `CNN_synth_testset_1/`

✔ 90,329 files (88,310 png, 2,000 jpeg, 19 bmp). `CNN_synth_testset.zip` itself is **not present locally**, so the folder cannot be re-verified against the archive.

| Generator | Real | Fake | Layout |
|---|---:|---:|---|
| progan | 4,000 | 4,000 | 20 categories |
| biggan | 2,000 | 2,000 | flat |
| crn | 6,382 | 6,382 | flat |
| cyclegan | 1,321 | 1,321 | apple, horse, orange, summer, winter, zebra |
| deepfake | 2,707 | 2,698 | flat (unbalanced by 9) |
| gaugan | 5,000 | 5,000 | flat |
| imle | 6,382 | 6,382 | flat |
| san | 219 | 219 | flat |
| seeingdark | 180 | 180 | flat |
| stargan | 1,999 | 1,999 | flat |
| stylegan | 5,991 | 5,991 | bedroom, car, cat |
| stylegan2 | 7,988 | 7,988 | car, cat, church, horse |
| whichfaceisreal | 1,000 | 1,000 | flat |

## 4. Manifests and leakage checks

Output folder: `DataSet/Our Dataset/CNN/manifests/` (`[datasets].manifests`).

| Step | Script | Output |
|---|---|---|
| Manifest + full decode, one pass per image (SHA-256 of bytes, PIL full decode → size/mode/format/errors). Labels come only from `0_real`/`1_fake` directories. Resumable 10k-image chunks. | `scripts/build_dataset_manifests.py` | `{train,val,test}_manifest.csv`, `{split}_summary.json`, `parts/` |
| Exact-duplicate report: within each split (including label-conflicting duplicates) and train↔val, train↔test, val↔test, with test hits broken down by generator | `scripts/check_leakage.py` | `leakage_report.json`, `leakage_pairs.csv` |
| Path sanity from the single root | `scripts/check_workspace.py --hash-bundle` | console |

Policy: the reports list duplicates; **no source file is ever deleted or modified**. If train↔val/test overlaps exist, the training sample list excludes them (fp-v3 rule) and the exclusion list is saved with the fp-v4 run. Near-duplicate detection (re-encoded or resized copies) is not covered by exact hashing and is recorded as a limitation.

Results are recorded in `DATASET_AUDIT_RESULTS.md`.
