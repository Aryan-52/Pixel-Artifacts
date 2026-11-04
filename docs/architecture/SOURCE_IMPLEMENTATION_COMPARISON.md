# Source Implementation Comparison

Audit date: 2026-10-31. Everything below was checked against the source files, bundles or published outputs named in each row. Where something was not inspected, this document says so.

## 1. Sources inspected

| ID | Source | Location | What was inspected |
|---|---|---|---|
| A | **fp-v3.0** (locked baseline) | `Implementation/five-pillar-detector/` | Vendored training script `backend/app/model/vendor/five_pillar_detector_v3.py` (1,709 lines), `model/handoff/*`, `model/CHECKSUMS.json`, `CHANGELOG.md`, file tree, `results/` layout |
| B | HPC multi-pillar script (never trained) | `Implementation/multipillar_progan_detector.py` (821 lines) | Complete file |
| C | Sahil's hybrid detector | HF `sahilSpit/checkpoints` (`eval_cnn_testset`, `_v2`, `_v3`); Colab `17x7-yLcoRruGlPiz2-CzXmrW6rGSmEhC` | HF file listing, `combinations/overall.csv` (v2, v3), `feature_cache*/meta.json`, `data_leakage_report.txt`, `full_hybrid/metrics.json` keys. The Colab notebook was seen **only** in a screenshot of its header cell. Its training code has **not** been read. |
| D | Pixel-Artifacts repo | `github.com/Aryan-52/Pixel-Artifacts` | 2 commits, README only. No code. |
| E | `ai-image-detector`, `AIGeneratedImageForensics` | local folders | Directory listing, dataset folders and hardcoded paths only. The AIGIF code is **not audited**. Finding: AIGIF trains on a re-split of the CNN synth **test** set (`datasets/final`), so its results are not zero-shot (see WORKSPACE_ORGANIZATION.md §2). |

## 2. Architecture comparison

| Aspect | A: fp-v3.0 | B: HPC script | C: Sahil v3 eval |
|---|---|---|---|
| Training regime | Frozen encoders → sharded feature cache → train head only | End-to-end; EfficientNet fine-tuned, CLIP frozen | Per-branch heads + score/fusion combos (exact regime unverified) |
| Spatial | EfficientNet-B0 IMAGENET1K_V1, frozen; stages 2,3,4,6 mean+std + final → **1,952** | EfficientNet-B0 trainable, final pool → 1,280 | "stage1" EfficientNet-B0 (unverified) |
| Frequency | FFT, 64 radial bins over r∈[0,√2] + stats → **408** | FFT 32 radial + 8 stats → 40 | `freq` 64-d |
| Color | RGB stats → **133** | RGB moments/corr/hist → 27 | none |
| Physical | Image-derived physical-consistency cues → **89** (not PRNU) | 24 (not PRNU) | `physics` 22 + `prnu` 9 (named PRNU) |
| Semantic | OpenCLIP ViT-B/32 `openai`, frozen → **512** | Same model → 512 | v2: `vit_small_patch16_224` (384); v3: `vit_large_patch14_clip_224.openai` (768) |
| Fusion | Per-pillar projector (256) → gated fusion → MLP (512); pillar dropout 0.15; gate-entropy bonus | Projector 128 → softmax gate → MLP | Gated `full_hybrid`, plus score averaging |
| Params | 1,654,534 trainable | ~5.9M trainable (EffNet) + head | unknown |
| Loss | BCE + label smoothing 0.02 + gate entropy 0.01 | BCE | unknown |
| Optimizer | AdamW lr 5e-4, wd 1e-2 (decay on ndim≥2 only) | AdamW 2e-4, wd 1e-4 | unknown |
| Schedule | 2-epoch warmup + cosine to 0.02×lr; max 40; early stop patience 8 | Cosine, T_max = epochs | unknown |
| Selection | Clean validation BCE loss | Validation F1 | unknown |
| Augmentation | Extra cached copy: blur p=0.5 σ~U(0,3); JPEG p=0.5 q~U(30,100) | Flip and random crop only | v3 meta: `augment: false` |
| Crop | Center crop 224, no downscale (protects pixel-grid artifacts) | Resize 256 → crop 224 (destroys pixel-grid traces) | `crop` 224 |
| Calibration | Temperature fit on clean val, T = max(1, fit) → 1.0 | none | Temperature and calibrated threshold fields present |
| Reproducibility | Feature-code hash, bundle SHA-256, split signatures, atomic checkpoints, resume | config.json, last/best checkpoints | `seed 42` in meta |

## 3. Published results (as stored; protocols are **not** comparable)

| Run | Test subset | N | Acc | AUC | AP | Note |
|---|---|---:|---:|---:|---:|---|
| fp-v3.0 | 12 GANs, ProGAN excluded, balanced subsample, seed 42 | 8,480 | 0.7225 | 0.8375 | 0.8590 | Precision 0.915, recall 0.491 |
| Sahil v2 `stage1` | full test set | 90,245 | 0.6381 | 0.8566 | 0.8714 | real_acc 0.992, fake_acc 0.284 |
| Sahil v2 best combo (`semantic+physics`) | full test set | 90,245 | 0.6843 | 0.7783 | 0.8102 | |
| Sahil v3 `full_hybrid` | 4 generators only (crn, imle, seeingdark, whichfaceisreal) | — | 0.6919 | 0.9518 | 0.9532 | real_acc 0.998, fake_acc 0.386 |

**Common pattern.** Every model ranks well (AUC 0.84–0.95) but is badly calibrated on unseen generators. At threshold 0.5 almost every real image is classified correctly, while most fakes are missed. The problem is the decision boundary under shift, not ranking ability. This matters for how fp-v4.0 reports thresholds (see protocol §7).

**Issues in Sahil's outputs:**
- `data_leakage_report.txt` reports "Training images checked: **0**". The leakage check compared nothing, so it proves nothing.
- The v2 test count (90,245) is 84 lower than the local test set (90,329). The difference is unexplained.
- v3 metrics cover 4 of the 13 generators.
- Branch called `prnu` is a residual statistic; same naming caution as fp-v3 applies.

## 4. Verdict per component

| Component | Decision | Reason |
|---|---|---|
| fp-v3 frozen-encoder + feature-cache design | **Reuse** | Only design that fits 720K images × 50 epochs on the local RTX 5050 (8 GB). Encoders run once per cached copy; epochs then cost seconds to minutes. |
| fp-v3 feature extractors (5 pillars, 3,094-d) | **Reuse unchanged** | Verified, hashed, golden-parity tested in the backend. Changing them breaks v3↔v4 comparability. |
| fp-v3 head, loss, warmup+cosine, selection on clean val BCE | **Reuse** | Already implemented and documented |
| fp-v3 blur+JPEG cached augmentation | **Reuse** | Matches the CNNDetection "Blur+JPEG (0.5)" recipe and the fp-v4 requirement |
| fp-v3 in-RAM cache concat + full-GPU standardisation | **Must change** | 1.44M rows × 3,094 → ~10.7 GB host cache, ~17.8 GB as fp32 on GPU. Does not fit 16 GB RAM / 8 GB VRAM. Needs disk-backed (memmap) standardised store and batch streaming. |
| fp-v3 `collect_samples` from a directory | **Extend** | Train images can be read directly from the split archive (see dataset plan). Add a zip-member sample source. |
| fp-v3 backend / frontend / tests | **Keep, do not touch until fp-v4 locked** | Working, tested (47 passed). Integrate later via bundle swap. |
| HPC script B | **Do not use** | Weaker pillars (40/27/24-d), resize-then-crop destroys pixel traces, never trained, end-to-end cost infeasible locally. Keep as history only. |
| Sahil's per-generator/combination tables, calibrated-threshold reporting | **Adopt the idea** | Useful report format (mean/worst generator accuracy, real_acc vs fake_acc). Re-implement in our evaluator; do not import code. |
| Sahil's ViT-L/14 CLIP semantic branch | **Candidate experiment only** | Higher AUC on 4 generators, but changes semantic dim and compute. Not part of fp-v4.0 (keeps v3↔v4 comparison clean). Recorded in DECISION_LOG. |

## 5. Not yet inspected (remaining audit debt)

- Sahil's Colab training code (optimizer, splits, training data, augmentation). It needs a readable export (`.ipynb`) or repo link.
- `ai-image-detector` and `AIGeneratedImageForensics` folders.
- fp-v3 backend engine internals beyond file inventory (already covered by its own tests).
