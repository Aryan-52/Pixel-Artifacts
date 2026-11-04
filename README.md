# Pixel-Artifacts

Detects whether an image is a real photograph or AI-generated. The detector looks at every image through five independent **pillars** and combines them with a learned per-image gate. It ships as a web application that explains each verdict.

MCA Major Project, Sardar Patel Institute of Technology, Mumbai.
Team: Aryan Dandekar, Gyana Karn and Sahil Sawant. Guides: Prof. Sakina Salmani and Prof. Harshil Kanakia.

## Results (model `fp-v4.1`)

The model is trained on ProGAN only and tested zero-shot on the 13-generator CNNDetection test set (90,329 images). Every setting was chosen on ProGAN validation data, the threshold is fixed at 0.5, and the test set was scored once.

| Protocol | Accuracy | ROC-AUC | AP | Real kept real | AI caught |
|---|---|---|---|---|---|
| 224 px centre crop (training protocol) | 83.15% | 0.943 | 0.952 | 97.0% | 69.3% |
| Full-image window scan (app protocol) | 86.03% | 0.947 | 0.956 | 98.0% | 74.1% |

- **Comparison:** on the 11 generators shared with UniversalFakeDetect (Ojha et al., CVPR 2023, Table 3), the mean accuracy is 81.3%, against 80.9% for UniversalFakeDetect and 73.3–77.6% for Wang et al. (CVPR 2020).
- **Per-generator results:** `models/fp-v4.1/test_report.json`.
- **31-subset ablation:** `results/ablation/`.
- **Limitation:** the detector is GAN-trained. On diffusion images (Tiny-GenImage) it catches only 22.9% of AI images.

## How it works

```
image ─► 224×224 window at full resolution (never downscaled)
          ├─ spatial    EfficientNet-B0, fine-tuned          1952-d
          ├─ frequency  radial FFT statistics (image + residual) 408-d
          ├─ colour     RGB / YCbCr / opponent statistics       133-d
          ├─ physical   illumination, shading, noise-residual   89-d   (image-derived; not camera PRNU)
          └─ semantic   OpenCLIP ViT-B/32, frozen               512-d
       ─► train-only standardisation ─► 5 projectors (256-d) ─► softmax gate ─► MLP ─► p(AI)
```

The app covers large images with a grid of 224 px windows that reads every pixel, and averages the window scores. Alongside the verdict it shows:
- the influence of each pillar;
- Grad-CAM, the Fourier spectrum, the noise residual and error-level analysis;
- metadata;
- a reliability statement from the benchmark;
- the likely source generator (separate 8-class attribution head).

## Repository layout

| Path | Contents |
|---|---|
| `src/models/five_pillar_v3.py` | Feature extractors and the gated fusion head (feature-code hash checked at load time) |
| `src/features/` | Resumable feature extraction and the standardised fp16 feature store |
| `src/training/` | Head trainer (resumable, ablation runner) and spatial EfficientNet fine-tuning |
| `src/models/attribution.py` | Source-attribution head |
| `scripts/` | Command-line entry points (see below) |
| `backend/app/` | FastAPI server: engine, window scan, visual evidence, batch jobs, research data |
| `frontend/` | React + Vite web application |
| `models/` | Packaged models with SHA-256 checksums: `fp-v4.1` (current), `attribution-v1`, and the earlier `fp-v3.0` / `fp-v4.0` |
| `results/` | Ablation tables, training logs, test reports |
| `docs/` | Decision log, experiment protocol, dataset audit |
| `tests/` | Training, app-parity and attribution tests |
| `configs/paths.toml` | All data and output paths, relative to the workspace root |

## Setup

Python 3.12 and Node 18+.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt     # see the note in requirements.txt for CUDA
cd frontend && npm install && npm run build && cd ..
```

## Run the app

Run from the repository root:

```
.venv\Scripts\python -m uvicorn backend.app.main:app --port 8000
```

Then open http://127.0.0.1:8000. The model files in `models/fp-v4.1/` are included in the repository. The CLIP and EfficientNet backbones download on first run.

## Reproduce the results

The data is the CNNDetection / ForenSynths benchmark (Wang et al., CVPR 2020): `progan_train`, `progan_val` and `CNN_synth_testset`. Place it where `configs/paths.toml` expects it.

```
python scripts/build_dataset_manifests.py        # hash and decode every image
python scripts/check_leakage.py                  # cross-split duplicates
python scripts/extract_features.py               # 5 pillars, resumable shards
python scripts/build_feature_store.py            # train-only standardisation, fp16
python scripts/finetune_spatial.py               # fine-tune EfficientNet-B0 (fp-v4.1)
python scripts/build_ft_store.py                 # re-extract the spatial pillar
python scripts/train_heads.py --group p1         # full model; p2/p3 = ablation subsets
python scripts/ablation_table.py                 # 31-subset ablation table
python scripts/package_model.py                  # copy into models/fp-v4.1 with checksums
python scripts/eval_fullscan.py                  # full-image window scan, scored once
python scripts/train_attribution.py              # source attribution head
```

Protocol, from the guide:
- train on ProGAN only;
- batch 64, at most 50 epochs, with early stopping;
- choose every setting on ProGAN validation only;
- fix the threshold at 0.5;
- score the test set once.

Every decision is recorded in `docs/architecture/DECISION_LOG.md` and every change in `CHANGELOG.md`.

`finetune_spatial_v2.py` and `build_store_v42.py` belong to an experimental version (fp-v4.2, decision D-22). It did not improve on fp-v4.1 and is not used by the app.
