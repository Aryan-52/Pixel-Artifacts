# Recommended Architecture — fp-v4.0

**Recommendation:** fp-v4.0 = the fp-v3.0 architecture, unchanged, trained on the full 720,119-image ProGAN set. The only engineering change is a disk-backed feature store so the larger cache fits local hardware.

The research question for v4 is "does 30× more ProGAN data improve cross-generator generalisation?". Changing the architecture at the same time would confound that comparison.

## Pipeline

```
image (PNG from split archive / folder)
  └─ RGB, center-crop 224 (bicubic upscale only if min side < 224; never downscale)
       ├─ [train only] cached aug copy: blur p=0.5 σ~U(0,3) → JPEG p=0.5 q~U(30,100)
       ▼
  frozen extractors (fp-v3 code, feature-code hash must match)
       spatial  EfficientNet-B0 IMAGENET1K_V1, stages 2/3/4/6 mean+std + final   1952
       frequency FFT log-spectrum, 64 radial bins r∈[0,√2] + stats                 408
       color    RGB statistics                                                      133
       physical image-derived physical-consistency cues (NOT camera PRNU)            89
       semantic OpenCLIP ViT-B/32 'openai', L2-normalised                           512
                                                                           total  3094
  ▼
  sharded feature cache (fp16 deep / fp32 handcrafted)        ← unchanged from v3
      stored in Pixel-Artifacts/results/training/feature_cache
  ▼
  NEW: standardise with TRAIN-only mean/std, clip ±10, write one fp16 memmap per split
  ▼
  head (unchanged): per-pillar projector 256 → gated fusion → MLP 512 → 1 logit
  ▼
  BCE (label smoothing 0.02) + gate-entropy 0.01; AdamW; warmup + cosine; AMP
  ▼
  select on clean-val BCE → temperature on clean val (T = max(1, fit)) → threshold 0.5
```

## What changes vs fp-v3.0

| Area | fp-v3.0 | fp-v4.0 |
|---|---|---|
| Training data | 4 categories × 6,000 = 24,000 | 20 categories, 720,119 |
| Validation | 1,600 (4 categories) | 8,000 (`progan_val`, 20 categories) |
| Cache rows (train) | 48,000 | 1,440,238 (clean + 1 aug copy) |
| Feature store | concatenated in RAM, standardised on GPU | standardised fp16 memmap on disk (~8.9 GB), streamed in batches |
| Sample source | directory walk | val/test: directory walk; train: zip members read in place from the split archive (`src/data/split_archive.py`) |
| Paths | hardcoded `/kaggle/working/...` constants | `configs/paths.toml`, relative to the single workspace root |
| Epoch budget / batch | 40 max, batch 256 | Per spec: 50 epochs, batch 64. See open decision D-5. |
| Everything else | — | identical (extractor code hash must equal `369cfa92c5a260ea` under the same Python) |

## Hardware fit (local: RTX 5050 Laptop 8 GB, 16 GB RAM, NVMe)

- Feature extraction: two extractor passes over 720K images. This is the dominant cost. Throughput will be benchmarked on 2,048 images before the full run, and no runtime is promised until then.
- Head training: 1.65M parameters, memmap batches. RAM stays bounded and VRAM use is small.

## Explicitly not in fp-v4.0

Fine-tuned encoders, ViT-L/14 semantic branch, new pillars, DCT, HSV/LAB. These are candidate follow-up experiments, versioned separately.
