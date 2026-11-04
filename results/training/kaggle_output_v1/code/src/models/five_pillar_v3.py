# ================================================================
# FIVE-PILLAR AI-GENERATED IMAGE DETECTOR  -  v3 (pre-training candidate)
# ================================================================
#
# Pillars (all five always extracted + cached; ACTIVE_PILLARS selects
# which ones the trainable fusion model uses, so ablations reuse the cache):
#
#   spatial   : EfficientNet-B0 (frozen, multi-stage mean/std pooling + final)
#   frequency : FFT log-spectrum statistics (Hann-windowed image + residual),
#               radial bins cover the FULL spectrum out to the FFT corner (sqrt 2)
#   color     : RGB / YCbCr / opponent distribution statistics
#   physical  : image-derived physical-consistency cues (illumination field,
#               shading ratio, gradient structure, noise-residual statistics).
#               NOT PRNU / iPRNU - no camera reference data is available.
#   semantic  : OpenCLIP ViT-B/32 (frozen, OpenAI weights)
#
# Pipeline:  feature cache (sharded, resumable)  ->  train-only per-feature
#            standardisation -> per-pillar projectors -> gated fusion ->
#            MLP -> BCE (+ label smoothing, gate-entropy term)
#
# METHODOLOGY NOTES
#  * Validation is IN-DISTRIBUTION (same ProGAN generator, same categories).
#    It is used for early stopping / checkpoint choice / temperature fitting
#    ONLY, and ONLY on the CLEAN validation images. It is NOT a zero-shot result.
#  * Zero-shot / perturbation sets must never influence training, model
#    selection, temperature, or the threshold (fixed at 0.5).
#  * Training augmentation (blur + JPEG on an extra copy of each training image)
#    is ON by default (TRAIN_AUG_COPIES = 1). Report blur/JPEG robustness as
#    "trained with augmentation", not as pure zero-shot. Set 0 to disable.
#  * Gate values are learned mixing weights, not causal evidence for an artifact.
#  * Confidence = temperature-scaled predicted probability (T >= 1, fitted on
#    clean in-distribution validation). It is NOT a demonstrated-calibrated
#    probability on unseen generators.
#
# RUN AS A .py FILE (python five_pillar_detector_v3.py) or as a notebook cell.
# The feature-code hash needs inspect.getsource; if it fails the script stops.
# ================================================================


import os
import sys
import gc
import io
import json
import math
import time
import random
import hashlib
import inspect
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

warnings.filterwarnings("ignore")
# NOTE: ImageFile.LOAD_TRUNCATED_IMAGES is deliberately NOT enabled.
# Truncated files must fail loudly and be dropped, not decode to grey blocks.


def _ensure_package(import_name, pip_name):
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pip_name} (needs Kaggle Internet = ON)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pip_name])
        except Exception as e:
            raise RuntimeError(
                f"Could not install {pip_name}. Turn Internet ON in the Kaggle "
                f"notebook settings, or attach the wheel as a dataset.") from e


_ensure_package("open_clip", "open_clip_torch")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import open_clip
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

# ================================================================
# 1. CONFIGURATION
# ================================================================
CODE_VERSION = "fp-v3.0"
SEED = 42

BASE_DIR = Path("/kaggle/working/cnndetection_project")
DATASET_DIR = BASE_DIR / "datasets" / "ForenSynths_train"
RUN_ROOT = BASE_DIR / "five_pillar_detector_v3"
FEATURE_DIR = RUN_ROOT / "feature_cache"          # shared by every run / ablation
PRETRAINED_DIR = RUN_ROOT / "pretrained_cache"

ALL_PILLARS = ["spatial", "frequency", "color", "physical", "semantic"]


def validate_active_pillars(active):
    """Reject empty / unknown / duplicate pillar names; return canonical pillar order."""
    if not isinstance(active, (list, tuple)) or len(active) == 0:
        raise ValueError("ACTIVE_PILLARS must be a non-empty list.")
    unknown = [p for p in active if p not in ALL_PILLARS]
    if unknown:
        raise ValueError(f"Unknown pillar name(s) {unknown}. Allowed: {ALL_PILLARS}")
    if len(set(active)) != len(active):
        raise ValueError(f"Duplicate pillar name(s) in ACTIVE_PILLARS: {list(active)}")
    return [p for p in ALL_PILLARS if p in active]      # canonical, deterministic order


def make_run_tag(active):
    omitted = [p for p in ALL_PILLARS if p not in active]
    return "full" if not omitted else "without_" + "+".join(omitted)


# For ablations edit this list only, e.g. remove "physical" for "without_physical".
ACTIVE_PILLARS = validate_active_pillars(
    ["spatial", "frequency", "color", "physical", "semantic"])
RUN_TAG = make_run_tag(ACTIVE_PILLARS)

# ---- image handling ----
IMAGE_SIZE = 224          # centre crop (NO downscale: resizing destroys pixel-grid artifacts)
CROP_POLICY = "center_crop; bicubic_upscale_only_if_min_side<IMAGE_SIZE; no_downscale; RGB; uint8"
TRAIN_AUG_COPIES = 1      # extra blur/JPEG-augmented copy per training image (0 = off)
VAL_AUG = False           # MUST stay False for model selection. (True only builds an extra
                          # augmented val copy that is logged for monitoring, never selected on.)
AUG_P_BLUR, AUG_BLUR_SIGMA = 0.5, (0.0, 3.0)
AUG_P_JPEG, AUG_JPEG_Q = 0.5, (30, 100)

# ---- feature extraction ----
BATCH_SIZE_FEATURE = 64   # keep <= ~300: torch.quantile has a 16M-element input limit
NUM_WORKERS = min(4, os.cpu_count() or 2)
SHARD_SIZE = 2048         # cache is written/resumed shard by shard
FREQ_BINS = 64            # radial bins over r in [0, sqrt(2)] (bin width ~2.5 px at 224)
FREQ_R_MAX = math.sqrt(2.0)   # r = 1.0 is Nyquist along the axes; corner of the FFT grid = sqrt(2)
SPATIAL_TAP_STAGES = (2, 3, 4, 6)   # EfficientNet-B0 stages pooled as mean+std
FEATURE_CLIP = 10.0       # standardised features are clipped to +-this
CHECK_DUPLICATES = True
MAX_DROP_WARN, MAX_DROP_ABORT = 0.01, 0.05   # fraction of entries dropped per split

# ---- pretrained identifiers ----
EFFNET_WEIGHTS_NAME = "IMAGENET1K_V1"          # torchvision EfficientNet_B0_Weights.<name>
CLIP_MODEL_NAME, CLIP_PRETRAINED = "ViT-B-32", "openai"

# Hand-counted expectations (only a sanity WARNING if they differ; real dims come from the cache)
EXPECTED_DIMS = {"spatial": 1952, "frequency": 408, "color": 133, "physical": 89, "semantic": 512}

# ---- model ----
PILLAR_DIM = 256
HIDDEN_DIM = 512
DROPOUT = 0.30
HEAD_DROPOUT2 = 0.20
INPUT_DROPOUT = 0.10
PROJ_DROPOUT = 0.15
GATE_DROPOUT = 0.10
PILLAR_DROPOUT = 0.15     # training-time whole-pillar dropout (prevents one pillar dominating)

# ---- training ----
BATCH_SIZE_TRAIN = 256
MAX_EPOCHS = 40
WARMUP_EPOCHS = 2
LEARNING_RATE = 5e-4
MIN_LR_FRAC = 0.02
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.02
GATE_ENTROPY_WEIGHT = 0.01   # small bonus against one-hot gate collapse
EARLY_STOPPING_PATIENCE = 8
MIN_DELTA = 1e-4
MAX_GRAD_NORM = 1.0
USE_AMP = True
CLASSIFICATION_THRESHOLD = 0.5   # FIXED; never tuned on zero-shot data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_DIR = RUN_ROOT / "runs" / RUN_TAG
CKPT_DIR = RUN_DIR / "checkpoints"
RESULT_DIR = RUN_DIR / "results"

LATEST_CKPT = CKPT_DIR / "latest_checkpoint.pth"
PREV_CKPT = CKPT_DIR / "latest_checkpoint.prev.pth"
BEST_CKPT = CKPT_DIR / "best_model.pth"
HISTORY_FILE = RESULT_DIR / "training_history.csv"

EPS = 1e-6

STORE_DTYPE = {
    "spatial": torch.float16, "semantic": torch.float16,      # deep features: fp16 is fine
    "frequency": torch.float32, "color": torch.float32,       # handcrafted: keep precision
    "physical": torch.float32,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_cfg():
    return {"pillar_dim": PILLAR_DIM, "hidden_dim": HIDDEN_DIM, "head_dropout": DROPOUT,
            "head_dropout2": HEAD_DROPOUT2, "input_dropout": INPUT_DROPOUT,
            "proj_dropout": PROJ_DROPOUT, "gate_dropout": GATE_DROPOUT,
            "pillar_dropout": PILLAR_DROPOUT}


def library_versions():
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "open_clip": getattr(open_clip, "__version__", "unknown"),
            "numpy": np.__version__, "pandas": pd.__version__}


def pretrained_spec():
    return {
        "spatial": {"library": "torchvision", "model": "efficientnet_b0",
                    "weights": f"EfficientNet_B0_Weights.{EFFNET_WEIGHTS_NAME}",
                    "frozen": True, "tapped_stages": list(SPATIAL_TAP_STAGES),
                    "final_descriptor": "avgpool 1280-d"},
        "semantic": {"library": "open_clip", "model": CLIP_MODEL_NAME,
                     "pretrained": CLIP_PRETRAINED, "frozen": True,
                     "output": "encode_image, L2-normalised 512-d"},
    }


def preprocessing_spec():
    return {
        "decode": "PIL open -> load() -> convert('RGB'); unreadable/truncated files are dropped",
        "crop": f"centre crop {IMAGE_SIZE}x{IMAGE_SIZE}; only if min(side) < {IMAGE_SIZE} the image "
                f"is first bicubic-UPSCALED so min side == {IMAGE_SIZE}. No downscaling. Use "
                f"center_crop_or_upscale() for EVERY evaluation set.",
        "image_size": IMAGE_SIZE, "crop_policy": CROP_POLICY,
        "tensor": "uint8 HWC -> CHW; /255 -> float RGB in [0,1] on the GPU",
        "handcrafted_input": "float RGB [0,1], no mean/std normalisation",
        "spatial_norm": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "semantic_norm": {"mean": CLIP_MEAN, "std": CLIP_STD},
        "frequency": {"bins": FREQ_BINS, "r_max": FREQ_R_MAX, "r_unit": "1.0 = axis Nyquist",
                      "window": "hann (periodic=False), separable outer product",
                      "spectra": ["gray (mean removed)", "gray minus 3x3 local mean"],
                      "fft_dims": "(-2,-1) only"},
        "spatial_tap_stages": list(SPATIAL_TAP_STAGES),
        "store_dtypes": {k: str(v) for k, v in STORE_DTYPE.items()},
        "standardisation": "(x - mean) / std per feature, stats from TRAIN split only "
                           "(bundle['stats']), then clip to +-feature_clip",
        "feature_clip": FEATURE_CLIP,
        "train_aug": {"copies": TRAIN_AUG_COPIES, "p_blur": AUG_P_BLUR, "blur_sigma": AUG_BLUR_SIGMA,
                      "p_jpeg": AUG_P_JPEG, "jpeg_quality": AUG_JPEG_Q},
    }


def build_config():
    return {
        "code_version": CODE_VERSION, "seed": SEED, "run_tag": RUN_TAG,
        "all_pillars": ALL_PILLARS, "active_pillars": ACTIVE_PILLARS,
        "dataset_dir": str(DATASET_DIR),
        "image_size": IMAGE_SIZE, "train_aug_copies": TRAIN_AUG_COPIES, "val_aug": VAL_AUG,
        "feature_extraction": {"batch_size": BATCH_SIZE_FEATURE, "shard_size": SHARD_SIZE,
                               "freq_bins": FREQ_BINS, "freq_r_max": FREQ_R_MAX,
                               "spatial_tap_stages": list(SPATIAL_TAP_STAGES),
                               "feature_clip": FEATURE_CLIP},
        "model": model_cfg(),
        "training": {"batch_size": BATCH_SIZE_TRAIN, "max_epochs": MAX_EPOCHS,
                     "warmup_epochs": WARMUP_EPOCHS, "lr": LEARNING_RATE,
                     "min_lr_frac": MIN_LR_FRAC, "weight_decay": WEIGHT_DECAY,
                     "label_smoothing": LABEL_SMOOTHING,
                     "gate_entropy_weight": GATE_ENTROPY_WEIGHT,
                     "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                     "min_delta": MIN_DELTA, "max_grad_norm": MAX_GRAD_NORM,
                     "use_amp": USE_AMP, "selection_metric": "clean_val_bce_loss"},
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "temperature_rule": "fit on clean validation logits only; final T = max(1.0, fitted T)",
    }


# ================================================================
# 2. ATOMIC IO
# ================================================================
def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(text, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_latest_checkpoint(state):
    """Write new latest; keep the previous one as a fallback."""
    tmp = LATEST_CKPT.with_name(LATEST_CKPT.name + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    if LATEST_CKPT.exists():
        os.replace(LATEST_CKPT, PREV_CKPT)
    os.replace(tmp, LATEST_CKPT)


# ================================================================
# 3. DATA DISCOVERY, FORMAT AUDIT, LEAKAGE CHECK
# ================================================================
def category_of(path):
    parts = Path(path).parts
    for i, s in enumerate(parts):
        if s in ("0_real", "1_fake") and i > 0:
            return parts[i - 1]
    return "unknown"


def collect_samples(split_dir):
    samples = []
    for cat_dir in sorted(p for p in Path(split_dir).iterdir() if p.is_dir()):
        for sub, label in (("0_real", 0), ("1_fake", 1)):
            d = cat_dir / sub
            if d.exists():
                files = sorted(p for p in d.rglob("*")
                               if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
                samples += [(str(p), label) for p in files]
    return samples


def _dom(d):
    return max(d, key=d.get) if d else None


def audit_dataset(samples, name, n=300):
    """Real-vs-fake FORMAT SHORTCUT audit. Prints everything needed to inspect it by eye and
    saves dataset_audit_<name>.json. Warning only: it never stops the run."""
    rng = random.Random(SEED)
    by = {0: [], 1: []}
    cat_ext = defaultdict(Counter)
    for p, l in samples:
        by[l].append(p)
        cat_ext[(category_of(p), l)][Path(p).suffix.lower()] += 1

    banner(f"[{name.upper()}] REAL-vs-FAKE FILE-FORMAT AUDIT (shortcut check)")
    stats = {}
    for l, lname in ((0, "REAL"), (1, "FAKE")):
        paths = by[l]
        exts = Counter(Path(p).suffix.lower() for p in paths)
        sub = rng.sample(paths, min(n, len(paths)))
        fmts, sizes, bpp, qt, unread = Counter(), Counter(), [], [], 0
        for p in sub:
            try:
                with Image.open(p) as im:
                    fmts[im.format] += 1
                    w, h = im.size
                    sizes[(w, h)] += 1
                    q = getattr(im, "quantization", None)
                    if q:
                        qt.append(float(np.mean(list(q[0]))))
                bpp.append(os.path.getsize(p) / float(w * h))
            except Exception:
                unread += 1
        tot = max(len(sub) - unread, 1)
        st = {
            "n_files": len(paths),
            "extensions_all_files": dict(exts),
            "sample_size": len(sub), "unreadable_in_sample": unread,
            "decoded_format_pct": {str(k): round(100.0 * v / tot, 1) for k, v in fmts.items()},
            "top_resolutions_pct": {f"{w}x{h}": round(100.0 * v / tot, 1)
                                    for (w, h), v in sizes.most_common(3)},
            "pct_min_side_below_crop": round(
                100.0 * sum(v for (w, h), v in sizes.items() if min(w, h) < IMAGE_SIZE) / tot, 1),
            "bytes_per_pixel_mean": float(np.mean(bpp)) if bpp else None,
            "bytes_per_pixel_median": float(np.median(bpp)) if bpp else None,
            "jpeg_qtable0_mean_median": float(np.median(qt)) if qt else None,
            "n_jpeg_with_qtable_in_sample": len(qt),
        }
        stats[l] = st
        print(f"\n  {lname}: {st['n_files']:,} files")
        print(f"    extensions (all files)  : {st['extensions_all_files']}")
        print(f"    decoded format (sample) : {st['decoded_format_pct']}  "
              f"[n={len(sub)}, unreadable={unread}]")
        print(f"    top resolutions         : {st['top_resolutions_pct']}")
        print(f"    % with min side < {IMAGE_SIZE}  : {st['pct_min_side_below_crop']}  "
              f"(these get bicubic-upscaled before cropping)")
        if bpp:
            print(f"    bytes/pixel             : mean {st['bytes_per_pixel_mean']:.3f} | "
                  f"median {st['bytes_per_pixel_median']:.3f}")
        print(f"    JPEG quant-table mean   : "
              f"{st['jpeg_qtable0_mean_median'] if qt else 'n/a (no JPEG in sample)'}"
              f"  (lower = higher JPEG quality)")

    print("\n  Per-category file extensions (all files):")
    for c in sorted({c for c, _ in cat_ext}):
        print(f"    {c:10s} real={dict(cat_ext.get((c, 0), {}))} | fake={dict(cat_ext.get((c, 1), {}))}")

    r, f_ = stats[0], stats[1]
    flags = []
    if set(r["extensions_all_files"]) != set(f_["extensions_all_files"]):
        flags.append(f"file extensions differ: real={sorted(r['extensions_all_files'])} "
                     f"vs fake={sorted(f_['extensions_all_files'])}")
    if _dom(r["decoded_format_pct"]) != _dom(f_["decoded_format_pct"]):
        flags.append(f"dominant decoded format differs: real={_dom(r['decoded_format_pct'])} "
                     f"vs fake={_dom(f_['decoded_format_pct'])}")
    if _dom(r["top_resolutions_pct"]) != _dom(f_["top_resolutions_pct"]):
        flags.append(f"dominant resolution differs: real={_dom(r['top_resolutions_pct'])} "
                     f"vs fake={_dom(f_['top_resolutions_pct'])}")
    if r["bytes_per_pixel_mean"] and f_["bytes_per_pixel_mean"]:
        ratio = max(r["bytes_per_pixel_mean"], f_["bytes_per_pixel_mean"]) / \
            max(min(r["bytes_per_pixel_mean"], f_["bytes_per_pixel_mean"]), 1e-9)
        if ratio > 1.5:
            flags.append(f"bytes/pixel differ by {ratio:.1f}x (different compression)")
    if (r["jpeg_qtable0_mean_median"] is None) != (f_["jpeg_qtable0_mean_median"] is None):
        flags.append("JPEG quantisation tables found in only ONE class")
    if abs(r["pct_min_side_below_crop"] - f_["pct_min_side_below_crop"]) > 5:
        flags.append("the two classes are upscaled at different rates")

    print("\n  " + "-" * 76)
    if flags:
        print(f"  !! FORMAT DIFFERENCES DETECTED between REAL and FAKE ({name}):")
        for x in flags:
            print(f"     - {x}")
        print("  The model may learn FORMAT/compression instead of generation artifacts.\n"
              "  Inspect the numbers above BEFORE interpreting validation performance. The\n"
              "  blur/JPEG augmentation does not remove this (clean copies are kept). If the\n"
              "  difference is real, document it as a dataset limitation.")
    else:
        print("  No obvious real-vs-fake format difference in this sample "
              "(absence in a sample is not proof of absence).")
    print("  " + "-" * 76)
    atomic_write_text(json.dumps({"split": name, "flags": flags,
                                  "real": stats[0], "fake": stats[1]}, indent=2, default=str),
                      RUN_ROOT / f"dataset_audit_{name}.json")
    return {"flags": flags, "stats": stats}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_cross_split_duplicates(train_samples, val_samples):
    key = hashlib.md5(("|".join(p for p, _ in train_samples) + "##" +
                       "|".join(p for p, _ in val_samples)).encode()).hexdigest()[:12]
    cache_file = RUN_ROOT / f"duplicate_check_{key}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            dup_paths = set(json.load(f))
    else:
        print("\nChecking for byte-identical images shared by train and val (MD5)...")
        with ThreadPoolExecutor(8) as ex:
            train_hashes = set(ex.map(_md5, [p for p, _ in train_samples]))
            val_hashes = list(ex.map(_md5, [p for p, _ in val_samples]))
        dup_paths = {p for (p, _), h in zip(val_samples, val_hashes) if h in train_hashes}
        atomic_write_text(json.dumps(sorted(dup_paths)), cache_file)
    print(f"Validation images identical to a training image: {len(dup_paths)}")
    if dup_paths:
        print("  -> removed from validation to prevent leakage.")
    return [(p, l) for p, l in val_samples if p not in dup_paths]


# ================================================================
# 4. IMAGE LOADING (centre crop, no downscale) AND AUGMENTATION
# ================================================================
def apply_aug(img, rng):
    if rng.random() < AUG_P_BLUR:
        sigma = rng.uniform(*AUG_BLUR_SIGMA)
        if sigma > 0.05:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    if rng.random() < AUG_P_JPEG:
        q = rng.randint(*AUG_JPEG_Q)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


def center_crop_or_upscale(img, size):
    w, h = img.size
    if min(w, h) < size:     # rare; only upscale when the image is smaller than the crop
        s = size / float(min(w, h))
        img = img.resize((max(size, math.ceil(w * s)), max(size, math.ceil(h * s))), _BICUBIC)
        w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


class ImageEntryDataset(Dataset):
    """entries: list of (path, label, copy_index). copy 0 = clean, >0 = augmented."""

    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        path, label, copy = self.entries[i]
        try:
            img = Image.open(path)
            img.load()
            img = img.convert("RGB")
            if copy > 0:
                img = apply_aug(img, random.Random(f"{SEED}|{path}|{copy}"))
            img = center_crop_or_upscale(img, IMAGE_SIZE)
            arr = np.asarray(img, dtype=np.uint8)
            if arr.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                raise ValueError("bad shape")
            t = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
            return t, float(label), 1, i
        except Exception:
            # Flagged invalid -> dropped from the cache (never trained on as a fake "black" sample).
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8), float(label), 0, i


# ================================================================
# 5. HANDCRAFTED PILLARS  (input: float RGB in [0,1], shape [B,3,H,W])
# ================================================================
def _cat_features(name, feats, B):
    """Shape-normalise every feature to [B,k] and concatenate.
    NON-FINITE VALUES ARE NOT REPLACED HERE: extract_batch counts them, persists the counts
    per shard, and marks the affected rows invalid (they are dropped and listed)."""
    cols = []
    for f in feats:
        f = f.float()
        if f.dim() == 1:
            f = f.unsqueeze(1)
        elif f.dim() > 2:
            f = f.flatten(1)
        assert f.shape[0] == B, f"{name}: feature with wrong batch dim {tuple(f.shape)}"
        cols.append(f)
    return torch.cat(cols, dim=1)


def _skew_kurt(v, dims):
    c = v - v.mean(dim=dims, keepdim=True)
    s = c.pow(2).mean(dim=dims, keepdim=True).sqrt() + EPS
    z = c / s
    return z.pow(3).mean(dim=dims), z.pow(4).mean(dim=dims) - 3.0


def _pearson(a, b):
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    return (a * b).mean(1) / (a.pow(2).mean(1).sqrt() * b.pow(2).mean(1).sqrt() + EPS)


# ---------------- FREQUENCY ----------------
_FREQ_TABLES = {}


def _freq_tables(H, W, nbins, device):
    key = (H, W, nbins, str(device))
    if key in _FREQ_TABLES:
        return _FREQ_TABLES[key]
    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                            indexing="ij")
    # torch.fft.fftshift places DC at index n//2 (NOT (n-1)/2).
    dy = (yy - H // 2).float()
    dx = (xx - W // 2).float()
    r = torch.sqrt(dy * dy + dx * dx) / (min(H, W) / 2.0)   # 1.0 == Nyquist along the axes
    assert float(r.max()) <= FREQ_R_MAX + 1e-4, "radius exceeds FREQ_R_MAX (non-square input?)"
    bins = torch.clamp((r / FREQ_R_MAX * nbins).long(), 0, nbins - 1).flatten()   # FULL spectrum
    onehot = F.one_hot(bins, nbins).float()                 # [H*W, nbins]
    counts = onehot.sum(0).clamp_min(1.0)
    region = torch.stack([
        r < 0.2,                          # low
        (r >= 0.2) & (r < 0.5),           # mid
        (r >= 0.5) & (r < 1.0),           # high (up to axis Nyquist)
        r >= 1.0,                         # corner: beyond axis Nyquist, up to sqrt(2)
    ]).flatten(1).float()
    ring = r > 0.05
    dirs = torch.stack([
        (dy.abs() <= 2) & (dx.abs() > 4) & ring,          # horizontal axis
        (dx.abs() <= 2) & (dy.abs() > 4) & ring,          # vertical axis
        ((dy - dx).abs() <= 2) & ring,                    # main diagonal (now reaches the corners)
        ((dy + dx).abs() <= 2) & ring,                    # anti diagonal
    ]).flatten(1).float()
    win = torch.outer(torch.hann_window(H, periodic=False, device=device),
                      torch.hann_window(W, periodic=False, device=device))
    t = dict(bins=bins, onehot=onehot, counts=counts,
             region=region, region_counts=region.sum(1).clamp_min(1.0),
             dirs=dirs, dir_counts=dirs.sum(1).clamp_min(1.0), win=win)
    _FREQ_TABLES[key] = t
    return t


def _log_spectrum(img2d, win):
    # dim=(-2,-1) is REQUIRED: fftshift without dim also rotates the batch axis.
    spec = torch.fft.fftshift(torch.fft.fft2(img2d * win), dim=(-2, -1))
    return torch.log1p(spec.abs()).flatten(1)


def extract_frequency_features(x):
    """Per spectrum (gray, residual): mean/std/peak per radial bin (3*FREQ_BINS) + 10 summary
    values; plus 4 directional values (gray only). Dim = 2*(3*FREQ_BINS+10)+4 = 408 for 64 bins."""
    B, _, H, W = x.shape
    t = _freq_tables(H, W, FREQ_BINS, x.device)
    gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    gray = gray - gray.mean((1, 2), keepdim=True)
    resid = gray - F.avg_pool2d(gray.unsqueeze(1), 3, 1, 1, count_include_pad=False).squeeze(1)
    idx = t["bins"].unsqueeze(0).expand(B, -1).contiguous()
    feats = []
    for k, img in enumerate((gray, resid)):
        mag = _log_spectrum(img, t["win"])                      # [B, H*W]
        mean_r = (mag @ t["onehot"]) / t["counts"]
        sq = ((mag * mag) @ t["onehot"]) / t["counts"]
        std_r = (sq - mean_r * mean_r).clamp_min(0).sqrt()
        mx = torch.full((B, FREQ_BINS), float("-inf"), device=x.device, dtype=mag.dtype)
        mx = mx.scatter_reduce(1, idx, mag, reduce="amax", include_self=True)
        mx = torch.where(torch.isfinite(mx), mx, mean_r)
        peak = mx - mean_r                                      # keeps periodic-peak fingerprints
        reg = (mag @ t["region"].T) / t["region_counts"]        # [B,4]
        low, mid, high, corner = reg[:, 0], reg[:, 1], reg[:, 2], reg[:, 3]
        summ = torch.stack([low, mid, high, corner,
                            high / (low + EPS), high / (mid + EPS), corner / (high + EPS),
                            mag.mean(1), mag.std(1), mag.amax(1)], dim=1)
        feats += [mean_r, std_r, peak, summ]
        if k == 0:
            d = (mag @ t["dirs"].T) / t["dir_counts"]
            feats.append(d / (mag.mean(1, keepdim=True) + EPS))
    return _cat_features("frequency", feats, B)


# ---------------- COLOR ----------------
def extract_color_features(x):
    B, C, H, W = x.shape
    flat = x.flatten(2)                                         # [B,3,P]
    P = flat.shape[-1]
    feats = [flat.mean(2), flat.std(2), flat.amin(2), flat.amax(2)]
    sk, ku = _skew_kurt(flat, (2,))
    feats += [sk, ku]
    srt = torch.sort(flat, dim=2).values                        # one sort for all quantiles
    for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99):
        feats.append(srt[:, :, int(round(q * (P - 1)))])
    bins = (flat * 16).long().clamp(0, 15)
    hist = torch.zeros(B, 3, 16, device=x.device).scatter_add_(2, bins, torch.ones_like(flat)) / P
    feats.append(hist.flatten(1))
    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    feats.append(torch.stack([_pearson(r, g), _pearson(r, b), _pearson(g, b)], dim=1))
    maxv, minv = x.amax(1).flatten(1), x.amin(1).flatten(1)
    sat = (maxv - minv) / (maxv + EPS)
    feats += [sat.mean(1), sat.std(1), maxv.mean(1), maxv.std(1)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (b - y) * 0.564
    cr = 0.5 + (r - y) * 0.713
    feats += [cb.mean(1), cb.std(1), cr.mean(1), cr.std(1), _pearson(cb, cr)]
    rg, yb = r - g, 0.5 * (r + g) - b
    feats += [rg.mean(1), rg.std(1), yb.mean(1), yb.std(1)]
    feats += [(flat <= 0.0).float().mean(2), (flat >= 1.0).float().mean(2)]    # clipping fractions
    h2, w2 = H // 2, W // 2
    for q in (x[:, :, :h2, :w2], x[:, :, :h2, w2:], x[:, :, h2:, :w2], x[:, :, h2:, w2:]):
        feats += [q.mean((2, 3)), q.std((2, 3))]
    return _cat_features("color", feats, B)


# ---------------- PHYSICAL-CONSISTENCY CUES ----------------
def _sobel(t):
    tp = F.pad(t, (1, 1, 1, 1), mode="replicate")
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=t.device, dtype=t.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3).contiguous()
    return F.conv2d(tp, kx), F.conv2d(tp, ky)


def _highpass(t):
    return t - F.avg_pool2d(t, 3, 1, 1, count_include_pad=False)


def extract_physical_features(x):
    B, _, H, W = x.shape
    L = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    illum = F.avg_pool2d(L, 31, 1, 15, count_include_pad=False)
    local_mean = F.avg_pool2d(L, 15, 1, 7, count_include_pad=False)
    feats = []

    # A. illumination field and its dominant direction
    il = illum.flatten(1)
    feats += [il.mean(1), il.std(1), il.amin(1), il.amax(1), il.amax(1) - il.amin(1)]
    gxi, gyi = _sobel(illum)
    mag_i = torch.sqrt(gxi * gxi + gyi * gyi + 1e-12)
    mgx, mgy = gxi.mean((1, 2, 3)), gyi.mean((1, 2, 3))
    feats += [mgx, mgy, mag_i.mean((1, 2, 3)),
              torch.sqrt(mgx * mgx + mgy * mgy + 1e-12) / (mag_i.mean((1, 2, 3)) + EPS)]

    # B. gradient / edge structure
    gx, gy = _sobel(L)
    gm = torch.sqrt(gx * gx + gy * gy + 1e-12)
    gmf = gm.flatten(1)
    mean_g = gmf.mean(1, keepdim=True)
    feats += [gmf.mean(1), gmf.std(1), torch.quantile(gmf, 0.95, dim=1),
              (gmf > mean_g).float().mean(1), (gmf > 2 * mean_g).float().mean(1)]
    ang = torch.remainder(torch.atan2(gy, gx), math.pi)
    ob = (ang / math.pi * 8).long().clamp(0, 7).flatten(1)
    oh = torch.zeros(B, 8, device=x.device).scatter_add_(1, ob, gmf)
    feats.append(oh / (oh.sum(1, keepdim=True) + EPS))
    jxx, jyy, jxy = (gx * gx).mean((1, 2, 3)), (gy * gy).mean((1, 2, 3)), (gx * gy).mean((1, 2, 3))
    feats.append(torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2 + 1e-12) / (jxx + jyy + EPS))

    # C. local contrast
    lv = (F.avg_pool2d(L * L, 15, 1, 7, count_include_pad=False) - local_mean ** 2).clamp_min(0)
    ls = torch.sqrt(lv + 1e-8).flatten(1)
    feats += [ls.mean(1), ls.std(1)]

    # D. shading ratio (log luminance minus log illumination; a reflectance-LIKE proxy only)
    sr = (torch.log(L + 1e-3) - torch.log(illum + 1e-3)).flatten(1)
    sk, _ = _skew_kurt(sr, (1,))
    feats += [sr.mean(1), sr.std(1), sk]

    # E. shadow proxy: dark-region fraction and edge softness inside vs outside dark areas
    Lf = L.flatten(1)
    dark = (Lf < 0.6 * Lf.mean(1, keepdim=True)).float()
    gd = (gmf * dark).sum(1) / (dark.sum(1) + 1.0)
    gb = (gmf * (1 - dark)).sum(1) / ((1 - dark).sum(1) + 1.0)
    feats += [dark.mean(1), gd, gb, gd / (gb + EPS)]

    # F. noise-residual statistics (image-derived; NOT PRNU)
    r = _highpass(L)[:, :, 4:-4, 4:-4]
    h, w = r.shape[-2:]
    rf = r.flatten(1)
    sk, ku = _skew_kurt(rf, (1,))
    feats += [rf.std(1), rf.abs().mean(1), sk, ku]
    var = rf.var(1) + 1e-10
    for dy, dx in ((0, 1), (1, 0), (0, 2), (2, 0), (1, 1)):
        a, b = r[:, :, dy:, dx:], r[:, :, :h - dy, :w - dx]
        feats.append((a * b).mean((1, 2, 3)) / var)

    def wstd(mask):
        n = mask.sum(1) + 1.0
        m = (rf * mask).sum(1) / n
        return torch.sqrt((((rf - m.unsqueeze(1)) ** 2) * mask).sum(1) / n + 1e-12)

    lm = local_mean[:, :, 4:-4, 4:-4].flatten(1)
    qt = torch.tensor([1.0 / 3, 2.0 / 3], device=x.device, dtype=lm.dtype)
    q = torch.quantile(lm, qt, dim=1)                                        # [2,B]
    lo = (lm <= q[0].unsqueeze(1)).float()
    hi = (lm >= q[1].unsqueeze(1)).float()
    mid = ((lm > q[0].unsqueeze(1)) & (lm < q[1].unsqueeze(1))).float()
    s_lo, s_mid, s_hi = wstd(lo), wstd(mid), wstd(hi)                        # signal-dependent noise
    feats += [s_lo, s_mid, s_hi, s_hi / (s_lo + EPS)]
    gmc = gm[:, :, 4:-4, 4:-4]
    gc_ = gmc.flatten(1)
    thr = torch.quantile(gc_, 0.10, dim=1, keepdim=True)
    feats.append(wstd((gc_ <= thr).float()))                                 # noise floor in flat areas
    rc = _highpass(x)[:, :, 4:-4, 4:-4].flatten(2)                           # [B,3,P]
    feats += [rc.std(2), _pearson(rc[:, 0], rc[:, 1]), _pearson(rc[:, 0], rc[:, 2]),
              _pearson(rc[:, 1], rc[:, 2])]

    # G. spatial consistency of noise and edges over a 4x4 grid (no raw block brightness ->
    #    avoids encoding scene layout)
    bh, bw = h // 4, w // 4
    rb, gbk = [], []
    for i in range(4):
        for j in range(4):
            ys, xs = slice(i * bh, (i + 1) * bh), slice(j * bw, (j + 1) * bw)
            rb.append(r[:, :, ys, xs].std((1, 2, 3)))
            gbk.append(gmc[:, :, ys, xs].mean((1, 2, 3)))
    rb, gbk = torch.stack(rb, 1), torch.stack(gbk, 1)
    feats += [rb, gbk, rb.std(1), rb.std(1) / (rb.mean(1) + EPS), rb.amax(1) / (rb.amin(1) + EPS)]

    feats += [(L[:, :, :, :W // 2].mean((1, 2, 3)) - L[:, :, :, W // 2:].mean((1, 2, 3))).abs(),
              (L[:, :, :H // 2, :].mean((1, 2, 3)) - L[:, :, H // 2:, :].mean((1, 2, 3))).abs()]
    return _cat_features("physical", feats, B)


# ---------------- SPATIAL (EfficientNet-B0) ----------------
def extract_spatial_features(effnet, x_norm):
    h, outs = x_norm, []
    for i, layer in enumerate(effnet.features):
        h = layer(h)
        if i in SPATIAL_TAP_STAGES:
            hf = h.float()
            outs += [hf.mean((2, 3)), hf.std((2, 3))]      # texture/edge statistics of mid-level maps
    outs.append(effnet.avgpool(h).flatten(1).float())      # standard 1280-d global descriptor
    return torch.cat(outs, dim=1)


# ================================================================
# 6. ENCODERS AND BATCH EXTRACTION
# ================================================================
class Encoders:
    def __init__(self):
        from torchvision import models
        PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(PRETRAINED_DIR / "torch_hub"))
        try:
            print(f"Loading EfficientNet-B0 (torchvision {EFFNET_WEIGHTS_NAME})...")
            self.effnet = models.efficientnet_b0(
                weights=getattr(models.EfficientNet_B0_Weights, EFFNET_WEIGHTS_NAME))
            print(f"Loading OpenCLIP {CLIP_MODEL_NAME} ({CLIP_PRETRAINED} weights)...")
            self.clip, _, _ = open_clip.create_model_and_transforms(
                CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED,
                cache_dir=str(PRETRAINED_DIR / "open_clip"))
        except Exception as e:
            raise RuntimeError(
                "Could not load pretrained weights. Turn Internet ON in Kaggle settings "
                "(first run only; weights are then cached under " + str(PRETRAINED_DIR) + "). "
                "Original error: " + repr(e)) from e
        for m in (self.effnet, self.clip):
            m.to(DEVICE).eval()
            for p in m.parameters():
                p.requires_grad_(False)
        v = lambda t: torch.tensor(t, device=DEVICE).view(1, 3, 1, 1)
        self.im_mean, self.im_std = v(IMAGENET_MEAN), v(IMAGENET_STD)
        self.cl_mean, self.cl_std = v(CLIP_MEAN), v(CLIP_STD)


_ENC = None


def get_encoders():
    global _ENC
    if _ENC is None:
        _ENC = Encoders()
        smoke_test_extractors(_ENC)
    return _ENC


def release_encoders():
    global _ENC
    _ENC = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def extract_batch(x_uint8, enc):
    """uint8 [B,3,H,W] -> dict of RAW (un-standardised) pillar features, float32."""
    x = x_uint8.to(DEVICE, non_blocking=True).float() / 255.0
    out = {
        "frequency": extract_frequency_features(x),       # fp32, outside autocast (cuFFT + precision)
        "color": extract_color_features(x),
        "physical": extract_physical_features(x),
    }
    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE.type == "cuda")):
        out["spatial"] = extract_spatial_features(enc.effnet, (x - enc.im_mean) / enc.im_std)
        sem = enc.clip.encode_image((x - enc.cl_mean) / enc.cl_std)
    out["semantic"] = F.normalize(sem.float(), dim=-1)
    for k, v in out.items():
        assert v.ndim == 2 and v.shape[0] == x.shape[0], f"{k}: bad feature shape {tuple(v.shape)}"
    return out


@torch.inference_mode()
def batch_consistency_check():
    """Row i of a batch must equal the same image processed alone.
    (This is exactly the check that catches batch-axis bugs such as a bare fftshift.)"""
    g = torch.Generator(device="cpu").manual_seed(0)
    imgs = torch.rand(6, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    ramp = torch.linspace(0, 1, IMAGE_SIZE).view(1, 1, 1, -1)
    imgs[3:] = (0.6 * ramp + 0.4 * imgs[3:] * 0.05 + 0.2).clamp(0, 1)
    x = ((imgs * 255).round() / 255).to(DEVICE)
    for fn in (extract_frequency_features, extract_color_features, extract_physical_features):
        whole = fn(x)
        single = torch.cat([fn(x[i:i + 1]) for i in range(x.shape[0])])
        if not torch.allclose(whole, single, rtol=1e-3, atol=1e-3):
            worst = (whole - single).abs().max().item()
            raise RuntimeError(f"{fn.__name__} is not batch-independent (max diff {worst:.4g}). "
                               f"Features would be misaligned with labels.")
    print("Batch-consistency check passed (handcrafted features are per-image).")


@torch.inference_mode()
def smoke_test_extractors(enc):
    batch_consistency_check()
    x = (torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE) * 255).to(torch.uint8)
    feats = extract_batch(x, enc)
    dims = {k: int(v.shape[1]) for k, v in feats.items()}
    print("Feature dimensions:", dims)
    for k, v in feats.items():
        if not torch.isfinite(v).all():
            raise RuntimeError(f"Non-finite values in {k} features on the smoke test.")
    diff = {k: (dims[k], EXPECTED_DIMS[k]) for k in dims if dims[k] != EXPECTED_DIMS[k]}
    if diff:
        print("WARNING: feature dims differ from the hand-counted expectations "
              f"(actual, expected): {diff}. Not fatal; the ACTUAL dims are what is cached.")


# ================================================================
# 7. SHARDED, RESUMABLE FEATURE CACHE
# ================================================================
def _pixel_preprocessing_objects():
    """EVERY function/class that directly touches pixels before feature extraction.
    Listed explicitly so a preprocessing change never relies on another object's source
    happening to capture it."""
    return [center_crop_or_upscale,      # crop / upscale geometry
            apply_aug,                   # blur + JPEG augmentation
            ImageEntryDataset]           # decode, RGB convert, crop call, uint8 tensor layout


def _pixel_preprocessing_constants():
    return {"IMAGE_SIZE": IMAGE_SIZE, "CROP_POLICY": CROP_POLICY,
            "INTERPOLATION": str(_BICUBIC), "SEED": SEED,
            "AUG": [AUG_P_BLUR, AUG_BLUR_SIGMA, AUG_P_JPEG, AUG_JPEG_Q]}


def _feature_code_hash():
    """Hash of EVERY function / class / constant that can change a cached feature value.
    Raises if any source cannot be read (no silent fallback).
    Two explicit groups: (A) pixel preprocessing, (B) feature extraction + encoders."""
    group_a = _pixel_preprocessing_objects()
    group_b = [extract_frequency_features, _freq_tables, _log_spectrum,
               extract_color_features, extract_physical_features, _sobel, _highpass,
               _cat_features, _skew_kurt, _pearson,
               extract_spatial_features, extract_batch, Encoders]
    parts = [CODE_VERSION]
    for o in group_a + group_b:
        try:
            parts.append(inspect.getsource(o))
        except (OSError, TypeError) as e:
            raise RuntimeError(
                f"Cannot read the source of {getattr(o, '__name__', o)} for the feature-code hash "
                f"({e!r}). Run the script as a .py file or from a notebook cell; the cache "
                f"signature cannot be trusted without it.") from e
    consts = dict(_pixel_preprocessing_constants())
    consts.update({
        "CODE_VERSION": CODE_VERSION,
        "FREQ_BINS": FREQ_BINS, "FREQ_R_MAX": FREQ_R_MAX,
        "SPATIAL_TAP_STAGES": list(SPATIAL_TAP_STAGES), "EPS": EPS,
        "IMAGENET_MEAN": IMAGENET_MEAN, "IMAGENET_STD": IMAGENET_STD,
        "CLIP_MEAN": CLIP_MEAN, "CLIP_STD": CLIP_STD,
        "STORE_DTYPE": {k: str(v) for k, v in STORE_DTYPE.items()},
        "EFFNET_WEIGHTS_NAME": EFFNET_WEIGHTS_NAME,
        "CLIP_MODEL_NAME": CLIP_MODEL_NAME, "CLIP_PRETRAINED": CLIP_PRETRAINED,
    })
    parts.append(json.dumps(consts, sort_keys=True, default=str))
    return hashlib.md5("\n#####\n".join(parts).encode()).hexdigest()[:16]


def split_signature(name, samples, n_copies):
    h = hashlib.md5()
    h.update(json.dumps({"name": name, "copies": n_copies, "code": _feature_code_hash()},
                        sort_keys=True).encode())
    for p, l in samples:
        h.update(f"{p}|{l}\n".encode())
    return h.hexdigest()[:12]


def _extract_shard(entries, start, label):
    enc = get_encoders()
    kwargs = dict(batch_size=BATCH_SIZE_FEATURE, shuffle=False, num_workers=NUM_WORKERS,
                  pin_memory=(DEVICE.type == "cuda"))
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 4
    loader = DataLoader(ImageEntryDataset(entries), **kwargs)
    acc = {k: [] for k in ALL_PILLARS}
    labels, valid = [], []
    n_decode_fail, n_nonfinite_any = 0, 0
    nf_rows = {k: 0 for k in ALL_PILLARS}
    nf_vals = {k: 0 for k in ALL_PILLARS}
    t0, nb = time.time(), len(loader)
    for step, (imgs, lab, ok, _) in enumerate(loader, 1):
        feats = extract_batch(imgs, enc)
        ok_dev = ok.to(DEVICE).bool()
        finite_rows = torch.ones(imgs.shape[0], dtype=torch.bool, device=DEVICE)
        for k in ALL_PILLARS:
            s = feats[k].to(STORE_DTYPE[k])               # cast first: fp16 overflow is also caught
            fin = torch.isfinite(s)
            row_ok = fin.all(1)
            nf_rows[k] += int(((~row_ok) & ok_dev).sum())
            nf_vals[k] += int(((~fin).sum(1) * ok_dev).sum())
            finite_rows &= row_ok
            acc[k].append(s.cpu())
        n_decode_fail += int((~ok_dev).sum())
        n_nonfinite_any += int(((~finite_rows) & ok_dev).sum())
        labels.append(lab.float())
        valid.append(ok.bool() & finite_rows.cpu())
        if step % 10 == 0 or step == nb:
            print(f"\r  {label}: batch {step}/{nb} | {time.time() - t0:5.0f}s", end="")
    print()
    secs = time.time() - t0
    shard = {k: torch.cat(v) for k, v in acc.items()}
    shard["labels"] = torch.cat(labels)
    shard["valid"] = torch.cat(valid)
    shard["start"] = start
    shard["stats"] = {
        "n_entries": len(entries), "n_decode_fail": n_decode_fail,
        "n_nonfinite_rows_any": n_nonfinite_any,
        "nonfinite_rows_by_pillar": nf_rows, "nonfinite_values_by_pillar": nf_vals,
        "n_kept": int(shard["valid"].sum()), "seconds": secs,
        "images_per_second": len(entries) / max(secs, 1e-9),
        "created": datetime.now().isoformat(),
    }
    return shard


def aggregate_extraction_stats(shard_stats):
    agg = {"n_shards": len(shard_stats), "n_entries": 0, "n_decode_fail": 0,
           "n_nonfinite_rows_any": 0, "n_kept": 0, "extract_seconds": 0.0,
           "nonfinite_rows_by_pillar": {k: 0 for k in ALL_PILLARS},
           "nonfinite_values_by_pillar": {k: 0 for k in ALL_PILLARS}}
    for s in shard_stats:
        for key in ("n_entries", "n_decode_fail", "n_nonfinite_rows_any", "n_kept"):
            agg[key] += s[key]
        agg["extract_seconds"] += s["seconds"]
        for k in ALL_PILLARS:
            agg["nonfinite_rows_by_pillar"][k] += s["nonfinite_rows_by_pillar"][k]
            agg["nonfinite_values_by_pillar"][k] += s["nonfinite_values_by_pillar"][k]
    return agg


def build_or_load_split(name, samples, n_copies):
    sig = split_signature(name, samples, n_copies)
    entries = [(p, l, c) for c in range(n_copies) for (p, l) in samples]
    n_shards = math.ceil(len(entries) / SHARD_SIZE)
    banner(f"{name.upper()} FEATURE CACHE  ({len(entries):,} entries, {n_shards} shards, sig {sig})")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    shards = []
    for k in range(n_shards):
        path = FEATURE_DIR / f"{name}_{sig}_s{k:04d}.pt"
        lo, hi = k * SHARD_SIZE, min((k + 1) * SHARD_SIZE, len(entries))
        shard = None
        if path.exists():
            try:
                shard = torch.load(path, map_location="cpu", weights_only=False)
                ok = (shard["start"] == lo and len(shard["labels"]) == hi - lo
                      and len(shard["valid"]) == hi - lo and "stats" in shard
                      and all(len(shard[p]) == hi - lo for p in ALL_PILLARS))
                if not ok:
                    raise ValueError("shard does not match the expected sample range")
            except Exception as e:
                print(f"  shard {k} unreadable/stale ({e}); rebuilding")
                path.unlink(missing_ok=True)
                shard = None
        if shard is None:
            shard = _extract_shard(entries[lo:hi], lo, f"shard {k + 1}/{n_shards}")
            atomic_torch_save(shard, path)
        else:
            print(f"  shard {k + 1}/{n_shards}: loaded from cache")
        shards.append(shard)

    agg = aggregate_extraction_stats([s["stats"] for s in shards])
    atomic_write_text(json.dumps(agg, indent=2), RUN_ROOT / f"extraction_stats_{name}_{sig}.json")
    print(f"  extraction stats: decode failures {agg['n_decode_fail']} | "
          f"non-finite rows {agg['n_nonfinite_rows_any']} | kept {agg['n_kept']}/{agg['n_entries']}")
    if agg["n_nonfinite_rows_any"]:
        print(f"    non-finite rows by pillar  : {agg['nonfinite_rows_by_pillar']}")
        print(f"    non-finite values by pillar: {agg['nonfinite_values_by_pillar']}")

    cache = {k: torch.cat([s[k] for s in shards]) for k in ALL_PILLARS}
    labels = torch.cat([s["labels"] for s in shards])
    valid = torch.cat([s["valid"] for s in shards])
    assert len(labels) == len(entries), "cache/sample-list length mismatch"
    copies = torch.tensor([e[2] for e in entries])
    n_bad = int((~valid).sum())
    if n_bad:
        bad_rows = [(entries[i][0], entries[i][1], entries[i][2])
                    for i in torch.nonzero(~valid).flatten().tolist()]
        pd.DataFrame(bad_rows, columns=["path", "label", "copy"]).to_csv(
            RUN_ROOT / f"invalid_samples_{name}.csv", index=False)
        print(f"  WARNING: {n_bad} unreadable / non-finite samples dropped "
              f"(listed in invalid_samples_{name}.csv)")
        frac = n_bad / max(len(entries), 1)
        if frac > MAX_DROP_ABORT:
            raise RuntimeError(f"{frac:.1%} of the {name} entries were dropped (> {MAX_DROP_ABORT:.0%}); "
                               f"this indicates a bug or a broken dataset. Aborting.")
        if frac > MAX_DROP_WARN:
            print(f"  WARNING: more than {MAX_DROP_WARN:.0%} of the data was dropped - inspect it.")
    keep = valid
    out = {k: cache[k][keep] for k in ALL_PILLARS}
    out["labels"] = labels[keep]
    out["copy"] = copies[keep]
    out["paths"] = [entries[i][0] for i in torch.nonzero(keep).flatten().tolist()]
    out["sig"] = sig
    out["split"] = name
    out["extraction_stats"] = agg
    return out


# ================================================================
# 8. FEATURE NORMALISATION (fit on TRAIN only)
# ================================================================
def compute_stats(train_cache):
    """Per-feature mean/std. Refuses anything that is not the training split."""
    if train_cache.get("split") != "train":
        raise RuntimeError("compute_stats() may only be called on the TRAIN split cache.")
    stats = {}
    for k in ALL_PILLARS:
        x = train_cache[k].float()
        stats[k] = {"mean": x.mean(0), "std": x.std(0).clamp_min(1e-6)}
    return stats


def standardize(feats, stats, keys, clip=FEATURE_CLIP):
    """Apply STORED (train-only) statistics. There is no code path here that computes statistics."""
    out = {}
    for k in keys:
        x = feats[k].float()
        m, s = stats[k]["mean"].to(x.device), stats[k]["std"].to(x.device)
        out[k] = ((x - m) / s).clamp_(-clip, clip).to(DEVICE)
    return out


# ================================================================
# 9. MODEL
# ================================================================
class PillarProjector(nn.Module):
    def __init__(self, d_in, d_out, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(cfg["input_dropout"]),
            nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(cfg["proj_dropout"]),
            nn.Linear(d_out, d_out), nn.LayerNorm(d_out),
        )

    def forward(self, x):
        return self.net(x)


class GatedFusion(nn.Module):
    """Softmax gate over pillars, computed from all (masked) pillar embeddings."""

    def __init__(self, dim, n, cfg):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * n), nn.Linear(dim * n, dim), nn.GELU(),
            nn.Dropout(cfg["gate_dropout"]), nn.Linear(dim, n),
        )

    def forward(self, z, mask=None):                 # z: [B,n,D], mask: [B,n] (1=keep)
        if mask is not None:
            z = z * mask.unsqueeze(-1)
        logits = self.gate(z.flatten(1)).float()
        if mask is not None:
            logits = logits.masked_fill(mask < 0.5, float("-inf"))
        w = torch.softmax(logits, dim=1)
        fused = (z * w.unsqueeze(-1).to(z.dtype)).sum(1)
        return fused, w


class FivePillarDetector(nn.Module):
    def __init__(self, dims, active, cfg):
        super().__init__()
        self.active = list(active)
        d = cfg["pillar_dim"]
        self.proj = nn.ModuleDict({k: PillarProjector(dims[k], d, cfg) for k in self.active})
        self.fusion = GatedFusion(d, len(self.active), cfg)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, cfg["hidden_dim"]), nn.GELU(),
            nn.Dropout(cfg["head_dropout"]), nn.Linear(cfg["hidden_dim"], 128), nn.GELU(),
            nn.Dropout(cfg["head_dropout2"]), nn.Linear(128, 1),
        )

    def forward(self, feats, mask=None):
        z = torch.stack([self.proj[k](feats[k]) for k in self.active], dim=1)
        fused, w = self.fusion(z, mask)
        return self.classifier(fused).squeeze(-1), w


def sample_pillar_mask(B, n, p, device):
    m = (torch.rand(B, n, device=device) > p).float()
    dead = (m.sum(1) == 0).nonzero(as_tuple=True)[0]
    if len(dead) > 0:
        m[dead, torch.randint(0, n, (len(dead),), device=device)] = 1.0
    return m


# ================================================================
# 10. LOSS / METRICS / PREDICTION / EXPLANATION / BUNDLE LOADING
# ================================================================
def make_loss_fn(pos_weight):
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)

    def loss_fn(logits, y, gates):
        target = y * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
        bce = F.binary_cross_entropy_with_logits(logits.float(), target, pos_weight=pw)
        ent = -(gates * torch.log(gates + 1e-8)).sum(1).mean()
        return bce - GATE_ENTROPY_WEIGHT * ent, bce, ent

    return loss_fn


@torch.no_grad()
def predict(model, feats, bs=2048, mask_col=None):
    """Returns logits [N], gates [N,n]. mask_col: drop that pillar (index within model.active)."""
    model.eval()
    N = feats[model.active[0]].shape[0]
    n = len(model.active)
    L, G = [], []
    for i in range(0, N, bs):
        fb = {k: feats[k][i:i + bs] for k in model.active}
        mask = None
        if mask_col is not None:
            mask = torch.ones(fb[model.active[0]].shape[0], n, device=DEVICE)
            mask[:, mask_col] = 0.0
        lg, g = model(fb, mask)
        L.append(lg.float().cpu())
        G.append(g.float().cpu())
    return torch.cat(L), torch.cat(G)


def metrics_from(y, logits, thr=CLASSIFICATION_THRESHOLD, full=True):
    y = np.asarray(y).astype(int)
    p = torch.sigmoid(torch.as_tensor(logits)).numpy()
    pred = (p >= thr).astype(int)
    m = {"accuracy": float(accuracy_score(y, pred))}
    two = len(np.unique(y)) == 2
    m["roc_auc"] = float(roc_auc_score(y, p)) if two else float("nan")
    m["average_precision"] = float(average_precision_score(y, p)) if two else float("nan")
    if full:
        m["precision"] = float(precision_score(y, pred, zero_division=0))
        m["recall"] = float(recall_score(y, pred, zero_division=0))
        m["f1"] = float(f1_score(y, pred, zero_division=0))
        m["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return m


def fit_temperature(logits, y):
    """Temperature scaling on CLEAN validation logits. Returns (T_final, T_unconstrained).
    T_final = max(1.0, T_unconstrained): calibration may only SOFTEN predictions. (On an
    in-distribution near-separable set the unconstrained optimum tends to sharpen, which would
    increase over-confidence on unseen generators.) T > 0 never changes the decision at 0.5."""
    logits = torch.as_tensor(logits).float()
    y = torch.as_tensor(y).float()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        log_t.data.zero_()
    t_unc = float(log_t.exp().clamp(0.05, 20.0))
    if not math.isfinite(t_unc):
        t_unc = 1.0
    return max(1.0, t_unc), t_unc


@torch.no_grad()
def explain(model, feats, temperature=1.0):
    """Per-image output: prediction, temperature-scaled predicted probability, confidence,
    fusion weights and leave-one-pillar-out logit changes.
    Fusion weights = how the model mixes pillar embeddings (NOT causal evidence).
    Delta-logit = change in the AI-generated logit when that pillar is removed (an association
    measured on this model, still not proof of a specific artifact)."""
    logits, gates = predict(model, feats)
    prob = torch.sigmoid(logits / temperature)
    n = len(model.active)
    deltas = torch.zeros(len(logits), n)
    if n > 1:
        for j in range(n):
            lj, _ = predict(model, feats, mask_col=j)
            deltas[:, j] = logits - lj
    return {"prob_fake": prob, "label": (prob >= CLASSIFICATION_THRESHOLD).long(),
            "confidence": torch.maximum(prob, 1 - prob), "gates": gates,
            "delta_logit": deltas, "pillars": list(model.active)}


def verify_bundle_code(bundle):
    """Stops an evaluator from extracting features with code that differs from training."""
    cur = _feature_code_hash()
    if bundle["feature_code_hash"] != cur:
        raise RuntimeError(
            f"Feature-extraction code differs from the code used for training "
            f"(bundle {bundle['feature_code_hash']} vs current {cur}). Use the same script version "
            f"({bundle['code_version']}) for zero-shot extraction.")


def load_bundle(path, device=DEVICE, check_code=True):
    """Zero-shot evaluator entry point:
         model, b = load_bundle(path)
         raw   = extract_batch(uint8_batch, get_encoders())     # same extractors
         feats = standardize(raw, b['stats'], b['active'], b['feature_clip'])  # stored train stats
         logits, gates = predict(model, feats)
         prob = sigmoid(logits / b['temperature']); label = prob >= b['threshold']
       Load every evaluation image with center_crop_or_upscale(img, b['preprocessing']['image_size'])."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    if check_code:
        verify_bundle_code(b)
    model = FivePillarDetector(b["dims"], b["active"], b["model_cfg"]).to(device)
    model.load_state_dict(b["model_state_dict"])
    model.eval()
    return model, b


# ================================================================
# 11. MAIN
# ================================================================
def main():
    banner(f"FIVE-PILLAR DETECTOR {CODE_VERSION}")
    print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Enable the T4 accelerator in Kaggle settings.")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Active pillars:", ACTIVE_PILLARS, "| run tag:", RUN_TAG)
    print(f"VAL_AUG = {VAL_AUG} | TRAIN_AUG_COPIES = {TRAIN_AUG_COPIES}")
    if VAL_AUG:
        print("NOTE: VAL_AUG=True builds an augmented validation copy for MONITORING only. "
              "Checkpoint selection, early stopping and temperature use the CLEAN copy only.")
    for d in (RUN_ROOT, FEATURE_DIR, RUN_DIR, CKPT_DIR, RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    feature_code_hash = _feature_code_hash()     # raises if it cannot be computed
    print("Feature-code hash:", feature_code_hash)

    # ---------------- data ----------------
    train_dir, val_dir = DATASET_DIR / "train", DATASET_DIR / "val"
    for d in (DATASET_DIR, train_dir, val_dir):
        if not d.exists():
            raise FileNotFoundError(f"Missing: {d}")
    train_samples, val_samples = collect_samples(train_dir), collect_samples(val_dir)
    if not train_samples or not val_samples:
        raise RuntimeError("No images found. Expected <split>/<category>/{0_real,1_fake}/")
    audit_dataset(train_samples, "train")
    audit_dataset(val_samples, "val")
    if CHECK_DUPLICATES:
        val_samples = remove_cross_split_duplicates(train_samples, val_samples)
    for nm, s in (("train", train_samples), ("val", val_samples)):
        ys = [l for _, l in s]
        print(f"{nm}: {len(s):,} images | real {ys.count(0):,} | fake {ys.count(1):,}")

    config = build_config()
    config["created"] = datetime.now().isoformat()
    config["gpu"] = torch.cuda.get_device_name(0)
    config["library_versions"] = library_versions()
    atomic_write_text(json.dumps(config, indent=2, default=str), RUN_DIR / "config.json")

    # ---------------- caches ----------------
    train_cache = build_or_load_split("train", train_samples, 1 + TRAIN_AUG_COPIES)
    val_cache = build_or_load_split("val", val_samples, 2 if VAL_AUG else 1)
    release_encoders()

    dims = {k: int(train_cache[k].shape[1]) for k in ALL_PILLARS}
    banner("FEATURE CACHE SUMMARY")
    for k in ALL_PILLARS:
        flag = "" if dims[k] == EXPECTED_DIMS[k] else f"  (expected {EXPECTED_DIMS[k]})"
        print(f"{k:10s} dim {dims[k]:5d} | train {tuple(train_cache[k].shape)} | "
              f"val {tuple(val_cache[k].shape)}{flag}")
    concat_dim = sum(dims[k] for k in ACTIVE_PILLARS)
    print(f"Raw concatenated dim (active pillars): {concat_dim} | projected: "
          f"{len(ACTIVE_PILLARS)} x {PILLAR_DIM} = {len(ACTIVE_PILLARS) * PILLAR_DIM} "
          f"-> fused {PILLAR_DIM}")

    stats = compute_stats(train_cache)                 # TRAIN ONLY (enforced inside)
    train_feats = standardize(train_cache, stats, ACTIVE_PILLARS)
    val_feats = standardize(val_cache, stats, ACTIVE_PILLARS)
    train_y = train_cache["labels"].to(DEVICE)
    val_y_np = val_cache["labels"].numpy()
    val_copy = val_cache["copy"].numpy()
    clean_idx, aug_idx = np.where(val_copy == 0)[0], np.where(val_copy > 0)[0]
    if not VAL_AUG:
        assert len(aug_idx) == 0, "VAL_AUG is False but augmented validation rows exist"
    clean_idx_t = torch.as_tensor(clean_idx)
    n_train_entries = len(train_y)
    train_sig, val_sig = train_cache["sig"], val_cache["sig"]
    train_extraction, val_extraction = train_cache["extraction_stats"], val_cache["extraction_stats"]
    del train_cache["spatial"], train_cache["semantic"]   # free host RAM
    gc.collect()

    n_real = int((train_y == 0).sum())
    n_fake = int((train_y == 1).sum())
    pos_weight = n_real / max(n_fake, 1)
    print(f"Training samples (clean+aug): {n_train_entries:,} | pos_weight {pos_weight:.3f} "
          f"(= n_real/n_fake; a no-op on balanced data) | clean val: {len(clean_idx):,}")

    # ---------------- model / optimiser ----------------
    MCFG = model_cfg()
    model = FivePillarDetector(dims, ACTIVE_PILLARS, MCFG).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")
    loss_fn = make_loss_fn(pos_weight)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}], lr=LEARNING_RATE)

    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / float(WARMUP_EPOCHS)
        prog = (ep - WARMUP_EPOCHS) / float(max(1, MAX_EPOCHS - WARMUP_EPOCHS))
        return MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    meta = {"train_sig": train_sig, "val_sig": val_sig, "active": list(ACTIVE_PILLARS),
            "dims": dims, "code_version": CODE_VERSION, "feature_code_hash": feature_code_hash,
            "model_cfg": MCFG}

    # ---------------- resume ----------------
    start_epoch, best_loss, best_epoch, patience, history, finished = 0, float("inf"), -1, 0, [], False
    ckpt = None
    for cand in (LATEST_CKPT, PREV_CKPT):
        if cand.exists():
            try:
                ckpt = torch.load(cand, map_location="cpu", weights_only=False)
                print(f"\nFound checkpoint: {cand.name}")
                break
            except Exception as e:
                print(f"Checkpoint {cand.name} unreadable ({e}); trying fallback.")
    if ckpt is not None:
        if ckpt["meta"] != meta:
            raise RuntimeError(
                "The checkpoint was created with different features/pillars/code. "
                "Delete " + str(RUN_DIR) + " (or change ACTIVE_PILLARS) to start a fresh run.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss, best_epoch = ckpt["best_loss"], ckpt["best_epoch"]
        patience, history = ckpt["patience"], ckpt["history"]
        finished = ckpt.get("finished", False)
        try:
            torch.set_rng_state(ckpt["rng"]["torch"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
            np.random.set_state(ckpt["rng"]["numpy"])
            random.setstate(ckpt["rng"]["python"])
        except Exception:
            pass
        print(f"Resuming at epoch {start_epoch + 1} | best clean val loss {best_loss:.6f} | "
              f"finished={finished}")
    if BEST_CKPT.exists():                      # best file may be ahead of latest after a crash
        try:
            b = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
            if b.get("meta") != meta:
                print("Existing best-model file belongs to a different configuration; ignored.")
            elif b["val_loss"] < best_loss:
                best_loss, best_epoch = b["val_loss"], b["epoch"]
        except Exception:
            print("Best-model file unreadable; it will be rewritten on the next improvement.")

    protocol = {
        "train_dir": str(train_dir), "val_dir": str(val_dir),
        "n_train_entries_clean_plus_aug": n_train_entries,
        "n_clean_val": int(len(clean_idx)), "train_aug_copies": TRAIN_AUG_COPIES, "val_aug": VAL_AUG,
        "selection_metric": "clean validation BCE loss (clean copy only)",
        "temperature_fit_on": "clean validation logits only, T>=1",
        "threshold_policy": "fixed at 0.5; must not be tuned on zero-shot/perturbation data",
        "validation_is_zero_shot": False,
        "normalisation_fit_on": "train split only (clean + augmented training copies)",
        "physical_pillar": "image-derived physical-consistency cues; NOT PRNU/iPRNU",
        "gate_interpretation": "learned mixing weights, not causal attribution",
        "train_extraction_stats": train_extraction, "val_extraction_stats": val_extraction,
    }

    def make_bundle(epoch, val_loss, finalized=False):
        return {
            "bundle_format": 3, "code_version": CODE_VERSION,
            "feature_code_hash": feature_code_hash,
            "model_state_dict": model.state_dict(), "model_cfg": MCFG,
            "active": list(ACTIVE_PILLARS), "pillar_order": list(ACTIVE_PILLARS),
            "all_pillars": list(ALL_PILLARS), "dims": dims,
            "raw_concat_dim": sum(dims[k] for k in ACTIVE_PILLARS),
            "projected_dim": PILLAR_DIM,
            "stats": {k: {"mean": v["mean"].cpu(), "std": v["std"].cpu()} for k, v in stats.items()},
            "feature_clip": FEATURE_CLIP,
            "preprocessing": preprocessing_spec(), "pretrained": pretrained_spec(),
            "library_versions": library_versions(),
            "threshold": CLASSIFICATION_THRESHOLD,
            "temperature": 1.0, "temperature_unconstrained": None,
            "config": config, "protocol": protocol, "meta": meta,
            "epoch": epoch, "val_loss": val_loss, "selection_metric": "clean_val_bce_loss",
            "finalized": finalized, "created": datetime.now().isoformat(),
        }

    # ---------------- training ----------------
    banner("TRAINING")
    N = n_train_entries
    n_batches = math.ceil(N / BATCH_SIZE_TRAIN)
    print(f"{N:,} samples | {n_batches} batches/epoch | selection metric: CLEAN validation loss "
          f"({len(clean_idx):,} images)")

    for epoch in range(start_epoch, MAX_EPOCHS):
        if finished:
            break
        t0 = time.time()
        model.train()
        g = torch.Generator().manual_seed(SEED * 1000 + epoch)
        perm = torch.randperm(N, generator=g)
        lr_now = optimizer.param_groups[0]["lr"]
        tot = tot_bce = tot_ent = 0.0
        seen = skipped = 0
        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE_TRAIN:(b + 1) * BATCH_SIZE_TRAIN].to(DEVICE)
            fb = {k: train_feats[k][idx] for k in ACTIVE_PILLARS}
            yb = train_y[idx]
            mask = (sample_pillar_mask(len(idx), len(ACTIVE_PILLARS), PILLAR_DROPOUT, DEVICE)
                    if len(ACTIVE_PILLARS) > 1 else None)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                logits, gates = model(fb, mask)
            loss, bce, ent = loss_fn(logits, yb, gates)
            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            n_b = len(idx)
            tot += loss.item() * n_b
            tot_bce += bce.item() * n_b
            tot_ent += ent.item() * n_b
            seen += n_b
            if (b + 1) % 20 == 0 or b + 1 == n_batches:
                print(f"\rEpoch {epoch + 1:02d}/{MAX_EPOCHS} | batch {b + 1}/{n_batches} | "
                      f"loss {tot / max(seen, 1):.4f}", end="")
        print()
        if skipped > 0.1 * n_batches:
            raise RuntimeError(f"{skipped}/{n_batches} batches had non-finite loss - aborting.")

        # ---- validation: SELECTION USES THE CLEAN COPY ONLY ----
        logits_v, gates_v = predict(model, val_feats)
        yv = torch.as_tensor(val_y_np)
        val_loss = float(F.binary_cross_entropy_with_logits(
            logits_v[clean_idx_t], yv[clean_idx_t]))                     # clean only
        mc = metrics_from(val_y_np[clean_idx], logits_v[clean_idx])
        ma = (metrics_from(val_y_np[aug_idx], logits_v[aug_idx], full=False)   # monitor only
              if len(aug_idx) else {"accuracy": float("nan"), "roc_auc": float("nan"),
                                    "average_precision": float("nan")})
        gm = gates_v[clean_idx].mean(0).numpy()
        rec = {"epoch": epoch + 1, "train_loss": tot / max(seen, 1), "train_bce": tot_bce / max(seen, 1),
               "gate_entropy": tot_ent / max(seen, 1), "val_loss_clean": val_loss,
               "accuracy": mc["accuracy"], "precision": mc["precision"], "recall": mc["recall"],
               "f1": mc["f1"], "roc_auc": mc["roc_auc"], "average_precision": mc["average_precision"],
               "aug_accuracy_monitor": ma["accuracy"], "aug_roc_auc_monitor": ma["roc_auc"],
               "lr": lr_now, "skipped_batches": skipped, "epoch_seconds": time.time() - t0}
        for k, v in zip(ACTIVE_PILLARS, gm):
            rec[f"gate_{k}"] = float(v)
        history.append(rec)

        print(f"  clean val loss {val_loss:.5f} | acc {mc['accuracy']:.4f} | F1 {mc['f1']:.4f} | "
              f"AUC {mc['roc_auc']:.4f} | AP {mc['average_precision']:.4f}")
        if len(aug_idx):
            print(f"  (monitor only) aug val acc {ma['accuracy']:.4f} | aug AUC {ma['roc_auc']:.4f}")
        print("  fusion weights: " + " ".join(f"{k}={v:.3f}" for k, v in zip(ACTIVE_PILLARS, gm)))

        improved = val_loss < best_loss - MIN_DELTA
        if improved:
            best_loss, best_epoch, patience = val_loss, epoch + 1, 0
            atomic_torch_save(make_bundle(epoch + 1, val_loss), BEST_CKPT)   # best first, latest second
            print("  *** new best (clean validation loss) ***")
        else:
            patience += 1
        scheduler.step()
        stop = patience >= EARLY_STOPPING_PATIENCE
        finished = stop or (epoch == MAX_EPOCHS - 1)
        save_latest_checkpoint({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_loss": best_loss, "best_epoch": best_epoch, "patience": patience,
            "history": history, "finished": finished, "meta": meta,
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(), "python": random.getstate()},
            "timestamp": datetime.now().isoformat(),
        })
        atomic_write_text(pd.DataFrame(history).to_csv(index=False), HISTORY_FILE)
        print(f"  checkpoint saved | best epoch {best_epoch} ({best_loss:.5f}) | "
              f"patience {patience}/{EARLY_STOPPING_PATIENCE} | {time.time() - t0:.1f}s")
        if stop:
            print("\nEarly stopping (clean validation loss stopped improving).")
            break

    # ---------------- final report on the BEST model ----------------
    banner("FINAL VALIDATION REPORT (best model; in-distribution, NOT zero-shot)")
    if BEST_CKPT.exists():
        best = torch.load(BEST_CKPT, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state_dict"])
    else:
        best = make_bundle(best_epoch, best_loss)
    logits_v, gates_v = predict(model, val_feats)
    yc = val_y_np[clean_idx]
    lc = logits_v[clean_idx]
    temperature, t_unc = fit_temperature(lc, yc)
    nll_before = float(F.binary_cross_entropy_with_logits(lc, torch.as_tensor(yc).float()))
    nll_after = float(F.binary_cross_entropy_with_logits(lc / temperature, torch.as_tensor(yc).float()))
    best["temperature"] = temperature
    best["temperature_unconstrained"] = t_unc
    best["threshold"] = CLASSIFICATION_THRESHOLD
    best["finalized"] = True
    atomic_torch_save(best, BEST_CKPT)
    print(f"Best epoch: {best['epoch']} | temperature: {temperature:.3f} "
          f"(unconstrained fit {t_unc:.3f}; clamped to T>=1) | "
          f"val NLL {nll_before:.5f} -> {nll_after:.5f}")

    report = {"best_epoch": int(best["epoch"]), "temperature": temperature,
              "temperature_unconstrained": t_unc, "nll_before": nll_before, "nll_after": nll_after,
              "threshold": CLASSIFICATION_THRESHOLD,
              "active_pillars": list(ACTIVE_PILLARS), "run_tag": RUN_TAG,
              "report_population": {"n_clean_val": int(len(clean_idx)),
                                    "note": "ALL clean validation samples are used (no subsetting)."},
              "clean_val": metrics_from(yc, lc),
              "augmented_val_monitor": (metrics_from(val_y_np[aug_idx], logits_v[aug_idx])
                                        if len(aug_idx) else None),
              "per_category_clean": {}, "mean_fusion_weights": {}, "leave_one_pillar_out": {},
              "extraction_stats": {"train": train_extraction, "val": val_extraction},
              "notes": [
                  "Validation is in-distribution (same ProGAN generator / categories); not zero-shot.",
                  "Physical pillar = image-derived physical-consistency cues, not PRNU/iPRNU.",
                  "Fusion weights are learned mixing weights, not causal attributions.",
                  "Temperature was fit on clean in-distribution validation data and clamped to T>=1; "
                  "confidence on unseen generators is expected to be over-optimistic.",
                  "Threshold is fixed at 0.5 and must not be tuned on zero-shot data.",
              ]}
    cats = np.array([category_of(p) for p in val_cache["paths"]])
    for c in sorted(set(cats)):
        sel = np.where((cats == c) & (val_copy == 0))[0]
        report["per_category_clean"][c] = metrics_from(val_y_np[sel], logits_v[sel])
    gm = gates_v[clean_idx].mean(0).numpy()
    report["mean_fusion_weights"] = {k: float(v) for k, v in zip(ACTIVE_PILLARS, gm)}

    ci = torch.as_tensor(clean_idx, device=DEVICE)
    sub = {k: v[ci] for k, v in val_feats.items()}          # ALL clean validation samples
    info = explain(model, sub, temperature)
    ysub = torch.as_tensor(yc)
    for cls, nm in ((0, "real"), (1, "fake")):
        m = ysub == cls
        if m.any():
            report["leave_one_pillar_out"][nm] = {
                k: float(v) for k, v in zip(ACTIVE_PILLARS, info["delta_logit"][m].mean(0))}

    print(f"\nClean validation ({len(clean_idx)} images): acc {report['clean_val']['accuracy']:.4f} | "
          f"F1 {report['clean_val']['f1']:.4f} | AUC {report['clean_val']['roc_auc']:.4f} | "
          f"AP {report['clean_val']['average_precision']:.4f}")
    print("Confusion matrix [[TN,FP],[FN,TP]]:", report["clean_val"]["confusion_matrix"])
    if report["augmented_val_monitor"]:
        print(f"Augmented validation (monitor only): acc {report['augmented_val_monitor']['accuracy']:.4f} "
              f"| AUC {report['augmented_val_monitor']['roc_auc']:.4f}")
    print("\nPer-category (clean):")
    for c, m in report["per_category_clean"].items():
        print(f"  {c:8s} acc {m['accuracy']:.4f} | AUC {m['roc_auc']:.4f} | AP {m['average_precision']:.4f}")
    print("\nMean learned fusion weights:", {k: round(v, 3) for k, v in report["mean_fusion_weights"].items()})
    print("Mean drop-one delta-logit:", {nm: {k: round(v, 2) for k, v in d.items()}
                                          for nm, d in report["leave_one_pillar_out"].items()})

    rs = np.random.RandomState(SEED)                        # deterministic example selection
    yl = ysub.numpy()
    shown = []
    for cls in (0, 1):
        pool = np.where(yl == cls)[0]
        if len(pool):
            shown += list(rs.choice(pool, size=min(2, len(pool)), replace=False))
    print("\nExample per-image outputs (2 real + 2 fake, deterministic random draw from clean val):")
    for i in shown:
        lab = "AI-GENERATED" if info["label"][i] == 1 else "REAL"
        print(f"\n  truth={'FAKE' if yl[i] == 1 else 'REAL'} | prediction: {lab} | "
              f"temperature-scaled p(AI-generated)={info['prob_fake'][i]:.3f} | "
              f"confidence={info['confidence'][i]:.3f}")
        print("    fusion weights : " + " ".join(
            f"{k}={info['gates'][i][j]:.2f}" for j, k in enumerate(ACTIVE_PILLARS)))
        print("    drop-one dLogit: " + " ".join(
            f"{k}={info['delta_logit'][i][j]:+.2f}" for j, k in enumerate(ACTIVE_PILLARS)))
    print("\n  (Fusion weights and drop-one changes describe this model's behaviour; they do not "
          "prove that a specific artifact caused the decision.)")

    atomic_write_text(json.dumps(report, indent=2, default=str),
                      RESULT_DIR / "final_validation_report.json")
    banner("DONE")
    print("Best deployable bundle :", BEST_CKPT)
    print("Latest checkpoint      :", LATEST_CKPT)
    print("History CSV            :", HISTORY_FILE)
    print("Report JSON            :", RESULT_DIR / "final_validation_report.json")
    print("Dataset audits         :", RUN_ROOT / "dataset_audit_train.json", "|",
          RUN_ROOT / "dataset_audit_val.json")
    print("\nKaggle reminder: copy best_model.pth, the feature_cache folder and the results to a "
          "Kaggle Dataset / download them. /kaggle/working may not survive a session that is "
          "terminated.")


if __name__ == "__main__":
    main()
