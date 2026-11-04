"""fp-v4.x inference engine.

All feature extraction and the fusion model are the training code's own functions
(src/models/five_pillar_v3.py). Differences from fp-v4.0: the spatial encoder is a fine-tuned
EfficientNet-B0 (fp-v4.1: D-20; fp-v4.2: fresh degradations, D-22), and from fp-v4.2 the
semantic pillar is CLIP ViT-L/14, named in the model folder's store_meta.json.

One image, exactly as the feature store was built for a clean (copy 0) image:
  PIL decode -> RGB -> center_crop_or_upscale(224) -> uint8 CHW
  -> extract_batch (raw features) -> STORE_DTYPE cast -> standardize with the TRAIN-ONLY stats in
     the bundle, clip +-10 -> fp16 round trip (the store is fp16) -> gated fusion -> logit
  -> sigmoid (temperature 1.0) -> fixed threshold 0.5.

The app reads the WHOLE image (scan, DECISION_LOG D-21): the same per-window pipeline on a grid of
224 px windows covering every pixel, averaged. The centre crop above is the training/paper protocol.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import sys
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.config import REPO, Settings

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

log = logging.getLogger("pixel_artifacts.engine")

PILLARS = ["spatial", "frequency", "color", "physical", "semantic"]


class ImageError(ValueError):
    """The upload is not a usable image; the message is safe to show to the user."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def decode(data: bytes, max_pixels: int) -> Image.Image:
    if not data:
        raise ImageError("The file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        if img.size[0] * img.size[1] > max_pixels:
            raise ImageError(f"The image is too large ({img.size[0]}x{img.size[1]} pixels).")
        img.load()
        return img
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ImageError("The file could not be read as an image.")


class Engine:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self.loaded = False

    # ------------------------------------------------------------------ loading
    def load(self) -> "Engine":
        import torch
        from torchvision import models

        from src.models import five_pillar_v3 as v3
        from src.paths import section

        d = self.s.model_dir
        sums = json.loads((d / "CHECKSUMS.json").read_text())["files"]
        for name in ("head.pth", "spatial_effnet.pth"):
            if sha256_file(d / name) != sums[name]["sha256"]:
                raise RuntimeError(f"{d / name} does not match CHECKSUMS.json")

        dev = self.s.device
        self.device = torch.device("cuda" if dev == "cuda" or (dev == "auto" and torch.cuda.is_available())
                                   else "cpu")
        v3.DEVICE = self.device
        v3.PRETRAINED_DIR = section("outputs")["pretrained"]
        self.torch, self.v3 = torch, v3

        self.model, self.bundle = v3.load_bundle(str(d / "head.pth"), device=self.device, check_code=False)
        expected = f"{v3._feature_code_hash()}+spatial_ft:{sums['spatial_effnet.pth']['sha256'][:12]}"
        meta = d / "store_meta.json"                  # fp-v4.2+: semantic pillar from another CLIP (D-22)
        sem = json.loads(meta.read_text()).get("semantic_encoder") if meta.exists() else None
        if sem:
            expected += f"+semantic:{sem['model']}/{sem['pretrained']}"
            v3.CLIP_MODEL_NAME, v3.CLIP_PRETRAINED = sem["model"], sem["pretrained"]
            v3.EXPECTED_DIMS = {**v3.EXPECTED_DIMS, "semantic": sem["dim"]}
        if self.bundle["feature_code_hash"] != expected:
            raise RuntimeError(f"feature code hash {self.bundle['feature_code_hash']} != {expected}")

        self.enc = v3.get_encoders()                  # OpenCLIP + (replaced below) EfficientNet
        net = models.efficientnet_b0(weights=None)
        net.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(1280, 1))
        net.load_state_dict(torch.load(d / "spatial_effnet.pth", map_location="cpu",
                                       weights_only=False)["effnet"])
        net.to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        self.enc.effnet = net

        self.threshold = float(self.bundle["threshold"])
        self.temperature = float(self.bundle["temperature"])
        self.version = self.bundle.get("model_version", self.s.model_version)
        self._load_attribution()
        self.loaded = True
        log.info("model %s on %s", self.version, self.device)
        return self

    def _load_attribution(self):
        """Optional source-attribution head (models/attribution-v1). It reads the detector's own
        features, so it is only used when it was trained on exactly these features."""
        from src.models import attribution as attr

        self.attr, self.attr_info = None, None
        d = self.s.attribution_dir
        if not (d / "attribution.pth").exists():
            log.info("no source-attribution head in %s", d)
            return
        sums = json.loads((d / "CHECKSUMS.json").read_text())["files"]
        if sha256_file(d / "attribution.pth") != sums["attribution.pth"]["sha256"]:
            raise RuntimeError(f"{d / 'attribution.pth'} does not match CHECKSUMS.json")
        model, ck = attr.load(d / "attribution.pth", self.device)
        if ck["meta"]["feature_code_hash"] != self.bundle["feature_code_hash"]:
            raise RuntimeError("attribution head was trained on different detector features")
        report = json.loads((d / "report.json").read_text())
        ck["test_accuracy"] = report["test"]["accuracy"]
        self.attr, self.attr_info, self._concat = model, ck, attr.concat

    def _attribute(self, feats):
        """Source-attribution probabilities [B, classes] (numpy), or None without the head."""
        if self.attr is None:
            return None
        return self.attr(self._concat(feats)).softmax(1).float().cpu().numpy()

    def source(self, prob) -> dict | None:
        """Mean attribution probabilities over windows -> ranked answer (None top = unknown)."""
        if prob is None:
            return None
        classes, tau = self.attr_info["classes"], float(self.attr_info["unknown_below"])
        ranked = sorted(zip(classes, map(float, prob)), key=lambda t: -t[1])
        return {"top": ranked[0][0] if ranked[0][1] >= tau else None, "unknown_below": tau,
                "test_accuracy": self.attr_info["test_accuracy"],
                "ranked": [{"generator": g, "p": p} for g, p in ranked]}

    # ------------------------------------------------------------------ preprocessing
    def to_tensor(self, crop: Image.Image):
        arr = np.asarray(crop, dtype=np.uint8)
        return self.torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()

    def prepare(self, img: Image.Image):
        """Training preprocessing. EXIF orientation is applied first so the crop matches what people see."""
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        size = self.v3.IMAGE_SIZE
        crop = self.v3.center_crop_or_upscale(img, size)
        if min(w, h) < size:
            box = None                                   # whole (upscaled) image was used
        else:
            box = [(w - size) // 2, (h - size) // 2, size, size]
        return img, crop, {"width": w, "height": h, "crop_box": box, "upscaled": min(w, h) < size}

    # ------------------------------------------------------------------ scoring
    def _features(self, x_uint8):
        v3, b = self.v3, self.bundle
        raw = v3.extract_batch(x_uint8, self.enc)
        stored = {k: v.to(v3.STORE_DTYPE[k]) for k, v in raw.items()}
        feats = v3.standardize(stored, b["stats"], b["active"], b["feature_clip"])
        return {k: v.half().float() for k, v in feats.items()}

    def score(self, x_uint8):
        """uint8 [B,3,224,224] -> (p_ai [B], gates [B,5]) as numpy."""
        torch = self.torch
        with self._lock, torch.inference_mode():
            logits, gates = self.v3.predict(self.model, self._features(x_uint8))
        p = torch.sigmoid(logits.float() / self.temperature)
        return p.numpy(), gates.float().numpy()

    def explain_batch(self, x_uint8):
        """uint8 [B,3,224,224] -> per-window p_ai, gates, leave-one-pillar-out logit changes and
        source-attribution probabilities (numpy)."""
        torch = self.torch
        with self._lock, torch.inference_mode():
            feats = self._features(x_uint8)
            info = self.v3.explain(self.model, feats, self.temperature)
            src = self._attribute(feats)
        return {"p_ai": info["prob_fake"].float().cpu().numpy(), "gates": info["gates"].float().cpu().numpy(),
                "delta_logit": info["delta_logit"].float().cpu().numpy(), "source": src}

    def scan(self, img: Image.Image, chunk: int = 64) -> dict:
        """Full-image protocol (DECISION_LOG D-21): every 224 px window of the (EXIF-oriented RGB)
        image is scored at full resolution; image p(AI) = mean of window probabilities. Images
        smaller than 224 px on a side are read whole (upscaled), as in training."""
        size = self.v3.IMAGE_SIZE
        boxes = window_boxes(*img.size, size)
        parts = []
        for a in range(0, len(boxes), chunk):
            crops = ([self.v3.center_crop_or_upscale(img, size)] if boxes == [None] else
                     [img.crop((x, y, x + size, y + size)) for x, y in boxes[a:a + chunk]])
            parts.append(self.explain_batch(self.torch.stack([self.to_tensor(c) for c in crops])))
        out = {k: (None if parts[0][k] is None else np.concatenate([p[k] for p in parts])) for k in parts[0]}
        p = out["p_ai"]
        pm = float(p.mean())
        focus = int(p.argmax())
        return {"p_ai": pm, "logit": float(np.log(max(pm, 1e-7) / max(1 - pm, 1e-7))),
                "gates": dict(zip(PILLARS, map(float, out["gates"].mean(0)))),
                "delta_logit": dict(zip(PILLARS, map(float, out["delta_logit"].mean(0)))),
                "source": self.source(None if out["source"] is None else out["source"].mean(0)),
                "windows": [{"x": b[0], "y": b[1], "p_ai": float(pi)} for b, pi in zip(boxes, p)] if boxes != [None] else [],
                "size": size, "focus": focus, "max_p_ai": float(p.max()), "share_ai": float((p >= self.threshold).mean())}

    def gradcam(self, x_uint8):
        """Grad-CAM of the fine-tuned spatial EfficientNet (its own real/fake logit), last conv stage.
        Returns (cam [224,224] in [0,1], spatial-expert probability)."""
        torch, F = self.torch, self.torch.nn.functional
        net, enc = self.enc.effnet, self.enc
        x = ((x_uint8.to(self.device).float() / 255.0) - enc.im_mean) / enc.im_std
        with self._lock, torch.enable_grad():
            acts = {}
            h = x
            for i, layer in enumerate(net.features):
                h = layer(h)
                if i == len(net.features) - 1:
                    h.requires_grad_(True)
                    acts["a"] = h
            logit = net.classifier(net.avgpool(h).flatten(1))[0, 0]
            (grad,) = torch.autograd.grad(logit, acts["a"])
        a = acts["a"].detach()
        w = grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / cam.max().clamp_min(1e-8)).cpu().numpy()
        return cam, float(torch.sigmoid(logit.detach()).item())

    def describe(self) -> dict:
        return {"model_version": self.version, "device": str(self.device), "pillars": PILLARS,
                "threshold": self.threshold, "temperature": self.temperature,
                "image_size": self.v3.IMAGE_SIZE, "crop_policy": self.v3.CROP_POLICY,
                "parameters_head": sum(p.numel() for p in self.model.parameters())}


def window_boxes(w: int, h: int, size: int = 224) -> list:
    """Top-left corners of the D-21 grid: ceil(W/size) x ceil(H/size) windows, evenly spaced, first
    and last on the image edges, so every pixel is read. [None] = image too small (read whole)."""
    if min(w, h) < size:
        return [None]
    nx, ny = -(-w // size), -(-h // size)
    xs = [round(i * (w - size) / (nx - 1)) for i in range(nx)] if nx > 1 else [0]
    ys = [round(j * (h - size) / (ny - 1)) for j in range(ny)] if ny > 1 else [0]
    return [(x, y) for y in ys for x in xs]


def confidence_band(p_ai: float) -> str:
    m = abs(p_ai - 0.5)
    return "high" if m >= 0.4 else "moderate" if m >= 0.2 else "low"


def verdict(p_ai: float, threshold: float) -> str:
    return "ai" if p_ai >= threshold else "real"
