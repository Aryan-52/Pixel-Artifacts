"""Forensic views of one image, returned as PNG data URLs.

All views are computed on the same 224x224 crop the model scores, so they line up with the verdict.
They are aids for a human reviewer; only the Grad-CAM comes from the model itself.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageFilter

# ink -> teal -> saffron -> coral -> cream: the app's palette, readable on the light UI
_STOPS = np.array([[16, 35, 42], [14, 94, 111], [240, 162, 2], [228, 87, 46], [255, 240, 220]], float)


def _lut() -> np.ndarray:
    x = np.linspace(0, 1, 256)
    pos = np.linspace(0, 1, len(_STOPS))
    return np.stack([np.interp(x, pos, _STOPS[:, c]) for c in range(3)], 1).astype(np.uint8)


LUT = _lut()


def png_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def jpeg_url(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def colorize(m: np.ndarray) -> Image.Image:
    m = np.clip(m, 0, 1)
    return Image.fromarray(LUT[(m * 255).astype(np.uint8)])


def _norm(m: np.ndarray, lo_pct: float = 1, hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(m, [lo_pct, hi_pct])
    return (m - lo) / max(hi - lo, 1e-8)


def gradcam_overlay(crop: Image.Image, cam: np.ndarray) -> Image.Image:
    base = np.asarray(crop.convert("L").convert("RGB"), float) * 0.55
    heat = np.asarray(colorize(cam), float)
    a = (0.25 + 0.6 * cam)[..., None]
    return Image.fromarray((base * (1 - a) + heat * a).astype(np.uint8))


def spectrum(crop: Image.Image) -> Image.Image:
    """Centred log-magnitude Fourier spectrum. Upsampling layers of many generators leave
    periodic peaks (bright dots / grid lines) away from the centre."""
    g = np.asarray(crop.convert("L"), float)
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g * win))))
    return colorize(_norm(mag, 5, 99.9))


def noise_residual(crop: Image.Image) -> Image.Image:
    """Image minus its 3x3 median: the fine texture left after removing content."""
    g = crop.convert("L")
    r = np.abs(np.asarray(g, float) - np.asarray(g.filter(ImageFilter.MedianFilter(3)), float))
    return colorize(_norm(r, 1, 99.5))


def ela(crop: Image.Image, quality: int = 90) -> Image.Image:
    """Error level analysis: difference after one JPEG re-save. Regions that react differently
    were often processed differently; uniform response is normal for untouched images."""
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=quality)
    re = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    d = np.abs(np.asarray(crop, float) - np.asarray(re, float)).max(2)
    return colorize(_norm(d, 1, 99.7))


def histograms(crop: Image.Image, bins: int = 32) -> dict:
    a = np.asarray(crop, np.uint8)
    out = {}
    for i, ch in enumerate("rgb"):
        h, _ = np.histogram(a[..., i], bins=bins, range=(0, 256))
        out[ch] = (h / h.sum()).round(5).tolist()
    return out


def thumbnail(img: Image.Image, side: int = 320) -> Image.Image:
    t = img.convert("RGB")
    t.thumbnail((side, side))
    return t
