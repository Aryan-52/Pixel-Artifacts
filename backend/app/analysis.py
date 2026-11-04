"""Turn one image into the response the app shows: verdict first, evidence and forensic views after."""
from __future__ import annotations

import hashlib
import time

from PIL import Image

from backend.app import metadata, visuals
from backend.app.engine import Engine, confidence_band, verdict
from backend.app.evidence import Evidence

LENSES = {
    "spatial": {"name": "Texture & detail", "tech": "EfficientNet-B0, fine-tuned on ProGAN",
                "plain": "Looks at fine textures, edges and small visual glitches."},
    "frequency": {"name": "Hidden patterns", "tech": "Fourier (FFT) spectrum statistics",
                  "plain": "Looks for repeating pixel patterns left by image generators."},
    "color": {"name": "Colour balance", "tech": "RGB / YCbCr / opponent-colour statistics",
              "plain": "Checks whether colours relate to each other the way a camera records them."},
    "physical": {"name": "Light & noise", "tech": "Image-derived shading and noise-consistency cues",
                 "plain": "Checks if lighting, shading and grain behave consistently. (Not sensor PRNU.)"},
    "semantic": {"name": "Overall look", "tech": "OpenCLIP ViT-B/32 embedding",
                 "plain": "Judges the image as a whole, the way a person would glance at it."},
}

HEADLINES = {
    ("ai", "high"): "Very likely AI-generated", ("ai", "moderate"): "Likely AI-generated",
    ("ai", "low"): "Leans AI-generated", ("real", "high"): "Very likely a real photo",
    ("real", "moderate"): "Likely a real photo", ("real", "low"): "Leans real",
}


def summary_text(label: str, band: str, p_ai: float, deltas: dict) -> dict:
    toward = {k: v for k, v in deltas.items() if (v > 0) == (label == "ai")}
    top = sorted(toward, key=lambda k: abs(toward[k]), reverse=True)[:2]
    against = {k: v for k, v in deltas.items() if k not in toward}
    lead = (f"The detector gives this image a {p_ai:.0%} chance of being AI-generated."
            if label == "ai" else
            f"The detector gives this image a {1 - p_ai:.0%} chance of being a real photo.")
    why = ""
    if top:
        why = "The strongest signals came from " + " and ".join(LENSES[k]["name"].lower() for k in top) + "."
    caution = {
        "high": "The signals agree strongly.",
        "moderate": "Most signals agree, but not all of them.",
        "low": "The signals are mixed. Treat this result as uncertain and check the image another way.",
    }[band]
    if against and band != "high":
        k = max(against, key=lambda k: abs(against[k]))
        caution += f" {LENSES[k]['name']} pointed the other way."
    return {"headline": HEADLINES[(label, band)], "lead": lead, "why": why, "caution": caution}


def analyze(engine: Engine, evidence: Evidence, data: bytes, filename: str, fmt: str | None,
            img: Image.Image, with_visuals: bool = True) -> dict:
    t0 = time.perf_counter()
    meta = metadata.inspect(img, fmt)
    full, _, info = engine.prepare(img)
    sc = engine.scan(full)
    p = sc["p_ai"]
    label, band = verdict(p, engine.threshold), confidence_band(p)
    size = sc["size"]
    if sc["windows"]:                                  # focus = the window that looked most AI-generated
        fw = sc["windows"][sc["focus"]]
        focus = full.crop((fw["x"], fw["y"], fw["x"] + size, fw["y"] + size))
        info["focus_box"] = [fw["x"], fw["y"], size, size]
    else:
        focus = engine.v3.center_crop_or_upscale(full, size)
        info["focus_box"] = None
    info["windows"] = len(sc["windows"]) or 1
    result = {
        "filename": filename, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        "verdict": label, "p_ai": p, "p_real": 1 - p, "confidence": max(p, 1 - p), "band": band,
        "logit": sc["logit"], "threshold": engine.threshold,
        "summary": summary_text(label, band, p, sc["delta_logit"]),
        "lenses": [{"key": k, **LENSES[k], "weight": sc["gates"][k], "delta_logit": sc["delta_logit"][k]}
                   for k in LENSES],
        "scan": {"windows": sc["windows"], "size": size, "max_p_ai": sc["max_p_ai"], "share_ai": sc["share_ai"]},
        "source": sc["source"],
        "image": info, "metadata": meta, "evidence": evidence.for_score(p),
        "model_version": engine.version,
    }
    if with_visuals:
        x = engine.to_tensor(focus).unsqueeze(0)
        cam, p_spatial = engine.gradcam(x)
        result["visuals"] = {
            "original": visuals.jpeg_url(visuals.thumbnail(full, 720)),
            "crop": visuals.png_url(focus),
            "gradcam": visuals.png_url(visuals.gradcam_overlay(focus, cam)),
            "spectrum": visuals.png_url(visuals.spectrum(focus)),
            "noise": visuals.png_url(visuals.noise_residual(focus)),
            "ela": visuals.png_url(visuals.ela(focus)),
        }
        result["spatial_expert_p_ai"] = p_spatial
        result["histograms"] = visuals.histograms(focus)
    result["inference_ms"] = (time.perf_counter() - t0) * 1000
    return result
