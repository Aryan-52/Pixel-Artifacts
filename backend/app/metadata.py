"""File metadata clues, independent of the model.

Metadata is easy to strip or fake, so a clue here is supporting context only: its presence can
point to a tool, its absence proves nothing.
"""
from __future__ import annotations

from PIL import ExifTags, Image

# Text that generators and editing tools commonly write into files.
AI_MARKERS = {
    "stable diffusion": "Stable Diffusion", "sd-webui": "Stable Diffusion WebUI", "comfyui": "ComfyUI",
    "midjourney": "Midjourney", "dall-e": "DALL-E", "dall·e": "DALL-E", "openai": "OpenAI",
    "firefly": "Adobe Firefly", "imagen": "Google Imagen", "novelai": "NovelAI", "invokeai": "InvokeAI",
    "leonardo.ai": "Leonardo.Ai", "c2pa": "C2PA content credentials",
    "trainedalgorithmicmedia": "IPTC 'trained algorithmic media' tag",
}
PROMPT_KEYS = {"parameters", "prompt", "workflow", "negative_prompt", "sd-metadata", "dream"}
EXIF_KEEP = {"Make", "Model", "Software", "DateTimeOriginal", "DateTime", "LensModel", "Artist",
             "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "ImageDescription"}


def _text(v) -> str:
    if isinstance(v, bytes):
        v = v[:400].decode("utf-8", "replace")
    s = str(v)
    return s if len(s) <= 300 else s[:300] + "..."


def inspect(img: Image.Image, fmt: str | None) -> dict:
    exif = {}
    try:
        raw = img.getexif()
        for k, v in raw.items():
            name = ExifTags.TAGS.get(k, str(k))
            if name in EXIF_KEEP:
                exif[name] = _text(v)
        for k, v in raw.get_ifd(0x8769).items():          # Exif sub-IFD: camera settings
            name = ExifTags.TAGS.get(k, str(k))
            if name in EXIF_KEEP:
                exif[name] = _text(v)
    except Exception:
        pass

    chunks = {str(k): _text(v) for k, v in (img.info or {}).items()
              if isinstance(v, (str, bytes)) and k not in ("icc_profile", "exif")}
    blob = " ".join([*exif.values(), *chunks.keys(), *chunks.values()]).lower()
    found = sorted({label for key, label in AI_MARKERS.items() if key in blob})
    prompt = sorted(k for k in chunks if k.lower() in PROMPT_KEYS)
    if prompt:
        found.append("generation prompt stored in file (" + ", ".join(prompt) + ")")

    camera = bool(exif.get("Make") or exif.get("Model"))
    if found:
        note = "The file carries text written by an AI generation tool. Treat this as a strong hint."
    elif camera:
        note = "Camera details are present. They can be copied or edited, so this is weak evidence."
    else:
        note = "No camera or AI-tool metadata. Common for screenshots, web images and generated images alike."
    return {"format": fmt, "mode": img.mode, "exif": exif, "text_chunks": chunks,
            "ai_markers": found, "camera_info": camera, "note": note}
