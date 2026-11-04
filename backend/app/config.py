"""App settings. Every path is inside the repository; limits can be overridden by environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    model_version: str = os.environ.get("PA_MODEL_VERSION", "fp-v4.1")
    device: str = os.environ.get("PA_DEVICE", "auto")                      # auto | cuda | cpu
    max_image_mb: float = _env("PA_MAX_IMAGE_MB", 25)                       # one image
    max_image_pixels: int = int(_env("PA_MAX_IMAGE_PIXELS", 40_000_000))
    max_direct_images: int = int(_env("PA_MAX_DIRECT_IMAGES", 12))           # /api/analyze, full visuals
    max_batch_images: int = int(_env("PA_MAX_BATCH_IMAGES", 5000))           # folder / zip job
    max_batch_mb: float = _env("PA_MAX_BATCH_MB", 2048)                     # upload or unzipped total
    job_ttl_hours: float = _env("PA_JOB_TTL_HOURS", 6)
    work_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "PA_WORK_DIR", REPO / "results" / "app_jobs")))

    @property
    def model_dir(self) -> Path:
        return REPO / "models" / self.model_version

    @property
    def attribution_dir(self) -> Path:
        """The attribution head reads the detector's features, so each detector has its own."""
        return REPO / "models" / {"fp-v4.1": "attribution-v1"}.get(self.model_version, "attribution-v2")

    @property
    def frontend_dist(self) -> Path:
        return REPO / "frontend" / "dist"
