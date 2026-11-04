"""Resolve project paths from configs/paths.toml relative to the single workspace root."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get("PIXEL_ARTIFACTS_WORKSPACE", REPO_ROOT.parent)).resolve()
CONFIG_PATH = REPO_ROOT / "configs" / "paths.toml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else WORKSPACE_ROOT / p


def section(name: str) -> dict:
    """Return a config section with every *path-like* string value resolved to a Path."""
    out = {}
    for k, v in load_config()[name].items():
        out[k] = v if (k.endswith("sha256") or k.endswith("_volume")) else resolve(v)
    return out
