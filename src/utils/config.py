"""
Centralised config loader.

Usage:
    from src.utils.config import load_config, get_path

    cfg = load_config()
    meta_path = get_path(cfg, "meta_clean")   # returns pathlib.Path
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels above this file: src/utils/config.py → root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and return the YAML config as a plain dict.

    The result is cached so repeated calls are free.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve cfg['paths'][key] to an absolute Path under PROJECT_ROOT."""
    raw = cfg["paths"][key]
    p = PROJECT_ROOT / raw
    return p


def get_embedding_dir(cfg: dict[str, Any]) -> Path:
    """Return the embedding directory for the active pipeline version.

    Reads pipeline.active_version from cfg; defaults to 'v3'.
    """
    version = cfg.get("pipeline", {}).get("active_version", "v3")
    if version == "v2":
        rel = cfg["pipeline"]["v2_embedding_dir"]
    else:
        rel = cfg["pipeline"]["v3_embedding_dir"]
    root = Path(__file__).resolve().parents[2]
    return root / rel


def get_embedding_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve cfg['embeddings'][key] to an absolute Path in the active version dir."""
    filename = cfg["embeddings"][key]
    return get_embedding_dir(cfg) / filename


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create all output directories that are listed in cfg['paths'] if they do not exist."""
    dir_keys = [
        "data_processed",
        "data_embeddings",
        "outputs_charts",
        "outputs_results",
    ]
    for key in dir_keys:
        get_path(cfg, key).mkdir(parents=True, exist_ok=True)
