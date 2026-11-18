"""Tiny config loader: merges every YAML file in configs/ into one dict."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_dir: str | Path | None = None) -> dict:
    config_dir = Path(config_dir) if config_dir else ROOT / "configs"
    cfg: dict = {}
    for f in sorted(config_dir.glob("*.yaml")):
        with open(f, encoding="utf-8") as fh:
            part = yaml.safe_load(fh) or {}
        for k, v in part.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def path(cfg: dict, key: str) -> Path:
    """Resolve a path from the config relative to the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p
