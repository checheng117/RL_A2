"""YAML merge, CLI overrides, and shared training utilities."""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any

import yaml

from src.utils.path_utils import find_project_root, resolve_path

logger = logging.getLogger(__name__)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config(config_paths: list[str], root: Path | None = None) -> dict[str, Any]:
    r = root or find_project_root()
    merged: dict[str, Any] = {}
    for p in config_paths:
        path = resolve_path(p, r)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        merged = deep_merge(merged, data)
    proj = merged.get("project", {}) or {}
    if merged.get("paths", {}).get("project_root") is None:
        merged.setdefault("paths", {})["project_root"] = str(r)
    merged["project"] = proj
    return merged


def apply_dot_overrides(cfg: dict[str, Any], pairs: list[str]) -> dict[str, Any]:
    for raw in pairs:
        if "=" not in raw:
            logger.warning("Skip override (expected key=value): %s", raw)
            continue
        key, val = raw.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            parsed: Any = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            parsed = val
        parts = key.split(".")
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = parsed
    return cfg


def output_checkpoint_dir(cfg: dict[str, Any], root: Path | None = None) -> Path:
    r = root or find_project_root()
    paths = cfg.get("paths", {}) or {}
    ckpt = paths.get("checkpoints_dir", "outputs/checkpoints")
    sub = (cfg.get("training", {}) or {}).get("output_subdir", "run")
    return resolve_path(str(Path(ckpt) / sub), r)


def log_file_path(cfg: dict[str, Any], name: str, root: Path | None = None) -> Path:
    r = root or find_project_root()
    logs = (cfg.get("paths", {}) or {}).get("logs_dir", "outputs/logs")
    sub = (cfg.get("training", {}) or {}).get("output_subdir", "run")
    return resolve_path(str(Path(logs) / sub / name), r)
