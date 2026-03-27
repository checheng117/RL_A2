"""Path helpers: resolve project root and normalize paths."""
from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from start (or cwd) until a directory containing configs/ is found."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "configs").is_dir() and (p / "src").is_dir():
            return p
    return cur


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    """If path is relative, resolve against project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    r = root or find_project_root()
    return (r / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_or_default(name: str, default: str) -> str:
    return os.environ.get(name, default)
