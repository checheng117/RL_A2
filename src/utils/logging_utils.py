"""Dual stdout + file logging for training scripts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO


class _TeeStream(TextIO):
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            n = st.write(s)
        return n

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def setup_run_logging(log_path: str | Path, level: int = logging.INFO) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    tee = _TeeStream(sys.stdout, log_file)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(tee)],
        force=True,
    )
    return logging.getLogger("rlhw2")


def attach_file_logger(logger: logging.Logger, log_path: str | Path) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(fh)
