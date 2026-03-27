"""Cleaning and validation for raw rows."""
from __future__ import annotations

from typing import Any


def is_valid_en_row(
    row: dict[str, Any],
    answer_key: str,
    chosen_key: str,
    rejected_key: str | None = None,
    min_len: int = 10,
) -> bool:
    a = (row.get(answer_key) or "").strip()
    c = (row.get(chosen_key) or "").strip()
    if len(a) < min_len or len(c) < min_len:
        return False
    if rejected_key:
        r = (row.get(rejected_key) or "").strip()
        if len(r) < min_len:
            return False
    return True


def is_valid_teacher_row(row: dict[str, Any], lang: str = "zh", min_len: int = 10) -> bool:
    """Exploration / legacy: require answer + chosen + rejected for zh|en (teacher process_data schema)."""
    a = (row.get(f"answer_{lang}") or "").strip()
    c = (row.get(f"summary_{lang}_chosen") or "").strip()
    rj = (row.get(f"summary_{lang}_rejected") or "").strip()
    return len(a) >= min_len and len(c) >= min_len and len(rj) >= min_len


def basic_stats(rows: list[dict[str, Any]], chosen_key: str, rejected_key: str) -> dict[str, Any]:
    lc = [len((r.get(chosen_key) or "")) for r in rows]
    lr = [len((r.get(rejected_key) or "")) for r in rows]
    return {
        "n": len(rows),
        "chosen_len_mean": sum(lc) / len(lc) if lc else 0.0,
        "rejected_len_mean": sum(lr) / len(lr) if lr else 0.0,
    }
