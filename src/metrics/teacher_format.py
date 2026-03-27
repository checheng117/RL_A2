"""Format check aligned with train_code_with_data/eval/evaluate.py."""
from __future__ import annotations

import re

_FORMAT_PATTERN: re.Pattern = re.compile(
    r"\[point\]\s*(.+?)\s*\[reason\]\s*(.+?)\s*\[summary\]\s*(.+?)$",
    re.DOTALL,
)


def check_format(response: str) -> bool:
    match = _FORMAT_PATTERN.search(response)
    if not match:
        return False
    return all(group.strip() for group in match.groups())


def check_format_strict(response: str) -> bool:
    """Exactly one of each section, order, non-empty, 2+ numbered reasons in [reason]."""
    from src.metrics.strict_format_adherence import strict_format_adherence_one

    return bool(strict_format_adherence_one(response)["pass_strict"])
