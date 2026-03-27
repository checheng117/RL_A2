"""Strict format: exactly one [point]/[reason]/[summary], correct order, non-empty, 2+ numbered reasons."""
from __future__ import annotations

import re

POINT = "[point]"
REASON = "[reason]"
SUMMARY = "[summary]"

_TAG_POINT = re.compile(re.escape(POINT), re.IGNORECASE)
_TAG_REASON = re.compile(re.escape(REASON), re.IGNORECASE)
_TAG_SUMMARY = re.compile(re.escape(SUMMARY), re.IGNORECASE)
# Single occurrence of each tag in order; bodies non-empty; summary to end.
_STRUCTURE = re.compile(
    r"\[point\]\s*(.+?)\s*\[reason\]\s*(.+?)\s*\[summary\]\s*(.+)$",
    re.DOTALL | re.IGNORECASE,
)
# At least two "1. / 2. / …" style markers (same line or multiple lines; avoids matching "4.5").
_NUMBERED_MARKER = re.compile(r"(?<!\d)\d+\.\s+")


def strict_format_adherence_one(text: str) -> dict[str, bool | str]:
    """Return pass_strict and a short failure code for debugging."""
    if not text or not text.strip():
        return {"pass_strict": False, "strict_fail_code": "empty"}

    p_n = len(_TAG_POINT.findall(text))
    r_n = len(_TAG_REASON.findall(text))
    s_n = len(_TAG_SUMMARY.findall(text))
    if p_n != 1 or r_n != 1 or s_n != 1:
        return {"pass_strict": False, "strict_fail_code": "tag_count"}

    tl = text.lower()
    lp, lr, ls = tl.find(POINT), tl.find(REASON), tl.find(SUMMARY)
    if lp < 0 or lr < 0 or ls < 0 or not (lp < lr < ls):
        return {"pass_strict": False, "strict_fail_code": "order"}

    m = _STRUCTURE.search(text)
    if not m:
        return {"pass_strict": False, "strict_fail_code": "parse"}

    point_body, reason_body, summary_body = m.group(1), m.group(2), m.group(3)
    if not point_body.strip() or not reason_body.strip() or not summary_body.strip():
        return {"pass_strict": False, "strict_fail_code": "empty_section"}

    # ≥2 list markers like `1. ` / `2. ` (same line or multiple lines; see reward_fn heuristics).
    if len(_NUMBERED_MARKER.findall(reason_body)) < 2:
        return {"pass_strict": False, "strict_fail_code": "reason_numbered"}

    return {"pass_strict": True, "strict_fail_code": "ok"}


def batch_strict_format_adherence(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {"strict_format_rate": 0.0, "strict_pass_count": 0}
    passed = 0
    for t in texts:
        if strict_format_adherence_one(t)["pass_strict"]:
            passed += 1
    n = len(texts)
    return {"strict_format_rate": passed / n, "strict_pass_count": passed}
