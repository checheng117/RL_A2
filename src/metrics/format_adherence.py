"""Loose format adherence: tags present + coarse order (legacy / compatibility).

For assignment reporting, prefer `format_adherence_strict` from evaluation output:
duplicate valid-looking blocks can still score 1.0 here. See
`src.metrics.strict_format_adherence`.
"""
from __future__ import annotations

from src.rewards.reward_fn import _order_ok, _tags_present


def format_adherence_score(text: str) -> dict[str, float | bool]:
    p, r, s = _tags_present(text)
    tags = (p + r + s) / 3.0
    ordered = _order_ok(text)
    return {"tag_coverage": float(tags), "ordered_sections": bool(ordered), "score": float(tags * (1.0 if ordered else 0.5))}


def batch_format_adherence(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {"format_rate": 0.0, "avg_tag_coverage": 0.0, "avg_score": 0.0}
    scores = [format_adherence_score(t)["score"] for t in texts]
    cov = [format_adherence_score(t)["tag_coverage"] for t in texts]
    return {
        "format_rate": sum(1 for s in scores if s >= 0.99) / len(scores),
        "avg_tag_coverage": sum(cov) / len(cov),
        "avg_score": sum(scores) / len(scores),
    }
