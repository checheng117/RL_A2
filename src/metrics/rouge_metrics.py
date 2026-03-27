"""ROUGE-L via rouge-score (CPU-friendly)."""
from __future__ import annotations

from rouge_score import rouge_scorer


def rouge_l_f1(preds: list[str], refs: list[str]) -> float:
    if not preds or not refs or len(preds) != len(refs):
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f1s: list[float] = []
    for p, r in zip(preds, refs, strict=True):
        s = scorer.score(r, p)
        f1s.append(s["rougeL"].fmeasure)
    return sum(f1s) / len(f1s)
