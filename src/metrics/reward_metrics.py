"""Aggregate reward statistics for evaluation."""
from __future__ import annotations

from src.rewards.reward_fn import compute_reward


def avg_reward(prompts: list[str], completions: list[str], sources: list[str], variant: str) -> float:
    if not completions:
        return 0.0
    total = 0.0
    for i, c in enumerate(completions):
        p = prompts[i] if i < len(prompts) else ""
        s = sources[i] if i < len(sources) else ""
        total += compute_reward(variant, p, c, s)
    return total / len(completions)
