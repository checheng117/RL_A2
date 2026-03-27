"""GRPO reward variants V1–V5 (V5 stub)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Sequence

POINT = "[point]"
REASON = "[reason]"
SUMMARY = "[summary]"


def _tags_present(text: str) -> tuple[bool, bool, bool]:
    t = text.lower()
    return POINT in t, REASON in t, SUMMARY in t


def _order_ok(text: str) -> bool:
    lp = text.lower().find(POINT)
    lr = text.lower().find(REASON)
    ls = text.lower().find(SUMMARY)
    if lp < 0 or lr < 0 or ls < 0:
        return False
    return lp < lr < ls


_NUMBERED_IN_REASON = re.compile(r"(?<!\d)\d+\.\s+")


def _reason_numbered(text: str) -> bool:
    """At least two `1.`/`2.`-style markers inside [reason] (same line OK; matches strict eval heuristics)."""
    m = re.search(r"\[reason\](.*?)(\[summary\]|$)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return False
    body = m.group(1)
    return len(_NUMBERED_IN_REASON.findall(body)) >= 2


def _repetition_penalty(text: str) -> float:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 6:
        return 0.0
    ctr = Counter(words)
    dup_ratio = 1.0 - (len(set(words)) / max(len(words), 1))
    return max(0.0, min(1.0, dup_ratio))


def _length_score(text: str, lo: int = 40, hi: int = 800) -> float:
    n = len(text.strip())
    if n < lo:
        return n / max(lo, 1)
    if n > hi:
        return max(0.2, hi / n)
    return 1.0


def _token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-zA-Z]{3,}", a.lower()))
    tb = set(re.findall(r"[a-zA-Z]{3,}", b.lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(union, 1)


def reward_v1(_prompt: str, completion: str, _source: str) -> float:
    p, r, s = _tags_present(completion)
    return float(p + r + s) / 3.0


def reward_v2(prompt: str, completion: str, source: str) -> float:
    base = reward_v1(prompt, completion, source)
    return base * (1.0 if _order_ok(completion) else 0.3)


def reward_v3(prompt: str, completion: str, source: str) -> float:
    base = reward_v2(prompt, completion, source)
    if base < 0.2:
        return base
    num_ok = 1.0 if _reason_numbered(completion) else 0.4
    len_s = _length_score(completion)
    rep = _repetition_penalty(completion)
    return max(0.0, base * num_ok * len_s * (1.0 - 0.7 * rep))


def reward_v4(prompt: str, completion: str, source: str) -> float:
    v3 = reward_v3(prompt, completion, source)
    overlap = _token_jaccard(source, completion)
    return max(0.0, 0.5 * v3 + 0.5 * overlap)


def reward_v5_stub(prompt: str, completion: str, source: str) -> float:
    """Placeholder: V4 plus mild preference for mid-length outputs."""
    v4 = reward_v4(prompt, completion, source)
    mid = 1.0 if 80 <= len(completion) <= 600 else 0.85
    return max(0.0, v4 * mid)


_VARIANTS: dict[str, Callable[[str, str, str], float]] = {
    "v1": reward_v1,
    "v2": reward_v2,
    "v3": reward_v3,
    "v4": reward_v4,
    "v5": reward_v5_stub,
}


def _normalize_variant(variant: str) -> str:
    v = (variant or "v1").lower().strip()
    if v.isdigit():
        return f"v{v}"
    if not v.startswith("v"):
        return f"v{v}"
    return v


def compute_reward(variant: str, prompt: str, completion: str, source_answer: str) -> float:
    key = _normalize_variant(variant)
    fn = _VARIANTS.get(key)
    if fn is None:
        raise ValueError(f"Unknown reward variant: {variant} (resolved {key})")
    return float(fn(prompt, completion, source_answer))


def make_trl_reward_fn(variant: str) -> Callable[..., list[float]]:
    """Build a reward callable for TRL GRPOTrainer (returns list[float])."""

    def reward_func(completions: Sequence[str], prompts: Sequence[str] | None = None, **kwargs: Any) -> list[float]:
        comps = list(completions)
        n = len(comps)
        prs = list(prompts) if prompts is not None else [""] * n
        if len(prs) < n:
            prs.extend([""] * (n - len(prs)))
        srcs = kwargs.get("answer_en")
        if srcs is None:
            src_list = [""] * n
        elif isinstance(srcs, str):
            src_list = [srcs] * n
        else:
            src_list = list(srcs)
        if len(src_list) < n:
            src_list.extend([""] * (n - len(src_list)))
        return [compute_reward(variant, prs[i], comps[i], src_list[i]) for i in range(n)]

    return reward_func
