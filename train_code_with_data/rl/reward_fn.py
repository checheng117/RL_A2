# rl/reward_fn.py
"""
Reward Functions for GRPO Training.

Each version adds stricter requirements on response quality.
Work through v1 → v4 in order and observe:
  - What shortcuts (hacks) does the model find at each level?
  - What additional signal is needed to close each loophole?

Expected response format:
  [point]  <core viewpoint, 10-60 chars>
  [reason] <numbered list of supporting reasons>
  [summary] <conclusion that synthesizes the reasons>
"""

import re
import jieba


# Structural tags the model must produce in order
KEYWORDS = ["[point]", "[reason]", "[summary]"]


# ============================================================
# V1 — Structural keyword presence
#
# Signal : checks whether all three tags appear anywhere in the response.
# Reward : (number of tags found) / 3  ∈ {0.0, 0.33, 0.67, 1.0}
#
# Expected hack:
#   Model outputs "[point][reason][summary]" followed by gibberish,
#   or repeats the tags multiple times to guarantee a hit.
# ============================================================
def reward_v1(response: str, reference: str, original_answer: str, **kwargs) -> float:
    hit = sum(1 for kw in KEYWORDS if kw in response)
    return hit / len(KEYWORDS)


# ============================================================
# V2 — Keyword presence + correct order
#
# Signal : all three tags must appear AND in the correct sequence.
# Reward : 1.0 if ordered correctly, 0.2 if tags exist but out of order, 0.0 if any tag missing.
#
# Expected hack:
#   Model learns the correct order but leaves each section empty or
#   fills it with a single character to minimise generation cost.
# ============================================================
def reward_v2(response: str, reference: str, original_answer: str, **kwargs) -> float:
    positions = []
    for kw in KEYWORDS:
        idx = response.find(kw)
        if idx == -1:
            return 0.0                      # missing tag → zero reward
        positions.append(idx)

    is_ordered = all(
        positions[i] < positions[i + 1]
        for i in range(len(positions) - 1)
    )
    return 1.0 if is_ordered else 0.2


# ============================================================
# V3 — Correct structure + substantive content in each section
#
# Signal : regex extracts the three sections; each is scored independently.
#   [point]   must be 10–60 characters (concise but non-trivial)
#   [reason]  must contain a numbered list  (1. / 1、/ 1．)
#   [summary] must not be near-identical to [point] (no copy-paste)
#
# Final reward: mean of the three section scores ∈ [0.0, 1.0]
#
# Expected hack:
#   Model writes a valid numbered list in [reason] but the items are
#   meaningless, or [summary] slightly paraphrases [point] to stay
#   just below the similarity threshold.
# ============================================================
def reward_v3(response: str, reference: str, original_answer: str, **kwargs) -> float:
    pattern = r'\[point\](.+?)\[reason\](.+?)\[summary\](.+?)$'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return 0.0

    point, reason, summary = [g.strip() for g in match.groups()]
    scores = []

    # ── [point]: length check ────────────────────────────────────────
    # Too short → vague; too long → not a concise viewpoint
    num_point_words = len(point.split())
    point_score = 1.0 if 5 < num_point_words < 100 else 0.3
    scores.append(point_score)

    # ── [reason]: numbered list check ───────────────────────────────
    # Accepts formats: "1." / "1、" / "1．"
    has_numbered_list = bool(re.search(r'[1-3][\.、．]', reason))
    reason_score = 1.0 if has_numbered_list else 0.3
    scores.append(reason_score)

    # ── [summary]: novelty check (character-level Jaccard) ──────────
    # Penalises summaries that are near-copies of the point
    point_chars   = set(point)
    summary_chars = set(summary)
    jaccard = len(point_chars & summary_chars) / \
              max(len(point_chars | summary_chars), 1)
    summary_score = 1.0 if jaccard < 0.8 else 0.2
    scores.append(summary_score)

    return sum(scores) / len(scores)


# ============================================================
# V4 — V3 + keyword coverage of the reference answer
#
# Signal : combines V3 format score with a content-relevance score.
#   Content score measures how many content words from the reference
#   answer appear in the model's response (after removing stop words).
#
# Final reward: 0.5 × format_score + 0.5 × keyword_score
#
# Expected hack:
#   Model produces a well-formatted response that is completely
#   unrelated to the reference, or stuffs reference keywords into
#   [point] without integrating them meaningfully.
# ============================================================

# Common Chinese stop words to exclude from keyword matching
_STOP_WORDS = {
    '的', '了', '是', '在', '我', '有', '和', '就',
    '不', '人', '都', '一', '这', '那', '也', '但', '而', '与',
}

def reward_v4(response: str, reference: str, original_answer: str, **kwargs) -> float:
    format_score = reward_v3(response, reference, original_answer, **kwargs)

    # Skip expensive jieba segmentation if format is already poor
    if format_score < 0.5:
        return format_score

    original_words = set(jieba.cut(reference)) - _STOP_WORDS
    response_words = set(jieba.cut(response))         - _STOP_WORDS

    if not original_words:
        keyword_score = 0.0
    else:
        overlap = len(original_words & response_words)
        # Require coverage of at least 30% of reference keywords for full score
        keyword_score = min(overlap / (len(original_words) * 0.3), 1.0)

    return 0.5 * format_score + 0.5 * keyword_score


# ============================================================
# V5 — [Student Implementation] Your improved reward function
#
# Hints to guide your design:
#   1. Does [reason] contain EXACTLY 3 numbered items?
#      (v3 only checks for the presence of a list, not its length)
#   2. Does [summary] genuinely synthesise the reasons,
#      rather than just repeating [point]?
#   3. Can you use sentence-transformers to compute semantic similarity
#      between [summary] and the reference answer?
#   4. Can you penalise responses that are suspiciously short overall?
#
# Your reward should return a float in [0.0, 1.0].
# ============================================================
def reward_v5(response: str, reference: str, **kwargs) -> float:
    """
    TODO: Design your own composite reward function.

    Suggested structure:
        score = w1 * format_score      # build on reward_v3 or reward_v4
              + w2 * length_score      # penalise too-short responses
              + w3 * semantic_score    # sentence-transformer similarity
        return min(score, 1.0)
    """
    raise NotImplementedError("Implement reward_v5 in reward_fn.py!")
