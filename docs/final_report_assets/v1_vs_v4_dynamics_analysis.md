# GRPO-V1 vs GRPO-V4 — checkpoint dynamics (test split)

Setup: **same** `grpo_test.jsonl` (81 rows), **greedy** decoding, tokenizer length from each checkpoint.
On-disk GRPO checkpoints are **700** and **final (723)** only; curves are **sparse**.

## Reward dynamics

- **V1** optimizes a **tag-heavy** scalar that can **saturate** near 1.0 in training logs; small test-window changes in avg_reward may hide quality shifts.
- **V1** test avg_reward: **1.0000 → 1.0000** (700 → final).
- **V4** adds ordering, numbered reasons, overlap, and length terms; train reward stays **below saturation** in logs. Test avg_reward: **0.4990 → 0.4992**.
- **Comparison:** V4’s reward scale and gradients reflect **more objectives**; improvements in reward are **less trivially gameable** than V1’s “tags present” signal.

## ROUGE-L dynamics

- **V1:** 0.4428 → 0.4396.
- **V4:** 0.4480 → 0.4495.
- If either run shows **reward flat/up** while ROUGE **falls** in the 700→723 window, treat it as a **hacking warning** on this split (not a universal law).

## Strict format dynamics

- **V1** strict rate: 0.988 → 0.988.
- **V4** strict rate: 1.000 → 1.000.
- **V1** is not trained on strict compliance; **high loose / lower strict** is the classic repetition pattern. **V4** partially aligns optimization with structure, so strict tends to be **more stable**, though not guaranteed.

## Output length dynamics

- **V1** mean tokens: 123.6 → 123.6.
- **V4** mean tokens: 118.7 → 119.5.
- Large upward moves with **flat ROUGE** suggest verbosity or template cycling.

## Why V1 is more hacking-prone

1. **Objective mismatch:** V1 largely rewards **surface tags**; the model can **repeat** valid-looking blocks and still score well under loose format.
2. **Saturated signal:** When train reward sits near **1.0**, GRPO has **little gradient** to discourage subtle bad habits on held-out quality.
3. **Strict metric gap:** What we report as “good format” in Part I/II (loose) **under-detects** multi-block outputs.

## Why V4 mitigates but need not solve hacking

1. **Richer reward** penalizes **order**, **reason numbering**, **length/repetition**, and **source overlap** — many failure modes of V1 cost points.
2. **Trade-offs:** overlap and heuristics can still be **gamed** (e.g., copying source phrases) without faithful summarization.
3. **Evaluation vs training:** strict format in the report is **not identical** to the reward’s internal checks; residual gaps remain.

---

*Auto-filled numeric deltas from `reward_hacking_dynamics_metrics.csv` via `python -m src.evaluation.export_v1_vs_v4_dynamics_analysis`.*