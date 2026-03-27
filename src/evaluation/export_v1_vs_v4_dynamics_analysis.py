"""Generate English dynamics comparison markdown from E2 CSV."""
from __future__ import annotations

import csv
from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def _f(x: str) -> float:
    return float(x) if x and str(x).strip() else float("nan")


def main() -> None:
    root = find_project_root()
    csv_path = resolve_path("outputs/report_assets/reward_hacking_dynamics_metrics.csv", root)
    out = resolve_path("outputs/report_assets/v1_vs_v4_dynamics_analysis.md", root)
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    by = {r["run_id"]: r for r in rows}

    def pair(tag700: str, tagf: str) -> tuple[dict, dict]:
        return by[tag700], by[tagf]

    v1_700, v1_f = pair("grpo_v1_checkpoint_700", "grpo_v1_final")
    v4_700, v4_f = pair("grpo_v4_checkpoint_700", "grpo_v4_final")

    lines = [
        "# GRPO-V1 vs GRPO-V4 — checkpoint dynamics (test split)",
        "",
        "Setup: **same** `grpo_test.jsonl` (81 rows), **greedy** decoding, tokenizer length from each checkpoint.",
        "On-disk GRPO checkpoints are **700** and **final (723)** only; curves are **sparse**.",
        "",
        "## Reward dynamics",
        "",
        "- **V1** optimizes a **tag-heavy** scalar that can **saturate** near 1.0 in training logs; small test-window "
        "changes in avg_reward may hide quality shifts.",
        f"- **V1** test avg_reward: **{_f(v1_700['avg_reward']):.4f} → {_f(v1_f['avg_reward']):.4f}** (700 → final).",
        f"- **V4** adds ordering, numbered reasons, overlap, and length terms; train reward stays **below saturation** "
        f"in logs. Test avg_reward: **{_f(v4_700['avg_reward']):.4f} → {_f(v4_f['avg_reward']):.4f}**.",
        "- **Comparison:** V4’s reward scale and gradients reflect **more objectives**; improvements in reward are "
        "**less trivially gameable** than V1’s “tags present” signal.",
        "",
        "## ROUGE-L dynamics",
        "",
        f"- **V1:** {_f(v1_700['rouge_l_f1']):.4f} → {_f(v1_f['rouge_l_f1']):.4f}.",
        f"- **V4:** {_f(v4_700['rouge_l_f1']):.4f} → {_f(v4_f['rouge_l_f1']):.4f}.",
        "- If either run shows **reward flat/up** while ROUGE **falls** in the 700→723 window, treat it as a "
        "**hacking warning** on this split (not a universal law).",
        "",
        "## Strict format dynamics",
        "",
        f"- **V1** strict rate: {_f(v1_700['strict_format_rate']):.3f} → {_f(v1_f['strict_format_rate']):.3f}.",
        f"- **V4** strict rate: {_f(v4_700['strict_format_rate']):.3f} → {_f(v4_f['strict_format_rate']):.3f}.",
        "- **V1** is not trained on strict compliance; **high loose / lower strict** is the classic repetition pattern. "
        "**V4** partially aligns optimization with structure, so strict tends to be **more stable**, though not guaranteed.",
        "",
        "## Output length dynamics",
        "",
        f"- **V1** mean tokens: {_f(v1_700['avg_output_length_tokens']):.1f} → {_f(v1_f['avg_output_length_tokens']):.1f}.",
        f"- **V4** mean tokens: {_f(v4_700['avg_output_length_tokens']):.1f} → {_f(v4_f['avg_output_length_tokens']):.1f}.",
        "- Large upward moves with **flat ROUGE** suggest verbosity or template cycling.",
        "",
        "## Why V1 is more hacking-prone",
        "",
        "1. **Objective mismatch:** V1 largely rewards **surface tags**; the model can **repeat** valid-looking blocks and "
        "still score well under loose format.",
        "2. **Saturated signal:** When train reward sits near **1.0**, GRPO has **little gradient** to discourage subtle "
        "bad habits on held-out quality.",
        "3. **Strict metric gap:** What we report as “good format” in Part I/II (loose) **under-detects** multi-block outputs.",
        "",
        "## Why V4 mitigates but need not solve hacking",
        "",
        "1. **Richer reward** penalizes **order**, **reason numbering**, **length/repetition**, and **source overlap** — "
        "many failure modes of V1 cost points.",
        "2. **Trade-offs:** overlap and heuristics can still be **gamed** (e.g., copying source phrases) without faithful summarization.",
        "3. **Evaluation vs training:** strict format in the report is **not identical** to the reward’s internal checks; "
        "residual gaps remain.",
        "",
        "---",
        "",
        "*Auto-filled numeric deltas from `reward_hacking_dynamics_metrics.csv` via "
        "`python -m src.evaluation.export_v1_vs_v4_dynamics_analysis`.*",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
