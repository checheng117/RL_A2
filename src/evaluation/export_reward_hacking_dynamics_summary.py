"""Write a short English guide to E2 plots (report-ready)."""
from __future__ import annotations

from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def main() -> None:
    root = find_project_root()
    out = resolve_path("outputs/report_assets/reward_hacking_dynamics_summary.md", root)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = """# E2 plot guide — reward hacking dynamics

All figures live under `outputs/report_assets/plots/` and are produced from
`outputs/report_assets/reward_hacking_dynamics_metrics.csv` (held-out **test** split,
greedy decoding per `configs/inference.yaml`).

## `reward_vs_step.png`

**GRPO-V1** and **GRPO-V4** test-set **avg_reward** (each scored with its training
reward variant: v1 vs v4). **SFT** is not shown: the policy was not trained with
that scalar. This curve shows how the optimized reward moves during late GRPO when
only checkpoints at steps **700** and **723** are available on disk.

## `rouge_vs_step.png`

**ROUGE-L F1** against `summary_en_chosen` for **SFT best** (horizontal reference at
step 0) and GRPO checkpoints. Divergence between rising reward and flat or falling
ROUGE is a standard hacking warning signal (interpret cautiously on a small split).

## `strict_format_vs_step.png`

**Strict format rate** (exactly one `[point]` / `[reason]` / `[summary]`, order,
non-empty bodies, numbered reasons). Declines alongside high loose format often
indicates template repetition or structural shortcuts.

## `output_length_vs_step.png`

Mean **output length in tokenizer tokens** (tokenizer from each evaluated
checkpoint). Sudden increases can indicate verbosity hacks or runaway templates.

## `combined_dynamics_summary.png`

Four-panel overview for slides or appendix: reward, ROUGE-L, strict format, and
length on a shared step axis, with **SFT** baselines as gray dashed lines where
applicable.
"""
    out.write_text(text.strip() + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
