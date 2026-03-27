"""Assemble Part V / E2 main English report from sparse + optional dense dynamics CSV."""
from __future__ import annotations

import csv

from src.utils.path_utils import find_project_root, resolve_path


def _f(x: str) -> float:
    return float(x) if x and str(x).strip() else float("nan")


def main() -> None:
    root = find_project_root()
    csv_path = resolve_path("outputs/report_assets/reward_hacking_dynamics_metrics.csv", root)
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    by = {r["run_id"]: r for r in rows}
    sft = by["step0_sft_best"]
    v1_700, v1_f = by["grpo_v1_checkpoint_700"], by["grpo_v1_final"]
    v4_700, v4_f = by["grpo_v4_checkpoint_700"], by["grpo_v4_final"]

    dense_path = resolve_path("outputs/report_assets/reward_hacking_dynamics_dense_metrics.csv", root)
    dense_block = ""
    if dense_path.is_file():
        drows = list(csv.DictReader(open(dense_path, newline="", encoding="utf-8")))
        dsft = next(r for r in drows if r["model_family"] == "SFT")
        dv1 = [r for r in drows if r["model_family"] == "GRPO-V1-dense"]
        dv1.sort(key=lambda r: int(r["checkpoint_step"]))
        tbl_lines = [
            "| GRPO-V1 dense | {step} | {rl:.4f} | {lo:.3f} | {st:.3f} | {tok:.1f} | {rw:.4f} |".format(
                step=int(r["checkpoint_step"]),
                rl=_f(r["rouge_l_f1"]),
                lo=_f(r["format_rate_loose"]),
                st=_f(r["strict_format_rate"]),
                tok=_f(r["avg_output_length_tokens"]),
                rw=_f(r["avg_reward"]),
            )
            for r in dv1
        ]
        dense_block = f"""
## 3b. Dense V1 dynamics (E2 supplement — **not** the official full V1 run)

To obtain **fine-grained** test curves, we added a **short** GRPO-V1 run (`configs/grpo_v1_e2_dense.yaml`):
**300** training steps, `save_steps=50`, from **SFT best**, same data and **v1** reward as the mainline.
Checkpoints live under `outputs/checkpoints/grpo_v1_e2_dense/`. This run **does not replace** `grpo_v1_3090` for Part III tables.

**Table** (`reward_hacking_dynamics_dense_metrics.csv`):

| Run | Step | ROUGE-L | loose | strict | len(tok) | avg_reward |
|-----|-----:|--------:|------:|-------:|---------:|-----------|
| SFT best | 0 | {_f(dsft['rouge_l_f1']):.4f} | {_f(dsft['format_rate_loose']):.3f} | {_f(dsft['strict_format_rate']):.3f} | {_f(dsft['avg_output_length_tokens']):.1f} | — |
{chr(10).join(tbl_lines)}

**Figures:** `outputs/report_assets/plots_dense/*_dense.png` · guide: `reward_hacking_dynamics_dense_summary.md`  
**Onset (dense):** `reward_hacking_onset_dense_analysis.md`  
**Early stop (dense):** `grpo_v1_earlystop_dense_vs_final.*`, `grpo_v1_earlystop_dense_qualitative.md`

**V4** dynamics in this report remain based on the **sparse** full-run checkpoints (700 / 723) unless a separate dense V4 run is added.
"""
    else:
        dense_block = """
## 3b. Dense V1 dynamics (optional supplement)

If `reward_hacking_dynamics_dense_metrics.csv` is missing: train with `bash scripts/run_grpo_v1_e2_dense.sh`, then
`python -m src.evaluation.evaluate_checkpoint_series --preset dense`, regenerate dense plots and exports, and re-run this report generator.
"""

    body = f"""# Part V / E2 — Reward hacking dynamics (English mainline)

## 1. Motivation

Reinforcement learning with **proxy rewards** can improve a metric while hurting **faithfulness**
or **template discipline**. We report **(A)** the original **sparse** checkpoint evaluation on the
full GRPO-V1/V4 runs, and **(B)** an optional **dense** short GRPO-V1 run to localize **onset** and
**early stopping** with finer step resolution.

## 2. Experimental setup

- **Data:** English-only pipeline; **test** split `grpo_test.jsonl` (**81** examples), identical
  prompts to `sft_test.jsonl` (verified line-wise).
- **Decoding:** greedy, `configs/base.yaml` + `configs/inference.yaml` (plain prompt, no chat template).
- **Sparse checkpoints (mainline):** SFT `best`; official GRPO **V1/V4** under `grpo_*_3090/` with
  **`checkpoint-700`** and **`best`** (step **723**). Used for **Part III/IV** alignment with published runs.
- **Dense checkpoints (supplement):** `grpo_v1_e2_dense` — short V1 run, **v1 reward unchanged**, see §3b.
- **Metrics:** `avg_reward` (variant per stage), **ROUGE-L F1** vs `summary_en_chosen`, **loose** and **strict**
  format rates, **mean output length** (tokenizer from each checkpoint).

**Sparse artifacts:** `reward_hacking_dynamics_metrics.csv`, `outputs/report_assets/plots/`,
`reward_hacking_onset_analysis.md`, `grpo_v1_earlystop_vs_final.*`, `grpo_v1_earlystop_qualitative.md`,
`v1_vs_v4_dynamics_analysis.md`.

{dense_block}

## 3. Dynamics results — sparse (summary table)

| Run | Step | ROUGE-L | loose | strict | len(tok) | avg_reward |
|-----|-----:|--------:|------:|-------:|---------:|-----------|
| SFT best | 0 | {_f(sft['rouge_l_f1']):.4f} | {_f(sft['format_rate_loose']):.3f} | {_f(sft['strict_format_rate']):.3f} | {_f(sft['avg_output_length_tokens']):.1f} | — |
| GRPO-V1 | 700 | {_f(v1_700['rouge_l_f1']):.4f} | {_f(v1_700['format_rate_loose']):.3f} | {_f(v1_700['strict_format_rate']):.3f} | {_f(v1_700['avg_output_length_tokens']):.1f} | {_f(v1_700['avg_reward']):.4f} |
| GRPO-V1 | 723 | {_f(v1_f['rouge_l_f1']):.4f} | {_f(v1_f['format_rate_loose']):.3f} | {_f(v1_f['strict_format_rate']):.3f} | {_f(v1_f['avg_output_length_tokens']):.1f} | {_f(v1_f['avg_reward']):.4f} |
| GRPO-V4 | 700 | {_f(v4_700['rouge_l_f1']):.4f} | {_f(v4_700['format_rate_loose']):.3f} | {_f(v4_700['strict_format_rate']):.3f} | {_f(v4_700['avg_output_length_tokens']):.1f} | {_f(v4_700['avg_reward']):.4f} |
| GRPO-V4 | 723 | {_f(v4_f['rouge_l_f1']):.4f} | {_f(v4_f['format_rate_loose']):.3f} | {_f(v4_f['strict_format_rate']):.3f} | {_f(v4_f['avg_output_length_tokens']):.1f} | {_f(v4_f['avg_reward']):.4f} |

**Figures (sparse):** `outputs/report_assets/plots/` — see `reward_hacking_dynamics_summary.md`.

## 4. Onset of hacking

- **Sparse:** `reward_hacking_onset_analysis.md` — limited to **700→723** on test; combines per-example proxies and trainer logs.
- **Dense (if available):** `reward_hacking_onset_dense_analysis.md` — step-wise table and **finer** onset narrative for **V1** only.

## 5. Early stopping

- **Sparse proxy:** `grpo_v1_earlystop_vs_final.*` (V1 **700** vs **723** on the **official** run).
- **Dense (if available):** `grpo_v1_earlystop_dense_*` — auto-picked **early** checkpoint vs **dense final** (step 300).

## 6. V1 vs V4 (reward design)

See `v1_vs_v4_dynamics_analysis.md`. **V4** evidence here remains **sparse-checkpoint**; the dense run **does not retrain V4**.

## 7. Key findings

- **V1** tag-only reward can **saturate** while quality metrics move slowly; **dense** curves make **saturation timing** and **decoupling** easier to see on the report page.
- **V4** adds **richer constraints**; compare using **sparse** endpoints and qualitative cases from Part III.
- **Dense E2** strengthens **onset / early-stop** claims for **V1** without changing the **official** V1/V4 checkpoint paths used elsewhere.

## 8. Limitations

- **n=81** test split; **single seed** in the reported runs.
- **Dense V1** is **shorter** than the full 723-step run — use it for **dynamics**, not as the sole “best” V1 model.
- **avg_reward** ≠ human judgment; **strict** format ≠ training objective for V1.

---
*Generated by `python -m src.evaluation.export_partv_e2_report`.*
"""
    out = resolve_path("outputs/report_assets/partv_e2_reward_hacking_dynamics.md", root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
