"""English plot guide for E2 dense dynamics."""
from __future__ import annotations

import csv

from src.utils.path_utils import find_project_root, resolve_path


def main() -> None:
    root = find_project_root()
    out = resolve_path("outputs/report_assets/reward_hacking_dynamics_dense_summary.md", root)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = resolve_path("outputs/report_assets/reward_hacking_dynamics_dense_metrics.csv", root)
    disk_block = ""
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        dense = sorted(
            (r for r in rows if r.get("model_family") == "GRPO-V1-dense"),
            key=lambda r: int(r["checkpoint_step"]),
        )
        if dense:
            steps = ", ".join(str(int(r["checkpoint_step"])) for r in dense)
            disk_block = f"""
## Checkpoints actually evaluated (disk reality)

The curves and tables use **only** the rows present in `reward_hacking_dynamics_dense_metrics.csv`.
**GRPO-V1-dense** test metrics were computed at optimizer steps **{steps}** (plus **SFT step 0**).
There are **no** interpolated points between saved checkpoints.

"""
    text = """# E2 dense dynamics — figure guide

Figures: `outputs/report_assets/plots_dense/*_dense.png`, built from
`reward_hacking_dynamics_dense_metrics.csv` (same **test** split and **greedy** decoding as the sparse E2 run).
"""
    text = text.rstrip() + "\n" + disk_block + """
## Main curve

**GRPO-V1 (E2-dense)** is a **short** auxiliary training run (`configs/grpo_v1_e2_dense.yaml`, **300** optimizer steps,
`save_steps=50`) from **SFT best**. It does **not** replace the official full **grpo_v1_3090** result.

## `reward_vs_step_dense.png`

Test **avg_reward** under **v1** scoring along dense steps. Saturation near **1.0** indicates the proxy is **easy to max**
on this split; compare with ROUGE/strict/length panels for **decoupling**.

## `rouge_vs_step_dense.png`

**ROUGE-L** vs `summary_en_chosen`; **gray line** = SFT baseline. **Orange squares** (if present) = **GRPO-V4** from the
**sparse** main E2 CSV (different run, for qualitative scale only).

## `strict_format_vs_step_dense.png`

**Strict** template compliance rate. Drops or plateaus below SFT while reward is high suggest **structural shortcuts**.

## `output_length_vs_step_dense.png`

Mean **output tokens** per checkpoint tokenizer. Upward drift with **flat/falling ROUGE** is a **verbosity / repetition** warning.

## `combined_dynamics_summary_dense.png`

Four-panel overview for the report appendix.
"""
    out.write_text(text.strip() + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
