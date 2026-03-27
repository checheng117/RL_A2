# Final report assets (public-facing)

## 1. What is in this folder

This directory is a **curated snapshot** of the assignment’s **final quantitative tables, analyses, qualitative case studies, and key figures**, copied from `outputs/report_assets/` (and its `plots/` / `plots_dense/` subfolders) for **version control and easy review**.

- **Why it exists:** The full `outputs/` tree (checkpoints, logs, raw predictions, all intermediate exports) is intentionally **gitignored** to keep the repository small and to avoid leaking large binaries. This folder is the **public window** onto the results a grader can trust without cloning local artifacts.
- **What was not copied:** Checkpoints, training logs, prediction JSONL files, scratch templates, and redundant tables superseded by the unified Part III/IV export. See `appendix/` for numeric backups and sparse early-stop / dynamics tables.

**Renamed files (same bytes as source; only the filename is clearer for readers):**

| File in `plots/` | Original path under `outputs/report_assets/` |
|------------------|-----------------------------------------------|
| `sft_training_loss_curves.png` | `sft_loss_curves.png` |
| `e2_sparse_grpo_checkpoint_dynamics_combined.png` | `plots/combined_dynamics_summary.png` |
| `e2_dense_grpo_v1_dynamics_combined.png` | `plots_dense/combined_dynamics_summary_dense.png` |
| `e2_dense_reward_vs_step.png` | `plots_dense/reward_vs_step_dense.png` |
| `e2_dense_rouge_vs_step.png` | `plots_dense/rouge_vs_step_dense.png` |

---

## 2. Recommended reading order (TA-friendly: 3 steps)

For a **quick, fair review** (~10–15 minutes before diving into appendices), follow this order:

### Step 1 — Start with the **main results table**

Open **`unified_part34_metrics.md`** (CSV optional: `unified_part34_metrics.csv`).

- **What you get:** One row each for **SFT**, **DPO retune v2**, **GRPO-V1**, **GRPO-V4** on the **same English test split**, greedy decode — ROUGE-L, loose/strict format, mean length, and GRPO **avg_reward** where applicable.
- **Why first:** This is the single source of truth for “did the methods work on the reported split?” Everything else in this folder elaborates or illustrates these numbers.

### Step 2 — Read the **unified narrative** (Part III / IV framing)

Open **`unified_alignment_analysis.md`**.

- **What you get:** Why **V1** is easy to hack, how **V4** mitigates (without solving) hacking, the **loose vs strict** lesson, and short trade-offs among SFT / DPO / GRPO.
- **Why second:** Turns the table into an interpretable story graders expect in the write-up; avoids reading long qualitative files before you know the headline conclusions.

### Step 3 — **Part V (dynamics)** and **hacking / qualitative evidence**

Use this step to verify **exploration (E2)** and **concrete failure modes**.

1. **Part V anchor:** **`partv_e2_reward_hacking_dynamics.md`** — sparse vs dense setup, tables, limitations (**n** = 81, dense V1 is supplementary).  
2. **Supporting dynamics (short):** `v1_vs_v4_dynamics_analysis.md`, `reward_hacking_onset_dense_analysis.md`, `grpo_v1_earlystop_dense_vs_final.md` — then skim **`plots/`** (`e2_sparse_*`, `e2_dense_*`, plus `sft_training_loss_curves.png` for Part I).  
3. **Hacking cases (Part III evidence):** `grpo_v1_reward_hacking_cases.md`, `grpo_v4_reward_hacking_cases.md`, and optionally `grpo_qualitative_analysis.md`.  
4. **DPO story (Part II):** `dpo_final_decision.md` (numbers), then **`dpo_qualitative_analysis.md`** (original failure) and **`dpo_retune_v2_qualitative_analysis.md`** (recovery).  
5. **SFT (Part I):** `sft_loss_summary.md` + `sft_qualitative_analysis.md` if you need training-curve context and test-set examples.

**After the three steps:** use **`appendix/`** for step-level CSVs, sparse early-stop vs final, and `sft_vs_dpo_all_metrics` if you need full DPO variant tables or raw loss series (`sft_loss_history.csv`).

---

## 3. Mapping from assignment parts to files

| Part | Primary files (this folder) |
|------|-----------------------------|
| **I — SFT** | `sft_loss_summary.md`, `plots/sft_training_loss_curves.png`, `sft_qualitative_analysis.md`; aggregate row in `unified_part34_metrics.md` |
| **II — DPO** | `dpo_final_decision.md`, `dpo_qualitative_analysis.md` (original failure), `dpo_retune_v2_qualitative_analysis.md`; `appendix/sft_vs_dpo_all_metrics.{md,csv}` |
| **III — GRPO & hacking** | `grpo_v1_reward_hacking_cases.md`, `grpo_v4_reward_hacking_cases.md`, `grpo_qualitative_analysis.md`; V1/V4 rows in `unified_part34_metrics.md` |
| **IV — Unified analysis** | `unified_part34_metrics.{md,csv}`, `unified_alignment_analysis.md` |
| **V — E2 dynamics** | `partv_e2_reward_hacking_dynamics.md`, `v1_vs_v4_dynamics_analysis.md`, `reward_hacking_onset_dense_analysis.md`, `grpo_v1_earlystop_dense_vs_final.{md,csv}`, `plots/e2_*` figures; `appendix/reward_hacking_dynamics_*`, `appendix/grpo_v1_earlystop_*` |

---

## 4. Key figures and tables

| File | Used for | Recommended report section | Notes |
|------|----------|----------------------------|--------|
| `unified_part34_metrics.md` / `.csv` | Main comparison table | Part IV (and III intro) | Same split, greedy decode; ROUGE-L, loose/strict, length, avg reward for GRPO |
| `unified_alignment_analysis.md` | Loose vs strict, V1 hackability, paradigm discussion | Part IV | Companion text to the unified table |
| `dpo_final_decision.md` | DPO variant selection vs SFT | Part II | Original vs retune v1/v2 metrics |
| `plots/sft_training_loss_curves.png` | Training dynamics | Part I | Parsed from SFT `trainer_state.json`; see `sft_loss_summary.md` |
| `sft_loss_summary.md` | Numeric summary for SFT loss | Part I | Points to `appendix/sft_loss_history.csv` for raw series |
| `plots/e2_sparse_grpo_checkpoint_dynamics_combined.png` | Sparse checkpoint dynamics (700 → 723) | Part V | Official GRPO-V1/V4 run checkpoints only |
| `plots/e2_dense_grpo_v1_dynamics_combined.png` | Dense GRPO-V1 (50–300 steps) | Part V | **Supplementary** short run; does not replace official V1 `best` in Part III |
| `plots/e2_dense_reward_vs_step.png` | Reward saturation vs step | Part V | Read with ROUGE/strict panels |
| `plots/e2_dense_rouge_vs_step.png` | Quality proxy vs step | Part V | Pair with dense reward plot |
| `partv_e2_reward_hacking_dynamics.md` | Part V narrative + tables | Part V | Single anchor for sparse + dense setup |
| `reward_hacking_onset_dense_analysis.md` | Onset heuristic (dense) | Part V | Small **n**; cautious wording |
| `grpo_v1_earlystop_dense_vs_final.md` / `.csv` | Early vs final (dense) | Part V | Auto-picked early checkpoint vs step 300 |
| `v1_vs_v4_dynamics_analysis.md` | V1 vs V4 sparse deltas | Part III / V | Tied to `appendix/reward_hacking_dynamics_metrics.*` |
| `sft_qualitative_analysis.md` | Stratified test examples | Part I / IV | High / mid / low ROUGE |
| `dpo_qualitative_analysis.md` | Original DPO failure examples | Part II | Contrasts with retune path |
| `dpo_retune_v2_qualitative_analysis.md` | SFT vs v1 vs v2 on fixed indices | Part II | Per-example can disagree with aggregate |
| `grpo_v1_reward_hacking_cases.md` | High reward, weak summary (V1) | Part III | Strict eval ≠ V1 training objective |
| `grpo_v4_reward_hacking_cases.md` | Residual failure modes (V4) | Part III | Mitigation ≠ elimination |
| `grpo_qualitative_analysis.md` | Short GRPO qualitative note | Part III | Cross-model examples |

---

## 5. Notes on reproducibility

- These files are **selected exports** from a full local pipeline. **Regenerating** them requires the same checkpoints, `data/processed/*.jsonl`, and evaluation scripts under `src/evaluation/` (see the repository root `README.md` and `docs/REPORT_GUIDE.md`).
- **Not in git:** merged model checkpoints, full `outputs/logs/`, `outputs/predictions/*.jsonl`, and the complete `outputs/report_assets/` mirror (except what you copy here). Submitters typically zip code + this folder + PDF; checkpoints only if the course requires them.
- **Figures** here are **bit-identical copies** of the PNGs produced by the plotting scripts; renamed paths above point to the originals under `outputs/report_assets/` on the machine that ran the experiments.

---

## Folder layout

```
docs/final_report_assets/
├── README.md                          (this index)
├── unified_part34_metrics.md / .csv
├── unified_alignment_analysis.md
├── sft_loss_summary.md
├── sft_qualitative_analysis.md
├── dpo_final_decision.md
├── dpo_qualitative_analysis.md
├── dpo_retune_v2_qualitative_analysis.md
├── grpo_qualitative_analysis.md
├── grpo_v1_reward_hacking_cases.md
├── grpo_v4_reward_hacking_cases.md
├── partv_e2_reward_hacking_dynamics.md
├── reward_hacking_onset_dense_analysis.md
├── grpo_v1_earlystop_dense_vs_final.md / .csv
├── v1_vs_v4_dynamics_analysis.md
├── plots/
│   ├── sft_training_loss_curves.png
│   ├── e2_sparse_grpo_checkpoint_dynamics_combined.png
│   ├── e2_dense_grpo_v1_dynamics_combined.png
│   ├── e2_dense_reward_vs_step.png
│   └── e2_dense_rouge_vs_step.png
└── appendix/
    ├── reward_hacking_dynamics_metrics.md / .csv
    ├── reward_hacking_dynamics_dense_metrics.md / .csv
    ├── reward_hacking_dynamics_dense_summary.md
    ├── reward_hacking_onset_analysis.md
    ├── grpo_v1_earlystop_vs_final.md / .csv
    ├── grpo_v1_earlystop_qualitative.md
    ├── grpo_v1_earlystop_dense_qualitative.md
    ├── sft_loss_history.csv
    └── sft_vs_dpo_all_metrics.md / .csv
```
