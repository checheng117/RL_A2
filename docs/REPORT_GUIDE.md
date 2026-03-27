# Report guide — quick index for the homework PDF

**Relationship to `README.md`:** The **README** is the **full experiment archive** (chronology, numbers, caveats, commands). This file is a **fast lookup**: what to cite where, and the safest one-line framing.

**Quick browse of frozen results in git:** Open **`docs/final_report_assets/README.md`** first — it indexes the curated copies of metrics, analyses, and plots for graders or public clones (without local `outputs/`).

---

## Part I — SFT

| Need | File |
|------|------|
| Loss artifact | `outputs/report_assets/sft_loss_curves.png`, `sft_loss_summary.md`, `sft_loss_history.csv` |
| Test metrics | `outputs/metrics/sft_test_metrics.json` |
| Qualitative | `outputs/report_assets/sft_qualitative_analysis.md` |

**Stable wording:** Report **strict_format_rate** alongside legacy loose format; cite English-only mainline + seed/split from `README.md` §3–4.

---

## Part II — DPO

| Need | File |
|------|------|
| Failure vs recovery | `outputs/report_assets/dpo_final_decision.md` |
| Multi-way table | `outputs/report_assets/sft_vs_dpo_all_metrics.md` (or `.csv`) |
| v2 qualitative | `outputs/report_assets/dpo_retune_v2_qualitative_analysis.md` |
| Original DPO narrative | `outputs/report_assets/dpo_qualitative_analysis.md` (if present) |

**Stable wording:** Original DPO **failed**; **DPO retune v2** is the reported preference model; aggregate metrics **slightly** favor v2 over SFT on ROUGE-L — do not oversell.

---

## Part III — GRPO + hacking

| Need | File |
|------|------|
| Reward code | `src/rewards/reward_fn.py` |
| Metrics | `outputs/metrics/grpo_v1_test_metrics.json`, `grpo_v4_test_metrics.json` |
| Cases | `outputs/report_assets/grpo_v1_reward_hacking_cases.md`, `grpo_v4_reward_hacking_cases.md` |

**Stable wording:** V1 reward **saturates** (~1.0); V4 is **richer** and **mitigates** some failures; **high reward + low ROUGE** still appears — cite cases.

---

## Part IV — Unified comparison

| Need | File |
|------|------|
| Master table | `outputs/report_assets/unified_part34_metrics.md` + `.csv` |
| Narrative | `outputs/report_assets/unified_alignment_analysis.md` |

**Stable wording:** Same test split, greedy decode; **loose vs strict** lesson is central; small ROUGE gaps are **not** definitive on n=81.

---

## Part V — E2 reward hacking dynamics

| Need | File |
|------|------|
| **Single paste anchor** | `outputs/report_assets/partv_e2_reward_hacking_dynamics.md` |
| Sparse metrics | `outputs/report_assets/reward_hacking_dynamics_metrics.md`, `outputs/report_assets/plots/*.png` |
| Dense (V1) metrics + plots | `reward_hacking_dynamics_dense_metrics.md`, `plots_dense/*_dense.png`, `reward_hacking_dynamics_dense_summary.md` |
| Onset | `reward_hacking_onset_analysis.md` (sparse), `reward_hacking_onset_dense_analysis.md` (dense) |
| Early stop | `grpo_v1_earlystop_vs_final.md` (sparse 700 vs 723), `grpo_v1_earlystop_dense_vs_final.md` + `grpo_v1_earlystop_dense_qualitative.md` |
| V1 vs V4 narrative | `v1_vs_v4_dynamics_analysis.md` |

**Stable wording:** Dense V1 run is **supplementary**; official Part III GRPO-V1 remains **`grpo_v1_3090/best`**. Acknowledge **sparse** checkpoints on full runs. See README §9 for SFT baseline sparse vs dense.

---

## Figures worth prioritizing

1. `sft_loss_curves.png`  
2. `unified_part34_metrics` (table)  
3. One Part V panel: `plots/combined_dynamics_summary.png` **or** `plots_dense/combined_dynamics_summary_dense.png`  
4. Optional: single-metric dense curves (`reward_vs_step_dense.png`, `rouge_vs_step_dense.png`)

---

## Regenerating exports (optional)

After `make eval-all`: `make report-assets`. Unified + Part V modules are listed in **`README.md` §10**. Do not regenerate for the frozen hand-in unless you changed checkpoints or code.

---

## Packaging

- Prefer **code zip** without multi-GB `outputs/checkpoints/` if rules require; keep **configs + src + scripts + README + docs + report_assets**.
