# Report draft scaffold (not a full report)

Use this as a **section outline** while writing the PDF. Fill prose with evidence from paths cited in each bullet. **Numbers:** copy from `outputs/report_assets/unified_part34_metrics.md` and Part V MDs unless you re-ran eval.

---

## Title / Abstract

- **Cite:** `README.md` §1 (overview), §6 (headline table).
- **Write:** Task (English structured summarization); methods (SFT, DPO retune v2, GRPO-V1/V4); main finding (V4 best aggregate trade-off here; V1 saturation + hacking; DPO recovery story).

---

## 1. Introduction

- **Cite:** Assignment wording + `unified_alignment_analysis.md` §5 (paradigm trade-offs).
- **Write:** Why proxy rewards and format matter; roadmap of sections.

---

## 2. Data and experimental setup

- **Cite:** `README.md` §4; `data/splits/split_manifest.json`; `outputs/metrics/dataset_split_summary.json`.
- **Write:** Source JSONL path; **90/5/5**, **seed 42**, **shuffle on** (`sequential: false`); English-only field choice; prompt/response format (`[point]` / `[reason]` / `[summary]`).

---

## 3. Part I — Supervised fine-tuning (SFT)

- **Cite:** `sft_loss_curves.png`, `sft_loss_summary.md`; `sft_test_metrics.json`; `sft_qualitative_analysis.md`.
- **Write:** Model, training objective, hyperparameters (point to `configs/sft_full_3090.yaml`); loss trend; **test** ROUGE-L **0.4408**, strict **0.988**, length **124.46** (unified table); 1–2 qualitative examples (good + failure).

---

## 4. Part II — Direct preference optimization (DPO)

- **Cite:** `dpo_final_decision.md`; `sft_vs_dpo_all_metrics.md`; `dpo_retune_v2_qualitative_analysis.md`; optional `dpo_qualitative_analysis.md`.
- **Write:** Pipeline from merged SFT; **original failure** (ROUGE **0.2665**, strict **0**); retune v1/v2 motivation; **final v2** metrics vs SFT; honesty: per-example v1 can beat v2 on some indices; **β / data** lessons in prose.

---

## 5. Part III — GRPO and reward design

- **Cite:** `src/rewards/reward_fn.py`; `grpo_v1_reward_hacking_cases.md`, `grpo_v4_reward_hacking_cases.md`; unified table rows for V1/V4.
- **Write:** V1 definition + why it saturates; V4 components; training setup (SFT init + LoRA); **avg_reward** interpretation; **≥2 cases** per variant with “high reward, weak summary” diagnosis.

---

## 6. Part IV — Unified comparison

- **Cite:** `unified_part34_metrics.md`; `unified_alignment_analysis.md` §2–5.
- **Write:** Same split + greedy decode; full comparison table; **loose vs strict** paragraph; conservative statement on small ROUGE differences; when RL “wins” vs “teaches failure modes.”

---

## 7. Part V — Exploration: E2 reward hacking dynamics

- **Cite:** `partv_e2_reward_hacking_dynamics.md`; `reward_hacking_dynamics_metrics.md` + `plots/`; `reward_hacking_dynamics_dense_metrics.md` + `plots_dense/`; `reward_hacking_onset_dense_analysis.md`; `grpo_v1_earlystop_dense_vs_final.md`; `v1_vs_v4_dynamics_analysis.md`.
- **Write:** Motivation; **sparse** official checkpoints (700, 723); **dense V1** supplement (50–300); reward saturation + ROUGE/strict panels; onset heuristic; early-stop result (early **200** vs final **300** on dense); limitations (**n=81**, single seed, sparse V4); **SFT baseline** consistency note if comparing across sparse/dense CSVs (`README.md` §9).

---

## 8. Discussion and limitations

- **Cite:** `unified_alignment_analysis.md` §6; `README.md` §9.
- **Write:** What would change with more data/seeds; reward ≠ human judgment; strict ≠ training objective for V1; no over-claim on GRPO.

---

## 9. Conclusion

- **Write:** 3–4 bullets: baseline quality; DPO failure/recovery; V1 hacking lesson; V4 mitigation; dynamics/early-stop as **hypothesis-generating** evidence.

---

## Appendix ideas (optional)

- Extra qualitative rows; full CSV imports; hyperparameter tables from YAML snapshots; one-page reproduction checklist from `README.md` §10–11.
