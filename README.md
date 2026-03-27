# CSC6129 RL Assignment 2 — Experiment archive & report-writing hub

This document is the **single master record** for finished work in this repo: what was run, where artifacts live, which numbers to cite, and how to write each assignment Part without hunting the tree. For a **short index** of the same paths, see `docs/REPORT_GUIDE.md`. For engineering detail (trainer alignment, VRAM, E2-dense audit), see `docs/IMPLEMENTATION_NOTES.md`. A **section-by-section report scaffold** is in `docs/REPORT_DRAFT_SCAFFOLD.md`.

**Freeze policy:** No new training is assumed beyond the checkpoints and metrics already on disk. Cite paths below; do not overwrite `outputs/metrics/*.json` or unified CSVs unless you intentionally re-run evaluation.

## Public final assets (GitHub / TA)

**Curated, version-controlled** tables, analyses, qualitative write-ups, and key figures live in **`docs/final_report_assets/`**. Start from **`docs/final_report_assets/README.md`** (reading order, Part mapping, figure index). The training pipeline still emits the full mirror under **`outputs/report_assets/`**, which is **gitignored** alongside checkpoints, logs, and raw predictions — clone the repo alone and use `docs/final_report_assets/` as the public results window.

---

## 1. Project Overview

### What this assignment studies

Structured **English** summarization with fixed tags: `[point]`, `[reason]` (numbered lines), `[summary]`. The pipeline trains **SFT → DPO (with retunes) → GRPO-V1 / GRPO-V4**, compares alignment paradigms, and analyzes **reward hacking** and **checkpoint dynamics** (Part V / E2).

### Parts completed in this repository

| Part | Scope (assignment) | Status in repo |
|------|-------------------|----------------|
| **I** | SFT baseline, data, format metrics | Full SFT on 3090; loss export; qualitative + test metrics |
| **II** | DPO from SFT | Original DPO failed badly; **retune v2** is the reported DPO |
| **III** | GRPO, reward design, hacking | **GRPO-V1** (tag-only reward) vs **GRPO-V4** (richer reward); hacking case MDs |
| **IV** | Unified comparison | `unified_part34_metrics` + `unified_alignment_analysis.md` |
| **V** | Exploration — E2 dynamics | **Sparse** checkpoint series on official V1/V4 runs + **dense** short V1 run for onset / early-stop |

### Current mainline (what the report should center on)

- **Task language:** English only (`main_task_lang: en` in `configs/data.yaml`); prompts/responses use `answer_en` and `summary_en_*`.
- **Split:** **90% / 5% / 5%**, **`sequential: false`**, **`seed: 42`** — see `data/splits/split_manifest.json` and `outputs/metrics/dataset_split_summary.json` (**81** test examples).
- **Checkpoints of record:** SFT `outputs/checkpoints/sft_full_3090/best`; DPO `outputs/checkpoints/dpo_lora_3090_retune_v2/best`; GRPO **V1/V4** `outputs/checkpoints/grpo_v1_3090/best`, `outputs/checkpoints/grpo_v4_3090/best` (full runs, step **723**).
- **Part V supplement:** `outputs/checkpoints/grpo_v1_e2_dense/` (**300** steps, saves every **50**) — **does not replace** official GRPO-V1 for Part III tables.

### Why these results are enough for a strong report

You have: (1) a clear **failure → recovery** story for DPO; (2) **loose vs strict** format metrics exposing template repetition; (3) **unified table** + narrative for SFT / DPO / GRPO; (4) **per-model hacking cases**; (5) **sparse + dense dynamics** for reward saturation, onset heuristics, and early stopping — all with file paths and conservative wording already drafted in `outputs/report_assets/*.md` (and mirrored for publication under `docs/final_report_assets/`).

---

## 2. Assignment-to-Artifact Map

For each Part: **question → experiments → cite these files → body vs appendix.**

### Part I — SFT

1. **Questions:** Can the model learn the template? How does loss evolve? What do outputs look like on the test split?
2. **Experiments:** Teacher-aligned **full SFT** (`configs/sft_full_3090.yaml`, `scripts/run_sft_full.sh`).
3. **Cite:** `outputs/metrics/sft_test_metrics.json`; `outputs/report_assets/sft_loss_summary.md`, `sft_loss_curves.png`, `sft_loss_history.csv`; `outputs/report_assets/sft_qualitative_analysis.md`; predictions `outputs/predictions/sft_test_greedy.jsonl`.
4. **Best for report body:** Loss figure + summary MD; 1–2 qualitative examples; table row from `unified_part34_metrics.md` (SFT line).
5. **Appendix:** Full qualitative MD; raw `metrics_summary.md` if you need a wide stage list.

### Part II — DPO

1. **Questions:** Does preference optimization help? What goes wrong with bad hyperparameters or rejects?
2. **Experiments:** Original DPO (`dpo_lora_3090`); retune v1/v2 (`run_dpo_retune.sh`, `run_dpo_retune_v2.sh`).
3. **Cite:** `outputs/metrics/dpo_test_metrics.json`, `dpo_retune_test_metrics.json`, `dpo_retune_v2_test_metrics.json`; `outputs/report_assets/dpo_final_decision.md`; `sft_vs_dpo_all_metrics.md` / `.csv`; `dpo_retune_v2_qualitative_analysis.md`; optional `dpo_qualitative_analysis.md` for the original failure narrative.
4. **Best for report body:** **Three-way or four-way table** (original vs v1 vs v2 vs SFT); **dpo_final_decision** bullets; 1 qualitative row showing **original** collapse + 1 showing **v2** alignment with SFT.
5. **Appendix:** Long qualitative MDs; per-example notes where v1 beats v2 on ROUGE (index 30).

### Part III — GRPO + reward hacking

1. **Questions:** How does online RL with a **scalar reward** change behavior? Why is V1 hackable? Does V4 mitigate?
2. **Experiments:** GRPO-V1 and GRPO-V4 from **merged SFT** + new LoRA (`configs/grpo_v1_3090.yaml`, `grpo_v4_3090.yaml`).
3. **Cite:** `outputs/metrics/grpo_v1_test_metrics.json`, `grpo_v4_test_metrics.json`; `outputs/report_assets/grpo_v1_reward_hacking_cases.md`, `grpo_v4_reward_hacking_cases.md`; `src/rewards/reward_fn.py` (definitions).
4. **Best for report body:** Unified table rows for V1/V4; **1–2 hacking cases per variant** from the MDs; short explanation of V1 tag-only vs V4 composite reward.
5. **Appendix:** Extra cases; training notes in `outputs/logs/` (if needed).

### Part IV — Unified analysis

1. **Questions:** How do SFT, DPO, and GRPO compare on **the same test split** and decoding settings? What is the **loose vs strict** lesson?
2. **Experiments:** Single greedy eval settings in `configs/inference.yaml`; exports from `export_unified_part34` / `export_unified_alignment_analysis`.
3. **Cite:** `outputs/report_assets/unified_part34_metrics.md` (and `.csv`); `outputs/report_assets/unified_alignment_analysis.md`.
4. **Best for report body:** The **unified markdown table** + §4–5 of `unified_alignment_analysis.md` (loose vs strict + paradigm trade-offs).
5. **Appendix:** CSV import for LaTeX; diagnostic v4 scores mentioned in unified table notes.

### Part V — E2 Reward hacking dynamics

1. **Questions:** How do **reward**, **ROUGE-L**, **format**, and **length** evolve over training? When does hacking “show up”? Can **early stopping** help V1?
2. **Experiments:** (A) **Sparse:** evaluate SFT best + GRPO-V1/V4 **checkpoint-700** and **best (723)**. (B) **Dense:** short GRPO-V1 only, steps 50–300 every 50.
3. **Cite:** `outputs/report_assets/partv_e2_reward_hacking_dynamics.md` (**master**); `reward_hacking_dynamics_metrics.md`, `reward_hacking_dynamics_dense_metrics.md`; `reward_hacking_onset_analysis.md`, `reward_hacking_onset_dense_analysis.md`; `grpo_v1_earlystop_dense_vs_final.md` (+ qualitative `grpo_v1_earlystop_dense_qualitative.md` if present); `v1_vs_v4_dynamics_analysis.md`; plots `outputs/report_assets/plots/` and `plots_dense/*_dense.png`; `reward_hacking_dynamics_dense_summary.md`.
4. **Best for report body:** Part V §3 (sparse table) + §3b (dense) from `partv_e2_reward_hacking_dynamics.md`; **1–2 figures** (e.g. `reward_vs_step_dense.png`, `rouge_vs_step_dense.png`); one **early-stop** table from `grpo_v1_earlystop_dense_vs_final.md`.
5. **Appendix:** Sparse-only onset note; full CSVs; comparison of SFT “step 0” row between sparse and dense tables (see §9 caveats).

---

## 3. Final Environment and Reproducibility

| Item | Value |
|------|--------|
| **Conda env** | `rlhw2_qwen35_3090` (see `scripts/setup_conda.sh`) |
| **Python** | 3.10 recommended (`environment/environment.yml` / `requirements.txt`) |
| **GPU** | **NVIDIA RTX 3090 24GB** (full SFT uses bf16 full weights; GRPO needs headroom for rollouts) |
| **CUDA** | Match PyTorch build in env; `python environment/check_env.py` checks imports |
| **HF Hub** | Model `Qwen/Qwen3.5-0.8B`. Put **`HF_TOKEN`** in repo-root **`.env`** (loaded by `src/utils/hf_env.py`; not echoed by check script) |

### Reproducing vs not re-running

- **To reproduce data only:** `make prepare-data` (uses `configs/data.yaml`: **random** split, seed **42**, English fields).
- **To reproduce metrics from existing checkpoints:** `make eval-all` then `make report-assets`, plus Part V modules listed in §10 (if checkpoints exist).
- **Already completed for this archive:** Full SFT, DPO variants, GRPO V1/V4, E2 sparse + dense eval exports. **Re-running full GRPO is slow and unnecessary** for the frozen report unless you changed code or data.

### Hard constants (copy into report methods)

- **Seed:** `42` (`configs/base.yaml` `project.seed`, `configs/data.yaml` `split.seed`).
- **Split:** train **1446** / val **80** / test **81** (`dataset_split_summary.json`); **`sequential_split`: false**.
- **Decoding (main metrics):** greedy, **`use_chat_template: false`** (`configs/inference.yaml`).
- **GRPO official final step:** **723** for V1/V4 `best` (sparse dynamics table).

---

## 4. Data Pipeline

### Teacher bundle location

- **Raw JSONL:** `train_code_with_data/data/train.jsonl` (fields `answer_zh` / `answer_en`, `summary_*_chosen` / `summary_*_rejected`, optional `question`).
- **Teacher reference code:** `train_code_with_data/` (SFT script, `data/process_data.py`, eval).

### Fields used for the **English main experiment**

Configured in `configs/data.yaml` under `fields` + `main_task_lang: en`:

- **Answer text:** `answer_en`
- **Chosen / rejected summaries:** `summary_en_chosen`, `summary_en_rejected`

Chinese fields remain in rows for **legacy / exploration** only; processed JSONL still carries them where split script passthrough allows.

### English-only correction (why Chinese is not the mainline)

The assignment pipeline was aligned to teacher code (Chinese prompt template) first; the **report mainline** switches to **English** summaries and `answer_en` so ROUGE and human-readable qualitative discussion match the structured English references. Document this as an explicit **design choice** in the report.

### Split and processed outputs

- **Manifest:** `data/splits/split_manifest.json` (hashes, counts, `sequential_split`, `main_task_lang`).
- **Splits:** `data/splits/{train,val,test}.jsonl`
- **Processed:** `data/processed/sft_{train,val,test}.jsonl`, `dpo_*.jsonl`, `grpo_*.jsonl`  
- **GRPO test = same prompts as SFT test** for unified eval (`grpo_test.jsonl` vs `sft_test.jsonl`, verified in Part V docs).

### Prompt / response shape

- **SFT rows:** `prompt`, `response` (teacher-style seq2seq; no chat template in training — see `src/training/sft_hf_trainer.py`).
- **DPO / GRPO:** `chosen` / `rejected` or `prompt` + `answer_en` per stage configs; rewards read batch fields (e.g. `answer_en`).

---

## 5. Experiment Chronology (detailed)

### 5.1 Initial repository bootstrap on 3090

- **Why:** Run full fine-tuning and RL stages on a **single 24GB** GPU with conservative batches.
- **Issues:** Possible `bitsandbytes` friction; code **falls back** to bf16 + LoRA when 4-bit unavailable (`docs/IMPLEMENTATION_NOTES.md`).
- **Artifacts:** `configs/base.yaml`, `environment/check_env.py`, `make env-check`.

### 5.2 Alignment with teacher code

- **Why:** Match course reference for SFT tokenization and prompt/response masking.
- **Result:** `sft_hf_trainer` + `DataCollatorForSeq2Seq` path; eval without chat template.
- **Files:** `docs/IMPLEMENTATION_NOTES.md`, `train_code_with_data/sft/train_sft.py` (reference).

### 5.3 English-only correction

- **Why:** Single coherent language for ROUGE, qualitative, and GRPO rewards using `answer_en`.
- **Result:** `configs/data.yaml` `main_task_lang: en` and processed manifests record English fields.
- **Files:** `outputs/metrics/dataset_alignment_summary.json`, `split_manifest.json`.

### 5.4 Full SFT

- **Why:** Strong supervised baseline and initialization for DPO/GRPO.
- **Result:** `outputs/checkpoints/sft_full_3090/best`; test ROUGE-L **0.4408**, strict **0.988**, len **124.46** tok (`unified_part34_metrics.md`).
- **Files:** `sft_loss_summary.md`, `sft_loss_curves.png`, `sft_qualitative_analysis.md`, `sft_test_metrics.json`.

### 5.5 DPO initial failure

- **Why:** First DPO hyperparameters / data effects collapsed format and quality.
- **Result:** Original DPO ROUGE-L **0.2665**, strict **0.000**, mean len **388.6** (`dpo_final_decision.md`).
- **Files:** `dpo_test_metrics.json`, `dpo_qualitative_analysis.md`.

### 5.6 Strict format metric introduction

- **Why:** **Loose** format stayed **1.0** for degenerate multi-block outputs; **strict** exposes true template compliance (`src/metrics/strict_format_adherence.py`).
- **Result:** All main metrics JSONs report **both** `format_adherence` (loose) and `strict_format_rate`.

### 5.7 DPO retune v1 and v2

- **Why:** Recover from collapse with conservative training.
- **Result:** **v2** best aggregate: ROUGE-L **0.4448**, strict **0.988**, len **121.30** — matches or slightly beats SFT on this split (`dpo_final_decision.md`).
- **Files:** `dpo_retune_v2_test_metrics.json`, `dpo_retune_v2_qualitative_analysis.md`, `sft_vs_dpo_all_metrics.md`.

### 5.8 GRPO-V1 / GRPO-V4

- **Why:** Part III — compare **weak** vs **richer** proxy rewards under online RL.
- **Result (best checkpoints):** See §6. V1 **avg_reward 1.0** (saturation); V4 **0.4992** with better ROUGE-L / strict / length on this split.
- **Files:** `grpo_*_test_metrics.json`, hacking MDs, `unified_part34_metrics.md`.

### 5.9 Part V / E2 — sparse + dense V1 dynamics

- **Why:** Assignment exploration — dynamics, onset, early stopping; sparse disk checkpoints only **700** and **723** for full runs.
- **Result:** Dense **300-step** V1 run fills in step-wise curves; **early (step 200)** beats dense final on ROUGE-L by **0.0062** on n=81 (`grpo_v1_earlystop_dense_vs_final.md`).
- **Files:** `partv_e2_reward_hacking_dynamics.md`, `plots/`, `plots_dense/`, onset + earlystop MDs.

---

## 6. Final Experimental Results (numbers + sources)

All values below are from **`outputs/report_assets/unified_part34_metrics.md`** unless marked.

| Model | Avg reward | ROUGE-L | loose | strict | len (tok) | Interpretation (conservative) |
|-------|------------|---------|-------|--------|-----------|------------------------------|
| **SFT (full)** | — | **0.4408** | 1.000 | **0.988** | 124.46 | Stable baseline; not RL-optimized. |
| **DPO retune v2** | — | **0.4448** | 1.000 | **0.988** | 121.30 | **~match or slight lift** vs SFT on ROUGE-L; same strict; **shorter** outputs — **not** a large effect size on n=81. |
| **GRPO-V1** | **1.0000** | 0.4396 | 1.000 | 0.988 | 123.62 | **Roughly on par** with SFT on ROUGE/strict; reward **saturated** — use hacking narrative, not “RL clearly wins.” |
| **GRPO-V4** | **0.4992** | **0.4495** | 1.000 | **1.000** | **119.48** | **Best point estimate** here on ROUGE-L, strict, length; still **not** human-level summarization — cite hacking cases for honesty. |

**DPO choice:** Report **retune v2** as the final DPO (`dpo_final_decision.md`). Original DPO is the **failure** anchor.

**Part V sparse dynamics (same test file, from `reward_hacking_dynamics_metrics.md`):**

- SFT (step 0): ROUGE-L **0.4408**, strict **0.988**, len **124.5**
- GRPO-V1: 700 → 723: ROUGE **0.4428 → 0.4396**, avg_reward **1.0** throughout
- GRPO-V4: 700 → 723: ROUGE **0.4480 → 0.4495**, avg_reward **0.4990 → 0.4992**

**Part V dense (V1 only, auxiliary):** full table in `reward_hacking_dynamics_dense_metrics.md` and §3b of `partv_e2_reward_hacking_dynamics.md`.

**Significance language:** Treat **Δ ROUGE &lt; ~0.01–0.02** on **81** examples as **“similar / slightly better/worse”** unless you add CIs. **Strict format** moving **0.988 → 1.000** is a **visible** compliance gain for GRPO-V4 vs SFT/DPO/V1 on this run.

---

## 7. Key Figures and Tables for the Final Report

| Purpose | Suggested file(s) | Report section | What to say (1 line) |
|---------|-------------------|----------------|----------------------|
| SFT training curve | `outputs/report_assets/sft_loss_curves.png`, `sft_loss_summary.md` | Part I | Train loss ↓; eval loss tracked at epoch granularity. |
| SFT qualitative | `sft_qualitative_analysis.md` | Part I / IV | Mix of high/mid/low ROUGE; failure mode with strict pass but poor content. |
| SFT vs DPO (all variants) | `sft_vs_dpo_all_metrics.md`, `dpo_final_decision.md` | Part II | Original collapse vs retune recovery; **v2** final. |
| DPO qualitative (v2) | `dpo_retune_v2_qualitative_analysis.md` | Part II | Per-example can disagree with aggregate (v1 vs v2 on index 30). |
| Unified main table | `unified_part34_metrics.md` (or CSV) | Part IV (and III intro) | Same split, greedy decode — **the** comparison table. |
| Alignment narrative | `unified_alignment_analysis.md` | Part IV | Loose vs strict; V1 hackability; paradigm trade-offs. |
| GRPO-V1 hacking | `grpo_v1_reward_hacking_cases.md` | Part III | High v1 reward + low ROUGE / strict fail (e.g. repetition). |
| GRPO-V4 hacking / limits | `grpo_v4_reward_hacking_cases.md` | Part III | Strict pass but low ROUGE — reward ≠ quality. |
| Part V master text | `partv_e2_reward_hacking_dynamics.md` | Part V | Sparse + dense setup and findings in one place. |
| Dynamics plots (sparse) | `outputs/report_assets/plots/*.png` | Part V | Reward / ROUGE / strict / length vs step (coarse). |
| Dynamics plots (dense) | `outputs/report_assets/plots_dense/*_dense.png`, `reward_hacking_dynamics_dense_summary.md` | Part V | Finer V1 curves; saturation vs quality decoupling. |
| Onset (dense) | `reward_hacking_onset_dense_analysis.md` | Part V | Heuristic onset window; **n=81** caution. |
| Early stop (dense) | `grpo_v1_earlystop_dense_vs_final.md` | Part V | Auto early &gt; final on ROUGE-L; small delta. |
| V1 vs V4 dynamics | `v1_vs_v4_dynamics_analysis.md` | Part III / V | Sparse checkpoint narrative. |

---

## 8. Part-by-Part Writing Guide

### Part I — SFT

- **Structure:** Data + split → model/training setup → **loss curves** → **aggregate test metrics** (ROUGE, loose, strict, length) → **2–3 qualitative** examples (good + bad).
- **Emphasize:** English-only mainline; **strict** format catches issues loose misses.
- **Pitfall:** Claiming success on **loose=1.0** alone.
- **Safe template:** “SFT establishes a strong baseline on the English structured summarization task; training loss decreases over epochs, and on the held-out test split we observe ROUGE-L **0.4408** and strict format **0.988** (`unified_part34_metrics.md`).”

### Part II — DPO

- **Structure:** Preference data construction → original DPO **failure** (metrics + example) → retune motivation → **v2** metrics vs SFT → qualitative.
- **Emphasize:** **Original DPO is not success**; **v2** is the reported model; per-example **≠** always ranking v2 above v1.
- **Pitfall:** Hiding the failed run; overstating DPO gains (**~0.004** ROUGE vs SFT).
- **Safe template:** “Initial DPO training degraded both automatic metrics and template compliance; after conservative retuning, **DPO retune v2** matches or slightly improves ROUGE-L relative to SFT on this split while maintaining strict format **0.988** (`dpo_final_decision.md`).”

### Part III — GRPO + hacking

- **Structure:** Reward definitions (V1 vs V4) → training setup from SFT → **aggregate** comparison → **hacking cases** + why V1 saturates.
- **Emphasize:** V1 **avg_reward = 1.0** is a **pedagogical** signal; V4 **mitigates** but see V4 cases.
- **Pitfall:** “RL beats everything” without citing **unified** numbers; ignoring **strict** vs **reward objective** mismatch.
- **Safe template:** “GRPO-V1’s tag-only reward quickly saturates, enabling high-reward outputs with visible quality issues; GRPO-V4’s richer shaping improves strict compliance and ROUGE-L on this split, though high-reward–low-ROUGE cases remain (`grpo_*_reward_hacking_cases.md`).”

### Part IV — Unified

- **Structure:** Single evaluation protocol → **unified table** → **loose vs strict** lesson → three-paradigm trade-offs (from `unified_alignment_analysis.md`).
- **Emphasize:** Same test JSONL and decoding; **honesty clause** in analysis doc if RL underperforms (here V4 is best point estimate).
- **Pitfall:** Cherry-picking one metric; ignoring **length**.
- **Safe template:** “Under greedy decoding on the shared English test split, GRPO-V4 achieves the strongest aggregate trade-off among RL runs, while SFT and DPO retune v2 remain competitive; differences in ROUGE-L between SFT, DPO v2, and GRPO-V1 are small in absolute terms (`unified_part34_metrics.md`).”

### Part V — E2 dynamics

- **Structure:** Motivation (proxy reward risk) → **sparse** official checkpoints → **dense V1** supplement → onset + early stopping → limitations (**n**, single seed).
- **Emphasize:** Dense run is **auxiliary**; **does not replace** official GRPO-V1 **best** for Part III.
- **Pitfall:** Mixing sparse and dense SFT baselines **without** noting the strict-rate discrepancy (§9).
- **Safe template:** “Checkpoint-level evaluation shows that V1’s test reward is saturated early, while ROUGE-L and strict format evolve more slowly; a dense short run suggests an early checkpoint can improve ROUGE-L relative to the dense final step under our selection rule (`grpo_v1_earlystop_dense_vs_final.md`), but evidence is limited to **n=81**.”

---

## 9. Important Caveats and Honest Limitations

- **Small test set (81):** Rankings can flip with resplits; describe **effect sizes** modestly.
- **Single seed** for reported runs; no variance bands in repo unless you compute them.
- **Sparse GRPO curves:** Full V1/V4 runs only have **700** and **723** on disk for dynamics tables — **do not imply** you have continuous training logs unless you export them.
- **Dense E2 is V1-only, 300 steps:** Shorter than full **723**-step V1; use for **dynamics narrative**, not as the sole “official” V1 model in Part III.
- **SFT baseline discrepancy (sparse vs dense):** `reward_hacking_dynamics_metrics.md` lists SFT strict **0.988** and ROUGE **0.4408** (aligned with `unified_part34_metrics.md`). `reward_hacking_dynamics_dense_metrics.md` lists SFT strict **1.000** and ROUGE **0.4429** at “step 0” — likely **evaluation pipeline / tokenizer-from-checkpoint** differences between series. **For Part III/IV, prefer the unified + sparse E2 SFT row**; in Part V, **stay within one series** when comparing steps or explain the two baselines explicitly.
- **avg_reward ≠ human quality; strict ≠ training objective** for V1/V4 in full — always pair with ROUGE and qualitatives.
- **Trainer log gap for dense run:** `outputs/logs/grpo_v1_e2_dense/train.log` may be incomplete; rely on **saved checkpoints + exported metrics** (`docs/IMPLEMENTATION_NOTES.md` E2-dense audit).

---

## 10. Command Reference

| Command / action | Purpose | Executed in this project? | Re-run now? |
|------------------|---------|---------------------------|-------------|
| `make env-check` | Verify imports, CUDA, optional bnb | Yes | **Yes** (quick sanity) |
| `make prepare-data` | Build splits + processed JSONL from `data.yaml` | Yes | Only if raw data or split config changes |
| `bash scripts/run_sft_full.sh` / `make sft-full` | Full SFT | Yes | **No** (long); use existing `best` |
| `bash scripts/run_dpo_full.sh` | Original DPO | Yes | **No** unless documenting failure again |
| `bash scripts/run_dpo_retune.sh` / `run_dpo_retune_v2.sh` | DPO retunes | Yes | **No** for freeze |
| `bash scripts/run_grpo_v1_full.sh` / `run_grpo_v4_full.sh` | GRPO | Yes | **No** for freeze |
| `bash scripts/run_grpo_v1_e2_dense.sh` | Dense V1 supplement | Yes | **No** unless redoing Part V |
| `make eval-all` | Batch eval bests → predictions + metrics | Yes | Optional refresh if checkpoints unchanged |
| `make report-assets` | Summaries, SFT loss export, compare tables, qualitative templates | Yes | Safe to refresh exports |
| `python -m src.evaluation.evaluate --config configs/base.yaml configs/inference.yaml --stage <sft|dpo|grpo_v1|grpo_v4> --checkpoint <path>` | Single-stage eval | Yes | As needed |
| `python -m src.evaluation.export_unified_part34` / `export_unified_alignment_analysis` | Unified Part III/IV MD/CSV | Yes | Regenerate after eval |
| `python -m src.evaluation.evaluate_checkpoint_series --preset sparse` | E2 sparse metrics | Yes | If checkpoints exist |
| `python -m src.evaluation.evaluate_checkpoint_series --preset dense` | E2 dense metrics | Yes | If dense ckpts exist |
| `python -m src.evaluation.plot_reward_hacking_dynamics --style sparse` / `--style dense` | Dynamics plots | Yes | After CSV refresh |
| `python -m src.evaluation.export_partv_e2_report` | Single Part V master MD | Yes | After sub-exports |

---

## 11. Final Submission Checklist

- [ ] **PDF report** covers Parts I–V with **citations** to paths or copied tables from `outputs/report_assets/`.
- [ ] **Code zip** includes `src/`, `configs/`, `scripts/`, `environment/`, `docs/`, **README.md**; exclude huge `outputs/checkpoints/` if submission size forbids (then document how to reproduce).
- [ ] **Figures embedded:** SFT loss; ≥1 Part V dynamics figure (sparse or dense); optional unified comparison chart if you build one.
- [ ] **Tables:** `unified_part34_metrics` + DPO comparison (`sft_vs_dpo_all_metrics` or `dpo_final_decision` summary).
- [ ] **Qualitative:** At least one block each from SFT + DPO + GRPO hacking MDs (trim for page limit).
- [ ] **Honesty:** Limitations §9 echoed in report discussion (n=81, sparse curves, dense auxiliary).
- [ ] **HF / license:** Qwen license + course rules mentioned if required.
- [ ] **Internal consistency:** Do not mix sparse/dense SFT baselines in one figure axis without a footnote.

---

## Legacy pointers (not the English mainline focus)

- Teacher default language in **reference** code is Chinese; this repo’s **frozen** mainline uses **English** processed fields — state that clearly in the report.
- `outputs/report_assets/reward_hacking_cases.md` is a **template** stub from `export_report_assets.sh`; the **real** cases are `grpo_v1_reward_hacking_cases.md` and `grpo_v4_reward_hacking_cases.md`.

## License

Course / research use; comply with Qwen model licenses and your institution’s rules.
