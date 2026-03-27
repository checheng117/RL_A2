# Implementation notes

## Final engineering decisions (report freeze summary)

These are the **locked** choices the report should describe; full narrative lives in **`README.md`**.

1. **English-only mainline** — `configs/data.yaml` sets `main_task_lang: en` and uses `answer_en` / `summary_en_*` for training and eval. Chinese fields remain for exploration/legacy only.
2. **Split** — **90/5/5**, **`sequential: false`**, **`seed: 42`** (`split_manifest.json`: train 1446, val 80, test **81**).
3. **Strict format metric** — Added because **loose** `format_adherence` could stay **1.0** for degenerate multi-block outputs; **strict** (`strict_format_rate`) matches single-template intent (`src/metrics/strict_format_adherence.py`).
4. **DPO** — Original run collapsed (see `dpo_final_decision.md`); **DPO retune v2** (`outputs/checkpoints/dpo_lora_3090_retune_v2/best`) is the **submission** DPO — conservative retune, aggregate metrics ≥ SFT on this split.
5. **GRPO** — Policy init from **merged full SFT** + new LoRA; **V1** = tag-heavy reward (saturates, hacking-friendly); **V4** = order, numbered reasons, length/repetition, overlap — **better aggregate proxies**, not human judgment.
6. **Part III/IV numbers** — **Source of record:** `outputs/report_assets/unified_part34_metrics.{md,csv}` (greedy, `configs/inference.yaml`).
7. **Part V / E2** — **Sparse:** official `grpo_v1_3090` / `grpo_v4_3090` checkpoints **700** and **723**. **Dense:** `grpo_v1_e2_dense` only — **300** steps, `save_steps=50`, for **finer V1 dynamics**; **does not replace** official V1 for the main table. Post-hoc exports + log caveat: see **E2-dense recovery audit** below.

---

## Scope

This repository implements a **single-GPU (RTX 3090 24GB)** pipeline for:

- Fixed **90/5/5** split, **seed 42**, manifest + summary JSON
- **Teacher-aligned SFT** (`train_code_with_data/sft/train_sft.py`): HuggingFace `Trainer` + `DataCollatorForSeq2Seq`, **prompt/response** with label masking (see `src/training/sft_hf_trainer.py`)
- **QLoRA path** (legacy / assignment skeleton): **SFT** (`trl.SFTTrainer`), **DPO** (`trl.DPOTrainer`), **GRPO** (`trl.GRPOTrainer`) on `Qwen/Qwen3.5-0.8B` — expects a `text` column; current processed JSONL from `split_dataset` is **teacher-shaped** (`prompt`/`response`), so **full SFT is the supported default** (`scripts/run_sft_full.sh`)
- Unified **evaluation** (format, ROUGE-L, length, scalar reward) and **report asset** export

## Teacher alignment (source of truth: `train_code_with_data/`)

1. **Raw data** — `train_code_with_data/data/train.jsonl` (JSONL; fields `answer_zh/en`, `summary_*_chosen/rejected`, optional `question`).
2. **Prompt** — Chinese template from `data/process_data.py` (`PROMPT_TEMPLATE`), not English chat instructions.
3. **Processed columns** — SFT: `prompt`, `response` (+ passthrough raw fields). DPO/GRPO: same as teacher (`chosen`/`rejected`; `reference` + `original_answer`).
4. **Split** — Default **sequential** 90/5/5 (no shuffle) to mirror teacher’s deterministic ordering philosophy while keeping assignment 5% val + 5% test; set `split.sequential: false` to shuffle with `seed`.
5. **SFT trainer** — Tokenization matches teacher: `encode(prompt)` + `encode(response)+EOS`, labels `-100` on prompt tokens; **no** `apply_chat_template` during training.
6. **Inference / eval** — `configs/inference.yaml` sets `use_chat_template: false` and uses `generate_from_plain_prompt` (left padding), consistent with `train_code_with_data/eval/evaluate.py`.
7. **Format metrics** — **Loose** (`format_adherence` / `batch_format_adherence`): tags + coarse order; can stay at 1.0 when the model emits **multiple** `[point]/[reason]/[summary]` blocks. **Strict** (`strict_format_rate` / `src/metrics/strict_format_adherence.py`): exactly one of each tag, correct order, non-empty bodies, and ≥2 numbered lines in `[reason]` (aligned with reward `_reason_numbered`). Evaluation JSON keeps **both**. Teacher `check_format` remains the legacy single-match regex; `check_format_strict` wraps the strict checker.
8. **HF auth** — `src/utils/hf_env.py` loads `HF_TOKEN` from repo `.env` (never logged); also set in `modeling.py` before Hub downloads.

## Key decisions (legacy stack)

1. **Config merge order** — CLI accepts multiple YAML files; later files override earlier keys (`src/training/common.py::deep_merge`). Standard triple: `base.yaml` + `data.yaml` + stage YAML.

2. **QLoRA vs fallback** — `configs/base.yaml` sets `quantization.load_in_4bit: true`. If `bitsandbytes` is absent or rejected by `transformers` (version gate), `load_causal_lm_base` **falls back** to bf16/fp32 full weights + LoRA. **Full SFT** (`sft_hf_trainer`) loads **bf16 full weights** without PEFT.

3. **TRL API drift** — `SFTConfig` uses `max_length` (not `max_seq_length`). `DPOConfig` in current `trl` has `max_length` only (no `max_prompt_length`). `GRPOConfig` exposes `max_completion_length` but **no** `max_prompt_length`; prompt truncation relies on trainer/tokenizer defaults. YAML still documents `max_prompt_length` for human tuning notes.

4. **GRPO rewards** — Implemented as pure Python callables compatible with [TRL GRPO custom reward functions](https://huggingface.co/docs/trl/main/en/grpo_trainer): `completions`, `prompts`, plus dataset columns forwarded via `**kwargs`. Variants **v1–v4** are substantive; **v5** is a **stub** (extends v4 with a mild length prior) for extension / report exploration.

5. **DPO reference model** — `ref_model` is a **second** frozen copy of the **same** initialization as the policy. If `sft_adapter_path` is a **merged full SFT** directory (no `adapter_config.json`), policy = that checkpoint + **new LoRA**, ref = frozen full copy (`load_dpo_policy_and_ref_from_full_sft`). If it is a **LoRA-on-Hub** SFT dir, the legacy `load_model_with_sft_adapter` path is used. This duplicates weights (0.8B on 24GB is usually OK; if OOM, reduce `max_seq_length` / batch).

6. **Evaluation output length** — Report the primary length metric as **`avg_output_length_tokens`** (tokenizer from the evaluated checkpoint, `encode(pred, add_special_tokens=False)`). **`avg_output_length_chars`** is auxiliary only.

7. **PEFT eval after DPO-on-full-SFT** — `evaluation.peft_base_checkpoint` in `configs/inference.yaml` must point at the **merged SFT** directory used to build the policy; do not load Hub base + DPO adapter.

8. **SFT/DPO tokenizer warnings** — TRL LoRA path may log *tokenized prompt vs prompt+completion mismatch* when using string `text` built from `apply_chat_template`. Prefer **teacher full SFT** for strict alignment.

## VRAM risk points

- **Full SFT 0.8B (3090 24GB)**: initial `max_length=1536` + `per_device_train_batch_size=2` **OOM**; production settings are `max_length: 1024`, micro-batch **1**, `gradient_accumulation_steps: 16` (effective 16), plus **gradient checkpointing** — still mirrors teacher label-masking logic, only VRAM trade-offs differ.
- **DPO**: two quantized/full bases if not careful — watch `nvidia-smi`.
- **GRPO**: **online generation** + **multiple completions** (`num_generations`). Defaults are small (`2`). Increase only after smoke tests pass.
- **Fallback bf16 base**: without 4-bit, **LoRA full forward** uses more memory — lower `per_device_train_batch_size` and `max_seq_length` / `max_completion_length`.

## Optional optimizations (not default)

- **flash-attn** / **fast kernels** for Qwen: optional, install pain on some systems — omitted from default env.
- **vLLM** in GRPO (`GRPOConfig.use_vllm`): fastest, but extra setup and memory tuning; see TRL docs.
- **W&B / TensorBoard**: training uses `report_to="none"` by default; enable in YAML if desired.

## Checkpoint selection

- **Teacher SFT**: `load_best_model_at_end` + `save_strategy: epoch`; `best/` copied from `trainer.state.best_model_checkpoint` (or latest `checkpoint-*`).
- **SFT/DPO (LoRA)**: Hugging Face `load_best_model_at_end` when eval runs; script copies `trainer.state.best_model_checkpoint` to `outputs/checkpoints/<stage>/best`.
- **GRPO**: best checkpoint API varies; script copies **last** `checkpoint-*` to `best` for a stable path for inference scripts.

## Pre-GRPO checklist (English main line)

- **DPO retune** — v1: `configs/dpo_lora_3090_retune.yaml`, `bash scripts/run_dpo_retune.sh`. v2 (second pass, more conservative): `configs/dpo_lora_3090_retune_v2.yaml`, `bash scripts/run_dpo_retune_v2.sh` → `outputs/checkpoints/dpo_lora_3090_retune_v2/best`. Four-way metrics: `python -m src.evaluation.compare_sft_dpo_all`. Part II closure: `python -m src.evaluation.export_dpo_final_decision`. Older triple table: `compare_sft_dpo_triple.py`; pre-GRPO note: `export_pre_grpo_decision`.
- **Reward vs strict format** — `reward_v1` only checks tag presence → **cannot** penalize repeated blocks. `reward_v4` adds order, numbered reasons, length/repetition, source overlap → **closer** to strict reporting but not identical. Prefer **V4** (or a future strict-aware variant) if GRPO is used to mitigate DPO-style template repetition.
- **GRPO init (English mainline)** — `configs/grpo_v1_3090.yaml` / `grpo_v4_3090.yaml` set `training.sft_adapter_path: outputs/checkpoints/sft_full_3090/best`. `load_grpo_policy_from_sft_checkpoint` loads **merged full SFT** + **new trainable LoRA**; legacy **LoRA-only** SFT dirs still use `PeftModel.from_pretrained` on the quantized base.
- **GRPO data** — `data/processed/grpo_{train,val,test}.jsonl` with `prompt`, `answer_en`, etc.; reward functions read `answer_en` from the batch (`make_trl_reward_fn`).
- **Part III exports** — `export_grpo_reward_hacking.py` → `grpo_v1_reward_hacking_cases.md` / `grpo_v4_reward_hacking_cases.md`; `export_unified_part34.py` → `unified_part34_metrics.*`; `export_grpo_qualitative_analysis.py`, `export_unified_alignment_analysis.md`.
- **Part V / E2 (checkpoint dynamics)** — **Sparse (mainline):** official `grpo_v1_3090` / `grpo_v4_3090` keep **`checkpoint-700`** and **`best`** (step **723**). Eval: `python -m src.evaluation.evaluate_checkpoint_series --preset sparse` → `reward_hacking_dynamics_metrics.csv`; plots `plot_reward_hacking_dynamics --style sparse`. **Dense (V1 supplement only):** `configs/grpo_v1_e2_dense.yaml` — **300** `max_steps`, `save_steps=50`, `save_total_limit=20`, same **v1** reward and SFT init; **does not replace** Part III `grpo_v1_3090`. Train: `bash scripts/run_grpo_v1_e2_dense.sh` (conda `rlhw2_qwen35_3090`). Eval: `python -m src.evaluation.evaluate_checkpoint_series --preset dense` → `reward_hacking_dynamics_dense_metrics.csv`, `outputs/metrics/e2_dynamics_dense/`; plots `plot_reward_hacking_dynamics --style dense`; `export_reward_hacking_dynamics_dense_summary`, `export_reward_hacking_onset_dense_analysis`, `export_grpo_v1_earlystop_dense`. Full E2 write-up: `export_partv_e2_report` → `partv_e2_reward_hacking_dynamics.md`.
- **Risks** — Runaway length, duplicate structure blocks, reward hacking on **loose** format, train/eval metric mismatch (optimize V1 but report strict).

## Strict format (why it was added)

DPO can score **format_rate = 1.0** while repeating several full template cycles; strict format makes that failure visible for reports and for deciding whether GRPO is justified.

## E2-dense recovery audit (2026-03-27)

- **Checkpoints on disk:** `outputs/checkpoints/grpo_v1_e2_dense/` has **`checkpoint-50` … `checkpoint-300`** (step every 50 up to 300) plus **`best`**. Each folder contains full LoRA artifacts (`adapter_model.safetensors`, `trainer_state.json`, etc.). There is **no** repo-root `trainer_state.json` under that directory (only inside each save folder).
- **`best` vs final:** `diff` shows `best/trainer_state.json` matches `checkpoint-300/trainer_state.json`; both record **`global_step`: 300**, **`epoch`** ≈ **0.4149**. Log line: *Copied last GRPO checkpoint to …/best* — **best** is the **last** save, not a separate early-best selection (`best_global_step` is null in state).
- **Training log gap:** `outputs/logs/grpo_v1_e2_dense/train.log` on disk is only a **few lines** (stdout tee failed when the log directory was missing earlier); **do not** rely on it for full training traces or `train_runtime` (user saw those in another console session).
- **Post-hoc pipeline (no retrain):** Ran `python -m src.evaluation.evaluate_checkpoint_series --preset dense`, then dense plots (`plot_reward_hacking_dynamics --style dense`), `export_reward_hacking_dynamics_dense_summary`, `export_reward_hacking_onset_dense_analysis`, `export_grpo_v1_earlystop_dense`, `export_partv_e2_report`. Dense figures and the dense summary markdown state **explicitly** which checkpoint steps were evaluated (from CSV / plot footers).
