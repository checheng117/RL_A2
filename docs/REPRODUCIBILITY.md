# Reproducibility

## Randomness

- **Global seed**: `project.seed` in `configs/base.yaml` (default **42**), applied via `src/utils/seed.py` in training scripts.
- **Split**: `split.seed` in `configs/data.yaml` (**42**). If `split.sequential` is **false**, shuffling uses `random.Random(seed).shuffle` before slicing; default **sequential** preserves file order.

## Data split

- Ratios: **90% / 5% / 5%** (`train_ratio`, `val_ratio`, `test_ratio`).
- **Default (teacher-aligned)**: `split.sequential: true` — **no shuffle**; preserves JSONL order (deterministic). To reproduce the older shuffled split, set `sequential: false` (still uses `seed` when shuffling).
- **Do not re-split** after experiments; all stages read `data/processed/*` built from the same manifest.
- `data/splits/split_manifest.json` records counts, source file SHA-256 hashes, field names, UTC timestamp, and `teacher_lang`.
- `outputs/metrics/dataset_split_summary.json` stores split stats + basic length statistics.
- `outputs/metrics/dataset_alignment_summary.json` records teacher prompt excerpt + alignment metadata.

## Software versions

- Pin training stack via `environment/requirements.txt` / `environment/environment.yml`.
- After upgrades, record `pip freeze` (or conda export) alongside checkpoints for the report.

## Checkpoint selection

- **SFT / DPO**: best checkpoint chosen by `metric_for_best_model` (`eval_loss` by default); copied to `outputs/checkpoints/<experiment>/best`.
- **GRPO**: `best` mirrors the latest `checkpoint-*` after training (see `IMPLEMENTATION_NOTES.md`).

## How to reproduce a result

1. Same commit hash + same `data/raw` inputs (verify hashes in `split_manifest.json`).
2. Same merged YAML (store a copy under `outputs/logs/` or commit to git).
3. Same command line (including `--override` flags).
4. For inference metrics: same `--checkpoint` path and `configs/inference.yaml` generation settings.

## HF Hub access

- Model weights: `Qwen/Qwen3.5-0.8B` from Hugging Face. Prefer repo-root `.env` with `HF_TOKEN=...` (loaded by `src/utils/hf_env.py`; not printed by `environment/check_env.py`).
