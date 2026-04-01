# RL_A2 Public Showcase Repository

This repository is a **public code + minimal verification window** for my RL assignment project on structured summarization (`SFT -> DPO -> GRPO`). It is intentionally curated for external readers (e.g., instructor, reviewer) to quickly understand the method, run core code paths, and check key reported metrics without exposing full course submission materials or local experiment workspace artifacts.

## What this repository contains

- Core implementation in `src/` (training, evaluation, rewards, metrics, inference)
- Final/primary experiment configurations in `configs/`
- Minimal runnable scripts in `scripts/`
- Environment setup files in `environment/`
- Test split manifest in `data/splits/split_manifest.json`
- A compact public verification window in `public_results/`

## What is intentionally NOT public

The following are intentionally excluded from this public repository and remain local/private:

- Report workspace and course documents (`report/`, course PDFs/DOCX)
- Local docs/work notes and process artifacts (`docs/`, `analysis/`)
- Teacher/reference bundle (`train_code_with_data/`)
- Large runtime artifacts (`outputs/checkpoints/`, `outputs/logs/`, predictions, full metrics dumps)
- Process-only scripts, one-off exports, scratch/temp/legacy backups

For complete course deliverables, use the official submission package instead of this GitHub repository.

## Repository structure

```text
.
|-- src/                # training/evaluation/reward/metric/inference code
|-- configs/            # experiment and inference configs
|-- scripts/            # minimal entry scripts kept for public use
|-- environment/        # requirements and environment checks
|-- data/
|   `-- splits/
|       `-- split_manifest.json
|-- public_results/     # small, curated CSV snapshot for result verification
|-- tests/
|-- README.md
`-- Makefile
```

## Minimal setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
python environment/check_env.py
```

Or use Conda:

```bash
bash scripts/setup_conda.sh
```

## Minimal reproduction commands

1) Prepare processed data/splits:

```bash
make prepare-data
```

2) Run core training stages (long-running):

```bash
make sft-full
make dpo-retune-v2
make grpo-v1-full
make grpo-v4-full
```

3) Run tests:

```bash
make test
```

## Public result snapshot (for quick verification)

The directory `public_results/` contains a compact set of final CSVs used as a public verification window:

- `unified_part34_metrics.csv`: unified SFT/DPO/GRPO comparison table
- `paired_bootstrap_ci_summary.csv`: bootstrap confidence intervals for key ROUGE comparisons
- `pairwise_win_tie_loss.csv`: per-example win/tie/loss summary
- `reward_hacking_dynamics_metrics.csv`: sparse E2 checkpoint dynamics summary
- `e2_kl_probe_metrics.csv`: E2 KL schedule probe result
- `seed_sensitivity_summary.csv`: cross-seed stability snapshot

See `public_results/README.md` for per-file notes and interpretation boundaries.

## Public vs submission scope

- This GitHub repo is a **showcase**: core code, configs, minimal scripts, and a small results snapshot.
- It is **not** a full mirror of local training artifacts or course submission materials.
- If a reviewer needs full report assets, large logs/checkpoints, or complete appendix materials, those are available only in the official submission package.

## License

Course/research use only. Please comply with model licenses and course policies.
