# CSC6129 RL_A2

This is the public code repository for my CSC6129 reinforcement learning assignment.  
The project studies structured summarization with a staged alignment pipeline (`SFT -> DPO -> GRPO`).  
For this public release, I keep the core implementation, experiment configs, and a small set of result snapshots so that an external reader can quickly understand what was done and sanity-check the main claims.

## What you can find here

- Main implementation in `src/` (training, evaluation, rewards, metrics, inference)
- Experiment configs in `configs/`
- Minimal entry scripts in `scripts/`
- Environment setup files in `environment/`
- Split manifest in `data/splits/split_manifest.json`
- A compact result snapshot in `public_results/`

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
|-- public_results/     # compact CSV snapshot for quick verification
|-- tests/
|-- README.md
`-- Makefile
```

## Setup

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

## Main entry points

The commands below are kept to document the primary code paths used in the project.

1) Prepare processed data/splits:

```bash
make prepare-data
```

The public repo does not include the full raw course dataset, so data preparation and full reruns require access to the non-public dataset bundle.

2) Run core training stages (long-running):

```bash
make sft-full
make dpo-retune-v2
make grpo-v1-full
make grpo-v4-full
```

These are the main training entry points, but running the full pipeline still depends on non-public data and local runtime artifacts (for example, checkpoint/logging workspace).

3) Run tests:

```bash
make test
```

## Public result files

`public_results/` contains a small set of CSV artifacts for quick cross-checking of reported outcomes:

- `unified_part34_metrics.csv`: unified SFT/DPO/GRPO comparison
- `paired_bootstrap_ci_summary.csv`: bootstrap confidence intervals for key ROUGE comparisons
- `pairwise_win_tie_loss.csv`: pairwise win/tie/loss summary on the shared test split
- `reward_hacking_dynamics_metrics.csv`: sparse E2 checkpoint dynamics summary
- `e2_kl_probe_metrics.csv`: E2 KL-schedule probe
- `seed_sensitivity_summary.csv`: seed sensitivity snapshot
- `qualitative_case_snapshot.csv`: minimal case-level examples (GRPO-V1/GRPO-V4) supporting qualitative mismatch discussion

These files are meant as lightweight evidence for key conclusions, not as a full release of all experiment artifacts.

## License

Course/research use only. Please comply with model licenses and course policies.
