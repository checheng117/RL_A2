ROOT := $(abspath .)
export PYTHONPATH := $(ROOT):$(PYTHONPATH)

.PHONY: env-check prepare-data smoke-sft smoke-dpo smoke-grpo-v1 smoke-grpo-v4 eval-all report-assets test lint

env-check:
	python environment/check_env.py

prepare-data:
	bash scripts/prepare_data.sh

sft-full:
	bash scripts/run_sft_full.sh

smoke-sft:
	bash scripts/run_sft_smoke.sh

smoke-dpo:
	bash scripts/run_dpo_smoke.sh

smoke-grpo-v1:
	bash scripts/run_grpo_v1_smoke.sh

smoke-grpo-v4:
	bash scripts/run_grpo_v4_smoke.sh

eval-all:
	bash scripts/run_eval_all.sh

report-assets:
	bash scripts/export_report_assets.sh

test:
	pytest -q tests/

lint:
	python -m compileall -q src
