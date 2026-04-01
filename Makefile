ROOT := $(abspath .)
export PYTHONPATH := $(ROOT):$(PYTHONPATH)

.PHONY: env-check prepare-data sft-full dpo-retune-v2 grpo-v1-full grpo-v4-full test lint

env-check:
	python environment/check_env.py

prepare-data:
	bash scripts/prepare_data.sh

sft-full:
	bash scripts/run_sft_full.sh

dpo-retune-v2:
	bash scripts/run_dpo_retune_v2.sh

grpo-v1-full:
	bash scripts/run_grpo_v1_full.sh

grpo-v4-full:
	bash scripts/run_grpo_v4_full.sh

test:
	pytest -q tests/

lint:
	python -m compileall -q src
