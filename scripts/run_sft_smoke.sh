#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
# Smoke uses teacher-aligned full SFT trainer (tiny subset); QLoRA entry remains `src.training.sft` + legacy `text` field if you add it.
python -m src.training.sft_hf_trainer \
  --config configs/base.yaml configs/data.yaml configs/sft_full_3090.yaml \
  --smoke_test
