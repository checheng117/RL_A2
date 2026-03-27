#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
python -m src.training.dpo \
  --config configs/base.yaml configs/data.yaml configs/dpo_lora_3090_retune_v2.yaml \
  "$@"
