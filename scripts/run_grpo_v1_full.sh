#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
python -m src.training.grpo \
  --config configs/base.yaml configs/data.yaml configs/grpo_v1_3090.yaml \
  "$@"
