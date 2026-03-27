#!/usr/bin/env bash
# Part V / E2: short GRPO-V1 run with dense checkpoints (dense dynamics only; not the official V1 table).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate rlhw2_qwen35_3090
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
python -m src.training.grpo \
  --config configs/base.yaml configs/data.yaml configs/grpo_v1_e2_dense.yaml \
  "$@"
