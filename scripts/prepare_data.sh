#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
# English-only main splits: random 90/5/5 (seed in configs/data.yaml), processed SFT/DPO/GRPO from EN fields.
python -m src.data.split_dataset --config configs/base.yaml configs/data.yaml
echo "Data ready: data/splits + data/processed"
