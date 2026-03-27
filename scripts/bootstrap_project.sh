#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p data/raw data/processed data/splits
mkdir -p outputs/checkpoints outputs/logs outputs/predictions outputs/metrics outputs/examples outputs/report_assets
mkdir -p configs src tests

for d in outputs/checkpoints outputs/logs outputs/predictions outputs/metrics outputs/examples outputs/report_assets data/raw data/processed data/splits; do
  touch "$d/.gitkeep"
done

echo "Bootstrap done under $ROOT"
