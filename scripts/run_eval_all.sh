#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
CFG="configs/base.yaml configs/inference.yaml"

run_one() {
  local stage="$1" ck="$2"
  if [[ -d "$ck" ]]; then
    python -m src.evaluation.evaluate --config $CFG --stage "$stage" --checkpoint "$ck"
  else
    echo "Skip $stage (no checkpoint at $ck)"
  fi
}

if [[ -d outputs/checkpoints/sft_full_3090/best ]]; then
  run_one sft outputs/checkpoints/sft_full_3090/best
else
  run_one sft outputs/checkpoints/sft_lora_3090/best
fi
run_one dpo outputs/checkpoints/dpo_lora_3090/best
run_one grpo_v1 outputs/checkpoints/grpo_v1_3090/best
run_one grpo_v4 outputs/checkpoints/grpo_v4_3090/best

python -m src.evaluation.summarize_results

bash "$ROOT/scripts/export_report_assets.sh"
