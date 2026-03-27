#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"
mkdir -p outputs/report_assets outputs/examples

python -m src.evaluation.summarize_results \
  --out_csv outputs/report_assets/metrics_summary.csv \
  --out_md outputs/report_assets/metrics_summary.md

python -m src.evaluation.export_sft_loss_curves || true

PRED="outputs/predictions/sft_test_greedy.jsonl"
if [[ -f "$PRED" ]]; then
  python -m src.evaluation.export_examples \
    --predictions_jsonl "$PRED" \
    --output_md outputs/report_assets/selected_examples.md
  python -m src.evaluation.export_sft_examples \
    --predictions_jsonl "$PRED" \
    --output_md outputs/examples/sft_test_examples.md \
    --n 5
  cp -f outputs/examples/sft_test_examples.md outputs/report_assets/sft_test_examples.md 2>/dev/null || true
  python -m src.evaluation.export_sft_qualitative_report --predictions_jsonl "$PRED" || true
else
  cat > outputs/report_assets/selected_examples.md << 'EOF'
# Selected examples

_No predictions yet._ Run `bash scripts/run_eval_all.sh` after training.
EOF
fi

cat > outputs/report_assets/reward_hacking_cases.md << 'EOF'
# Reward hacking cases (template)

Document cases where **reward is high** but **quality is low**.

## Case 1 — GRPO-V1
- **Output**:
- **Reward**:
- **Why it hacks**:

## Case 2 — GRPO-V4
- **Output**:
- **Reward**:
- **Why it hacks**:
EOF

if [[ -f outputs/metrics/dpo_test_metrics.json ]]; then
  python -m src.evaluation.compare_sft_dpo || true
  python -m src.evaluation.export_dpo_qualitative_report || true
fi
if [[ -f outputs/metrics/dpo_retune_test_metrics.json ]]; then
  python -m src.evaluation.compare_sft_dpo_triple || true
  python -m src.evaluation.export_pre_grpo_decision || true
fi

cat > outputs/report_assets/training_curves_grpo_note.md << 'EOF'
# GRPO curves (if applicable)

SFT loss figure: `outputs/report_assets/sft_loss_curves.png` (and CSV `sft_loss_history.csv`).
For DPO / GRPO, parse `outputs/checkpoints/<run>/checkpoint-*/trainer_state.json` similarly.
EOF

echo "Report assets under outputs/report_assets/"
