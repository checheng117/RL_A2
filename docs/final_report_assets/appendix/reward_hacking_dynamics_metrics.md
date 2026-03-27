# Reward hacking dynamics — checkpoint-level metrics (E2)

Test input: `/home/cc/Project/CC/RL_A2/data/processed/grpo_test.jsonl` (81 rows; same prompts as `sft_test.jsonl`).

| run_id | family | step | ROUGE-L | loose | strict | len(tok) | avg_reward |
|---|---:|---:|---:|---:|---:|---:|---|
| step0_sft_best | SFT | 0 | 0.4408 | 1.000 | 0.988 | 124.5 | — |
| grpo_v1_checkpoint_700 | GRPO-V1 | 700 | 0.4428 | 1.000 | 0.988 | 123.6 | 1.0000 |
| grpo_v1_final | GRPO-V1 | 723 | 0.4396 | 1.000 | 0.988 | 123.6 | 1.0000 |
| grpo_v4_checkpoint_700 | GRPO-V4 | 700 | 0.4480 | 1.000 | 1.000 | 118.7 | 0.4990 |
| grpo_v4_final | GRPO-V4 | 723 | 0.4495 | 1.000 | 1.000 | 119.5 | 0.4992 |

Per-run metrics JSON: `outputs/metrics/e2_dynamics/*.json`