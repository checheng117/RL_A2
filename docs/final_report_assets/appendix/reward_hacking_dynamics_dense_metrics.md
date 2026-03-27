# Reward hacking dynamics — dense GRPO-V1 (E2 supplement)

Test input: `/home/cc/Project/CC/RL_A2/data/processed/grpo_test.jsonl` (81 rows; same prompts as `sft_test.jsonl`).

| run_id | family | step | ROUGE-L | loose | strict | len(tok) | avg_reward |
|---|---:|---:|---:|---:|---:|---:|---|
| step0_sft_best | SFT | 0 | 0.4429 | 1.000 | 1.000 | 119.7 | — |
| grpo_v1_e2_dense_step50 | GRPO-V1-dense | 50 | 0.4461 | 1.000 | 1.000 | 119.7 | 1.0000 |
| grpo_v1_e2_dense_step100 | GRPO-V1-dense | 100 | 0.4440 | 1.000 | 1.000 | 118.7 | 1.0000 |
| grpo_v1_e2_dense_step150 | GRPO-V1-dense | 150 | 0.4420 | 1.000 | 0.988 | 124.4 | 1.0000 |
| grpo_v1_e2_dense_step200 | GRPO-V1-dense | 200 | 0.4476 | 1.000 | 1.000 | 119.0 | 1.0000 |
| grpo_v1_e2_dense_step250 | GRPO-V1-dense | 250 | 0.4458 | 1.000 | 1.000 | 119.4 | 1.0000 |
| grpo_v1_e2_dense_step300 | GRPO-V1-dense | 300 | 0.4414 | 1.000 | 1.000 | 121.0 | 1.0000 |

Per-run metrics JSON: `outputs/metrics/e2_dynamics_dense/*.json`