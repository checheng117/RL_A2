# Unified Part III / IV metrics

Same English test split, greedy decode (`configs/inference.yaml`). **Avg Reward** column uses **—** for SFT/DPO (not optimized with GRPO reward); see notes for diagnostic v4 score.

| Model | Avg Reward | ROUGE-L | loose | strict | len(tok) | faithful? | hacking? | Checkpoint | Notes |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| SFT (full) | — | 0.4408 | 1.000 | 0.988 | 124.46 | Yes | No | `/home/cc/Project/CC/RL_A2/outputs/checkpoints/sft_full_3090/best` | Supervised baseline; not RL-trained. (diagnostic v4=0.2621 in metrics JSON) |
| DPO retune v2 | — | 0.4448 | 1.000 | 0.988 | 121.30 | Yes | No | `/home/cc/Project/CC/RL_A2/outputs/checkpoints/dpo_lora_3090_retune_v2/best` | Best DPO in this repo run. (diagnostic v4=0.2606 in metrics JSON) |
| GRPO-V1 | 1.0000 | 0.4396 | 1.000 | 0.988 | 123.62 | Yes | Mild | `/home/cc/Project/CC/RL_A2/outputs/checkpoints/grpo_v1_3090/best` | RL with tag-only reward; hacking-prone. |
| GRPO-V4 | 0.4992 | 0.4495 | 1.000 | 1.000 | 119.48 | Yes | No | `/home/cc/Project/CC/RL_A2/outputs/checkpoints/grpo_v4_3090/best` | RL with richer reward; partial mitigation. |

_faithful? / hacking? are coarse heuristics for the report; see qualitative + hacking MD files._
