# GRPO-V1 E2-dense: early vs final

- **Early (auto-picked):** `grpo_v1_e2_dense_step200` — step **200** (among steps with avg_reward≥0.95 before final, maximize `ROUGE + 0.5*strict - 0.015*len/10`).
- **Final:** `grpo_v1_e2_dense_step300` — step **300**.

| | avg_reward | ROUGE-L | strict | len(tok) |
|--:|---:|---:|---:|---:|
| early | 1.0000 | 0.4476 | 1.000 | 119.0 |
| final | 1.0000 | 0.4414 | 1.000 | 121.0 |
| Δ (final − early) | +0.0000 | -0.0062 | +0.0000 | +2.0 |

## Conclusion (honest)

The auto-picked **early** checkpoint has **higher ROUGE-L** than dense final on this split — early stopping looks **more balanced** under the stated quality proxies (still small n=81).

See qualitative aligned rows in `grpo_v1_earlystop_dense_qualitative.md`.