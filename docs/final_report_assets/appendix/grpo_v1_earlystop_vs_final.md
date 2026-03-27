# GRPO-V1 early stop proxy: step **700** vs **final (723)**

No extra training: we compare **saved** adapters only. Ideal early targets (200/300) are **not** on disk;
**700** is the latest intermediate checkpoint available for both V1 and V4.

| Checkpoint | avg_reward (v1) | ROUGE-L | strict | len(tok) | loose |
|---|---:|---:|---:|---:|---:|
| step 700 | 1.0 | 0.4428 | 0.988 | 123.6 | 1.000 |
| final | 1.0 | 0.4396 | 0.988 | 123.6 | 1.000 |

## Takeaway

- ROUGE-L: **-0.0032** (final − early).
- Strict format rate: **+0.0000**.
- Mean output tokens: **+0.0**.
- avg_reward: **+0.0000**.

Interpret alongside `grpo_v1_earlystop_qualitative.md` (aligned row indices).