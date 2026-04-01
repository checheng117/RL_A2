# Public Results Snapshot

This folder provides a small, public verification window for key report claims.

It intentionally includes only compact final CSV summaries, not full predictions, logs, checkpoints, or process notes.

## Files

- `unified_part34_metrics.csv`  
  Unified comparison across SFT / DPO-retune-v2 / GRPO-V1 / GRPO-V4.

- `paired_bootstrap_ci_summary.csv`  
  Bootstrap confidence intervals for key ROUGE-L mean and deltas.

- `pairwise_win_tie_loss.csv`  
  Per-example pairwise win/tie/loss summary on the shared test split.

- `reward_hacking_dynamics_metrics.csv`  
  Sparse checkpoint dynamics used for Part V-style trend checking.

- `e2_kl_probe_metrics.csv`  
  KL-schedule probe comparison in the E2 dense setup.

- `seed_sensitivity_summary.csv`  
  Small cross-seed snapshot for stability sanity check.

- `qualitative_case_snapshot.csv`  
  Minimal 4-case snapshot (2x GRPO-V1, 2x GRPO-V4) for case-level mismatch checks.

## Notes

- These files are exported snapshots for public verification only.
- Local absolute paths and large artifacts are intentionally removed from this directory.
- For full course deliverables and appendices, refer to the official submission package.
