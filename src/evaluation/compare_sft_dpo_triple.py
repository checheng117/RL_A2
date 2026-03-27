"""SFT vs original DPO vs DPO retune — one Markdown/CSV table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluation.compare_sft_dpo import _row
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_metrics", type=str, default="outputs/metrics/sft_test_metrics.json")
    p.add_argument("--dpo_metrics", type=str, default="outputs/metrics/dpo_test_metrics.json")
    p.add_argument("--dpo_retune_metrics", type=str, default="outputs/metrics/dpo_retune_test_metrics.json")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/sft_vs_dpo_vs_dpo_retune_metrics.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/sft_vs_dpo_vs_dpo_retune_metrics.md")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    paths = [
        ("SFT (full fine-tune)", resolve_path(args.sft_metrics, root), "Baseline."),
        ("DPO (original)", resolve_path(args.dpo_metrics, root), "Original DPO LoRA run."),
        ("DPO (retune)", resolve_path(args.dpo_retune_metrics, root), "Light hyperparameter retune."),
    ]
    out_csv = resolve_path(args.out_csv, root)
    out_md = resolve_path(args.out_md, root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    loaded: list[tuple[str, dict, str]] = []
    for name, p, note in paths:
        if not p.is_file():
            raise SystemExit(f"Missing {p}")
        with open(p, encoding="utf-8") as f:
            loaded.append((name, json.load(f), note))

    rows = [_row(name, m, note) for name, m, note in loaded]
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# SFT vs DPO vs DPO retune",
        "",
        "| Model | format loose | strict | ROUGE-L | Avg. len (tok) | Checkpoint | Notes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['Model']} | {r['format_rate_loose']} | {r['strict_format_rate']} | "
            f"{r['ROUGE_L_F1']} | {r['avg_output_length_tokens']} | `{r['checkpoint_path']}` | {r['Notes']} |"
        )
    md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
