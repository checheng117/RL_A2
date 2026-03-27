"""Four-way table: SFT, DPO original, DPO retune v1, DPO retune v2."""
from __future__ import annotations

import argparse
import csv
import json

from src.evaluation.compare_sft_dpo import _row
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_metrics", type=str, default="outputs/metrics/sft_test_metrics.json")
    p.add_argument("--dpo_metrics", type=str, default="outputs/metrics/dpo_test_metrics.json")
    p.add_argument("--dpo_retune_metrics", type=str, default="outputs/metrics/dpo_retune_test_metrics.json")
    p.add_argument("--dpo_retune_v2_metrics", type=str, default="outputs/metrics/dpo_retune_v2_test_metrics.json")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/sft_vs_dpo_all_metrics.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/sft_vs_dpo_all_metrics.md")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    specs = [
        ("SFT (full fine-tune)", args.sft_metrics, "Stable baseline; teacher-aligned greedy outputs."),
        ("DPO (original)", args.dpo_metrics, "Repeated multi-block outputs; ROUGE collapse vs SFT."),
        ("DPO retune v1", args.dpo_retune_metrics, "Improved strict + ROUGE vs original; still verbose vs SFT."),
        ("DPO retune v2", args.dpo_retune_v2_metrics, "Second conservative pass; see table for vs v1."),
    ]
    rows: list[dict[str, str]] = []
    for name, rel, note in specs:
        p = resolve_path(rel, root)
        if not p.is_file():
            raise SystemExit(f"Missing {p}")
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        rows.append(_row(name, m, note))

    fieldnames = [
        "Model",
        "ROUGE_L_F1",
        "format_rate_loose",
        "strict_format_rate",
        "avg_output_length_tokens",
        "checkpoint_path",
        "Notes",
    ]
    out_rows = []
    for r in rows:
        out_rows.append(
            {
                "Model": r["Model"],
                "ROUGE_L_F1": r["ROUGE_L_F1"],
                "format_rate_loose": r["format_rate_loose"],
                "strict_format_rate": r["strict_format_rate"],
                "avg_output_length_tokens": r["avg_output_length_tokens"],
                "checkpoint_path": r["checkpoint_path"],
                "Notes": r["Notes"],
            }
        )

    out_csv = resolve_path(args.out_csv, root)
    out_md = resolve_path(args.out_md, root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    md = [
        "# SFT vs DPO (all stages, test split, English)",
        "",
        "Same decoding for all models (configs/inference.yaml, greedy).",
        "",
        "| Model | ROUGE-L | format loose | strict | len (tok) | Checkpoint | Notes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in out_rows:
        md.append(
            f"| {r['Model']} | {r['ROUGE_L_F1']} | {r['format_rate_loose']} | {r['strict_format_rate']} | "
            f"{r['avg_output_length_tokens']} | `{r['checkpoint_path']}` | {r['Notes']} |"
        )
    md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
