"""Write side-by-side SFT vs DPO test metrics for the report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sft_metrics", type=str, default="outputs/metrics/sft_test_metrics.json")
    p.add_argument("--dpo_metrics", type=str, default="outputs/metrics/dpo_test_metrics.json")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/sft_vs_dpo_metrics.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/sft_vs_dpo_metrics.md")
    return p


def _loose_rate(m: dict) -> float | None:
    fa = m.get("format_adherence") or {}
    r = fa.get("format_rate")
    return float(r) if r is not None else None


def _strict_rate(m: dict) -> float | None:
    fs = m.get("format_adherence_strict") or {}
    r = m.get("strict_format_rate")
    if r is not None:
        return float(r)
    r2 = fs.get("strict_format_rate")
    return float(r2) if r2 is not None else None


def _row(model_name: str, m: dict, notes: str) -> dict[str, str]:
    loose = _loose_rate(m)
    strict = _strict_rate(m)
    return {
        "Model": model_name,
        "format_rate_loose": f"{loose:.3f}" if loose is not None else "",
        "strict_format_rate": f"{strict:.3f}" if strict is not None else "",
        "ROUGE_L_F1": f"{float(m.get('rouge_l_f1', 0)):.4f}",
        "avg_output_length_tokens": f"{float(m.get('avg_output_length_tokens', 0)):.2f}",
        "checkpoint_path": str(m.get("checkpoint", "")),
        "Notes": notes,
    }


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    sft_p = resolve_path(args.sft_metrics, root)
    dpo_p = resolve_path(args.dpo_metrics, root)
    out_csv = resolve_path(args.out_csv, root)
    out_md = resolve_path(args.out_md, root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not sft_p.is_file():
        raise SystemExit(f"Missing {sft_p}")
    if not dpo_p.is_file():
        raise SystemExit(f"Missing {dpo_p}")

    with open(sft_p, encoding="utf-8") as f:
        sft = json.load(f)
    with open(dpo_p, encoding="utf-8") as f:
        dpo = json.load(f)

    rows = [
        _row(
            "SFT (full fine-tune)",
            sft,
            "English main task; greedy decode on test split.",
        ),
        _row(
            "DPO (LoRA on SFT init)",
            dpo,
            "Same test JSONL and tokenizer-length metric as SFT.",
        ),
    ]

    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    md = [
        "# SFT vs DPO (test split, English)",
        "",
        "- **format_rate_loose**: legacy metric (tags + coarse order); can stay 1.0 when blocks repeat.",
        "- **strict_format_rate**: exactly one `[point]`/`[reason]`/`[summary]`, correct order, non-empty sections, ≥2 numbered lines in `[reason]`.",
        "",
        "Lengths are **mean output tokens** (checkpoint tokenizer, `add_special_tokens=False`).",
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
