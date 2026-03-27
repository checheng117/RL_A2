"""Aggregate per-run metrics JSON into CSV/Markdown tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_glob", type=str, default="outputs/metrics/*_test_metrics.json")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/metrics_summary.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/metrics_summary.md")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    paths = sorted(root.glob(args.metrics_glob))
    if not paths:
        paths = sorted((root / "outputs" / "metrics").glob("*_test_metrics.json"))

    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))

    out_csv = resolve_path(args.out_csv, root)
    out_md = resolve_path(args.out_md, root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out_csv.write_text(
            "stage,n,rouge_l_f1,format_rate,strict_format_rate,avg_output_length_tokens,avg_output_length_chars,avg_reward,reward_variant\n",
            encoding="utf-8",
        )
        out_md.write_text("# Metrics summary\n\n_No metric files found._\n", encoding="utf-8")
        print("No metrics found")
        return

    fieldnames = [
        "stage",
        "n",
        "rouge_l_f1",
        "format_rate",
        "strict_format_rate",
        "avg_output_length_tokens",
        "avg_output_length_chars",
        "avg_reward",
        "reward_variant",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            fmt = r.get("format_adherence") or {}
            fr = fmt.get("format_rate", "")
            sr = r.get("strict_format_rate", "")
            if sr == "" and r.get("format_adherence_strict"):
                sr = (r.get("format_adherence_strict") or {}).get("strict_format_rate", "")
            flat = {
                "stage": r.get("stage", ""),
                "n": r.get("n", ""),
                "rouge_l_f1": r.get("rouge_l_f1", ""),
                "format_rate": fr,
                "strict_format_rate": sr,
                "avg_output_length_tokens": r.get("avg_output_length_tokens", r.get("avg_output_length", "")),
                "avg_output_length_chars": r.get("avg_output_length_chars", ""),
                "avg_reward": r.get("avg_reward", ""),
                "reward_variant": r.get("reward_variant", ""),
            }
            w.writerow(flat)

    def _fmt(x, spec: str, default: str = "") -> str:
        try:
            return format(float(x), spec)
        except (TypeError, ValueError):
            return default or str(x)

    md = [
        "# Metrics summary",
        "",
        "Mean output length uses **tokenizer tokens** (`avg_output_length_tokens`); chars are auxiliary.",
        "",
        "| stage | n | ROUGE-L | format loose | strict | len(tok) | len(chars) | avg reward | reward var |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        fmt = r.get("format_adherence") or {}
        fr = fmt.get("format_rate", "")
        sr = r.get("strict_format_rate", "")
        if sr == "" and r.get("format_adherence_strict"):
            sr = (r.get("format_adherence_strict") or {}).get("strict_format_rate", "")
        tok = r.get("avg_output_length_tokens", r.get("avg_output_length", ""))
        ch = r.get("avg_output_length_chars", "")
        md.append(
            f"| {r.get('stage','')} | {r.get('n','')} | {_fmt(r.get('rouge_l_f1'), '.4f')} | "
            f"{_fmt(fr, '.3f') if fr != '' else ''} | {_fmt(sr, '.3f') if sr != '' else ''} | "
            f"{_fmt(tok, '.1f')} | {_fmt(ch, '.1f')} | "
            f"{_fmt(r.get('avg_reward'), '.4f')} | {r.get('reward_variant','')} |"
        )
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
