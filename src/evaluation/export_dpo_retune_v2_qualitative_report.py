"""Qualitative: SFT vs DPO retune v1 vs v2 on same qualitative indices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.metrics.strict_format_adherence import strict_format_adherence_one
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--indices_json", type=str, default="outputs/report_assets/qualitative_pick_indices.json")
    p.add_argument("--sft_predictions", type=str, default="outputs/predictions/sft_test_greedy.jsonl")
    p.add_argument("--retune1_predictions", type=str, default="outputs/predictions/dpo_retune_test_greedy.jsonl")
    p.add_argument("--retune2_predictions", type=str, default="outputs/predictions/dpo_retune_v2_test_greedy.jsonl")
    p.add_argument("--output_md", type=str, default="outputs/report_assets/dpo_retune_v2_qualitative_analysis.md")
    return p


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _strict_label(text: str) -> str:
    return "Pass" if strict_format_adherence_one(text)["pass_strict"] else "Fail"


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    idx_path = resolve_path(args.indices_json, root)
    sft_p = resolve_path(args.sft_predictions, root)
    r1_p = resolve_path(args.retune1_predictions, root)
    r2_p = resolve_path(args.retune2_predictions, root)
    out_md = resolve_path(args.output_md, root)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    for p in (idx_path, sft_p, r1_p, r2_p):
        if not p.is_file():
            raise SystemExit(f"Missing {p}")

    indices = json.loads(idx_path.read_text(encoding="utf-8")).get("indices") or []
    sft_rows, r1_rows, r2_rows = _load_rows(sft_p), _load_rows(r1_p), _load_rows(r2_p)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    lines = [
        "# DPO retune v2 qualitative analysis",
        "",
        f"Indices from qualitative_pick_indices.json: {indices}.",
        "Models: SFT, DPO retune v1, DPO retune v2 (same greedy decoding).",
        "",
    ]

    for k, idx in enumerate(indices, 1):
        if idx >= len(sft_rows) or idx >= len(r1_rows) or idx >= len(r2_rows):
            continue
        sr, t1, t2 = sft_rows[idx], r1_rows[idx], r2_rows[idx]
        ref = (sr.get("summary_en_chosen") or "").strip()
        ps = (sr.get("prediction") or "").strip()
        p1 = (t1.get("prediction") or "").strip()
        p2 = (t2.get("prediction") or "").strip()
        r_ref = ref or " "
        rl_s = scorer.score(r_ref, ps or " ")["rougeL"].fmeasure
        rl_1 = scorer.score(r_ref, p1 or " ")["rougeL"].fmeasure
        rl_2 = scorer.score(r_ref, p2 or " ")["rougeL"].fmeasure
        ss, s1, s2 = _strict_label(ps), _strict_label(p1), _strict_label(p2)

        def clip(t: str, n: int = 1800) -> str:
            return t[:n] + ("..." if len(t) > n else "")

        lines.append(f"## Example {k} (row index {idx})")
        lines.append(f"- **Example ID**: {k} (index {idx})")
        lines.append("- **Reference summary**:")
        lines.append("```")
        lines.append(clip(ref))
        lines.append("```")
        lines.append("### SFT prediction")
        lines.append("```")
        lines.append(clip(ps))
        lines.append("```")
        lines.append(f"- strict_format: **{ss}**; ROUGE-L: **{rl_s:.4f}**")
        lines.append("### DPO retune v1")
        lines.append("```")
        lines.append(clip(p1))
        lines.append("```")
        lines.append(f"- strict_format: **{s1}**; ROUGE-L: **{rl_1:.4f}**")
        lines.append("### DPO retune v2")
        lines.append("```")
        lines.append(clip(p2))
        lines.append("```")
        lines.append(f"- strict_format: **{s2}**; ROUGE-L: **{rl_2:.4f}**")
        note = []
        if rl_2 > rl_1 + 0.02:
            note.append("v2 higher ROUGE-L than v1.")
        elif rl_1 > rl_2 + 0.02:
            note.append("v1 higher ROUGE-L than v2.")
        else:
            note.append("Similar ROUGE-L v1 vs v2.")
        if s2 == "Pass" and s1 == "Fail":
            note.append("strict fixed in v2.")
        if s2 == "Fail" and s1 == "Pass":
            note.append("strict worse in v2.")
        if len(p2) < len(p1) * 0.9:
            note.append("v2 shorter.")
        if len(p2) > len(p1) * 1.1:
            note.append("v2 longer.")
        lines.append(f"- **Notes**: {' '.join(note)}")
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            "### Retune v2 vs v1",
            "See aggregate `dpo_retune_v2_test_metrics.json` and four-way table.",
            "",
            "### Open issues",
            "May still lag SFT on length, strict, or ROUGE-L.",
            "",
            "### Part II final pick",
            "Prefer the variant with best ROUGE-L among DPO runs without collapsing strict; document gaps vs SFT honestly.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
