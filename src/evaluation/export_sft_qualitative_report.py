"""Stratified qualitative analysis markdown for report (Part I)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.metrics.teacher_format import check_format
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export report-style SFT qualitative analysis")
    p.add_argument("--predictions_jsonl", type=str, required=True)
    p.add_argument("--output_md", type=str, default="outputs/report_assets/sft_qualitative_analysis.md")
    p.add_argument("--indices_json", type=str, default="outputs/report_assets/qualitative_pick_indices.json")
    return p


def _faithfulness(rouge_f: float, fmt_ok: bool) -> tuple[str, str]:
    if not fmt_ok:
        return "Poor", "Format failure undermines comparability and often indicates template drift."
    if rouge_f >= 0.52:
        return "Good", "High lexical overlap with reference while keeping bracket structure."
    if rouge_f >= 0.30:
        return "Partial", "Captures part of the reference gist; may omit a reason or rephrase heavily."
    return "Poor", "Low overlap with reference; risk of omission, invention, or off-focus summary."


def _pick_indices(n: int, rouges: list[float]) -> list[int]:
    """Two high-ROUGE, two mid-rank, one low-ROUGE (by sorted rank)."""
    order = sorted(range(n), key=lambda i: rouges[i], reverse=True)
    seq: list[int] = [order[0]]
    if n > 1:
        seq.append(order[1])
    if n > 4:
        seq.append(order[n // 2])
        seq.append(order[min(n // 2 + 1, n - 1)])
    seq.append(order[-1])
    out: list[int] = []
    for p in seq:
        if p not in out:
            out.append(p)
    for p in order:
        if len(out) >= 5:
            break
        if p not in out:
            out.append(p)
    return out[:5]


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    pred_path = resolve_path(args.predictions_jsonl, root)
    out_md = resolve_path(args.output_md, root)
    idx_path = resolve_path(args.indices_json, root)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouges: list[float] = []
    for r in rows:
        pred = (r.get("prediction") or "").strip()
        ref = (r.get("summary_en_chosen") or r.get("response") or "").strip()
        rouges.append(scorer.score(ref or " ", pred or " ")["rougeL"].fmeasure)

    n = len(rows)
    indices = list(range(n)) if n <= 5 else _pick_indices(n, rouges)
    idx_path.write_text(json.dumps({"indices": indices, "source_predictions": str(pred_path)}, indent=2), encoding="utf-8")

    lines: list[str] = [
        "# SFT qualitative analysis (English test split)",
        "",
        "Representative examples were chosen by **stratifying on per-example ROUGE-L** (two high, two mid-rank, one low) so the discussion spans success and failure modes.",
        "",
        f"Predictions file: `{pred_path}`",
        "",
    ]

    strengths: list[str] = []
    failures: list[str] = []

    for k, idx in enumerate(indices, 1):
        r = rows[idx]
        pred = (r.get("prediction") or "").strip()
        ref = (r.get("summary_en_chosen") or r.get("response") or "").strip()
        ans = (r.get("answer_en") or "").strip()
        rouge_f = rouges[idx]
        fmt_ok = check_format(pred)
        fj, note_f = _faithfulness(rouge_f, fmt_ok)

        ans_show = ans[:2200] + ("\n\n[…truncated…]" if len(ans) > 2200 else "")

        if rouge_f >= 0.5 and fmt_ok:
            strengths.append(
                f"Example {k} (test index {idx}) reaches strong ROUGE-L ({rouge_f:.3f}) with valid `[point]/[reason]/[summary]` structure."
            )
        if rouge_f < 0.35 or not fmt_ok:
            failures.append(
                f"Example {k} (index {idx}): ROUGE-L {rouge_f:.3f}, format {'OK' if fmt_ok else 'FAIL'} — see notes below."
            )

        extra = []
        if len(pred) > len(ref) * 1.4:
            extra.append("Model output is more verbose than reference (especially in `[reason]`).")
        if rouge_f < 0.4 and fmt_ok:
            extra.append("Structure is correct but content diverges from reference (possible paraphrase or missed detail).")
        if not extra:
            extra.append("See prediction vs reference side-by-side.")

        lines.append(f"## Example {k} (test row index `{idx}`)")
        lines.append(f"- **Example ID**: `test_idx_{idx}`")
        lines.append(f"- **Format adherence**: **{'Pass' if fmt_ok else 'Fail'}**")
        lines.append(f"- **ROUGE-L (vs summary_en_chosen)**: **{rouge_f:.4f}**")
        lines.append(f"- **Faithfulness judgement**: **{fj}** — {note_f}")
        lines.append("### Source answer (`answer_en`)")
        lines.append("```")
        lines.append(ans_show)
        lines.append("```")
        lines.append("### Reference (`summary_en_chosen`)")
        lines.append("```")
        lines.append(ref)
        lines.append("```")
        lines.append("### Model prediction")
        lines.append("```")
        lines.append(pred)
        lines.append("```")
        lines.append(f"- **Notes**: {' '.join(extra)}")
        lines.append("")

    if not strengths:
        strengths.append("Even when ROUGE-L is moderate, outputs often preserve the bracketed scaffold, which helps downstream grading.")
    if not failures:
        failures.append("Main residual risk is paraphrase: correct format with partial semantic match to `summary_en_chosen`.")

    lines.extend(
        [
            "## Common strengths",
            "",
            *[f"- {s}" for s in strengths],
            "",
            "## Common failure modes",
            "",
            *[f"- {s}" for s in failures],
            "",
        ]
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md} and {idx_path}")


if __name__ == "__main__":
    main()
