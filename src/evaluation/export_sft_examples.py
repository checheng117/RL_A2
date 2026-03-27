"""Export SFT test examples (answer_en, reference, prediction, format, ROUGE-L) to markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.metrics.teacher_format import check_format
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export SFT test examples markdown")
    p.add_argument("--predictions_jsonl", type=str, required=True)
    p.add_argument("--output_md", type=str, required=True)
    p.add_argument("--n", type=int, default=5)
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    pred_path = resolve_path(args.predictions_jsonl, root)
    out_path = resolve_path(args.output_md, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    n = min(args.n, len(rows))
    pick = rows[:n]

    lines = [
        "# SFT test examples (English main task)",
        "",
        "Metrics aligned with assignment: format adherence, ROUGE-L vs `summary_en_chosen`, avg length in aggregate metrics JSON.",
        "",
        f"Predictions: `{pred_path}`",
        "",
    ]
    for i, r in enumerate(pick, 1):
        pred = (r.get("prediction") or "").strip()
        ref = ((r.get("summary_en_chosen") or r.get("response") or "").strip())
        ans_en = (r.get("answer_en") or "").strip()
        rouge_f = scorer.score(ref or " ", pred or " ")["rougeL"].fmeasure
        adherent = check_format(pred)
        lines.append(f"## Example {i}")
        lines.append("### answer_en (source)")
        lines.append("```")
        lines.append(ans_en[:4000] + ("..." if len(ans_en) > 4000 else ""))
        lines.append("```")
        lines.append("### summary_en_chosen (reference)")
        lines.append("```")
        lines.append(ref[:4000])
        lines.append("```")
        lines.append("### Model output")
        lines.append("```")
        lines.append(pred)
        lines.append("```")
        lines.append(f"- **Format adherence (structured tags)**: {'yes' if adherent else 'no'}")
        lines.append(f"- **ROUGE-L F (vs summary_en_chosen)**: {rouge_f:.4f}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
