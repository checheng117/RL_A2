"""Report-style DPO vs SFT comparison on the same test indices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.metrics.teacher_format import check_format
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--indices_json", type=str, default="outputs/report_assets/qualitative_pick_indices.json")
    p.add_argument("--sft_predictions", type=str, default="outputs/predictions/sft_test_greedy.jsonl")
    p.add_argument("--dpo_predictions", type=str, default="outputs/predictions/dpo_test_greedy.jsonl")
    p.add_argument("--output_md", type=str, default="outputs/report_assets/dpo_qualitative_analysis.md")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    idx_path = resolve_path(args.indices_json, root)
    sft_p = resolve_path(args.sft_predictions, root)
    dpo_p = resolve_path(args.dpo_predictions, root)
    out_md = resolve_path(args.output_md, root)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if not idx_path.is_file():
        raise SystemExit(f"Missing {idx_path} — run export_sft_qualitative_report first.")
    indices = json.loads(idx_path.read_text(encoding="utf-8")).get("indices") or []

    def load_rows(path: Path) -> list[dict]:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    sft_rows = load_rows(sft_p)
    dpo_rows = load_rows(dpo_p)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    lines = [
        "# DPO vs SFT qualitative comparison",
        "",
        f"Same **test row indices** as `sft_qualitative_analysis.md`: `{indices}`.",
        "",
    ]

    for k, idx in enumerate(indices, 1):
        if idx >= len(sft_rows) or idx >= len(dpo_rows):
            continue
        sr, dr = sft_rows[idx], dpo_rows[idx]
        ref = (sr.get("summary_en_chosen") or "").strip()
        ps = (sr.get("prediction") or "").strip()
        pd_ = (dr.get("prediction") or "").strip()
        rs = scorer.score(ref or " ", ps or " ")["rougeL"].fmeasure
        rd = scorer.score(ref or " ", pd_ or " ")["rougeL"].fmeasure
        fs, fd = check_format(ps), check_format(pd_)

        struct = "DPO" if (fd and not fs) else ("SFT" if (fs and not fd) else "Similar")
        if fs and fd:
            struct = "Both structured"
        if not fs and not fd:
            struct = "Both imperfect"

        if rd > rs + 0.02:
            faith = "DPO closer to reference by ROUGE-L."
        elif rs > rd + 0.02:
            faith = "SFT closer to reference by ROUGE-L."
        else:
            faith = "Roughly comparable faithfulness (ROUGE-L)."

        flow = ""
        if len(pd_) < len(ps) * 0.85:
            flow = "DPO output is shorter / tighter."
        elif len(pd_) > len(ps) * 1.15:
            flow = "DPO output is longer; check for verbosity."
        else:
            flow = "Similar verbosity."

        lines.append(f"## Example {k} (index `{idx}`)")
        lines.append(f"- **Reference (`summary_en_chosen`)**:")
        lines.append("```")
        lines.append(ref[:2000])
        lines.append("```")
        lines.append("### SFT prediction")
        lines.append("```")
        lines.append(ps[:2000])
        lines.append("```")
        lines.append(f"- ROUGE-L: **{rs:.4f}**; format: **{'Pass' if fs else 'Fail'}**")
        lines.append("### DPO prediction")
        lines.append("```")
        lines.append(pd_[:2000])
        lines.append("```")
        lines.append(f"- ROUGE-L: **{rd:.4f}**; format: **{'Pass' if fd else 'Fail'}**")
        lines.append("### Comparison")
        lines.append(f"- **Structure**: {struct}.")
        lines.append(f"- **Faithfulness (proxy)**: {faith}")
        lines.append(f"- **Fluency / length**: {flow}")
        lines.append("")

    lines.extend(
        [
            "## Summary",
            "",
            "### DPO vs SFT — improvements",
            "- DPO may better separate **chosen vs rejected** style when the preference signal aligns with format and concision.",
            "",
            "### DPO vs SFT — limitations",
            "- If the rejected summaries are noisy, DPO can **degrade** ROUGE or faithfulness; monitor test metrics.",
            "- LoRA capacity is smaller than full fine-tune; large behavior changes are unlikely without longer training.",
            "",
            "### Worth continuing to GRPO?",
            "- If DPO improves **format adherence** or human-judged quality without collapsing ROUGE, GRPO is a reasonable next step for reward shaping experiments.",
            "- If metrics regress, fix preference data or β before spending budget on online RL.",
            "",
        ]
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
