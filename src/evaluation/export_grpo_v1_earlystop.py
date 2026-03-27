"""E2 minimal intervention: GRPO-V1 checkpoint-700 vs final (best) on the same test rows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.metrics.strict_format_adherence import strict_format_adherence_one
from src.rewards.reward_fn import compute_reward
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dynamics_csv", type=str, default="outputs/report_assets/reward_hacking_dynamics_metrics.csv")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/grpo_v1_earlystop_vs_final.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/grpo_v1_earlystop_vs_final.md")
    p.add_argument("--out_qual", type=str, default="outputs/report_assets/grpo_v1_earlystop_qualitative.md")
    p.add_argument("--variant", type=str, default="v1")
    return p


def _float(x: str) -> float:
    return float(x) if x and str(x).strip() else float("nan")


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pred_path(root: Path, dynamics: list[dict], run_id: str) -> Path:
    r = next(x for x in dynamics if x["run_id"] == run_id)
    p = Path(r["predictions_jsonl"])
    return p if p.is_file() else resolve_path(r["predictions_jsonl"], root)


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    dyn = _load_rows(resolve_path(args.dynamics_csv, root))
    early_p = _pred_path(root, dyn, "grpo_v1_checkpoint_700")
    final_p = _pred_path(root, dyn, "grpo_v1_final")
    r_early = next(x for x in dyn if x["run_id"] == "grpo_v1_checkpoint_700")
    r_final = next(x for x in dyn if x["run_id"] == "grpo_v1_final")

    early_rows: list[dict] = []
    final_rows: list[dict] = []
    with open(early_p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                early_rows.append(json.loads(line))
    with open(final_p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                final_rows.append(json.loads(line))
    assert len(early_rows) == len(final_rows)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    variant = args.variant

    scored: list[tuple[int, float, dict]] = []
    for i, (e, f) in enumerate(zip(early_rows, final_rows)):
        pe = (e.get("prediction") or "").strip()
        pf = (f.get("prediction") or "").strip()
        ref = (e.get("summary_en_chosen") or "").strip()
        src = (e.get("answer_en") or "").strip()
        prompt = (e.get("prompt") or "").strip()
        len_delta = abs(len(pf) - len(pe))
        se = strict_format_adherence_one(pe)["pass_strict"]
        sf = strict_format_adherence_one(pf)["pass_strict"]
        rwe = compute_reward(variant, prompt, pe, src)
        rwf = compute_reward(variant, prompt, pf, src)
        rle = scorer.score(ref or " ", pe or " ")["rougeL"].fmeasure
        rlf = scorer.score(ref or " ", pf or " ")["rougeL"].fmeasure
        scored.append(
            (
                i,
                len_delta + (10.0 if se != sf else 0) + abs(rwf - rwe),
                {
                    "idx": i,
                    "early_pred": pe,
                    "final_pred": pf,
                    "ref": ref,
                    "src": src[:400],
                    "early_strict": se,
                    "final_strict": sf,
                    "early_rwd": rwe,
                    "final_rwd": rwf,
                    "early_rl": rle,
                    "final_rl": rlf,
                },
            )
        )

    scored.sort(key=lambda x: -x[1])
    picked = [x[2] for x in scored[:12]]
    # ensure at least 3 diverse indices
    qual = []
    seen_idx = set()
    for d in picked:
        if len(qual) >= 3:
            break
        if d["idx"] in seen_idx:
            continue
        qual.append(d)
        seen_idx.add(d["idx"])

    out_csv = resolve_path(args.out_csv, root)
    out_md = resolve_path(args.out_md, root)
    out_qual = resolve_path(args.out_qual, root)
    for p in (out_csv, out_md, out_qual):
        p.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {
            "checkpoint": "GRPO-V1 step 700 (early)",
            "avg_reward": r_early.get("avg_reward", ""),
            "rouge_l_f1": r_early["rouge_l_f1"],
            "strict_format_rate": r_early["strict_format_rate"],
            "avg_output_length_tokens": r_early["avg_output_length_tokens"],
            "format_rate_loose": r_early["format_rate_loose"],
        },
        {
            "checkpoint": "GRPO-V1 final (723)",
            "avg_reward": r_final.get("avg_reward", ""),
            "rouge_l_f1": r_final["rouge_l_f1"],
            "strict_format_rate": r_final["strict_format_rate"],
            "avg_output_length_tokens": r_final["avg_output_length_tokens"],
            "format_rate_loose": r_final["format_rate_loose"],
        },
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    def fmt_row(label: str, r: dict) -> str:
        return (
            f"| {label} | {r.get('avg_reward', '—')} | {_float(r['rouge_l_f1']):.4f} | "
            f"{_float(r['strict_format_rate']):.3f} | {_float(r['avg_output_length_tokens']):.1f} | "
            f"{_float(r['format_rate_loose']):.3f} |"
        )

    md = [
        "# GRPO-V1 early stop proxy: step **700** vs **final (723)**",
        "",
        "No extra training: we compare **saved** adapters only. Ideal early targets (200/300) are **not** on disk;",
        "**700** is the latest intermediate checkpoint available for both V1 and V4.",
        "",
        "| Checkpoint | avg_reward (v1) | ROUGE-L | strict | len(tok) | loose |",
        "|---|---:|---:|---:|---:|---:|",
        fmt_row("step 700", r_early),
        fmt_row("final", r_final),
        "",
        "## Takeaway",
        "",
        f"- ROUGE-L: **{_float(r_final['rouge_l_f1']) - _float(r_early['rouge_l_f1']):+.4f}** (final − early).",
        f"- Strict format rate: **{_float(r_final['strict_format_rate']) - _float(r_early['strict_format_rate']):+.4f}**.",
        f"- Mean output tokens: **{_float(r_final['avg_output_length_tokens']) - _float(r_early['avg_output_length_tokens']):+.1f}**.",
        f"- avg_reward: **{_float(str(r_final.get('avg_reward',''))) - _float(str(r_early.get('avg_reward',''))):+.4f}**.",
        "",
        "Interpret alongside `grpo_v1_earlystop_qualitative.md` (aligned row indices).",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")

    ql = [
        "# GRPO-V1 early (700) vs final — qualitative (aligned test indices)",
        "",
        f"Scoring: reward **{variant}**; ROUGE-L vs `summary_en_chosen`.",
        "",
    ]
    for k, d in enumerate(qual, 1):
        ql.append(f"## Example {k} — row index `{d['idx']}`")
        ql.append(f"- **Early strict** / **Final strict**: {d['early_strict']} / {d['final_strict']}")
        ql.append(
            f"- **Early reward** / **Final reward**: {d['early_rwd']:.4f} / {d['final_rwd']:.4f} · "
            f"**Early ROUGE-L** / **Final ROUGE-L**: {d['early_rl']:.4f} / {d['final_rl']:.4f}"
        )
        ql.append("- **Reference (`summary_en_chosen`)**:")
        ql.append("")
        ql.append("```")
        ql.append(d["ref"][:1200])
        ql.append("```")
        ql.append("- **Early prediction (700)**:")
        ql.append("")
        ql.append("```")
        ql.append(d["early_pred"][:2500])
        ql.append("```")
        ql.append("- **Final prediction (723)**:")
        ql.append("")
        ql.append("```")
        ql.append(d["final_pred"][:2500])
        ql.append("```")
        ql.append("")
    out_qual.write_text("\n".join(ql), encoding="utf-8")
    print(json.dumps({"csv": str(out_csv), "md": str(out_md), "qual": str(out_qual)}, indent=2))


if __name__ == "__main__":
    main()
