"""Early vs final using dense V1 checkpoints; auto-picks early by quality balance."""
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
    p.add_argument("--dynamics_csv", type=str, default="outputs/report_assets/reward_hacking_dynamics_dense_metrics.csv")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/grpo_v1_earlystop_dense_vs_final.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/grpo_v1_earlystop_dense_vs_final.md")
    p.add_argument("--out_qual", type=str, default="outputs/report_assets/grpo_v1_earlystop_dense_qualitative.md")
    return p


def _f(x: str) -> float:
    return float(x) if x and str(x).strip() else float("nan")


def _load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pred_path(root: Path, dynamics: list[dict], run_id: str) -> Path:
    r = next(x for x in dynamics if x["run_id"] == run_id)
    p = Path(r["predictions_jsonl"])
    return p if p.is_file() else resolve_path(r["predictions_jsonl"], root)


def _pick_early_row(dense: list[dict]) -> dict:
    """Prefer high reward (>=0.95) and best composite quality before final step."""
    dense = sorted(dense, key=lambda r: int(r["checkpoint_step"]))
    if len(dense) < 2:
        return dense[0]
    final_step = int(dense[-1]["checkpoint_step"])
    candidates = [r for r in dense if int(r["checkpoint_step"]) < final_step]
    if not candidates:
        return dense[0]

    def score_row(r: dict) -> float:
        rw = _f(r["avg_reward"])
        if rw < 0.95:
            return -1e9
        rl = _f(r["rouge_l_f1"])
        st = _f(r["strict_format_rate"])
        tok = _f(r["avg_output_length_tokens"])
        return rl + 0.5 * st - 0.015 * (tok / 10.0)

    ranked = sorted(candidates, key=score_row, reverse=True)
    return ranked[0]


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    dyn = _load_rows(resolve_path(args.dynamics_csv, root))
    dense_v1 = [r for r in dyn if r["model_family"] == "GRPO-V1-dense"]
    dense_v1.sort(key=lambda r: int(r["checkpoint_step"]))
    if len(dense_v1) < 2:
        raise SystemExit("Need at least two GRPO-V1-dense rows in CSV")

    early_row = _pick_early_row(dense_v1)
    final_row = dense_v1[-1]
    early_id, final_id = early_row["run_id"], final_row["run_id"]

    early_p = _pred_path(root, dyn, early_id)
    final_p = _pred_path(root, dyn, final_id)

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

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
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
        rwe = compute_reward("v1", prompt, pe, src)
        rwf = compute_reward("v1", prompt, pf, src)
        rle = scorer.score(ref or " ", pe or " ")["rougeL"].fmeasure
        rlf = scorer.score(ref or " ", pf or " ")["rougeL"].fmeasure
        scored.append(
            (
                i,
                len_delta + (8.0 if se != sf else 0) + abs(rwf - rwe) * 3 + abs(rlf - rle) * 5,
                {
                    "idx": i,
                    "early_pred": pe,
                    "final_pred": pf,
                    "ref": ref,
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
    qual = []
    seen = set()
    for _, __, d in scored:
        if len(qual) >= 3:
            break
        if d["idx"] in seen:
            continue
        qual.append(d)
        seen.add(d["idx"])

    out_csv = resolve_path(args.out_csv, root)
    out_md = resolve_path(args.out_md, root)
    out_qual = resolve_path(args.out_qual, root)
    for p in (out_csv, out_md, out_qual):
        p.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {
            "checkpoint": f"early ({early_id}, step {early_row['checkpoint_step']})",
            "avg_reward": early_row.get("avg_reward", ""),
            "rouge_l_f1": early_row["rouge_l_f1"],
            "strict_format_rate": early_row["strict_format_rate"],
            "avg_output_length_tokens": early_row["avg_output_length_tokens"],
        },
        {
            "checkpoint": f"final ({final_id}, step {final_row['checkpoint_step']})",
            "avg_reward": final_row.get("avg_reward", ""),
            "rouge_l_f1": final_row["rouge_l_f1"],
            "strict_format_rate": final_row["strict_format_rate"],
            "avg_output_length_tokens": final_row["avg_output_length_tokens"],
        },
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    dr = _f(final_row["rouge_l_f1"]) - _f(early_row["rouge_l_f1"])
    ds = _f(final_row["strict_format_rate"]) - _f(early_row["strict_format_rate"])
    dt = _f(final_row["avg_output_length_tokens"]) - _f(early_row["avg_output_length_tokens"])
    d_rw = _f(final_row["avg_reward"]) - _f(early_row["avg_reward"])

    if dr < -0.002:
        balance_early = (
            "The auto-picked **early** checkpoint has **higher ROUGE-L** than dense final on this split — "
            "early stopping looks **more balanced** under the stated quality proxies (still small n=81)."
        )
    elif abs(dr) <= 0.002 and abs(ds) <= 0.02 and abs(dt) <= 2.0:
        balance_early = (
            "Early vs final are **close** on ROUGE/strict/length; **no strong** case for either side from aggregates alone."
        )
    else:
        balance_early = (
            "**Final** matches or **beats** early on ROUGE (or other metrics favor continuing training) — "
            "early stopping is **not** clearly better here."
        )

    md = [
        "# GRPO-V1 E2-dense: early vs final",
        "",
        f"- **Early (auto-picked):** `{early_id}` — step **{early_row['checkpoint_step']}** "
        f"(among steps with avg_reward≥0.95 before final, maximize `ROUGE + 0.5*strict - 0.015*len/10`).",
        f"- **Final:** `{final_id}` — step **{final_row['checkpoint_step']}**.",
        "",
        "| | avg_reward | ROUGE-L | strict | len(tok) |",
        "|--:|---:|---:|---:|---:|",
        f"| early | {_f(early_row['avg_reward']):.4f} | {_f(early_row['rouge_l_f1']):.4f} | {_f(early_row['strict_format_rate']):.3f} | {_f(early_row['avg_output_length_tokens']):.1f} |",
        f"| final | {_f(final_row['avg_reward']):.4f} | {_f(final_row['rouge_l_f1']):.4f} | {_f(final_row['strict_format_rate']):.3f} | {_f(final_row['avg_output_length_tokens']):.1f} |",
        f"| Δ (final − early) | {d_rw:+.4f} | {dr:+.4f} | {ds:+.4f} | {dt:+.1f} |",
        "",
        "## Conclusion (honest)",
        "",
        balance_early,
        "",
        "See qualitative aligned rows in `grpo_v1_earlystop_dense_qualitative.md`.",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")

    ql = [
        "# GRPO-V1 E2-dense — early vs final qualitative",
        "",
        f"Early: **{early_id}** · Final: **{final_id}**",
        "",
    ]
    for k, d in enumerate(qual, 1):
        ql.append(f"## Example {k} (row `{d['idx']}`)")
        ql.append(f"- strict early/final: {d['early_strict']} / {d['final_strict']}")
        ql.append(
            f"- reward early/final: {d['early_rwd']:.4f} / {d['final_rwd']:.4f} · "
            f"ROUGE-L: {d['early_rl']:.4f} / {d['final_rl']:.4f}"
        )
        ql.append("- reference:")
        ql.append("```")
        ql.append(d["ref"][:1200])
        ql.append("```")
        ql.append("- early:")
        ql.append("```")
        ql.append(d["early_pred"][:2500])
        ql.append("```")
        ql.append("- final:")
        ql.append("```")
        ql.append(d["final_pred"][:2500])
        ql.append("```")
        ql.append("")
    out_qual.write_text("\n".join(ql), encoding="utf-8")
    print(json.dumps({"early_run_id": early_id, "final_run_id": final_id, "csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
