"""Export 2+ reward-hacking-style examples per GRPO run for Part III report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.metrics.format_adherence import format_adherence_score
from src.metrics.strict_format_adherence import strict_format_adherence_one
from src.rewards.reward_fn import compute_reward
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_jsonl", type=str, required=True)
    p.add_argument("--reward_variant", type=str, required=True, choices=["v1", "v4"])
    p.add_argument("--out_md", type=str, required=True)
    p.add_argument("--min_examples", type=int, default=2)
    return p


def _loose_pass(text: str) -> bool:
    return float(format_adherence_score(text)["score"]) >= 0.99


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    pred_p = resolve_path(args.predictions_jsonl, root)
    out_md = resolve_path(args.out_md, root)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    variant = args.reward_variant

    rows: list[dict] = []
    with open(pred_p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scored: list[tuple[float, int, dict]] = []
    for i, row in enumerate(rows):
        pred = (row.get("prediction") or "").strip()
        ref = (row.get("summary_en_chosen") or "").strip()
        src = (row.get("answer_en") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        rw = compute_reward(variant, prompt, pred, src)
        rl = scorer.score(ref or " ", pred or " ")["rougeL"].fmeasure
        st = strict_format_adherence_one(pred)["pass_strict"]
        scored.append((rw, i, {"row": row, "reward": rw, "rouge_l": rl, "strict": st}))

    # Prefer: high reward but (strict fail OR low ROUGE OR obvious duplicate-template risk)
    def hack_score(t: tuple[float, int, dict]) -> float:
        rw, _, d = t
        pred = (d["row"].get("prediction") or "")
        bad = (not d["strict"]) or d["rouge_l"] < 0.32
        dup = pred.lower().count("[point]") > 1
        if bad or dup:
            return rw + (0.2 if dup else 0)
        return rw * 0.1

    ranked = sorted(scored, key=hack_score, reverse=True)
    picked: list[tuple[int, dict]] = []
    seen_text = set()
    for rw, idx, d in ranked:
        pred = (d["row"].get("prediction") or "").strip()
        if pred in seen_text:
            continue
        bad = (not d["strict"]) or d["rouge_l"] < 0.35 or pred.lower().count("[point]") > 1
        if rw < 0.35 and not bad:
            continue
        if bad or rw >= 0.85:
            picked.append((idx, d))
            seen_text.add(pred)
        if len(picked) >= args.min_examples:
            break

    if len(picked) < args.min_examples:
        for rw, idx, d in sorted(scored, key=lambda x: -x[0]):
            pred = (d["row"].get("prediction") or "").strip()
            if pred in seen_text:
                continue
            picked.append((idx, d))
            seen_text.add(pred)
            if len(picked) >= args.min_examples:
                break

    title = f"GRPO-{variant.upper()} reward hacking / high-reward–low-quality cases"
    lines = [
        f"# {title}",
        "",
        f"Source: `{args.predictions_jsonl}` (test split). Reward = **{variant}** (not strict metric).",
        "Strict / loose are **evaluation** tools; the model was **not** directly optimized on strict format.",
        "",
    ]

    for k, (idx, d) in enumerate(picked, 1):
        row = d["row"]
        pred = (row.get("prediction") or "").strip()
        ref = (row.get("summary_en_chosen") or "").strip()
        src = (row.get("answer_en") or "").strip()
        lp = "Pass" if _loose_pass(pred) else "Fail"
        sp = "Pass" if d["strict"] else "Fail"
        why_high = []
        if variant == "v1":
            why_high.append("V1 only requires tag presence; empty or repeated sections can still yield ~1.0.")
        else:
            why_high.append("V4 mixes structure terms with source overlap; mechanical copying or templates can keep reward up.")
        if not d["strict"]:
            why_high.append("Strict metric fails (e.g. duplicate `[point]`/`[reason]`/`[summary]` blocks or missing numbered reasons).")
        if d["rouge_l"] < 0.35:
            why_high.append("ROUGE-L vs `summary_en_chosen` is low → poor summary quality despite reward.")
        why_bad = "The text may be verbose, repetitive, or not a faithful condensation of the reference summary."

        lines.append(f"## Example {k} (test row index `{idx}`, Example ID `{k}`)")
        lines.append(f"- **Example ID**: {k} (row {idx})")
        lines.append("- **Source answer (`answer_en`)**:")
        lines.append("```")
        lines.append(src[:2500])
        lines.append("```")
        lines.append("- **Reference summary (`summary_en_chosen`)**:")
        lines.append("```")
        lines.append(ref[:2500])
        lines.append("```")
        lines.append("- **Model prediction**:")
        lines.append("```")
        lines.append(pred[:3500])
        lines.append("```")
        lines.append(f"- **loose format**: {lp}")
        lines.append(f"- **strict format**: {sp}")
        lines.append(f"- **reward ({variant})**: **{d['reward']:.4f}**")
        lines.append(f"- **ROUGE-L (vs reference)**: **{d['rouge_l']:.4f}**")
        lines.append(f"- **Why high reward**: {' '.join(why_high)}")
        lines.append(f"- **Why still a bad summary**: {why_bad}")
        lines.append("")

    lines.append(
        "_If fewer than ideal hacking cases appear on test, examples above are still high-reward rows "
        "with quality issues (strict fail and/or low ROUGE) where available._\n"
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
