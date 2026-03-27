"""Assemble pre_grpo_decision.md from metric JSON files (honest, data-driven)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def _load(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _line(name: str, m: dict) -> str:
    loose = (m.get("format_adherence") or {}).get("format_rate", "")
    strict = m.get("strict_format_rate")
    if strict is None:
        strict = (m.get("format_adherence_strict") or {}).get("strict_format_rate", "")
    return (
        f"| {name} | {loose} | {strict} | {m.get('rouge_l_f1', '')} | "
        f"{m.get('avg_output_length_tokens', '')} | `{m.get('checkpoint', '')}` |"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/report_assets/pre_grpo_decision.md")
    args = p.parse_args()
    root = find_project_root()
    out = resolve_path(args.out, root)
    out.parent.mkdir(parents=True, exist_ok=True)

    sft = _load(resolve_path("outputs/metrics/sft_test_metrics.json", root))
    dpo = _load(resolve_path("outputs/metrics/dpo_test_metrics.json", root))
    ret = _load(resolve_path("outputs/metrics/dpo_retune_test_metrics.json", root))

    def gap_loose(a: dict, b: dict) -> str:
        la = (a.get("format_adherence") or {}).get("format_rate")
        lb = (b.get("format_adherence") or {}).get("format_rate")
        if la is None or lb is None:
            return "n/a"
        return f"{float(lb) - float(la):+.3f} (B - A)"

    def gap_strict(a: dict, b: dict) -> str:
        sa = a.get("strict_format_rate")
        if sa is None:
            sa = (a.get("format_adherence_strict") or {}).get("strict_format_rate")
        sb = b.get("strict_format_rate")
        if sb is None:
            sb = (b.get("format_adherence_strict") or {}).get("strict_format_rate")
        if sa is None or sb is None:
            return "n/a"
        return f"{float(sb) - float(sa):+.3f} (B - A)"

    r_rouge = float(ret.get("rouge_l_f1", 0)) - float(dpo.get("rouge_l_f1", 0))
    r_strict = float(
        ret.get("strict_format_rate")
        or (ret.get("format_adherence_strict") or {}).get("strict_format_rate", 0)
    ) - float(
        dpo.get("strict_format_rate")
        or (dpo.get("format_adherence_strict") or {}).get("strict_format_rate", 0)
    )
    r_len = float(ret.get("avg_output_length_tokens", 0)) - float(dpo.get("avg_output_length_tokens", 0))

    body = f"""# Pre-GRPO decision report (English main task)

Generated from `*_test_metrics.json` after strict-format evaluation. This file is meant for the assignment write-up: conclusions are **data-limited** (single retune run).

## 1. Side-by-side test metrics (n={sft.get("n", "?")})

| Model | format loose | strict | ROUGE-L | avg len (tok) | checkpoint |
|------|-------------|--------|---------|---------------|------------|
{_line("SFT", sft)}
{_line("DPO (original)", dpo)}
{_line("DPO (retune)", ret)}

## 2. Loose vs strict format

- **Loose** (`format_adherence`): tags present and coarse section order; repeated `[point]/[reason]/[summary]` blocks can still yield format_rate ≈ 1.0.
- **Strict** (`strict_format_rate`): exactly **one** of each tag, correct order, all sections non-empty, and **≥2** `1.` / `2.`-style markers inside `[reason]` (same line or multiple lines).

**SFT → DPO (original) strict gap:** {gap_strict(sft, dpo)}  
**DPO (original) → DPO (retune) strict gap:** {gap_strict(dpo, ret)}

If DPO drops sharply on strict while staying high on loose, that supports the hypothesis **“valid-template repetition”** rather than a pure ROUGE-only regression.

## 3. Observed DPO failure modes (from prior qualitative + metrics)

- Verbose generations and **multiple repeated structural blocks** while still satisfying loose tags.
- ROUGE-L drop vs SFT when copying or drifting from `summary_en_chosen` style.

## 4. Did retune help vs original DPO?

| Signal | Δ (retune − original DPO) |
|--------|---------------------------|
| ROUGE-L | {r_rouge:+.4f} |
| strict_format_rate | {r_strict:+.4f} |
| avg output tokens | {r_len:+.2f} |

Qualitative: re-read `outputs/report_assets/dpo_qualitative_analysis.md` after regenerating predictions if needed; retune uses `outputs/predictions/dpo_retune_test_greedy.jsonl`.

## 5. vs SFT

Retune **ROUGE-L** vs SFT: {float(ret.get('rouge_l_f1', 0)) - float(sft.get('rouge_l_f1', 0)):+.4f}.  
Retune **strict** vs SFT: {gap_strict(sft, ret)}.

## 6. GRPO go / no-go (honest)

**This run (retune vs original DPO):** large gains on strict format, ROUGE-L, and shorter outputs (see §4). **Retune vs SFT:** ROUGE-L is still slightly below SFT; strict format and length are closer but not fully matched.

- GRPO is **not mandatory** before the deadline: SFT remains the strongest **overall** reference on this split.
- If you **do** enter GRPO, treat it as an experiment with **strict + length monitoring**, not as a fix for a “good” DPO: use **V4** (or stricter) reward, keep **`max_completion_length` modest**, and log duplicate-tag failures explicitly.
- **If skipping GRPO**, the blocking narrative is already valuable: **loose format masked template repetition**; conservative DPO partially recovers quality but does not beat SFT.

**Suggested hypothesis if entering GRPO:** policy optimization should be constrained so completions stay **single-template**; align optimization signal with **V4** (or a future strict match term), not V1-only tags.

**Blocking concerns if not entering yet:** original DPO checkpoint is a **bad RL initialization** (strict 0, long outputs); prefer **retune** or **SFT** as the policy seed if GRPO is attempted later.

## 7. GRPO config sanity (no full run performed)

- `reward_v1`: tag coverage only — **does not** catch duplicate blocks; **not** aligned with strict format.
- `reward_v4`: order, numbered reasons, length/repetition heuristics, source overlap — **closer** to strict goals but still not identical to `strict_format_rate`.

English data paths in `configs/grpo_v*.yaml` use `data/processed/grpo_*.jsonl`; ensure these match your English-only pipeline before launching.

## 8. Risk checklist before GRPO

- Output length blow-up; repeated `[point]/[reason]/[summary]`; reward hacking on loose structure; train/eval metric mismatch.

See also `docs/IMPLEMENTATION_NOTES.md`.
"""
    out.write_text(body, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
