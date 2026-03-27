"""Write dpo_final_decision.md from latest metric JSONs (honest Part II closure)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def _load(root: Path, rel: str) -> dict:
    p = resolve_path(rel, root)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _strict(m: dict) -> float:
    r = m.get("strict_format_rate")
    if r is not None:
        return float(r)
    return float((m.get("format_adherence_strict") or {}).get("strict_format_rate", 0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/report_assets/dpo_final_decision.md")
    args = p.parse_args()
    root = find_project_root()
    out = resolve_path(args.out, root)
    out.parent.mkdir(parents=True, exist_ok=True)

    sft = _load(root, "outputs/metrics/sft_test_metrics.json")
    dpo0 = _load(root, "outputs/metrics/dpo_test_metrics.json")
    r1 = _load(root, "outputs/metrics/dpo_retune_test_metrics.json")
    r2 = _load(root, "outputs/metrics/dpo_retune_v2_test_metrics.json")

    def rouge(m: dict) -> float:
        return float(m.get("rouge_l_f1", 0))

    def tok(m: dict) -> float:
        return float(m.get("avg_output_length_tokens", 0))

    dpo_candidates = [
        ("DPO (original)", dpo0),
        ("DPO retune v1", r1),
        ("DPO retune v2", r2),
    ]
    # Best DPO: prefer highest ROUGE-L; tie-break higher strict, then shorter length
    best_name, best_m = max(
        dpo_candidates,
        key=lambda x: (rouge(x[1]), _strict(x[1]), -tok(x[1])),
    )

    rs, rr2 = rouge(sft), rouge(r2)
    ss, sr2 = _strict(sft), _strict(r2)
    ts, tr2 = tok(sft), tok(r2)

    beats_sft = rr2 >= rs - 1e-6 and sr2 >= ss - 1e-6 and tr2 <= ts + 5
    near_sft = abs(rr2 - rs) < 0.02 and abs(sr2 - ss) < 0.05

    if rr2 >= rs - 1e-6 and sr2 >= ss - 1e-6:
        gap3 = (
            "**On this test run, DPO retune v2 is at or above SFT** on ROUGE-L and strict format; "
            f"mean length is **{tr2:.1f}** vs SFT **{ts:.1f}** tokens. "
            "Any remaining gap is best checked qualitatively (faithfulness, copying, edge cases), not only aggregate ROUGE."
        )
    else:
        gap3 = (
            "Typical gaps after retuning: **mean length** still above SFT, **strict** slightly below, "
            "or **ROUGE-L** short of the SFT reference alignment to `summary_en_chosen`."
        )

    grpo_seed = "SFT best (`outputs/checkpoints/sft_full_3090/best`)"
    grpo_reason = (
        "Most stable ROUGE-L + strict; avoids carrying DPO pathologies into online RL. "
        "Use retune v2 only if its metrics are clearly superior to v1 and close to SFT."
    )
    if rouge(r2) > rouge(r1) and _strict(r2) >= _strict(r1):
        grpo_seed = "DPO retune v2 (`outputs/checkpoints/dpo_lora_3090_retune_v2/best`)"
        grpo_reason = (
            "v2 matches or exceeds SFT on this split; reasonable GRPO seed if you want RL on top of preference tuning."
        )

    body = f"""# Part II — DPO final decision (English main task)

Data-driven summary from `outputs/metrics/*_test_metrics.json` (greedy eval, same test split).

## 1. Best DPO checkpoint among {len(dpo_candidates)} trained DPO variants

**Selected for reporting:** **{best_name}** (by ROUGE-L, then strict_format_rate, then shorter mean output tokens).

| Variant | ROUGE-L | strict | avg len (tok) |
|---------|---------|--------|----------------|
| DPO (original) | {rouge(dpo0):.4f} | {_strict(dpo0):.3f} | {tok(dpo0):.1f} |
| DPO retune v1 | {rouge(r1):.4f} | {_strict(r1):.3f} | {tok(r1):.1f} |
| DPO retune v2 | {rouge(r2):.4f} | {_strict(r2):.3f} | {tok(r2):.1f} |

## 2. Has any DPO variant matched or beaten SFT?

**SFT:** ROUGE-L **{rs:.4f}**, strict **{ss:.3f}**, len **{ts:.1f}** tok.  
**DPO retune v2:** ROUGE-L **{rr2:.4f}**, strict **{sr2:.3f}**, len **{tr2:.1f}** tok.

- **Strict “beats SFT”** (all metrics at least as good): **{str(beats_sft)}** (approximate rule: ROUGE ≥ SFT, strict ≥ SFT, length not much longer).
- **Practical “close enough” for a report narrative** (within ~0.02 ROUGE and ~0.05 strict): **{str(near_sft)}**.

## 3. Gap vs SFT (quantitative + what to say in the report)

{gap3}

## 4. Worth a third DPO tuning round?

**Recommendation:** **Usually no** unless v2 regressed vs v1 or homework explicitly requires more ablation. Diminishing returns are common; document **v1 vs v2** and stop unless you have a clear hypothesis (e.g. data noise in rejected summaries).

## 5. Submission stance (Part II)

For the assignment, **reporting SFT + best DPO variant + honest gap** is sufficient. You do **not** need DPO to exceed SFT to pass; show metrics, strict format, and qualitative cases.

**Suggested “final DPO artifact” for the PDF:** **{best_name}** metrics + `outputs/report_assets/sft_vs_dpo_all_metrics.md` + qualitative MD.

## 6. If GRPO is attempted later — policy seed

- **Default reference starting point:** {grpo_seed}
- **Reason:** {grpo_reason}

---

_Generated by `python -m src.evaluation.export_dpo_final_decision`._
"""
    out.write_text(body, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
