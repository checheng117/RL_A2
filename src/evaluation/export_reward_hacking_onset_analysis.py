"""Part V / E2: cautious onset narrative from dynamics CSV + per-row prediction stats."""
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
    p.add_argument("--csv", type=str, default="outputs/report_assets/reward_hacking_dynamics_metrics.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/reward_hacking_onset_analysis.md")
    return p


def _float(x: str) -> float:
    if not x or not str(x).strip():
        return float("nan")
    return float(x)


def _hack_flags(pred: str, ref: str, scorer: rouge_scorer.RougeScorer) -> dict:
    pl = pred.lower()
    dup_point = pl.count("[point]") > 1
    st = strict_format_adherence_one(pred)["pass_strict"]
    rl = scorer.score(ref or " ", pred or " ")["rougeL"].fmeasure
    return {"dup_point": dup_point, "strict_pass": st, "rouge_l": rl}


def _scan_pred_jsonl(path: Path, variant: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    n = 0
    dup = 0
    strict_fail = 0
    high_rw_low_r = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            pred = (row.get("prediction") or "").strip()
            ref = (row.get("summary_en_chosen") or "").strip()
            src = (row.get("answer_en") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            fl = _hack_flags(pred, ref, scorer)
            if fl["dup_point"]:
                dup += 1
            if not fl["strict_pass"]:
                strict_fail += 1
            rw = compute_reward(variant, prompt, pred, src)
            if rw >= 0.85 and fl["rouge_l"] < 0.38:
                high_rw_low_r += 1
    return {
        "n": n,
        "dup_point_rate": dup / max(n, 1),
        "strict_fail_rate": strict_fail / max(n, 1),
        "high_reward_low_rouge_count": high_rw_low_r,
        "high_reward_low_rouge_rate": high_rw_low_r / max(n, 1),
    }


def _trainer_train_reward_series(ckpt_dir: Path) -> list[tuple[int, float, float]]:
    state = json.loads((ckpt_dir / "trainer_state.json").read_text())
    out: list[tuple[int, float, float]] = []
    for h in state.get("log_history", []):
        step = h.get("step")
        if step is None:
            continue
        rw = h.get("rewards/reward_func/mean", h.get("reward"))
        ml = h.get("completions/mean_length")
        if rw is None:
            continue
        out.append((int(step), float(rw), float(ml) if ml is not None else float("nan")))
    return out


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    csv_path = resolve_path(args.csv, root)
    out_md = resolve_path(args.out_md, root)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    by_id = {r["run_id"]: r for r in rows}

    def stats_for(run_id: str, variant: str) -> dict | None:
        r = by_id.get(run_id)
        if not r:
            return None
        pj = r.get("predictions_jsonl", "").strip()
        if not pj:
            return None
        p = Path(pj)
        if not p.is_file():
            p = resolve_path(pj, root)
        if not p.is_file():
            return None
        s = _scan_pred_jsonl(p, variant)
        s["run_id"] = run_id
        return s

    s_v1_700 = stats_for("grpo_v1_checkpoint_700", "v1")
    s_v1_f = stats_for("grpo_v1_final", "v1")
    s_v4_700 = stats_for("grpo_v4_checkpoint_700", "v4")
    s_v4_f = stats_for("grpo_v4_final", "v4")

    lines = [
        "# Onset of reward hacking — E2 (cautious, split-specific)",
        "",
        "Scope: **English-only** main line; **test** split (81 examples); **greedy** decoding;",
        "metrics as in `reward_hacking_dynamics_metrics.csv`. Only **two** GRPO adapter",
        "snapshots exist on disk (**700** and **final @723**); intermediate weights are not",
        "available, so **held-out test dynamics are coarse**. We therefore combine (i)",
        "checkpoint-level test metrics, (ii) **per-example** structure/ROUGE/reward counters",
        "on saved predictions, and (iii) **training logs** (`trainer_state.json`) as an",
        "auxiliary signal (on-policy training batch, not the test set).",
        "",
        "## 1. Held-out test: step 700 → 723",
        "",
    ]

    def fmt_delta(name: str, a: dict, b: dict, keys: list[str]) -> list[str]:
        out = [f"### {name}"]
        for k in keys:
            va, vb = _float(a[k]), _float(b[k])
            out.append(f"- **{k}**: {va:.4f} → {vb:.4f} (Δ {vb - va:+.4f})")
        return out

    r0 = by_id["step0_sft_best"]
    r700 = by_id["grpo_v1_checkpoint_700"]
    rf = by_id["grpo_v1_final"]
    lines.extend(
        fmt_delta(
            "GRPO-V1 (test)",
            r700,
            rf,
            ["rouge_l_f1", "strict_format_rate", "avg_output_length_tokens", "avg_reward"],
        )
    )
    lines.append("")
    r4_700 = by_id["grpo_v4_checkpoint_700"]
    r4_f = by_id["grpo_v4_final"]
    lines.extend(
        fmt_delta(
            "GRPO-V4 (test)",
            r4_700,
            r4_f,
            ["rouge_l_f1", "strict_format_rate", "avg_output_length_tokens", "avg_reward"],
        )
    )

    lines.extend(
        [
            "",
            "## 2. Per-example signals (same test JSONL, by checkpoint)",
            "",
            "Counts below use: **dup `[point]`** (loose hack proxy), **strict fail rate**,",
            "and **high reward + low ROUGE-L** (reward ≥ 0.85, ROUGE-L < 0.38 vs `summary_en_chosen`).",
            "",
        ]
    )
    for label, s in [
        ("GRPO-V1 @700", s_v1_700),
        ("GRPO-V1 @final", s_v1_f),
        ("GRPO-V4 @700", s_v4_700),
        ("GRPO-V4 @final", s_v4_f),
    ]:
        if s:
            lines.append(
                f"- **{label}**: dup-rate={s['dup_point_rate']:.3f}, strict-fail-rate={s['strict_fail_rate']:.3f}, "
                f"high-R/low-ROUGE={s['high_reward_low_rouge_count']}/{s['n']} ({s['high_reward_low_rouge_rate']:.3f})"
            )
        else:
            lines.append(f"- **{label}**: (missing predictions path)")

    lines.extend(
        [
            "",
            "## 3. When does hacking “show up” on the test checkpoints we have?",
            "",
            "### GRPO-V1",
            "",
        ]
    )

    # Narrative from numbers
    dr = _float(rf["avg_reward"]) - _float(r700["avg_reward"])
    drouge = _float(rf["rouge_l_f1"]) - _float(r700["rouge_l_f1"])
    dstrict = _float(rf["strict_format_rate"]) - _float(r700["strict_format_rate"])
    dlen = _float(rf["avg_output_length_tokens"]) - _float(r700["avg_output_length_tokens"])
    if s_v1_700 and s_v1_f:
        dd_dup = s_v1_f["dup_point_rate"] - s_v1_700["dup_point_rate"]
        dd_hr = s_v1_f["high_reward_low_rouge_rate"] - s_v1_700["high_reward_low_rouge_rate"]
    else:
        dd_dup = dd_hr = 0.0

    lines.append(
        f"Between **700** and **723**, avg **v1 reward** changes by **{dr:+.4f}**, ROUGE-L by **{drouge:+.4f}**, "
        f"strict format by **{dstrict:+.4f}**, mean length by **{dlen:+.1f}** tokens."
    )
    if s_v1_700 and s_v1_f:
        lines.append(
            f"Per-example proxies moved: duplicate-`[point]` rate **{s_v1_700['dup_point_rate']:.3f} → {s_v1_f['dup_point_rate']:.3f}** "
            f"(Δ {dd_dup:+.3f}); high-reward/low-ROUGE rate **{dd_hr:+.3f}**."
        )
    v1_hr_up = dd_hr > 0.005
    v1_rg_down = drouge < -0.001
    lines.append(
        "**Interpretation (conservative):** We cannot pin a sharp **onset step** on the test split "
        "with only two saved adapters. **In this run**, V1 **avg_reward stays at 1.0** (saturated), "
        "ROUGE-L **slightly decreases** 700→final, and the **high-reward/low-ROUGE** proxy "
        f"{'**increases** (weak late-stage pressure consistent with mild hacking)' if v1_hr_up and v1_rg_down else 'is flat or ambiguous'} — "
        "a **small-magnitude** pattern on **n=81**, not proof of a large qualitative collapse. "
        "Severe hacking may also have **stabilized before step 700** (no earlier checkpoints to verify)."
    )

    lines.extend(["", "### GRPO-V4", ""])
    dr4 = _float(r4_f["avg_reward"]) - _float(r4_700["avg_reward"])
    drouge4 = _float(r4_f["rouge_l_f1"]) - _float(r4_700["rouge_l_f1"])
    dstrict4 = _float(r4_f["strict_format_rate"]) - _float(r4_700["strict_format_rate"])
    dlen4 = _float(r4_f["avg_output_length_tokens"]) - _float(r4_700["avg_output_length_tokens"])
    lines.append(
        f"V4 test deltas (700 → final): reward **{dr4:+.4f}**, ROUGE-L **{drouge4:+.4f}**, "
        f"strict **{dstrict4:+.4f}**, length **{dlen4:+.1f}** tokens."
    )
    if s_v4_700 and s_v4_f:
        lines.append(
            f"Duplicate-`[point]` rate: **{s_v4_700['dup_point_rate']:.3f} → {s_v4_f['dup_point_rate']:.3f}**; "
            f"high-reward/low-ROUGE: **{s_v4_f['high_reward_low_rouge_rate'] - s_v4_700['high_reward_low_rouge_rate']:+.3f}**."
        )
    lines.append(
        "V4’s reward is **not saturated at 1.0** in training logs, so the policy faces **richer constraints** "
        "(order, numbered reasons, overlap, length). Any hacking-like trend on test is expected to be **weaker "
        "or later** than under V1, but this must be read **together** with the small step gap and n=81 test set."
    )

    lines.extend(["", "## 4. Auxiliary: training log (not test)", ""])
    for name, rel in [
        ("GRPO-V1", "outputs/checkpoints/grpo_v1_3090/best/trainer_state.json"),
        ("GRPO-V4", "outputs/checkpoints/grpo_v4_3090/best/trainer_state.json"),
    ]:
        p = resolve_path(rel, root)
        if p.is_file():
            series = _trainer_train_reward_series(p.parent)
            if series:
                early = series[: min(40, len(series))]
                late = series[-8:]
                lines.append(f"### {name}")
                lines.append(
                    f"- Logged **{len(series)}** points (e.g. every ~5 steps). Early mean train reward "
                    f"(first ~40 logs): **{sum(x[1] for x in early)/len(early):.4f}**; "
                    f"last 8 logs: **{sum(x[1] for x in late)/len(late):.4f}**."
                )
                ml_early = sum(x[2] for x in early if x[2] == x[2]) / max(
                    sum(1 for x in early if x[2] == x[2]), 1
                )
                ml_late = sum(x[2] for x in late if x[2] == x[2]) / max(
                    sum(1 for x in late if x[2] == x[2]), 1
                )
                lines.append(
                    f"- Training completion **mean length** (logged): early avg **{ml_early:.1f}** vs late avg **{ml_late:.1f}** tokens (trainer-reported)."
                )
        lines.append("")

    lines.append(
        "---\n\n*This file is generated by `python -m src.evaluation.export_reward_hacking_onset_analysis` "
        "and should be read as **evidence under the stated limitations**, not a claim about all seeds or datasets.*"
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
