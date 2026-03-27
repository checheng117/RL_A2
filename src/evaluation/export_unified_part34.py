"""Unified table: SFT, DPO retune v2, GRPO-V1, GRPO-V4."""
from __future__ import annotations

import argparse
import csv
import json

from src.evaluation.compare_sft_dpo import _loose_rate, _strict_rate
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sft", type=str, default="outputs/metrics/sft_test_metrics.json")
    p.add_argument("--dpo_v2", type=str, default="outputs/metrics/dpo_retune_v2_test_metrics.json")
    p.add_argument("--grpo_v1", type=str, default="outputs/metrics/grpo_v1_test_metrics.json")
    p.add_argument("--grpo_v4", type=str, default="outputs/metrics/grpo_v4_test_metrics.json")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/unified_part34_metrics.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/unified_part34_metrics.md")
    return p


def _load(root, rel: str) -> dict:
    with open(resolve_path(rel, root), encoding="utf-8") as f:
        return json.load(f)


def _faithful(m: dict) -> str:
    r = float(m.get("rouge_l_f1", 0))
    s = _strict_rate(m) or 0.0
    if r >= 0.42 and s >= 0.95:
        return "Yes"
    if r >= 0.32 and s >= 0.75:
        return "Mixed"
    return "No"


def _hacking_label(name: str, m: dict) -> str:
    s = _strict_rate(m) or 0.0
    l = _loose_rate(m) or 0.0
    if "GRPO-V1" in name or "V1" in name:
        if l >= 0.99 and s < 0.85:
            return "Yes"
        return "Mild"
    if "GRPO-V4" in name or "V4" in name:
        if l >= 0.99 and s < 0.7:
            return "Yes"
        if s < 0.9:
            return "Mild"
        return "No"
    return "No"


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    specs = [
        ("SFT (full)", args.sft, "Supervised baseline; not RL-trained."),
        ("DPO retune v2", args.dpo_v2, "Best DPO in this repo run."),
        ("GRPO-V1", args.grpo_v1, "RL with tag-only reward; hacking-prone."),
        ("GRPO-V4", args.grpo_v4, "RL with richer reward; partial mitigation."),
    ]
    rows_out = []
    for name, rel, notes in specs:
        m = _load(root, rel)
        loose = _loose_rate(m)
        strict = _strict_rate(m)
        ar = m.get("avg_reward", "")
        reward_disp = "—" if name.startswith("SFT") or "DPO" in name else f"{float(ar):.4f}" if ar != "" else "—"
        if name.startswith("SFT") or "DPO" in name:
            reward_note = f"(diagnostic v4={float(m.get('avg_reward', 0)):.4f} in metrics JSON)"
        else:
            reward_note = ""
        rows_out.append(
            {
                "Model": name,
                "Avg_Reward": reward_disp,
                "ROUGE_L_F1": f"{float(m.get('rouge_l_f1', 0)):.4f}",
                "format_rate_loose": f"{loose:.3f}" if loose is not None else "",
                "strict_format_rate": f"{strict:.3f}" if strict is not None else "",
                "avg_output_length_tokens": f"{float(m.get('avg_output_length_tokens', 0)):.2f}",
                "faithful_heuristic": _faithful(m),
                "hacking_heuristic": _hacking_label(name, m),
                "checkpoint_path": str(m.get("checkpoint", "")),
                "notes": notes + (" " + reward_note if reward_note else ""),
            }
        )

    fieldnames = list(rows_out[0].keys())
    out_csv = resolve_path(args.out_csv, root)
    out_md = resolve_path(args.out_md, root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    md = [
        "# Unified Part III / IV metrics",
        "",
        "Same English test split, greedy decode (`configs/inference.yaml`). "
        "**Avg Reward** column uses **—** for SFT/DPO (not optimized with GRPO reward); see notes for diagnostic v4 score.",
        "",
        "| Model | Avg Reward | ROUGE-L | loose | strict | len(tok) | faithful? | hacking? | Checkpoint | Notes |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows_out:
        md.append(
            f"| {r['Model']} | {r['Avg_Reward']} | {r['ROUGE_L_F1']} | {r['format_rate_loose']} | {r['strict_format_rate']} | "
            f"{r['avg_output_length_tokens']} | {r['faithful_heuristic']} | {r['hacking_heuristic']} | "
            f"`{r['checkpoint_path']}` | {r['notes']} |"
        )
    md.append("")
    md.append("_faithful? / hacking? are coarse heuristics for the report; see qualitative + hacking MD files._")
    md.append("")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_csv} and {out_md}")


if __name__ == "__main__":
    main()
