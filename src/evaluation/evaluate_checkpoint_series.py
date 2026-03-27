"""Batch-evaluate a fixed checkpoint series for Part V / E2 dynamics (same test JSONL for all rows).

Uses `grpo_test.jsonl` for every model so metrics are strictly comparable (prompts match `sft_test`).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.evaluation.evaluate import run_single_eval
from src.training.common import load_merged_config
from src.utils.path_utils import ensure_dir, find_project_root, resolve_path

DENSE_CKPT_ROOT = "outputs/checkpoints/grpo_v1_e2_dense"

# Disk reality (Mar 2025 runs): only checkpoint-700 and final (best == checkpoint-723); no 100/200/300 saves.
E2_CHECKPOINT_SPECS: list[dict] = [
    {
        "run_id": "step0_sft_best",
        "model_family": "SFT",
        "checkpoint_step": 0,
        "stage": "sft",
        "checkpoint_rel": "outputs/checkpoints/sft_full_3090/best",
        "pred_filename": "e2_step0_sft_best_test_greedy.jsonl",
        "reward_variant": None,
        "legacy_pred": "sft_test_greedy.jsonl",
    },
    {
        "run_id": "grpo_v1_checkpoint_700",
        "model_family": "GRPO-V1",
        "checkpoint_step": 700,
        "stage": "grpo_v1",
        "checkpoint_rel": "outputs/checkpoints/grpo_v1_3090/checkpoint-700",
        "pred_filename": "e2_grpo_v1_step700_test_greedy.jsonl",
        "reward_variant": "v1",
        "legacy_pred": None,
    },
    {
        "run_id": "grpo_v1_final",
        "model_family": "GRPO-V1",
        "checkpoint_step": 723,
        "stage": "grpo_v1",
        "checkpoint_rel": "outputs/checkpoints/grpo_v1_3090/best",
        "pred_filename": "e2_grpo_v1_final_test_greedy.jsonl",
        "reward_variant": "v1",
        "legacy_pred": "grpo_v1_test_greedy.jsonl",
    },
    {
        "run_id": "grpo_v4_checkpoint_700",
        "model_family": "GRPO-V4",
        "checkpoint_step": 700,
        "stage": "grpo_v4",
        "checkpoint_rel": "outputs/checkpoints/grpo_v4_3090/checkpoint-700",
        "pred_filename": "e2_grpo_v4_step700_test_greedy.jsonl",
        "reward_variant": "v4",
        "legacy_pred": None,
    },
    {
        "run_id": "grpo_v4_final",
        "model_family": "GRPO-V4",
        "checkpoint_step": 723,
        "stage": "grpo_v4",
        "checkpoint_rel": "outputs/checkpoints/grpo_v4_3090/best",
        "pred_filename": "e2_grpo_v4_final_test_greedy.jsonl",
        "reward_variant": "v4",
        "legacy_pred": "grpo_v4_test_greedy.jsonl",
    },
]


def build_dense_checkpoint_specs(root: Path) -> list[dict]:
    """SFT baseline + every checkpoint-* under grpo_v1_e2_dense (sorted by global step)."""
    specs: list[dict] = [
        {
            "run_id": "step0_sft_best",
            "model_family": "SFT",
            "checkpoint_step": 0,
            "stage": "sft",
            "checkpoint_rel": "outputs/checkpoints/sft_full_3090/best",
            "pred_filename": "e2_dense_step0_sft_best_test_greedy.jsonl",
            "reward_variant": None,
            "legacy_pred": "e2_step0_sft_best_test_greedy.jsonl",
        },
    ]
    ckpt_root = resolve_path(DENSE_CKPT_ROOT, root)
    if not ckpt_root.is_dir():
        raise SystemExit(f"Dense checkpoint root missing: {ckpt_root} (train with scripts/run_grpo_v1_e2_dense.sh first).")
    cps: list[Path] = []
    for p in ckpt_root.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-") and p.name != "checkpoint-":
            try:
                step = int(p.name.split("-", 1)[1])
            except ValueError:
                continue
            cps.append(p)
    cps.sort(key=lambda p: int(p.name.split("-", 1)[1]))
    if not cps:
        raise SystemExit(f"No checkpoint-* under {ckpt_root}")
    for p in cps:
        step = int(p.name.split("-", 1)[1])
        rel = str(p.relative_to(root))
        specs.append(
            {
                "run_id": f"grpo_v1_e2_dense_step{step}",
                "model_family": "GRPO-V1-dense",
                "checkpoint_step": step,
                "stage": "grpo_v1",
                "checkpoint_rel": rel,
                "pred_filename": f"e2_dense_grpo_v1_step{step}_test_greedy.jsonl",
                "reward_variant": "v1",
                "legacy_pred": None,
            }
        )
    return specs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="E2 checkpoint series eval on grpo_test.jsonl")
    p.add_argument(
        "--preset",
        choices=["sparse", "dense"],
        default="sparse",
        help="sparse: main E2 table; dense: SFT + all grpo_v1_e2_dense/checkpoint-*",
    )
    p.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/inference.yaml"])
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument(
        "--reuse_legacy_predictions",
        action="store_true",
        default=True,
        help="If skip_generate and E2 pred missing, copy legacy grpo_v1/v4/sft *_test_greedy.jsonl when paths match final/SFT best.",
    )
    p.add_argument(
        "--no_reuse_legacy_predictions",
        action="store_false",
        dest="reuse_legacy_predictions",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Default depends on --preset",
    )
    p.add_argument(
        "--out_md",
        type=str,
        default=None,
    )
    p.add_argument(
        "--per_metrics_dir",
        type=str,
        default=None,
        help="Default: outputs/metrics/e2_dynamics or e2_dynamics_dense",
    )
    return p


def _maybe_reuse_legacy(
    root: Path,
    pred_path: Path,
    legacy_name: str | None,
    *,
    reuse: bool,
    skip_generate: bool,
) -> bool:
    if not skip_generate or not reuse or not legacy_name:
        return False
    if pred_path.is_file():
        return True
    legacy = resolve_path(f"outputs/predictions/{legacy_name}", root)
    if legacy.is_file():
        ensure_dir(pred_path.parent)
        shutil.copy2(legacy, pred_path)
        return True
    return False


def _row_from_metrics(root: Path, spec: dict, metrics: dict) -> dict[str, str | float | int | None]:
    fam = spec["model_family"]
    step = int(spec["checkpoint_step"])
    fmt = metrics.get("format_adherence") or {}
    loose = float(fmt.get("format_rate", 0.0))
    strict = float(metrics.get("strict_format_rate", 0.0))
    rouge = float(metrics.get("rouge_l_f1", 0.0))
    avg_tok = float(metrics.get("avg_output_length_tokens", 0.0))
    if fam == "SFT":
        avg_rwd = None
    else:
        avg_rwd = float(metrics.get("avg_reward", 0.0))
    return {
        "run_id": spec["run_id"],
        "model_family": fam,
        "checkpoint_step": step,
        "checkpoint_path": str(resolve_path(spec["checkpoint_rel"], root)),
        "rouge_l_f1": rouge,
        "format_rate_loose": loose,
        "strict_format_rate": strict,
        "avg_output_length_tokens": avg_tok,
        "avg_reward": avg_rwd if avg_rwd is not None else "",
        "predictions_jsonl": metrics.get("predictions_jsonl", ""),
        "n": int(metrics.get("n", 0)),
    }


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    if args.preset == "dense":
        specs = build_dense_checkpoint_specs(root)
        out_csv = args.out_csv or "outputs/report_assets/reward_hacking_dynamics_dense_metrics.csv"
        out_md = args.out_md or "outputs/report_assets/reward_hacking_dynamics_dense_metrics.md"
        per_dir_s = args.per_metrics_dir or "outputs/metrics/e2_dynamics_dense"
    else:
        specs = E2_CHECKPOINT_SPECS
        out_csv = args.out_csv or "outputs/report_assets/reward_hacking_dynamics_metrics.csv"
        out_md = args.out_md or "outputs/report_assets/reward_hacking_dynamics_metrics.md"
        per_dir_s = args.per_metrics_dir or "outputs/metrics/e2_dynamics"

    cfg = load_merged_config(args.config, root)
    in_path = resolve_path("data/processed/grpo_test.jsonl", root)
    pred_dir = resolve_path(
        (cfg.get("paths", {}) or {}).get("predictions_dir", "outputs/predictions"), root
    )
    ensure_dir(pred_dir)
    per_dir = resolve_path(per_dir_s, root)
    ensure_dir(per_dir)

    rows_out: list[dict] = []
    for spec in specs:
        pred_path = pred_dir / spec["pred_filename"]
        ckpt = resolve_path(spec["checkpoint_rel"], root)
        _maybe_reuse_legacy(
            root,
            pred_path,
            spec.get("legacy_pred"),
            reuse=args.reuse_legacy_predictions,
            skip_generate=args.skip_generate,
        )
        metrics_path = per_dir / f"{spec['run_id']}_metrics.json"
        metrics = run_single_eval(
            root=root,
            cfg=cfg,
            stage=spec["stage"],
            checkpoint=ckpt,
            input_jsonl=in_path,
            pred_path=pred_path,
            skip_generate=args.skip_generate,
            smoke_test=args.smoke_test,
            reward_variant=spec.get("reward_variant"),
            metrics_json_out=metrics_path,
        )
        rows_out.append(_row_from_metrics(root, spec, metrics))

    out_csv = resolve_path(out_csv, root)
    out_md = resolve_path(out_md, root)
    ensure_dir(out_csv.parent)

    fieldnames = [
        "run_id",
        "model_family",
        "checkpoint_step",
        "checkpoint_path",
        "rouge_l_f1",
        "format_rate_loose",
        "strict_format_rate",
        "avg_output_length_tokens",
        "avg_reward",
        "n",
        "predictions_jsonl",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    title = (
        "# Reward hacking dynamics — dense GRPO-V1 (E2 supplement)\n\n"
        if args.preset == "dense"
        else "# Reward hacking dynamics — checkpoint-level metrics (E2)\n\n"
    )
    lines = [
        title.rstrip("\n"),
        "",
        f"Test input: `{in_path}` (81 rows; same prompts as `sft_test.jsonl`).",
        "",
        "| run_id | family | step | ROUGE-L | loose | strict | len(tok) | avg_reward |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows_out:
        ar = r["avg_reward"]
        ars = f"{float(ar):.4f}" if ar != "" and ar is not None else "—"
        lines.append(
            f"| {r['run_id']} | {r['model_family']} | {r['checkpoint_step']} | "
            f"{float(r['rouge_l_f1']):.4f} | {float(r['format_rate_loose']):.3f} | "
            f"{float(r['strict_format_rate']):.3f} | {float(r['avg_output_length_tokens']):.1f} | {ars} |"
        )
    lines.append("")
    lines.append(f"Per-run metrics JSON: `{per_dir_s}/*.json`")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"wrote_csv": str(out_csv), "wrote_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
