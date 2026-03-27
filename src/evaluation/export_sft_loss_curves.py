"""Export SFT train/eval loss series from HuggingFace Trainer trainer_state.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot SFT loss curves from trainer_state.json")
    p.add_argument(
        "--trainer_state",
        type=str,
        default="outputs/checkpoints/sft_full_3090/checkpoint-273/trainer_state.json",
        help="Prefer the final checkpoint copy (full log_history).",
    )
    p.add_argument("--out_png", type=str, default="outputs/report_assets/sft_loss_curves.png")
    p.add_argument("--out_csv", type=str, default="outputs/report_assets/sft_loss_history.csv")
    p.add_argument("--out_md", type=str, default="outputs/report_assets/sft_loss_summary.md")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    state_path = resolve_path(args.trainer_state, root)
    if not state_path.is_file():
        # fall back: any checkpoint under sft_full_3090 with largest step
        ckpt_root = resolve_path("outputs/checkpoints/sft_full_3090", root)
        candidates = sorted(ckpt_root.glob("checkpoint-*/trainer_state.json"), key=lambda p: p.parent.name)
        state_path = candidates[-1] if candidates else state_path
    if not state_path.is_file():
        raise SystemExit(f"Missing trainer_state.json at {state_path}")

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    history = state.get("log_history") or []

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for h in history:
        step = h.get("step")
        if step is None:
            continue
        if "loss" in h and "eval_loss" not in h:
            train_rows.append({"step": step, "train_loss": float(h["loss"])})
        if "eval_loss" in h:
            eval_rows.append({"step": step, "eval_loss": float(h["eval_loss"])})

    out_csv = resolve_path(args.out_csv, root)
    out_png = resolve_path(args.out_png, root)
    out_md = resolve_path(args.out_md, root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Wide CSV: one row per global step appearing in either series (sparse columns OK)
    all_steps = sorted({r["step"] for r in train_rows} | {r["step"] for r in eval_rows})
    train_by_s = {r["step"]: r["train_loss"] for r in train_rows}
    eval_by_s = {r["step"]: r["eval_loss"] for r in eval_rows}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "eval_loss"])
        w.writeheader()
        for s in all_steps:
            w.writerow(
                {
                    "step": s,
                    "train_loss": train_by_s.get(s, ""),
                    "eval_loss": eval_by_s.get(s, ""),
                }
            )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    if train_rows:
        ax.plot(
            [r["step"] for r in train_rows],
            [r["train_loss"] for r in train_rows],
            marker="o",
            markersize=2,
            linewidth=1.2,
            label="Training loss (logged)",
        )
    if eval_rows:
        ax.plot(
            [r["step"] for r in eval_rows],
            [r["eval_loss"] for r in eval_rows],
            marker="s",
            markersize=4,
            linewidth=1.5,
            label="Validation / eval loss",
        )
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Loss")
    ax.set_title("SFT (full fine-tune): training vs validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    last_train = train_rows[-1]["train_loss"] if train_rows else None
    last_eval = eval_rows[-1]["eval_loss"] if eval_rows else None
    md_lines = [
        "# SFT loss summary",
        "",
        f"Parsed from `{state_path.relative_to(root) if str(state_path).startswith(str(root)) else state_path}`.",
        "",
        "- **Training loss** rows come from `log_history` entries that include `loss` (no `eval_loss` on the same row).",
        "- **Eval loss** rows come from entries that include `eval_loss` (end-of-epoch evaluations; different logging frequency than train).",
        "",
        f"- Steps logged (train): **{len(train_rows)}**; eval points: **{len(eval_rows)}**.",
        f"- Last training loss (final log): **{last_train:.4f}**" if last_train is not None else "",
        f"- Last eval loss: **{last_eval:.4f}**" if last_eval is not None else "",
        "",
        f"Figure: `{out_png.relative_to(root) if str(out_png).startswith(str(root)) else out_png}`",
        f"CSV: `{out_csv.relative_to(root) if str(out_csv).startswith(str(root)) else out_csv}`",
        "",
    ]
    out_md.write_text("\n".join(ln for ln in md_lines if ln is not None) + "\n", encoding="utf-8")
    print(f"Wrote {out_png}, {out_csv}, {out_md}")


if __name__ == "__main__":
    main()
