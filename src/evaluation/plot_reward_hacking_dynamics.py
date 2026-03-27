"""Plot E2 dynamics from reward_hacking_dynamics_metrics.csv (sparse) or *_dense_metrics.csv."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--style",
        choices=["sparse", "dense"],
        default="sparse",
        help="dense: V1 E2-dense run + SFT; optional V4 reference from --ref_csv",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Default path depends on --style",
    )
    p.add_argument(
        "--ref_csv",
        type=str,
        default="outputs/report_assets/reward_hacking_dynamics_metrics.csv",
        help="Sparse E2 CSV for V4 reference markers (dense style only).",
    )
    p.add_argument("--out_dir", type=str, default=None)
    return p


def _load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(x: str, default: float = 0.0) -> float:
    if x is None or x.strip() == "":
        return default
    return float(x)


def _tight_layout_dense_caption(fig, style: str, v1_rows: list[dict]) -> None:
    """Dense plots: state explicitly which checkpoint steps were evaluated (no interpolation)."""
    if style == "dense" and v1_rows:
        steps = ", ".join(str(int(r["checkpoint_step"])) for r in v1_rows)
        fig.tight_layout(rect=[0, 0.09, 1, 1])
        fig.text(
            0.5,
            0.02,
            f"Dense run — only these GRPO-V1-dense steps on disk and evaluated: {steps}",
            ha="center",
            fontsize=7.5,
            color="0.35",
            transform=fig.transFigure,
        )
    else:
        fig.tight_layout()


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    if args.style == "dense":
        csv_path = resolve_path(
            args.csv or "outputs/report_assets/reward_hacking_dynamics_dense_metrics.csv", root
        )
        out_dir = resolve_path(args.out_dir or "outputs/report_assets/plots_dense", root)
        suffix = "_dense"
    else:
        csv_path = resolve_path(
            args.csv or "outputs/report_assets/reward_hacking_dynamics_metrics.csv", root
        )
        out_dir = resolve_path(args.out_dir or "outputs/report_assets/plots", root)
        suffix = ""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(csv_path)

    sft = [r for r in rows if r["model_family"] == "SFT"]
    if args.style == "dense":
        v1 = [r for r in rows if r["model_family"] == "GRPO-V1-dense"]
        v1.sort(key=lambda r: int(r["checkpoint_step"]))
        ref_path = resolve_path(args.ref_csv, root)
        v4: list[dict] = _load_rows(ref_path) if ref_path.is_file() else []
        v4 = [r for r in v4 if r["model_family"] == "GRPO-V4"]
    else:
        v1 = [r for r in rows if r["model_family"] == "GRPO-V1"]
        v4 = [r for r in rows if r["model_family"] == "GRPO-V4"]

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    def fname(base: str) -> str:
        return f"{base}{suffix}.png" if suffix else f"{base}.png"

    # A. reward vs step
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if v1:
        xs = [int(r["checkpoint_step"]) for r in v1]
        ys = [_float(str(r.get("avg_reward", ""))) for r in v1]
        lbl = "GRPO-V1 (E2-dense)" if args.style == "dense" else "GRPO-V1"
        ax.plot(xs, ys, marker="o", label=lbl, color="#1f77b4", linewidth=2)
    if args.style == "sparse":
        for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
            xs = [int(r["checkpoint_step"]) for r in data]
            ys = [_float(str(r.get("avg_reward", ""))) for r in data]
            ax.plot(xs, ys, marker=marker, label=label, color=color, linewidth=2)
    elif v4:
        xs = [int(r["checkpoint_step"]) for r in v4]
        ys = [_float(str(r.get("avg_reward", ""))) for r in v4]
        ax.scatter(xs, ys, marker="s", s=80, c="#ff7f0e", edgecolors="k", linewidths=0.5, zorder=5, label="GRPO-V4 (sparse ref)")
    ax.set_xlabel("Checkpoint step (training)")
    ax.set_ylabel("avg_reward (test, v1 variant for V1 curve)")
    ax.set_title("Test-set average reward vs training step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _tight_layout_dense_caption(fig, args.style, v1)
    fig.savefig(out_dir / fname("reward_vs_step"))
    plt.close(fig)

    # B. ROUGE-L
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if sft:
        r0 = sft[0]
        ax.axhline(
            _float(r0["rouge_l_f1"]),
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"SFT best (step 0): {_float(r0['rouge_l_f1']):.3f}",
        )
    if v1:
        xs = [int(r["checkpoint_step"]) for r in v1]
        ys = [_float(r["rouge_l_f1"]) for r in v1]
        ax.plot(xs, ys, marker="o", label="GRPO-V1 (E2-dense)" if args.style == "dense" else "GRPO-V1", color="#1f77b4", linewidth=2)
    if args.style == "sparse":
        for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
            xs = [int(r["checkpoint_step"]) for r in data]
            ys = [_float(r["rouge_l_f1"]) for r in data]
            ax.plot(xs, ys, marker=marker, label=label, color=color, linewidth=2)
    elif v4:
        xs = [int(r["checkpoint_step"]) for r in v4]
        ys = [_float(r["rouge_l_f1"]) for r in v4]
        ax.scatter(xs, ys, marker="s", s=80, c="#ff7f0e", edgecolors="k", linewidths=0.5, zorder=5, label="GRPO-V4 (sparse ref)")
    ax.set_xlabel("Checkpoint step (training)")
    ax.set_ylabel("ROUGE-L F1 (vs summary_en_chosen)")
    ax.set_title("Test ROUGE-L vs training step")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _tight_layout_dense_caption(fig, args.style, v1)
    fig.savefig(out_dir / fname("rouge_vs_step"))
    plt.close(fig)

    # C. strict format
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if sft:
        r0 = sft[0]
        ax.axhline(
            _float(r0["strict_format_rate"]),
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"SFT best (step 0): {_float(r0['strict_format_rate']):.3f}",
        )
    if v1:
        xs = [int(r["checkpoint_step"]) for r in v1]
        ys = [_float(r["strict_format_rate"]) for r in v1]
        ax.plot(xs, ys, marker="o", label="GRPO-V1 (E2-dense)" if args.style == "dense" else "GRPO-V1", color="#1f77b4", linewidth=2)
    if args.style == "sparse":
        for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
            xs = [int(r["checkpoint_step"]) for r in data]
            ys = [_float(r["strict_format_rate"]) for r in data]
            ax.plot(xs, ys, marker=marker, label=label, color=color, linewidth=2)
    elif v4:
        xs = [int(r["checkpoint_step"]) for r in v4]
        ys = [_float(r["strict_format_rate"]) for r in v4]
        ax.scatter(xs, ys, marker="s", s=80, c="#ff7f0e", edgecolors="k", linewidths=0.5, zorder=5, label="GRPO-V4 (sparse ref)")
    ax.set_xlabel("Checkpoint step (training)")
    ax.set_ylabel("Strict format rate")
    ax.set_title("Test strict format rate vs training step")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _tight_layout_dense_caption(fig, args.style, v1)
    fig.savefig(out_dir / fname("strict_format_vs_step"))
    plt.close(fig)

    # D. output length
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if sft:
        r0 = sft[0]
        ax.axhline(
            _float(r0["avg_output_length_tokens"]),
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"SFT best (step 0): {_float(r0['avg_output_length_tokens']):.1f} tok",
        )
    if v1:
        xs = [int(r["checkpoint_step"]) for r in v1]
        ys = [_float(r["avg_output_length_tokens"]) for r in v1]
        ax.plot(xs, ys, marker="o", label="GRPO-V1 (E2-dense)" if args.style == "dense" else "GRPO-V1", color="#1f77b4", linewidth=2)
    if args.style == "sparse":
        for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
            xs = [int(r["checkpoint_step"]) for r in data]
            ys = [_float(r["avg_output_length_tokens"]) for r in data]
            ax.plot(xs, ys, marker=marker, label=label, color=color, linewidth=2)
    elif v4:
        xs = [int(r["checkpoint_step"]) for r in v4]
        ys = [_float(r["avg_output_length_tokens"]) for r in v4]
        ax.scatter(xs, ys, marker="s", s=80, c="#ff7f0e", edgecolors="k", linewidths=0.5, zorder=5, label="GRPO-V4 (sparse ref)")
    ax.set_xlabel("Checkpoint step (training)")
    ax.set_ylabel("Mean output length (tokens)")
    ax.set_title("Test mean output length vs training step")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    _tight_layout_dense_caption(fig, args.style, v1)
    fig.savefig(out_dir / fname("output_length_vs_step"))
    plt.close(fig)

    # E. combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    v1_lbl = "GRPO-V1 (E2-dense)" if args.style == "dense" else "GRPO-V1"

    def plot_reward(ax_) -> None:
        if v1:
            xs = [int(r["checkpoint_step"]) for r in v1]
            ys = [_float(str(r.get("avg_reward", ""))) for r in v1]
            ax_.plot(xs, ys, marker="o", label=v1_lbl, color="#1f77b4", linewidth=1.8)
        if args.style == "sparse":
            for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
                xs = [int(r["checkpoint_step"]) for r in data]
                ys = [_float(str(r.get("avg_reward", ""))) for r in data]
                ax_.plot(xs, ys, marker=marker, label=label, color=color, linewidth=1.8)
        elif v4:
            xs = [int(r["checkpoint_step"]) for r in v4]
            ys = [_float(str(r.get("avg_reward", ""))) for r in v4]
            ax_.scatter(xs, ys, marker="s", s=70, c="#ff7f0e", edgecolors="k", linewidths=0.4, label="V4 ref", zorder=5)
        ax_.set_xlabel("Step")
        ax_.set_ylabel("avg_reward")
        ax_.set_title("Reward")
        ax_.grid(True, alpha=0.3)
        ax_.legend(fontsize=8)

    def plot_rouge(ax_) -> None:
        if sft:
            ax_.axhline(_float(sft[0]["rouge_l_f1"]), color="gray", linestyle="--", linewidth=1.5)
        if v1:
            xs = [int(r["checkpoint_step"]) for r in v1]
            ys = [_float(r["rouge_l_f1"]) for r in v1]
            ax_.plot(xs, ys, marker="o", label=v1_lbl, color="#1f77b4", linewidth=1.8)
        if args.style == "sparse":
            for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
                xs = [int(r["checkpoint_step"]) for r in data]
                ys = [_float(r["rouge_l_f1"]) for r in data]
                ax_.plot(xs, ys, marker=marker, label=label, color=color, linewidth=1.8)
        elif v4:
            xs = [int(r["checkpoint_step"]) for r in v4]
            ys = [_float(r["rouge_l_f1"]) for r in v4]
            ax_.scatter(xs, ys, marker="s", s=70, c="#ff7f0e", edgecolors="k", linewidths=0.4, label="V4 ref", zorder=5)
        ax_.set_xlabel("Step")
        ax_.set_ylabel("ROUGE-L")
        ax_.set_title("ROUGE-L (gray = SFT)")
        ax_.grid(True, alpha=0.3)
        ax_.legend(fontsize=8)

    def plot_strict(ax_) -> None:
        if sft:
            ax_.axhline(_float(sft[0]["strict_format_rate"]), color="gray", linestyle="--", linewidth=1.5)
        if v1:
            xs = [int(r["checkpoint_step"]) for r in v1]
            ys = [_float(r["strict_format_rate"]) for r in v1]
            ax_.plot(xs, ys, marker="o", label=v1_lbl, color="#1f77b4", linewidth=1.8)
        if args.style == "sparse":
            for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
                xs = [int(r["checkpoint_step"]) for r in data]
                ys = [_float(r["strict_format_rate"]) for r in data]
                ax_.plot(xs, ys, marker=marker, label=label, color=color, linewidth=1.8)
        elif v4:
            xs = [int(r["checkpoint_step"]) for r in v4]
            ys = [_float(r["strict_format_rate"]) for r in v4]
            ax_.scatter(xs, ys, marker="s", s=70, c="#ff7f0e", edgecolors="k", linewidths=0.4, label="V4 ref", zorder=5)
        ax_.set_xlabel("Step")
        ax_.set_ylabel("Strict rate")
        ax_.set_title("Strict format")
        ax_.set_ylim(-0.02, 1.05)
        ax_.grid(True, alpha=0.3)
        ax_.legend(fontsize=8)

    def plot_len(ax_) -> None:
        if sft:
            ax_.axhline(_float(sft[0]["avg_output_length_tokens"]), color="gray", linestyle="--", linewidth=1.5)
        if v1:
            xs = [int(r["checkpoint_step"]) for r in v1]
            ys = [_float(r["avg_output_length_tokens"]) for r in v1]
            ax_.plot(xs, ys, marker="o", label=v1_lbl, color="#1f77b4", linewidth=1.8)
        if args.style == "sparse":
            for data, label, color, marker in ((v4, "GRPO-V4", "#ff7f0e", "s"),):
                xs = [int(r["checkpoint_step"]) for r in data]
                ys = [_float(r["avg_output_length_tokens"]) for r in data]
                ax_.plot(xs, ys, marker=marker, label=label, color=color, linewidth=1.8)
        elif v4:
            xs = [int(r["checkpoint_step"]) for r in v4]
            ys = [_float(r["avg_output_length_tokens"]) for r in v4]
            ax_.scatter(xs, ys, marker="s", s=70, c="#ff7f0e", edgecolors="k", linewidths=0.4, label="V4 ref", zorder=5)
        ax_.set_xlabel("Step")
        ax_.set_ylabel("Tokens")
        ax_.set_title("Output length")
        ax_.grid(True, alpha=0.3)
        ax_.legend(fontsize=8)

    plot_reward(axes[0, 0])
    plot_rouge(axes[0, 1])
    plot_strict(axes[1, 0])
    plot_len(axes[1, 1])
    supt = "GRPO-V1 E2-dense dynamics (test)" if args.style == "dense" else "GRPO dynamics on held-out test (E2)"
    if args.style == "dense" and v1:
        steps = ", ".join(str(int(r["checkpoint_step"])) for r in v1)
        supt += f"\n(only checkpoints: {steps})"
    fig.suptitle(supt, fontsize=11 if args.style == "dense" and v1 else 13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / fname("combined_dynamics_summary"), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
