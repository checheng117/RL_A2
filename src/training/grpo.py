"""GRPO training with TRL GRPOTrainer and custom reward variants."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import TrainerCallback

from src.rewards.reward_fn import make_trl_reward_fn
from src.training.common import apply_dot_overrides, load_merged_config, log_file_path, output_checkpoint_dir
from src.training.modeling import (
    ensure_trl_model_compat,
    load_grpo_policy_from_sft_checkpoint,
    load_tokenizer_from_checkpoint_dir,
    maybe_gradient_checkpointing,
)
from src.utils.gpu_utils import pick_autocast_dtype
from src.utils.logging_utils import setup_run_logging
from src.utils.path_utils import find_project_root, resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class LinearBetaScheduleCallback(TrainerCallback):
    """Optional linear beta schedule for GRPO KL coefficient."""

    def __init__(
        self,
        trainer,
        *,
        start_beta: float,
        end_beta: float,
        start_step: int,
        end_step: int,
    ) -> None:
        self.trainer = trainer
        self.start_beta = float(start_beta)
        self.end_beta = float(end_beta)
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        if self.end_step <= self.start_step:
            raise ValueError("beta schedule requires end_step > start_step")
        self._last_logged_step = -1

    def _beta_at(self, step: int) -> float:
        if step <= self.start_step:
            return self.start_beta
        if step >= self.end_step:
            return self.end_beta
        ratio = (step - self.start_step) / (self.end_step - self.start_step)
        return self.start_beta + ratio * (self.end_beta - self.start_beta)

    def on_step_begin(self, args, state, control, **kwargs):
        step = int(state.global_step)
        beta_now = self._beta_at(step)
        self.trainer.beta = float(beta_now)
        if step == self.start_step or step == self.end_step or step % max(int(args.logging_steps), 1) == 0:
            if step != self._last_logged_step:
                logger.info("KL beta schedule update: step=%d beta=%.6f", step, beta_now)
                self._last_logged_step = step
        return control


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO from SFT adapter")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    cfg = apply_dot_overrides(load_merged_config(args.config, root), args.override)
    tcfg = cfg.get("training", {}) or {}
    if args.smoke_test:
        tcfg["smoke_test"] = True
    if args.dry_run:
        tcfg["dry_run"] = True
    if args.evaluate_only:
        tcfg["evaluate_only"] = True
    cfg["training"] = tcfg

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)

    out_dir = output_checkpoint_dir(cfg, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(log_file_path(cfg, "train.log", root))

    sft_path = resolve_path(
        tcfg.get("sft_adapter_path", "outputs/checkpoints/sft_full_3090/best"),
        root,
    )
    tokenizer = load_tokenizer_from_checkpoint_dir(cfg, sft_path)
    tokenizer.padding_side = "left"

    data_cfg = cfg.get("data", {}) or {}
    train_path = resolve_path(data_cfg.get("train_file", "data/processed/grpo_train.jsonl"), root)
    ds_train = load_dataset("json", data_files=str(train_path), split="train")

    if "prompt" not in ds_train.column_names:
        raise SystemExit("GRPO dataset must contain a 'prompt' column (see data pipeline).")

    if tcfg.get("smoke_test"):
        n = min(int(tcfg.get("smoke_max_samples", 16)), len(ds_train))
        ds_train = ds_train.select(range(max(n, 1)))
        tcfg["max_steps"] = int(tcfg.get("smoke_max_steps", 2))
        tcfg["save_steps"] = 1
        tcfg["logging_steps"] = 1

    variant = str(tcfg.get("reward_variant", "v1"))
    reward_fn = make_trl_reward_fn(variant)

    model = load_grpo_policy_from_sft_checkpoint(cfg, sft_path)
    maybe_gradient_checkpointing(model, cfg)
    ensure_trl_model_compat(model)

    _, dtype_flags = pick_autocast_dtype(prefer_bf16=bool((cfg.get("compute", {}) or {}).get("bf16", True)))

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise SystemExit(
            "trl with GRPOTrainer is required. Install environment/requirements.txt (trl>=0.18)."
        ) from e

    max_comp = int(tcfg.get("max_completion_length", 256))
    # max_prompt_length in YAML is advisory; TRL GRPOConfig truncates via model/tokenizer defaults.

    grpo_args = GRPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(tcfg.get("learning_rate", 1e-5)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 1)),
        logging_steps=int(tcfg.get("logging_steps", 5)),
        save_steps=int(tcfg.get("save_steps", 50)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        max_steps=int(tcfg.get("max_steps", -1)) if int(tcfg.get("max_steps", -1) or -1) > 0 else -1,
        max_completion_length=max_comp,
        num_generations=int(tcfg.get("num_generations", 2)),
        beta=float(tcfg.get("beta", 0.04)),
        scale_rewards=False,
        report_to="none",
        seed=seed,
        **dtype_flags,
    )

    trainer_kw = dict(
        model=model,
        args=grpo_args,
        train_dataset=ds_train,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )
    try:
        trainer = GRPOTrainer(**trainer_kw)
    except TypeError:
        trainer_kw.pop("processing_class", None)
        trainer_kw["tokenizer"] = tokenizer
        trainer = GRPOTrainer(**trainer_kw)

    beta_sched = (tcfg.get("beta_schedule") or {})
    if isinstance(beta_sched, dict) and str(beta_sched.get("type", "")).lower() == "linear":
        cb = LinearBetaScheduleCallback(
            trainer,
            start_beta=float(beta_sched.get("start_beta", tcfg.get("beta", 0.04))),
            end_beta=float(beta_sched.get("end_beta", tcfg.get("beta", 0.04))),
            start_step=int(beta_sched.get("start_step", 0)),
            end_step=int(beta_sched.get("end_step", tcfg.get("max_steps", 0))),
        )
        trainer.add_callback(cb)
        logger.info(
            "Enabled linear KL beta schedule: start_beta=%.6f end_beta=%.6f start_step=%d end_step=%d",
            cb.start_beta,
            cb.end_beta,
            cb.start_step,
            cb.end_step,
        )

    if tcfg.get("dry_run"):
        logger.info("dry_run OK (GRPO trainer built; no generation run)")
        return

    if tcfg.get("evaluate_only"):
        if hasattr(trainer, "evaluate"):
            logger.info("eval: %s", trainer.evaluate())
        else:
            logger.info("evaluate_only: no evaluate() on trainer; skip")
        return

    train_kw = {}
    if args.resume_from_checkpoint or tcfg.get("resume_from_checkpoint"):
        train_kw["resume_from_checkpoint"] = str(
            resolve_path(args.resume_from_checkpoint or tcfg.get("resume_from_checkpoint"), root)
        )
    trainer.train(**train_kw)

    cps = sorted(out_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    best_src = str(cps[-1]) if cps else str(out_dir)
    best_dest = out_dir / "best"
    if best_dest.exists():
        shutil.rmtree(best_dest)
    shutil.copytree(best_src, best_dest, dirs_exist_ok=True)
    logger.info("Copied last GRPO checkpoint to %s", best_dest)


if __name__ == "__main__":
    main()
