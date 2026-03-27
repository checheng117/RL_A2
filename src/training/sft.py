"""Supervised fine-tuning with TRL SFTTrainer + QLoRA."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from datasets import load_dataset

from src.training.common import apply_dot_overrides, load_merged_config, log_file_path, output_checkpoint_dir
from src.training.modeling import attach_lora, load_causal_lm_base, load_model_with_sft_adapter, load_tokenizer, maybe_gradient_checkpointing
from src.utils.gpu_utils import pick_autocast_dtype
from src.utils.logging_utils import setup_run_logging
from src.utils.path_utils import find_project_root, resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT (LoRA/QLoRA)")
    p.add_argument("--config", nargs="+", required=True, help="YAML files in merge order")
    p.add_argument("--override", action="append", default=[], help="dotted.key=value")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    root = find_project_root()
    cfg = load_merged_config(args.config, root)
    cfg = apply_dot_overrides(cfg, args.override)
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
    log_path = log_file_path(cfg, "train.log", root)
    setup_run_logging(log_path)

    tokenizer = load_tokenizer(cfg)
    data_cfg = cfg.get("data", {}) or {}
    train_path = resolve_path(data_cfg.get("train_file", "data/processed/sft_train.jsonl"), root)
    val_path = resolve_path(data_cfg.get("validation_file", "data/processed/sft_val.jsonl"), root)

    ds_train = load_dataset("json", data_files=str(train_path), split="train")
    ds_val = load_dataset("json", data_files=str(val_path), split="train")

    if tcfg.get("smoke_test"):
        n = min(int(tcfg.get("smoke_max_samples", 32)), len(ds_train))
        ds_train = ds_train.select(range(n))
        nv = min(max(4, n // 4), len(ds_val))
        ds_val = ds_val.select(range(nv))
        tcfg["max_steps"] = int(tcfg.get("smoke_max_steps", 3))
        tcfg["num_train_epochs"] = 1
        tcfg["save_steps"] = 1
        tcfg["eval_steps"] = 1
        tcfg["logging_steps"] = 1

    _, dtype_flags = pick_autocast_dtype(prefer_bf16=bool((cfg.get("compute", {}) or {}).get("bf16", True)))
    max_len = int(tcfg.get("max_seq_length", 1024))

    from trl import SFTConfig, SFTTrainer

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 3)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        eval_steps=int(tcfg.get("eval_steps", 200)),
        save_steps=int(tcfg.get("save_steps", 200)),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        load_best_model_at_end=bool(tcfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(tcfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(tcfg.get("greater_is_better", False)),
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=int(tcfg.get("max_steps", -1)) if int(tcfg.get("max_steps", -1) or -1) > 0 else -1,
        report_to="none",
        seed=seed,
        max_length=max_len,
        dataset_text_field="text",
        **dtype_flags,
    )

    resume_ckpt = args.resume_from_checkpoint or tcfg.get("resume_from_checkpoint")

    if tcfg.get("evaluate_only"):
        best = out_dir / "best"
        if not best.is_dir():
            raise SystemExit(f"Missing SFT adapter at {best}; train first.")
        model = load_model_with_sft_adapter(cfg, best)
        maybe_gradient_checkpointing(model, cfg)
    else:
        model = load_causal_lm_base(cfg)
        maybe_gradient_checkpointing(model, cfg)
        model = attach_lora(model, cfg)

    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
    )
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("processing_class", None)
        trainer_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**trainer_kwargs)

    if tcfg.get("dry_run"):
        sample = tokenizer(ds_train[0]["text"], return_tensors="pt")
        if torch.cuda.is_available():
            sample = {k: v.cuda() for k, v in sample.items()}
        with torch.no_grad():
            _ = model(**sample)
        logger.info("dry_run OK")
        return

    if tcfg.get("evaluate_only"):
        metrics = trainer.evaluate()
        logger.info("eval metrics: %s", metrics)
        return

    train_kw = {}
    if resume_ckpt:
        train_kw["resume_from_checkpoint"] = str(resolve_path(resume_ckpt, root))

    trainer.train(**train_kw)

    best_src = getattr(trainer.state, "best_model_checkpoint", None)
    if not best_src or not Path(best_src).is_dir():
        # fall back to latest checkpoint
        cps = sorted(out_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        best_src = str(cps[-1]) if cps else str(out_dir)
    best_dest = out_dir / "best"
    if best_dest.exists():
        shutil.rmtree(best_dest)
    shutil.copytree(best_src, best_dest, dirs_exist_ok=True)
    logger.info("Copied best checkpoint to %s (from %s)", best_dest, best_src)


if __name__ == "__main__":
    main()
