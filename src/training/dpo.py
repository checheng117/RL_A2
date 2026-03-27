"""DPO training with TRL DPOTrainer from SFT LoRA checkpoint."""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.common import apply_dot_overrides, load_merged_config, log_file_path, output_checkpoint_dir
from src.training.modeling import (
    ensure_trl_model_compat,
    load_dpo_policy_and_ref_from_full_sft,
    load_model_with_sft_adapter,
    load_tokenizer,
    maybe_gradient_checkpointing,
)
from src.utils.gpu_utils import pick_autocast_dtype
from src.utils.logging_utils import setup_run_logging
from src.utils.path_utils import find_project_root, resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DPO from SFT adapter")
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

    sft_path = resolve_path(tcfg.get("sft_adapter_path", "outputs/checkpoints/sft_lora_3090/best"), root)
    sft_has_peft = (sft_path / "adapter_config.json").is_file()
    tokenizer = (
        AutoTokenizer.from_pretrained(str(sft_path), trust_remote_code=True)
        if not sft_has_peft
        else load_tokenizer(cfg)
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_cfg = cfg.get("data", {}) or {}
    train_path = resolve_path(data_cfg.get("train_file", "data/processed/dpo_train.jsonl"), root)
    val_path = resolve_path(data_cfg.get("validation_file", "data/processed/dpo_val.jsonl"), root)
    ds_train = load_dataset("json", data_files=str(train_path), split="train")
    ds_val = load_dataset("json", data_files=str(val_path), split="train")

    if tcfg.get("smoke_test"):
        n = min(int(tcfg.get("smoke_max_samples", 24)), len(ds_train))
        ds_train = ds_train.select(range(n))
        nv = min(max(4, n // 3), len(ds_val))
        ds_val = ds_val.select(range(nv))
        tcfg["max_steps"] = int(tcfg.get("smoke_max_steps", 2))
        tcfg["save_steps"] = 1
        tcfg["eval_steps"] = 1
        tcfg["logging_steps"] = 1

    _, dtype_flags = pick_autocast_dtype(prefer_bf16=bool((cfg.get("compute", {}) or {}).get("bf16", True)))
    max_len = int(tcfg.get("max_seq_length", 1024))

    from trl import DPOConfig, DPOTrainer

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tcfg.get("learning_rate", 5e-5)),
        beta=float(tcfg.get("beta", 0.2)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 2)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.05)),
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
        max_length=max_len,
        report_to="none",
        seed=seed,
        **dtype_flags,
    )

    if tcfg.get("evaluate_only"):
        best = out_dir / "best"
        if not best.is_dir():
            raise SystemExit(f"Missing DPO adapter at {best}")
        if sft_has_peft:
            model = load_model_with_sft_adapter(cfg, best)
            ref = load_model_with_sft_adapter(cfg, sft_path)
        else:
            load_kw: dict = {"trust_remote_code": True}
            if torch.cuda.is_available():
                load_kw["torch_dtype"] = torch.bfloat16
                load_kw["device_map"] = "auto"
            else:
                load_kw["torch_dtype"] = torch.float32
            base = AutoModelForCausalLM.from_pretrained(str(sft_path), **load_kw)
            peft_mod = __import__("peft")
            model = peft_mod.PeftModel.from_pretrained(base, str(best))
            ref = AutoModelForCausalLM.from_pretrained(str(sft_path), **load_kw)
        for p in ref.parameters():
            p.requires_grad = False
        ref.eval()
    else:
        if sft_has_peft:
            model = load_model_with_sft_adapter(cfg, sft_path)
            ref = load_model_with_sft_adapter(cfg, sft_path)
        else:
            model, ref = load_dpo_policy_and_ref_from_full_sft(cfg, sft_path)
        for p in ref.parameters():
            p.requires_grad = False
        ref.eval()

    maybe_gradient_checkpointing(model, cfg)

    ensure_trl_model_compat(model)
    ensure_trl_model_compat(ref)

    trainer_kw = dict(
        model=model,
        ref_model=ref,
        args=dpo_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
    )
    try:
        trainer = DPOTrainer(**trainer_kw)
    except TypeError:
        trainer_kw.pop("processing_class", None)
        trainer_kw["tokenizer"] = tokenizer
        trainer = DPOTrainer(**trainer_kw)

    if tcfg.get("dry_run"):
        logger.info("dry_run OK (DPO model + ref loaded, trainer built)")
        return

    if tcfg.get("evaluate_only"):
        logger.info("eval: %s", trainer.evaluate())
        return

    train_kw = {}
    if args.resume_from_checkpoint or tcfg.get("resume_from_checkpoint"):
        train_kw["resume_from_checkpoint"] = str(
            resolve_path(args.resume_from_checkpoint or tcfg.get("resume_from_checkpoint"), root)
        )
    trainer.train(**train_kw)

    best_src = getattr(trainer.state, "best_model_checkpoint", None)
    if not best_src or not Path(best_src).is_dir():
        cps = sorted(out_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        best_src = str(cps[-1]) if cps else str(out_dir)
    best_dest = out_dir / "best"
    if best_dest.exists():
        shutil.rmtree(best_dest)
    shutil.copytree(best_src, best_dest, dirs_exist_ok=True)
    logger.info("Copied best DPO checkpoint to %s", best_dest)


if __name__ == "__main__":
    main()
