"""
Full-parameter SFT using HuggingFace Trainer + DataCollatorForSeq2Seq,
aligned with train_code_with_data/sft/train_sft.py (prompt/response + label masking).
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.training.common import apply_dot_overrides, load_merged_config, log_file_path, output_checkpoint_dir
from src.utils.hf_env import load_hf_token_from_dotenv
from src.utils.logging_utils import setup_run_logging
from src.utils.path_utils import find_project_root, resolve_path
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Teacher-style full SFT (HF Trainer)")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    return p


def tokenize_fn(examples, tokenizer, max_length: int):
    input_ids_list = []
    labels_list = []

    for prompt, response in zip(examples["prompt"], examples["response"]):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    load_hf_token_from_dotenv()
    cfg = load_merged_config(args.config, root)
    cfg = apply_dot_overrides(cfg, args.override)
    tcfg = cfg.get("training", {}) or {}

    if args.smoke_test:
        tcfg["smoke_test"] = True
    if args.dry_run:
        tcfg["dry_run"] = True
    cfg["training"] = tcfg

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)

    out_dir = output_checkpoint_dir(cfg, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_file_path(cfg, "train.log", root)
    setup_run_logging(log_path)

    model_name = cfg.get("project", {}).get("base_model", "Qwen/Qwen3.5-0.8B")
    max_length = int(tcfg.get("max_length", 2048))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if bool(tcfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    data_cfg = cfg.get("data", {}) or {}
    train_path = resolve_path(data_cfg.get("train_file", "data/processed/sft_train.jsonl"), root)
    val_path = resolve_path(data_cfg.get("validation_file", "data/processed/sft_val.jsonl"), root)

    ds_train = load_dataset("json", data_files=str(train_path), split="train")
    ds_val = load_dataset("json", data_files=str(val_path), split="train")

    if tcfg.get("smoke_test"):
        n = min(64, len(ds_train))
        ds_train = ds_train.select(range(n))
        nv = min(max(8, n // 4), len(ds_val))
        ds_val = ds_val.select(range(nv))
        tcfg["num_train_epochs"] = 1

    remove_cols = list(ds_train.column_names)

    ds_train = ds_train.map(
        lambda x: tokenize_fn(x, tokenizer, max_length),
        batched=True,
        remove_columns=remove_cols,
    )
    ds_val = ds_val.map(
        lambda x: tokenize_fn(x, tokenizer, max_length),
        batched=True,
        remove_columns=list(ds_val.column_names),
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(tcfg.get("num_train_epochs", 3)),
        learning_rate=float(tcfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        eval_strategy=str(tcfg.get("eval_strategy", "epoch")),
        save_strategy=str(tcfg.get("save_strategy", "epoch")),
        load_best_model_at_end=bool(tcfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(tcfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(tcfg.get("greater_is_better", False)),
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        report_to="none",
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 0)),
        seed=seed,
        bf16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,
        ),
    )

    if tcfg.get("dry_run"):
        batch = trainer.data_collator([ds_train[0]])
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
        with torch.no_grad():
            _ = model(**batch)
        logger.info("dry_run OK")
        return

    resume = args.resume_from_checkpoint or tcfg.get("resume_from_checkpoint")
    train_kw = {}
    if resume:
        train_kw["resume_from_checkpoint"] = str(resolve_path(str(resume), root))

    trainer.train(**train_kw)

    best_dest = out_dir / "best"
    if best_dest.exists():
        shutil.rmtree(best_dest)
    best_src = trainer.state.best_model_checkpoint
    if best_src and Path(best_src).is_dir():
        shutil.copytree(best_src, best_dest, dirs_exist_ok=True)
    else:
        cps = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if cps:
            shutil.copytree(cps[-1], best_dest, dirs_exist_ok=True)
        else:
            trainer.save_model(str(best_dest))
    tokenizer.save_pretrained(str(best_dest))
    logger.info("Saved best-style checkpoint to %s", best_dest)


if __name__ == "__main__":
    main()
