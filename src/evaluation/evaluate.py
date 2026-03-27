"""Unified evaluation: generate (optional) and score predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference.generate import generate_completion, generate_from_plain_prompt
from src.metrics.format_adherence import batch_format_adherence
from src.metrics.strict_format_adherence import batch_strict_format_adherence
from src.metrics.reward_metrics import avg_reward
from src.metrics.rouge_metrics import rouge_l_f1
from src.training.common import load_merged_config
from src.training.modeling import load_causal_lm_base
from src.utils.path_utils import ensure_dir, find_project_root, resolve_path

STAGE_INPUT = {
    "sft": "data/processed/sft_test.jsonl",
    "dpo": "data/processed/sft_test.jsonl",
    "dpo_retune": "data/processed/sft_test.jsonl",
    "dpo_retune_v2": "data/processed/sft_test.jsonl",
    "grpo_v1": "data/processed/grpo_test.jsonl",
    "grpo_v4": "data/processed/grpo_test.jsonl",
    "grpo_v5": "data/processed/grpo_test.jsonl",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate checkpoint on test split")
    p.add_argument("--config", nargs="+", default=["configs/base.yaml", "configs/inference.yaml"])
    p.add_argument("--stage", type=str, required=True, help="sft|dpo|grpo_v1|grpo_v4|...")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input_jsonl", type=str, default=None)
    p.add_argument("--predictions_jsonl", type=str, default=None)
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument(
        "--reward_variant",
        type=str,
        default=None,
        help="For avg_reward scoring; default: v4 (sft/dpo) or match grpo stage",
    )
    return p


def _load_model_for_eval(cfg: dict, checkpoint: Path, root: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    adapter_cfg = checkpoint / "adapter_config.json"
    eval_cfg = cfg.get("evaluation", {}) or {}
    peft_base_raw = eval_cfg.get("peft_base_checkpoint")
    if adapter_cfg.is_file():
        peft = __import__("peft")
        if peft_base_raw:
            base_path = resolve_path(str(peft_base_raw), root)
            load_kw: dict = {
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True,
            }
            if torch.cuda.is_available():
                load_kw["device_map"] = "auto"
            model = AutoModelForCausalLM.from_pretrained(str(base_path), **load_kw)
        else:
            model = load_causal_lm_base(cfg)
        model = peft.PeftModel.from_pretrained(model, str(checkpoint))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
    return model, tokenizer


def run_generate(
    cfg: dict,
    checkpoint: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    smoke: bool,
    root: Path,
) -> None:
    model, tokenizer = _load_model_for_eval(cfg, checkpoint, root)
    inf = cfg.get("inference", {}) or {}
    max_new = int(inf.get("max_new_tokens", 512))
    do_sample = bool(inf.get("do_sample", False))
    temp = float(inf.get("temperature", 0.7))
    top_p = float(inf.get("top_p", 0.9))
    use_chat = bool(inf.get("use_chat_template", False))
    max_in = int(inf.get("max_input_length", 1024))

    rows: list[dict] = []
    with open(input_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if smoke:
        rows = rows[: max(4, min(8, len(rows)))]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="eval-generate"):
            prompt = row.get("prompt")
            if not prompt:
                raise ValueError("Missing prompt in row")
            if use_chat:
                pred = generate_completion(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temp,
                    top_p=top_p,
                )
            else:
                pred = generate_from_plain_prompt(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=max_new,
                    do_sample=do_sample,
                    temperature=temp,
                    top_p=top_p,
                    max_input_length=max_in,
                )
            fout.write(json.dumps({**row, "prediction": pred}, ensure_ascii=False) + "\n")


def _count_output_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def score_file(predictions_jsonl: Path, reward_variant: str, tokenizer=None) -> dict:
    rows: list[dict] = []
    with open(predictions_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    preds = [r.get("prediction", "") for r in rows]
    # Assignment Part I: reference = summary_en_chosen (English); response mirrors training target.
    refs = [
        (r.get("summary_en_chosen") or r.get("response") or r.get("completion") or "").strip()
        for r in rows
    ]
    prompts = [r.get("prompt", "") for r in rows]
    sources = [r.get("answer_en", "") for r in rows]

    fmt = batch_format_adherence(preds)
    fmt_strict = batch_strict_format_adherence(preds)
    rouge = rouge_l_f1(preds, refs) if any(refs) else 0.0
    avg_len_chars = sum(len(p) for p in preds) / max(len(preds), 1)
    avg_len_tokens = 0.0
    if tokenizer is not None and preds:
        tok_lens = [_count_output_tokens(tokenizer, p) for p in preds]
        avg_len_tokens = sum(tok_lens) / len(tok_lens)
    rwd = avg_reward(prompts, preds, sources, reward_variant)

    return {
        "n": len(rows),
        "rouge_l_f1": rouge,
        "avg_output_length_tokens": avg_len_tokens,
        "avg_output_length_chars": avg_len_chars,
        "format_adherence": fmt,
        "format_adherence_strict": fmt_strict,
        "strict_format_rate": fmt_strict.get("strict_format_rate", 0.0),
        "avg_reward": rwd,
        "reward_variant": reward_variant,
    }


def run_single_eval(
    *,
    root: Path,
    cfg: dict,
    stage: str,
    checkpoint: Path,
    input_jsonl: Path,
    pred_path: Path,
    skip_generate: bool,
    smoke_test: bool,
    reward_variant: str | None,
    metrics_json_out: Path | None = None,
) -> dict:
    """Generate (unless skipped), score predictions, optionally write metrics JSON; return metrics dict."""
    stage = stage.lower()
    if not skip_generate:
        run_generate(cfg, checkpoint, input_jsonl, pred_path, smoke=smoke_test, root=root)
    elif not pred_path.is_file():
        raise FileNotFoundError(f"--skip_generate but missing {pred_path}")

    rw = reward_variant
    if rw is None:
        if "grpo_v1" in stage or stage == "grpo_v1":
            rw = "v1"
        elif "grpo_v4" in stage or stage == "grpo_v4":
            rw = "v4"
        elif "grpo_v5" in stage or stage == "grpo_v5":
            rw = "v5"
        else:
            rw = "v4"
    tok = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True)
    metrics = score_file(pred_path, rw, tokenizer=tok)
    metrics["stage"] = stage
    metrics["checkpoint"] = str(checkpoint)
    eval_cfg = cfg.get("evaluation", {}) or {}
    metrics["reference_field"] = eval_cfg.get("reference_field", "summary_en_chosen")
    metrics["output_length_unit"] = "tokens_primary_chars_auxiliary"
    metrics["predictions_jsonl"] = str(pred_path)
    if metrics_json_out is not None:
        ensure_dir(metrics_json_out.parent)
        with open(metrics_json_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    cfg = load_merged_config(args.config, root)
    stage = args.stage.lower()
    in_path = resolve_path(args.input_jsonl or STAGE_INPUT.get(stage, STAGE_INPUT["sft"]), root)
    pred_dir = resolve_path((cfg.get("paths", {}) or {}).get("predictions_dir", "outputs/predictions"), root)
    ensure_dir(pred_dir)
    pred_path = resolve_path(
        args.predictions_jsonl or str(pred_dir / f"{stage}_test_greedy.jsonl"),
        root,
    )
    ckpt = resolve_path(args.checkpoint, root)
    metrics_path = resolve_path(
        (cfg.get("paths", {}) or {}).get("metrics_dir", "outputs/metrics"), root
    ) / f"{stage}_test_metrics.json"
    metrics = run_single_eval(
        root=root,
        cfg=cfg,
        stage=stage,
        checkpoint=ckpt,
        input_jsonl=in_path,
        pred_path=pred_path,
        skip_generate=args.skip_generate,
        smoke_test=args.smoke_test,
        reward_variant=args.reward_variant,
        metrics_json_out=metrics_path,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
