"""Batch inference on JSONL with prompt field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.inference.generate import generate_completion
from src.training.common import load_merged_config
from src.training.modeling import load_causal_lm_base, load_model_with_sft_adapter, load_tokenizer
from src.utils.path_utils import find_project_root, resolve_path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch greedy generation")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="PEFT adapter directory")
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--smoke_n", type=int, default=0, help="If >0, only first N rows")
    p.add_argument("--dry_run", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    root = find_project_root()
    cfg = load_merged_config(args.config, root)
    inf = cfg.get("inference", {}) or {}
    max_new = int(args.max_new_tokens or inf.get("max_new_tokens", 256))
    do_sample = bool(inf.get("do_sample", False))
    temp = float(inf.get("temperature", 0.7))
    top_p = float(inf.get("top_p", 0.9))

    tokenizer = load_tokenizer(cfg)
    ckpt = resolve_path(args.checkpoint, root)
    if args.dry_run:
        print(f"Would load adapter from {ckpt} and read {args.input_jsonl}")
        return

    base = load_causal_lm_base(cfg)
    model = __import__("peft").PeftModel.from_pretrained(base, str(ckpt))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    in_path = resolve_path(args.input_jsonl, root)
    out_path = resolve_path(args.output_jsonl, root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.smoke_n > 0:
        rows = rows[: args.smoke_n]

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc="generate"):
            prompt = row.get("prompt") or row.get("instruction")
            if not prompt and "text" in row:
                # recover user prompt from stored field if missing
                prompt = row.get("prompt")
            if not prompt:
                raise ValueError("Row missing prompt")
            out = generate_completion(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new,
                do_sample=do_sample,
                temperature=temp,
                top_p=top_p,
            )
            rec = {**row, "prediction": out}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
