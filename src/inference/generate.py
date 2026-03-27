"""Single-sample generation helper."""
from __future__ import annotations

import torch


@torch.inference_mode()
def generate_completion(
    model,
    tokenizer,
    user_prompt: str,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    gen_kw: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p
    out = model.generate(**inputs, **gen_kw)
    text_out = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return text_out.strip()


@torch.inference_mode()
def generate_from_plain_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_input_length: int = 1024,
) -> str:
    """Teacher-style: encode raw prompt (no chat template), greedy or sample."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    ).to(device)
    gen_kw: dict = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = top_p
    else:
        gen_kw["do_sample"] = False
    out = model.generate(**gen_kw)
    plen = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()
