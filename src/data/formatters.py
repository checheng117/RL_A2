"""Prompt / response formatting for SFT, DPO, GRPO."""
from __future__ import annotations

from typing import Any

SUMMARY_INSTRUCTION = """You are a concise summarization assistant.
Read the source answer and produce a structured summary strictly in the format below.
[point] one-sentence core viewpoint
[reason] 1. ... 2. ... 3. ...
[summary] one-sentence closing summary

Source answer:
"""


def build_user_prompt(answer_en: str) -> str:
    return f"{SUMMARY_INSTRUCTION}{answer_en.strip()}"


def format_chat_text(tokenizer, user_text: str, assistant_text: str) -> str:
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def row_to_sft(
    row: dict[str, Any],
    tokenizer,
    answer_key: str,
    summary_key: str,
) -> dict[str, Any]:
    ans = (row.get(answer_key) or "").strip()
    summ = (row.get(summary_key) or "").strip()
    user = build_user_prompt(ans)
    text = format_chat_text(tokenizer, user, summ)
    return {
        **row,
        "prompt": user,
        "completion": summ,
        "text": text,
    }


def row_to_dpo(
    row: dict[str, Any],
    answer_key: str,
    chosen_key: str,
    rejected_key: str,
) -> dict[str, Any]:
    ans = (row.get(answer_key) or "").strip()
    chosen = (row.get(chosen_key) or "").strip()
    rejected = (row.get(rejected_key) or "").strip()
    prompt = build_user_prompt(ans)
    return {
        **row,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def row_to_grpo(row: dict[str, Any], answer_key: str) -> dict[str, Any]:
    ans = (row.get(answer_key) or "").strip()
    prompt = build_user_prompt(ans)
    return {**row, "prompt": prompt, "answer_en": ans}


def english_sft_record(
    row: dict[str, Any],
    answer_key: str = "answer_en",
    chosen_key: str = "summary_en_chosen",
) -> dict[str, Any]:
    """Assignment main task: plain English instruction + target summary_en_chosen."""
    ans = (row.get(answer_key) or "").strip()
    summ = (row.get(chosen_key) or "").strip()
    return {**row, "prompt": build_user_prompt(ans), "response": summ}


def english_dpo_record(
    row: dict[str, Any],
    answer_key: str = "answer_en",
    chosen_key: str = "summary_en_chosen",
    rejected_key: str = "summary_en_rejected",
) -> dict[str, Any]:
    ans = (row.get(answer_key) or "").strip()
    chosen = (row.get(chosen_key) or "").strip()
    rejected = (row.get(rejected_key) or "").strip()
    return {**row, "prompt": build_user_prompt(ans), "chosen": chosen, "rejected": rejected}


def english_grpo_record(
    row: dict[str, Any],
    answer_key: str = "answer_en",
    chosen_key: str = "summary_en_chosen",
) -> dict[str, Any]:
    ans = (row.get(answer_key) or "").strip()
    ref = (row.get(chosen_key) or "").strip()
    return {
        **row,
        "prompt": build_user_prompt(ans),
        "reference": ref,
        "original_answer": ans,
    }
