"""
Teacher-aligned data helpers (source: train_code_with_data/data/process_data.py).
Prompt template, column names, and Dataset schemas match the reference implementation.
"""
from __future__ import annotations

import json
from typing import Literal

from datasets import Dataset

SupportedLang = Literal["zh", "en"]

PROMPT_TEMPLATE: str = (
    "请将以下回答总结为结构化摘要：\n\n"
    "回答：{answer}\n\n"
    "摘要："
)

TRAIN_RATIO: float = 0.9


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def process_for_sft(data: list[dict], lang: SupportedLang = "zh") -> Dataset:
    samples: list[dict] = []
    for item in data:
        samples.append(
            {
                "prompt": PROMPT_TEMPLATE.format(answer=item[f"answer_{lang}"]),
                "response": item[f"summary_{lang}_chosen"],
            }
        )
    return Dataset.from_list(samples)


def process_for_dpo(data: list[dict], lang: SupportedLang = "zh") -> Dataset:
    samples: list[dict] = []
    for item in data:
        samples.append(
            {
                "prompt": PROMPT_TEMPLATE.format(answer=item[f"answer_{lang}"]),
                "chosen": item[f"summary_{lang}_chosen"],
                "rejected": item[f"summary_{lang}_rejected"],
            }
        )
    return Dataset.from_list(samples)


def process_for_grpo(data: list[dict], lang: SupportedLang = "zh") -> Dataset:
    samples: list[dict] = []
    for item in data:
        answer = item[f"answer_{lang}"]
        samples.append(
            {
                "prompt": PROMPT_TEMPLATE.format(answer=answer),
                "reference": item[f"summary_{lang}_chosen"],
                "original_answer": answer,
            }
        )
    return Dataset.from_list(samples)
