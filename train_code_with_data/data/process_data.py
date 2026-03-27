# data/process_data.py
"""
Data Processing Pipeline
========================

Transforms raw JSONL records into training-ready datasets for three
fine-tuning paradigms:

  - SFT  (Supervised Fine-Tuning)   : prompt → chosen response
  - DPO  (Direct Preference Opt.)   : prompt → (chosen, rejected) pair
  - GRPO (Group Relative Policy Opt.): prompt → reference + original answer

Raw Record Schema
-----------------
Each line in the source JSONL file must conform to the following structure::

    {
        "answer_zh":            str,   # Chinese source answer
        "answer_en":            str,   # English source answer
        "summary_zh_chosen":    str,   # Preferred Chinese structured summary
        "summary_zh_rejected":  str,   # Dis-preferred Chinese structured summary
        "summary_en_chosen":    str,   # Preferred English structured summary
        "summary_en_rejected":  str    # Dis-preferred English structured summary
    }

Target Output Format
--------------------
The model is trained to produce structured summaries in the following schema::

    [point]   one-sentence statement of the central claim
    [reason]
    1. first supporting reason
    2. second supporting reason
    3. third supporting reason
    [summary] a concise paragraph integrating the point and reasons

Usage
-----
    python data/process_data.py

Outputs (written to data/)
--------------------------
    sft_{train,test}.jsonl
    dpo_{train,test}.jsonl
    grpo_{train,test}.jsonl
"""

import json
from typing import Literal

from datasets import Dataset


# ============================================================
# CONSTANTS
# ============================================================

# Fraction of data allocated to the training split; remainder is held-out test.
TRAIN_RATIO: float = 0.9

# Supported language codes for bilingual processing.
SupportedLang = Literal["zh", "en"]


# ============================================================
# PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE: str = (
    "请将以下回答总结为结构化摘要：\n\n"
    "回答：{answer}\n\n"
    "摘要："
)


# ============================================================
# I/O UTILITIES
# ============================================================


def load_jsonl(path: str) -> list[dict]:
    """
    Load a newline-delimited JSON file into a list of dictionaries.

    Each non-empty line is parsed as an independent JSON object.
    Lines that are blank or contain only whitespace are silently skipped.

    Parameters
    ----------
    path : str
        Filesystem path to the ``.jsonl`` source file.

    Returns
    -------
    list[dict]
        Ordered list of parsed records, preserving source file order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    json.JSONDecodeError
        If any non-empty line cannot be parsed as valid JSON.
    """
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ============================================================
# DATASET PROCESSORS
# ============================================================


def process_for_sft(
    data: list[dict],
    lang: SupportedLang = "zh",
) -> Dataset:
    """
    Convert raw records into a supervised fine-tuning (SFT) dataset.

    Each sample pairs the formatted prompt with the *chosen* (preferred)
    reference summary, providing a direct input–output mapping for
    standard cross-entropy training.

    Parameters
    ----------
    data : list[dict]
        Raw records loaded from the source JSONL file.
    lang : {"zh", "en"}, optional
        Language variant to extract; defaults to ``"zh"`` (Chinese).

    Returns
    -------
    datasets.Dataset
        HuggingFace ``Dataset`` with columns:

        - ``prompt``   : formatted instruction string
        - ``response`` : chosen reference summary (target sequence)
    """
    samples: list[dict] = []
    for item in data:
        samples.append({
            "prompt":   PROMPT_TEMPLATE.format(answer=item[f"answer_{lang}"]),
            "response": item[f"summary_{lang}_chosen"],
        })
    return Dataset.from_list(samples)


def process_for_dpo(
    data: list[dict],
    lang: SupportedLang = "zh",
) -> Dataset:
    """
    Convert raw records into a Direct Preference Optimization (DPO) dataset.

    Each sample contains a prompt together with a *chosen* (preferred) and a
    *rejected* (dis-preferred) completion, enabling contrastive preference
    learning without an explicit reward model.

    Parameters
    ----------
    data : list[dict]
        Raw records loaded from the source JSONL file.
    lang : {"zh", "en"}, optional
        Language variant to extract; defaults to ``"zh"`` (Chinese).

    Returns
    -------
    datasets.Dataset
        HuggingFace ``Dataset`` with columns:

        - ``prompt``   : formatted instruction string
        - ``chosen``   : preferred reference summary
        - ``rejected`` : dis-preferred reference summary
    """
    samples: list[dict] = []
    for item in data:
        samples.append({
            "prompt":   PROMPT_TEMPLATE.format(answer=item[f"answer_{lang}"]),
            "chosen":   item[f"summary_{lang}_chosen"],
            "rejected": item[f"summary_{lang}_rejected"],
        })
    return Dataset.from_list(samples)


def process_for_grpo(
    data: list[dict],
    lang: SupportedLang = "zh",
) -> Dataset:
    """
    Convert raw records into a Group Relative Policy Optimization (GRPO) dataset.

    Each sample retains the original source answer alongside the chosen
    reference summary, enabling reward computation based on both content
    fidelity and structural adherence during online RL training.

    Parameters
    ----------
    data : list[dict]
        Raw records loaded from the source JSONL file.
    lang : {"zh", "en"}, optional
        Language variant to extract; defaults to ``"zh"`` (Chinese).

    Returns
    -------
    datasets.Dataset
        HuggingFace ``Dataset`` with columns:

        - ``prompt``          : formatted instruction string
        - ``reference``       : chosen reference summary (for reward scoring)
        - ``original_answer`` : raw source answer (for content-grounding reward)
    """
    samples: list[dict] = []
    for item in data:
        answer = item[f"answer_{lang}"]
        samples.append({
            "prompt":          PROMPT_TEMPLATE.format(answer=answer), # Include target answer that the model should learn to summarize. 
            "reference":       item[f"summary_{lang}_chosen"],
            "original_answer": answer, # text need to be summarized
        })
    return Dataset.from_list(samples)
    

# ============================================================
# ENTRY POINT
# ============================================================


def main() -> None:
    """
    Execute the full data processing pipeline.

    Steps
    -----
    1. Load the raw JSONL dataset from ``data/dataset.jsonl``.
    2. Partition records into a 90 % training split and a 10 % held-out
       test split using a deterministic sequential cut.
    3. For each split, produce SFT, DPO, and GRPO variants and serialize
       them as JSONL files under the ``data/`` directory.
    4. Print a brief summary of split sizes to stdout.
    """
    raw_data: list[dict] = load_jsonl("data/dataset.jsonl")

    # Deterministic sequential split — no shuffling to preserve reproducibility.
    cut: int = int(len(raw_data) * TRAIN_RATIO)
    splits: dict[str, list[dict]] = {
        "train": raw_data[:cut],
        "test":  raw_data[cut:],
    }

    processors = {
        "sft":  process_for_sft,
        "dpo":  process_for_dpo,
        "grpo": process_for_grpo,
    }

    for split_name, split_data in splits.items():
        for paradigm, processor_fn in processors.items():
            output_path = f"data/{paradigm}_{split_name}.jsonl"
            processor_fn(split_data).to_json(output_path, force_ascii=False)
            print(f"  Saved {len(split_data):>5} records → {output_path}")

    print(
        f"\nSplit summary — "
        f"train: {len(splits['train'])} samples, "
        f"test: {len(splits['test'])} samples."
    )


if __name__ == "__main__":
    main()
