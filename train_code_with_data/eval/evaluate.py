# eval/evaluate.py
"""
Evaluation Script: Structured Summary Quality Assessment
========================================================

This script evaluates the output quality of a fine-tuned causal language
model on the structured summarization task along two complementary axes:

Metrics
-------
1. **ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation — Longest
   Common Subsequence):
   Measures lexical overlap between generated and reference summaries at the
   sequence level.  Higher values indicate greater content coverage.
   The F-measure is reported as the primary scalar metric.

2. **Format Adherence**:
   Measures the proportion of model outputs that correctly instantiate the
   required three-section schema::

       [point]   <non-empty content>
       [reason]  <non-empty content>
       [summary] <non-empty content>

   A response is deemed adherent only when all three tags appear in the
   prescribed order and each section contains substantive (non-whitespace)
   text.

Evaluation Protocol
-------------------
By default, the script evaluates on the held-out 10 % of the data file
(i.e., records from index ``floor(N × 0.9)`` onward), mirroring the
train/test split applied during data processing.

Usage
-----
::

    python eval/evaluate.py --model_path ./outputs/sft/best
    python eval/evaluate.py --model_path ./outputs/dpo/best  --batch_size 8
    python eval/evaluate.py --model_path ./outputs/grpo/best --output_file results/grpo.jsonl

CLI Arguments
-------------
--model_path     Path to the model checkpoint directory (required or default).
--data_path      Path to the processed SFT ``.jsonl`` evaluation file.
--max_new_tokens Maximum number of tokens to generate per prompt.
--temperature    Sampling temperature; set to 0 for deterministic greedy decoding.
--batch_size     Number of prompts processed per forward pass (tune for VRAM).
--output_file    Optional path to persist per-sample results as a ``.jsonl`` file.
"""

import re
import json
import argparse
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("..")
from data.process_data import load_jsonl, process_for_grpo
from tqdm import tqdm

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================
# TODO (Mandatory): Set the default paths below to your model checkpoint and evaluation data to successfully run this script without CLI arguments.

DEFAULT_MODEL_PATH: str = "../rl/outputs/grpo/best"
DEFAULT_DATA_PATH: str  = "../data/train.jsonl"

# Fraction of the data file used for training; the complement is the eval split.
EVAL_SPLIT_RATIO: float = 0.9

# Generation hyper-parameters.
MAX_NEW_TOKENS: int   = 512
TEMPERATURE: float    = 0.7   # Set to 0.0 to enable greedy (deterministic) decoding.
DO_SAMPLE: bool       = True
DEFAULT_BATCH_SIZE: int = 4
MAX_INPUT_LENGTH: int = 1024


# ============================================================
# FORMAT VALIDATION
# ============================================================

# Compiled pattern enforcing the mandatory three-section output schema.
# Each section must contain at least one non-whitespace character.
_FORMAT_PATTERN: re.Pattern = re.compile(
    r"\[point\]\s*(.+?)\s*\[reason\]\s*(.+?)\s*\[summary\]\s*(.+?)$",
    re.DOTALL,
)


def check_format(response: str) -> bool:
    """
    Determine whether a model response adheres to the required output schema.

    A response is considered adherent if and only if:

    - All three structural tags (``[point]``, ``[reason]``, ``[summary]``)
      are present.
    - The tags appear in the prescribed sequential order.
    - Each section contains at least one non-whitespace character.

    Parameters
    ----------
    response : str
        Raw decoded model output string.

    Returns
    -------
    bool
        ``True`` if the response is structurally valid; ``False`` otherwise.
    """
    match = _FORMAT_PATTERN.search(response)
    if not match:
        return False
    # Verify that every captured section contains substantive content.
    return all(group.strip() for group in match.groups())


def compute_format_adherence(responses: list[str]) -> dict:
    """
    Compute aggregate format adherence statistics over a collection of responses.

    Parameters
    ----------
    responses : list[str]
        List of raw model-generated output strings.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - ``adherence_pct``  (float): percentage of structurally valid responses.
        - ``num_adherent``   (int):   count of valid responses.
        - ``num_total``      (int):   total number of responses evaluated.
    """
    num_adherent: int = sum(1 for r in responses if check_format(r))
    num_total: int    = len(responses)
    return {
        "adherence_pct": round(num_adherent / num_total * 100, 2) if num_total else 0.0,
        "num_adherent":  num_adherent,
        "num_total":     num_total,
    }


# ============================================================
# ROUGE-L COMPUTATION
# ============================================================


def compute_rouge_l(
    predictions: list[str],
    references:  list[str],
    use_stemmer: bool = False,
) -> dict:
    """
    Compute corpus-level ROUGE-L scores across paired prediction–reference sets.

    ROUGE-L is based on the Longest Common Subsequence (LCS) between a
    candidate and a reference string, capturing sentence-level structural
    similarity beyond simple n-gram overlap.

    Parameters
    ----------
    predictions : list[str]
        Model-generated summary strings.
    references : list[str]
        Gold-standard reference summary strings aligned with ``predictions``.
    use_stemmer : bool, optional
        Whether to apply Porter stemming prior to scoring.  Recommended for
        English text; should be disabled for Chinese (default: ``False``).

    Returns
    -------
    dict
        Corpus-averaged scores with the following keys:

        - ``rougeL_precision`` (float): mean precision across all pairs.
        - ``rougeL_recall``    (float): mean recall across all pairs.
        - ``rougeL_fmeasure``  (float): mean F-measure — the **primary metric**.
    """
    scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=use_stemmer)

    precision_scores: list[float] = []
    recall_scores:    list[float] = []
    fmeasure_scores:  list[float] = []

    for pred, ref in zip(predictions, references):
        # Guard against empty strings, which would cause division-by-zero
        # inside the rouge_score library.
        pred = pred.strip() or " "
        ref  = ref.strip()  or " "

        scores = scorer_obj.score(ref, pred)
        precision_scores.append(scores["rougeL"].precision)
        recall_scores.append(scores["rougeL"].recall)
        fmeasure_scores.append(scores["rougeL"].fmeasure)

    def _mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "rougeL_precision": _mean(precision_scores),
        "rougeL_recall":    _mean(recall_scores),
        "rougeL_fmeasure":  _mean(fmeasure_scores),
    }


# ============================================================
# RESPONSE GENERATION
# ============================================================


def generate_responses(
    model,
    tokenizer,
    prompts:        list[str],
    max_new_tokens: int   = MAX_NEW_TOKENS,
    temperature:    float = TEMPERATURE,
    do_sample:      bool  = DO_SAMPLE,
    batch_size:     int   = DEFAULT_BATCH_SIZE,
) -> list[str]:
    """
    Generate one structured summary per prompt using the provided model.

    Prompts are processed in mini-batches to bound peak GPU memory consumption.
    Left-padding is applied so that all sequences within a batch terminate at
    the same position, which is required for correct batch generation with
    causal language models.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The fine-tuned causal language model to evaluate.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer corresponding to ``model``.
    prompts : list[str]
        Formatted instruction strings, one per evaluation sample.
    max_new_tokens : int, optional
        Maximum number of tokens to generate beyond the prompt length.
    temperature : float, optional
        Softmax temperature for sampling.  A value of ``0.0`` triggers
        deterministic greedy decoding regardless of ``do_sample``.
    do_sample : bool, optional
        Whether to use stochastic sampling; ignored when ``temperature == 0``.
    batch_size : int, optional
        Number of prompts to process per forward pass.

    Returns
    -------
    list[str]
        Decoded response strings in the same order as ``prompts``.
        The prompt prefix is stripped from each output.
    """
    model.eval()
    responses: list[str] = []
    use_greedy: bool = (temperature == 0.0) or (not do_sample)

    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating Responses"):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        # Tokenize with left-padding to align sequence endings within the batch.
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(model.device)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        with torch.no_grad():
            if use_greedy:
                output_ids = model.generate(**generation_kwargs, do_sample=False)
            else:
                output_ids = model.generate(
                    **generation_kwargs,
                    do_sample=True,
                    temperature=temperature,
                )

        # Isolate newly generated token IDs by discarding the prompt prefix.
        prompt_length: int = inputs["input_ids"].shape[1]
        for ids in output_ids:
            new_token_ids = ids[prompt_length:]
            responses.append(
                tokenizer.decode(new_token_ids, skip_special_tokens=True)
            )

        num_done = min(batch_start + batch_size, len(prompts))
        print(f"  Generated {num_done} / {len(prompts)} responses", end="\r")

    print()  # Terminate the in-place progress line.
    return responses


# ============================================================
# CLI & ENTRY POINT
# ============================================================


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns
    -------
    argparse.Namespace
        Parsed argument object with attributes corresponding to each flag.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned structured summarization model using "
            "ROUGE-L and format adherence metrics."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the processed evaluation ``.jsonl`` file.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Maximum number of tokens to generate per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help="Sampling temperature; set to 0 for greedy decoding.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts to process per forward pass.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save per-sample evaluation results as ``.jsonl``.",
    )
    return parser.parse_args()


def _print_results(rouge: dict, fmt: dict) -> None:
    """
    Render a formatted evaluation results table to stdout.

    Parameters
    ----------
    rouge : dict
        Output of :func:`compute_rouge_l`.
    fmt : dict
        Output of :func:`compute_format_adherence`.
    """
    sep = "=" * 60
    print(f"\n{sep}")
    print("  EVALUATION RESULTS")
    print(sep)
    print(f"  ROUGE-L  Precision : {rouge['rougeL_precision']:.4f}")
    print(f"  ROUGE-L  Recall    : {rouge['rougeL_recall']:.4f}")
    print(f"  ROUGE-L  F-measure : {rouge['rougeL_fmeasure']:.4f}"
          "  ← primary metric")
    print(f"  {'─' * 40}")
    print(
        f"  Format Adherence   : {fmt['adherence_pct']:.2f}%"
        f"  ({fmt['num_adherent']} / {fmt['num_total']})"
    )
    print(f"{sep}\n")


def _save_per_sample_results(
    output_file: str,
    prompts:     list[str],
    predictions: list[str],
    references:  list[str],
) -> None:
    """
    Persist per-sample evaluation records to a JSONL file.

    Each record contains the prompt, model prediction, gold reference,
    sample-level ROUGE-L F-measure, and a boolean format adherence flag.

    Parameters
    ----------
    output_file : str
        Destination file path (parent directories are created if absent).
    prompts : list[str]
        Formatted instruction strings.
    predictions : list[str]
        Model-generated summary strings.
    references : list[str]
        Gold-standard reference summary strings.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate a single scorer object and reuse it across all samples
    # to avoid repeated object construction overhead.
    sample_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    with open(output_path, "w", encoding="utf-8") as fh:
        for prompt, pred, ref in zip(prompts, predictions, references):
            score = sample_scorer.score(
                ref.strip()  or " ",
                pred.strip() or " ",
            )
            record = {
                "prompt":           prompt,
                "prediction":       pred,
                "reference":        ref,
                "rougeL_fmeasure":  round(score["rougeL"].fmeasure, 4),
                "format_adherent":  check_format(pred),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Per-sample results saved to: {output_file}\n")


def main() -> None:
    """
    Orchestrate the full evaluation pipeline.

    Pipeline Steps
    --------------
    1. Parse CLI arguments and display the evaluation configuration.
    2. Load the tokenizer and model from the specified checkpoint directory.
    3. Load the evaluation data and extract the held-out test split.
    4. Generate model responses for all evaluation prompts.
    5. Compute ROUGE-L and format adherence metrics.
    6. Print a formatted results table to stdout.
    7. Optionally serialize per-sample results to a JSONL file.
    """
    args = parse_args()

    # ── 1. Display configuration ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Model : {args.model_path}")
    print(f"  Data  : {args.data_path}")
    print(f"{'=' * 60}\n")

    # ── 2. Load tokenizer and model ──────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    # Left-padding is mandatory for correct batch generation with causal LMs.
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}.\n")

    # ── 3. Load and partition evaluation data ────────────────────────────────
    data  = load_jsonl(args.data_path)
    split = int(len(data) * 0.9)           # 90% train / 10% eval

    train_dataset = process_for_grpo(data[:split])
    eval_dataset  = process_for_grpo(data[split:])
    print(f"Evaluating on {len(eval_dataset)} held-out samples.\n")

    prompts:    list[str] = [item["prompt"]      for item in eval_dataset]
    references: list[str] = [item["reference"] for item in eval_dataset]

    # ── 4. Generate model responses ──────────────────────────────────────────
    print("Generating responses...")
    predictions: list[str] = generate_responses(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    # ── 5. Compute evaluation metrics ────────────────────────────────────────
    rouge_results:  dict = compute_rouge_l(predictions, references)
    format_results: dict = compute_format_adherence(predictions)

    # ── 6. Display results ───────────────────────────────────────────────────
    _print_results(rouge_results, format_results)

    # ── 7. (Optional) Persist per-sample results ─────────────────────────────
    if args.output_file:
        _save_per_sample_results(
            args.output_file, prompts, predictions, references
        )


if __name__ == "__main__":
    main()
