# sft/train_sft.py
"""
SFT Training: Teach the model to generate summaries via supervised fine-tuning.
Student Task: Understand the upper limit of SFT — how good can the model get
              when trained directly on (prompt, response) pairs?
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
import sys
sys.path.append("..")
from data.process_data import load_jsonl, process_for_sft


# ============================================================
# CONFIG — Edit this section before running
# ============================================================
# TODO (Mandatory): Set the paths below of your own model and data to successfully run SFT training.
# Path to your base model (local directory or HuggingFace model ID)
# e.g. "Qwen/Qwen2.5-0.5B" or "/data/models/Qwen2.5-0.5B"
MODEL_NAME = "/data/models/Qwen2.5-0.5B"

# Path to your training data (.jsonl format)
DATA_PATH = "../data/train.jsonl"

# Maximum token length per sample (prompt + response).
# Longer sequences use more GPU memory. Try 1024 if you run out of memory.
MAX_LENGTH = 2048

# ============================================================


def tokenize_fn(examples, tokenizer):
    """
    Convert raw (prompt, response) pairs into token IDs for training.

    Key idea:
      - input_ids  = prompt tokens + response tokens
      - labels     = -100 (ignored) for prompt + response tokens for loss
      This ensures the model only learns to predict the response, not the prompt.
    """
    input_ids_list = []
    labels_list    = []

    for prompt, response in zip(examples["prompt"], examples["response"]):
        # Tokenize prompt (no special tokens — we handle EOS manually)
        prompt_ids   = tokenizer.encode(prompt, add_special_tokens=False)

        # Tokenize response and append EOS so the model learns when to stop
        response_ids = tokenizer.encode(response, add_special_tokens=False) \
                       + [tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids

        # -100 tells PyTorch's CrossEntropyLoss to skip prompt tokens
        labels    = [-100] * len(prompt_ids) + response_ids

        # Truncate to MAX_LENGTH if the combined sequence is too long
        input_ids = input_ids[:MAX_LENGTH]
        labels    = labels[:MAX_LENGTH]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


def main():
    # ── 1. Load tokenizer & model ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token   # required for left-pad models

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,             # bfloat16 saves memory; use float32 if your GPU doesn't support it
        trust_remote_code=True,
    )

    # ── 2. Load & split data ─────────────────────────────────────────
    data  = load_jsonl(DATA_PATH)
    split = int(len(data) * 0.9)               # 90% train / 10% eval

    train_dataset = process_for_sft(data[:split])
    eval_dataset  = process_for_sft(data[split:])

    # Tokenize datasets (batched for speed; original text columns are dropped)
    train_dataset = train_dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_fn(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # ── 3. Training arguments ────────────────────────────────────────
    # TODO (Optional): Tune the parameters below and observe the effect
    #   on training loss, eval loss, and final summary quality.
    training_args = TrainingArguments(
        output_dir="./outputs/sft",             # checkpoints are saved here

        # --- core hyperparameters ---
        num_train_epochs=3,                     # more epochs → better fit, but risk overfitting
        learning_rate=2e-5,                     # typical SFT range: 1e-5 ~ 5e-5

        # --- batch size & gradient accumulation ---
        # effective batch size = per_device_train_batch_size
        #                        × gradient_accumulation_steps
        #                        × num_GPUs
        # Here: 4 × 4 = 16 (single GPU)
        per_device_train_batch_size=4,          # reduce to 2 if you get OOM
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        # --- logging & evaluation ---
        logging_steps=10,                       # print loss every N steps
        eval_strategy="epoch",                  # evaluate once per epoch
        save_strategy="epoch",                  # save checkpoint once per epoch
        load_best_model_at_end=True,            # keep the checkpoint with lowest eval loss

        # --- misc ---
        report_to="none",                       # set to "wandb" if you want experiment tracking
        dataloader_num_workers=0,
    )

    # ── 4. Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,               # pads sequence length to multiples of 8 for efficiency
        ),
    )

    # ── 5. Train & save ──────────────────────────────────────────────
    trainer.train()
    trainer.save_model("./outputs/sft/best")
    tokenizer.save_pretrained("./outputs/sft/best")
    print("SFT training complete! Best model saved to ./outputs/sft/best")


if __name__ == "__main__":
    main()
