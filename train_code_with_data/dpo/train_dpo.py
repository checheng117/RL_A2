# dpo/train_dpo.py
"""
DPO Training: Improve summary quality through preference comparison.

How DPO works:
  Instead of a scalar reward, DPO uses (chosen, rejected) response pairs.
  The model is trained to increase the likelihood of the chosen response
  relative to the rejected one, while staying close to a reference policy
  (controlled by beta).

Student Task:
  1. Configure the basic training setup by specifying the model and data paths to successfully run DPO training.
  2. Try to configure the DPO training by tuning hyperparameters like learning rate, beta, and batch size.    
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import sys
sys.path.append("..")
from data.process_data import load_jsonl, process_for_dpo


# ============================================================
# CONFIG — Edit this section before running
# ============================================================
# TODO (Mandatory): Set the paths below of your own model and data to successfully run DPO training.
# Path to your base model (should match the model used in SFT,
# or use the SFT checkpoint at ./outputs/sft/best for better results)
MODEL_NAME = "/data/models/Qwen2.5-0.5B"

# Path to your training data (.jsonl format)
# Each record must contain a prompt, a chosen response, and a rejected response
DATA_PATH = "../data/train.jsonl"

# ============================================================


def main():
    # ── 1. Load tokenizer & model ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,         # bfloat16 saves memory; use float32 if unsupported
        trust_remote_code=True,
    )

    # ── 2. Load & split data ─────────────────────────────────────────
    data  = load_jsonl(DATA_PATH)
    split = int(len(data) * 0.9)           # 90% train / 10% eval

    train_dataset = process_for_dpo(data[:split])
    eval_dataset  = process_for_dpo(data[split:])

    # ── 3. DPO Config ────────────────────────────────────────────────
    # TODO (Optional): Tune the parameters below and observe their effect
    #   on training stability, reward margin, and summary quality.
    dpo_config = DPOConfig(
        output_dir="./outputs/dpo",         # checkpoints are saved here

        # --- core hyperparameters ---
        num_train_epochs=3,                 # more epochs → stronger preference signal, but risk overfitting
        learning_rate=5e-7,                 # DPO needs a much smaller lr than SFT (typical: 1e-7 ~ 5e-6)
                                            # too high → model collapses or diverges from reference

        # --- beta: KL divergence penalty ---
        # Controls how far the model is allowed to drift from the reference policy.
        #   high beta (e.g. 0.5) → stays close to reference, conservative updates
        #   low  beta (e.g. 0.01) → more aggressive, higher risk of reward hacking
        beta=0.1,

        # --- batch size & gradient accumulation ---
        # effective batch size = per_device_train_batch_size × gradient_accumulation_steps
        # Here: 2 × 8 = 16
        per_device_train_batch_size=2,      # DPO processes pairs, so memory usage is ~2× SFT
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,      # increase if you reduce batch size

        # --- sequence length ---
        max_length=512,                     # max tokens for (prompt + response); increase for longer texts

        # --- logging & evaluation ---
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",

        # --- misc ---
        report_to="none",                   # set to "wandb" if you want experiment tracking
    )

    # ── 4. DPO Trainer ───────────────────────────────────────────────
    trainer = DPOTrainer(
        model=model,
        # ref_model: the reference policy the KL penalty is computed against.
        # None → TRL uses the initial model weights as reference (via EMA internally).
        # You can also pass an explicit reference model loaded from a checkpoint.
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # ── 5. Train & save ──────────────────────────────────────────────
    trainer.train()
    trainer.save_model("./outputs/dpo/best")
    print("DPO training complete! Best model saved to ./outputs/dpo/best")


if __name__ == "__main__":
    main()
