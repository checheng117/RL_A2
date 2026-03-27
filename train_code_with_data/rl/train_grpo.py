# rl/train_grpo.py
"""
GRPO Training: Improve summary quality through reinforcement learning.

How GRPO works:
  For each prompt, the model samples a GROUP of responses (num_generations).
  A reward function scores each response. The model is updated to increase
  the probability of higher-reward responses relative to the group average —
  without needing a separate critic / value network (unlike PPO).

  Group-normalised advantage for response i:
      A_i = (R_i - mean(R)) / std(R)

  This makes GRPO much more memory-efficient than PPO for LLMs.

Student Task:
  1. Switch between reward_v1 ~ reward_v5 (see REWARD FUNCTION section below).
  2. For each version, observe the reward curve and generated summaries:
       - Does the model genuinely improve, or does it find a shortcut?
       - At which version does reward hacking disappear?
  3. Implement reward_v5 in reward_fn.py to close all remaining loopholes.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import sys
sys.path.append("..")
from data.process_data import load_jsonl, process_for_grpo


# ============================================================
# CONFIG — Edit this section before running
# ============================================================
# TODO (Mandatory): Set the paths below of your own model and data to successfully run GRPO training.
# Path to your base model (local directory or HuggingFace model ID).
# Tip: starting from the SFT checkpoint (../sft/outputs/sft/best)
#      usually leads to faster reward improvement than the raw base model.
MODEL_NAME = "../sft/outputs/sft/best_ckpt"

# Path to your training data (.jsonl format)
DATA_PATH = "../data/train.jsonl"

# ============================================================
# REWARD FUNCTION — Switch between versions to compare behaviour
# ============================================================
# Uncomment exactly ONE line. Work through v1 → v4 in order,
# then implement and test your own v5.
#
#   v1 — keyword presence only          (easiest to hack)
#   v2 — keyword presence + order
#   v3 — correct structure + content checks
#   v4 — v3 + reference keyword coverage
#   v5 — your design                    (hardest to hack)

# TODO (Mandatory): Switch between reward_v1 and one of reward_v2 ~ reward_v4 to observe how the model learns and hacks each reward design.
# from reward_fn import reward_v1 as reward_function; reward_type="v1"
# from reward_fn import reward_v2 as reward_function; reward_type="v2"
# from reward_fn import reward_v3 as reward_function; reward_type="v3"
from reward_fn import reward_v4 as reward_function; reward_type="v4"
# from reward_fn import reward_v5 as reward_function; reward_type="v5" --- IGNORE (uncomment after implementing v5) ---
print(f"="*20)
print(f"Using reward function: {reward_type}")
print(f"="*20)
# ============================================================


def make_reward_fn(tokenizer):
    """
    Wrap the chosen reward function to match the GRPO interface.

    GRPO expects: fn(prompts, completions, **kwargs) -> List[float]
      - prompts      : List[str] — the input prompts (unused here, but available)
      - completions  : List[str] — model-generated responses to score
      - **kwargs     : extra dataset columns forwarded automatically
                       (e.g. "original_answer" from process_for_grpo)

    Returns a list of scalar rewards, one per completion, in [0.0, 1.0].
    """
    def reward_fn(prompts, completions, **kwargs):
        # "original_answer" is passed through from the dataset column.
        # Falls back to empty strings if the column is absent.
        original_answers = kwargs.get("original_answer", [""] * len(completions))
        references = kwargs.get("reference", [""] * len(completions))

        rewards = []
        for completion, original, reference in zip(completions, original_answers, references):
            score = reward_function(
                response=completion,
                reference=reference,
                original_answer=original,
            )
            rewards.append(score)

        return rewards

    return reward_fn


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

    train_dataset = process_for_grpo(data[:split])
    eval_dataset  = process_for_grpo(data[split:])

    # ── 3. GRPO Config ───────────────────────────────────────────────
    # TODO (Optional): Tune the parameters below and observe their effect
    #   on reward curves, generation diversity, and summary quality.
    grpo_config = GRPOConfig(
        output_dir=f"./outputs_{reward_type}/grpo",        # checkpoints are saved here

        # --- core hyperparameters ---
        num_train_epochs=3,
        learning_rate=1e-6,                 # GRPO needs a very small lr (typical: 5e-7 ~ 5e-6)
                                            # too high → reward spikes then collapses

        # --- batch size & gradient accumulation ---
        # effective samples per update =
        #     per_device_train_batch_size × gradient_accumulation_steps × num_generations
        # Here: 2 × 8 × 4 = 64 response samples per update step
        per_device_train_batch_size=2,      # reduce to 1 if you get OOM
        gradient_accumulation_steps=8,

        # --- GRPO-specific parameters ---
        # num_generations: responses sampled per prompt per step.
        #   More → stable reward baseline, but higher memory cost.
        #   Must be ≥ 2 (GRPO needs within-group variance to compute advantages).
        num_generations=4,

        # temperature: sampling temperature during generation.
        #   Higher → more diverse group, better exploration of the reward landscape.
        #   Lower  → more deterministic outputs, less exploration.
        temperature=0.9,

        # --- logging & saving ---
        logging_steps=10,
        save_strategy="epoch",

        # --- misc ---
        report_to="none",                   # set to "wandb" for experiment tracking
    )

    # ── 4. GRPO Trainer ──────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=make_reward_fn(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── 5. Train & save ──────────────────────────────────────────────
    trainer.train()
    trainer.save_model(f"./outputs_{reward_type}/grpo/best")
    print(f"GRPO training complete! Best model saved to ./outputs_{reward_type}/grpo/best")


if __name__ == "__main__":
    main()
