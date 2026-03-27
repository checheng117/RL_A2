"""Model, tokenizer, quantization, and LoRA attachment."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.gpu_utils import torch_compute_dtype
from src.utils.hf_env import load_hf_token_from_dotenv


def ensure_trl_model_compat(module: nn.Module) -> None:
    """TRL trainers set model.warnings_issued; some HF+Peft stacks omit it."""
    if "warnings_issued" not in module.__dict__:
        module.warnings_issued = {}  # type: ignore[attr-defined]


def load_tokenizer(cfg: dict[str, Any]):
    load_hf_token_from_dotenv()
    name = cfg.get("project", {}).get("base_model", "Qwen/Qwen3.5-0.8B")
    tok_cfg = cfg.get("tokenizer", {}) or {}
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=tok_cfg.get("trust_remote_code", True),
    )
    ps = tok_cfg.get("padding_side", "right")
    tokenizer.padding_side = ps
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _bnb_config(cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    q = cfg.get("quantization", {}) or {}
    if not q.get("load_in_4bit", True):
        return None
    compute_dtype = torch_compute_dtype(q.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(q.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_causal_lm_base(cfg: dict[str, Any], device_map: str | dict | None = "auto"):
    import logging

    log = logging.getLogger(__name__)
    load_hf_token_from_dotenv()
    name = cfg.get("project", {}).get("base_model", "Qwen/Qwen3.5-0.8B")
    bnb = _bnb_config(cfg)
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if bnb is not None:
        try:
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = device_map
            return AutoModelForCausalLM.from_pretrained(name, **kwargs)
        except ImportError as e:
            log.warning(
                "4-bit load failed (%s); falling back to bf16/fp32 without bitsandbytes (uses more VRAM).",
                e,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("4-bit load failed (%s); retrying without quantization.", e)
    kwargs.pop("quantization_config", None)
    kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if torch.cuda.is_available():
        kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def lora_config_from_yaml(cfg: dict[str, Any]) -> LoraConfig:
    l = cfg.get("lora", {}) or {}
    return LoraConfig(
        r=int(l.get("r", 16)),
        lora_alpha=int(l.get("lora_alpha", 32)),
        lora_dropout=float(l.get("lora_dropout", 0.05)),
        bias=l.get("bias", "none"),
        task_type=l.get("task_type", "CAUSAL_LM"),
        target_modules=list(l.get("target_modules", [])),
    )


def attach_lora(model, cfg: dict[str, Any]):
    return get_peft_model(model, lora_config_from_yaml(cfg))


def load_dpo_policy_and_ref_from_full_sft(cfg: dict[str, Any], sft_dir: str | Path):
    """DPO after full SFT: frozen ref + policy = same merged weights + new LoRA adapters."""
    path = Path(sft_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"SFT checkpoint directory not found: {path}")
    if (path / "adapter_config.json").is_file():
        raise ValueError(
            f"Expected merged full-SFT dir without adapter_config.json; got {path}. "
            "Use load_model_with_sft_adapter for LoRA checkpoints."
        )
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    ref = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    policy_base = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    policy = get_peft_model(policy_base, lora_config_from_yaml(cfg))
    return policy, ref


def load_model_with_sft_adapter(cfg: dict[str, Any], adapter_path: str | Path):
    """Load 4-bit base and attach SFT LoRA adapter from disk."""
    base = load_causal_lm_base(cfg)
    path = Path(adapter_path)
    if not path.is_dir():
        raise FileNotFoundError(f"SFT adapter directory not found: {path}")
    model = PeftModel.from_pretrained(base, str(path), is_trainable=True)
    return model


def load_tokenizer_from_checkpoint_dir(cfg: dict[str, Any], ckpt: str | Path) -> AutoTokenizer:
    """Prefer tokenizer shipped with SFT/adapter dir; else Hub tokenizer from cfg."""
    p = Path(ckpt)
    if (p / "tokenizer_config.json").is_file() or (p / "tokenizer.json").is_file():
        tok_cfg = cfg.get("tokenizer", {}) or {}
        t = AutoTokenizer.from_pretrained(
            str(p),
            trust_remote_code=tok_cfg.get("trust_remote_code", True),
        )
        if t.pad_token_id is None:
            t.pad_token = t.eos_token
        return t
    return load_tokenizer(cfg)


def load_grpo_policy_from_sft_checkpoint(cfg: dict[str, Any], sft_dir: str | Path):
    """GRPO init: merged full-SFT dir → base weights + new trainable LoRA; LoRA SFT dir → resume PEFT."""
    path = Path(sft_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"SFT checkpoint directory not found: {path}")
    if (path / "adapter_config.json").is_file():
        return load_model_with_sft_adapter(cfg, path)
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    base = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    return get_peft_model(base, lora_config_from_yaml(cfg))


def maybe_gradient_checkpointing(model, cfg: dict[str, Any]) -> None:
    comp = cfg.get("compute", {}) or {}
    if comp.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
