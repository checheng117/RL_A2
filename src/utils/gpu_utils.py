"""GPU / dtype selection for single-card 3090-friendly training."""
from __future__ import annotations

from typing import Any


def pick_autocast_dtype(prefer_bf16: bool = True) -> tuple[str, dict[str, Any]]:
    """Return TrainingArguments-style bf16/fp16 flags."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "fp32", {"bf16": False, "fp16": False}
        cap = torch.cuda.get_device_capability(0)
        if prefer_bf16 and cap[0] >= 8:
            return "bf16", {"bf16": True, "fp16": False}
        return "fp16", {"bf16": False, "fp16": True}
    except ImportError:
        return "fp32", {"bf16": False, "fp16": False}


def torch_compute_dtype(name: str):
    import torch

    n = (name or "float16").lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16"):
        return torch.float16
    return torch.float32
