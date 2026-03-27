#!/usr/bin/env python3
"""Verify Python, CUDA, GPU, and key package imports for this project."""
from __future__ import annotations

import importlib
import os
import sys

# Project root on PYTHONPATH when run as `python environment/check_env.py` from repo root
try:
    from src.utils.hf_env import load_hf_token_from_dotenv
except ImportError:
    load_hf_token_from_dotenv = None  # type: ignore[misc,assignment]


def main() -> int:
    ok = True
    print(f"Python: {sys.version.split()[0]} (expect 3.10.x)")
    if not sys.version_info[:2] == (3, 10):
        print("  WARN: Project targets Python 3.10; other versions may work but are unsupported.")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"  cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  device: {torch.cuda.get_device_name(0)}")
            print(
                "  mem total (GB): "
                f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f}"
            )
        else:
            print("  No CUDA GPU visible — training expects CUDA; CPU ok for checks/smoke parsing.")
    except Exception as e:  # noqa: BLE001
        print(f"torch import failed: {e}")
        ok = False

    packages = [
        "transformers",
        "datasets",
        "peft",
        "trl",
        "accelerate",
        "rouge_score",
        "yaml",
        "bitsandbytes",
    ]
    if load_hf_token_from_dotenv:
        load_hf_token_from_dotenv()
    hf_ok = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print(f"HF_TOKEN available for hub downloads: {hf_ok} (value not printed)")

    for name in packages:
        try:
            mod = importlib.import_module("yaml" if name == "yaml" else name)
            ver = getattr(mod, "__version__", None)
            if ver is None and name == "yaml":
                import importlib.metadata as im

                try:
                    ver = im.version("PyYAML")
                except Exception:
                    ver = "?"
            elif ver is None:
                ver = "?"
            print(f"{name}: import ok (version {ver})")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: IMPORT FAILED — {e}")
            if name != "bitsandbytes":
                ok = False
            else:
                print("  (bitsandbytes optional for fp16 LoRA without 4-bit; QLoRA needs it)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
