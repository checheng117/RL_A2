"""Load Hugging Face token from project .env without logging secrets."""
from __future__ import annotations

import os

from src.utils.path_utils import find_project_root


def load_hf_token_from_dotenv() -> bool:
    """
    If HF_TOKEN is not already set, try to read it from <project_root>/.env.
    Returns True if HF_TOKEN is set after this call (either was already set or loaded).
    Never prints the token value.
    """
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    root = find_project_root()
    env_path = root / ".env"
    if not env_path.is_file():
        return False
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key == "HF_TOKEN" and val:
                os.environ["HF_TOKEN"] = val
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", val)
                return True
    except OSError:
        return False
    return bool(os.environ.get("HF_TOKEN"))
