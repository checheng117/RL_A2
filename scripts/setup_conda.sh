#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${CONDA_ENV_NAME:-rlhw2_qwen35_3090}"

if ! command -v conda &>/dev/null; then
  echo "conda not found. Install Miniconda/Anaconda first."
  exit 1
fi

echo "Creating conda env: $ENV_NAME (python=3.10)"
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda create -n "$ENV_NAME" python=3.10 -y

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing PyTorch (CUDA 12.1 wheels). For cu124 use: pip install torch --index-url https://download.pytorch.org/whl/cu124"
pip install --upgrade pip
pip install "torch>=2.5.0,<2.8.0" "torchvision>=0.20.0" "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cu121

echo "Installing project requirements (HF stack; must match TRL / transformers)"
pip install -r "$ROOT/environment/requirements.txt"

echo "Run: conda activate $ENV_NAME && python environment/check_env.py"
