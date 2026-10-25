#!/bin/bash
# Python environment for exllamav3-rocm: Python 3.12 + PyTorch ROCm 10.2 (TheRock nightly) wheels + runtime deps.
# usage: rocm/scripts/setup_env.sh [env_dir]   (default: ./.venv-rocm, needs conda or python3.12)
set -euo pipefail
ENV_DIR=${1:-$(pwd)/.venv-rocm}
GFX_TARGET=${GFX_TARGET:-gfx906}
INDEX_URL=${INDEX_URL:-https://nightly.repo.amd.com/rocm/whl-next/}
if command -v conda >/dev/null; then
    conda create -y -p "$ENV_DIR" python=3.12 >/dev/null
    source "$(conda info --base)/bin/activate" "$ENV_DIR"
else
    python3.12 -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
fi
pip install --upgrade pip
# torch pulls in the matching rocm[libraries]; the device-* extra adds the GPU-specific kernels
pip install --pre "torch[device-${GFX_TARGET}]==2.13.0" "torchvision[device-${GFX_TARGET}]==0.28.*" --index-url "$INDEX_URL"
pip install "huggingface_hub[hf_transfer]" tokenizers "numpy>=2.1" rich typing_extensions safetensors ninja \
            pillow pyyaml marisa_trie pydantic "llguidance>=1.7.0"
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
echo "environment ready: $ENV_DIR"
 