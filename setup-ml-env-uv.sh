#!/usr/bin/env bash
# setup-ml-env-uv.sh - Fine-tuning environment for WSL2 Ubuntu 24.04, uv edition.
# Target: RTX 4060 Ti 8GB + 16GB RAM, all installs via China mirrors.
# Safe to re-run: finished steps are skipped or are idempotent.

set -euo pipefail

step() { echo; echo "================================================================"; echo "  $1"; echo "================================================================"; }

step "1/6 apt mirror -> Aliyun + base packages (incl. python3.12-venv, pipx)"
if grep -q "archive.ubuntu.com\|security.ubuntu.com" /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; then
    sudo sed -i 's@//archive.ubuntu.com@//mirrors.aliyun.com@g; s@//security.ubuntu.com@//mirrors.aliyun.com@g' /etc/apt/sources.list.d/ubuntu.sources
    echo "apt sources switched to mirrors.aliyun.com"
else
    echo "apt sources already switched, skip."
fi
sudo apt update
sudo apt install -y build-essential git curl wget ca-certificates python3.12 python3.12-venv python3.12-dev pipx

step "2/6 install uv (via pipx, Tsinghua PyPI)"
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q '.local/bin' "$HOME/.bashrc"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
if command -v uv >/dev/null 2>&1; then
    echo "uv already installed: $(uv --version), skip."
else
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple pipx install uv
fi
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv --version

step "3/6 create venv at ~/venvs/ft (system python 3.12, no download)"
if [ -d "$HOME/venvs/ft" ]; then
    echo "venv already exists, skip."
else
    uv venv "$HOME/venvs/ft" --python /usr/bin/python3.12 --seed
fi
# shellcheck disable=SC1091
source "$HOME/venvs/ft/bin/activate"

step "4/6 install PyTorch 2.5.1+cu124 (Aliyun mirror)"
uv pip install torch==2.5.1 \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu124 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

step "5/6 install fine-tuning stack (Tsinghua PyPI)"
uv pip install llamafactory bitsandbytes accelerate peft trl datasets modelscope

step "6/6 HuggingFace mirror + verification"
if ! grep -q "HF_ENDPOINT" "$HOME/.bashrc"; then
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> "$HOME/.bashrc"
    echo "HF_ENDPOINT added to ~/.bashrc"
fi
export HF_ENDPOINT=https://hf-mirror.com

echo
echo "--- nvidia-smi ---"
nvidia-smi || echo "[WARN] nvidia-smi failed - check WSL GPU passthrough"
echo
echo "--- torch CUDA check ---"
python - <<'EOF'
import torch
print("torch        :", torch.__version__)
print("cuda built   :", torch.version.cuda)
print("cuda usable  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu          :", torch.cuda.get_device_name(0))
    print("vram (GB)    :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul test  : OK", y.shape)
EOF

echo
echo "================================================================"
echo "  ALL DONE."
echo "  Next: close this shell, reopen Ubuntu, then:"
echo "      source ~/venvs/ft/bin/activate"
echo "================================================================"
