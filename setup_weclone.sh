#!/usr/bin/env bash
# setup_weclone.sh — WeClone 独立环境一键搭建（WSL 内运行）
#
# ⚠️ 为什么必须独立环境：
#   WeClone 依赖 torch==2.7.1+cu126 / transformers==4.53.2 / llamafactory==0.9.4 / vllm==0.10.0,
#   与你的 ~/venvs/ft (unsloth + torch 2.14.1+cu130) **直接冲突**。
#   本脚本在 WeClone 项目内建自己的 .venv, 绝不碰你现有环境。
#
# 用法:
#   bash setup_weclone.sh                                   # 默认装到 ~/WeClone
#   DIR=/mnt/e/VMOS_workspace/WeClone bash setup_weclone.sh  # 指定目录
#
# 必须用 bash 运行（dash 不支持 pipefail）

set -euo pipefail

DIR="${DIR:-$HOME/WeClone}"
MIRROR="${MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple/}"

step() { echo; echo "================================================================"; echo "  $1"; echo "================================================================"; }

step "0/6 前置检查"
if ! command -v uv >/dev/null 2>&1; then
    echo "[错误] 未找到 uv。请先安装：pipx install uv   （并把 ~/.local/bin 加进 PATH）"
    exit 1
fi
echo "uv: $(uv --version)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
else
    echo "[警告] nvidia-smi 不可用 —— WSL 的 GPU 直通可能没配好"
fi

step "1/6 克隆 / 更新 WeClone"
if [ -d "$DIR/.git" ]; then
    git -C "$DIR" pull --ff-only
else
    git clone https://github.com/xming521/WeClone.git "$DIR"
fi
cd "$DIR"
echo "项目目录: $DIR"

step "2/6 创建独立虚拟环境 (Python 3.12)"
# WeClone 的 pyproject 限定 requires-python >=3.12,<3.13
if [ -d .venv ]; then
    echo ".venv 已存在，跳过"
else
    uv venv .venv --python=3.12
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "python: $(python -V)"

step "3/6 安装主依赖（清华镜像，较慢请耐心）"
uv pip install --group main -e . -i "$MIRROR"

step "4/6 安装中文脱敏模型（Presidio 需要）"
uv pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl

step "5/6 生成配置文件"
if [ -f settings.jsonc ]; then
    echo "settings.jsonc 已存在，保留不动"
else
    cp settings.template.jsonc settings.jsonc
    echo "已从模板生成 settings.jsonc"
fi

step "6/6 CUDA 自检"
python - <<'EOF'
import torch
print("torch       :", torch.__version__)
print("cuda built  :", torch.version.cuda)
print("cuda usable :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu         :", p.name, f"{p.total_memory/1024**3:.1f} GB")
    x = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print("matmul test : OK", (x @ x).shape)
EOF

cat <<'TIP'

==============================================================
 环境就绪。接下来：

 1) 改 settings.jsonc 适配 8GB 显存（详见 docs/WeClone-数字分身落地.md 第 3 节）
      common_args.finetuning_type            : "lora"  -> "qlora"
      train_sft_args.per_device_train_batch_size : 2   -> 1
      train_sft_args.gradient_accumulation_steps : 16  -> 32
      train_sft_args.cutoff_len              : 2048    -> 1024
      删除 train_sft_args.flash_attn 那一行

 2) 下模型（国内走魔搭）
      pip install modelscope
      modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models/Qwen2.5-7B-Instruct

 3) 放聊天数据，然后跑数据管线
      weclone-cli make-dataset

 4) 训练
      weclone-cli train-sft

 退出环境:  deactivate
 重新进入:  cd DIR && source .venv/bin/activate
==============================================================
TIP
