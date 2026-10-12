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
# 注意: spaCy 模型只发布在 GitHub Releases。
#   PyPI 上的同名包 zh-core-web-sm 是「防依赖混淆」的占位包(版本 999.9.9)，装了没用！
#   国内直连 GitHub Releases 常超时 -> 这里依次尝试加速代理，失败也只告警不中断。
WHL="zh_core_web_sm-3.8.0-py3-none-any.whl"
GH="explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/${WHL}"
TMP_WHL="${TMPDIR:-/tmp}/${WHL}"
URLS=(
    "https://ghfast.top/https://github.com/${GH}"
    "https://ghproxy.net/https://github.com/${GH}"
    "https://hk.gh-proxy.com/https://github.com/${GH}"
    "https://github.com/${GH}"
)

spacy_ok=0
if python -c "import spacy; spacy.load('zh_core_web_sm')" >/dev/null 2>&1; then
    echo "zh_core_web_sm 已安装，跳过。"
    spacy_ok=1
else
    for u in "${URLS[@]}"; do
        echo "尝试下载源: $u"
        if curl -fL --retry 3 --retry-delay 2 -m 300 -o "$TMP_WHL" "$u" 2>/dev/null; then
            if uv pip install "$TMP_WHL" >/dev/null 2>&1; then
                spacy_ok=1
                break
            fi
        fi
    done
fi

if [ "$spacy_ok" = 1 ]; then
    python - <<'EOF'
import spacy
nlp = spacy.load("zh_core_web_sm")
print("spaCy 版本 :", spacy.__version__)
print("中文模型   : zh_core_web_sm 可用")
EOF
else
    echo
    echo "[警告] 中文脱敏模型没装上（网络问题，不影响后续步骤继续跑）。"
    echo "       影响: WeClone 的 Presidio 中文 PII 识别会不可用。"
    echo "       兜底: 在 settings.jsonc 的 blocked_words 里手动列敏感词。"
    echo "       重试: 见 docs/WeClone-数字分身落地.md 第 7 节坑清单。"
fi

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
