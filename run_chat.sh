#!/usr/bin/env bash
# Qwen3.5-9B GGUF 本地聊天服务 (WSL + RTX 4060 Ti 8GB)
# 用法:  bash run_chat.sh          # 用下方默认参数启动
#        CTX=32768 bash run_chat.sh  # 自定义上下文长度
set -euo pipefail

# ---- 路径（模型在 Windows 盘，WSL 通过 /mnt 访问）----------------
WIN_DIR="/mnt/e/VMOS_workspace/model_tuning/models/Qwen3.5-9B-GGUF"
MODEL="${MODEL:-$WIN_DIR/Qwen3.5-9B-UD-Q4_K_XL.gguf}"

# ---- llama.cpp 可执行文件 --------------------------------------
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"

# ---- 运行参数（8GB 显存调优）------------------------------------
CTX="${CTX:-16384}"          # 上下文长度: 8GB 建议 8192~16384
NGL="${NGL:-99}"             # 放到 GPU 的层数, 爆显存就往下调(如 32)
PORT="${PORT:-8080}"
ENABLE_VISION="${ENABLE_VISION:-0}"   # =1 时启用视觉(多占约 900MB 显存)

ARGS=(
  -m "$MODEL"
  -ngl "$NGL"
  -c "$CTX"
  --host 0.0.0.0
  --port "$PORT"
  --jinja                       # 使用模型自带 chat template
  --flash-attn on
)

if [[ "$ENABLE_VISION" == "1" ]]; then
  ARGS+=( --mmproj "$WIN_DIR/mmproj-F16.gguf" )
fi

if [[ ! -f "$MODEL" ]]; then
  echo "[错误] 找不到模型文件: $MODEL" >&2
  echo "       请确认已下载, 或用 MODEL=/path/to.gguf bash run_chat.sh 指定" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "[错误] 找不到 llama-server: $LLAMA_SERVER" >&2
  echo "       先编译: git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp" >&2
  echo "       cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j" >&2
  exit 1
fi

echo "启动: $MODEL"
echo "上下文: $CTX  |  GPU层: $NGL  |  视觉: $ENABLE_VISION"
echo "浏览器打开: http://localhost:$PORT   (OpenAI 兼容: /v1)"
exec "$LLAMA_SERVER" "${ARGS[@]}"
