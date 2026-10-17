#!/usr/bin/env bash
# run_app.sh — 启动 E.V. 语音助手（P1）
#
# 用法:  bash run_app.sh
#
# 会先用 ~/venvs/evapp 环境起一个网页服务，然后你浏览器打开就能语音聊天。
# 模型服务（llama-server）需要你自己另开终端起 —— 这个脚本只负责检查它在不在。

set -euo pipefail

VENV="${VENV:-$HOME/venvs/evapp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8090}"
LLM_BASE="${LLM_BASE:-http://127.0.0.1:8080/v1}"

if [ ! -d "$VENV" ]; then
    echo "[错误] 没找到 $VENV —— 先跑:  bash setup_app.sh"
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "================================================================"
echo "  E.V. 语音助手"
echo "================================================================"

# ---- STT 模型: 优先用 setup_app.sh 下到本地的目录（避免运行时回 HF 重新下载）----
if [ -z "${WHISPER_MODEL:-}" ]; then
    for d in "$ROOT/models"/faster-whisper-*; do
        if [ -s "$d/model.bin" ]; then
            export WHISPER_MODEL="$d"
            break
        fi
    done
fi
if [ -n "${WHISPER_MODEL:-}" ] && [ -d "$WHISPER_MODEL" ]; then
    echo "[STT] 本地模型: $WHISPER_MODEL"
else
    echo "[STT] 模型名: ${WHISPER_MODEL:-large-v3}（首次会从 HF 下载，慢的话先跑 setup_app.sh）"
fi

# ---- STT 后端: ctranslate2 4.x 需要 CUDA 12 的 cublas/cudnn ----
# 你的 torch 是 cu13，soname 对不上（libcublas.so.13 ≠ .so.12），所以在 evapp 里单独装一份 cu12。
NV_LIBS="$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':' || true)"
if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
fi

if [ -z "${WHISPER_DEVICE:-}" ]; then
    if [ -n "$NV_LIBS" ]; then
        export WHISPER_DEVICE=cuda
        echo "[STT] 设备: cuda（已挂上 venv 内的 CUDA 12 运行库）"
    else
        export WHISPER_DEVICE=cpu
        export WHISPER_COMPUTE=int8
        echo "[STT] 设备: cpu —— 没找到 CUDA 12 运行库（libcublas.so.12），自动降级"
        echo "      想用 GPU 跑识别（快 3~5 倍），装一次就好（约 1GB）："
        echo "        pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 -i https://mirrors.aliyun.com/pypi/simple/"
    fi
fi
echo

# ---- 检查模型服务 ----
if curl -sf -m 4 "${LLM_BASE}/models" >/dev/null 2>&1; then
    echo "[LLM] 在线: $LLM_BASE"
else
    echo "[LLM] 连不上 $LLM_BASE"
    echo "      另开一个终端起模型服务："
    echo
    echo "        source ~/venvs/ft/bin/activate"
    echo "        cd $ROOT && python serve_hf.py --adapter ~/ft_outputs/ev-v2/lora_adapter"
    echo
    echo "      注意: 不要用 llama-server 跑自己导出的 GGUF ——"
    echo "            Qwen3.5 是混合注意力架构, GGUF 输出会是乱码（见 docs/踩坑记录.md 第六节）"
    echo
    echo "      （模型没起也能打开界面，但一发消息就会报错）"
fi
echo

# ---- 音色配置 ----
if [ -f "$ROOT/app/voice.json" ]; then
    echo "[TTS] 音色配置: $(python -c "
import json;d=json.load(open('$ROOT/app/voice.json',encoding='utf-8'))
print(d.get('voice','?'), 'rate='+str(d.get('rate','?')), 'pitch='+str(d.get('pitch','?')))" 2>/dev/null || echo '(读取失败)')"
    echo "      想换音色:  python app/try_voices.py  然后改 app/voice.json"
fi
echo

# ---- 提示访问方式 ----
IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "[WEB] 本机打开:     http://localhost:$PORT"
[ -n "$IP" ] && echo "[WEB] 局域网打开:   http://$IP:$PORT"
echo "      注意: 浏览器只在 localhost 或 HTTPS 下给麦克风权限。"
echo "            从别的设备访问 http://$IP:$PORT 会拿不到麦克风（P4 再做 HTTPS）。"
echo

PORT="$PORT" LLM_BASE="$LLM_BASE" exec python "$ROOT/app/server.py"
