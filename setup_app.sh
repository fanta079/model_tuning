#!/usr/bin/env bash
# setup_app.sh — E.V. 语音助手（P1）环境搭建
#
# 为什么要独立 venv:
#   ~/venvs/ft 是训练环境（torch/unsloth/trl…），动它会牵连训练。
#   这个 App 只需要 fastapi + 语音识别 + 语音合成，不需要 torch。
#   分开建环境，互不影响。
#
# 用法:  bash setup_app.sh

set -euo pipefail

VENV="${VENV:-$HOME/venvs/evapp}"
MIRROR="${MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"

step() { echo; echo "================================================================"; echo "  $1"; echo "================================================================"; }

step "0/4 前置检查"
command -v uv >/dev/null 2>&1 || { echo "[错误] 未找到 uv"; exit 1; }
echo "uv: $(uv --version)"
command -v nvidia-smi >/dev/null 2>&1 && \
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" || \
    echo "[警告] nvidia-smi 不可用，语音识别会退回 CPU（会慢）"

step "1/4 建立独立虚拟环境 (Python 3.12)"
if [ -d "$VENV" ]; then
    echo "$VENV 已存在，跳过创建"
else
    uv venv "$VENV" --python=3.12
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "python: $(python -V)"

step "2/4 安装依赖（阿里云源）"
uv pip install -i "$MIRROR" \
    fastapi "uvicorn[standard]" python-multipart httpx edge-tts faster-whisper \
    "av>=12,<14"

step "3/4 预下载语音识别模型（large-v3 约 3.1GB，走魔搭）"
# 为什么不再用 WhisperModel("large-v3") 直接触发下载:
#   hf-mirror 实测 ~0.4MB/s，3.1GB 得跑一两小时；魔搭同一个文件 ~11MB/s，5 分钟。
#   下到本地目录后按【路径】加载，顺带避免运行时又回 HF 偷偷拉一遍。
#   详见 docs/踩坑记录.md #19
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_MODEL="${WHISPER_MODEL:-large-v3}"
WHISPER_SOURCE="${WHISPER_SOURCE:-modelscope}"        # modelscope | hf
WHISPER_DIR="${WHISPER_DIR:-$ROOT/models/faster-whisper-$WHISPER_MODEL}"

# 魔搭上已验证存在的仓库名；其他模型规模走 HF 回退
case "$WHISPER_MODEL" in
    large-v3)       MS_REPO="Systran/faster-whisper-large-v3" ;;                 # 3.1GB
    large-v3-turbo) MS_REPO="pengzhendong/faster-whisper-large-v3-turbo" ;;      # 1.6GB，快 4 倍
    medium)         MS_REPO="Systran/faster-whisper-medium" ;;                   # 1.5GB
    *)              MS_REPO="" ;;
esac

mkdir -p "$WHISPER_DIR"
if [ "$WHISPER_SOURCE" = "modelscope" ] && [ -n "$MS_REPO" ]; then
    BASE="https://www.modelscope.cn/models/$MS_REPO/resolve/master"
    echo "源    : ModelScope  $MS_REPO"
    echo "目录  : $WHISPER_DIR"
    # -C - 断点续传：中断了就原样重跑本脚本，不会从头下
    for f in config.json preprocessor_config.json tokenizer.json vocabulary.json vocabulary.txt model.bin; do
        code=$(curl -sIL -m 15 -o /dev/null -w '%{http_code}' "$BASE/$f" || true)
        [ "$code" = "200" ] || continue
        echo "  ↓ $f"
        curl -L -C - --retry 5 --retry-delay 2 --connect-timeout 15 --progress-bar \
             -o "$WHISPER_DIR/$f" "$BASE/$f"
    done
    [ -s "$WHISPER_DIR/model.bin" ] || { echo "[错误] model.bin 没下全，重跑本脚本会续传"; exit 1; }
    python - <<EOF
from faster_whisper import WhisperModel
WhisperModel("$WHISPER_DIR", device="cpu", compute_type="int8")
print("就绪: $WHISPER_DIR")
EOF
else
    echo "源    : HuggingFace  $WHISPER_MODEL"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    # ⚠️ 必须关掉 Xet：
    #   hf-mirror 只代理 HF 主 API，不代理 Xet 的 CAS 服务（cas-server.xethub.hf.co），
    #   不关会报 401 Unauthorized —— 这是新版 huggingface_hub 的坑。
    export HF_HUB_DISABLE_XET=1
    python - <<EOF
import os
from faster_whisper import WhisperModel
m = os.environ.get("WHISPER_MODEL", "$WHISPER_MODEL")
print(f"下载/校验 {m} ...")
WhisperModel(m, device="cpu", compute_type="int8")   # 只为触发下载
print(f"{m} 就绪")
EOF
fi

step "4/4 自检"
python -c "
import edge_tts, fastapi, uvicorn, httpx
print('edge-tts :', edge_tts.__version__ if hasattr(edge_tts,'__version__') else 'ok')
print('fastapi  :', fastapi.__version__)
print('依赖全部就绪')
"

cat <<'TIP'

==============================================================
 环境就绪。接下来：

 0) 语音识别模型已落地（不用再等 HF 慢慢下）:
      models/faster-whisper-large-v3/
      run_app.sh 会自动按这个目录加载，别手动传模型名 "large-v3"
      （传名字会忽略本地目录、又回 HF 重下一遍）

 1) 先试听音色（选一个"冷"的女声给 E.V.）
      python app/try_voices.py
      # 听完把选中的 voice/rate/pitch 填进 app/voice.json

 2) 起 E.V. 模型服务（另开一个终端）
      ⛔ 别用 llama-server / GGUF：Qwen3.5 是混合注意力，转 GGUF 会输出乱码
         （详见 docs/踩坑记录.md 第 18 条，已验证 llama.cpp 这条路暂时走不通）

      source ~/venvs/ft/bin/activate        # 注意：模型服务在【训练环境】，不是 evapp
      python serve_hf.py --adapter /home/fanta/ft_outputs/ev-v2/lora_adapter
      # 接口 http://127.0.0.1:8080/v1 ，和 llama-server 完全兼容，P1 零改动

 3) 启动语音助手（回到 evapp 环境）
      bash run_app.sh
      浏览器打开 http://localhost:8090

 退出环境:  deactivate
==============================================================
TIP
