#!/usr/bin/env bash
# setup_tts.sh — 语音合成环境（音色设计 + 本地 TTS 服务）
#
# 为什么要【第三个】venv:
#   ~/venvs/ft    训练环境，torch/unsloth 版本被钉死，动它就牵连训练
#   ~/venvs/evapp 语音助手，靠一份独立的 cu12 运行库跑 faster-whisper（LD_LIBRARY_PATH 拼好的）
#   qwen-tts 会拉它自己的 torch/transformers —— 装进上面任何一个都可能把对方搞挂。
#   所以单独建 ~/venvs/tts，用完还能整个删掉，不影响主链路。
#
# 两个模型，分工不同，别搞混:
#   1.7B-VoiceDesign  一次性：用文字造音色 -> ref.wav（4.3GB，要独占显存）
#   0.6B-Base         常驻：拿 ref.wav 克隆，把任意文字念出来（1.5GB，能和 serve_hf 并存）
#
# 用法:
#   bash setup_tts.sh                 # 建环境 + 下两个模型（合计 ~5.8GB）
#   bash setup_tts.sh --only base     # 只补 0.6B Base（已经造好音色之后）
#   bash setup_tts.sh --only design   # 只补 1.7B VoiceDesign
#   bash setup_tts.sh --no-model      # 只建环境，不下载权重
#   VENV=~/venvs/tts2 bash setup_tts.sh
#
# 跑完怎么用:
#   source ~/venvs/tts/bin/activate
#   python app/design_voice.py --dry-run       # 不用显卡，先看描述词
#   bash start_ev.sh --stop                    # 1.7B 要独占显存
#   python app/design_voice.py --only 3        # 造候选
#   python app/design_voice.py --lock 3        # 锁定音色
#   python app/tts_server.py --check           # 自检：念一句给你听
#   TTS_BASE=http://127.0.0.1:8092 bash start_ev.sh   # 接进语音助手

set -euo pipefail

VENV="${VENV:-$HOME/venvs/tts}"
MIRROR="${MIRROR:-https://mirrors.aliyun.com/pypi/simple/}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_MODEL=1
WHICH="design base"

while [ $# -gt 0 ]; do
    case "$1" in
        --no-model) WITH_MODEL=0; shift ;;
        --only)     WHICH="${2:?--only 需要 design 或 base}"; shift 2 ;;
        -h|--help)  sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)          echo "未知参数: $1"; exit 1 ;;
    esac
done

step() { echo; echo "================================================================"; echo "  $1"; echo "================================================================"; }

step "0/4 前置检查"
command -v uv >/dev/null 2>&1 || { echo "[错误] 未找到 uv（跑一次 setup-ml-env-uv.sh 会装上）"; exit 1; }
echo "uv: $(uv --version)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)"
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${USED:-0}" -gt 1000 ]; then
        echo "[提醒] 显存已占用 ${USED}MiB。1.7B VoiceDesign 要 ~4.5GB，跑之前先:"
        echo "         bash start_ev.sh --stop"
    fi
else
    echo "[警告] 没有 nvidia-smi：只能用 --device cpu，1.7B 在 CPU 上很慢（仅验证用）"
fi

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
# 注意：不要装 flash-attn。
#   它要从源码编译（几十分钟、吃 10GB+ 内存），8GB 卡上收益极小。
#   默认 sdpa 完全够用，design_voice.py 的 --flash 是给"已经装过"的人准备的。
uv pip install -i "$MIRROR" qwen-tts soundfile modelscope

step "3/4 自检（torch / CUDA / 解码器）"
python - <<'EOF'
import importlib, sys
print("python  :", sys.version.split()[0])
import torch
print("torch   :", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
import soundfile
print("soundfile:", soundfile.__version__)
missing = [m for m in ("qwen_tts",) if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("[错误] 缺包: " + ", ".join(missing))
import qwen_tts
print("qwen_tts: ok", getattr(qwen_tts, "__version__", ""))
EOF

if [ "$WITH_MODEL" = 1 ]; then
    mkdir -p "$ROOT/models"
    step "4/4 下载权重（走魔搭）"

    download_one() {
        local name="$1" size="$2" purpose="$3"
        local dir="$ROOT/models/$name"
        echo
        echo "── $name  ($size)  $purpose"
        if [ -s "$dir/model.safetensors" ] || [ -s "$dir/model-00001-of-00001.safetensors" ]; then
            echo "   已存在，跳过：$dir"
            return 0
        fi
        # 为什么不用 from_pretrained 自动下:
        #   HF 直连不通；走 hf-mirror 实测 ~0.4MB/s，几个 GB 要跑几小时。魔搭快一个数量级。
        python - <<EOF
from modelscope import snapshot_download
p = snapshot_download("Qwen/$name", local_dir=r"$dir")
print("   下载完成:", p)
EOF
    }

    for one in $WHICH; do
        case "$one" in
            design) download_one "Qwen3-TTS-12Hz-1.7B-VoiceDesign" "约 4.3GB" "造音色用（一次性，要独占显存）" ;;
            base)   download_one "Qwen3-TTS-12Hz-0.6B-Base"          "约 1.5GB" "说话用（常驻，和 serve_hf 并存）" ;;
            *)      echo "[警告] --only 不认识: $one（可选 design / base）" ;;
        esac
    done

    echo
    echo "design_voice.py / tts_server.py 都会自动认 models/ 下的目录，不用传 --model"
else
    step "4/4 跳过权重下载（--no-model）"
fi

cat <<'TIP'

==============================================================
 环境就绪。两件事，顺序别搞反：

 A. 造音色（一次性，1.7B，要独占显存）
      source ~/venvs/tts/bin/activate
      python app/design_voice.py --dry-run     # 先看描述词，不用显卡
      bash start_ev.sh --stop                  # 腾显存，忘了就 OOM
      python app/design_voice.py --only 3
      # 听: app/voice_samples/design/index.html
      python app/design_voice.py --lock 3      # 选定 -> app/voice_lock/ref.wav

 B. 让她说话（常驻，0.6B，可以和模型服务并存）
      python app/tts_server.py --check         # 先自检：念一句存成 check.wav
      python app/tts_server.py                 # 起服务 :8092
      # 另开终端:
      TTS_BASE=http://127.0.0.1:8092 bash start_ev.sh

 换音色: 重跑 design_voice.py --lock N -> 重启 tts_server.py（server.py 不用动）
==============================================================
TIP
