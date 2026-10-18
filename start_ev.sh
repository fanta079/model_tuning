#!/usr/bin/env bash
# start_ev.sh — 一条命令把「模型服务 + 语音助手前端」拉起来
#
# 用法:
#   bash start_ev.sh                     # 用默认 adapter 起服务 + 起前端
#   bash start_ev.sh --adapter ~/ft_outputs/ev-v3/lora_adapter
#   bash start_ev.sh --no-llm            # 模型服务已经自己起好了，只起前端
#   bash start_ev.sh --tts               # 顺带起本地 TTS（:8092），换掉 edge-tts
#   bash start_ev.sh --stop              # 全停（模型服务 + TTS + 前端）
#
# 说明:
#   * 模型服务（serve_hf.py）在后台跑，日志写 logs/serve_hf.log
#   * 本地 TTS（app/tts_server.py）在后台跑，日志写 logs/tts_server.log
#   * 前端（app/server.py）在前台跑，Ctrl+C 只停前端；停全部用 --stop
#   * 三个服务分属不同 venv：ft / tts / evapp，各自用绝对路径的 python
#   * 显存提醒：LLM 3.4 + TTS 2.0 + whisper 1.6 ≈ 7GB，8GB 卡会紧。
#     --tts 时 whisper 仍留 GPU（int8_float16），CPU 识别实测 17.5 秒不可用

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FT_VENV="${FT_VENV:-$HOME/venvs/ft}"
EV_VENV="${EV_VENV:-$HOME/venvs/evapp}"
TTS_VENV="${TTS_VENV:-$HOME/venvs/tts}"
ADAPTER="${ADAPTER:-$HOME/ft_outputs/ev-v2/lora_adapter}"
LLM_PORT="${LLM_PORT:-8080}"
APP_PORT="${APP_PORT:-8090}"
TTS_PORT="${TTS_PORT:-8092}"
LOGDIR="$ROOT/logs"
PIDFILE="$LOGDIR/serve_hf.pid"
TTS_PIDFILE="$LOGDIR/tts_server.pid"

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

STOP_ONLY=0
START_LLM=1
START_TTS=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stop)     STOP_ONLY=1; shift ;;
        --no-llm)   START_LLM=0; shift ;;
        --tts)      START_TTS=1; shift ;;
        --adapter)  ADAPTER="${2:?--adapter 需要路径}"; shift 2 ;;
        --port)     APP_PORT="${2:?--port 需要端口}"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *)          echo "未知参数: $1"; echo; usage; exit 1 ;;
    esac
done

# ---------------- 停止 ----------------
if [ "$STOP_ONLY" = 1 ]; then
    echo "[停止] 正在关掉模型服务和前端 ……"
    if [ -f "$PIDFILE" ]; then
        kill "$(cat "$PIDFILE")" 2>/dev/null || true
        rm -f "$PIDFILE"
    fi
    pkill -f "serve_hf.py" 2>/dev/null || true
    pkill -f "app/server.py" 2>/dev/null || true
    if [ -f "$TTS_PIDFILE" ]; then
        kill "$(cat "$TTS_PIDFILE")" 2>/dev/null || true
        rm -f "$TTS_PIDFILE"
    fi
    pkill -f "app/tts_server.py" 2>/dev/null || true
    sleep 1
    if pgrep -f "serve_hf.py|app/server.py|app/tts_server.py" >/dev/null 2>&1; then
        echo "[停止] 还有残留，强制杀"
        pkill -9 -f "serve_hf.py" 2>/dev/null || true
        pkill -9 -f "app/server.py" 2>/dev/null || true
        pkill -9 -f "app/tts_server.py" 2>/dev/null || true
    fi
    echo "[停止] 完成（显存已释放）"
    exit 0
fi

mkdir -p "$LOGDIR"

# ---------------- 1) 模型服务 ----------------
if [ "$START_LLM" = 1 ]; then
    if curl -sf -m 2 "http://127.0.0.1:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[LLM] 端口 ${LLM_PORT} 已有服务，跳过启动"
    else
        if [ ! -x "$FT_VENV/bin/python" ]; then
            echo "[LLM] 找不到 $FT_VENV/bin/python"
            echo "      用 --no-llm 自己起模型服务，或改 FT_VENV=/path/to/venv"
            exit 1
        fi
        if [ ! -d "$ADAPTER" ]; then
            echo "[LLM] adapter 不存在: $ADAPTER"
            echo "      检查路径，或用 --adapter 指定别的版本"
            exit 1
        fi
        echo "[LLM] 启动 serve_hf.py"
        echo "      adapter : $ADAPTER"
        echo "      日志    : $LOGDIR/serve_hf.log"
        nohup "$FT_VENV/bin/python" "$ROOT/serve_hf.py" \
            --adapter "$ADAPTER" --port "$LLM_PORT" \
            > "$LOGDIR/serve_hf.log" 2>&1 &
        echo $! > "$PIDFILE"

        printf "[LLM] 等待就绪（首次加载模型 30~60 秒）"
        ok=0
        for i in $(seq 1 120); do
            if curl -sf -m 2 "http://127.0.0.1:${LLM_PORT}/v1/models" >/dev/null 2>&1; then
                ok=1; echo " ✅"; break
            fi
            printf "."
            sleep 1
        done
        if [ "$ok" = 0 ]; then
            echo " ❌ 超时"
            echo "      看日志: tail -50 $LOGDIR/serve_hf.log"
            exit 1
        fi
    fi
fi

# ---------------- 1.5) 本地 TTS（可选，--tts） ----------------
if [ "$START_TTS" = 1 ]; then
    if [ -n "${TTS_BASE:-}" ]; then
        echo "[TTS] 外部已指定 TTS_BASE=$TTS_BASE，跳过本地启动"
    elif curl -sf -m 2 "http://127.0.0.1:${TTS_PORT}/health" >/dev/null 2>&1; then
        echo "[TTS] 端口 ${TTS_PORT} 已有服务，跳过启动"
        export TTS_BASE="http://127.0.0.1:${TTS_PORT}"
    else
        if [ ! -x "$TTS_VENV/bin/python" ]; then
            echo "[TTS] 找不到 $TTS_VENV/bin/python"
            echo "      先跑 bash setup_tts.sh --only base，或改 TTS_VENV=/path/to/venv"
            exit 1
        fi

        # 权重放哪影响的是**加载**（9P 慢且会短读，whisper 栽过），不影响单次合成速度。
        TTS_MODEL_NAME="Qwen3-TTS-12Hz-0.6B-Base"
        FAST_MODEL="$HOME/models/$TTS_MODEL_NAME"
        SLOW_MODEL="$ROOT/models/$TTS_MODEL_NAME"
        if [ ! -d "$FAST_MODEL" ] && [ -d "$SLOW_MODEL" ]; then
            if [ "${EV_AUTO_EXT4:-0}" = 1 ]; then
                echo "[TTS] 把权重从 9P 拷到 ext4（1.8GB，几十秒）……"
                mkdir -p "$HOME/models"
                cp -r "$SLOW_MODEL" "$FAST_MODEL"
            else
                echo "[TTS] 权重在 9P 上（$SLOW_MODEL）—— 加载慢，建议拷到 ext4"
                echo "      建议先拷到 ext4（只需一次）："
                echo "        cp -r \"$SLOW_MODEL\" \"$FAST_MODEL\""
                echo "      或下次直接：EV_AUTO_EXT4=1 bash start_ev.sh --tts"
            fi
        fi

        echo "[TTS] 启动 app/tts_server.py（端口 ${TTS_PORT}）"
        echo "      日志    : $LOGDIR/tts_server.log"
        nohup "$TTS_VENV/bin/python" "$ROOT/app/tts_server.py" \
            --port "$TTS_PORT" > "$LOGDIR/tts_server.log" 2>&1 &
        echo $! > "$TTS_PIDFILE"

        printf "[TTS] 等待就绪（加载 0.6B + 建 clone prompt，约 30~60 秒）"
        ok=0
        for i in $(seq 1 180); do
            if curl -sf -m 2 "http://127.0.0.1:${TTS_PORT}/health" 2>/dev/null \
                    | grep -q '"ok": *true'; then
                ok=1; echo " ✅"; break
            fi
            printf "."
            sleep 1
        done
        if [ "$ok" = 0 ]; then
            echo " ❌ 超时"
            echo "      看日志: tail -50 $LOGDIR/tts_server.log"
            echo "      先单独验证音色: $TTS_VENV/bin/python app/tts_server.py --check"
            exit 1
        fi
        export TTS_BASE="http://127.0.0.1:${TTS_PORT}"
    fi
    # 显存：三个模型挤 8GB。实测把 STT 挪 CPU 会让识别慢到 17.5 秒（不可用），
    # 所以默认仍走 GPU，用 int8_float16 把 whisper 压到 ~1.6GB：
    #   LLM 3.4 + TTS ~1.5 + whisper 1.6 ≈ 6.5GB，刚好够。
    # 真 OOM 了 server.py 自己会退回 CPU（get_whisper 里的三层降级），
    # 但那时你会明显感到识别变慢 —— 那种情况下更要考虑换 small 模型。
    if [ -z "${WHISPER_DEVICE:-}" ]; then
        export WHISPER_DEVICE=cuda
        export WHISPER_COMPUTE="${WHISPER_COMPUTE:-int8_float16}"
        echo "[TTS] whisper 保持 GPU（${WHISPER_DEVICE}/${WHISPER_COMPUTE}），CPU 识别太慢不可用"
        echo "      若 OOM：WHISPER_DEVICE=cpu 或换 small 模型"
    fi
    echo "[TTS] 引擎: 本地 Qwen3-TTS 克隆 → ${TTS_BASE}"
fi

# ---------------- 2) 前端 ----------------
# whisper 优先用 ext4 家目录那份：/mnt/e 上读 3GB 级文件会偶发 9P 短读
# （报 "File model.bin is incomplete: failed to read a buffer ... at position 1.2GB"）
# 拷过去不只是修 bug，加载还快 5~9 倍。
if [ -z "${WHISPER_MODEL:-}" ]; then
    for d in "$HOME/models/faster-whisper-large-v3" "$HOME/models/faster-whisper-medium"; do
        if [ -s "$d/model.bin" ]; then
            export WHISPER_MODEL="$d"
            echo "[STT] 用 ext4 上的模型: $d"
            break
        fi
    done
fi

export PORT="$APP_PORT"
export LLM_BASE="http://127.0.0.1:${LLM_PORT}/v1"
echo "[APP] 启动语音助手前端（端口 ${APP_PORT}）"
[ -n "${TTS_BASE:-}" ] && echo "      TTS: ${TTS_BASE}"
echo "      Ctrl+C 只停前端；要全停: bash start_ev.sh --stop"
echo
exec bash "$ROOT/run_app.sh"
