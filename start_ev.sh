#!/usr/bin/env bash
# start_ev.sh — 一条命令把「模型服务 + 语音助手前端」拉起来
#
# 用法:
#   bash start_ev.sh                     # 用默认 adapter 起服务 + 起前端
#   bash start_ev.sh --adapter ~/ft_outputs/ev-v3/lora_adapter
#   bash start_ev.sh --no-llm            # 模型服务已经自己起好了，只起前端
#   bash start_ev.sh --stop              # 全停（模型服务 + 前端）
#
# 说明:
#   * 模型服务（serve_hf.py）在后台跑，日志写 logs/serve_hf.log
#   * 前端（app/server.py）在前台跑，Ctrl+C 只停前端；停全部用 --stop
#   * 两个服务分属不同 venv：模型服务用 ~/venvs/ft，前端用 ~/venvs/evapp

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FT_VENV="${FT_VENV:-$HOME/venvs/ft}"
EV_VENV="${EV_VENV:-$HOME/venvs/evapp}"
ADAPTER="${ADAPTER:-$HOME/ft_outputs/ev-v2/lora_adapter}"
LLM_PORT="${LLM_PORT:-8080}"
APP_PORT="${APP_PORT:-8090}"
LOGDIR="$ROOT/logs"
PIDFILE="$LOGDIR/serve_hf.pid"

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

STOP_ONLY=0
START_LLM=1
while [ $# -gt 0 ]; do
    case "$1" in
        --stop)     STOP_ONLY=1; shift ;;
        --no-llm)   START_LLM=0; shift ;;
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
    sleep 1
    if pgrep -f "serve_hf.py|app/server.py" >/dev/null 2>&1; then
        echo "[停止] 还有残留，强制杀"
        pkill -9 -f "serve_hf.py" 2>/dev/null || true
        pkill -9 -f "app/server.py" 2>/dev/null || true
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

# ---------------- 2) 前端 ----------------
export PORT="$APP_PORT"
export LLM_BASE="http://127.0.0.1:${LLM_PORT}/v1"
echo "[APP] 启动语音助手前端（端口 ${APP_PORT}）"
echo "      Ctrl+C 只停前端；要全停: bash start_ev.sh --stop"
echo
exec bash "$ROOT/run_app.sh"
