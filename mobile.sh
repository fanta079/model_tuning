#!/usr/bin/env bash
# mobile.sh — 让【手机浏览器】也能用 E.V.（关键是拿到 https）
#
# 为什么不能直接扫局域网 IP:
#   浏览器的麦克风只在"安全上下文"里给：https 或 localhost。
#   手机访问 http://192.168.x.x:8090 时 navigator.mediaDevices 直接是 undefined，
#   录音按钮会用不了 —— 打字还能聊，但那就不是语音助手了。
#
# 三条路，按省事程度排：
#   1) cloudflared 临时隧道  一条命令出 https，不用装证书，最适合"先试试"
#   2) tailscale serve       自己设备之间直连，稳定、不限速，适合长期用
#   3) 自签证书              最麻烦（手机要装 CA），不推荐
#
# 用法:
#   bash mobile.sh                 # 自动挑一条能用的
#   bash mobile.sh --lan           # 只打印局域网地址（明确告诉你麦克风不可用）
#   bash mobile.sh --port 8090     # 指定端口
#
# 前提: 语音助手已经在跑（另开终端 bash start_ev.sh）

set -euo pipefail

PORT="${PORT:-8090}"
MODE="auto"

while [ $# -gt 0 ]; do
    case "$1" in
        --lan)  MODE="lan"; shift ;;
        --port) PORT="$2"; shift 2 ;;
        -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || true
[ -n "${LAN_IP:-}" ] || LAN_IP="<本机局域网IP>"

echo "=============================================================="
echo "  E.V. 手机接入（端口 $PORT）"
echo "=============================================================="

# --- 0) 先确认服务在跑 ---
if curl -sf -m 3 "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
    echo "[检查] 语音助手在线：http://127.0.0.1:$PORT"
else
    echo "[检查] 端口 $PORT 上没服务 —— 先另开一个终端: bash start_ev.sh"
    echo "       （没起也能继续看方案，只是连不上）"
fi
echo

if [ "$MODE" = "lan" ]; then
    cat <<EOF
[局域网直连]
    http://$LAN_IP:$PORT

   ⚠️ 能打开、能打字聊天、回复也有声音，但**麦克风用不了**
      （非 https 不是安全上下文，浏览器直接不给 getUserMedia）。

   真要用嘴说，走下面两条 https 方案之一。自用的话 cloudflared 最省事。
EOF
    exit 0
fi

# --- 1) tailscale serve（优先：直连、稳定）---
if command -v tailscale >/dev/null 2>&1 && tailscale status >/dev/null 2>&1; then
    echo "[方案] tailscale serve（设备间直连，不经过第三方）"
    tailscale serve --bg "$PORT" 2>/dev/null || true
    URL="$(tailscale serve status 2>/dev/null | grep -o 'https://[^ ]*' | head -1 || true)"
    if [ -n "$URL" ]; then
        echo "  手机打开: ${URL%/}"
        echo "  （手机也装 tailscale 并登录同一账号）"
        exit 0
    fi
    echo "  [提示] tailscale serve 没拿到 https 地址，改试 cloudflared"
    echo
fi

# --- 2) cloudflared 临时隧道 ---
if command -v cloudflared >/dev/null 2>&1; then
    echo "[方案] cloudflared 临时隧道（给出一个 trycloudflare.com 的 https 地址）"
    echo "  注意: 流量会过 Cloudflare，自用没问题；Ctrl+C 即断。"
    echo "  等下面出现 https://xxx.trycloudflare.com 就把那个地址填到手机浏览器。"
    echo
    cloudflared tunnel --url "http://127.0.0.1:$PORT" 2>&1 | while IFS= read -r line; do
        # 只挑有用的行，别把整屏调试日志糊到用户脸上
        case "$line" in
            *trycloudflare.com*)
                echo
                echo "  ➜ 手机打开: $(echo "$line" | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | head -1)"
                echo
                ;;
            *ERR*|*error*|*failed*) echo "  $line" ;;
        esac
    done
    exit 0
fi

# --- 3) 都没有：给安装指引 ---
cat <<EOF
[没有可用的 https 工具]

  A. cloudflared（推荐，30 秒装好，免注册）
       wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
       sudo dpkg -i cloudflared-linux-amd64.deb
       bash mobile.sh

  B. tailscale（长期用更稳，设备间直连）
       curl -fsSL https://tailscale.com/install.sh | sh
       sudo tailscale up        # 手机也装 App 登录同一账号
       bash mobile.sh

  C. 只想先看看界面长什么样（没有麦克风）
       http://$LAN_IP:$PORT
     手机和 PC 要连同一个 WiFi；Windows 防火墙可能拦，放行 $PORT 即可。

  备注: 不想折腾 https，就直接做安卓包（WebView 壳），
        原生权限不受"安全上下文"限制，http 也能录音。
EOF
