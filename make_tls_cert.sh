#!/usr/bin/env bash
# ============================================================================
# make_tls_cert.sh —— 生成自签证书，让手机也能用麦克风
#
# 为什么需要
#   浏览器的 getUserMedia（麦克风）只在「安全上下文」可用：localhost 或 https。
#   手机通过局域网 IP 访问时是明文 http，麦克风直接被禁用 —— 跟你设备无关，
#   是浏览器规定。想用语音就必须 https。
#
# 用法
#   bash make_tls_cert.sh                       # 自动带 WSL 的 IP
#   LAN_IP=192.168.3.205 bash make_tls_cert.sh  # 再加一个地址（Windows 的局域网 IP）
#   bash make_tls_cert.sh certs                 # 指定输出目录
#
# 之后启动（run_app.sh / start_ev.sh 都会继承环境变量）
#   TLS_CERT=certs/server.crt TLS_KEY=certs/server.key bash start_ev.sh --tts
#   → 打开 https://<IP>:8090
#
# 手机要装一次 CA（否则只显示证书错误，且仍算不安全上下文）
#   Android：设置 → 安全 → 加密与凭据 → 安装证书 → CA 证书 → 选 certs/ca.crt
#   iOS    ：AirDrop/邮件打开 ca.crt → 安装描述文件 →
#            设置 → 通用 → 关于本机 → 证书信任设置 → 打开完全信任
# ============================================================================
set -euo pipefail

DIR="${1:-certs}"
LAN_IP="${LAN_IP:-}"
mkdir -p "$DIR"
cd "$DIR"

WSL_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -n "$WSL_IP" ] || WSL_IP="127.0.0.1"

SAN="DNS:localhost,IP:127.0.0.1,IP:${WSL_IP}"
[ -n "$LAN_IP" ] && SAN="${SAN},IP:${LAN_IP}"

echo "[1/3] 建根 CA（10 年）"
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
  -keyout ca.key -out ca.crt \
  -subj "/CN=E.V. Local CA" 2>/dev/null

echo "[2/3] 建服务器证书（825 天，含 SAN: ${SAN}）"
openssl req -newkey rsa:2048 -nodes \
  -keyout server.key -out server.csr \
  -subj "/CN=${WSL_IP}" 2>/dev/null

cat > san.cnf <<EOF
subjectAltName=${SAN}
extendedKeyUsage=serverAuth
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
  -days 825 -out server.crt -extfile san.cnf 2>/dev/null
rm -f server.csr san.cnf ca.srl

echo "[3/3] 完成"
echo
echo "  证书: $(pwd)/server.crt"
echo "  私钥: $(pwd)/server.key"
echo "  CA  : $(pwd)/ca.crt   ← 这个要装到手机上"
echo
echo "启动："
echo "  TLS_CERT=$(pwd)/server.crt TLS_KEY=$(pwd)/server.key bash start_ev.sh --tts"
echo "然后手机打开：https://${LAN_IP:-$WSL_IP}:8090"
echo
ls -l ca.crt server.crt server.key
