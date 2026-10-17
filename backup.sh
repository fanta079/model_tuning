#!/usr/bin/env bash
# backup.sh — 把一个版本的「数据 + 配置 + 评测 + adapter」打成一份归档，方便传到远程
#
# 用法:
#   bash backup.sh                                  # 自动找最新的 ev-v* adapter
#   bash backup.sh --adapter ~/ft_outputs/ev-v4/lora_adapter
#   bash backup.sh --no-adapter                     # 只备数据和配置（几百 KB）
#
# 产出: artifacts/ev-v4_20261017-1530.tar.gz
#
# 说明:
#   * 不含 checkpoint-*（每个 157MB，训完就该删，不值得备份）
#   * 不含 voice_samples / logs / models（都能重新生成）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/artifacts"
ADAPTER=""
WITH_ADAPTER=1

while [ $# -gt 0 ]; do
    case "$1" in
        --adapter)    ADAPTER="${2:?}"; shift 2 ;;
        --no-adapter) WITH_ADAPTER=0; shift ;;
        -h|--help)    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "未知参数: $1"; exit 1 ;;
    esac
done

# 没指定就找最新的 ev-v*
if [ "$WITH_ADAPTER" = 1 ] && [ -z "$ADAPTER" ]; then
    ADAPTER="$(ls -dt "$HOME"/ft_outputs/ev-v*/lora_adapter 2>/dev/null | head -1 || true)"
    if [ -z "$ADAPTER" ]; then
        echo "[警告] 没找到 ~/ft_outputs/ev-v*/lora_adapter，改用 --no-adapter 只备数据"
        WITH_ADAPTER=0
    fi
fi

if [ -n "${ADAPTER:-}" ] && [ -d "$ADAPTER" ]; then
    VERSION="$(basename "$(dirname "$ADAPTER")")"      # .../ev-v4/lora_adapter -> ev-v4
else
    VERSION="ev-local"
fi
STAMP="$(date +%Y%m%d-%H%M)"
NAME="${VERSION}_${STAMP}"
mkdir -p "$OUT"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PKG="$TMP/$NAME"
mkdir -p "$PKG"

# ---- 1) 数据（只带 json，体积小、可复现）----
mkdir -p "$PKG/my_data"
find "$ROOT/my_data" -maxdepth 1 -name '*.json' -exec cp {} "$PKG/my_data/" \; 2>/dev/null || true

# ---- 2) 人设与配置 ----
mkdir -p "$PKG/personas" "$PKG/app"
cp "$ROOT"/personas/*.json "$PKG/personas/" 2>/dev/null || true
cp "$ROOT"/personas/*.md   "$PKG/personas/" 2>/dev/null || true
cp "$ROOT/app/system.txt"  "$PKG/app/" 2>/dev/null || true
cp "$ROOT/app/voice.json"  "$PKG/app/" 2>/dev/null || true
cp "$ROOT"/eval_result_*.md "$PKG/" 2>/dev/null || true

# ---- 3) adapter（只要最终产物，不要 checkpoint）----
if [ "$WITH_ADAPTER" = 1 ] && [ -d "$ADAPTER" ]; then
    mkdir -p "$PKG/lora_adapter"
    rsync -a --exclude 'checkpoint-*' "$ADAPTER"/ "$PKG/lora_adapter"/ 2>/dev/null \
        || cp -r "$ADAPTER"/. "$PKG/lora_adapter/"
fi

# ---- 4) 版本说明 ----
{
    echo "版本: $VERSION"
    echo "时间: $(date '+%Y-%m-%d %H:%M')"
    echo "adapter: ${ADAPTER:-（未包含）}"
    echo
    echo "数据："
    for f in "$PKG"/my_data/*.json; do
        [ -e "$f" ] || continue
        printf '  %-26s %s 条\n' "$(basename "$f")" "$(python -c "
import json,sys
try: print(len(json.load(open(sys.argv[1],encoding='utf-8'))))
except Exception: print('?')
" "$f" 2>/dev/null || echo '?')"
    done
    echo
    echo "评测：$(ls "$PKG"/eval_result_*.md 2>/dev/null | wc -l) 份报告"
} > "$PKG/版本说明.txt"

tar -czf "$OUT/$NAME.tar.gz" -C "$TMP" "$NAME"

echo "[打包] $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
echo
echo "传到远程（任选一种）:"
echo "  · Gitee 私有仓库（数据和配置，几 MB）"
echo "    git remote add gitee git@gitee.com:<你>/<仓库>.git && git push gitee master"
echo "  · ModelScope 私有模型（adapter，85MB）"
echo "    pip install modelscope && modelscope upload <你>/ev-$VERSION <解包后的>/$NAME/lora_adapter"
