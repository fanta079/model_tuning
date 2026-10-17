#!/usr/bin/env bash
# make_data_repo.sh — 把「训练数据 + 评测报告」导出成一个可独立 git 的数据仓库
#
# 用法:
#   bash make_data_repo.sh                 # 导出到 ../ev-data（默认）
#   bash make_data_repo.sh --out ~/repos/ev-data
#
# 产出结构:
#   ev-data/
#   ├── data/       所有 my_data/*.json（原始 + 合并 + 清洗后）
#   ├── reports/    eval_result_*.md
#   ├── README.md   自动生成：每个文件多少条、什么时候生成、对应哪个代码版本
#   └── .gitignore
#
# 说明:
#   数据仓库**只放数据**。人设卡/场景库属于代码仓库，不在这里复制一份，
#   否则两份会分叉 —— 用 README 里的 commit sha 指回代码仓库即可。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/../ev-data"

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="${2:?--out 需要路径}"; shift 2 ;;
        -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

PY="${PYTHON:-python}"
CODE_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '未提交')"
CODE_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"

mkdir -p "$OUT/data" "$OUT/reports"

echo "[导出] 数据 -> $OUT/data"
# 只导「真正的训练数据」，跳过单条测试样本 / smoke 用例 / 中间清洗产物
for f in "$ROOT"/my_data/*.json; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"
    case "$b" in
        example_*|haibara_smoke*|haibara_v3.json|all.json|train.json|val.json|dropped.json) continue ;;
    esac
    [ "$(wc -c <"$f")" -lt 1024 ] && continue          # 太小 = 测试残留
    cp "$f" "$OUT/data/"
done

echo "[导出] 报告 -> $OUT/reports"
cp "$ROOT"/eval_result_*.md "$OUT/reports/" 2>/dev/null || true

# ---- 自动生成 README（provenance）----
{
    echo "# E.V. 训练数据仓库"
    echo
    echo "> 只放数据与评测报告。人设卡、场景库、生成脚本在 **代码仓库**。"
    echo
    echo "## 对应代码版本"
    echo
    echo "- commit: \`$CODE_SHA\`（分支 \`$CODE_BRANCH\`）"
    echo "- 导出时间: $(date '+%Y-%m-%d %H:%M')"
    echo
    echo "## 数据文件"
    echo
    echo "| 文件 | 条数 | 大小 | 说明 |"
    echo "|---|---|---|---|"
    "$PY" - "$OUT/data" <<'PY' 2>/dev/null || true
import json, os, sys
d = sys.argv[1]
note = {
    'ev2_all.json': 'v2 训练集（闲聊 800）',
    'ev3_all.json': 'v3 = v2 + 自我介绍专项 80',
    'intro_api.json': '自我介绍专项包生成结果',
    'help_api.json': '助手能力专项包生成结果',
    'ev4_all.json': 'v4 = v3 + 助手 200',
    'ev_manual_40.json': '手写精品 40 条（few-shot 锚点）',
}
for f in sorted(os.listdir(d)):
    if not f.endswith('.json'):
        continue
    p = os.path.join(d, f)
    try:
        n = len(json.load(open(p, encoding='utf-8')))
    except Exception:
        n = '?'
    kb = os.path.getsize(p) / 1024
    print(f"| `{f}` | {n} | {kb:.0f} KB | {note.get(f, '')} |")
PY
    echo
    echo "## 评测报告"
    echo
    echo "| 文件 | 说明 |"
    echo "|---|---|"
    for f in "$OUT"/reports/*.md; do
        [ -e "$f" ] || continue
        echo "| \`$(basename "$f")\` | $(basename "$f" .md) 的 25 题人设评测 |"
    done
    echo
    echo "## 版本对应关系"
    echo
    echo "| 版本 | 数据 | 权重 |"
    echo "|---|---|---|"
    echo "| ev-v3 | \`ev3_all.json\` | ModelScope: \`<你的账号>/ev-v3\` |"
    echo "| ev-v4 | \`ev4_all.json\` | ModelScope: \`<你的账号>/ev-v4\` |"
    echo
    echo "> 三个仓库用**同一个 tag** 绑定，例如都打 \`v4\`。"
} > "$OUT/README.md"

cat > "$OUT/.gitignore" <<'EOF'
# 数据仓库里，json 要入库 —— 只忽略临时文件
*.tmp
*.tar.gz
.DS_Store
__pycache__/
EOF

echo
echo "[完成] $OUT"
echo "       数据 $(ls "$OUT"/data/*.json 2>/dev/null | wc -l) 个文件"
echo "       报告 $(ls "$OUT"/reports/*.md 2>/dev/null | wc -l) 份"
echo
echo "下一步:"
echo "  cd $OUT && git init && git add -A && git commit -m 'data: v4'"
echo "  git remote add origin git@github.com:<你>/ev-data.git   # 私有！"
echo "  git push -u origin master"
echo "  git tag v4 && git push --tags"
