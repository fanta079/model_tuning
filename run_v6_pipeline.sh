#!/usr/bin/env bash
# ============================================================================
# run_v6_pipeline.sh —— v6：补「自述」和「机制解释」两件 v5 实测最差的事
#
#   export OPENAI_API_KEY=sk-xxx
#   bash run_v6_pipeline.sh                 # 全套：生成 320 条 + 重建 + 训练
#   bash run_v6_pipeline.sh --skip-gen      # 已经有数据，只重建 + 训练
#   bash run_v6_pipeline.sh --skip-train    # 只生成 + 重建，先看配比报告
#   bash run_v6_pipeline.sh --fg            # 训练放前台
#   N=400 EPOCHS=3 bash run_v6_pipeline.sh  # 改数据量 / 轮数
#
# 为什么有 v6（证据在 docs/踩坑记录.md 第 28 条）：
#   1) 真实对话里问「介绍一下你自己」，v5 答「E.V.。……就这样。」—— 等于没答。
#      根因是 ev3 的 30 条自述样本本身就长这样，其中还有「问这个做什么」这种
#      人设卡明令禁止的反问。rebuild_dataset.py 会把这些敷衍样本丢掉，
#      由本包生成的 120+ 条替换。
#   2) 验收时「为什么电脑越用越慢」被答成「怎么办」—— 缺机制解释。
#
# 配比：help 34% / v6 22% / chat 44%（按 assistant 轮次）
#   v6 不能给太高：自述有几十条就够，给多了模型会把「介绍自己」当默认反应。
# ============================================================================
set -o pipefail

BASE="/mnt/e/VMOS_workspace/model_tuning"
PY="${PY:-$HOME/venvs/ft/bin/python}"
N="${N:-320}"
EPOCHS="${EPOCHS:-2}"
OUT_DIR="${OUT_DIR:-$HOME/ft_outputs/ev-v6}"
SKIP_GEN=0
SKIP_TRAIN=0
FG=0

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-gen)   SKIP_GEN=1 ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --fg)         FG=1 ;;
    --n)          N="$2"; shift ;;
    --epochs)     EPOCHS="$2"; shift ;;
    --out-dir)    OUT_DIR="$2"; shift ;;
    -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
    *)            echo "未知参数: $1"; exit 1 ;;
  esac
  shift
done

cd "$BASE" || { echo "[x] 进不去 $BASE"; exit 1; }
mkdir -p logs

[ -x "$PY" ] || { echo "[x] 找不到 python: $PY"; echo "    先跑 bash setup-ml-env-uv.sh"; exit 1; }

# ---------------- 0) 显存互斥 ----------------
if pgrep -f "serve_hf.py" >/dev/null 2>&1; then
  echo "[!] serve_hf.py 正在跑 —— 8GB 卡扛不住训练 + 推理同时占卡"
  echo "    先执行：pkill -f serve_hf.py"
  exit 1
fi

# ---------------- 1) 生成 v6 数据 ----------------
if [ "$SKIP_GEN" -eq 0 ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[x] 需要 OPENAI_API_KEY：export OPENAI_API_KEY=sk-xxx"
    exit 1
  fi
  echo "[1/3] 生成 v6 数据（自述 + 机制解释，目标 $N 条，可续传）"
  "$PY" gen_persona_data.py \
      --scenes personas/v6_pack.json \
      --n "$N" --per-call 10 --num-seeds 18 --temperature 1.10 \
      --out my_data/v6_api.json || { echo "[x] 生成失败"; exit 1; }
else
  echo "[1/3] 跳过生成（--skip-gen）"
fi

# ---------------- 2) 重建训练集 ----------------
echo "[2/3] 重建训练集：help 34% / v6 22% / chat 44%（按 assistant 轮次）"
echo "      注意 chat 来源会自动丢掉「自述题 + 敷衍答」的样本"
"$PY" rebuild_dataset.py \
    --src help=my_data/help_api.json+my_data/help_api_v2.json:0.34 \
    --src v6=my_data/v6_api.json:0.22 \
    --src chat=my_data/ev3_all.json:0.44 \
    --out my_data/ev6_all.json --keep-ellipsis 0.08 || { echo "[x] 重建失败"; exit 1; }

if [ "$SKIP_TRAIN" -eq 1 ]; then
  echo "[3/3] 跳过训练（--skip-train）—— 先看上面的报告，确认没问题再跑："
  echo "      bash run_v6_pipeline.sh --skip-gen"
  exit 0
fi

# ---------------- 3) 训练 ----------------
echo "[3/3] 训练 v6 → $OUT_DIR"
export DATA_PATH="$BASE/my_data/ev6_all_train.json"
export DATA_FORMAT=chat
export RESPONSES_ONLY=1
export NUM_EPOCHS="$EPOCHS"
export OUTPUT_DIR="$OUT_DIR"
export SMOKE_TEST=0

if [ "$FG" -eq 1 ]; then
  "$PY" train_lora.py 2>&1 | tee "logs/train_ev_v6.log"
else
  nohup "$PY" train_lora.py > "logs/train_ev_v6.log" 2>&1 &
  echo "    已放后台，PID $!"
  echo "    看进度：tail -f $BASE/logs/train_ev_v6.log"
fi

echo ""
echo "训练完成后："
echo "  先验收  python probe_v5.py --tag v6"
echo "  adapter bash start_ev.sh --adapter $OUT_DIR/lora_adapter"
echo "  归档    cp -r $OUT_DIR/lora_adapter /mnt/e/VMOS_workspace/ev-models/ev-v6/"
