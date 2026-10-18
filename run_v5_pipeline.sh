#!/usr/bin/env bash
# ============================================================================
# run_v5_pipeline.sh —— 一条命令跑完「生成数据 → 重建训练集 → 训练」
#
#   export OPENAI_API_KEY=sk-xxx
#   bash run_v5_pipeline.sh                 # 全套：生成 520 条 + 重建 + 训练
#   bash run_v5_pipeline.sh --skip-gen      # 已经有数据，只重建 + 训练
#   bash run_v5_pipeline.sh --skip-train    # 只生成 + 重建，看数据报告
#   bash run_v5_pipeline.sh --fg            # 训练放前台（能直接看进度条）
#   N=800 EPOCHS=3 bash run_v5_pipeline.sh  # 改数据量 / 轮数
#
# 必须先激活环境或用脚本内的 ~/venvs/ft。本脚本自带显存互斥检查。
# ============================================================================
set -o pipefail

BASE="/mnt/e/VMOS_workspace/model_tuning"
PY="${PY:-$HOME/venvs/ft/bin/python}"
N="${N:-520}"
EPOCHS="${EPOCHS:-2}"
OUT_DIR="${OUT_DIR:-$HOME/ft_outputs/ev-v5}"
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
    -h|--help)    sed -n '2,15p' "$0"; exit 0 ;;
    *)            echo "未知参数: $1"; exit 1 ;;
  esac
  shift
done

cd "$BASE" || { echo "[x] 进不去 $BASE"; exit 1; }
mkdir -p logs

[ -x "$PY" ] || { echo "[x] 找不到 python: $PY"; echo "    先跑 bash setup-ml-env-uv.sh"; exit 1; }

# ---------------- 0) 显存互斥：8GB 卡不能边推理边训练 ----------------
if pgrep -f "serve_hf.py" >/dev/null 2>&1; then
  echo "[!] 检测到 serve_hf.py 正在跑 —— 8GB 显存扛不住训练 + 推理同时占卡"
  echo "    先执行：pkill -f serve_hf.py"
  exit 1
fi

# ---------------- 1) 生成 help 数据 ----------------
if [ "$SKIP_GEN" -eq 0 ]; then
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[x] 需要 OPENAI_API_KEY：export OPENAI_API_KEY=sk-xxx"
    exit 1
  fi
  echo "[1/3] 生成 help 数据（目标 $N 条，中断后重跑同一条命令即续传）"
  "$PY" gen_persona_data.py \
      --scenes personas/help_pack.json \
      --n "$N" --per-call 10 --num-seeds 12 --temperature 1.15 \
      --out my_data/help_api_v2.json || { echo "[x] 生成失败"; exit 1; }
else
  echo "[1/3] 跳过生成（--skip-gen）"
fi

# ---------------- 2) 重建训练集（配比 + 风格清洗）----------------
echo "[2/3] 重建训练集：help 40% / chat 60%（按 assistant 轮次）"
"$PY" rebuild_dataset.py \
    --src help=my_data/help_api.json+my_data/help_api_v2.json:0.40 \
    --src chat=my_data/ev3_all.json:0.60 \
    --out my_data/ev5_all.json --keep-ellipsis 0.08 || { echo "[x] 重建失败"; exit 1; }

if [ "$SKIP_TRAIN" -eq 1 ]; then
  echo "[3/3] 跳过训练（--skip-train）—— 先看上面的报告，确认没问题再跑："
  echo "      bash run_v5_pipeline.sh --skip-gen"
  exit 0
fi

# ---------------- 3) 训练 ----------------
echo "[3/3] 训练 v5 → $OUT_DIR"
export DATA_PATH="$BASE/my_data/ev5_all_train.json"
export DATA_FORMAT=chat
export RESPONSES_ONLY=1
export NUM_EPOCHS="$EPOCHS"
export OUTPUT_DIR="$OUT_DIR"
export SMOKE_TEST=0

if [ "$FG" -eq 1 ]; then
  "$PY" train_lora.py 2>&1 | tee "logs/train_ev_v5.log"
else
  nohup "$PY" train_lora.py > "logs/train_ev_v5.log" 2>&1 &
  echo "    已放后台，PID $!"
  echo "    看进度：tail -f $BASE/logs/train_ev_v5.log"
fi

echo ""
echo "训练完成后："
echo "  adapter  bash start_ev.sh --adapter $OUT_DIR/lora_adapter"
echo "  备份     bash backup.sh"
