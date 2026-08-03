#!/usr/bin/env bash
# Stage-1 生死实验的端到端驱动：等 Figure 11 复现跑完 → 训练 → 五档评测。
#
# 为什么等：8 张卡现在全被 Figure 11 复现占着，挤进去会互相拖慢甚至 OOM。
#
# 档位分工（varikv/config.py:Config.ablation）：
#   tier 1 discard  KVzip            无参数，不训练
#   tier 2 point    Infini-attention 需训练
#   tier 3 moment   MomentKV         **training-free**，不训练 ← 真实门槛
#   tier 4 fe+point IndexMem 加强    需训练
#   tier 5 fe+dist  VariKV           需训练
# 需要训练的三档并行占三张卡；1 和 3 直接进评测。
#
# 最关键的对比是 3 vs 5：两边都持有二阶信息，自变量收敛到
# 「贝叶斯信念 + KL 门控 + 方差感知读出」相对「频率派矩统计」是否真的有用。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
PY=$ROOT/.venv/bin/python
LOG=$ROOT/scratch_stage1_logs
mkdir -p "$LOG"
cd "$ROOT"

BUDGET=256
STEPS=${STEPS:-1500}
EVAL_LIMIT=${EVAL_LIMIT:-120}

echo "STAGE1_DRIVER_START: $(date)"

# ---------- 1. 等 Figure 11 复现退出 ----------
while pgrep -f "scratch_repro_full.py --run" >/dev/null; do sleep 300; done
echo "FIG11_DONE, GPUs free: $(date)"
sleep 30

# ---------- 2. 并行训练需要训练的三档 ----------
echo "TRAIN_START: $(date)  steps=$STEPS budget=$BUDGET"
for spec in "2:0" "4:1" "5:2"; do
  tier="${spec%%:*}"; gpu="${spec##*:}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY varikv/train.py \
      --tier "$tier" --budget "$BUDGET" --steps "$STEPS" \
      > "$LOG/train_tier${tier}.log" 2>&1 &
  echo "  tier$tier → GPU$gpu (pid $!)"
done
wait
echo "TRAIN_DONE: $(date)"
for t in 2 4 5; do
  echo "  tier$t 末行: $(tail -3 "$LOG/train_tier${t}.log" | tr '\n' ' ')"
done

# ---------- 3. 五档评测 ----------
echo
echo "EVAL_START: $(date)  limit=$EVAL_LIMIT"
CUDA_VISIBLE_DEVICES=0 $PY varikv/evaluate.py \
    --tier 1 2 3 4 5 --budget "$BUDGET" --limit "$EVAL_LIMIT" \
    > "$LOG/eval_all.log" 2>&1
echo "EVAL_DONE: $(date)"
echo
echo "############ 结果 ############"
grep -A40 "四档对比\|五档对比\|=== tier" "$LOG/eval_all.log" | tail -60
echo
echo "STAGE1_DRIVER_DONE: $(date)"
