#!/usr/bin/env bash
# Stage-1 容量扫描 + 连续指标重测（2026-08-07）
#
# 起因：修好分词 bug 后，五档 exact-match 仍然全 0，但那是**指标没有分辨率**——
# 把 3.5k 上下文压进 K 个 latent 高斯是 ~377:1，逐字符复原高熵随机串对任何有损
# 记忆都不可能。同一批样本上 nll 却把各档清楚分开（无记忆 5.04 → 有记忆 2.60），
# 生成文本也显示记忆带回了部分信息（gold jade-shrike-85 → 生成 jade-otter-59）。
#
# 所以本次做两件事：
#   1. 主指标换成 answer 上的 nll，跑满 4 个干扰档 × 5 档位
#   2. 扫 K=16/32/64 —— 「容量太小压不出效果」是 CLAUDE.md 已列的已知风险，
#      而槽参数只有 K×d_z，加 K 几乎免费（实测 K=16→32 仍是 0.40M）
#
# 判据：tier5 相对 tier2/4 在 K=16 时只差 0.02~0.07 nats（n=12，基本是噪声）。
# 若 K 拉大后仍不分离，那才是有说服力的 NO-GO。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
PY=$ROOT/.venv/bin/python
LOG=$ROOT/scratch_stage1_sweep_logs
CKPT=$ROOT/varikv/ckpt
mkdir -p "$LOG" "$CKPT"
cd "$ROOT"

BUDGET=256
STEPS=${STEPS:-1500}
PER_LEVEL=${PER_LEVEL:-40}      # 每个干扰档取多少样本（4 档 → 每个配置 160 个）
KS=${KS:-"16 32 64"}
TIERS="1 2 3 4 5"

echo "SWEEP_START: $(date)   K=[$KS] steps=$STEPS per_level=$PER_LEVEL budget=$BUDGET"

# ---------- 1. 训练：K=32/64 的 tier 2/4/5（K=16 已有 ckpt） ----------
echo
echo "TRAIN_START: $(date)"
gpu=0
pids=()
for k in $KS; do
  for t in 2 4 5; do
    if [ -f "$CKPT/k${k}_tier${t}.pt" ]; then
      echo "  skip K=$k tier$t（ckpt 已存在）"
      continue
    fi
    CUDA_VISIBLE_DEVICES=$gpu $PY varikv/train.py \
        --tier "$t" --budget "$BUDGET" --steps "$STEPS" --num_slots "$k" \
        > "$LOG/train_k${k}_tier${t}.log" 2>&1 &
    pids+=($!)
    echo "  K=$k tier$t → GPU$gpu (pid $!)"
    gpu=$(( (gpu + 1) % 8 ))
  done
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}"; fi
echo "TRAIN_DONE: $(date)"
for k in $KS; do
  for t in 2 4 5; do
    f="$LOG/train_k${k}_tier${t}.log"
    [ -f "$f" ] && echo "  K=$k tier$t: $(grep '^step' "$f" | tail -1)"
  done
done

# ---------- 2. 评测：3 个 K × 5 档，按 GPU 分发 ----------
echo
echo "EVAL_START: $(date)"
gpu=0
pids=()
for k in $KS; do
  for t in $TIERS; do
    CUDA_VISIBLE_DEVICES=$gpu $PY varikv/evaluate.py \
        --tier "$t" --budget "$BUDGET" --num_slots "$k" \
        --per_level "$PER_LEVEL" --json_out "$LOG/res_k${k}_tier${t}.json" \
        > "$LOG/eval_k${k}_tier${t}.log" 2>&1 &
    pids+=($!)
    gpu=$(( (gpu + 1) % 8 ))
    # 一次最多 8 个并发，占满就等
    if [ ${#pids[@]} -ge 8 ]; then wait "${pids[@]}"; pids=(); fi
  done
done
if [ ${#pids[@]} -gt 0 ]; then wait "${pids[@]}"; fi
echo "EVAL_DONE: $(date)"

# ---------- 3. 汇总（含配对比较） ----------
echo
echo "############ 汇总 ############"
$PY scratch_stage1_sweep_report.py --logdir "$LOG" --ks $KS
echo
echo "SWEEP_DONE: $(date)"
