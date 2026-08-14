#!/usr/bin/env bash
# 控制臂第一批：Retr.KV @ratio 0.1，8 个臂。**训练无关**——β/ρ/src 都是超参。
#
# 为什么是 Retr.KV @0.1：冻结掩码 2×2 曾在这里测到「强行换用有记忆臂的 valid 掩码
# 得到 +44 分」，也就是 FastKVzip 的选择在这个面板上**可证明不是最优的**。
# 这是"更好的 selection 到底值多少分"已知有正答案的唯一位置。
#
# 基线不重跑：直接复用 r05 的 `_r05b`（同模型/同 chunk/同 window/同 level/同 100 条，
# 且它的 VARIKV_RATIOS 里就含 0.1）。
#
# 幅度为什么取两档：验收实测 β=0.5 只翻转 **0.895%** 的掩码，而历史参照是
# v1（拿到 +51）位移 2.07%、v2b（零收益）位移 3.18%。0.895% 不到 v1 的一半，
# 说明 0.5 可能还没进入有效决策区间。这是对实测数字的反应，不是钓超参。
#
# **预注册判据**：某臂必须同时 (a) 显著优于基线 且 (b) 显著优于**它自己的 shuffle**。
# 只满足 (a) 只能说明"扰动分数有用"，不能说明历史几何含可用信号。
#
# 用法： setsid nohup bash scratch_ctrl1.sh > scratch_ctrl1.log 2>&1 &
set -u
cd "$(dirname "$0")"
ROOT=$PWD
PY=$ROOT/.venv/bin/python
LOGS=$ROOT/scratch_ctrl1_logs; mkdir -p "$LOGS"
NUM=${NUM:-100}
DATA=${DATA:-scbench_kv}

# name|src|beta|shuffle
ARMS=(
  "ret05|retained|0.5|0"    "ret05s|retained|0.5|1"
  "ret15|retained|1.5|0"    "ret15s|retained|1.5|1"
  "evi05|evicted|-0.5|0"    "evi05s|evicted|-0.5|1"
  "evi15|evicted|-1.5|0"    "evi15s|evicted|-1.5|1"
)

free_gpu() {   # 进程存在性判空 + 20 秒复核（生成阶段显存会掉到很低，不能用显存判）
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] || continue
    sleep 20
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] && { echo "$g"; return; }
  done
  echo ""
}

echo "共 ${#ARMS[@]} 个臂（$DATA @0.1，$NUM 条）"
i=0
while [ $i -lt ${#ARMS[@]} ]; do
  IFS='|' read -r name src beta shuf <<< "${ARMS[$i]}"
  mk="$LOGS/.done__$name"
  [ -f "$mk" ] && { echo "跳过 $name"; i=$((i+1)); continue; }
  g=$(free_gpu); [ -z "$g" ] && { sleep 60; continue; }
  extra=""; [ "$shuf" = "1" ] && extra="--ctrl_shuffle"
  echo "[$(date -u +%H:%M)] $name (src=$src β=$beta shuffle=$shuf) → GPU$g"
  (
    cd "$ROOT/external/FastKVzip/prefill"
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$g $PY -B eval_chunk.py \
      -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num "$NUM" \
      --prefill_chunk 16000 --window_size 4096 --level pair \
      -d "$DATA" --tag "_c1$name" \
      --ctrl --ctrl_src "$src" --ctrl_beta "$beta" $extra \
      > "$LOGS/$name.log" 2>&1
    rc=$?
    # 完成判据是 harness 自己打的 `Finished.`，不是 rc——被杀的进程 rc 也可能是 0
    if [ $rc -eq 0 ] && tail -3 "$LOGS/$name.log" | grep -q "Finished."; then
      touch "$mk"; echo "[$(date -u +%H:%M)] $name 完成"
    else
      echo "[$(date -u +%H:%M)] **$name 失败** rc=$rc"
    fi
  ) &
  sleep 30
  i=$((i+1))
done
wait
echo "[$(date -u +%H:%M)] ALL DONE"
for a in "${ARMS[@]}"; do n=${a%%|*}; [ -f "$LOGS/.done__$n" ] && echo "  ✓ $n" || echo "  ✗ $n"; done
