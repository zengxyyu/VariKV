#!/usr/bin/env bash
# 等 ratio 0.3/0.2 扫描的队列排空、且有卡**真正空闲**，再跑升级版 (mass × direction) 探针。
#
# 为什么要等：`scratch_cen23.sh` 的调度器会抢走任何空出来的卡，两个调度器抢同一张卡
# 会撞（CLAUDE.md 记过）。所以触发条件是「22 个任务全部派出去」——那时调度器不再要卡。
# 也不共卡：n=2 那版是共卡跑的（带 0.45 显存上限，没扰动邻居），但用户要求不再这么做。
#
# 为什么要重跑：n=2 版的 280 个观测**嵌套在 2 条样本里**，不独立，对「最优 γ」这类
# 结论的有效 n 接近 2。而且 Oracle-Mass 的 P90 已经是 1.1491 > 1 —— 最差 10% 的头上
# 「把质量修准」比不修还差，说明头级散布很宽、「加一个全局常数」有过冲风险。
#
# 三个数据集是故意挑的：
#   scbench_kv            质心收益最大（+11.00 ★），headroom +35.60
#   scbench_prefix_suffix headroom 更大（+41.40）但质心只 +3.60 ⇒ 收益不由 headroom 决定
#   scbench_vt            headroom **负**（基线 49.47 > 满缓存 41.07）⇒ 最优 γ 可能是 0，
#                         如果是，收缩故事就是数据集相关的，不能当通用方法
#
# 用法： nohup bash scratch_massdir_wait.sh > scratch_massdir_wait.log 2>&1 &
set -u
cd "$(dirname "$0")"

echo "[$(date -u +%H:%M)] 等 cen23 队列排空（22 个全部派出）…"
while [ "$(grep -c '→ GPU' scratch_cen23.log 2>/dev/null || echo 0)" -lt 22 ]; do sleep 120; done
echo "[$(date -u +%H:%M)] 队列已排空，开始等真正空闲的卡"

free_gpu() {
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] || continue
    sleep 20
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] && { echo $g; return; }
  done
  echo ""
}

for job in "scbench_kv 16" "scbench_vt 16" "scbench_prefix_suffix 16" "scbench_kv 1024"; do
  set -- $job; d=$1; K=$2
  out="scratch_massdir_${d}_K${K}.log"
  [ -s "$out" ] && grep -q "判读" "$out" && { echo "跳过 $job（已完成）"; continue; }
  while :; do g=$(free_gpu); [ -n "$g" ] && break; sleep 120; done
  echo "[$(date -u +%H:%M)] $job → GPU$g"
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u scratch_probe_massdir.py \
      --data "$d" --K "$K" --n 0 --mem_frac 0 > "$out" 2>&1
  echo "[$(date -u +%H:%M)] $job 完成 rc=$?"
done
echo "[$(date -u +%H:%M)] ALL DONE"
