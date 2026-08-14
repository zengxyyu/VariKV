#!/usr/bin/env bash
# 等第一张卡空出来 → 跑 ControlRetainCache 的五项验收。
#
# r05 的 15 个任务已全部派出（队列为空，调度器在 wait），所以卡一旦空出来就是永久空的，
# 不存在和它抢卡的竞态。仍然用「进程存在性 + 20 秒复核」判空，因为生成阶段显存会掉到很低。
set -u
cd "$(dirname "$0")"
free_gpu() {
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
echo "[$(date -u +%H:%M)] 等待空卡…"
while :; do
  g=$(free_gpu); [ -n "$g" ] && break
  sleep 60
done
echo "[$(date -u +%H:%M)] GPU$g 空出来了，跑控制臂验收"
CUDA_VISIBLE_DEVICES=$g .venv/bin/python scratch_verify_ctrl.py
echo "[$(date -u +%H:%M)] 验收退出码 $?"
