#!/usr/bin/env bash
# 质心 @ ratio 0.3 / 0.2，Figure-11 全部 11 个面板 × K∈{16,1024}
#
# **这是论文生死实验。** 现有全部质心结果都在 ratio 0.1，而论文 Figure 11 的 x 轴是
# 0.2–1.0 ⇒ 没有任何可对照的点。要回答的问题只有一个：
#     这个方法是不是只在 FastKVzip 已经崩掉的 0.1 regime 才有效？
# 若 0.3 处 FastKVzip 已近满缓存而质心 ≈ +0.1，那它只是 extreme-compression rescue；
# 若 0.2/0.3 仍稳定 +2…+6，方法强得多。
#
# 注意 `eval_chunk.py` 是 `for ratio in set_ratios(): prefill(chunk_ratio=ratio)` ——
# **每个 ratio 各自预填一次**，所以两个 ratio 的成本约等于单 ratio 的两倍，
# 不像非 chunked 路径那样能共享预填。
#
# 基线：0.3/0.2 在 `__full_chunk16k_w4096`（`scratch_stage2b_sweep.py` 那批）里**已经有**，
# 同配置同 100 条 ⇒ 不需要重跑基线。但 `scbench_kv` 的 0.3/0.2 也在 `_full` 里。
# 报告端读 `__full` 取 0.3/0.2 两行即可。
#
# 用法： nohup bash scratch_cen23.sh > scratch_cen23.log 2>&1 &
set -u
cd "$(dirname "$0")"
ROOT=$PWD
LOGS=$ROOT/scratch_cen23_logs; mkdir -p "$LOGS"
cd external/FastKVzip/prefill

# 长任务优先（实测单 ratio：repoqa 最贵，choice_eng 最便宜）
PANELS="scbench_repoqa scbench_prefix_suffix scbench_kv scbench_mf scbench_vt \
scbench_summary gsm squad scbench_many_shot scbench_qa_eng scbench_choice_eng"
KS="1024 16"

JOBS=()
for d in $PANELS; do for K in $KS; do JOBS+=("$d:$K"); done; done
echo "共 ${#JOBS[@]} 个任务（11 面板 × 2 个 K），ratio 0.3,0.2"

free_gpu() {   # 进程存在性是二值的；显存会在生成阶段掉到很低，用显存判空会把两个任务派到同一张卡
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] || continue
    sleep 20                                  # 20 秒后再确认一次，避免抢到正在启动的卡
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] && { echo $g; return; }
  done
  echo ""
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  job=${JOBS[$i]}; d=${job%:*}; K=${job#*:}
  tag="_c23${K}_${d}"
  mk="$LOGS/.done__${d}_K${K}"
  [ -f "$mk" ] && { echo "跳过 $job（已完成）"; i=$((i+1)); continue; }
  g=$(free_gpu)
  [ -z "$g" ] && { sleep 60; continue; }
  echo "[$(date -u +%H:%M)] $job → GPU$g"
  (
    CUDA_VISIBLE_DEVICES=$g VARIKV_RATIOS=0.3,0.2 \
      ../../../.venv/bin/python -B eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M \
      -g fastkvzip --num 100 --prefill_chunk 16000 --window_size 4096 --level pair \
      -d "$d" --tag "$tag" --centroid_k "$K" \
      > "$LOGS/c23${K}_${d}.log" 2>&1
    grep -q "Finished\." "$LOGS/c23${K}_${d}.log" && touch "$mk" \
      && echo "[$(date -u +%H:%M)] ✓ $job" || echo "[$(date -u +%H:%M)] ✗ $job 失败"
  ) &
  sleep 45                      # 错开启动，别让两个任务同时抢同一张刚空出来的卡
  i=$((i+1))
done
wait
echo "ALL DONE $(date -u)"
ls "$LOGS"/.done__* 2>/dev/null | wc -l | xargs echo "完成标记数:"
