#!/usr/bin/env bash
# 三个 v3 方差隔离 run 训完就自动评测（否则 ckpt 会躺着没人评，卡被调度器拿去跑队列）。
# 用 CANDIDATES 里的卡，与 p234 的候选（2-7）错开风险：这里也用进程列表判空闲，
# 且每起一个 sleep 60 错峰。
set -u; cd "$(dirname "$0")" || exit 1
declare -A CK=( [v3a]=varikv/ckpt_kl_v3a [v3b]=varikv/ckpt_kl_v3b [v3c]=varikv/ckpt_kl_v3c )
free_gpu(){ for g in 0 1 2 3 4 5 6 7; do
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
   | grep -qF "$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i $g)" || { echo $g; return 0; }
 done; return 1; }
for k in v3a v3b v3c; do
  f="${CK[$k]}/s2b_dist_k16.pt"
  ( for i in $(seq 1 480); do [ -f "$f" ] && break; sleep 30; done
    [ -f "$f" ] || { echo "$k 超时"; exit 1; }
    for i in $(seq 1 120); do g=$(free_gpu) && break || sleep 30; done
    echo "$(date -u +%H:%M) [$k] ckpt 就绪，起评测于 GPU$g"
    cd external/FastKVzip/prefill
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$g ../../../.venv/bin/python -B eval_chunk.py \
      -m Qwen/Qwen2.5-7B-Instruct-1M -d scbench_kv -g fastkvzip --num 100 \
      --prefill_chunk 16000 --window_size 4096 --level pair --tag "_kl${k}_dist" \
      --varikv_ckpt "../../../${CK[$k]}/s2b_dist_k16.pt" --varikv_residual --varikv_slots 16 \
      > "../../../scratch_centroid_logs/kl${k}_dist.log" 2>&1
    echo "$(date -u +%H:%M) [$k] 评测完成" ) &
  sleep 5
done
wait; echo "$(date -u +%H:%M) 三个 v3 评测全部结束"
