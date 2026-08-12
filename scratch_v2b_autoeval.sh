#!/usr/bin/env bash
# v2b dist 训练一结束就立刻在 GPU1 上开评测（省掉等人来看的空转）。
# GPU1 是 v2b dist 自己的卡，训练退出就空出来；p234 的候选卡是 2-7，不冲突。
set -u; cd "$(dirname "$0")" || exit 1
CK=varikv/ckpt_kl_v2b/s2b_dist_k16.pt
for i in $(seq 1 480); do [ -f "$CK" ] && break; sleep 30; done
[ -f "$CK" ] || { echo "超时：ckpt 未出现"; exit 1; }
echo "$(date -u +%H:%M) ckpt 已出现，等 GPU1 空"
for i in $(seq 1 40); do
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    | grep -qF "$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i 1)" || break
  sleep 15
done
echo "$(date -u +%H:%M) 起 v2b dist 的下游评测"
cd external/FastKVzip/prefill
VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=1 ../../../.venv/bin/python -B eval_chunk.py \
  -m Qwen/Qwen2.5-7B-Instruct-1M -d scbench_kv -g fastkvzip --num 100 \
  --prefill_chunk 16000 --window_size 4096 --level pair --tag _klv2b_dist \
  --varikv_ckpt ../../../varikv/ckpt_kl_v2b/s2b_dist_k16.pt --varikv_residual \
  --varikv_slots 16 > ../../../scratch_centroid_logs/klv2b_dist.log 2>&1
echo "$(date -u +%H:%M) v2b dist 评测结束"
