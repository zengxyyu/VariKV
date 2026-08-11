#!/bin/bash
# 等 MGF 全量跑完 → 跑 ε_MGF 的 query 方向敏感性（决定"每簇一个校正标量"是否可行）
set -u
cd /home/ubuntu/zxy/vlm-memory
for i in $(seq 1 120); do
  grep -q "判读（v5 Q2/Q3）" scratch_probe_mgf.log 2>/dev/null && break
  sleep 60
done
echo "MGF 全量完成 $(date)"
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -B scratch_probe_mgf_stability.py \
    --n_samples 2 --N 64000 --Ws 2048 8192 --n_queries 4 --gq 2 --layer_stride 6 \
    > scratch_probe_mgf_stability.log 2>&1
echo "STABILITY DONE $(date)"
