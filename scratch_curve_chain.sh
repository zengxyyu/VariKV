#!/bin/bash
# 等当前 intervention 跑完 → 跑恢复曲线（top5/20/80/all/low）
set -u; cd /home/ubuntu/zxy/vlm-memory
for i in $(seq 1 60); do
  grep -q "CHAIN DONE" scratch_p0_results.log 2>/dev/null && break
  sleep 60
done
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -B scratch_probe_intervene.py \
    --n_samples 4 --n_queries 3 --curve 5 20 80 \
    > scratch_probe_curve.log 2>&1
echo "CURVE DONE $(date)" >> scratch_probe_curve.log
