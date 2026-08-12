#!/usr/bin/env bash
set -u; cd "$(dirname "$0")" || exit 1
for i in $(seq 1 240); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$m" -lt 2000 ] && break; sleep 30
done
echo "$(date -u +%H:%M) GPU0 空了，起 v2b point"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scratch_stage2b_train.py \
  --obj kl --residual --num_slots 16 --ratio 0.1 --max_ctx 32768 --chunk 16000 \
  --window 4096 --target_len 256 --kl_weight sensitive --ctx_pos random --seed 42 \
  --min_chunks 0 --steps 1500 --log_every 50 --val_windows 4 --val_every 100 \
  --mode point --out varikv/ckpt_kl_v2b > scratch_stage2b_logs/train_kl_v2b_point.log 2>&1
echo "$(date -u +%H:%M) v2b point 完成"
