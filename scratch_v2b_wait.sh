#!/usr/bin/env bash
# 等 GPU 0-3 的 v2 评测跑完，再起 v2b —— 「+21.60 能不能复现」的干净判据。
# v2b = 修复后的代码 + --min_chunks 0（全 34 篇，与 v1 同一文档集合）。
set -u
cd "$(dirname "$0")" || exit 1
for i in $(seq 1 720); do
  n=0
  for t in klv2a_dist klv2a_point klv2s_dist klv2s_point; do
    [ -f "scratch_centroid_logs/$t.log" ] && grep -q "Finished." "scratch_centroid_logs/$t.log" && n=$((n+1))
  done
  [ "$n" -ge 4 ] && break
  sleep 30
done
echo "$(date -u +%H:%M) v2 评测已完成，起 v2b"
C="--obj kl --residual --num_slots 16 --ratio 0.1 --max_ctx 32768 --chunk 16000
   --window 4096 --target_len 256 --kl_weight sensitive --ctx_pos random --seed 42
   --min_chunks 0 --steps 1500 --log_every 50 --val_windows 4 --val_every 100
   --out varikv/ckpt_kl_v2b"
for m in dist point; do
  g=$([ "$m" = dist ] && echo 0 || echo 1)
  CUDA_VISIBLE_DEVICES=$g .venv/bin/python -u scratch_stage2b_train.py $C --mode $m \
      > scratch_stage2b_logs/train_kl_v2b_$m.log 2>&1 &
done
wait
echo "$(date -u +%H:%M) v2b 训练完成"
