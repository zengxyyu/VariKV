#!/bin/bash
# 真 7B 上的 v2 等价性验收，排队等卡（不与在跑的评测抢显存）。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory; LOG=$ROOT/scratch_ctrl_logs
. "$ROOT/scratch_gpu_lock.sh"
G=$(gpu_claim)
echo "$(date +%H:%M) [GPU$G] verify-real"
CUDA_VISIBLE_DEVICES=$G "$ROOT/.venv/bin/python" -B "$ROOT/varikv_v2.py" verify-real \
    --num 2 > "$LOG/v2_verify_real.log" 2>&1
tail -5 "$LOG/v2_verify_real.log"
gpu_release "$G"
