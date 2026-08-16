#!/bin/bash
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
LOG=$ROOT/scratch_ctrl_logs
. "$ROOT/scratch_gpu_lock.sh"
G=$(gpu_claim)
echo "$(date +%H:%M) [GPU$G] massdir K=16 重跑（真 joint + 精确 MGF oracle）"
CUDA_VISIBLE_DEVICES=$G "$ROOT/.venv/bin/python" -B "$ROOT/scratch_probe_massdir.py" \
    --data scbench_kv --K 16 --n 20 > "$LOG/massdir_K16.log" 2>&1
gpu_release "$G"
