#!/bin/bash
set -u
ROOT=/home/ubuntu/zxy/vlm-memory; LOG=$ROOT/scratch_ctrl_logs
. "$ROOT/scratch_gpu_lock.sh"
G=$(gpu_claim)
echo "$(date +%H:%M) [GPU$G] 粒度扫描冒烟"
CUDA_VISIBLE_DEVICES=$G "$ROOT/.venv/bin/python" -B "$ROOT/scratch_probe_nll_grain.py" \
    -d scbench_kv --num 1 --fracs 1e-4 1e-2 --n_rand 2 > "$LOG/nllgrain_smoke.log" 2>&1
if grep -q "判读：" "$LOG/nllgrain_smoke.log"; then
    echo "$(date +%H:%M) [GPU$G] 冒烟过，正式跑"
    CUDA_VISIBLE_DEVICES=$G "$ROOT/.venv/bin/python" -B "$ROOT/scratch_probe_nll_grain.py" \
        -d scbench_kv --num 8 > "$LOG/nllgrain_scbench_kv.log" 2>&1
else
    echo "$(date +%H:%M) 冒烟失败"; tail -20 "$LOG/nllgrain_smoke.log"
fi
gpu_release "$G"
