#!/bin/bash
# U^NLL 有效性对照 + massdir 重跑（补上第 16-20 列的打印与存盘）。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
LOG=$ROOT/scratch_ctrl_logs
. "$ROOT/scratch_gpu_lock.sh"
G=$(gpu_claim)
echo "$(date +%H:%M) [GPU$G] nll_stab 冒烟"
CUDA_VISIBLE_DEVICES=$G "$ROOT/.venv/bin/python" -B "$ROOT/scratch_probe_nll_stab.py" \
    -d scbench_kv --num 1 --n_cand 4 --block 128 > "$LOG/nllstab_smoke.log" 2>&1
if grep -q "^C 跨存活集合" "$LOG/nllstab_smoke.log"; then
    echo "$(date +%H:%M) [GPU$G] 冒烟过，正式跑"
    CUDA_VISIBLE_DEVICES=$G "$ROOT/.venv/bin/python" -B "$ROOT/scratch_probe_nll_stab.py" \
        -d scbench_kv --num 6 --n_cand 24 > "$LOG/nllstab_scbench_kv.log" 2>&1
else
    echo "$(date +%H:%M) 冒烟失败，见 nllstab_smoke.log"
fi
gpu_release "$G"
