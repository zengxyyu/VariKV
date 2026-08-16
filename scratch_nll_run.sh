#!/bin/bash
# U^NLL oracle：两个 panel，一个残差涨 4.4、一个跌 9.96。用共享锁等卡。
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT" || exit 1
. "$ROOT/scratch_gpu_lock.sh"
# 先冒烟：1 篇 4 个候选，让错误快速暴露而不是跑两小时才崩
G=$(gpu_claim)
CUDA_VISIBLE_DEVICES=$G .venv/bin/python -u scratch_probe_nll_oracle.py \
    -d scbench_kv --num 1 --n_cand 4 > scratch_ctrl_logs/nll_smoke.log 2>&1
gpu_release "$G"
grep -q "===" scratch_ctrl_logs/nll_smoke.log || { echo "冒烟失败，不继续"; exit 1; }
for D in scbench_kv scbench_vt; do
    G=$(gpu_claim)
    CUDA_VISIBLE_DEVICES=$G nohup .venv/bin/python -u scratch_probe_nll_oracle.py \
        -d "$D" --num 20 --n_cand 32 > "scratch_ctrl_logs/nll_$D.log" 2>&1
    gpu_release "$G"
done
