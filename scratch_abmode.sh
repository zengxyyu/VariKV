#!/bin/bash
# 干净版 compat ckpt 在 memoryless vs stateful 下的 A/B —— 直接量化那个默认值 bug 的代价。
set -u
R=/home/ubuntu/zxy/vlm-memory; L=$R/scratch_ctrl_logs
cd "$R/external/FastKVzip/prefill" || exit 1
. "$R/scratch_gpu_lock.sh"
for M in memoryless stateful; do
    G=$(gpu_claim)
    VARIKV_RATIOS=0.2,0.1 CUDA_VISIBLE_DEVICES=$G nohup "$R/.venv/bin/python" -B eval_chunk.py \
        -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 40 --prefill_chunk 16000 \
        --window_size 4096 --level pair -d scbench_kv --tag "_abmode_$M" \
        --ctrlm_mode $M --ctrlm_ckpt ../../../varikv/v2c_s0_compat.pt \
        > "$L/abmode_$M.log" 2>&1 &
    echo "$(date +%H:%M) [GPU$G] A/B $M"
    sleep 20
done
wait
for g in 0 1 2 3 4 5 6 7; do gpu_release "$g"; done
echo "$(date +%H:%M) A/B 完成"
