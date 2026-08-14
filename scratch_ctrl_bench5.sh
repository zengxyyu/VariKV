#!/bin/bash
# memoryless seed0 拿到 +4.40★（三臂最好），补 seed 1/2 验证可复现性。
# stateful 已有三种子 +2.60 ± 0.53，可复现；memoryless 必须同等对待。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
freegpu() {
    while true; do
        for g in 0 1 2 3 4 5 6 7; do
            [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader|wc -l)" -eq 0 ] \
                && { echo $g; return; }
        done; sleep 120
    done
}
for S in 1 2; do
    G=$(freegpu)
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$G nohup ../../../.venv/bin/python -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
        --tag "_b2s${S}me" --ctrlm_mode memoryless \
        --ctrlm_ckpt "../../../varikv/ctrl_b_a1_s$S.pt/memoryless.pt" \
        > "../../../scratch_ctrl_logs/bench_s${S}_memoryless.log" 2>&1 &
    sleep 90
done
wait
