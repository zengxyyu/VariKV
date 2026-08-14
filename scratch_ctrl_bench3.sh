#!/bin/bash
# seed 0 的 stateful 下游拿到 +3.00★，但**单次训练不是一次测量**——v1 的 +21.60
# 就是这么来的（三次重训跨度 39 分）。补 seed 1/2 的 stateful，等 GPU 空出来就上。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
run() {
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$2 ../../../.venv/bin/python -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
        --tag "_b1s$1st" --ctrlm_mode stateful \
        --ctrlm_ckpt "../../../varikv/ctrl_b_a1_s$1.pt/stateful.pt" \
        > "../../../scratch_ctrl_logs/bench_s$1_stateful.log" 2>&1
}
run 1 3 &
# GPU 6/7 上的 sm_cont 训练结束后再上 seed 2
( while pgrep -f "ctrl_train.py --traces scratch_ctrl_traces_sm_cont" >/dev/null; do sleep 60; done
  run 2 7 ) &
wait
