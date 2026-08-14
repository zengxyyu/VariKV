#!/bin/bash
# 重跑两个对照臂：**必须显式给 --ctrlm_mode**。args.py:114 的默认值是 "stateful"
# 而不是空串，`_mode = args.ctrlm_mode or _ck.get("mode")` 于是永远取 stateful，
# 三次运行都以 stateful 模式执行、只换了权重文件（目录名全是 ctrlmstat8 是指纹）。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
i=0
for M in memoryless shuffled; do
    G=$(echo "4 5" | cut -d' ' -f$((i + 1)))
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$G nohup ../../../.venv/bin/python -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
        --tag "_b2$M" --ctrlm_mode "$M" \
        --ctrlm_ckpt "../../../varikv/ctrl_b_a1_s0.pt/$M.pt" \
        > "../../../scratch_ctrl_logs/bench2_$M.log" 2>&1 &
    i=$((i + 1))
done
wait
