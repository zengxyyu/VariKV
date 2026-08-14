#!/bin/bash
# 学习版 B 的**下游评测**：Retr.KV @0.1，三条臂各 100 样本。
# 训练侧判据（Δ_history）没通过，但 CLAUDE.md 记着"训练侧指标与下游反相关"
# （ckpt_kl_v2a 验证最好、下游最差），所以不能拿训练侧结论替代下游测量。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
i=0
for M in stateful memoryless shuffled; do
    G=$(echo "3 4 5" | cut -d' ' -f$((i + 1)))
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$G nohup ../../../.venv/bin/python -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
        --tag "_b1$M" --ctrlm_ckpt "../../../varikv/ctrl_b_a1_s0.pt/$M.pt" \
        > "../../../scratch_ctrl_logs/bench_$M.log" 2>&1 &
    i=$((i + 1))
done
wait
