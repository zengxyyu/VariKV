#!/bin/bash
# **全部既有结果都只在 ratio 0.1 上。** 而论文 Figure 11 的横轴是 0.2→1.0，
# 0.1 在论文里没有对应点；headroom 表也显示 0.3/0.2 的余量远小于 0.1。
# 方法与 ratio 的耦合：margin=(s0−τ)/σ_g 的分布、以及"近阈值候选池"的位置，
# 都由训练时的 ratio 0.1 决定。测它是否迁移。
# VARIKV_RATIOS 支持一次预填算多个 ratio，所以三个工作点几乎不额外花钱。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
VARIKV_RATIOS=0.4,0.3,0.2 CUDA_VISIBLE_DEVICES=0 nohup ../../../.venv/bin/python -B \
    eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
    --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv --tag _rtb \
    > ../../../scratch_ctrl_logs/ratio_base.log 2>&1 &
VARIKV_RATIOS=0.4,0.3,0.2 CUDA_VISIBLE_DEVICES=1 nohup ../../../.venv/bin/python -B \
    eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
    --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv --tag _rtm \
    --ctrlm_mode memoryless \
    --ctrlm_ckpt ../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt \
    > ../../../scratch_ctrl_logs/ratio_mem.log 2>&1 &
wait
