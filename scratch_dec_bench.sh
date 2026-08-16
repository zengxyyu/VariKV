#!/bin/bash
# 拆解的下游评测：4 个架构 × seed 0，同一 ratio 网格，和 v2/v3 完全可比。
# 判读：affine(225 参数) ≈ v2(638K) ⇒ +4.27 是跨层/头的尺度重校准，不是 KV 语义。
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT/external/FastKVzip/prefill" || exit 1
for A in affine bias scalar kv; do
    while true; do
        for G in 0 1 2 3 4 5 6 7; do
            U=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader|wc -l)
            [ "$U" -eq 0 ] && { SLOT=$G; break 2; }
        done; sleep 90
    done
    VARIKV_RATIOS=1.0,0.5,0.4,0.3,0.2,0.1,0.05 CUDA_VISIBLE_DEVICES=$SLOT \
        nohup "$ROOT/.venv/bin/python" -B eval_chunk.py \
        -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 --prefill_chunk 16000 \
        --window_size 4096 --level pair -d scbench_kv --tag "_dc$A" \
        --ctrlm_ckpt "../../../varikv/dec_${A}_s0.pt/memoryless.pt" \
        > "$ROOT/scratch_ctrl_logs/decbench_$A.log" 2>&1 &
    sleep 60
done
wait
