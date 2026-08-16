#!/bin/bash
# 预算门控：只在 ratio ≤ rho_max 时施加 Δs。**不需要重训**，纯推理期开关，
# 而且 ratio 推理时已知 ⇒ 可部署（headroom 不可部署，要跑满缓存才知道）。
# 用两个 panel 检验：Retr.KV（残差有大收益）+ Retr.MultiHop（残差有大损失）。
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT/external/FastKVzip/prefill" || exit 1
for CFG in "scbench_kv 0.25" "scbench_vt 0.25" "scbench_kv 0.15" "scbench_vt 0.15"; do
    set -- $CFG; D=$1; RM=$2
    while true; do
        for G in 0 1 2 3 4 5 6 7; do
            [ "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader|wc -l)" -eq 0 ] \
                && { SLOT=$G; break 2; }
        done; sleep 90
    done
    VARIKV_RATIOS=1.0,0.5,0.4,0.3,0.2,0.1,0.05 CUDA_VISIBLE_DEVICES=$SLOT \
        nohup "$ROOT/.venv/bin/python" -B eval_chunk.py \
        -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 --prefill_chunk 16000 \
        --window_size 4096 --level pair -d "$D" --tag "_rg" --ctrlm_rho_max "$RM" \
        --ctrlm_ckpt ../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt \
        > "$ROOT/scratch_ctrl_logs/rhogate_${D}_${RM}.log" 2>&1 &
    sleep 60
done
wait
