#!/bin/bash
# 补齐拆解四臂的种子 1/2 —— s0 单点不构成测量。
# v1 同一份代码三次重训跨度 39 分，本项目的硬规矩是 n≥3 种子并报跨种子散布。
# 用共享 GPU 锁，不自己扫卡（三个调度器同时扫会把作业叠到同一张卡直到 OOM，已发生过）。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT/external/FastKVzip/prefill" || exit 1
LOG=$ROOT/scratch_ctrl_logs
. "$ROOT/scratch_gpu_lock.sh"
declare -A PID=() LAST=()
for S in 1 2; do
for A in bias affine scalar kv; do
    [ -f "$LOG/.done_dcs_${A}_$S" ] && { echo "跳过 $A s$S"; continue; }
    for G in 0 1 2 3 4 5 6 7; do
        P=${PID[$G]:-}
        if [ -n "$P" ] && ! kill -0 "$P" 2>/dev/null; then
            LD=${LAST[$G]:-}
            [ -n "$LD" ] && tail -3 "$LOG/decbench_$LD.log" 2>/dev/null \
                | grep -q Finished && touch "$LOG/.done_dcs_$LD"
            gpu_release "$G"; unset "PID[$G]"
        fi
    done
    SLOT=$(gpu_claim)
    VARIKV_RATIOS=1.0,0.5,0.4,0.3,0.2,0.1,0.05 CUDA_VISIBLE_DEVICES=$SLOT \
        nohup "$ROOT/.venv/bin/python" -B eval_chunk.py \
        -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 --prefill_chunk 16000 \
        --window_size 4096 --level pair -d scbench_kv --tag "_dc${A}_s$S" \
        --ctrlm_ckpt "../../../varikv/dec_${A}_s$S.pt/memoryless.pt" \
        > "$LOG/decbench_${A}_s$S.log" 2>&1 &
    PID[$SLOT]=$!; LAST[$SLOT]="${A}_s$S"
    echo "$(date +%H:%M)  [GPU$SLOT] $A 种子 $S"
    sleep 40
done; done
wait
for S in 1 2; do for A in bias affine scalar kv; do
    tail -3 "$LOG/decbench_${A}_s$S.log" 2>/dev/null|grep -q Finished \
        && touch "$LOG/.done_dcs_${A}_s$S"
done; done
echo "$(date +%H:%M)  拆解四臂 × 种子 1/2 完成（8 作业）"
