#!/bin/bash
# 补齐质心：2 个 K × 11 panel × 8 ratio。**22 个作业而不是 176 格**——
# VARIKV_RATIOS 在一次预填里算完所有 ratio，只有生成阶段按 ratio 重复。
# 基线已有（__g8base 覆盖全 11 panel × 8 ratio），不用重跑。
# 跑完后 RESULTS_GRID.md 里质心那 12 个空格会被填上，两条线才在同一张表上完全可比。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT/external/FastKVzip/prefill" || exit 1
LOG="$ROOT/scratch_ctrl_logs"
. "$ROOT/scratch_gpu_lock.sh"   # 跨调度器 GPU 锁
RATIOS=1.0,0.75,0.5,0.4,0.3,0.2,0.1,0.05
DATASETS="scbench_kv scbench_mf scbench_vt scbench_qa_eng scbench_choice_eng \
scbench_summary scbench_prefix_suffix scbench_repoqa scbench_many_shot squad gsm"

declare -A PID=() LAST=()
for D in $DATASETS; do for K in 16 1024; do
    [ -f "$LOG/.done_cg_${D}_$K" ] && { echo "跳过 $D K=$K"; continue; }
    # 用共享锁抢卡，不再自维护槽位表 —— 三个调度器同时扫 0-7 会把作业叠到
    # 同一张卡直到 OOM（已发生）。同时回收上一个作业的锁。
    for G in 0 1 2 3 4 5 6 7; do
        P=${PID[$G]:-}
        if [ -n "$P" ] && ! kill -0 "$P" 2>/dev/null; then
            LD=${LAST[$G]:-}
            [ -n "$LD" ] && tail -3 "$LOG/cengrid_$LD.log" 2>/dev/null \
                | grep -q Finished && touch "$LOG/.done_cg_$LD"
            gpu_release "$G"; unset "PID[$G]"
        fi
    done
    SLOT=$(gpu_claim)
    VARIKV_RATIOS=$RATIOS CUDA_VISIBLE_DEVICES=$SLOT nohup "$ROOT/.venv/bin/python" -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d "$D" \
        --tag "_cg$K" --centroid_k "$K" > "$LOG/cengrid_${D}_${K}.log" 2>&1 &
    PID[$SLOT]=$!; LAST[$SLOT]="${D}_${K}"
    echo "$(date +%H:%M)  [GPU$SLOT] $D K=$K"
    sleep 40
done; done
wait
for D in $DATASETS; do for K in 16 1024; do
    tail -3 "$LOG/cengrid_${D}_${K}.log" 2>/dev/null | grep -q Finished \
        && touch "$LOG/.done_cg_${D}_${K}"
done; done
echo "$(date +%H:%M)  质心 22 个作业完成"
