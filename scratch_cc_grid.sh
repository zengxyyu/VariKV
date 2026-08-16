#!/bin/bash
# 组合臂：质心读出 + 学习残差改分同时开。11 panel × 8 ratio，与 RESULTS_GRID.md 同基线。
# **最有信息量的四个 panel 排前面**，这样不必等全部跑完就能判断交互是超可加还是次可加：
#   Retr.KV        残差峰在 0.2(+18.80)、质心峰在 0.1(+11.00) —— 两个峰能否同时拿到
#   Retr.MultiHop  两条线同向失败(−9.96 / −6.53) —— 叠加是更糟还是抵消
#   Code.RepoQA    质心强(+9.09★)、残差弱(+0.91) —— 质心主导时残差会不会拖后腿
#   Retr.PrefSuf   两条都中等
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT/external/FastKVzip/prefill" || exit 1
LOG="$ROOT/scratch_ctrl_logs"
. "$ROOT/scratch_gpu_lock.sh"   # 跨调度器 GPU 锁
RATIOS=1.0,0.75,0.5,0.4,0.3,0.2,0.1,0.05
CK=../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt
DATASETS="scbench_kv scbench_vt scbench_repoqa scbench_prefix_suffix \
scbench_mf scbench_qa_eng scbench_choice_eng scbench_summary \
scbench_many_shot squad gsm"
# **K=16 才是等预算的那一档。** 质心开销 K=1024 是 6.08%、K=16 只有 0.095%
# （CLAUDE.md 的核算），所以只有 K=16 时四条臂（base / residual / centroid /
# combined）才在同一个字节预算上，`Combined − Base` 才是公平比较。
# K=1024 那档仍然有用，但只能拿来比 `Combined − Centroid`（两者预算相同）。
KLIST="16 1024"

declare -A PID=() LAST=()
for K in $KLIST; do
for D in $DATASETS; do
    [ -f "$LOG/.done_cc_${D}_$K" ] && { echo "跳过 $D K=$K"; continue; }
    # 用共享锁抢卡，不再自维护槽位表 —— 三个调度器同时扫 0-7 会把作业叠到
    # 同一张卡直到 OOM（已发生）。同时回收上一个作业的锁。
    for G in 0 1 2 3 4 5 6 7; do
        P=${PID[$G]:-}
        if [ -n "$P" ] && ! kill -0 "$P" 2>/dev/null; then
            LD=${LAST[$G]:-}
            [ -n "$LD" ] && tail -3 "$LOG/ccgrid_$LD.log" 2>/dev/null \
                | grep -q Finished && touch "$LOG/.done_cc_$LD"
            gpu_release "$G"; unset "PID[$G]"
        fi
    done
    SLOT=$(gpu_claim)
    VARIKV_RATIOS=$RATIOS CUDA_VISIBLE_DEVICES=$SLOT nohup "$ROOT/.venv/bin/python" -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d "$D" \
        --tag "_cc$K" --centroid_k "$K" --ctrlm_ckpt "$CK" \
        > "$LOG/ccgrid_${D}_$K.log" 2>&1 &
    PID[$SLOT]=$!; LAST[$SLOT]="${D}_$K"
    echo "$(date +%H:%M)  [GPU$SLOT] $D 组合臂 K=$K"
    sleep 40
done; done
wait
for K in $KLIST; do for D in $DATASETS; do
    tail -3 "$LOG/ccgrid_${D}_$K.log" 2>/dev/null | grep -q Finished \
        && touch "$LOG/.done_cc_${D}_$K"
done; done
echo "$(date +%H:%M)  组合臂 11 个作业完成"
