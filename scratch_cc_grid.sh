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
RATIOS=1.0,0.75,0.5,0.4,0.3,0.2,0.1,0.05
CK=../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt
DATASETS="scbench_kv scbench_vt scbench_repoqa scbench_prefix_suffix \
scbench_mf scbench_qa_eng scbench_choice_eng scbench_summary \
scbench_many_shot squad gsm"

declare -A PID=() LAST=()
for D in $DATASETS; do
    [ -f "$LOG/.done_cc_$D" ] && { echo "跳过 $D"; continue; }
    while true; do
        SLOT=""
        for G in 0 1 2 3 4 5 6 7; do
            P=${PID[$G]:-}
            if [ -z "$P" ] || ! kill -0 "$P" 2>/dev/null; then
                LD=${LAST[$G]:-}
                [ -n "$LD" ] && tail -3 "$LOG/ccgrid_$LD.log" 2>/dev/null \
                    | grep -q Finished && touch "$LOG/.done_cc_$LD"
                SLOT=$G; break
            fi
        done
        [ -n "$SLOT" ] && break
        sleep 60
    done
    VARIKV_RATIOS=$RATIOS CUDA_VISIBLE_DEVICES=$SLOT nohup "$ROOT/.venv/bin/python" -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d "$D" \
        --tag "_cc" --centroid_k 1024 --ctrlm_ckpt "$CK" \
        > "$LOG/ccgrid_$D.log" 2>&1 &
    PID[$SLOT]=$!; LAST[$SLOT]="$D"
    echo "$(date +%H:%M)  [GPU$SLOT] $D 组合臂"
    sleep 40
done
wait
for D in $DATASETS; do
    tail -3 "$LOG/ccgrid_$D.log" 2>/dev/null | grep -q Finished && touch "$LOG/.done_cc_$D"
done
echo "$(date +%H:%M)  组合臂 11 个作业完成"
