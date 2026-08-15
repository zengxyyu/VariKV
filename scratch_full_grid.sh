#!/bin/bash
# 全 ratio × 全 panel × {基线, v2, v3}。8 卡满载，有卡就上。
#
#   ratio  1.0 / 0.75 / 0.5 / 0.4 / 0.3 / 0.2 / 0.1 / 0.05
#   panel  11 个（Figure 11 除 MRCR）
#   臂     base            = 纯 FastKVzip
#          v2  ctrl_b_a1_s0/memoryless   教师靶子 U^full        (8 训练/2 验证篇)
#          v3  ctrl_smc_s0/memoryless    教师靶子 U^setmarginal (23 训练/7 验证篇)
#
# **v2 与 v3 同时差了靶子和语料规模**，所以两者之差不能单独归因给靶子；这张表的
# 用途是看**同一模型在 ratio 上的形状**，以及各自相对基线的曲线。
#
# 为什么 33 个作业而不是 33×8：`VARIKV_RATIOS` 在**一次预填**里算完所有 ratio，
# 只有生成阶段按 ratio 重复。所以加 ratio 很便宜，加数据集才贵。
#
# 调度：自己维护 8 个槽位并记录 PID，**不靠显存或进程数判断空闲**——
# CLAUDE.md 记着评测作业在生成阶段显存会掉到 2GB 以下、按显存判会重复派发；
# 而按进程数判会把同卡上的小训练误当成占用。槽位表两个问题都没有。
#
# 断点续传：按日志尾部的 `Finished.` 写标记。**不能按结果文件计数**——
# choice_eng 只有 18 条、qa_eng 20、many_shot 54、vt 90、repoqa 88，计数法会永远重跑。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT/external/FastKVzip/prefill" || exit 1
LOG="$ROOT/scratch_ctrl_logs"
RATIOS=1.0,0.75,0.5,0.4,0.3,0.2,0.1,0.05
declare -A CK=( [v2]="../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt"
                [v3]="../../../varikv/ctrl_smc_s0.pt/memoryless.pt" )

# 长的排前面：最长的作业决定墙钟时间。长度见 CLAUDE.md 的上下文表。
DATASETS="scbench_kv scbench_mf scbench_vt scbench_qa_eng scbench_choice_eng \
scbench_summary scbench_prefix_suffix scbench_repoqa scbench_many_shot squad gsm"

JOBS=()
for D in $DATASETS; do for A in base v2 v3; do JOBS+=("$D:$A"); done; done

declare -A PID=()
declare -A LAST=()      # 槽位 → 上一个作业名，用来在槽位释放时写完成标记
launch() {
    local D=${1%%:*} A=${1##*:} G=$2 EXTRA=""
    [ "$A" != base ] && EXTRA="--ctrlm_mode memoryless --ctrlm_ckpt ${CK[$A]}"
    VARIKV_RATIOS=$RATIOS CUDA_VISIBLE_DEVICES=$G nohup "$ROOT/.venv/bin/python" -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d "$D" \
        --tag "_g8$A" $EXTRA > "$LOG/grid_${D}_${A}.log" 2>&1 &
    PID[$G]=$!
    echo "$(date +%H:%M)  [GPU$G] $D $A"
}

for J in "${JOBS[@]}"; do
    D=${J%%:*}; A=${J##*:}
    [ -f "$LOG/.done_grid_${D}_${A}" ] && { echo "跳过 $D $A"; continue; }
    while true; do                                    # 等一个空槽
        SLOT=""
        for G in 0 1 2 3 4 5 6 7; do
            P=${PID[$G]:-}
            if [ -z "$P" ] || ! kill -0 "$P" 2>/dev/null; then
                # 槽位空了：若上一个作业跑完就写标记
                if [ -n "$P" ]; then
                    LD=${LAST[$G]:-}; [ -n "$LD" ] && tail -3 "$LOG/grid_$LD.log" 2>/dev/null \
                        | grep -q Finished && touch "$LOG/.done_grid_$LD"
                fi
                SLOT=$G; break
            fi
        done
        [ -n "$SLOT" ] && break
        sleep 60
    done
    LAST[$SLOT]="${D}_${A}"
    launch "$J" "$SLOT"
    sleep 40                                          # 错开模型加载
done
wait
for J in "${JOBS[@]}"; do
    D=${J%%:*}; A=${J##*:}
    tail -3 "$LOG/grid_${D}_${A}.log" 2>/dev/null | grep -q Finished \
        && touch "$LOG/.done_grid_${D}_${A}"
done
echo "$(date +%H:%M)  全部 33 个作业完成"
