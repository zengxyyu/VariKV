#!/bin/bash
# ratio 0.05 上的 12 个 panel（Figure 11 全集）× {基线, 学习残差}。
#
# **只用 GPU 2-7 六张卡**，0 与 1 留给方法探索。
#
# 两条必须先知道的事实：
#
# 1. `scbench_many_shot` (26,474 tok) 与 `scbench_repoqa` (72,499 tok) 在 0.05 下
#    **构造性退化**：`ratio·clen` 分别是 1324 / 3625，都 ≤ window 4096，于是
#    `wrapper.py:275` 把 chunk_ratio 置 0、`valid` 恒为全 False，保留集**等于局部窗口**、
#    与门控分无关。这两格的基线与残差臂**必然逐位相同**，跑它们是为了留证据，
#    不是为了比较。有效 ratio 见下表（都远小于 0.05，因为窗口先占掉了预算）：
#       gsm 0.0388 / squad 0.0309 / prefix_suffix 0.0141 / summary 0.0158
#       choice_eng 0.0162 / qa_eng 0.0170 / vt 0.0177 / mf 0.0233 / kv 0.0265
#
# 2. 5 个数据集在 0.05 上**已有基线**（tag `_r05b`，含 0.05 与 0.1 两个 ratio）：
#    kv / vt / summary / prefix_suffix / repoqa。所以只补另外 6 个基线。
set -u
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
CKPT=../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt
LOG=../../../scratch_ctrl_logs
GPUS="2 3 4 5 6 7"

# 任务表："dataset|arm"。长的排前面（最长作业决定墙钟时间）。
JOBS=""
for D in scbench_kv scbench_mf scbench_vt scbench_qa_eng scbench_choice_eng \
         scbench_summary scbench_prefix_suffix scbench_repoqa scbench_many_shot \
         squad gsm; do
    JOBS="$JOBS $D|mem"
done
for D in scbench_mf scbench_qa_eng scbench_choice_eng scbench_many_shot squad gsm; do
    JOBS="$JOBS $D|base"          # 缺的 6 个基线
done
JOBS="$JOBS mrcr|base mrcr|mem"   # 第 12 个 panel，最贵，放最后

launch() {  # $1=dataset $2=arm $3=gpu
    local D=$1 A=$2 G=$3 SCRIPT=eval_chunk.py EXTRA="" TAG
    [ "$D" = mrcr ] && SCRIPT=eval_chunk_mrcr.py
    if [ "$A" = mem ]; then
        EXTRA="--ctrlm_mode memoryless --ctrlm_ckpt $CKPT"; TAG=_r5m
    else TAG=_r5b; fi
    VARIKV_RATIOS=0.05 CUDA_VISIBLE_DEVICES=$G nohup ../../../.venv/bin/python -B \
        $SCRIPT -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d "$D" \
        --tag "$TAG" $EXTRA > "$LOG/r005_${D}_${A}.log" 2>&1 &
    echo "  [$G] $D $A"
}

for J in $JOBS; do
    D=${J%%|*}; A=${J##*|}
    M="$LOG/.done_r005_${D}_${A}"
    [ -f "$M" ] && { echo "  跳过 $D $A"; continue; }
    while true; do                       # 等一张空卡
        FREE=""
        for G in $GPUS; do
            [ "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader|wc -l)" -eq 0 ] \
                && { FREE=$G; break; }
        done
        [ -n "$FREE" ] && break
        sleep 90
    done
    launch "$D" "$A" "$FREE"
    sleep 75                             # 让显存占用稳定后再判下一张卡
done
wait
# 完成标记按日志尾部的 Finished. 判定，不按结果文件计数——
# choice_eng 只有 18 条、many_shot 54 条、vt 90 条，计数法会永远重跑。
for J in $JOBS; do
    D=${J%%|*}; A=${J##*|}
    tail -3 "$LOG/r005_${D}_${A}.log" 2>/dev/null | grep -q "Finished" \
        && touch "$LOG/.done_r005_${D}_${A}"
done
echo "全部完成"
