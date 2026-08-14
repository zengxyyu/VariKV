#!/bin/bash
# **靶子对照**：同一架构、同一训练配置，只有教师的 U 定义不同。
#   U^full        = 满缓存单 token 移除损伤（per-token 内在量）→ ctrl_b_a1_s0
#   U^setmarginal = 固定预算下条件于存活集合的边际价值      → ctrl_smc_s0 / ctrl_smr3_s0
# 用 memoryless 臂：历史已按 TOST 判为实际为零，memoryless 才是干净的"学习打分器"。
# 命题：现有门控分对 U^setmarginal 的排序准确率只有 0.532（对 U^full 是 0.577），
# 若靶子换对能带来下游增益，则贡献在**蒸馏靶子的选择**上，与历史无关。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
freegpu() {  # 等到某张卡上没有计算进程
    while true; do
        for g in 0 1 2 3 4 5 6 7; do
            u=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)
            [ "$u" -eq 0 ] && { echo $g; return; }
        done
        sleep 120
    done
}
for CK in ctrl_smc_s0 ctrl_smr3_s0; do
    G=$(freegpu)
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$G nohup ../../../.venv/bin/python -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
        --tag "_tg${CK#ctrl_}" --ctrlm_mode memoryless \
        --ctrlm_ckpt "../../../varikv/$CK.pt/memoryless.pt" \
        > "../../../scratch_ctrl_logs/bench_tgt_$CK.log" 2>&1 &
    sleep 90
done
wait
