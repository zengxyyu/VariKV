#!/bin/bash
# 缺的那一格：正确靶子(set_marginal) × typed 架构 × α 冻结 1.0 × 三臂 × 多种子。
# stateful−shuffled 对"池化线性模型跨文档权重不稳"这条批评免疫：两臂结构与参数
# 完全一致，只差内容归属。
cd /home/ubuntu/zxy/vlm-memory || exit 1
i=0
for S in 0 1 2; do
    G=$(echo "0 1 2" | cut -d' ' -f$((i + 1)))
    CUDA_VISIBLE_DEVICES=$G nohup .venv/bin/python -u scratch_ctrl_train.py \
        --traces scratch_ctrl_traces_sm_r3 --epochs 40 --seed "$S" --split_seed 42 \
        --alpha_init 1.0 --freeze_alpha --pair_w linear --lam_global 1.0 \
        --out "varikv/ctrl_smr3_s$S.pt" > "scratch_ctrl_logs/train_smr3_s$S.log" 2>&1 &
    i=$((i + 1))
done
i=0
for S in 0 1; do
    G=$(echo "6 7" | cut -d' ' -f$((i + 1)))
    CUDA_VISIBLE_DEVICES=$G nohup .venv/bin/python -u scratch_ctrl_train.py \
        --traces scratch_ctrl_traces_sm_cont --epochs 40 --seed "$S" --split_seed 42 \
        --alpha_init 1.0 --freeze_alpha --pair_w linear --lam_global 1.0 \
        --out "varikv/ctrl_smc_s$S.pt" > "scratch_ctrl_logs/train_smc_s$S.log" 2>&1 &
    i=$((i + 1))
done
wait
