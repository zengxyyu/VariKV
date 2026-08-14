#!/bin/bash
# VariKV-B：固定 train/val 划分（split_seed 42），3 个种子只变初始化/采样/顺序。
# α 冻结在 1.0：自学时 40 epoch 只从 0.050 爬到 0.0555，Δs 满幅仅为近阈值池内
# 典型 |Δs0| 的 12%，只有 24% 的成对翻得动 —— 判据被构造性封顶。
cd /home/ubuntu/zxy/vlm-memory || exit 1
i=0
for S in 0 1 2; do
    G=$(echo "0 1 3" | cut -d' ' -f$((i + 1)))
    CUDA_VISIBLE_DEVICES=$G nohup .venv/bin/python -u scratch_ctrl_train.py \
        --traces scratch_ctrl_traces_v2 --epochs 40 \
        --seed "$S" --split_seed 42 \
        --alpha_init 1.0 --freeze_alpha \
        --pair_w linear --lam_global 1.0 \
        --out "varikv/ctrl_b_a1_s$S.pt" \
        > "scratch_ctrl_logs/train_b_a1_s$S.log" 2>&1 &
    i=$((i + 1))
done
wait
