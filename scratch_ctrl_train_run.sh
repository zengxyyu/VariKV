#!/bin/bash
# VariKV-B 真实训练：3 个种子。单次训练不是一次测量（v1 的 +21.60 是这么来的）。
cd /home/ubuntu/zxy/vlm-memory || exit 1
i=0
for S in 0 1 2; do
    G=$([ $i -eq 0 ] && echo 0 || ([ $i -eq 1 ] && echo 1 || echo 3))
    CUDA_VISIBLE_DEVICES=$G nohup .venv/bin/python -u scratch_ctrl_train.py \
        --traces scratch_ctrl_traces_v2 --epochs 40 --seed "$S" \
        --pair_w linear --lam_global 1.0 \
        --out "varikv/ctrl_b_s$S.pt" > "scratch_ctrl_logs/train_b_s$S.log" 2>&1 &
    i=$((i + 1))
done
wait
