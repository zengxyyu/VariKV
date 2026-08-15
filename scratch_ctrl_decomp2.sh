#!/bin/bash
# 打分器拆解：与 GPU 0/1 上的 ratio 评测**共卡**。训练约 7GB、评测约 30GB，
# 80GB 的卡塞得下，没必要为了独占而串行等几小时。
# 两条通道（GPU 0 / GPU 1）各自串行，避免同卡堆太多。
cd /home/ubuntu/zxy/vlm-memory || exit 1
lane() {   # $1=gpu  $2..=任务列表 "arch:seed"
    local G=$1; shift
    for J in "$@"; do
        A=${J%%:*}; S=${J##*:}
        O="varikv/dec_${A}_s$S.pt"
        [ -d "$O" ] && continue
        CUDA_VISIBLE_DEVICES=$G .venv/bin/python -u scratch_ctrl_train.py \
            --traces scratch_ctrl_traces_v2 --epochs 40 --seed "$S" --split_seed 42 \
            --arch "$A" --alpha_init 1.0 --freeze_alpha --pair_w linear \
            --lam_global 1.0 --out "$O" \
            > "scratch_ctrl_logs/dec_${A}_s$S.log" 2>&1
    done
}
lane 0 affine:0 affine:1 affine:2 scalar:0 scalar:1 scalar:2 &
lane 1 bias:0 bias:1 bias:2 kv:0 kv:1 kv:2 &
wait
echo "12 个拆解训练完成"
