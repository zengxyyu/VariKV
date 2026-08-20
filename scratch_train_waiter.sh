#!/bin/bash
# 等到某张卡**完全没有 compute 进程**再放一个训练上去，严格一卡一进程。
#
# 判空规则（项目纪律）：用 --query-compute-apps 的**进程存在性**，不用显存；
# 且必须**相隔 20 秒的两次读数都为空**，避免评测作业在生成阶段的瞬时空档被误判。
set -u
cd "$(dirname "$0")"
BASE="--arch chead --d_kv 128 --dim 128 --epochs 40 --freeze_alpha --lam_global 1.0 \
--lr 0.0003 --n_pairs 256 --pair_w linear --slots 8 --split_seed 42 \
--traces scratch_ctrl_traces_v2_10 --val_frac 0.25 --scale global \
--alpha_max 1.0 --alpha_init 0.999"

busy() {   # busy <gpu_idx> -> 0 表示有进程
    local uuid; uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
                       | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -qF "$uuid"
}

CAND="${CAND:-0 1 2 3 4 5 6 7}"
for s in 0 1 2; do
    out="varikv/chead10_s$s.pt"
    [ -e "$out" ] && { echo "[skip] $out 已存在"; continue; }
    placed=0
    while [ $placed -eq 0 ]; do
        for g in $CAND; do
            busy "$g" && continue
            sleep 20
            busy "$g" && continue          # 20 秒后仍为空才认
            echo "[launch] seed $s -> GPU$g  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python -u \
                scratch_ctrl_train.py $BASE --seed "$s" --out "$out" \
                > "scratch_ctrl_logs/train_chead10_s$s.log" 2>&1 &
            sleep 30                        # 让它占住卡，避免下一轮重复选中
            placed=1; break
        done
        [ $placed -eq 0 ] && sleep 60
    done
done
echo "[done] 三个 chead10 全部已投放 $(date +%H:%M:%S)"
