#!/bin/bash
cd /home/ubuntu/zxy/vlm-memory || exit 1
i=2
for V in read_typed full_typed; do
    CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python -u scratch_ctrl_dirseed.py \
        --variant "$V" --seeds 0 1 2 3 4 > "scratch_ctrl_logs/ladder_$V.log" 2>&1 &
    i=$((i + 4))          # GPU 2 和 6，避开正在跑的 0/1/3/4/5
done
wait
