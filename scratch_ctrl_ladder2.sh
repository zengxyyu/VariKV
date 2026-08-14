#!/bin/bash
cd /home/ubuntu/zxy/vlm-memory || exit 1
i=0
for V in read_dironly read_signed; do
    CUDA_VISIBLE_DEVICES=$((i)) nohup .venv/bin/python -u scratch_ctrl_dirseed.py \
        --variant "$V" --seeds 0 1 2 3 4 > "scratch_ctrl_logs/ladder_$V.log" 2>&1 &
    i=$((i + 1))
done
wait
