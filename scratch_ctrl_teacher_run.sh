#!/bin/bash
cd /home/ubuntu/zxy/vlm-memory || exit 1
CUDA_VISIBLE_DEVICES=7 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --out scratch_ctrl_traces_v2 \
    > scratch_ctrl_logs/teacher_v2.log 2>&1 &
wait
