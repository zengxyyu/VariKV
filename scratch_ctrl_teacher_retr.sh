#!/bin/bash
# 检索型教师，两个变体。语料仍是 fineweb ⇒ 与续写版唯一的变量是查询类型。
cd /home/ubuntu/zxy/vlm-memory || exit 1
CUDA_VISIBLE_DEVICES=7 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --task retrieval --n_dup 1 --out scratch_ctrl_traces_retr1 \
    > scratch_ctrl_logs/teacher_retr1.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --task retrieval --n_dup 3 --out scratch_ctrl_traces_retr3 \
    > scratch_ctrl_logs/teacher_retr3.log 2>&1 &
wait
