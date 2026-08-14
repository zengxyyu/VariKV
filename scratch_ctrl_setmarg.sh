#!/bin/bash
# 条件于存活集合的边际效用教师，三个任务变体各 30 篇。
cd /home/ubuntu/zxy/vlm-memory || exit 1
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --utility set_marginal --n_cat 30 --out scratch_ctrl_traces_sm_cont \
    > scratch_ctrl_logs/teacher_sm_cont.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --utility set_marginal --task retrieval --n_dup 1 --n_cat 30 \
    --out scratch_ctrl_traces_sm_r1 > scratch_ctrl_logs/teacher_sm_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --utility set_marginal --task retrieval --n_dup 3 --n_cat 30 \
    --out scratch_ctrl_traces_sm_r3 > scratch_ctrl_logs/teacher_sm_r3.log 2>&1 &
wait
echo "=== 教师完成，跑凸探针 ==="
for T in sm_cont sm_r1 sm_r3; do
    echo "════════ $T ════════"
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scratch_probe_histinfo.py \
        --traces "scratch_ctrl_traces_$T"
done
