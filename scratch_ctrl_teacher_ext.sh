#!/bin/bash
# 把三个任务变体都扩到 30 篇长文档（前 10 篇与既有 trace 逐字节相同，断点续传会跳过）。
# n=6 时可检出效应 0.033 而 n_dup=3 的点估计是 0.016 —— 功效不足，不是证据为零。
cd /home/ubuntu/zxy/vlm-memory || exit 1
while pgrep -f "ctrl_teacher.py --task retrieval" >/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --n_cat 30 --out scratch_ctrl_traces_v2 \
    >> scratch_ctrl_logs/teacher_v2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --task retrieval --n_dup 1 --n_cat 30 --out scratch_ctrl_traces_retr1 \
    >> scratch_ctrl_logs/teacher_retr1.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup .venv/bin/python -u scratch_ctrl_teacher.py \
    --task retrieval --n_dup 3 --n_cat 30 --out scratch_ctrl_traces_retr3 \
    >> scratch_ctrl_logs/teacher_retr3.log 2>&1 &
wait
echo "=== 三个教师扩容完成，跑凸探针 ==="
for T in v2 retr1 retr3; do
    echo "════════ $T (n=30) ════════"
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scratch_probe_histinfo.py \
        --traces "scratch_ctrl_traces_$T"
done
