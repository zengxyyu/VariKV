#!/bin/bash
cd /home/ubuntu/zxy/vlm-memory || exit 1
while pgrep -f "ctrl_teacher.py --task retrieval" >/dev/null; do sleep 20; done
for T in retr1 retr3; do
    echo "════════ $T ════════"
    CUDA_VISIBLE_DEVICES=7 .venv/bin/python -u scratch_probe_histinfo.py \
        --traces "scratch_ctrl_traces_$T"
done
