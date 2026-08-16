#!/bin/bash
# 2×2 的两格：Retr.KV（残差涨 +4.4）与 Retr.MultiHop（残差跌 −9.96）。
# 若两者都是 D_ours < D_base，而分数一涨一跌，就得到
# 「更忠实于满缓存 ≠ 任务上更好」——这比任何分数曲线都硬。
ROOT=/home/ubuntu/zxy/vlm-memory
cd "$ROOT" || exit 1
for D in scbench_kv scbench_vt; do
    while true; do
        for G in 7 6 5 4 3 2 1 0; do
            [ "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader|wc -l)" -eq 0 ] \
                && { SLOT=$G; break 2; }
        done; sleep 90
    done
    CUDA_VISIBLE_DEVICES=$SLOT nohup .venv/bin/python -u scratch_probe_fidelity.py \
        -d "$D" --num 30 > "scratch_ctrl_logs/fid_$D.log" 2>&1 &
    sleep 60
done
wait
