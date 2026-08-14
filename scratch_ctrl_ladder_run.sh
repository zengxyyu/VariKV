#!/bin/bash
# A→E 分段短接阶梯：6 变体 × 5 种子，每变体独占一张卡。
# 单独成脚本，是因为在 bash 工具里直接写 pkill/pgrep 会匹配到工具自己的命令行
# （`bash -c` 的 argv 里含有同样的字符串），把发起命令的 shell 一起杀掉。
cd /home/ubuntu/zxy/vlm-memory || exit 1
mkdir -p scratch_ctrl_logs
i=0
for V in raw_oracle proj_oracle ema_direct read_exact full full_mean; do
    CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python -u scratch_ctrl_dirseed.py \
        --variant "$V" --seeds 0 1 2 3 4 \
        > "scratch_ctrl_logs/ladder_$V.log" 2>&1 &
    i=$((i + 1))
done
wait
