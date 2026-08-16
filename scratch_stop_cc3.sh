#!/bin/bash
# 停组合臂（质心线，用户已决定不再做）+ 清掉上一轮调度器留下的孤儿 eval。
# **必须写成脚本文件**：把 pkill 的模式写在 bash 工具的命令行里，模式会匹配到
# 工具自己那条命令行，把发起的 shell 一起杀掉 —— 本会话已经踩过 4 次。
pkill -f 'scratch_cc_grid[.]sh'
sleep 1
for p in $(pgrep -f 'eval_chunk[.]py.*_cc16'); do kill "$p" 2>/dev/null; done
sleep 4
for p in $(pgrep -f 'eval_chunk[.]py.*_cc16'); do kill -9 "$p" 2>/dev/null; done
sleep 2
# 释放锁：只删没有进程占着的那些，别动别人的
for d in /tmp/varikv_gpulock/*; do
    [ -d "$d" ] || continue
    g=$(basename "$d")
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g")" ] \
        && rmdir "$d" 2>/dev/null && echo "释放 GPU$g 的锁"
done
echo "剩余 _cc16 进程: $(pgrep -cf 'eval_chunk[.]py.*_cc16')"
