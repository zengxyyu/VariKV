#!/bin/bash
# 按用户要求：清空 d10 四臂的全部产物，从头重跑。
pkill -f 'scratch_d10_arms[.]sh'
sleep 1
for p in $(pgrep -f 'ctrl_train[.]py.*traces_v2_10'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f 'ctrl_train[.]py.*traces_v2_10'); do kill -9 "$p" 2>/dev/null; done
sleep 2
ROOT=/home/ubuntu/zxy/vlm-memory
echo "删除 ckpt:";  rm -rfv $ROOT/varikv/d10_*.pt 2>/dev/null | wc -l
echo "删除训练日志:"; rm -fv $ROOT/scratch_ctrl_logs/d10_*_s*.log 2>/dev/null | wc -l
echo "删除评测日志与 marker:"; rm -fv $ROOT/scratch_ctrl_logs/d10bench_*.log \
    $ROOT/scratch_ctrl_logs/.done_d10_* 2>/dev/null | wc -l
echo "删除评测结果目录:"; rm -rf $ROOT/external/FastKVzip/prefill/results/scbench_kv/*_d10*; echo ok
for d in /tmp/varikv_gpulock/*; do
    [ -d "$d" ] || continue; g=$(basename "$d")
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g")" ] && rm -rf "$d"
done
echo "残留 d10 文件: $(ls -d $ROOT/varikv/d10_* 2>/dev/null | wc -l)"
echo "残留训练进程: $(pgrep -cf 'ctrl_train[.]py')"
echo "锁: $(ls /tmp/varikv_gpulock 2>/dev/null | tr '\n' ' ')"
