#!/bin/bash
pkill -f 'scratch_pool[.]sh'; sleep 1
for p in $(pgrep -f 'eval_chunk[.]py.*_v2c_s'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f 'eval_chunk[.]py.*_v2c_s'); do kill -9 "$p" 2>/dev/null; done
sleep 3
R=/home/ubuntu/zxy/vlm-memory
rm -rf $R/external/FastKVzip/prefill/results/*/*_v2c_s*   # CVD 错乱下产出的，一律不要
rm -f $R/scratch_ctrl_logs/v2cbench_*.log $R/scratch_ctrl_logs/.done_v2c_*
for d in /tmp/varikv_gpulock/*; do
    [ -d "$d" ] || continue; g=$(basename "$d")
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g)" ] && rm -rf "$d"
done
echo "残留 v2c: $(pgrep -cf 'eval_chunk[.]py.*_v2c_s')  池子: $(pgrep -cf 'scratch_pool[.]sh')"
echo "锁: $(ls /tmp/varikv_gpulock 2>/dev/null|tr '\n' ' ')"
