#!/bin/bash
# 停掉 30 篇档的四臂种子扫（用户只要 v2 档）。marker 可续跑，随时能重开。
pkill -f 'scratch_dec_seeds[.]sh'
sleep 1
for p in $(pgrep -f 'eval_chunk[.]py.*_dc[a-z]*_s[12]'); do kill "$p" 2>/dev/null; done
sleep 4
for p in $(pgrep -f 'eval_chunk[.]py.*_dc[a-z]*_s[12]'); do kill -9 "$p" 2>/dev/null; done
sleep 2
for d in /tmp/varikv_gpulock/*; do
    [ -d "$d" ] || continue; g=$(basename "$d")
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g")" ] && rmdir "$d" 2>/dev/null
done
echo "剩余 30 篇档进程: $(pgrep -cf 'eval_chunk[.]py.*_dc[a-z]*_s[12]')"
