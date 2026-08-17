#!/bin/bash
for p in $(pgrep -f 'eval_chunk[.]py.*_d10'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f 'eval_chunk[.]py.*_d10'); do kill -9 "$p" 2>/dev/null; done
sleep 3
for d in /tmp/varikv_gpulock/*; do [ -d "$d" ] || continue; g=$(basename "$d")
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g)" ] && rm -rf "$d"; done
cd /home/ubuntu/zxy/vlm-memory
setsid ./scratch_master_queue.sh > scratch_ctrl_logs/master_run.log 2>&1 < /dev/null &
disown
