#!/bin/bash
set -u
ROOT=/home/ubuntu/zxy/vlm-memory; LOG=$ROOT/scratch_ctrl_logs; Q=$LOG/.mq_jobs
J=()
for A in bias affine scalar kv; do
    J+=("cd $ROOT && $ROOT/.venv/bin/python -B scratch_probe_armdiag.py \
--ckpt $ROOT/varikv/d10_${A}_s0.pt/memoryless.pt --num 3 > $LOG/armdiag_d10_$A.log 2>&1")
done
( flock 9; tmp=$(mktemp); printf '%s\n' "${J[@]}" > "$tmp"
  [ -f "$Q" ] && cat "$Q" >> "$tmp"; mv "$tmp" "$Q" ) 9>"$Q.lock"
echo "插入 ${#J[@]} 个 armdiag（v2 档）"
