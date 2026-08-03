#!/usr/bin/env bash
# Wait for the 8 real reproduce shards (kv x3, prefix_suffix x3, many_shot x2) to finish,
# then run the parser for all three datasets and print the near-lossless summary.
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
source /home/ubuntu/zxy/vlm-memory/.venv/bin/activate
LOGDIR=/home/ubuntu/zxy/vlm-memory/scratch/repro_0730_qwen3/repro_logs
LOGS=$(ls "$LOGDIR"/scbench_kv_*.log "$LOGDIR"/scbench_prefix_suffix_*.log "$LOGDIR"/scbench_many_shot_*.log)
N=$(echo "$LOGS" | wc -l)
echo "waiting for $N shards to finish..."

while :; do
  done_cnt=0; crash_cnt=0
  for f in $LOGS; do
    if grep -q "Finished" "$f" 2>/dev/null; then done_cnt=$((done_cnt+1))
    elif grep -qE "Traceback|OutOfMemoryError|CUDA out of memory|Killed" "$f" 2>/dev/null; then crash_cnt=$((crash_cnt+1)); fi
  done
  [ $((done_cnt+crash_cnt)) -ge "$N" ] && break
  sleep 20
done
echo "SHARDS_SETTLED done=$done_cnt crash=$crash_cnt : $(date)"

TAG=qwen3-8b_fastkvzip_chunk16k_w4096
for d in scbench_kv_short scbench_prefix_suffix_short scbench_many_shot; do
  echo "============================================================"
  echo "PARSE $d"
  echo "============================================================"
  python -B -m results.parse -m "$TAG" -d "$d" 2>&1 | grep -vE "include_score" | tail -25
done
echo "PARSE_ALL_DONE"
