#!/usr/bin/env bash
# 实测每个未测数据集的 per-example 耗时（每个 2 条，8 卡并行）。
# 目的：把完整复现的时间估计从"外推"换成"实测"。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
PY=/home/ubuntu/zxy/vlm-memory/.venv/bin/python
LOGDIR=/home/ubuntu/zxy/vlm-memory/scratch/probe/timing_logs
mkdir -p "$LOGDIR"
M="Qwen/Qwen3-8B"

probe() {  # gpu dataset
  CUDA_VISIBLE_DEVICES=$1 $PY -B eval_chunk.py -g fastkvzip -m "$M" -d "$2" \
    --idx 0 --num 2 --tag probe > "$LOGDIR/$2.log" 2>&1 &
}

echo "probe start: $(date)"
probe 0 squad
probe 1 gsm
probe 2 scbench_repoqa
probe 3 scbench_mf
probe 4 scbench_summary
probe 5 scbench_vt
probe 6 scbench_choice_eng
probe 7 scbench_qa_eng
wait
echo "probe done: $(date)"

echo
printf "%-26s %8s %10s\n" dataset tokens sec_per_ex
for f in "$LOGDIR"/*.log; do
  d=$(basename "$f" .log)
  tok=$(grep -oE "[0-9]+ tokens, KV cache" "$f" | head -1 | grep -oE "^[0-9]+")
  sec=$(grep -oE "## Time: [0-9.]+s" "$f" | awk '{gsub("s","",$3); s+=$3; n++} END {if(n>0) printf "%.1f", s/n; else print "FAIL"}')
  printf "%-26s %8s %10s\n" "$d" "${tok:-?}" "${sec:-FAIL}"
done
