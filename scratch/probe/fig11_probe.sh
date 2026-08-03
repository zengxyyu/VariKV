#!/usr/bin/env bash
# Figure 11 复现前的耗时探测：Qwen2.5-7B-1M（论文主结果图用的模型）。
# 注意：该模型不触发 _short/_mid 替换，跑的是完整长上下文版本，比 Qwen3-8B 的 _short 版贵。
# 每个数据集 2 条，三行各取一个代表任务。顺带触发模型下载。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
PY=/home/ubuntu/zxy/vlm-memory/.venv/bin/python
LOGDIR=/home/ubuntu/zxy/vlm-memory/scratch/probe/fig11_probe_logs
mkdir -p "$LOGDIR"
M="Qwen/Qwen2.5-7B-Instruct-1M"

probe() {  # gpu dataset
  CUDA_VISIBLE_DEVICES=$1 $PY -B eval_chunk.py -g fastkvzip -m "$M" -d "$2" \
    --idx 0 --num 2 --tag f11probe > "$LOGDIR/$2.log" 2>&1 &
}

echo "probe start: $(date)"
probe 0 squad                  # Contextual QA 行
probe 1 scbench_many_shot      # ICL.ManyShot
probe 2 scbench_prefix_suffix  # Retrieval 行（完整版，非 _short）
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
