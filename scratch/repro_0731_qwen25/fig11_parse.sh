#!/usr/bin/env bash
# 等 Figure 11 子集跑完，然后按 (数据集 × 方法) 解析出准确率随压缩比的曲线。
# 论文 Figure 11 的比较对象是「5 个方法在同一压缩比下的相对高低」，所以按数据集分组打印。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
source /home/ubuntu/zxy/vlm-memory/.venv/bin/activate

# 等调度器进程退出
while pgrep -f "scratch_repro_full.py --run" >/dev/null; do sleep 60; done
echo "SCHEDULER_DONE: $(date)"
echo

M=qwen2.5-7b-instruct-1m
for d in squad scbench_many_shot scbench_prefix_suffix; do
  echo "############################################################"
  echo "# $d   (压缩比顺序: 1.0 0.75 0.5 0.4 0.3 0.2)"
  echo "############################################################"
  for spec in "fastkvzip:_fastkvzip_chunk16k_w4096:pair" \
              "kvzip::pair" \
              "duoattn:_head_chunk16k_w4096:pair" \
              "expected:_expect_chunk16k_w4096:adakv-layer" \
              "snapkv:_snap_chunk16k_w4096:pair-head"; do
    label="${spec%%:*}"; rest="${spec#*:}"; tag="${rest%%:*}"; level="${rest##*:}"
    echo "--- $label ---"
    python -B -m results.parse -m "${M}${tag}" -d "$d" 2>&1 \
      | grep -vE "include_score|^\[" | tail -18
  done
  echo
done
echo "PARSE_ALL_DONE"
