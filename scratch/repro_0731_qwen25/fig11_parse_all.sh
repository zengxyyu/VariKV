#!/usr/bin/env bash
# 等 Figure 11 全量复现跑完，然后按 (数据集 × 方法) 解析准确率随压缩比的曲线。
# 覆盖 -d all 的全部 11 个数据集（Figure 11 的第 12 个是 MRCR，需 eval_chunk_mrcr.py，单独处理）。
# 取代只覆盖 3 个数据集的 scratch_fig11_parse.sh。
#
# 注意：Qwen2.5-7B-1M 不触发 _short/_mid 替换，所以 results.parse 用的就是数据集原名。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
source /home/ubuntu/zxy/vlm-memory/.venv/bin/activate

while pgrep -f "scratch_repro_full.py --run" >/dev/null; do sleep 120; done
echo "SCHEDULER_DONE: $(date)"
echo

M=qwen2.5-7b-instruct-1m
DATASETS="gsm squad scbench_many_shot scbench_repoqa scbench_mf scbench_kv \
          scbench_prefix_suffix scbench_summary scbench_vt scbench_choice_eng scbench_qa_eng"

for d in $DATASETS; do
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
