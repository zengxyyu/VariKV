#!/usr/bin/env bash
# Figure 11 全量复现的收尾驱动：等主体 40 个作业跑完 → 跑 MRCR 5 个作业 → 解析全部 12 个数据集。
#
# 为什么 MRCR 要单独一步：它的数据格式不经 DataWrapper、打分用 SequenceMatcher、
# 结果由 results/parse_mrcr.py（而非 results.parse）解析，且需换用专用脚本
# （eval_chunk_mrcr.py / 本地新增的 eval_mrcr.py），见 scratch_repro_full.py:MRCR_SCRIPT。
# 主体跑的时候 8 张卡已占满，所以 MRCR 只能排在后面。
#
# 取代 scratch_fig11_parse_all.sh（那个只解析 11 个数据集、且不跑 MRCR）。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
PY=$ROOT/.venv/bin/python
cd "$ROOT"

echo "DRIVER_START: $(date)"

# ---------- 1. 等主体调度器退出 ----------
while pgrep -f "scratch_repro_full.py --run" >/dev/null; do sleep 120; done
echo "MAIN_SCHEDULER_DONE: $(date)"

# ---------- 2. 跑 MRCR（Figure 11 的第 12 个数据集）----------
echo "MRCR_START: $(date)"
$PY scratch_repro_full.py --run \
    --models Qwen/Qwen2.5-7B-Instruct-1M --datasets mrcr
echo "MRCR_DONE: $(date)"

# ---------- 3. 解析全部 12 个数据集 ----------
cd "$ROOT/external/FastKVzip/prefill"
source "$ROOT/.venv/bin/activate"

M=qwen2.5-7b-instruct-1m
# Qwen2.5-7B-1M 不触发 _short/_mid 替换，所以这里用的就是数据集原名。
DATASETS="gsm squad scbench_many_shot scbench_repoqa scbench_qa_eng scbench_choice_eng \
          scbench_prefix_suffix scbench_summary scbench_vt scbench_mf scbench_kv"
SPECS="fastkvzip:_fastkvzip_chunk16k_w4096:pair
kvzip::pair
duoattn:_head_chunk16k_w4096:pair
expected:_expect_chunk16k_w4096:adakv-layer
snapkv:_snap_chunk16k_w4096:pair-head"

for d in $DATASETS; do
  echo "############################################################"
  echo "# $d   (压缩比顺序: 1.0 0.75 0.5 0.4 0.3 0.2)"
  echo "############################################################"
  while IFS= read -r spec; do
    label="${spec%%:*}"; rest="${spec#*:}"; tag="${rest%%:*}"; level="${rest##*:}"
    echo "--- $label ---"
    python -B -m results.parse -m "${M}${tag}" -d "$d" 2>&1 \
      | grep -vE "include_score|^\[" | tail -18
  done <<< "$SPECS"
  echo
done

# MRCR 走独立解析器：输出是 {ratio: 平均分} 的 JSON，不是 results.parse 的分行格式。
echo "############################################################"
echo "# mrcr   (results/parse_mrcr.py，输出为 {压缩比: 平均分})"
echo "############################################################"
while IFS= read -r spec; do
  label="${spec%%:*}"; rest="${spec#*:}"; tag="${rest%%:*}"; level="${rest##*:}"
  echo "--- $label ---"
  python -B -m results.parse_mrcr -m "${M}${tag}" -l "$level" 2>&1 | tail -12
done <<< "$SPECS"

echo
echo "PARSE_ALL_DONE: $(date)"
