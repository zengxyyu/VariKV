#!/bin/bash
# P0 自动链：等探针跑完 → 出报告 → 跑 top-head intervention。
# 见 NEXT_STEPS.md v5 §4（P0-B、P0-B'）。
set -u
cd /home/ubuntu/zxy/vlm-memory
PY=.venv/bin/python

for i in $(seq 1 180); do
  grep -q "DONE 恒等式自检全部通过" scratch_probe_damage.log 2>/dev/null && break
  sleep 60
done
echo "=== 探针状态 $(date) ==="
grep -c "问题" scratch_probe_damage.log

echo "=== P0-B 报告 ===" > scratch_p0_results.log
$PY -B scratch_probe_damage_report.py >> scratch_p0_results.log 2>&1

echo "" >> scratch_p0_results.log
echo "=== P0-B' top-head intervention $(date) ===" >> scratch_p0_results.log
CUDA_VISIBLE_DEVICES=1 $PY -B scratch_probe_intervene.py \
    --n_samples 5 --n_queries 3 --topk 5 \
    >> scratch_p0_results.log 2>&1
echo "CHAIN DONE $(date)" >> scratch_p0_results.log
