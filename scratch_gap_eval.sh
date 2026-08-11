#!/bin/bash
# 评测 2026-08-09 16:06 训出的三个 ckpt（gap 目标 + 残差读出），此前从未评过。
#   ckpt_gap_fix03/dist   —— obj=gap, ratio 固定 0.3
#   ckpt_gap_rand/dist    —— obj=gap, ratio 每步随机
#   ckpt_gap_rand/point   —— 同上，point 对照
# 三者都带 residual_gate，必须 --varikv_residual。
# 两个区间：标准 5 比例（与已有 ret/rb 表可比）与低比例 0.1/0.05
# （eval.py:set_ratios 的注释：基线在 ratio>=0.3 已无损，可回收空间只在低比例）。
# 基线两档均已存在（ret_/rb_ 与 low_），不重跑。
set -u
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
PY=/home/ubuntu/zxy/vlm-memory/.venv/bin/python
CK=/home/ubuntu/zxy/vlm-memory/varikv
LOG=/home/ubuntu/zxy/vlm-memory/scratch_stage2b_logs
D=scbench_many_shot

run() {   # gpu tag ckpt ratios logname
  local gpu=$1 tag=$2 ckpt=$3 ratios=$4 name=$5
  (
    if [ -n "$ratios" ]; then export VARIKV_RATIOS="$ratios"; fi
    CUDA_VISIBLE_DEVICES=$gpu $PY -B eval_chunk.py \
      -g fastkvzip -m Qwen/Qwen2.5-7B-Instruct-1M -d $D \
      --tag "$tag" --varikv_ckpt "$ckpt" --varikv_slots 16 --varikv_residual \
      > "$LOG/$name.log" 2>&1
    echo "DONE $name $(date +%H:%M:%S)" >> /home/ubuntu/zxy/vlm-memory/scratch_gap_eval.log
  ) &
}

echo "START $(date)" > /home/ubuntu/zxy/vlm-memory/scratch_gap_eval.log

run 0 gapf  $CK/ckpt_gap_fix03/s2b_dist_k16.pt  ""          gapf_dist
run 1 gapr  $CK/ckpt_gap_rand/s2b_dist_k16.pt   ""          gapr_dist
run 2 gapr  $CK/ckpt_gap_rand/s2b_point_k16.pt  ""          gapr_point
run 3 gapfl $CK/ckpt_gap_fix03/s2b_dist_k16.pt  "0.1,0.05"  gapf_dist_low
run 4 gaprl $CK/ckpt_gap_rand/s2b_dist_k16.pt   "0.1,0.05"  gapr_dist_low
run 5 gaprl $CK/ckpt_gap_rand/s2b_point_k16.pt  "0.1,0.05"  gapr_point_low

wait
echo "ALL DONE $(date)" >> /home/ubuntu/zxy/vlm-memory/scratch_gap_eval.log
