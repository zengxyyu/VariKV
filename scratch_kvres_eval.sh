#!/bin/bash
# 残差版在 scbench_kv（检索密集，169k 上下文）上的评测。
#
# 为什么是这个数据集：many_shot 是高冗余类，压缩摘要本来最难显价值，且基线
# 在 ratio>=0.3 已无损、天花板只有 5.93 分。scbench_kv 相反 —— 基线在 0.2
# 处相对性能仅 66.28，真有可回收空间；而旧的 KV 注入版在这里崩到 0.29/0.00，
# 是全部数据集里最惨的一格。残差版从未在此跑过。
#
# 取舍（每作业约 7 GPU-小时，100 条 × ~260s）：
#   标准区间只跑门**开**的那对（ckpt_stage2b_res, lm 目标, sigmoid 0.186/0.287）
#   —— 门关的几档在标准区间已由 rbkv 证明逐字等于基线，再跑无信息；
#      且标准区间基线本身几乎无损（0.3 处 95.89），没有可回收空间。
#   低比例 0.1/0.05 全部档位都跑，含基线（该区间尚无 scbench_kv 基线）。
#
# tag 必须逐 ckpt 区分：三个 dist 档若同 tag 会写进同一个结果目录互相覆盖
# （目录名只带 mode，不带 ckpt 名）。
set -u
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
PY=/home/ubuntu/zxy/vlm-memory/.venv/bin/python
CK=/home/ubuntu/zxy/vlm-memory/varikv
LOG=/home/ubuntu/zxy/vlm-memory/scratch_stage2b_logs
STAMP=/home/ubuntu/zxy/vlm-memory/scratch_kvres_eval.log
D=scbench_kv

run() {   # gpu tag ckpt ratios logname
  local gpu=$1 tag=$2 ckpt=$3 ratios=$4 name=$5
  (
    if [ -n "$ratios" ]; then export VARIKV_RATIOS="$ratios"; fi
    local extra=""
    if [ -n "$ckpt" ]; then
      extra="--varikv_ckpt $ckpt --varikv_slots 16 --varikv_residual"
    fi
    CUDA_VISIBLE_DEVICES=$gpu $PY -B eval_chunk.py \
      -g fastkvzip -m Qwen/Qwen2.5-7B-Instruct-1M -d $D \
      --tag "$tag" $extra > "$LOG/$name.log" 2>&1
    echo "DONE $name rc=$? $(date +%H:%M:%S)" >> "$STAMP"
  ) &
}

echo "START $(date)" > "$STAMP"

# 标准区间（0.75→0.2）—— 只跑门开的一对，基线复用已有的 rb tag（100 条，同配置）
run 0 kvres  $CK/ckpt_stage2b_res/s2b_dist_k16.pt   ""         kvres_dist
run 1 kvres  $CK/ckpt_stage2b_res/s2b_point_k16.pt  ""         kvres_point

# 低比例 0.1 / 0.05 —— 全档位 + 该区间的基线
run 2 kvlb   ""                                     "0.1,0.05" kvl_baseline
run 3 kvlres $CK/ckpt_stage2b_res/s2b_dist_k16.pt   "0.1,0.05" kvl_res_dist
run 4 kvlres $CK/ckpt_stage2b_res/s2b_point_k16.pt  "0.1,0.05" kvl_res_point
run 5 kvlgf  $CK/ckpt_gap_fix03/s2b_dist_k16.pt     "0.1,0.05" kvl_gapf_dist
run 6 kvlgr  $CK/ckpt_gap_rand/s2b_dist_k16.pt      "0.1,0.05" kvl_gapr_dist
run 7 kvlgr  $CK/ckpt_gap_rand/s2b_point_k16.pt     "0.1,0.05" kvl_gapr_point

wait
echo "ALL DONE $(date)" >> "$STAMP"
