#!/bin/bash
# 补齐 gap 目标三个 ckpt 在 scbench_kv（论文 Retr.KV）标准区间的结果。
#
# 为什么要补：昨晚 (scratch_kvres_eval.sh) 只给这三个跑了 0.1/0.05，理由之一是
# 「门关着的档已由 rbkv 证明逐字等于基线」—— 那条站不住：rbkv 加载的是
# ckpt_stage2b_retain，门停在**初值** 0.018 且从未训过；这三个的门是训过的
# （0.014/0.024/0.032，max 0.26~0.40，4~12% 的 head-group 超过 0.1），实测
# gapf 在 0.1 处是 30.60 而基线 32.60，本就不逐字相同。
# 所以标准区间对这三个是真缺数据，而且 0.2 是论文横轴内唯一还有 23 个绝对分
# 空间的位置（基线 45.20 vs 满缓存 68.20）。
#
# 比例：不设 VARIKV_RATIOS，走 eval.py:set_ratios 的默认 [0.75,0.5,0.4,0.3,0.2]，
# 与已有的 rb 基线口径完全一致（解析端 parse 会自动补 1.0 参照行）。
# 基线不重跑（rb tag 已有 100 条）。
#
# tag 必须逐 ckpt 区分：gap_fix03/dist 与 gap_rand/dist 都是 dist 模式，
# 结果目录名只带 mode 不带 ckpt 名，同 tag 会互相覆盖。
# 首字符不能是下划线，否则目录出现双下划线、手搓 -m 串会静默解析出 0 条。
#
# 成本：昨晚实测标准区间 7h07m / 低比例(2档) 3h01m ⇒ 每档约 1.4h、固定开销约
# 0.3h，本次每 job 约 7 小时，三个并行占 3 张卡。
set -u
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
PY=/home/ubuntu/zxy/vlm-memory/.venv/bin/python
CK=/home/ubuntu/zxy/vlm-memory/varikv
LOG=/home/ubuntu/zxy/vlm-memory/scratch_stage2b_logs
STAMP=/home/ubuntu/zxy/vlm-memory/scratch_gapstd_eval.log
D=scbench_kv

run() {   # gpu tag ckpt logname
  local gpu=$1 tag=$2 ckpt=$3 name=$4
  (
    CUDA_VISIBLE_DEVICES=$gpu $PY -B eval_chunk.py \
      -g fastkvzip -m Qwen/Qwen2.5-7B-Instruct-1M -d $D \
      --tag "$tag" --varikv_ckpt "$ckpt" --varikv_slots 16 --varikv_residual \
      > "$LOG/$name.log" 2>&1
    echo "DONE $name rc=$? $(date +%H:%M:%S)" >> "$STAMP"
  ) &
}

echo "START $(date)" > "$STAMP"

run 0 gfsd $CK/ckpt_gap_fix03/s2b_dist_k16.pt  gapstd_fix_dist
run 1 grsd $CK/ckpt_gap_rand/s2b_dist_k16.pt   gapstd_rand_dist
run 2 grsp $CK/ckpt_gap_rand/s2b_point_k16.pt  gapstd_rand_point

wait
echo "ALL DONE $(date)" >> "$STAMP"
