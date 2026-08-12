#!/usr/bin/env bash
# teacher-KL dist 在另外七个代表数据集上的评测（2026-08-12）。
#
# 为什么必须做：+21.60 那个结果**只在 scbench_kv 上**，而它是 11 个数据集里唯一
# 有大 headroom 的（ratio 0.1 时 35.6 分）。先例就在眼前 —— 三个 gap_* ckpt 在
# scbench_kv 上 null，在其余 9 个上也 null，一致；但"在最有利的格子上有效"完全
# 可能是单数据集现象。这一批把它变成 8 个格子。
#
# 七个数据集覆盖论文 Figure 11 的三个类别 + headroom 的两个极端：
#   prefix_suffix  retrieval,   headroom +10.80（次高）
#   gsm            contextual,  +7.00
#   choice_eng     contextual,  +6.95（n=18，小样本，别单独下结论）
#   many_shot      redundancy,  +4.82
#   repoqa         retrieval,   +0.91（最贵，先起）
#   vt             retrieval,   **−5.02（负 headroom！压缩反而更好）**
#                               ← 关键对照：记忆会不会在"本来不需要修"的地方添害
#   squad          contextual,  +0.56（近零 headroom 对照）
#
# 每个数据集跑两臂：teacher-KL dist（ckpt_kl/dist）与同批 ratio-0.1 纯基线。
# 不复用旧的 _full 基线 —— 那些跑的是 [1.0…0.2] 六档，**没有 0.1**。
#
# 成本：单档单比例，七个数据集 ≈ 3.1 GPU-h/臂，两臂 ≈ 7 GPU-h，4 卡并行 ≈ 1.8 h。
set -u
cd "$(dirname "$0")/external/FastKVzip/prefill" || exit 1
PY=../../../.venv/bin/python
LOG=../../../scratch_klsweep_logs
mkdir -p "$LOG"
CKPT=../../../varikv/ckpt_kl/s2b_dist_k16.pt
GPUS=(${GPUS:-4 5 6 7})

COMMON="-m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100
        --prefill_chunk 16000 --window_size 4096 --level pair"

# 长的先起（longest-first），否则最后会被 repoqa 拖尾
JOBS=(
  "scbench_repoqa kl" "scbench_repoqa base"
  "scbench_prefix_suffix kl" "scbench_prefix_suffix base"
  "scbench_vt kl" "scbench_vt base"
  "scbench_mf_SKIP kl"
  "scbench_many_shot kl" "scbench_many_shot base"
  "gsm kl" "gsm base"
  "scbench_choice_eng kl" "scbench_choice_eng base"
  "squad kl" "squad base"
)
JOBS=("${JOBS[@]/scbench_mf_SKIP kl/}")     # mf 不在这七个里

run_one () {   # run_one <gpu> <dataset> <arm>
  local gpu=$1 ds=$2 arm=$3
  local name="${ds}__${arm}"
  local f="$LOG/$name.log"
  [ -f "$LOG/.done_$name" ] && { echo "[skip] $name"; return 0; }
  local extra=""
  [ "$arm" = kl ] && extra="--varikv_ckpt $CKPT --varikv_residual --varikv_slots 16"
  VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$gpu $PY -B eval_chunk.py $COMMON \
      -d "$ds" --tag "_kls_$arm" $extra > "$f" 2>&1
  if grep -q "Finished." "$f"; then touch "$LOG/.done_$name"; echo "[done] $name"
  else echo "[FAIL] $name (见 $f)"; fi
}

# 每张卡一个 worker，从共享队列取活
QUEUE="$LOG/.queue"
: > "$QUEUE"
for j in "${JOBS[@]}"; do [ -n "$j" ] && echo "$j" >> "$QUEUE"; done
LOCK="$LOG/.lock"

worker () {
  local gpu=$1
  while :; do
    local job=""
    { flock 9
      job=$(head -1 "$QUEUE" 2>/dev/null || true)
      [ -n "$job" ] && sed -i 1d "$QUEUE"
    } 9>"$LOCK"
    [ -z "$job" ] && break
    set -- $job
    echo "[gpu$gpu] 取活 $1 $2"
    run_one "$gpu" "$1" "$2"
  done
  echo "[gpu$gpu] 队列空，退出"
}

for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "ALL DONE"
