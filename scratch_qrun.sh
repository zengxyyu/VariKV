#!/bin/bash
# $1=gpu $2=dataset $3=ratio $4=tag $5=table.npy(或 "-" 表示不注入) $6=num $7=dumpfile(可选)
set -u
GPU=$1; DS=$2; R=$3; TAG=$4; TAB=$5; NUM=$6; DUMP=${7:-}
QMODE=${8:-}
XENV=${9:-}
# 第 10/11 个字段：模型与打分器 ckpt。**默认值与扩展前逐字相同**，
# 所以既有队列行（9 字段）行为不变。第二 backbone 用得上：
#   Qwen/Qwen3-8B + varikv/qwen3_untrained.pt/memoryless.pt
# 未训练打分器对 floor/floorcov 是**安全的**（已读代码路径确认）：配额模式下
# 保留集由 learned_ctrlcache.py 的 `argsort(sc)` 重建、`sc = score0[:,0]` 是基线
# 分数，扰动分数算出的 mask 被整块覆盖，故 ckpt 权重不可能影响这两族。
# **⚠ 任何依赖方向的臂（full/within/across/floorproj/pathproj/maxlift）禁止用它。**
MODEL=${10:-Qwen/Qwen2.5-7B-Instruct-1M}
CKPT=${11:-../../../varikv/d10_scalar_s0.pt/memoryless.pt}
case "$MODEL" in
  *Qwen3*|*qwen3*)
    # **必须精确匹配**：`*floor*` 会连 floorproj / floorpath 一起放行，
    # 而那两个走 reachable_project、依赖 alpha_eff 与学到的方向，正是要挡的。
    case "$QMODE" in
      floor|floorcov) : ;;
      *) echo "REFUSE $TAG: Qwen3 用的是未训练打分器，只允许 floor/floorcov（收到 QMODE=$QMODE）" >&2; exit 3 ;;
    esac ;;
esac
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
export CUDA_VISIBLE_DEVICES=$GPU VARIKV_RATIOS=$R
[ "$TAB" != "-" ] && export VARIKV_QUOTA_INJECT=/home/ubuntu/zxy/vlm-memory/$TAB
[ -n "$DUMP" ] && [ "$DUMP" != "-" ] && export VARIKV_QUOTA_DUMP=/home/ubuntu/zxy/vlm-memory/$DUMP
[ -n "$QMODE" ] && [ "$QMODE" != "-" ] && export VARIKV_QUOTA_MODE=$QMODE
# XENV 支持**逗号分隔多个**变量：队列行按空格分字段，所以一个字段里不能有空格。
# pathproj 需要同时给 VARIKV_QUOTA_FLOOR 与 VARIKV_PROJ_LAMBDA。
if [ -n "$XENV" ] && [ "$XENV" != "-" ]; then
  IFS=',' read -ra _KVS <<< "$XENV"
  for _kv in "${_KVS[@]}"; do [ -n "$_kv" ] && export "$_kv"; done
fi
../../../.venv/bin/python -B eval_chunk.py -m "$MODEL" -g fastkvzip \
  --num $NUM --prefill_chunk 16000 --window_size 4096 --level pair -d $DS --tag $TAG \
  --ctrlm_ckpt "$CKPT" --ctrlm_mode memoryless \
  > /home/ubuntu/zxy/vlm-memory/scratch_ctrl_logs/${TAG#_}.log 2>&1
echo "DONE $TAG rc=$?" >> /home/ubuntu/zxy/vlm-memory/scratch_ctrl_logs/qrun_done.log
