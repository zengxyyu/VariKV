#!/usr/bin/env bash
# 质心容量扫描（2026-08-12）—— 免训练，回答「被删掉的 KV 值不值得用额外内存概括」。
#
# 6 个 job，每个 = scbench_kv 100 条 @ratio 0.1，一张卡约 1.5 h，并行 ⇒ ~2 h。
# 前置：scratch_verify_centroid.py 四项验收全过（已过，见 48e8989）。
#
#   cen16    每头 16 簇 —— **与失败版本容量完全相同**。若这个就涨，病根全在接线
#   cen109   离线恢复 oracle 70% 的那个配置，原样搬过来
#   cen256   更多有没有更好
#   cen1024  同上，看饱和
#   mb1024   **matched-budget 对照**：不加摘要，改成多留同样字节的真实 KV。
#            一个质心 2d+1=257 scalars、一条 exact KV 2d=256 ⇒ K 个质心 ≈ K 条 KV。
#            实测保留量 16903/(层,头) ⇒ +1024 条 = +6.06% ⇒ ratio 0.1061。
#            **不加这一档，就算涨了也答不出「为什么不直接少压一点」**
#   cen109inv  RoPE 对照：K 越大簇越窄 ⇒ 内容分辨率与相位一致性同时改善，
#            容量曲线混淆两者。inv 逆旋到无位置帧再按位置质心旋回，用来分开它们
#
# 基线复用已有的 rb 运行（scbench_kv @0.1 = 32.60），不重跑。
set -u
cd "$(dirname "$0")/external/FastKVzip/prefill" || exit 1
PY=../../../.venv/bin/python
LOG=../../../scratch_centroid_logs
mkdir -p "$LOG"
# 比例由 VARIKV_RATIOS 决定（eval_chunk.py 用 set_ratios() 的值做 chunk_ratio），
# **没有 --chunk_ratio 这个参数**。matched-budget 档因此靠改这个环境变量实现。
COMMON="-m Qwen/Qwen2.5-7B-Instruct-1M -d scbench_kv -g fastkvzip --num 100
        --prefill_chunk 16000 --window_size 4096 --level pair"

run () {  # run <gpu> <name> <ratio> <extra args...>
  local gpu=$1 name=$2 ratio=$3; shift 3
  local f="$LOG/$name.log"
  if [ -f "$LOG/.done_$name" ]; then echo "[skip] $name"; return; fi
  echo "[gpu$gpu] $name : $*"
  ( export VARIKV_RATIOS=$ratio
    CUDA_VISIBLE_DEVICES=$gpu $PY -B eval_chunk.py $COMMON --tag "_$name" "$@" \
        > "$f" 2>&1
    if grep -q "Finished." "$f"; then touch "$LOG/.done_$name"; echo "[done] $name"
    else echo "[FAIL] $name (见 $f)"; fi ) &
}

run 0 cen16       0.1     --centroid_k 16
run 1 cen109      0.1     --centroid_k 109
run 2 cen256      0.1     --centroid_k 256
run 3 cen1024     0.1     --centroid_k 1024
run 4 cen109inv   0.1     --centroid_k 109 --centroid_rope inv
# matched-budget：纯基线（无质心），只把比例提到等字节的 0.1061
#   保留 16903/(层,头) × 1.0608 = +1028 条 ≈ 1024 个质心的字节
run 5 mb1024      0.1061
wait
echo "ALL DONE"
