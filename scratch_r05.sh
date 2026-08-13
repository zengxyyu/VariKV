#!/usr/bin/env bash
# 训练无关质心 @ ratio 0.05（外加 0.1 做同批一致性校验）× 5 个代表性面板 × {基线, K=16, K=1024}
#
# 为什么是 0.05：现有质心数据全在 0.1 和 0.3/0.2。而 RestoreKV 报告的规律是**预算越紧增益越大**
# （KVzip 在 r=0.05 掉到 38.2，恢复到 73.2；r=0.2 时几乎没差别）。所以 0.05 才是恢复类方法
# 该被检验的档位。我们自己的 headroom 图也支持：Retr.KV 在 0.1 时基线 32.60、满缓存 68.20。
#
# 为什么排除 gsm / squad：上下文 86 / 203 token，比 window_size=4096 还短，永远不触发驱逐，
# 在任何 ratio 下都测不到东西。
#
# 为什么同批带 0.1：eval_chunk 是 `for ratio in set_ratios(): prefill(ratio)`，两档各自预填一次，
# 成本翻倍；但 0.1 已有独立数据（tag `_cen16`/`_b01`），同批复现得上才能证明这批没跑歪。
#
# 用法： setsid nohup bash scratch_r05.sh > scratch_r05.log 2>&1 &
#       冒烟： NUM=2 DATASETS=scbench_vt bash scratch_r05.sh
set -u
cd "$(dirname "$0")"
ROOT=$PWD
PY=$ROOT/.venv/bin/python
LOGS=$ROOT/scratch_r05_logs; mkdir -p "$LOGS"
NUM=${NUM:-100}
RATIOS=${RATIOS:-0.1,0.05}
# 长任务优先（单 ratio 实测：repoqa ≈ kv > prefix_suffix > vt > summary）
DATASETS=${DATASETS:-"scbench_kv scbench_repoqa scbench_prefix_suffix scbench_vt scbench_summary"}

# arm: tag后缀|额外参数
ARMS=("r05b|" "r05c16|--centroid_k 16" "r05c1024|--centroid_k 1024")

JOBS=()
for d in $DATASETS; do for a in "${ARMS[@]}"; do JOBS+=("$d|$a"); done; done
echo "共 ${#JOBS[@]} 个任务（${NUM} 条 × ratio ${RATIOS}）"

free_gpu() {   # 用进程存在性判空；显存在生成阶段会掉到很低，用显存会把两个任务派到同一张卡
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] || continue
    sleep 20                                    # 复核一次，避开正在启动的卡
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] && { echo "$g"; return; }
  done
  echo ""
}

i=0
while [ $i -lt ${#JOBS[@]} ]; do
  IFS='|' read -r d tag extra <<< "${JOBS[$i]}"
  name="${tag}_${d}"
  mk="$LOGS/.done__$name"
  [ -f "$mk" ] && { echo "跳过 $name（已完成）"; i=$((i+1)); continue; }
  g=$(free_gpu)
  [ -z "$g" ] && { sleep 60; continue; }
  echo "[$(date -u +%H:%M)] $name → GPU$g"
  (
    cd "$ROOT/external/FastKVzip/prefill"
    # --num 必须显式给；ratio 必须经 VARIKV_RATIOS，解析端也要看到同一个值
    VARIKV_RATIOS=$RATIOS CUDA_VISIBLE_DEVICES=$g $PY -B eval_chunk.py \
      -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num "$NUM" \
      --prefill_chunk 16000 --window_size 4096 --level pair \
      -d "$d" --tag "_$tag" $extra > "$LOGS/$name.log" 2>&1
    rc=$?
    # 完成判据是 harness 自己打的 `Finished.`，不是 rc——被杀的进程 rc 也可能是 0
    if [ $rc -eq 0 ] && tail -3 "$LOGS/$name.log" | grep -q "Finished."; then
      touch "$mk"; echo "[$(date -u +%H:%M)] $name 完成"
    else
      echo "[$(date -u +%H:%M)] **$name 失败** rc=$rc（无 Finished.，不写 marker）"
    fi
  ) &
  sleep 30                                      # 错开加载，避免同时抢显存
  i=$((i+1))
done
wait
echo "[$(date -u +%H:%M)] ALL DONE"
for j in "${JOBS[@]}"; do
  IFS='|' read -r d tag extra <<< "$j"
  [ -f "$LOGS/.done__${tag}_${d}" ] && echo "  ✓ ${tag}_${d}" || echo "  ✗ ${tag}_${d}"
done
