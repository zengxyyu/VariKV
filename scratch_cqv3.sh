#!/usr/bin/env bash
# probe v3+fix：transductive vs held-out C_q 的 2×2，四个臂同一份代码同一份口径
#
# **为什么不是只补跑那个断掉的 job。** 断掉的是 `scbench_kv` 的 held-out 臂
# （停在 87/100，无 traceback，最后一张表只有 80 条）。但它原来的对照——转导臂——
# 跑的是 v2 代码，而 v3 改掉了 C_q 的标定分布（all-token → 末 16 个 query token，
# 与「评估只用最后一个 query token」对齐），并新增了度量感知的二阶标量列。
# 拿 v3 的 held-out 去比 v2 的转导，比的是两件事的叠加。四个臂一起重跑，
# 挂钟时间与只跑一个相同（卡是空的），但结论才是可解释的。
#
# 判据（**看数之前写死**）：held-out C_q 在 Retr.KV 上回收
#   (eucl-Lloyd − Cq-Lloyd) / (eucl-Lloyd − score-oracle) ≥ 20%
# 且样本级配对 CI 不含 0 ⇒ C_q 方法线继续；否则降级为分析结论。
#
# 用法： setsid nohup bash scratch_cqv3.sh > scratch_cqv3.log 2>&1 &
set -u
cd "$(dirname "$0")"
ROOT=$PWD
LOGS=$ROOT/scratch_cqv3_logs; mkdir -p "$LOGS"
N=${N:-100}
# 完成判据里的 N−11 松弛量是给 `scbench_vt` 的：它整个数据集只有 90 条，
# 所以 --n 100 时它最多跑到样本 89，那是**跑完**，不是被截断。

# arm: 名字:数据集:cq_from（空=转导）
ARMS=(
  "kv_trans:scbench_kv:"
  "kv_held:scbench_kv:scbench_vt"
  "vt_trans:scbench_vt:"
  "vt_held:scbench_vt:scbench_kv"
)

free_gpu() {   # 进程存在性是二值的；显存在生成阶段会掉到很低，用显存判空会把两个任务派到同一张卡
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] || continue
    sleep 20                                   # 20 秒后复核，避开正在启动的卡
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] && { echo "$g"; return; }
  done
  echo ""
}

i=0
while [ $i -lt ${#ARMS[@]} ]; do
  IFS=: read -r name data cqfrom <<< "${ARMS[$i]}"
  mk="$LOGS/.done__$name"
  [ -f "$mk" ] && { echo "跳过 $name（已完成）"; i=$((i+1)); continue; }
  g=$(free_gpu)
  [ -z "$g" ] && { sleep 60; continue; }
  extra=""
  [ -n "$cqfrom" ] && extra="--cq_from $cqfrom"
  echo "[$(date -u +%H:%M)] $name ($data, cq_from='${cqfrom:-transductive}') → GPU$g"
  (
    # **`--n` 必须显式给**：默认是 8，不给就安静地只跑 8 条（第一次重跑就栽在这）。
    CUDA_VISIBLE_DEVICES=$g "$ROOT/.venv/bin/python" -B scratch_probe_cluster.py \
      --data "$data" --K 16 --ratio 0.1 --n "$N" --report_every 10 \
      --out "$ROOT/scratch_cqv3_$name" $extra \
      > "$LOGS/$name.log" 2>&1
    rc=$?
    # **完成判据 = 最终表 + 跑满样本，不是 rc**：上一轮那个 job 是被杀的，rc 也可能是 0
    # （父 shell 被收走时子进程无输出即退出）。同时不能用「表的张数≥2」——report_every
    # 比 n 大时成功的 run 也只打 1 张表，那样会把成功判成失败（这是第一次重跑的第二个 bug）。
    last=$(grep -o '样本 [0-9]* 完成' "$LOGS/$name.log" | tail -1 | tr -dc 0-9)
    if [ $rc -eq 0 ] && grep -q "判读（预注册）" "$LOGS/$name.log" \
       && [ -n "$last" ] && [ "$last" -ge $((N - 11)) ]; then
      touch "$mk"; echo "[$(date -u +%H:%M)] $name 完成"
    else
      echo "[$(date -u +%H:%M)] **$name 失败** rc=$rc 最后样本=${last:-无}（不写 marker）"
    fi
  ) &
  sleep 30                                     # 错开加载，避免同时抢显存
  i=$((i+1))
done
wait
echo "[$(date -u +%H:%M)] ALL DONE"
for a in "${ARMS[@]}"; do
  n=${a%%:*}
  [ -f "$LOGS/.done__$n" ] && echo "  ✓ $n" || echo "  ✗ $n"
done
