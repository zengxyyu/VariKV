#!/usr/bin/env bash
# P2/P3/P4 —— 把 +21.60 这个结果**解释清楚**，而不是继续加机制。
#
# 全部在 ratio 0.1、chunk 16000 / window 4096 / level pair 上跑，与已有基线同批可比。
# 基线复用：`_kls_base`（七数据集扫描那批）与 `_b01`（scbench_kv），配置逐字相同。
#
# ── P2：centroid（免训练）在 prefix_suffix / repoqa 上 ────────────────────────
#   这是最便宜的高信息实验，而且它的结果决定 P3/P4 还值不值得做：
#     centroid 也涨  ⇒ 问题在 learned readout 的泛化，P3/P4 值得做
#     centroid 也是 0 ⇒ 这两类任务的信息压不进小摘要，任务选择性是本质的，
#                       P3/P4 降级为次要
#
# ── P3：gate surgery ───────────────────────────────────────────────────────
#   point 的门学到 σ=0.265，dist 只有 0.131，而"门越开分越低"是本项目已建立的
#   规律，且 point 的生成长度只有 48.9（基线 120.5）= 退化输出。
#   所以 14.60 可能纯粹是**幅度失控**，与"方差携带信息"无关。
#     gs 0.5 / 0.25  把 point 的注入幅度缩小
#     gfrom dist     直接把 dist 的门装到 point 身上（最锐利）
#   若 point 因此回到 45~50 ⇒ 39.6 分的差主要是 gate，不是表示。
#
# ── P4：dist 外科消融 ──────────────────────────────────────────────────────
#   dist 与 point 同时差**四处**（memory.py 顶部）：τ_obs / τ_old /
#   η_i=σ(α·z(KL)−β) 是否随内容变 / decoder 是否读 logvar。
#   逐个关掉，看 54.20 各掉到哪里。**这是"论文能不能叫 distributional"的唯一判据。**
#     ablate logvar     读出不看方差（只剩写入侧）
#     ablate precision  τ≡1（与 point 同）
#     ablate eta        写入强度换成本批均值（去内容相关性、保住写入总量）
set -u
cd "$(dirname "$0")/external/FastKVzip/prefill" || exit 1
PY=../../../.venv/bin/python
LOG=../../../scratch_p234_logs
mkdir -p "$LOG"
KL=../../../varikv/ckpt_kl/s2b_dist_k16.pt
KLP=../../../varikv/ckpt_kl/s2b_point_k16.pt
C="-m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100
   --prefill_chunk 16000 --window_size 4096 --level pair"

# 每行：<名字> | <数据集> | <额外参数>
JOBS=(
  # P2 —— 免训练 centroid（最先跑，它决定后面的优先级）
  "p2_cen16_ps|scbench_prefix_suffix|--centroid_k 16"
  "p2_cen1024_ps|scbench_prefix_suffix|--centroid_k 1024"
  # Retr.MultiHop（补加 2026-08-12 晚）：teacher-KL 在这里 **−18.36★**，是全表最大的
  # 负结果，而它的基线 49.47 > 满缓存 41.07（压缩本身有益）。
  # 质心是免训练 + 正确代数 + 忠实还原质量，所以它能区分：
  #   质心也害 ⇒ 伤害来自"把被删的东西加回来"本身 ⇒ 整个吸收范式在高冗余任务上有根本问题
  #   质心不害 ⇒ 伤害是**学出来的记忆**特有的 ⇒ 修读出/门就够
  # 预测（可证伪）：质心应该害得**更狠**，因为它按 log n_j 忠实还原质量，
  # 而学出来的记忆是经一个小门缩放后注入的。
  "p2_cen16_vt|scbench_vt|--centroid_k 16"
  "p2_cen1024_vt|scbench_vt|--centroid_k 1024"
  "p2_cen16_rq|scbench_repoqa|--centroid_k 16"
  "p2_cen1024_rq|scbench_repoqa|--centroid_k 1024"
  # P3 —— gate surgery（全部在 scbench_kv 上，与 54.20/14.60 直接可比）
  "p3_pt_gs0p5|scbench_kv|--varikv_ckpt $KLP --varikv_residual --varikv_slots 16 --varikv_gate_scale 0.5"
  "p3_pt_gs0p25|scbench_kv|--varikv_ckpt $KLP --varikv_residual --varikv_slots 16 --varikv_gate_scale 0.25"
  "p3_pt_gfrom|scbench_kv|--varikv_ckpt $KLP --varikv_residual --varikv_slots 16 --varikv_gate_from $KL"
  # 对照：把 dist 的门放大到 point 的水平 —— 若 dist 也因此崩，规律得到双向确认
  "p3_di_gs2|scbench_kv|--varikv_ckpt $KL --varikv_residual --varikv_slots 16 --varikv_gate_scale 2.0"
  # P4 —— dist 外科消融
  "p4_ab_logvar|scbench_kv|--varikv_ckpt $KL --varikv_residual --varikv_slots 16 --varikv_ablate logvar"
  "p4_ab_precision|scbench_kv|--varikv_ckpt $KL --varikv_residual --varikv_slots 16 --varikv_ablate precision"
  "p4_ab_eta|scbench_kv|--varikv_ckpt $KL --varikv_residual --varikv_slots 16 --varikv_ablate eta"
)

QUEUE="$LOG/.queue"
# RESUME=1 时**不重建队列**，接着已有的 .queue 跑（避免把正在跑的 job 重新排一遍）
if [ "${RESUME:-0}" != "1" ]; then
  : > "$QUEUE"
  for j in "${JOBS[@]}"; do echo "$j" >> "$QUEUE"; done
fi
LOCK="$LOG/.lock"

# 等空卡（显存 <2 GB 视为空闲）。
# **必须排除 GPU 0/1**：scratch_v2b_wait.sh 硬编码用它们训 v2b，四个 v2 评测一结束
# 就会占用。两个调度器都按"显存空了就抢"来判断，不排除就会抢同一张卡而 OOM。
# 默认放开全部 8 张卡。之前排除 0/1 是为 scratch_v2b_wait.sh 留的，v2b 已训完。
# 若再有手动占卡的任务，用 CANDIDATES="..." 覆盖。
CANDIDATES="${CANDIDATES:-0 1 2 3 4 5 6 7}"
# 空闲判据必须**严于**"显存 <2 GB"。踩过的坑：评测 job 在生成阶段显存会短暂回落
# 到阈值以下，于是同一张卡被派了第二个 job（实测 GPU6 上 _kls_kl 与
# _p2_cen16_rq 挤在一起）。两条加强：
#   ① 该卡上不能有**任何** compute 进程（用 nvidia-smi 的进程列表，不看显存）
#   ② 间隔 20 s 连采两次都满足，才算真空闲
gpu_has_proc () {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    | grep -qF "$(nvidia-smi --query-gpu=gpu_uuid --format=csv,noheader -i "$1")"
}
free_gpu () {
  for g in $CANDIDATES; do
    gpu_has_proc "$g" && continue
    sleep 20
    gpu_has_proc "$g" && continue
    echo "$g"; return 0
  done
  return 1
}

echo "$(date -u +%H:%M) 排入 ${#JOBS[@]} 个 job，候选卡 [$CANDIDATES]，等空卡"

while :; do
  n=$(wc -l < "$QUEUE")
  [ "$n" -eq 0 ] && break
  g=$(free_gpu) || { sleep 60; continue; }
  job=""
  { flock 9
    job=$(head -1 "$QUEUE" 2>/dev/null || true)
    [ -n "$job" ] && sed -i 1d "$QUEUE"
  } 9>"$LOCK"
  [ -z "$job" ] && break
  name="${job%%|*}"; rest="${job#*|}"; ds="${rest%%|*}"; extra="${rest#*|}"
  if [ -f "$LOG/.done_$name" ]; then echo "[skip] $name"; continue; fi
  echo "$(date -u +%H:%M) [gpu$g] $name  ($ds)"
  ( VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$g $PY -B eval_chunk.py $C \
        -d "$ds" --tag "_$name" $extra > "$LOG/$name.log" 2>&1
    if grep -q "Finished." "$LOG/$name.log"; then touch "$LOG/.done_$name"
      echo "$(date -u +%H:%M) [done] $name"
    else echo "$(date -u +%H:%M) [FAIL] $name"; fi ) &
  sleep 45          # 错开启动，避免同时抢显存
done
wait
echo "$(date -u +%H:%M) ALL DONE"
