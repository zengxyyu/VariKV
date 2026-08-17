#!/bin/bash
# 主队列：干净版 v2（3 种子 × 11 panel × 9 ratio）+ v2 档四臂（scbench_kv @0.2/0.1）。
#
# ── 为什么是工作池而不是"一轮 8 个再 wait" ──────────────────────────────
# 首版用轮次栅栏：派 8 个 → wait 全部 → 派下一轮。两处硬伤，实测都发生了：
#   · 作业数不足 8 时其余卡空转（B 阶段只有 3 个 ⇒ 5 张卡闲着）；
#   · 同轮耗时差极大时按最慢的那个对齐（D 里 gsm 86 token vs scbench_kv 169k，
#     33 个作业分 5 轮，最后一轮只剩 1 个 ⇒ 7 张卡空转最长 3 小时）。
# 改成 8 个常驻 worker 各占一张卡，从共享队列 flock 原子取任务，做完立刻取下一个。
# **全程无栅栏**，且 C/E/D 合成一个按优先级排好的队列 —— C 取完 worker 直接接 E/D。
#
# ── ratio 0.02 是阴性对照，不是结果 ──────────────────────────────────────
# `wrapper.py:286-292`：`ratio*clen < window` 时 `window = int(ratio*clen)` 且
# `chunk_ratio = 0`，于是预算照给、但**选择从"按门控分数"变成"纯按最近"**，
# 门控分数完全不参与 ⇒ 任何改分数的方法恒为 no-op。门槛 `clen > 4096/ratio`：
#   0.2  → 20,480   只有 gsm(86) / squad(203) 过不去
#   0.05 → 81,920   repoqa(72k) 及更短的过不去
#   0.02 → 204,800  **11 个 panel 全部过不去**（最长 scbench_kv 只有 169,428）
# 所以 0.02 那一列的 Δ 必然是 0；跑它是为了当阴性对照 —— 测出非零就说明实现坏了。
# 注意"退化"指的是**分数不再被使用**，不是精度崩掉：一个占 20% 上下文的滑动窗口
# 常常分数很好看，FastKVzip 图 11 在那些点上正常，与这里说的是两回事。
#
# 四臂只评 0.2/0.1：0.1 是 +4.27 的参照工作点；0.2 上 v2 有 +18.80★、效应量大
# 4 倍，分辨四条臂的功效高得多。其余 ratio 对"机制是什么"没有信息。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
LOG=$ROOT/scratch_ctrl_logs
Q=$LOG/.mq_jobs
cd "$ROOT" || exit 1
. "$ROOT/scratch_gpu_lock.sh"
release_all() { for g in 0 1 2 3 4 5 6 7; do gpu_release "$g"; done; }
trap release_all EXIT

PANELS="scbench_kv scbench_vt scbench_mf scbench_qa_eng scbench_choice_eng \
scbench_summary scbench_prefix_suffix scbench_repoqa scbench_many_shot squad gsm"
R9=1.0,0.75,0.5,0.4,0.3,0.2,0.1,0.05,0.02
EVAL="$ROOT/.venv/bin/python -B eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M \
-g fastkvzip --num 100 --prefill_chunk 16000 --window_size 4096 --level pair"

pop_job() {   # 原子弹出队首；队空返回 1
    ( flock 9
      line=$(head -1 "$Q" 2>/dev/null)
      [ -z "$line" ] && exit 1
      sed -i '1d' "$Q"
      printf '%s' "$line" ) 9>"$Q.lock"
}

run_pool() {   # 8 个 worker 常驻，各占一卡，边取边做
    local pids=() i
    for i in 0 1 2 3 4 5 6 7; do
        ( G=$(gpu_claim)
          while j=$(pop_job); do
              echo "$(date +%H:%M)  [GPU$G] ${j:0:130}"
              eval "CUDA_VISIBLE_DEVICES=$G $j"
          done
          gpu_release "$G" ) &
        pids+=($!); sleep 3
    done
    wait "${pids[@]}"
    echo "$(date +%H:%M)  ---- 队列排空 ----"
}

# ── 阶段 A：等 d10 四臂 12 个 ckpt 就绪（另一个脚本在训）────────────────
echo "=== 等 d10 四臂训练完成 ==="
while [ "$(ls varikv/d10_*.pt/memoryless.pt 2>/dev/null | wc -l)" -lt 12 ]; do sleep 60; done
pkill -f 'scratch_d10_arms[.]sh' 2>/dev/null; sleep 5
BAD=$(grep -L "训练 8 篇 / 验证 2 篇" "$LOG"/d10_*_s*.log 2>/dev/null | wc -l)
if [ "$BAD" -ne 0 ]; then echo "✗ $BAD 个训练不是 8/2 划分，中止"; exit 1; fi
echo "$(date +%H:%M)  四臂 12 个 ckpt 就绪，全部 8/2 划分"

# ── 阶段 B：干净版 v2 训练，3 种子（便宜，先做完好排后面的评测）─────────
while pgrep -f 'varikv_v2[.]py train' >/dev/null; do sleep 30; done   # 等在飞的
: > "$Q"
for S in 0 1 2; do
    [ -f "varikv/v2c_s$S.pt" ] || echo "$ROOT/.venv/bin/python -u $ROOT/varikv_v2.py \
train --data v2 --seed $S --out varikv/v2c_s$S.pt > $LOG/v2c_train_s$S.log 2>&1" >> "$Q"
done
if [ -s "$Q" ]; then
    echo "=== B 干净版 v2 训练（$(wc -l < "$Q") 个）==="; release_all; run_pool
fi
for S in 0 1 2; do
    grep -q "训练 8 / 验证 2" "$LOG/v2c_train_s$S.log" 2>/dev/null \
        || { echo "✗ v2c s$S 划分不对或未完成"; exit 1; }
    [ -f "varikv/v2c_s${S}_compat.pt" ] || { echo "✗ v2c s$S 缺 compat 版"; exit 1; }
done
echo "$(date +%H:%M)  干净版 v2 三个种子就绪（8/2 划分 + compat 版都已核对）"

# ── C + E + D 合成一个按优先级排好的队列，无栅栏 ─────────────────────────
cd "$ROOT/external/FastKVzip/prefill" || exit 1
: > "$Q"
# C 四臂评测（12）—— 最值钱、最便宜，排最前
for S in 0 1 2; do for A in bias affine scalar kv; do
    [ -f "$LOG/.done_d10_${A}_s$S" ] && continue
    echo "env VARIKV_RATIOS=0.2,0.1 $EVAL -d scbench_kv --tag _d10${A}_s$S \
--ctrlm_ckpt ../../../varikv/d10_${A}_s$S.pt/memoryless.pt \
> $LOG/d10bench_${A}_s$S.log 2>&1" >> "$Q"
done; done
# E 基线补 ratio 0.02（11）—— 只为让 D 的 0.02 那格能配对
for P in $PANELS; do
    [ -f "$LOG/.done_b002_$P" ] && continue
    echo "env VARIKV_RATIOS=0.02 $EVAL -d $P --tag _b002 > $LOG/b002_$P.log 2>&1" >> "$Q"
done
# D 干净版 v2 评测（33）—— 长作业排前面，短的垫后，减少池尾空转
for S in 0 1 2; do for P in $PANELS; do
    [ -f "$LOG/.done_v2c_${P}_s$S" ] && continue
    echo "env VARIKV_RATIOS=$R9 $EVAL -d $P --tag _v2c_s$S \
--ctrlm_ckpt ../../../varikv/v2c_s${S}_compat.pt > $LOG/v2cbench_${P}_s$S.log 2>&1" >> "$Q"
done; done
echo "=== C+E+D 合并队列共 $(wc -l < "$Q") 个作业 ==="
release_all; run_pool

for S in 0 1 2; do for A in bias affine scalar kv; do
    tail -3 "$LOG/d10bench_${A}_s$S.log" 2>/dev/null | grep -q Finished \
        && touch "$LOG/.done_d10_${A}_s$S"
done; done
for P in $PANELS; do
    tail -3 "$LOG/b002_$P.log" 2>/dev/null | grep -q Finished && touch "$LOG/.done_b002_$P"
    for S in 0 1 2; do
        tail -3 "$LOG/v2cbench_${P}_s$S.log" 2>/dev/null | grep -q Finished \
            && touch "$LOG/.done_v2c_${P}_s$S"
    done
done
echo "$(date +%H:%M)  === 主队列全部完成 ==="
