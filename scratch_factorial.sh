#!/bin/bash
# 因子消融：把 `scalar` 的三个标量输入拆开，判定 +4.73 的胜因是**非线性**还是
# **知道全局阈值在哪**。同时测"去掉头身份还剩多少"。
#
# 为什么必须做：`scalar ≫ affine` 同时改了两件事 —— 函数类（线性 → MLP）与输入集
# （只有 z → z+margin+rs）。现在的证据分不清哪个是胜因。
#
# 实测支撑（2026-08-17，在真实 trace 上算的，不是推导）：
#   margin 与 z 在头内是精确的仿射关系  m_i = A_h·z_i + B_h,  残差 2.4e-7（float32 精度）
#   其中 A_h = σ_h/σ_g、B_h = (μ_h−τ)/σ_g，**逐 chunk 变化**：同一 (层13,头0) 上
#   A_h 在 8 个 chunk 里从 0.024 变到 0.056（2.3×）。
#   而 `affine` 的 a_{l,h}、b_{l,h} 是**固定常数** ⇒ 它在原理上表达不了 margin，
#   不是"能表达但没给它"。这就是要拆开测的理由。
#   另：A_h 跨头从 0.011 到 0.93（约 75×）—— 各头分数尺度差两个数量级，而
#   level="pair" 是全局阈值化。
#
# 判读（预注册）：
#   sz ≈ szr ≈ 0 而 szm ≈ scalar   ⇒ **全局阈值位置是关键缺失状态**，命题干净
#   sz 就 ≈ scalar                 ⇒ 胜因是非线性，与 margin 无关，故事要改写
#   szmr0 ≈ scalar                 ⇒ 存在与"我是哪个头"无关的**普适修正律**，
#                                     参数可再降到 2.4K 且不需要逐头表
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
LOG=$ROOT/scratch_ctrl_logs
Q=$LOG/.mq_jobs
TR=scratch_ctrl_traces_v2_10
ARCHS="sz szr szm szmr0"

# 与主队列共用同一个队列文件和同一把 flock —— 直接插队，不另起调度器，
# 否则两个调度器各扫各的卡，会把作业叠到同一张卡上（本项目已发生过 OOM）。
prepend() {   # 把若干行插到队首（短作业优先，减少长尾）
    ( flock 9
      tmp=$(mktemp); printf '%s\n' "$@" > "$tmp"
      [ -f "$Q" ] && cat "$Q" >> "$tmp"
      mv "$tmp" "$Q" ) 9>"$Q.lock"
}
append() { ( flock 9; printf '%s\n' "$@" >> "$Q" ) 9>"$Q.lock"; }

# ── 1. 训练（插队首，每个约 13 分钟）────────────────────────────────────
T=()
for S in 0 1 2; do for A in $ARCHS; do
    [ -f "varikv/d10_${A}_s$S.pt/memoryless.pt" ] && continue
    T+=("cd $ROOT && $ROOT/.venv/bin/python -u scratch_ctrl_train.py \
--traces $TR --epochs 40 --seed $S --split_seed 42 --arch $A \
--alpha_init 1.0 --freeze_alpha --pair_w linear --lam_global 1.0 \
--out varikv/d10_${A}_s$S.pt > $LOG/d10_${A}_s$S.log 2>&1")
done; done
[ ${#T[@]} -gt 0 ] && { prepend "${T[@]}"; echo "$(date +%H:%M)  插入 ${#T[@]} 个因子臂训练"; }

# ── 2. 等 ckpt 齐了再追加评测（不能和训练同时排，会读到不存在的 ckpt）──
echo "$(date +%H:%M)  等 12 个因子臂 ckpt..."
while :; do
    n=0; for S in 0 1 2; do for A in $ARCHS; do
        [ -f "varikv/d10_${A}_s$S.pt/memoryless.pt" ] && n=$((n+1)); done; done
    [ "$n" -ge 12 ] && break; sleep 120
done
BAD=0
for S in 0 1 2; do for A in $ARCHS; do
    grep -q "训练 8 篇 / 验证 2 篇" "$LOG/d10_${A}_s$S.log" || BAD=$((BAD+1)); done; done
[ "$BAD" -ne 0 ] && { echo "✗ $BAD 个训练不是 8/2 划分，不排评测"; exit 1; }
echo "$(date +%H:%M)  12 个 ckpt 就绪，全部 8/2 划分"

E=()
for S in 0 1 2; do for A in $ARCHS; do
    E+=("cd $ROOT/external/FastKVzip/prefill && env VARIKV_RATIOS=0.2,0.1 \
$ROOT/.venv/bin/python -B eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip \
--num 100 --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
--tag _d10${A}_s$S --ctrlm_ckpt ../../../varikv/d10_${A}_s$S.pt/memoryless.pt \
> $LOG/d10bench_${A}_s$S.log 2>&1")
done; done
prepend "${E[@]}"
echo "$(date +%H:%M)  插入 ${#E[@]} 个因子臂评测"
