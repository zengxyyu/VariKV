#!/bin/bash
# 四臂 × 3 种子，训在 **v2 那 10 篇**上（8/2 划分、320 步），再下游评测。
#
# 与既有 `varikv/dec_*` 的区别（后者是废的）：dec_* 训在扩容后的 30 篇上
# （23/7、920 步），而 v2 训在 10 篇上（8/2、320 步）。要解释 v2 的 +4.27，
# 消融臂必须与它同数据、同划分、同步数、同教师靶子，否则架构差异与
# 数据量/步数混在一起。分母用 v2 自己的三个种子
# （下游 tag: __b2memoryless / __b2s1me / __b2s2me，各 n=100 @ratio 0.1）。
#
# 训练参数与 v2 逐字相同，**只多一个 --arch**，这正是原 scratch_ctrl_decomp.sh 的做法。
# `--traces` 指向只含那 10 篇的符号链接目录（sha256 已核对 = v2 读到的那批）。
#
# --------------------------------------------------------------------------
# **按「整轮 wait 后统一释放锁」组织，不再按 PID 回收。** 首版用
# `kill -0 $PID` 判断作业结束来放锁 —— 那是错的：后台子进程结束后变成**僵尸**，
# `kill -0` 对僵尸返回成功，于是锁永不释放。实测后果：8 张卡全上锁、卡上却
# 一个进程都没有，下一轮 gpu_claim 永远阻塞（16:14 起卡死到 16:28）。
# 整轮 wait 之后再释放，不依赖任何存活判断。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
LOG=$ROOT/scratch_ctrl_logs
TR=scratch_ctrl_traces_v2_10
cd "$ROOT" || exit 1
. "$ROOT/scratch_gpu_lock.sh"

release_all() { for g in 0 1 2 3 4 5 6 7; do gpu_release "$g"; done; }
trap release_all EXIT        # 被 kill 也不留死锁

echo "=== 阶段 1：训练（缺哪个补哪个）==="
JOBS=()
for S in 0 1 2; do for A in bias affine scalar kv; do
    [ -f "varikv/d10_${A}_s$S.pt/memoryless.pt" ] || JOBS+=("$A:$S")
done; done
echo "  待训练 ${#JOBS[@]} 个: ${JOBS[*]:-（无，全部已完成）}"
while [ ${#JOBS[@]} -gt 0 ]; do
    R=("${JOBS[@]:0:8}"); JOBS=("${JOBS[@]:8}")
    for j in "${R[@]}"; do
        A=${j%:*}; S=${j#*:}
        G=$(gpu_claim)
        CUDA_VISIBLE_DEVICES=$G nohup "$ROOT/.venv/bin/python" -u scratch_ctrl_train.py \
            --traces "$TR" --epochs 40 --seed "$S" --split_seed 42 \
            --arch "$A" --alpha_init 1.0 --freeze_alpha --pair_w linear \
            --lam_global 1.0 --out "varikv/d10_${A}_s$S.pt" \
            > "$LOG/d10_${A}_s$S.log" 2>&1 &
        echo "$(date +%H:%M)  [GPU$G] 训练 $A s$S"
        sleep 15
    done
    wait; release_all
    echo "$(date +%H:%M)  ---- 一轮训练完成 ----"
done

# 硬门：每个训练都必须是 8/2，否则 trace 目录不对，不许进评测
BAD=$(grep -L "训练 8 篇 / 验证 2 篇" "$LOG"/d10_*_s*.log 2>/dev/null | wc -l)
if [ "$BAD" -ne 0 ]; then echo "✗ 有 $BAD 个训练不是 8/2 划分，中止"; exit 1; fi
echo "$(date +%H:%M)  === 12 个 ckpt 全部 40 epoch / 8-2 划分 ==="

echo "=== 阶段 2：下游评测，先 s0/s1 再 s2 ==="
cd "$ROOT/external/FastKVzip/prefill" || exit 1
EJOBS=()
for S in 0 1 2; do for A in bias affine scalar kv; do
    [ -f "$LOG/.done_d10_${A}_s$S" ] || EJOBS+=("$A:$S")
done; done
echo "  待评测 ${#EJOBS[@]} 个"
while [ ${#EJOBS[@]} -gt 0 ]; do
    R=("${EJOBS[@]:0:8}"); EJOBS=("${EJOBS[@]:8}")
    for j in "${R[@]}"; do
        A=${j%:*}; S=${j#*:}
        G=$(gpu_claim)
        VARIKV_RATIOS=1.0,0.5,0.4,0.3,0.2,0.1,0.05 CUDA_VISIBLE_DEVICES=$G \
            nohup "$ROOT/.venv/bin/python" -B eval_chunk.py \
            -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
            --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
            --tag "_d10${A}_s$S" \
            --ctrlm_ckpt "../../../varikv/d10_${A}_s$S.pt/memoryless.pt" \
            > "$LOG/d10bench_${A}_s$S.log" 2>&1 &
        echo "$(date +%H:%M)  [GPU$G] 评测 $A s$S"
        sleep 30
    done
    wait; release_all
    for j in "${R[@]}"; do
        A=${j%:*}; S=${j#*:}
        tail -3 "$LOG/d10bench_${A}_s$S.log" 2>/dev/null | grep -q Finished \
            && touch "$LOG/.done_d10_${A}_s$S"
    done
    echo "$(date +%H:%M)  ---- 一轮评测完成 ----"
done
echo "$(date +%H:%M)  === v2 档四臂全部完成（12 训练 + 12 评测）==="
