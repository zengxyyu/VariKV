#!/bin/bash
# ρ=0.3 全流程：teacher → 3 个 chead 训练。用于直接检验「方法是 ρ-specific」这个诊断。
#
# 假说：`chd10`（训在 ρ=0.1）在 ρ=0.3 上平均 −6.12、5/6 panel 为负，是因为
# teacher 的候选池 `Near(τ_ρ)` 依赖 ρ。若在 ρ=0.3 上重训一个控制器，
# 它在 ρ=0.3 上应当**转正**。若仍为负，诊断就错了，得另找原因。
#
# 调度纪律：与评测 worker 用**同一把每卡目录锁** /tmp/vq/lock/<g>，原子 mkdir 获取；
# **不暂停评测队列**（全网格是主交付物）；判空用 --query-compute-apps 的进程存在性，
# 且相隔 20 秒两次读数都为空才认。单实例锁防重复启动。
set -u
cd "$(dirname "$0")"
L=/tmp/vq/lock
SELF=/tmp/vq/r03.lock
mkdir "$SELF" 2>/dev/null || { echo "[abort] 已有实例在跑"; exit 1; }
HELD=""
cleanup() { for g in $HELD; do rmdir "$L/$g" 2>/dev/null; done; rmdir "$SELF" 2>/dev/null; }
trap cleanup EXIT

occupied() {
    local u; u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
                 | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -qF "$u"
}

# 抢一张卡，回显卡号（阻塞直到拿到）
grab() {
    while :; do
        for g in 0 1 2 3 4 5 6 7; do
            occupied "$g" && continue
            mkdir "$L/$g" 2>/dev/null || continue
            sleep 20
            if occupied "$g"; then rmdir "$L/$g"; continue; fi
            HELD="$HELD $g"; echo "$g"; return
        done
        # 评测 worker 有 **8 个实例**在轮询，本脚本只有 1 个 —— 外层等 60 秒会长期
        # 饿死。降到 5 秒只是提高抢锁频率，**安全性不变**：真正的保护是
        # `mkdir` 的原子性 + 拿到锁后那 20 秒二次确认，两者都没动。
        sleep 5
    done
}

TR=scratch_ctrl_traces_r03
if [ ! -f "$TR/doc009.pt" ]; then
    grab; G=$GRABBED
    echo "[teacher] ρ=0.3 -> GPU$G  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$G .venv/bin/python -u scratch_ctrl_teacher.py \
        --ratio 0.3 --n_long 10 --n_short 0 --out "$TR" \
        > scratch_ctrl_logs/teacher_r03.log 2>&1
    rc=$?
    echo "[teacher] rc=$rc  $(date +%H:%M:%S)  docs=$(ls $TR/doc*.pt 2>/dev/null | wc -l)"
    [ $rc -ne 0 ] && { echo "[abort] teacher 失败"; exit 1; }
    # 教师占的那张卡用完就还回去
    for g in $HELD; do rmdir "$L/$g" 2>/dev/null; done; HELD=""
fi

BASE="--arch chead --d_kv 128 --dim 128 --epochs 40 --freeze_alpha --lam_global 1.0 \
--lr 0.0003 --n_pairs 256 --pair_w linear --slots 8 --split_seed 42 \
--traces $TR --val_frac 0.25 --scale global --alpha_max 1.0 --alpha_init 0.999"
PIDS=""
for s in 0 1 2; do
    out="varikv/chr03_s$s.pt"
    [ -f "$out/memoryless.pt" ] && { echo "[skip] $out"; continue; }
    grab; G=$GRABBED
    echo "[train] seed $s -> GPU$G  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$G setsid nohup .venv/bin/python -u scratch_ctrl_train.py \
        $BASE --seed "$s" --out "$out" > "scratch_ctrl_logs/train_chr03_s$s.log" 2>&1 &
    sleep 8
    # setsid 会 fork，$! 不是真 PID：从 /proc 里按 --out 找（**禁止 pgrep -f**）
    for p in $(ps -eo pid,args | grep '[s]cratch_ctrl_train.py' | awk '{print $1}'); do
        c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
        case "$c" in *"--out $out"*) PIDS="$PIDS $p";; esac
    done
    sleep 30
done
echo "[wait] $PIDS"
for p in $PIDS; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
echo "[done] $(date +%H:%M:%S)  ckpt=$(ls -d varikv/chr03_s*.pt 2>/dev/null | wc -l)"
