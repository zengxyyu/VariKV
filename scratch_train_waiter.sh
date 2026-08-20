#!/bin/bash
# 把剩余的 chead10 训练放上 GPU，严格一卡一进程，且**不与评测 worker 抢卡**。
#
# 为什么不能简单轮询空卡：/tmp/vq/worker.sh 有 8 个实例，各自也在轮询，
# 且用 **每卡目录锁 /tmp/vq/lock/<g>** 互斥。直接抢必输（实测 GPU6 被 _q3mL08 抢走）。
# 正确做法两条一起用：
#   ① 暂停队列 —— jobs.txt 为空时 worker 只 sleep，不占卡；
#   ② 用**同一把目录锁**原子获取（mkdir 成功才算拿到），与 worker 同规则竞争。
# 两者都在 EXIT trap 里恢复，脚本被杀也不会把队列卡死。
set -u
cd "$(dirname "$0")"

# ⚠ 单实例保护。2026-08-20 15:14 的事故：脚本改过后重启，**只杀了一个实例**，
# 两个等卡器并存，各自把 seed 2 派到不同卡 —— 同一个种子跑了两份，
# 交叉写同一个 ckpt 与同一个日志，产物只能全部作废重来。
SELFLOCK=/tmp/vq/waiter.lock
mkdir "$SELFLOCK" 2>/dev/null || { echo "[abort] 已有等卡器在运行（$SELFLOCK 存在）"; exit 1; }
trap 'rmdir "$SELFLOCK" 2>/dev/null' EXIT
L=/tmp/vq/lock; Q=/tmp/vq/jobs.txt; HOLD=/tmp/vq/jobs.hold_waiter
# 可参数化：ARCH / ALPHA_INIT / ALPHA_MAX / PREFIX 由环境变量给，默认沿用
# 首次使用时的 chead10 配置，**老的调用方式行为不变**。
ARCH="${ARCH:-chead}"; A_INIT="${A_INIT:-0.999}"; A_MAX="${A_MAX:-1.0}"
PREFIX="${PREFIX:-chead10}"
BASE="--arch $ARCH --d_kv 128 --dim 128 --epochs 40 --freeze_alpha --lam_global 1.0 \
--lr 0.0003 --n_pairs 256 --pair_w linear --slots 8 --split_seed 42 \
--traces scratch_ctrl_traces_v2_10 --val_frac 0.25 --scale global \
--alpha_max $A_MAX --alpha_init $A_INIT"
HELD=""; PIDS=""

cleanup() {
    for g in $HELD; do rmdir "$L/$g" 2>/dev/null; done
    if [ -f "$HOLD" ]; then cat "$HOLD" >> "$Q"; rm -f "$HOLD"; echo "[restore] 队列已恢复"; fi
}
trap 'cleanup; rmdir "$SELFLOCK" 2>/dev/null' EXIT

# ① 暂停队列。**NOHOLD=1 时跳过** —— 当评测队列本身就是主交付物、
# 且在跑的作业很长（全网格一个作业 7 个 ratio x 100 样本，数小时）时，
# 暂停队列会把交付物卡住，而训练照样拿不到卡。那种情况下只公平抢锁即可。
if [ "${NOHOLD:-0}" != "1" ] && [ -s "$Q" ]; then
    cp "$Q" "$HOLD"; : > "$Q"; echo "[hold] 队列暂存 $(grep -cve '^\s*$' "$HOLD") 行"
fi

occupied() {   # 该卡是否有 compute 进程
    local uuid; uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
                       | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -qF "$uuid"
}

for s in 0 1 2; do
    out="varikv/${PREFIX}_s$s.pt"
    [ -e "$out" ] && { echo "[skip] $out 已存在"; continue; }
    placed=0
    while [ $placed -eq 0 ]; do
        for g in 0 1 2 3 4 5 6 7; do
            occupied "$g" && continue
            # ⚠ 若这把锁**已经是我们自己**持有的（该卡上一个训练已结束、卡已空），
            # 就不能再 mkdir —— 它必然失败，导致我们永远用不了自己占着的卡。
            if ! printf ' %s ' $HELD | grep -q " $g "; then
                mkdir "$L/$g" 2>/dev/null || continue    # 原子获取，与 worker 同规则
                HELD="$HELD $g"
            fi
            sleep 20
            if occupied "$g"; then rmdir "$L/$g"; continue; fi
            echo "[launch] seed $s -> GPU$g  $(date +%H:%M:%S)"
            CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python -u \
                scratch_ctrl_train.py $BASE --seed "$s" --out "$out" \
                > "scratch_ctrl_logs/train_${PREFIX}_s$s.log" 2>&1 &
            PIDS="$PIDS $!"
            sleep 30
            placed=1; break
        done
        [ $placed -eq 0 ] && sleep 30
    done
done
# 收尾等待：只等**本脚本自己启动的 PID**。
# 本项目禁止 pgrep/pkill -f —— 按名字匹配会误伤同名的其他作业。
echo "[wait] 等自启的 PID 收尾: $PIDS"
for pid in $PIDS; do while kill -0 "$pid" 2>/dev/null; do sleep 20; done; done
echo "[done] $(date +%H:%M:%S)"
