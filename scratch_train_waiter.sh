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
L=/tmp/vq/lock; Q=/tmp/vq/jobs.txt; HOLD=/tmp/vq/jobs.hold_waiter
BASE="--arch chead --d_kv 128 --dim 128 --epochs 40 --freeze_alpha --lam_global 1.0 \
--lr 0.0003 --n_pairs 256 --pair_w linear --slots 8 --split_seed 42 \
--traces scratch_ctrl_traces_v2_10 --val_frac 0.25 --scale global \
--alpha_max 1.0 --alpha_init 0.999"
HELD=""; PIDS=""

cleanup() {
    for g in $HELD; do rmdir "$L/$g" 2>/dev/null; done
    if [ -f "$HOLD" ]; then cat "$HOLD" >> "$Q"; rm -f "$HOLD"; echo "[restore] 队列已恢复"; fi
}
trap cleanup EXIT

# ① 暂停队列
if [ -s "$Q" ]; then cp "$Q" "$HOLD"; : > "$Q"; echo "[hold] 队列暂存 $(grep -cve '^\s*$' "$HOLD") 行"; fi

occupied() {   # 该卡是否有 compute 进程
    local uuid; uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
                       | awk -F', ' -v i="$1" '$1==i{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -qF "$uuid"
}

for s in 0 1 2; do
    out="varikv/chead10_s$s.pt"
    [ -e "$out" ] && { echo "[skip] $out 已存在"; continue; }
    placed=0
    while [ $placed -eq 0 ]; do
        for g in 0 1 2 3 4 5 6 7; do
            occupied "$g" && continue
            mkdir "$L/$g" 2>/dev/null || continue        # 原子获取，与 worker 同规则
            sleep 20
            if occupied "$g"; then rmdir "$L/$g"; continue; fi
            echo "[launch] seed $s -> GPU$g  $(date +%H:%M:%S)"
            HELD="$HELD $g"
            CUDA_VISIBLE_DEVICES=$g setsid nohup .venv/bin/python -u \
                scratch_ctrl_train.py $BASE --seed "$s" --out "$out" \
                > "scratch_ctrl_logs/train_chead10_s$s.log" 2>&1 &
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
