#!/bin/bash
# 纯工作池：8 个常驻 worker，各占一张卡，从共享队列 `.mq_jobs` 原子取任务。
#
# 与 `scratch_master_queue.sh` 的区别：那个脚本把"分阶段编排"和"执行池"混在一起，
# 阶段跑完就退出，于是后来往队列里追加的作业**没人取**（实测：队列 45 个、
# 调度器 0 个、只有 2 个无关作业在跑）。这个脚本只做执行，队列空了就等，
# 所以任何时候往 `.mq_jobs` 追加都会被自动捡起来。
#
# 不按 PID 回收锁 —— 后台子进程结束后是僵尸，`kill -0` 对僵尸返回成功，锁永不释放
# （已实测把 8 张卡全锁死）。这里每个 worker 全程持有自己那张卡，退出时才释放。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
Q=$ROOT/scratch_ctrl_logs/.mq_jobs
cd "$ROOT" || exit 1
. "$ROOT/scratch_gpu_lock.sh"
touch "$Q"

pop_job() {   # 原子弹出队首；队空返回 1
    ( flock 9
      line=$(head -1 "$Q" 2>/dev/null)
      [ -z "$line" ] && exit 1
      sed -i '1d' "$Q"
      printf '%s' "$line" ) 9>"$Q.lock"
}

IDLE_EXIT=${IDLE_EXIT:-900}    # 连续空转这么久就收工，别永远占着卡
pids=()
for i in 0 1 2 3 4 5 6 7; do
    ( G=$(gpu_claim)
      idle=0
      while :; do
          if j=$(pop_job); then
              idle=0
              echo "$(date +%H:%M)  [GPU$G] ${j:0:130}"
              # **必须 export，不能写成前缀赋值。** 作业串形如
              #     cd X && env VAR=... python ...
              # 而 `CUDA_VISIBLE_DEVICES=$G cd X && env ... python`
              # 里那个赋值**只对 `cd` 生效**，`&&` 之后的 python 拿不到 ⇒ 它看到
              # 全部 8 张卡自己挑，多个作业叠到同一张卡上（实测 GPU1/GPU2 各叠了
              # 3 个进程，显存 34 GB）。用子 shell + export 才能覆盖整串。
              ( export CUDA_VISIBLE_DEVICES=$G; eval "$j" )
          else
              sleep 30; idle=$((idle+30))
              [ "$idle" -ge "$IDLE_EXIT" ] && break
          fi
      done
      gpu_release "$G"
      echo "$(date +%H:%M)  [GPU$G] worker 收工" ) &
    pids+=($!); sleep 3
done
wait "${pids[@]}"
echo "$(date +%H:%M)  池子全部收工"
