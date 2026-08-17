#!/bin/bash
# 单卡 top-up worker：补 `scratch_pool.sh` 里因 IDLE_EXIT 收工而空出来的卡。
#
# 为什么需要：池子的 worker 空转 900 秒就收工释放卡（设计如此，"别永远占着卡"）。
# 但在**间歇性排队**的用法下——队列空一阵、然后又追加作业——收工的 worker 不会
# 回来，于是出现"卡空着、队列有活"。实测 GPU0 空闲而队列 2 个。
#
# 用法：./scratch_topup.sh <gpu>   幂等：卡上有进程就直接退出，不会叠。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
Q=$ROOT/scratch_ctrl_logs/.mq_jobs
G=${1:?用法: scratch_topup.sh <gpu>}
IDLE_EXIT=${IDLE_EXIT:-7200}      # 比池子的 900 长得多：这里就是为间歇排队服务的

n=$(nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" -ne 0 ] && { echo "GPU$G 上已有 $n 个进程，不叠，退出"; exit 0; }

pop_job() {   # 与池子同一把 flock，原子弹出队首
    ( flock 9
      line=$(head -1 "$Q" 2>/dev/null)
      [ -z "$line" ] && exit 1
      sed -i '1d' "$Q"
      printf '%s' "$line" ) 9>"$Q.lock"
}

cd "$ROOT" || exit 1
idle=0
while :; do
    if j=$(pop_job); then
        idle=0
        echo "$(date +%H:%M)  [topup GPU$G] ${j:0:130}"
        # 与池子同样的坑：必须 export，前缀赋值只对 `cd` 生效
        ( export CUDA_VISIBLE_DEVICES=$G; eval "$j" )
    else
        sleep 30; idle=$((idle+30))
        [ "$idle" -ge "$IDLE_EXIT" ] && break
    fi
done
echo "$(date +%H:%M)  [topup GPU$G] 收工"
