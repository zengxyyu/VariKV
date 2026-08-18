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

# **每卡一把锁，不能只查「卡上有没有进程」。** 实测踩过：同卡已有一个 topup
# worker 正处在两个作业之间的轮询间隙时，卡上确实是空的，于是第二个 topup 也起来了，
# 两个一起抢队列 ⇒ GPU0 上叠了 2 个作业、53 GB，而另外 4 张卡全空。
# mkdir 是原子的；锁里记 PID，持有者死了就回收。
TLOCK=/tmp/varikv_topup/$G
mkdir -p /tmp/varikv_topup
if ! mkdir "$TLOCK" 2>/dev/null; then
    o=$(cat "$TLOCK/pid" 2>/dev/null)
    if [ -n "$o" ] && kill -0 "$o" 2>/dev/null; then
        echo "GPU$G 已有 topup worker (pid $o)，退出"; exit 0
    fi
    rm -rf "$TLOCK"; mkdir -p "$TLOCK"        # 持有者已死，回收
fi
echo $$ > "$TLOCK/pid"
trap 'rm -rf "$TLOCK"' EXIT

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
