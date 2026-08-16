# 跨调度器的 GPU 占用锁。source 进来用。
#
# 为什么需要：每个调度器只知道**自己**派出去的 PID，不知道别的调度器占了哪张卡。
# 三个调度器同时扫 0-7，就会把作业叠到同一张卡上直到 OOM（已发生：GPU 0 上
# 30.47 + 30.14 GB 两个进程，第三个作业 OOM）。CLAUDE.md 早记过这条。
#
# `mkdir` 是原子的：成功即抢到，失败即已被占。比 nvidia-smi 查显存可靠——
# 评测作业在生成阶段显存会掉到 2GB 以下，按显存判会重复派发。
GPULOCK=/tmp/varikv_gpulock
mkdir -p "$GPULOCK"

gpu_claim() {   # 抢一张卡，阻塞直到成功；回显卡号
    #
    # **两道检查缺一不可。** 锁只挡得住调度器之间的竞态；挡不住卡上还有**别人**
    # 遗留的作业（竞态期间已经叠上去的、或手工起的）。所以拿到锁之后还要确认
    # 卡上真的没有计算进程 —— 有就立刻放锁重试，否则第二个作业照样 OOM。
    while true; do
        for g in ${GPU_CANDIDATES:-0 1 2 3 4 5 6 7}; do
            if mkdir "$GPULOCK/$g" 2>/dev/null; then
                if [ "$(nvidia-smi -i $g --query-compute-apps=pid \
                        --format=csv,noheader | wc -l)" -ne 0 ]; then
                    rmdir "$GPULOCK/$g" 2>/dev/null; continue   # 卡上有别人，放回
                fi
                echo "$$" > "$GPULOCK/$g/owner"
                echo "$g"; return
            fi
            # 锁的持有者已经死了 ⇒ 回收（防止调度器被 kill 后卡永久锁死）
            o=$(cat "$GPULOCK/$g/owner" 2>/dev/null)
            [ -n "$o" ] && ! kill -0 "$o" 2>/dev/null && rm -rf "$GPULOCK/$g"
        done
        sleep 45
    done
}

gpu_release() { rm -rf "$GPULOCK/$1"; }
