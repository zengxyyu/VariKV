#!/bin/bash
# 一次调用看全局。给自动续跑用 —— 每次醒来只跑这一条。
cd /home/ubuntu/zxy/vlm-memory
L=scratch_ctrl_logs
P="scbench_kv scbench_vt scbench_mf scbench_qa_eng scbench_choice_eng scbench_summary \
scbench_prefix_suffix scbench_repoqa scbench_many_shot squad gsm"
echo "===== $(date '+%m-%d %H:%M') ====="
echo "池子 worker $(pgrep -cf '[s]cratch_pool.sh')  eval 进程 $(pgrep -cf '[e]val_chunk')  队列 $(wc -l < $L/.mq_jobs 2>/dev/null)"
n=0; for g in 0 1 2 3 4 5 6 7; do c=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g|wc -l); n=$((n+c)); [ "$c" -gt 1 ] && echo "  ⚠ GPU$g 有 $c 个进程（应为 1）"; done
echo "GPU 上进程总数 $n/8"
echo "--- 因子臂 12 ---"
e=0
for A in sz szr szm szmr0; do for S in 0 1 2; do f=$L/d10bench_${A}_s$S.log
  if tail -3 $f 2>/dev/null|grep -q Finished; then e=$((e+1))
  else k=$(grep -cE 'Results saved' $f 2>/dev/null); [ "${k:-0}" -gt 0 ] && echo "  跑着 ${A}_s$S $k/100"; fi
done; done; echo "  完成 $e/12"
echo "--- 干净版 v2 33 ---"
d=0; for S in 0 1 2; do for x in $P; do tail -3 $L/v2cbench_${x}_s$S.log 2>/dev/null|grep -q Finished && d=$((d+1)); done; done
echo "  完成 $d/33"
b=0; for f in $L/v2cbench_*.log; do tail -3 $f 2>/dev/null|grep -q Finished || continue
  grep -q "mode=memoryless" $f || { b=$((b+1)); echo "  ✗ mode 不对: $(basename $f)"; }; done
[ "$b" -eq 0 ] && echo "  已完成作业的 mode 全部 memoryless ✓"
echo "--- 卡死检测（日志 30 分钟没动的在跑作业）---"
now=$(date +%s); stuck=0
for p in $(pgrep -f '[e]val_chunk.py'); do
  t=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null|grep -oE '\-\-tag [^ ]+'|sed 's/--tag //')
  dd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null|grep -oE '\-d [a-z_]+'|sed 's/-d //')
  for f in $L/*${dd}*.log; do case "$f" in *"$(echo $t|tr -d _)"*|*) ;; esac; done
done
for f in $L/v2cbench_*.log $L/d10bench_*.log; do
  [ -f "$f" ] || continue; tail -3 $f 2>/dev/null|grep -q Finished && continue
  k=$(grep -cE 'Results saved' $f 2>/dev/null); [ "${k:-0}" -eq 0 ] && continue
  age=$(( (now - $(stat -c %Y $f)) / 60 ))
  [ "$age" -gt 30 ] && { echo "  ⚠ $(basename $f) 已 $age 分钟无更新（$k/100）"; stuck=1; }
done
[ "$stuck" -eq 0 ] && echo "  无卡死"
