#!/bin/bash
# RESULTS_GRID.md 的自动刷新看门狗。
#
# 表格**从来不会自己更新** —— 它由 `scratch_all_report.py --write` 生成，
# 一次约 4 分钟（要扫全部 panel x ratio x 臂的逐样本 JSON）。整夜跑实验时
# 很容易忘记重跑，于是表格停在几小时前。这个脚本消除那个失误面。
#
# 做法：每 10 分钟看一次结果目录里最新的 mtime；只要它比表格新，就重算，
# 然后跑跨实现复核。**去抖**：连续两次探测都没有更新才算稳定，避免在一批
# 作业写盘中途生成半截表。
R=/home/ubuntu/zxy/vlm-memory
RES=$R/external/FastKVzip/prefill/results
LOG=$R/scratch_ctrl_logs/grid_refresh.log
cd $R
while :; do
  newest=$(find $RES -maxdepth 2 -name 'output-*.json' -newer RESULTS_GRID.md -print -quit 2>/dev/null)
  if [ -n "$newest" ]; then
    sleep 600                                  # 去抖：等这一批写完
    n2=$(find $RES -maxdepth 2 -name 'output-*.json' -newer RESULTS_GRID.md -print -quit 2>/dev/null)
    if [ -n "$n2" ]; then
      echo "$(date +%m-%d_%H:%M) 检测到新结果，重算表格" >> $LOG
      .venv/bin/python scratch_all_report.py --write RESULTS_GRID.md >> $LOG 2>&1
      echo "$(date +%m-%d_%H:%M) 重算完成，跑跨实现复核" >> $LOG
      .venv/bin/python scratch_verify_grid.py >> $LOG 2>&1
      echo "$(date +%m-%d_%H:%M) 完成" >> $LOG
    fi
  fi
  sleep 600
done
