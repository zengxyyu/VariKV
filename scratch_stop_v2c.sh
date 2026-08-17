#!/bin/bash
# 干净版 v2 的评测全部跑在 mode=stateful 上（--ctrlm_mode 默认值覆盖了 ckpt），
# 而 compat ckpt 的 writer 模块是填零的 ⇒ 结果全部作废。停掉并清理。
for p in $(pgrep -f 'eval_chunk[.]py.*_v2c_s'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f 'eval_chunk[.]py.*_v2c_s'); do kill -9 "$p" 2>/dev/null; done
sleep 2
R=/home/ubuntu/zxy/vlm-memory
rm -rf $R/external/FastKVzip/prefill/results/*/*_v2c_s*
rm -f $R/scratch_ctrl_logs/v2cbench_*.log $R/scratch_ctrl_logs/.done_v2c_*
echo "残留 v2c 进程: $(pgrep -cf 'eval_chunk[.]py.*_v2c_s')"
echo "残留 v2c 结果目录: $(ls -d $R/external/FastKVzip/prefill/results/*/*_v2c_s* 2>/dev/null|wc -l)"
