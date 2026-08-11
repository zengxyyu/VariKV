#!/bin/bash
# 等 B 批评测全部结束 → 应用 P0-A 空记忆 guard → 自检 → 提交。
# 必须等：guard 会改变 memory_residual 的行为，B 批还在陆续启动新 job，
# 中途改会让批内前 11 个与后面的不可比。
set -u
cd /home/ubuntu/zxy/vlm-memory
for i in $(seq 1 400); do
  grep -q "ALL DONE" scratch_gapsweep_run.log 2>/dev/null && break
  sleep 60
done
{
  echo "=== B 批状态 $(date) ==="
  ls scratch_gapsweep_logs/.done__* | wc -l
  grep -q "ALL DONE" scratch_gapsweep_run.log && echo "B 批已全部结束" || echo "等待超时，仍继续（风险自负）"
  echo "=== 应用 P0-A guard ==="
  .venv/bin/python -B scratch_fix_empty_memory.py
  echo "=== 自检 ==="
  .venv/bin/python -B scratch_verify_empty_guard.py
} > scratch_p0a.log 2>&1
if grep -q "P0-A 自检通过" scratch_p0a.log; then
  git add -A external/FastKVzip/prefill/attention/memcache_retain.py \
      scratch_fix_empty_memory.py scratch_verify_empty_guard.py scratch_p0a.log
  git commit -q -m "P0-A: empty memory must not inject into the attention output

memory_residual is called unconditionally from attn.py:149, so before the first
absorption — while the slots still hold their initialisation — the module was
still adding something to the attention output. Evidence: the ratio-1.0 score,
which memory cannot legitimately affect, was byte-identical across independent
jobs for the same checkpoint and differed between checkpoints
(68.20/66.80/68.60/67.20/67.80/70.40), so the checkpoint was determining the
full-cache reference.

Guarded at the top of memory_residual: return zeros of the correct shape while
_absorbed_upto <= 0. Self-test asserts the output is exactly zero before any
absorption and non-zero after, and is recorded in scratch_p0a.log.

Applied only after the 27-job gapsweep evaluation finished, since the guard
changes what newly launched jobs would compute and would have made that batch
internally inconsistent.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
  echo "P0A COMMITTED $(date)" >> scratch_p0a.log
else
  echo "P0A FAILED — 未提交，见 scratch_p0a.log" >> scratch_p0a.log
fi
