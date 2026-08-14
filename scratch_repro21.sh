#!/usr/bin/env bash
# 用 **v1 一模一样的代码** 重训 4 个模型，再各自评测，看 +21.60 是不是可复现的。
#
# 隔离方式：`git worktree` 在 222cef7（v1 训练时的 HEAD）单独检出到
# /home/ubuntu/zxy/vlm-memory-repro21。主仓库工作区一个字节不动，ckpt 与 results
# 都写进 worktree 自己的目录，不可能覆盖现有结果。
#
# 三个已排除的陷阱：
#  1. `eval_chunk.py:22` 有硬编码 `sys.path.insert(0,"/home/ubuntu/zxy/vlm-memory")`，
#     不改的话 worktree 里会导入**今天**的 varikv/memory.py（`c216a45` 改过随机读出
#     和 slot codebook，直接影响 dist）。已在 worktree 内改指向自己。
#  2. `0bd84fb`（07:30，v1 跑到一半时的提交）动了 memcache_retain.py，但那 15 行
#     全在 `if self.train_write:` 里，评测 train_write=False ⇒ 不影响。
#  3. v1 优化器只拿 `mem.parameters()` ⇒ backbone 从未被更新，尽管当时没有显式冻结。
#     `0bd84fb` 的 "freeze the backbone" 是卫生性修正，不是说 v1 训了 7B。
#
# argv 由日志指纹反推并**已验证**：`wrapper.py:236` 的短上下文自适应窗口
# (`clen<chunk ⇒ window=0.1·clen`) 使窗口序列成为 argv 的确定性指纹；probe 跑出的
# 310 310 279 279 215 215 2095 2095 259 259 与 v1 日志逐一吻合。
# 同时它证明 v1 用的是 `ctx_pos=tail`（旧默认）——random 会让 ctx 恒为 max_ctx、
# 窗口变成常数。CLAUDE.md 说 kl 目标"在随机放置的窗口后监督"，那是**修正后**的行为。
#
# **v1 没有 --seed**（`--seed` 是 `0bd84fb` 才加的）。所以"同代码同参数重跑"天然就是
# 换一次随机轨迹 —— 这正是要测的东西：+21.60 是稳定结果还是一次幸运轨迹。
set -u
W=/home/ubuntu/zxy/vlm-memory-repro21
MAIN=/home/ubuntu/zxy/vlm-memory
PY=$MAIN/.venv/bin/python
LOGS=$MAIN/scratch_repro21_logs; mkdir -p "$LOGS"
N=${N:-4}
TRAIN_ARGS="--obj kl --residual --mode dist --num_slots 16 --ratio 0.1 \
--max_ctx 32768 --chunk 16000 --window 4096 --target_len 256 \
--kl_weight sensitive --steps 1500 --log_every 50"

free_gpu() {
  for g in 0 1 2 3 4 5 6 7; do
    u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $g)
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] || continue
    sleep 20
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
    [ "$n" = "0" ] && { echo "$g"; return; }
  done
  echo ""
}

# ---------- 阶段 1：训练 N 个副本 ----------
echo "[$(date -u +%H:%M)] 阶段 1：训练 $N 个副本（v1 代码，无 seed ⇒ 轨迹各不相同）"
for i in $(seq 1 $N); do
  [ -f "$W/varikv/ckpt_r$i/s2b_dist_k16.pt" ] && { echo "  跳过 r$i（已存在）"; continue; }
  while :; do g=$(free_gpu); [ -n "$g" ] && break; sleep 60; done
  echo "[$(date -u +%H:%M)] train r$i → GPU$g"
  ( cd "$W" && CUDA_VISIBLE_DEVICES=$g $PY -u scratch_stage2b_train.py $TRAIN_ARGS \
      --out varikv/ckpt_r$i > "$LOGS/train_r$i.log" 2>&1
    echo "[$(date -u +%H:%M)] train r$i 退出 $?" ) &
  sleep 30
done
wait
echo "[$(date -u +%H:%M)] 训练全部结束"

# ---------- 阶段 2：逐个评测（与 v1 完全相同的评测命令）----------
echo "[$(date -u +%H:%M)] 阶段 2：评测"
for i in $(seq 1 $N); do
  CK=$W/varikv/ckpt_r$i/s2b_dist_k16.pt
  [ -f "$CK" ] || { echo "  r$i 无 ckpt，跳过"; continue; }
  [ -f "$LOGS/.done_eval_r$i" ] && { echo "  跳过 r$i 评测"; continue; }
  while :; do g=$(free_gpu); [ -n "$g" ] && break; sleep 60; done
  echo "[$(date -u +%H:%M)] eval r$i → GPU$g"
  ( cd "$W/external/FastKVzip/prefill" && \
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$g $PY -B eval_chunk.py \
      -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
      --prefill_chunk 16000 --window_size 4096 --level pair \
      -d scbench_kv --tag "_r$i" \
      --varikv_ckpt "$CK" --varikv_residual --varikv_slots 16 \
      > "$LOGS/eval_r$i.log" 2>&1
    tail -3 "$LOGS/eval_r$i.log" | grep -q "Finished." && touch "$LOGS/.done_eval_r$i"
    echo "[$(date -u +%H:%M)] eval r$i 结束" ) &
  sleep 30
done
wait
echo "[$(date -u +%H:%M)] ALL DONE"
