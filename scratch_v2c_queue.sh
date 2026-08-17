#!/bin/bash
# 干净版 v2：3 种子 × 11 panel × 9 ratio，**用修复后的 mode 解析重跑**。
#
# ── 为什么要重跑 ──────────────────────────────────────────────────────
# 上一批 33 个作业全部跑在 `mode=stateful` 上：`--ctrlm_mode` 曾默认 `"stateful"`，
# 而 eval_chunk 用 `args.ctrlm_mode or _ck["mode"]` —— 非空字符串恒为真，ckpt 存的
# `memoryless` 从来不生效。日志实证 `v2c_* → mode=stateful` 而 `ctrl_b_a1_* →
# mode=memoryless`，两批根本不在同一个方法下。已修（默认改 None + 覆盖时告警），
# 旧结果已全部删除。
#
# ── 与原版 v2 逐字段对齐（核对过，不是凭印象）─────────────────────────
# 原版实际调用（`scratch_ctrl_bench2.sh` / `bench5.sh`，并由结果目录名反证）：
#     -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100
#     --prefill_chunk 16000 --window_size 4096 --level pair
#     --ctrlm_mode memoryless      其余全默认
# 目录名 `__b2memoryless_chunk16k_w4096_ctrlmmemo8` **没有 `_a…` 也没有 `_rm…`**
# ⇒ 反证 `--ctrlm_alpha`(默认 −1.0=不覆盖) 与 `--ctrlm_rho_max`(默认 1.0) 都没传；
# `ctrl_seed 0` / `ctrlm_slots 8` / `ctrlm_dim 128` 同为默认。
# 加载后的模块签名也一致：`mode=memoryless slots=8 typed=True alpha=0.9990
# params=637.8K`，两边逐字相同。
#
# 本脚本**显式传 `--ctrlm_mode memoryless`**。虽然修复后不传也会跟随 ckpt，但显式写
# 出来才能在日志里自证，不依赖"默认值这次是对的"。
#
# 三处有意不同，都不影响与原版在 ratio 0.1 上的可比性：
#   · 11 个 panel（原版只跑 scbench_kv）
#   · 9 个 ratio（原版只跑 0.1）—— 每个 ratio 走**独立的分块预填**，所以 0.1 那格
#     与只跑 0.1 时逐位相同
#   · ckpt 是干净版训练产物的 compat 导出（这正是要检验的对象）
#
# ratio 0.02 是**阴性对照**：`ratio×clen < window` ⇒ 只留最近的 token、门控分数
# 不参与 ⇒ Δ 应恒为 0，非零就说明实现坏了。
set -u
ROOT=/home/ubuntu/zxy/vlm-memory
LOG=$ROOT/scratch_ctrl_logs
Q=$LOG/.mq_jobs
PANELS="scbench_kv scbench_vt scbench_mf scbench_qa_eng scbench_choice_eng \
scbench_summary scbench_prefix_suffix scbench_repoqa scbench_many_shot squad gsm"
R9=1.0,0.75,0.5,0.4,0.3,0.2,0.1,0.05,0.02
cd "$ROOT" || exit 1

for S in 0 1 2; do
    [ -f "varikv/v2c_s${S}_compat.pt" ] || { echo "✗ 缺 v2c_s${S}_compat.pt"; exit 1; }
done

# 追加到主队列（共用同一把 flock），不另起调度器 —— 两个调度器各扫各的卡会叠作业。
J=()
for S in 0 1 2; do for P in $PANELS; do
    [ -f "$LOG/.done_v2c_${P}_s$S" ] && continue
    J+=("cd $ROOT/external/FastKVzip/prefill && env VARIKV_RATIOS=$R9 \
$ROOT/.venv/bin/python -B eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip \
--num 100 --prefill_chunk 16000 --window_size 4096 --level pair -d $P \
--tag _v2c_s$S --ctrlm_mode memoryless \
--ctrlm_ckpt ../../../varikv/v2c_s${S}_compat.pt \
> $LOG/v2cbench_${P}_s$S.log 2>&1")
done; done
( flock 9; printf '%s\n' "${J[@]}" >> "$Q" ) 9>"$Q.lock"
echo "$(date +%H:%M)  追加 ${#J[@]} 个干净版 v2 评测，队列现有 $(wc -l < "$Q") 个"

# 等排空后统一打 marker，并**逐个核对日志里的 mode**（这次绝不再默认它是对的）
while [ -s "$Q" ] || pgrep -f 'eval_chunk[.]py.*_v2c_s' >/dev/null; do sleep 120; done
BAD=0
for S in 0 1 2; do for P in $PANELS; do
    f="$LOG/v2cbench_${P}_s$S.log"
    tail -3 "$f" 2>/dev/null | grep -q Finished && touch "$LOG/.done_v2c_${P}_s$S"
    grep -q "mode=memoryless" "$f" 2>/dev/null || BAD=$((BAD+1))
done; done
echo "$(date +%H:%M)  干净版 v2 完成；**mode 不是 memoryless 的作业数 = $BAD**（必须为 0）"
