#!/bin/bash
# 三件事，按优先级串在一张卡上：
#  1. 修好的信度曲线（等预算互换，ε 相对 |S|，同时测 U^attn）—— 撤回结论的替代实验
#  2. 同一探针的 boundary 档 —— v2 残差实际所处的工作点
#  3. 四臂的机制诊断（头内秩保持 + 逐(层,头)预算再分配）
set -u
ROOT=/home/ubuntu/zxy/vlm-memory; LOG=$ROOT/scratch_ctrl_logs
. "$ROOT/scratch_gpu_lock.sh"
G=$(gpu_claim); PY="$ROOT/.venv/bin/python"
run(){ echo "$(date +%H:%M) [GPU$G] $1"; shift; CUDA_VISIBLE_DEVICES=$G "$PY" -B "$@"; }

run "信度曲线冒烟" "$ROOT/scratch_probe_nll_stab.py" --num 1 --n_cand 4 \
    --eps 0.001 0.02 > "$LOG/nllstab2_smoke.log" 2>&1
if grep -q "^判读" "$LOG/nllstab2_smoke.log"; then
    run "信度曲线 random" "$ROOT/scratch_probe_nll_stab.py" --num 6 --n_cand 20 \
        --swap_mode random > "$LOG/nllstab2_random.log" 2>&1
    run "信度曲线 boundary" "$ROOT/scratch_probe_nll_stab.py" --num 6 --n_cand 20 \
        --swap_mode boundary > "$LOG/nllstab2_boundary.log" 2>&1
else
    echo "冒烟失败"; tail -25 "$LOG/nllstab2_smoke.log"
fi
for A in bias affine scalar kv; do
    run "臂诊断 $A" "$ROOT/scratch_probe_armdiag.py" \
        --ckpt "$ROOT/varikv/dec_${A}_s0.pt/memoryless.pt" --num 3 \
        > "$LOG/armdiag_$A.log" 2>&1
done
gpu_release "$G"
echo "$(date +%H:%M) P31 全部完成"
