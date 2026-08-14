#!/bin/bash
# **泛化才是这个 +4.27 能不能成篇的分水岭。** v1 就是死在这里（8 panel 均值 +1.41）。
# headroom 表：Retr.KV 是唯一有真余量的 panel，Retr.MultiHop 与 En.QA 的余量是**负的**
# （压缩比满缓存还好），所以 vt 是危险 panel、必须测。
# 每个 panel 都跑基线 + memoryless，因为 ratio 0.1 的基线大多没有现成的。
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill || exit 1
freegpu() {
    while true; do
        for g in 0 1 2 3 4 5 6 7; do
            [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader|wc -l)" -eq 0 ] \
                && { echo $g; return; }
        done; sleep 120
    done
}
run() {  # $1=dataset $2=arm(base|mem)
    G=$(freegpu)
    if [ "$2" = base ]; then EXTRA=""; TAG="_pnb"; else
        EXTRA="--ctrlm_mode memoryless --ctrlm_ckpt ../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt"
        TAG="_pnm"; fi
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$G nohup ../../../.venv/bin/python -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d "$1" \
        --tag "$TAG" $EXTRA > "../../../scratch_ctrl_logs/panel_${1}_$2.log" 2>&1 &
    sleep 60
}
for D in scbench_vt scbench_prefix_suffix gsm scbench_many_shot scbench_choice_eng squad; do
    run "$D" base
    run "$D" mem
done
wait
