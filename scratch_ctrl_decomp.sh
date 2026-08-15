#!/bin/bash
# P0：把 memoryless 的 +4.27 拆开。只用 GPU 0/1（其余 6 张在跑 ratio 0.05 全景）。
# 同一批 trace、同一 split、同一 α、同一 epoch 数，唯一变量是**控制器能看到什么**。
cd /home/ubuntu/zxy/vlm-memory || exit 1
wait_gpu() {  # 只在 0/1 里等
    while true; do
        for g in 0 1; do
            [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader|wc -l)" -eq 0 ] \
                && { echo $g; return; }
        done; sleep 90
    done
}
for A in affine bias scalar kv; do
  for S in 0 1 2; do
    O="varikv/dec_${A}_s$S.pt"; [ -d "$O" ] && continue
    G=$(wait_gpu)
    CUDA_VISIBLE_DEVICES=$G nohup .venv/bin/python -u scratch_ctrl_train.py \
        --traces scratch_ctrl_traces_v2 --epochs 40 --seed "$S" --split_seed 42 \
        --arch "$A" --alpha_init 1.0 --freeze_alpha --pair_w linear --lam_global 1.0 \
        --out "$O" > "scratch_ctrl_logs/dec_${A}_s$S.log" 2>&1 &
    sleep 45
  done
done
wait
echo "=== 训练完成，下游评测 seed 0 的四个架构 ==="
cd external/FastKVzip/prefill || exit 1
for A in affine bias scalar kv; do
    G=$(wait_gpu 2>/dev/null || echo 0)
    while [ "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ] \
       && [ "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ]; do sleep 90; done
    G=0; [ "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader|wc -l)" -ne 0 ] && G=1
    VARIKV_RATIOS=0.1 CUDA_VISIBLE_DEVICES=$G nohup ../../../.venv/bin/python -B \
        eval_chunk.py -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
        --prefill_chunk 16000 --window_size 4096 --level pair -d scbench_kv \
        --tag "_dc$A" --ctrlm_ckpt "../../../varikv/dec_${A}_s0.pt/memoryless.pt" \
        > "../../../scratch_ctrl_logs/dec_bench_$A.log" 2>&1 &
    sleep 90
done
wait
