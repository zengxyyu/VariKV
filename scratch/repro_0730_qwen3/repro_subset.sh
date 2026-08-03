#!/usr/bin/env bash
# Stage 0 reproduce: 3 compression-exercising datasets, n=100 each, sharded across 8 H100s.
# Base names given; get_data_list substitutes _short/_mid for Qwen3 automatically.
cd /home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill
source /home/ubuntu/zxy/vlm-memory/.venv/bin/activate
LOGDIR=/home/ubuntu/zxy/vlm-memory/scratch/repro_0730_qwen3/repro_logs
mkdir -p "$LOGDIR"
M="Qwen/Qwen3-8B"

run() {  # gpu data idx num
  CUDA_VISIBLE_DEVICES=$1 python -B eval_chunk.py -g fastkvzip -m "$M" -d "$2" --idx "$3" --num "$4" \
    > "$LOGDIR/${2}_gpu$1_idx$3.log" 2>&1 &
}

echo "launch: $(date)"
# scbench_kv (retrieval) -> _short : GPUs 0,1,2
run 0 scbench_kv 0  34
run 1 scbench_kv 34 33
run 2 scbench_kv 67 33
# scbench_prefix_suffix (retrieval) -> _short : GPUs 3,4,5
run 3 scbench_prefix_suffix 0  34
run 4 scbench_prefix_suffix 34 33
run 5 scbench_prefix_suffix 67 33
# scbench_mf (numerical find) -> _mid : GPUs 6,7
run 6 scbench_mf 0  50
run 7 scbench_mf 50 50

wait
echo "ALL_SHARDS_DONE: $(date)"
