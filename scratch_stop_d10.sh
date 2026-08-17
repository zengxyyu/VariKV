#!/bin/bash
pkill -f 'scratch_d10_arms[.]sh'
sleep 2
for d in /tmp/varikv_gpulock/*; do
    [ -d "$d" ] || continue; g=$(basename "$d")
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g")" ] \
        && rm -rf "$d" && echo "释放 GPU$g"
done
