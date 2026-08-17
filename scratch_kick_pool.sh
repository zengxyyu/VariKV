#!/bin/bash
cd /home/ubuntu/zxy/vlm-memory
setsid ./scratch_pool.sh > scratch_ctrl_logs/pool_run.log 2>&1 < /dev/null &
disown
