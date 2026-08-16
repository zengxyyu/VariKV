#!/bin/bash
pkill -f "scratch_cc_grid.sh"; pkill -f "tag _cc "; pkill -f 'tag _cc$'
ps -eo pid,cmd | grep "[t]ag _cc" | awk '{print $1}' | xargs -r kill
sleep 3
