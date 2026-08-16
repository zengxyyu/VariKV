#!/bin/bash
ps -eo pid,cmd | grep "[t]ag _cc" | awk '{print $1}' | xargs -r kill
ps -eo pid,cmd | grep "[s]cratch_cc_grid" | awk '{print $1}' | xargs -r kill
sleep 3
