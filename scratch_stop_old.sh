#!/bin/bash
# 停掉被新计划取代的作业：r005 全景（只有 0.05）与 ratio 迁移（只有 0.4/0.3/0.2）。
# **不动 GPU 0/1 上的拆解训练**——它们与评测共卡、且是另一条独立的线。
pkill -f "scratch_r005_sweep.sh"
pkill -f "scratch_ctrl_ratio.sh"
pkill -f "tag _r5m"; pkill -f "tag _r5b"; pkill -f "tag _rtb"; pkill -f "tag _rtm"
sleep 5
