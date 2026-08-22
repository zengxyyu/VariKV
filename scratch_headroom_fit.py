#!/usr/bin/env python3
"""逐 panel 拟 `Δ = a·headroom + b` —— **数据直接从生成的表读，不手抄**。

问的是：静态 `u` 表在一个 panel 上能收回多少 headroom（斜率 a），
以及它在那里额外造成多少与 headroom 无关的恒定损伤（截距 b）。
"""
import re
import subprocess
import sys

import numpy as np

# 透传 `--pre`，同一份拟合口径服务多张表（kv `_uq01` / psyn `_up01` / r05 `_ur5`）。
_ARGS = sys.argv[1:]
TAB = subprocess.run([sys.executable, "scratch_utab_report.py"] + _ARGS,
                     capture_output=True, text=True).stdout
rows = {}
for ln in TAB.splitlines():
    m = re.match(r"\| ([^|]+?) \| ([\d.]+) \| ([-+][\d.]+) \| [\d.]+ \| [\d.]+ \| "
                 r"\*\*([-+][\d.]+)\*\*", ln)
    if m:
        rows.setdefault(m.group(1).strip(), []).append(
            (float(m.group(2)), float(m.group(3)), float(m.group(4))))

print(f"{'panel':16s}{'n':>3s}{'斜率 a':>9s}{'截距 b':>9s}{'RMS':>7s}{'Pearson':>9s}"
      f"   最大 headroom 格的 Δ")
for p, rs in sorted(rows.items(), key=lambda kv: -len(kv[1])):
    if len(rs) < 3:
        print(f"{p:16s}{len(rs):>3d}   （少于 3 点，不拟）")
        continue
    h = np.array([r[1] for r in rs]); d = np.array([r[2] for r in rs])
    a, b = np.polyfit(h, d, 1)
    res = d - (a * h + b)
    i = int(np.argmax(h))
    print(f"{p:16s}{len(rs):>3d}{a:>+9.3f}{b:>+9.2f}"
          f"{np.sqrt((res ** 2).mean()):>7.2f}{np.corrcoef(h, d)[0, 1]:>+9.3f}"
          f"   ρ={rs[i][0]} headroom {h[i]:+.2f} → Δ {d[i]:+.2f}")
