#!/usr/bin/env python3
"""把 5 个基线方法与本项目的各臂合并成一张可比的表。

**覆盖面的硬事实（先说清楚，避免误读）**：
  · 5 方法复现（fastkvzip / kvzip / duoattn / expected / snapkv）：11 个数据集全覆盖。
  · 本项目的 `scalar` / `kv` 等因子臂：**只在 scbench_kv 上评过**，只有 ρ∈{0.1, 0.2}。
  ⇒ **ρ=0.2 是唯一的共同比例**，跨方法比较只能在那里做；其余 ratio 只有基线方法有。
"""
import contextlib
import glob
import io
import json
import os
import sys

import numpy as np

_PRE = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                    "external/FastKVzip/prefill")
sys.path.insert(0, _PRE)
_M = contextlib.redirect_stdout(io.StringIO())
MODEL = "qwen2.5-7b-instruct-1m"

PANEL = {"scbench_kv": "Retr.KV", "scbench_prefix_suffix": "Retr.PrefSuf",
         "scbench_repoqa": "Code.RepoQA", "squad": "SQuAD", "gsm": "GSM8K",
         "scbench_qa_eng": "En.QA", "scbench_choice_eng": "En.MultiChoice",
         "scbench_summary": "En.Summary", "scbench_vt": "Retr.MultiHop",
         "scbench_mf": "Math.Find", "scbench_many_shot": "ICL.ManyShot"}
# 复现里 5 个方法的目录后缀（kvzip 走 eval.py，目录无 gate 段也无 chunk 段）
BASE = [("FastKVzip", f"*_{MODEL}_fastkvzip_chunk16k_w4096"),
        ("KVzip", f"*_{MODEL}"),
        ("DuoAttn", f"*_{MODEL}_head_chunk16k_w4096"),
        ("ExpectAttn", f"*_{MODEL}_expect_chunk16k_w4096"),
        ("SnapKV", f"*_{MODEL}_snap_chunk16k_w4096")]


def score(ds, pat, ratio):
    from results.parse import parse_answer, evaluate_answer
    cwd = os.getcwd(); os.chdir(_PRE)
    try:
        with _M:
            ans, sub = parse_answer(ds)
        out = {}
        for f in sorted(glob.glob(f"results/{ds}/{pat}/output-*.json")):
            i = int(os.path.basename(os.path.dirname(f)).split("_")[0])
            try:
                d = json.load(open(f))
            except Exception:
                continue
            p, g = [], []
            for k in [x for x in d if x.startswith("qa")]:
                for info, rec in d[k]:
                    if abs(float(info[0]) - ratio) < 1e-9:
                        p.append(rec["pruned"]); g.append(rec["answer"])
            if p:
                with _M:
                    out[i] = float(np.mean(evaluate_answer(
                        p, ans[i] if ans else g, ds, "qa",
                        subtask=sub[i] if sub else None)))
        return out
    finally:
        os.chdir(cwd)


def boot(x, n=4000, s=0):
    rg = np.random.default_rng(s)
    b = np.array([x[rg.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


R = 0.2
print(f"## 表 A：5 个基线方法 × 11 panel，绝对分 @ρ={R}（各数据集全量）\n")
print(f"{'panel':<15}{'full':>7}" + "".join(f"{n:>12}" for n, _ in BASE))
rows = {}
for ds, nm in PANEL.items():
    # 复现批次只跑了 [0.2..0.75]，**没有 ratio 1.0** —— 满缓存参考必须取自
    # 我们自己的 `__g8base` 批次（覆盖 [0.05..1.0]）。两批在 ρ=0.2 上给出同一个
    # 45.20，是评测确定性的一次跨批次核对。
    full = score(ds, f"*_{MODEL}_fastkvzip__g8base_*", 1.0)
    fm = np.mean(list(full.values())) * 100 if full else float("nan")
    cells, rows[nm] = [], {}
    for lab, pat in BASE:
        o = score(ds, pat, R)
        if len(o) < 5:
            cells.append("—"); continue
        rows[nm][lab] = o
        cells.append(f"{np.mean(list(o.values()))*100:.2f}")
    print(f"{nm:<15}{fm:>7.2f}" + "".join(f"{c:>12}" for c in cells))

print(f"\n\n## 表 B：Retr.KV 上的全部臂 @ρ={R}（唯一有本项目各臂的 panel）\n")
B = score("scbench_kv", f"*_{MODEL}_fastkvzip_chunk16k_w4096", R)
F = score("scbench_kv", f"*_{MODEL}_fastkvzip__g8base_*", 1.0)
bm, fm = np.mean(list(B.values())) * 100, np.mean(list(F.values())) * 100
print(f"基线 FastKVzip {bm:.2f}　full {fm:.2f}　headroom {fm-bm:+.2f}\n")
print(f"{'臂':<26}{'绝对':>8}{'Δ vs FKV':>11}{'95% CI':>19}{'恢复率':>8}{'n':>5}")
ARMS = [("KVzip", f"*_{MODEL}"), ("DuoAttn", f"*_{MODEL}_head_chunk16k_w4096"),
        ("ExpectAttn", f"*_{MODEL}_expect_chunk16k_w4096"),
        ("SnapKV", f"*_{MODEL}_snap_chunk16k_w4096")]
for s in (0, 1, 2):
    ARMS.append((f"本项目 scalar s{s}", f"*_{MODEL}_fastkvzip__d10scalar_s{s}_*"))
for s in (0, 1, 2):
    ARMS.append((f"本项目 kv s{s}", f"*_{MODEL}_fastkvzip__d10kv_s{s}_*"))
ARMS += [("本项目 静态表 s0", f"*_{MODEL}_fastkvzip__p02own_*"),
         ("平凡地板 b_min=8", f"*_{MODEL}_fastkvzip__flr8_*"),
         ("平凡地板 b_min=32", f"*_{MODEL}_fastkvzip__flr32_*")]
for lab, pat in ARMS:
    o = score("scbench_kv", pat, R)
    c = sorted(set(o) & set(B))
    if len(c) < 5:
        print(f"{lab:<26}  —— 无数据"); continue
    dv = (np.array([o[j] for j in c]) - np.array([B[j] for j in c])) * 100
    m, lo, hi = boot(dv)
    print(f"{lab:<26}{np.mean([o[j] for j in c])*100:>8.2f}{m:>+11.2f}"
          f"{f'[{lo:+.2f},{hi:+.2f}]':>19}{'★' if (lo>0 or hi<0) else ' '}"
          f"{m/(fm-bm)*100:>7.0f}%{len(c):>5}")

print(f"\n\n## 表 C：Retr.KV @ρ=0.1（本项目各臂的另一个比例）\n")
R1 = 0.1
# ρ=0.1 复现批次没有，基线取 `__g8base`；5 个复现方法在 0.1 上无数据，
# 所以表 C 只会有本项目的臂 —— 这是覆盖面的事实，不是缺失。
B1 = score("scbench_kv", f"*_{MODEL}_fastkvzip__g8base_*", R1)
if len(B1) >= 5:
    b1 = np.mean(list(B1.values())) * 100
    print(f"基线 FastKVzip {b1:.2f}　full {fm:.2f}　headroom {fm-b1:+.2f}\n")
    print(f"{'臂':<26}{'绝对':>8}{'Δ vs FKV':>11}{'95% CI':>19}{'恢复率':>8}{'n':>5}")
    for lab, pat in ARMS:
        o = score("scbench_kv", pat, R1)
        c = sorted(set(o) & set(B1))
        if len(c) < 5:
            continue
        dv = (np.array([o[j] for j in c]) - np.array([B1[j] for j in c])) * 100
        m, lo, hi = boot(dv)
        print(f"{lab:<26}{np.mean([o[j] for j in c])*100:>8.2f}{m:>+11.2f}"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>19}{'★' if (lo>0 or hi<0) else ' '}"
              f"{m/(fm-b1)*100:>7.0f}%{len(c):>5}")
else:
    print("  基线在 ρ=0.1 无数据（复现只跑了 0.2 及以上）")
