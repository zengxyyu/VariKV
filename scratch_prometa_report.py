#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ProMeta v3 下游读数 —— 逐样本配对 bootstrap，**绝对分**，对 `__g8base`。

    .venv/bin/python -B scratch_prometa_report.py

读数一律走 `scratch_read_scores.read_scores`（gate 无关、输出文件名无关、
匹配为空即抛错）。★ = 95% CI 排除 0。**不可分 ≠ 等价。**

三条预注册判据（写在结果之前）：
 ① `zero`（γ=0）必须与基线**逐样本相同**。不同 ⇒ 整表作废。
 ② `s0/s1/s2` 对 `blind` 的配对差：**这才是「上下文路径有没有下游价值」**。
    三个种子都不显著 ⇒ 上下文路径在下游没有可测贡献（与训练侧的 val 是否
    改善**无关** —— 本仓库已测到训练侧指标与下游反相关）。
 ③ `s0` 对 `shuf`：标签内容有没有下游价值。
参照上界：oracle 轮同剂量（γ=0.5 resid、β=0、n=40、ρ=0.1）在
Retr.KV 上 +3.50 ns、Retr.PrefSuf 上 +18.50★ —— **Student 不可能超过它**，
所以 Retr.KV 上读到接近 0 是**预期之内**，不构成方法失败。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")
from scratch_read_scores import read_scores            # noqa: E402

PANEL = {"kv": ("scbench_kv", "Retr.KV"),
         "ps": ("scbench_prefix_suffix", "Retr.PrefSuf"),
         "vt": ("scbench_vt", "Retr.MultiHop")}
ARMS = ["s0", "s1", "s2", "blind", "shuf", "zero"]
SUF = {"zero": "_pmresg0b0L14"}
DEF = "_pmresg0p5b0L14"
RHOS = [0.2, 0.1]


def boot(d, n=4000, seed=0):
    r = np.random.default_rng(seed)
    b = np.array([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def diff(a, b):
    """→ (均值×100, 显著, n)。b 是分母（基线或对照）。"""
    c = sorted(set(a) & set(b))
    if not c:
        return None
    d = (np.array([a[j] for j in c]) - np.array([b[j] for j in c])) * 100
    m, lo, hi = boot(d)
    return m, (lo > 0 or hi < 0), len(c), lo, hi


def fmt(t):
    if t is None:
        return "   —    "
    m, sig, n, lo, hi = t
    return f"{m:+6.2f}{'★' if sig else ' '}(n={n})"


def main():
    print(__doc__.split("三条")[0].strip())
    for p, (ds, name) in PANEL.items():
        base = {r: read_scores(ds, "__g8base", r) for r in RHOS}
        got = {}
        for arm in ARMS:
            tag = f"_pmev{p}" + SUF.get(arm, DEF)
            for r in RHOS:
                try:
                    s = read_scores(ds, tag, r)
                except Exception:
                    s = {}
                if s:
                    got[(arm, r)] = s
        if not got:
            print(f"\n### {name}：还没有结果")
            continue
        print(f"\n### {name}（{ds}）　基线 `__g8base`")
        print("| 臂 | " + " | ".join(f"ρ={r} 对基线" for r in RHOS)
              + " | " + " | ".join(f"ρ={r} 对 blind" for r in RHOS) + " |")
        print("|---" * (1 + 2 * len(RHOS)) + "|")
        for arm in ARMS:
            row = [f"`{arm}`"]
            for r in RHOS:
                row.append(fmt(diff(got[(arm, r)], base[r])) if (arm, r) in got else "   —    ")
            for r in RHOS:
                if (arm, r) in got and ("blind", r) in got and arm != "blind":
                    row.append(fmt(diff(got[(arm, r)], got[("blind", r)])))
                else:
                    row.append("   —    ")
            print("| " + " | ".join(row) + " |")
        # 判据①：零点必须逐位相同
        for r in RHOS:
            if ("zero", r) in got:
                c = sorted(set(got[("zero", r)]) & set(base[r]))
                mx = max(abs(got[("zero", r)][k] - base[r][k]) for k in c) if c else None
                print(f"  判据①　γ=0 零点 ρ={r}：n={len(c)}　max|Δ| = {mx}"
                      f"　{'**逐样本相同**' if mx == 0 else '**不同 ⇒ 本表作废**'}")
        # 判据②：种子对 blind
        for r in RHOS:
            ds_ = [diff(got[(s_, r)], got[("blind", r)]) for s_ in ("s0", "s1", "s2")
                   if (s_, r) in got and ("blind", r) in got]
            if ds_:
                sig = sum(1 for t in ds_ if t[1])
                print(f"  判据②　ρ={r} 上 {len(ds_)} 个种子对 blind：均值 "
                      f"{np.mean([t[0] for t in ds_]):+.2f}、显著 {sig}/{len(ds_)}"
                      + ("　⇒ **上下文路径在下游没有可测贡献**" if sig == 0 else
                         "　⇒ 上下文路径确有下游价值"))
    print("\n⚠ 只在 n == 数据集满量（kv/ps 100、vt 90）时引用绝对分；"
          "配对差在子集上稳定，绝对分不稳定。")


if __name__ == "__main__":
    main()
