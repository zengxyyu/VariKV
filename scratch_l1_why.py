#!/usr/bin/env python3
"""**探索性**：在四个同样饿死、同样剂量的层里，oracle 未来需求能不能把 L1 分出来？

（零 GPU，只读 `scratch_prometa_oracle_*.npz`。）

────────────────────────────────────────────────────────────────────────────
为什么这个对照是干净的
────────────────────────────────────────────────────────────────────────────
§十一之十七 给了一个**天然受控**的四元组：Retr.KV@0.1 上 L0/L1/L6/L13 的
饿死率都在 99.7–100%、实测 lift/chunk 都在 3.989–4.000，**唯独 L1 给 +24.80★**，
其余三层 −0.80 / +0.60 / +0.40 全 ns。⇒ 剂量与饿死这两个自变量已经被控住，
剩下的差别只能来自「那 4 个头里装的是什么」。

本脚本用 oracle 未来需求 `U[m,l,h,i]`（真实未来查询算的）问：
**L1 的头是不是「未来最想要、却被整层清零」的那一批？**

────────────────────────────────────────────────────────────────────────────
⚠ 三条必须随数字一起写的限制
────────────────────────────────────────────────────────────────────────────
① **事后的、探索性的**。四元组的下游结果先出来，才来找解释变量 —— 这不是
   预注册。任何结论只能当假说。
② **只有 4 个样本**（oracle dump 就这么多），而下游是 n=100。
③ `U` 是 `max_t softmax_i`，**跨头尺度不严格可比**（弥散注意力的头行和更大）。
   所以同时报**绝对量**与**层内归一化的秩**，并把判读建立在
   「L1 在 28 层里排第几」而不是绝对值上。
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FOUR = [0, 1, 6, 13]                 # 同样饿死、同样剂量的四层
DELTA = {0: -0.80, 1: +24.80, 6: +0.60, 13: +0.40}    # 下游 Δ（n=100）


def main():
    files = sorted(glob.glob("scratch_prometa_oracle_scbench_kv_*.npz"))
    assert files, "没有 Retr.KV 的 oracle dump"
    acc = None
    for f in files:
        U = np.load(f)["U"].astype(np.float64)        # [M,L,H,N]
        M, L, H, N = U.shape
        Um = U.max(0)                                 # [L,H,N] 最需要它的那个未来
        stat = dict(
            D=Um.sum(-1),                             # 该头对可驱逐区的总未来需求
            peak=Um.max(-1),                          # 单个位置的峰值需求
            top4=np.sort(Um, -1)[..., -4:].sum(-1),   # **地板只给 1 个名额/头**，
                                                      # 但 band 抬的是 4 个头 ⇒ 看每头前几名
        )
        stat["conc"] = stat["peak"] / np.maximum(stat["D"], 1e-30)
        if acc is None:
            acc = {k: [v] for k, v in stat.items()}
        else:
            for k, v in stat.items():
                acc[k].append(v)
    S = {k: np.mean(v, 0) for k, v in acc.items()}    # [L,H] 跨样本均值
    L, H = S["D"].shape
    print(f"# 四个等剂量层的 oracle 未来需求对照　{len(files)} 个样本 × {L} 层 × {H} 头")
    print("**探索性、事后、n=4，只能当假说**\n")

    for key, name in [("D", "总未来需求 Σ_i max_m U"),
                      ("peak", "峰值 max_i max_m U"),
                      ("top4", "前 4 个位置之和"),
                      ("conc", "集中度 peak/D")]:
        v = S[key].mean(1)                            # 逐层（层内 4 头取均值）
        order = np.argsort(-v)
        rank = {int(l): int(np.where(order == l)[0][0]) + 1 for l in range(L)}
        print(f"## {name}")
        for l in FOUR:
            print(f"   L{l:<3} 值={v[l]:.4f}　**28 层里排第 {rank[l]}**"
                  f"　下游 Δ={DELTA[l]:+.2f}")
        # 判据：L1 是不是在这个量上唯一突出的
        others = [v[l] for l in FOUR if l != 1]
        sep = (v[1] > max(others)) or (v[1] < min(others))
        print(f"   ⇒ L1 在四元组里{'**是极值**' if sep else '不是极值'}；"
              f"L1={v[1]:.4f} vs 其余 {['%.4f' % x for x in others]}\n")

    # 全层秩相关：这个量能不能解释「哪些层值得抬」？**只有 4 个下游点，功效极低**
    print("## ⚠ 与下游 Δ 的关系：**只有 4 个点**，不做相关系数（无意义），只列表")
    for key in ("D", "peak", "top4", "conc"):
        vals = [(l, S[key].mean(1)[l], DELTA[l]) for l in FOUR]
        vals.sort(key=lambda x: -x[1])
        print(f"   {key:<5} 按该量降序: " +
              " > ".join(f"L{l}(Δ{d:+.1f})" for l, _, d in vals))
    print("\n判读规则（写死）：只有当 L1 在某个量上**同时**是四元组极值、"
          "且该量的层序与 Δ 的层序一致时，才算找到一个候选解释变量；"
          "否则记为「oracle 未来需求解释不了 L1」，并入待做。")


if __name__ == "__main__":
    main()
