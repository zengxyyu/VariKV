#!/usr/bin/env python3
"""**规则 × 目标 矩阵** —— 终结「β 的方向是不是定反了」这个争论（零 GPU）。

────────────────────────────────────────────────────────────────────────────
争点
────────────────────────────────────────────────────────────────────────────
`ρ_β(U) = (1/β)log E[e^{βU}]` 里 `U` 是**保留效用**。外部复核提出：
效用语义下 `β>0` 是 soft-**max**，所以「β>0 = 最坏未来保护」是把 utility
与 loss 的风险方向搞混了，正确写法应是 soft-min `−(1/β)log E[e^{−βU}]`。

**这个推理漏了一步。** 驱逐 `i` 给未来 `m` 造成的**损失**就是 `L_{m,i}=U_{m,i}`。
于是「最坏未来损失」是

    L_worst(S) = max_m Σ_{i∉S} U_{m,i}

而 **max 是次可加的**，所以

    max_m Σ_{i∉S} U_{m,i}  ≤  Σ_{i∉S} max_m U_{m,i}          (†)

⇒ **按 `max_m U` 取 top-B 就是在最小化最坏未来损失的一个上界**，
`β→∞` 有严格的松弛依据。反过来 soft-min 那一侧**没有**对应的界
（`min_m Σ ≠ Σ min_m`，且不等号方向不利）。

**真正错的是我自己**：先前用 `max_S min_m Σ_{i∈S} U`（集合级**平权福利**）
去判一个为**最坏损失**设计的规则 —— 那是两个不同的目标。
⇒ 本脚本把「规则」与「目标」拆成两维，同时报，让目标依赖性无处可藏。

────────────────────────────────────────────────────────────────────────────
四个目标（都逐 (层,头) 独立、固定预算 B）
────────────────────────────────────────────────────────────────────────────
  L_mean  = (1/M) Σ_m Σ_{i∉S} U_{m,i}                    最小化。**下游指标
            是 M 个问题的平均分**，若准确率随丢失质量近似线性，这就是它。
  L_worst = max_m Σ_{i∉S} U_{m,i}                        最小化（最坏损失）
  W_minmax= min_m Σ_{i∈S} U_{m,i}                        最大化（平权福利）
  W_pf    = Σ_m log(ε + Σ_{i∈S} U_{m,i})                 最大化（比例公平）
            **它对 S 单调且次模** ⇒ **exact-marginal** 贪心有 (1−1/e) 保证。
            ⚠ 首版实现用的是连续梯度 `Σ_m U/(ε+t_m)`，**那没有这个保证**；
            已改成 `Σ_m log1p(U/(ε+t_m))`，即精确的 `Δ_i`。

**预注册的三条自检**（判据本身要能被证伪）：
  ① `mean` 规则在 `L_mean` 上必须**恰好最优**（那是个线性目标，top-B 即精确解）；
  ② `max` 规则在 `L_worst` 上应当**优于 mean**（由 (†) 的上界论证）；
  ③ `greedy_pf` 在 `W_pf` 上必须优于任何逐 token 规则（贪心直接优化它）。
若 ① 或 ③ 不成立 ⇒ **是这个脚本错了，不是数据在说话**。
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prometa.risk import entropic_risk                              # noqa: E402


def topb(score, B):
    return np.argsort(-score, kind="stable")[:B]


def greedy(U, B, kind, eps):
    """贪心最大化 W_pf 或 W_minmax。U: [M,N]。返回被选下标。"""
    M, N = U.shape
    t = np.zeros(M)
    avail = np.ones(N, bool)
    sel = []
    for _ in range(B):
        if kind == "pf":
            # **必须是真实的离散边际增益**，不是连续梯度（外部复核指出，采纳）：
            #     Δ_i = Σ_m [ log(ε+t_m+U_{m,i}) − log(ε+t_m) ]
            # 只有 exact-marginal 贪心才有单调次模的 (1−1/e) 保证；梯度式
            #     g_i = Σ_m U_{m,i}/(ε+t_m)
            # 是它的一阶近似（`log(1+x) ≤ x`，所以梯度式**系统性高估**大 U 的项），
            # 二者只在 `U ≪ ε+t` 时接近。首版写了「有 (1−1/e) 保证」而实现是梯度式
            # —— 那是理论与实现不一致，属第⑤类错。
            g = np.log1p(U / (eps + t)[:, None]).sum(0)
        else:                                            # minmax
            g = (t[:, None] + U).min(0)
        g = np.where(avail, g, -np.inf)
        j = int(np.argmax(g))
        sel.append(j)
        avail[j] = False
        t = t + U[:, j]
    return np.array(sel)


def objectives(U, sel, eps):
    M, N = U.shape
    keep = np.zeros(N, bool)
    keep[sel] = True
    F = U[:, keep].sum(1)                                # 保住的质量
    Lost = U.sum(1) - F
    return dict(L_mean=float(Lost.mean()), L_worst=float(Lost.max()),
                W_minmax=float(F.min()), W_pf=float(np.log(eps + F).sum()))


OBJ_SIGN = dict(L_mean=-1, L_worst=-1, W_minmax=+1, W_pf=+1)   # +1 = 越大越好


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--nsub", type=int, default=6000)
    ap.add_argument("--layer_step", type=int, default=4)
    ap.add_argument("--glob", default="scratch_prometa_oracle_scbench_*.npz")
    a = ap.parse_args()

    rules = ["mean", "max", "erisk_b2", "erisk_b20", "softmin_b2",
             "greedy_pf", "greedy_minmax"]
    acc = {r: {o: [] for o in OBJ_SIGN} for r in rules}
    ncell = 0
    files = sorted(glob.glob(a.glob))
    assert files, a.glob
    for f in files:
        U4 = np.load(f)["U"].astype(np.float64)          # [M,L,H,N]
        M, L, H, N = U4.shape
        rs = np.random.default_rng(1)
        sub = np.sort(rs.choice(N, min(a.nsub, N), replace=False))
        B = max(1, int(round(a.rho * len(sub))))
        for l in range(0, L, a.layer_step):
            for h in range(H):
                U = U4[:, l, h, sub]                     # [M,n]
                eps = 0.01 * float(U.sum(1).mean())
                sels = {
                    "mean": topb(U.mean(0), B),
                    "max": topb(U.max(0), B),
                    "erisk_b2": topb(entropic_risk(U, 2.0), B),
                    "erisk_b20": topb(entropic_risk(U, 20.0), B),
                    # soft-min（外部复核推荐的方向）：−ρ_{β}(−U)
                    "softmin_b2": topb(-entropic_risk(-U, 2.0), B),
                    "greedy_pf": greedy(U, B, "pf", eps),
                    "greedy_minmax": greedy(U, B, "minmax", eps),
                }
                for r, s_ in sels.items():
                    o = objectives(U, s_, eps)
                    for k, v in o.items():
                        acc[r][k].append(v)
                ncell += 1

    print(f"# 规则 × 目标 矩阵　{len(files)} 个 dump × {ncell//len(files)} 个 (层,头)"
          f" = {ncell} 格　ρ={a.rho} B={B}/{len(sub)}\n")
    print(f"{'规则':<16}" + "".join(f"{o:>14}" for o in OBJ_SIGN))
    base = {o: np.mean(acc['mean'][o]) for o in OBJ_SIGN}
    for r in rules:
        row = f"{r:<16}"
        for o in OBJ_SIGN:
            v = np.mean(acc[r][o])
            row += f"{v:>14.5f}"
        print(row)
    print(f"\n{'规则':<16}" + "".join(f"{o+'(相对mean)':>18}" for o in OBJ_SIGN))
    for r in rules:
        row = f"{r:<16}"
        for o in OBJ_SIGN:
            d = (np.mean(acc[r][o]) - base[o]) * OBJ_SIGN[o]     # 正 = 比 mean 好
            w = int(np.sum((np.array(acc[r][o]) - np.array(acc['mean'][o]))
                           * OBJ_SIGN[o] > 0))
            row += f"{d:>+11.5f} ({w:>3}/{ncell})"
        print(row)

    # ── 预注册自检 ─────────────────────────────────────────────────────
    print("\n== 预注册自检（失败 ⇒ 是脚本错了，不是数据在说话）==")
    ok = True
    d1 = np.mean(acc["mean"]["L_mean"])
    worst_other = min(np.mean(acc[r]["L_mean"]) for r in rules if r != "mean")
    c1 = d1 <= worst_other + 1e-12
    print(f"① mean 在 L_mean 上恰好最优：{'PASS' if c1 else 'FAIL'}"
          f"（mean {d1:.5f} vs 最好的其它 {worst_other:.5f}）")
    ok &= c1
    c2 = np.mean(acc["max"]["L_worst"]) < np.mean(acc["mean"]["L_worst"])
    print(f"② max 在 L_worst 上优于 mean（由 max 次可加的上界论证）："
          f"{'PASS' if c2 else 'FAIL'}"
          f"（max {np.mean(acc['max']['L_worst']):.5f} vs "
          f"mean {np.mean(acc['mean']['L_worst']):.5f}）")
    c3 = all(np.mean(acc["greedy_pf"]["W_pf"]) > np.mean(acc[r]["W_pf"])
             for r in ("mean", "max", "erisk_b2", "erisk_b20", "softmin_b2"))
    print(f"③ greedy_pf 在 W_pf 上优于全部逐 token 规则：{'PASS' if c3 else 'FAIL'}")
    ok &= c3
    print(f"\n自检总判：{'PASS' if ok else '**FAIL —— 结果不可读**'}")
    print("（② 是被检验的**命题**不是自检：它为真 ⇒ β>0 不是「方向反了」；"
          "为假 ⇒ 上界论证在真实数据上不起作用，那才是新发现。）")


if __name__ == "__main__":
    main()
