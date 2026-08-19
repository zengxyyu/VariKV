#!/usr/bin/env python3
"""`floorproj`（把地板配额投影回可达集）的单测，零 GPU、零模型。

**它要证伪的东西**：我打算用「投影后还剩多少分」来判断表示能力是不是收益的瓶颈。
如果投影本身写错了（比如投出去的配额其实仍然不可达、或者预算没守住、或者在本来
就可达时还乱动），那个判断就完全没意义。所以先在合成数据上把三条不变量钉死：

  ① 输出配额**确实可达**（用 slack 判据独立复核，而不是用投影自己的中间量）；
  ② 预算严格守恒 `Σq = B`；
  ③ 地板目标本来就可达时，投影是**恒等**的（否则会凭空引入一个干预）。
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "external/FastKVzip/prefill"))
from attention.quota_project import reachable_project     # noqa: E402


def slack_of(q, sc, alpha=1.0):
    """独立实现的 slack 判据（**不复用投影内部的量**）：>0 ⟺ 该配额可达。"""
    G = q.numel(); X = sc.reshape(G, -1)
    a = alpha * X.std(dim=-1)
    hi, lo = np.inf, -np.inf
    for g in range(G):
        sh = torch.sort(X[g], descending=True).values
        n = sh.numel(); k = int(q[g])
        s_q = float(sh[k-1]) if k >= 1 else np.inf
        s_q1 = float(sh[k]) if k < n else -np.inf
        hi = min(hi, s_q + float(a[g])); lo = max(lo, s_q1 - float(a[g]))
    return hi - lo


def main():
    rng = np.random.default_rng(0)
    bad = 0
    n_ident = n_moved = 0
    print(f"{'例':>3}  {'B':>6} {'L1(投影−地板)':>14} {'slack(地板)':>12} {'slack(投影)':>12}  结果")
    for t in range(40):
        G, n = 12, 60
        sc = torch.tensor(rng.normal(size=(G, n)) * rng.uniform(0.2, 1.5, size=(G, 1)),
                          dtype=torch.float32)
        Btot = int(rng.integers(G, G * n // 2))
        # 造一个地板式目标：先按分数排名给基线配额，再压一个 b_min
        thr = torch.quantile(sc.reshape(-1), 1 - Btot / (G * n))
        b0 = (sc > thr).sum(-1).float()
        bmin = float(rng.integers(1, 12))
        tg = torch.maximum(b0, torch.full_like(b0, min(bmin, Btot // G)))
        ex = float(tg.sum() - Btot)
        if ex > 0:
            room = (b0 - bmin).clamp(min=0)
            if float(room.sum()) < ex:
                continue
            tg = tg - room * (ex / float(room.sum()))
        tg = tg.round().long().clamp(0, n)
        d = Btot - int(tg.sum())
        if d != 0:
            idx = torch.argsort(-b0)
            for k in range(abs(d)):
                tg[int(idx[k % G])] += int(np.sign(d))
        tg = tg.clamp(0, n)
        if int(tg.sum()) != Btot:
            continue
        s_before = slack_of(tg, sc)
        q, l1 = reachable_project(tg, sc, b0, Btot)
        if q is None:
            print(f"{t:>3}  {Btot:>6} {'投影无解':>14}")
            bad += 1
            continue
        s_after = slack_of(q, sc)
        ok_budget = int(q.sum()) == Btot                      # ②
        ok_reach = s_after > 0                                # ①
        ok_ident = (s_before <= 0) or bool((q == tg).all())   # ③
        ok = ok_budget and ok_reach and ok_ident
        bad += (not ok)
        n_ident += int(bool((q == tg).all())); n_moved += int(not bool((q == tg).all()))
        if t < 12 or not ok:
            print(f"{t:>3}  {Btot:>6} {l1:>14.0f} {s_before:>+12.4f} {s_after:>+12.4f}  "
                  f"{'OK' if ok else '**FAIL** ' + ('预算 ' if not ok_budget else '') + ('不可达 ' if not ok_reach else '') + ('非恒等 ' if not ok_ident else '')}")
    print(f"\n  恒等（地板本就可达）{n_ident} 例，真的投影了 {n_moved} 例")

    # 上面的合成数据界很宽松，地板多半本来就可达 —— 那样只测到恒等分支。
    # **真实工作点恰恰是界很紧的那一侧**（74/74 不可达），所以这里把 α 压小，
    # 强制走投影分支，再查同样三条不变量。
    print("\n【紧界压力测试：α 小 ⇒ 地板应当不可达，必须真的投影】")
    rng = np.random.default_rng(7); nb = ntot = 0; l1s = []
    for t in range(30):
        G, n = 12, 60
        sc = torch.tensor(rng.normal(size=(G, n)) * rng.uniform(0.2, 1.5, size=(G, 1)),
                          dtype=torch.float32)
        Btot = int(rng.integers(2 * G, G * n // 2))
        thr = torch.quantile(sc.reshape(-1), 1 - Btot / (G * n))
        b0 = (sc > thr).sum(-1).float()
        tg = torch.maximum(b0, torch.full_like(b0, min(8.0, Btot // G)))
        ex = float(tg.sum() - Btot)
        if ex > 0:
            room = (b0 - 8.0).clamp(min=0)
            if float(room.sum()) < ex:
                continue
            tg = tg - room * (ex / float(room.sum()))
        tg = tg.round().long().clamp(0, n)
        d = Btot - int(tg.sum())
        if d != 0:
            idx = torch.argsort(-b0)
            for k in range(abs(d)):
                tg[int(idx[k % G])] += int(np.sign(d))
        if int(tg.sum()) != Btot:
            continue
        AL = 0.02
        if slack_of(tg, sc, AL) > 0:
            continue                                   # 本例地板仍可达，不算压力样本
        q, l1 = reachable_project(tg, sc, b0, Btot, alpha=AL)
        ntot += 1
        if q is None:
            nb += 1; continue
        ok = (int(q.sum()) == Btot) and (slack_of(q, sc, AL) > 0) and l1 > 0
        nb += (not ok); l1s.append(l1)
    print(f"  强制不可达样本 {ntot} 个；投影后仍违反不变量的 **{nb}** 个"
          f"；L1(投影−地板) 中位 {np.median(l1s) if l1s else float('nan'):.0f}")
    bad += nb + (ntot == 0)
    print(f"\n{'全部通过' if bad == 0 else f'**{bad} 项 FAIL**'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
