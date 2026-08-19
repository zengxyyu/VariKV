#!/usr/bin/env python3
"""`floorproj`（把地板配额投影到放宽的有界分数可达集 `Q_box`）的单测，零 GPU、零模型。

**它要证伪什么**：我打算用「投影后还剩多少分」判断可达性是不是收益的瓶颈。
投影一旦写错——投出去的点其实不可达、预算没守住、本来可达时却乱动、
或者**找不到可行 τ 时悄悄退回地板**——那个判断就完全没意义。

五组检查：

  ① 输出**确实可达**：用独立实现的 slack 判据复核（不复用投影内部的量）；
  ② 预算严格守恒 `Σq = B`；
  ③ 地板本就可达时投影是**恒等**的（否则会凭空引入一个干预）；
  ④ **可行 τ 区间的闭式端点正确**：与细网格暴力枚举逐点对拍；
  ⑤ **窄区间回归**：构造一个可行 τ 区间极窄的例子。旧实现在整个分数范围上撒
     1024 个均匀点，必然整个错过它，然后走到「找不到解」的分支并**静默退回地板**。
     新实现先闭式定位区间再在**区间内**采样，必须找到。
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "external/FastKVzip/prefill"))
from attention.quota_project import reachable_project, slack_of   # noqa: E402


def indep_slack(q, sc, alpha):
    """**独立实现**的 slack（纯 numpy、按定义重写），用于交叉复核。"""
    G = len(q); X = sc.reshape(G, -1).float().numpy()
    a = alpha * np.maximum(X.std(-1, ddof=1), 1e-6)
    hi, lo = np.inf, -np.inf
    for g in range(G):
        sh = np.sort(X[g])[::-1]; n = len(sh); k = int(q[g])
        s_q = sh[k-1] if k >= 1 else np.inf
        s_q1 = sh[k] if k < n else -np.inf
        hi = min(hi, s_q + a[g]); lo = max(lo, s_q1 - a[g])
    return hi - lo


def feasible_at(tau, sc, alpha, Btot):
    """τ 处是否存在预算为 B 的可达配额：Σq_min(τ) ≤ B ≤ Σq_max(τ)。"""
    G = sc.shape[0]; X = sc.reshape(G, -1).float().numpy()
    a = alpha * np.maximum(X.std(-1, ddof=1), 1e-6)
    qmax = sum(int((X[g] > tau - a[g]).sum()) for g in range(G))
    qmin = sum(int((X[g] > tau + a[g]).sum()) for g in range(G))
    return qmin <= Btot <= qmax


def make_floor_target(sc, Btot, bmin):
    """复刻 project_quota 的地板目标构造，返回 (b0, tgt) 或 None（本例不适用）。"""
    G, n = sc.shape
    flat = sc.reshape(-1)
    thr = torch.sort(flat, descending=True).values[Btot - 1]
    b0 = (sc > thr).sum(-1).float()
    if int(b0.sum()) == 0:
        return None
    eff = min(float(bmin), Btot // G)
    tg = torch.maximum(b0, torch.full_like(b0, eff))
    ex = float(tg.sum() - Btot)
    if ex > 0:
        room = (b0 - eff).clamp(min=0)
        if float(room.sum()) < ex:
            return None
        tg = tg - room * (ex / float(room.sum()))
    tg = tg.round().long().clamp(0, n)
    d = Btot - int(tg.sum())
    if d != 0:
        idx = torch.argsort(-b0)
        for k in range(abs(d)):
            tg[int(idx[k % G])] += int(np.sign(d))
    tg = tg.clamp(0, n)
    return (b0, tg) if int(tg.sum()) == Btot else None


def main():
    bad = 0
    AL = 0.999                                    # 与 ckpt 的 α 同量级

    print("【1-3】宽界 / 紧界两个总体上查：可达性 · 预算守恒 · 本可达时恒等")
    for name, al, seed in [("宽界 α=0.999", AL, 0), ("紧界 α=0.02", 0.02, 7)]:
        rng = np.random.default_rng(seed)
        nid = nmv = nb = ntot = 0; l1s = []
        for t in range(40):
            G, n = 12, 60
            sc = torch.tensor(rng.normal(size=(G, n)) * rng.uniform(0.2, 1.5, size=(G, 1)),
                              dtype=torch.float32)
            Btot = int(rng.integers(2 * G, G * n // 2))
            mk = make_floor_target(sc, Btot, 8)
            if mk is None:
                continue
            b0, tg = mk
            ntot += 1
            s_before = indep_slack(tg, sc, al)
            q, l1 = reachable_project(tg, sc, Btot, alpha=al)   # 失败会抛错，不返回 None
            ok_b = int(q.sum()) == Btot                                    # ②
            ok_r = indep_slack(q, sc, al) > 0                              # ①
            ok_i = (s_before <= 0) or bool((q == tg).all())                # ③
            nb += (not (ok_b and ok_r and ok_i))
            nid += int(bool((q == tg).all())); nmv += int(not bool((q == tg).all()))
            l1s.append(l1)
        bad += nb
        print(f"  {name:<14} 样本 {ntot:>3}  违反不变量 **{nb}**  "
              f"恒等 {nid} / 真投影 {nmv}  L1 中位 {np.median(l1s) if l1s else float('nan'):.0f}")

    print("\n【4】可行 τ 区间的闭式端点：与细网格暴力枚举对拍")
    rng = np.random.default_rng(3); mism = 0; ncase = 0
    for t in range(15):
        G, n = 6, 40
        sc = torch.tensor(rng.normal(size=(G, n)), dtype=torch.float32)
        Btot = int(rng.integers(2 * G, G * n // 2))
        X = sc.float(); a = AL * X.std(-1).clamp_min(1e-6)
        hi_pool = (X + a[:, None]).reshape(-1); lo_pool = (X - a[:, None]).reshape(-1)
        N = hi_pool.numel()
        t_hi = float(torch.topk(hi_pool, min(Btot, N), largest=True).values[-1])
        t_lo = float(torch.topk(lo_pool, min(Btot + 1, N), largest=True).values[-1])
        span = float(X.max() - X.min()) + 2 * float(a.max())
        for tau in np.linspace(float(X.min()) - float(a.max()) - 0.1,
                               float(X.max()) + float(a.max()) + 0.1, 4001):
            pred = (t_lo <= tau < t_hi)
            truth = feasible_at(tau, sc, AL, Btot)
            ncase += 1; mism += int(pred != truth)
    bad += (mism > 0)
    print(f"  {ncase} 个 τ 逐点对拍，闭式判据与暴力枚举不一致 **{mism}** 处"
          f"  {'OK' if mism == 0 else '**FAIL**'}")

    print("\n【5】窄区间回归：旧的「全范围撒 1024 点」必然错过，新实现必须找到")
    # 区间宽度 ≈ 阈值处**相邻次序统计量的间隔** + 2a。要让它相对量程极窄，就得
    # 「量程巨大 + 阈值落在一个稠密簇里」。真实分数分布正是这样——teacher trace
    # 本身就是按 |s−τ| 近阈值采样的，说明阈值附近候选极稠密。所以这不是人造刁难。
    G, n = 8, 50                                   # 400 个值
    rng = np.random.default_rng(11)
    vals = np.concatenate([
        rng.uniform(500, 1000, 100),               # 高簇：排名 1–100
        rng.uniform(0.0, 0.01, 200),               # **稠密簇**：排名 101–300
        rng.uniform(-1000, -500, 100),             # 低簇：排名 301–400
    ])
    rng.shuffle(vals)
    sc = torch.tensor(vals.reshape(G, n), dtype=torch.float32)
    Btot = 200                                     # 阈值落在稠密簇正中
    ALN = 1e-6                                     # 界极小，2a 不会撑宽区间
    X = sc.float(); a = ALN * X.std(-1).clamp_min(1e-6)
    hi_pool = (X + a[:, None]).reshape(-1); lo_pool = (X - a[:, None]).reshape(-1)
    t_hi = float(torch.topk(hi_pool, Btot, largest=True).values[-1])
    t_lo = float(torch.topk(lo_pool, Btot + 1, largest=True).values[-1])
    width = t_hi - t_lo
    g_lo = float(X.min() - a.max()); g_hi = float(X.max() + a.max())
    grid_old = np.linspace(g_lo, g_hi, 1024)
    old_hits = int(np.sum((grid_old >= t_lo) & (grid_old < t_hi)))
    mk = make_floor_target(sc, Btot, 8)
    ok5 = mk is not None
    if not ok5:
        print("    地板目标构造失败，本例无效")
    else:
        b0, tg = mk
        try:
            q, l1 = reachable_project(tg, sc, Btot, alpha=ALN)
            ok5 = (int(q.sum()) == Btot) and (indep_slack(q, sc, ALN) > 0)
        except RuntimeError as e:
            ok5 = False; print(f"    抛错：{e}")
    bad += (not ok5)
    print(f"  可行区间宽 {width:.3e}，分数量程 {g_hi-g_lo:.1f}，比值 {width/(g_hi-g_lo):.1e}")
    print(f"  旧的全范围 1024 均匀点落在区间内的个数：**{old_hits}**"
          f"{'  ⇒ 旧实现在此必然走到「找不到解」并静默退回地板' if old_hits == 0 else ''}")
    print(f"  新实现（闭式定位后区间内采样）：{'OK' if ok5 else '**FAIL**'}")

    print("\n【6】走**生产入口** `project_quota(mode=\"floorproj\")` 的集成检查")
    # 前五组测的是 helper。生产路径还多三件事会出错：alpha 有没有真的传进去、
    # 失败会不会静默回退、floor 与 floorproj 在「地板本就可达」时是否逐位相同。
    from attention.quota_project import project_quota                # noqa: E402
    rng = np.random.default_rng(21)
    L, H = 4, 3; G = L * H; n = 40
    # **必须造出饿死**，否则地板目标 ≡ 基线、任何界下都可达，(c)(d) 就测不到东西。
    # 真实数据的关键特征是跨头尺度差极大（饿死头 σ_h 中位 0.0048 vs 非饿死 0.1589，
    # 33 倍），于是低尺度头整体压在全局阈值之下。这里照此构造。
    scale = np.array([1.0] * 6 + [0.03] * 6)[:, None]       # 一半头尺度小 33 倍
    shift = np.array([0.0] * 6 + [-1.5] * 6)[:, None]       # 且整体偏低 ⇒ 饿死
    scf = torch.tensor(rng.normal(size=(G, n)) * scale + shift, dtype=torch.float32)
    sc = scf.reshape(L, H, n)
    Btot = 200
    thr = torch.sort(scf.reshape(-1), descending=True).values[Btot - 1]
    b0 = (scf > thr).sum(-1).float()
    delta = torch.zeros_like(b0)
    n6 = 0

    # (a) 不给 alpha_eff 必须 assert，而不是偷偷用某个默认值
    try:
        project_quota(b0, delta, n, "floorproj", L, H, sc=sc)
        print("    (a) **FAIL** 缺 alpha_eff 却没报错"); n6 += 1
    except AssertionError:
        print("    (a) OK   缺 alpha_eff 时 assert 生效")

    # (b) 宽界：地板目标本就可达 ⇒ floorproj 必须与 floor **逐位相同**
    os.environ["VARIKV_QUOTA_FLOOR"] = "16"    # = Btot/G，强制真的搬动
    bf = project_quota(b0.clone(), delta, n, "floor", L, H)
    bp = project_quota(b0.clone(), delta, n, "floorproj", L, H, sc=sc, alpha_eff=0.999)
    same = bool((bf == bp).all()); reach = indep_slack(bf, sc, 0.999) > 0
    ok = (not reach) or same
    n6 += (not ok)
    print(f"    (b) 宽界 α=0.999：地板可达={reach}  floorproj==floor={same}  "
          f"{'OK' if ok else '**FAIL**'}")

    # (c) 紧界：地板目标不可达 ⇒ floorproj 必须**动**，且结果通过独立可达性复核
    bp2 = project_quota(b0.clone(), delta, n, "floorproj", L, H, sc=sc, alpha_eff=1e-4)
    moved = not bool((bf == bp2).all())
    ok_c = (int(bp2.sum()) == int(bf.sum())) and (indep_slack(bp2, sc, 1e-4) > 0)
    n6 += (not (moved and ok_c))
    print(f"    (c) 紧界 α=1e-4：地板可达={indep_slack(bf, sc, 1e-4) > 0}  "
          f"投影动了={moved}  预算守恒+可达={ok_c}  "
          f"{'OK' if moved and ok_c else '**FAIL**'}")

    # (d) 检验一个**真正的数学性质**而不是「换个数结果就该变」：
    #     α₁ ≤ α₂ ⇒ Q_box(α₁) ⊆ Q_box(α₂)（界更宽、可达集更大）
    #     ⇒ 到地板目标的最小 L1 距离**随 α 单调不增**。
    # 这同时也证明 α 确实接上了：若被吞掉，整条曲线会是常数。
    # （注意：搜索是近似的，所以单调性理论上可能被近似误差破坏；真出现就要看见它。）
    from attention.quota_project import reachable_project as _rp
    tgt_floor = bf.clone()
    ds = []
    for al in [1e-4, 1e-3, 1e-2, 1e-1, 0.999, 5.0]:
        _, dd = _rp(tgt_floor, sc, int(bf.sum()), alpha=al)
        ds.append((al, dd))
    viol = sum(1 for i in range(1, len(ds)) if ds[i][1] > ds[i - 1][1] + 1e-9)
    spread = ds[0][1] - ds[-1][1]
    ok_d = (viol == 0) and (spread > 0)
    n6 += (not ok_d)
    print("    (d) L1(投影−地板) 随 α：" +
          "  ".join(f"α={a_:g}:{d_:.0f}" for a_, d_ in ds))
    print(f"        单调不增违例 {viol}，跨度 {spread:.0f}  "
          f"{'OK' if ok_d else '**FAIL：α 未接上或单调性被破坏**'}")
    bad += n6

    print(f"\n{'全部通过' if bad == 0 else f'**{bad} 项 FAIL**'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
