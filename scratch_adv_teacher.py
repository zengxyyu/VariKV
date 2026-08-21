"""任务基准的**反事实配额优势教师**（2026-08-21）。

────────────────────────────────────────────────────────────────────────────
它解决什么
────────────────────────────────────────────────────────────────────────────
现有两种教师都以**满缓存注意力输出**为参照：

    full_single :  U_i = err(S \\ i) − err(S)            条件=满缓存，参照=满缓存
    set_marginal:  U_i = err(S \\ i) − err(S ∪ {i})      条件=真实存活集合，参照=满缓存

但实测有 **28/77 格 headroom < 0**（压缩比满缓存还好），在那些格上
「向满缓存靠拢」**方向就是错的**。而 `chr03`（在评测工作点重训）仍 −12.87、
损伤占 Δ 方差 **86%** —— 都指向参照系而非条件。

本教师换掉参照系：

    J(b)          = − NLL(答案 token | 上下文, 配额 b)          ← **任务效用**
    A_{i←j}^{(k)} = J(b⁰ + k·e_i − k·e_j) − J(b⁰)              ← **相对当前基线**

预算严格守恒：受主 i 加 k 个、施主 j 减 k 个，`Σ_h b_h` 不变（脚本内断言）。
答案由 `make_retrieval` **构造时已知**，所以标签不需要人工标注，且是稠密的
对数概率而非二值命中。

────────────────────────────────────────────────────────────────────────────
为什么便宜：改配额**不需要重新预填**
────────────────────────────────────────────────────────────────────────────
`RetainCache` 物理上保留全部 KV，保留集只是 `self.valid` 这个掩码
（`prepare()` 用 `_get_valid()` 挑进注意力）。所以：

    预填一次（贵）  →  换掩码 → 前向答案 token（便宜）→ `slice()` 回滚  → 重复

每个动作只多一次 ~30 token 的前向。

────────────────────────────────────────────────────────────────────────────
三个自检（都会打印，任一失败即中止）
────────────────────────────────────────────────────────────────────────────
① **零动作**（k=0）必须给出 `A == 0`（逐位），否则说明掩码写入/回滚有副作用；
② **预算守恒**：每个动作后 `Σ_h b_h` 必须与基线相同；
③ **信度**：把**问句**按奇偶交错分半，各算一次标签，报两半的相关与符号一致率。
   **不按答案 token 位置对半**（2026-08-21 更正）—— 教师强制下后半答案条件于
   正确前缀、对缓存依赖更弱，两半估的不是同一个潜在量，那种分法的低一致率
   无法区分"估计量噪声大"与"真效应随位置变"。按独立事实分才是重复测量。
   取奇/偶而非前/后，是让两半的平均问句位置相同。
   若两半对不上，这个标签就不能拿来训练（本项目 `U^NLL` 那次正是栽在没测这个）。
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))

from model import ModelKVzip                                        # noqa: E402
from data.load import load_fineweb
from attention.kvcache import RetainCache                                  # noqa: E402


# ────────────────────────────── 任务构造 ──────────────────────────────
def make_task(m, ids, max_ctx, window, n_fact, rng):
    """插入 `n_fact` 条**互不相同**的合成事实，返回 (上下文, [(q,a)…], meta)。

    与 `scratch_ctrl_teacher.make_retrieval` 同构（同样的 key/val 格式、同样的
    插入区间与倒序插入），差别是**多事实多问句**：要估的不是某一个随机 query
    下的优势，而是对未来查询的期望

        Ā_{i←j} = E_{q∼Q(x)} [ J(q, S') − J(q, S) ]

    **每个问句单独一次前向，不拼接**（2026-08-21 更正）。曾经拼成一条
    `[q1,a1,…,qM,aM]` 走一次前向，理由写的是"每条各跑一次成本乘 M，负担不起"
    —— **那个理由是错的**：prefill 是共享的、每次只回滚 target，所以 M 次独立
    前向的 target token 总数与拼接**完全相同**，注意力 FLOPs 一样，只多 M 倍
    kernel 启动和 M 次读 KV cache（几十毫秒）。而拼接会真的改变被估的量：

        拼接算的是 p(a_m | S, q₁,a₁,…,q_{m-1},a_{m-1}, q_m)
        要的是     p(a_m | S, q_m)

    后面的问句能看到前面的**正确答案**和问答格式（few-shot），对缓存的依赖被
    系统性削弱，而且 M 个 NLL 不再独立 —— bootstrap 与分半信度都会反偏。
    """
    ctx = list(ids[-max_ctx:])
    hx = "0123456789abcdef"
    facts, qas = [], []
    for _ in range(n_fact):
        key = "".join(rng.choices(hx, k=16))
        val = "".join(rng.choices(hx, k=16))
        facts.append(m.encode(f" The secret key {key} maps to the value {val}. ")[0].tolist())
        qas.append((m.encode(f"\nQuestion: What value does the secret key {key} "
                             f"map to?\nAnswer:")[0].tolist(),
                    m.encode(f" {val}")[0].tolist()))
    # 插入点分散在可驱逐区（末尾 window 恒保留，插那里等于没驱逐）；倒序插以免下标失效
    lo, hi = int(0.05 * (len(ctx) - window)), int(0.90 * (len(ctx) - window))
    pos = sorted(rng.sample(range(lo, hi), n_fact), reverse=True)
    for p_, f_ in zip(pos, facts):
        ctx[p_:p_] = f_
    return ctx, qas, dict(n_fact=n_fact, pos=sorted(pos),
                          n_ans=sum(len(a) for _, a in qas),
                          n_tgt=sum(len(q) + len(a) for q, a in qas))


# ────────────────────────────── 效用 J ──────────────────────────────
@torch.no_grad()
def answer_nll(m, kv, qas_t, n_seen):
    """→ `(逐问句平均 NLL 的均值, 逐问句 NLL 向量[M])`，越小越好。

    每个问句**单独**一次 `m.model(q+a)`，算完立刻 `kv.slice(n_seen)` 回滚，
    所以 M 个问句都条件于**同一个**上下文缓存 S，互不可见。

    **必须回滚**：`m.model(...)` 会把 target 的 K/V 追加进 cache，不回滚则下一个
    问句/动作看到的上下文已被污染（`teacher_state` 的注释记过这个坑）。

    对齐：预测 `inp[t]` 用 `logits[t-1]`；答案有 `n_a` 个 token，所以取
    `logits[-n_a-1:-1]` 对 `inp[-n_a:]`。
    """
    per = []
    for t_t, n_a in qas_t:
        out = m.model(t_t, past_key_values=kv, use_cache=True)
        kv.slice(n_seen)
        lg = out.logits[0, -n_a - 1:-1].float()
        per.append(float(F.cross_entropy(lg, t_t[0, -n_a:], reduction="mean")))
    per = np.array(per)
    return float(per.mean()), per


# ────────────────────────────── 动作构造 ──────────────────────────────
def apply_one_side(valid, score, g, k, add):
    """只动一侧：`add=True` 给 g 加 k 个最好的被驱逐者，否则减 k 个最差的保留者。

    **预算故意不守恒** —— 它只用来做分解项 `J(S∪{i})` / `J(S\\{j})`，
    以便量出交互项 `I_ij`，不作为动作本身。
    """
    v = valid.clone()
    l, h = g
    if add:
        ev = (~v[l, h]).nonzero(as_tuple=True)[0]
        kk = int(min(k, len(ev)))
        if kk:
            v[l, h, ev[torch.argsort(score[l, h][ev], descending=True)[:kk]]] = True
    else:
        rt = v[l, h].nonzero(as_tuple=True)[0]
        kk = int(min(k, len(rt)))
        if kk:
            v[l, h, rt[torch.argsort(score[l, h][rt])[:kk]]] = False
    return v, kk


def apply_transfer(valid, score, i, j, k):
    """在 `valid` 的副本上执行「从 j 拿 k 个给 i」。

    受主 i：把它**被驱逐者里分数最高的 k 个**置 True；
    施主 j：把它**保留者里分数最低的 k 个**置 False。
    —— 这是「最小代价的施予/最大收益的接收」，与 `frontier` 探针同一约定。

    返回 (新 valid, 实际转移数)。若任一侧不足 k，按较小者转移以**严格守恒预算**。
    """
    v = valid.clone()
    li, hi_ = i
    lj, hj = j
    ev = (~v[li, hi_]).nonzero(as_tuple=True)[0]
    rt = v[lj, hj].nonzero(as_tuple=True)[0]
    kk = int(min(k, len(ev), len(rt)))
    if kk == 0:
        return v, 0
    add = ev[torch.argsort(score[li, hi_][ev], descending=True)[:kk]]
    rem = rt[torch.argsort(score[lj, hj][rt])[:kk]]
    v[li, hi_, add] = True
    v[lj, hj, rem] = False
    return v, kk


def frontier_density(valid, score, chunks, w_frac=0.02):
    """逐头**前沿密度** `ρ_h` = 阈值附近单位分数区间内的候选条目数，**逐 chunk 累加**。

    它是把配额空间的目标翻译回分数空间的唯一桥梁。全局 top-B 下
    `b_h = n_h(τ − c_h)`，`n_h` 是该头分数的生存函数，于是（`P = Σρ`）

        R = ∂b/∂c = diag(ρ) − ρρᵀ/P              对称，null(R) = span(1)

    已用有限差分对拍：相关 +0.967、列和恒 0、`‖R·1‖ = 3e−13`。解 `Rc = d` 得

        c_h = d_h / ρ_h + κ                       **除以密度**

    **必须逐 chunk（2026-08-21 修正，首版是错的）**：`kvcache.py:316-318` 的
    `prune_chunk` 只在 `score[..., evict_range]` 上调 `threshold`，所以**每个 chunk
    有自己的 τ_c**，旧决策从不回溯。用一个混合全局 τ 去数密度，会把"分数整体
    落在别的 chunk 阈值附近"的头误判成零密度 —— 首版据此报的「55/112 头零密度」
    是这个错误的产物，已作废。`level="pair"` 下 τ_c 在层与头上是同一个标量，
    所以每 chunk 一个 τ、跨 chunk 求和即可。

    `chunks` 是各 chunk 在**已拼接的可驱逐区**内的 `[(off_lo, off_hi)]`。
    """
    L, H, _ = valid.shape
    rho = np.zeros(L * H)
    w = w_frac * float(score.std())
    taus = []
    for lo, hi in chunks:
        v = valid[..., lo:hi]
        sq = score[..., lo:hi]
        if v.numel() == 0 or not bool(v.any()) or not bool((~v).any()):
            continue                       # 该 chunk 全保留或全驱逐 ⇒ 无前沿
        tau = 0.5 * (float(sq[v].min()) + float(sq[~v].max()))
        taus.append(tau)
        rho += (((sq - tau).abs() < w).sum(-1).float().reshape(-1).cpu().numpy()
                / (2 * w))
    return rho, taus


def presort(valid, score):
    """每个 (层,头) 预排一次序：被驱逐者按分数**降序**、保留者按分数**升序**。

    分数在整轮里不变，所以这一步只做一次；此后构造任何动作都只是切片，
    不必反复对 10 万元素做 argsort（384 个方向 × 56 对 × 2 次排序会主导耗时）。
    """
    L, H, _ = valid.shape
    ev_s, rt_s = {}, {}
    for l in range(L):
        for h in range(H):
            ev = (~valid[l, h]).nonzero(as_tuple=True)[0]
            rt = valid[l, h].nonzero(as_tuple=True)[0]
            ev_s[(l, h)] = ev[torch.argsort(score[l, h][ev], descending=True)]
            rt_s[(l, h)] = rt[torch.argsort(score[l, h][rt])]
    return ev_s, rt_s


def random_delta(valid, ev_s, rt_s, k_nom, rng):
    """一个**严格保预算**的随机配额扰动。

    把全部 G 个 (层,头) 随机两两配对，每对里前者当受主 (+k)、后者当施主 (−k)，
    `k = min(k_nom, 可驱逐数, 可移除数)` 逐对取，所以 `Σ_h Δb_h = 0` **按构造成立**
    （不是靠事后修正凑的 —— 凑法在可行性截断后很难保证整数守恒）。

    为什么要**稠密**扰动而不是逐对：单对挪 16 条只改答案 NLL 约 0.002 nats，
    与问句噪声同量级（实测符号一致 58.3%、可分 2.8%）。同时动 G/2 对能把 |A|
    抬高一到两个量级，再用最小二乘从 N 个方向反解出逐头边际效用。
    """
    L, H, _ = valid.shape
    gs = list(range(L * H)); rng.shuffle(gs)
    v = valid.clone()
    d = np.zeros(L * H, dtype=np.int64)
    for a_, b_ in zip(gs[0::2], gs[1::2]):
        i, j = (a_ // H, a_ % H), (b_ // H, b_ % H)
        kk = int(min(k_nom, len(ev_s[i]), len(rt_s[j])))
        if kk == 0:
            continue
        v[i[0], i[1], ev_s[i][:kk]] = True
        v[j[0], j[1], rt_s[j][:kk]] = False
        d[a_] += kk; d[b_] -= kk
    assert d.sum() == 0, f"预算不守恒 {d.sum()}"
    return v, d


def pick_actions(valid, score, n_recv, n_don, rng):
    """挑候选受主/施主。

    受主：**最好的被驱逐者分数**最高的头 —— 它最"憋屈"；
    施主：**最差的保留者分数**最低的头 —— 它最"浪费"。
    这是纯启发式，只用来把 G² 个 pair 砍到 n_recv×n_don；
    另加 `n_recv` 个**随机受主**作对照，防止启发式本身把结论选出来。
    """
    L, H, _ = valid.shape
    best_ev, worst_rt = [], []
    for l in range(L):
        for h in range(H):
            ev = (~valid[l, h]).nonzero(as_tuple=True)[0]
            rt = valid[l, h].nonzero(as_tuple=True)[0]
            best_ev.append(float(score[l, h][ev].max()) if len(ev) else -1e9)
            worst_rt.append(float(score[l, h][rt].min()) if len(rt) else 1e9)
    best_ev = np.array(best_ev); worst_rt = np.array(worst_rt)
    recv = list(np.argsort(-best_ev)[:n_recv])
    don = list(np.argsort(worst_rt)[:n_don])
    pool = [g for g in range(L * H) if worst_rt[g] < 1e8]
    rnd_recv = rng.sample(pool, min(n_recv, len(pool)))
    return ([(int(g) // H, int(g) % H) for g in recv],
            [(int(g) // H, int(g) % H) for g in don],
            [(int(g) // H, int(g) % H) for g in rnd_recv])


def report_grad(recs, a):
    """从保预算随机方向反解逐头边际效用 `u`，用**留出方向**判定一阶结构。

    模型 `A(Δb) = uᵀΔb + ½Δbᵀ H Δb + 噪声`，只拟一阶项，`u = ∂J/∂b_h`。

    **可识别性**：所有方向满足 `1ᵀΔb = 0`，所以 `u` 只在加常数意义下可识别
    （与 §四之五 的 gauge 自由度同构）。设计矩阵各行和为 0 ⇒ `1 ∈ null(X)` ⇒
    岭解落在行空间里、自动 `⊥ 1`；再显式中心化只是把这一点写明。

    **CV 单位是「方向」**：一个方向的 M 个问句先平均成一个 `A`，然后按方向分折。
    把同一方向的问句拆到 train/val 两边会泄漏。

    **判据只到局部一阶可预测性。** `R²=0.7` 只能说「在所采样的扰动分布下，线性的
    逐头配额势能解释了 70% 的留出效用方差」，**不能**说 `J(b) = Σ_h J_h(b_h)`，
    也不能说全局可分。
    """
    from scipy import stats as st
    ALL = np.array([r["d"] for r in recs], dtype=np.float64)
    EPS = np.array([r.get("eps", -1.0) for r in recs])
    DOC = np.array([r.get("doc", 0) for r in recs])
    # **ρ 是逐篇的**（每篇自己的分数分布与逐 chunk 阈值），不能拿 recs[0] 当全体。
    # 这里按篇平均，与「u 在多篇上合解」保持同一口径。
    rho = None
    if "rho" in recs[0]:
        rho = np.mean([np.array(recs[np.where(DOC == d_)[0][0]]["rho"])
                       for d_ in sorted(set(DOC))], axis=0)
    G = ALL.shape[1]

    def cv_r2(Xa, ya, lam, folds=5, seed=0):
        rs = np.random.default_rng(seed); idx = rs.permutation(len(ya))
        pred = np.zeros_like(ya)
        for f in range(folds):
            te = idx[f::folds]; tr = np.setdiff1d(idx, te)
            Xt, yt = Xa[tr], ya[tr] - ya[tr].mean()
            w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(Xa.shape[1]), Xt.T @ yt)
            pred[te] = Xa[te] @ w + ya[tr].mean()
        return 1 - ((ya - pred) ** 2).sum() / ((ya - ya.mean()) ** 2).sum()

    best_u, summary = None, []
    for ep in sorted(set(EPS)):
        sel = EPS == ep
        X = ALL[sel]; y = np.array([r["A"] for r in recs])[sel]
        Aq = np.array([r["Aq"] for r in recs])[sel]
        N = len(y)
        X = X - X.mean(0); sx = X.std() or 1.0; Xn = X / sx
        print(f"\n════ ε={ep:g}：{N} 个方向 × {G} 组 ════")
        print(f"  |A| 中位 {np.median(np.abs(y)):.5f}  A 均值 {y.mean():+.5f} "
              f"sd {y.std():.5f}  为正 {np.mean(y>0):.1%}")
        print(f"  实际搬动 ½‖Δb‖₁ 中位 {np.median([r['mb'] for r in np.array(recs)[sel]]):.0f}")
        # 设计覆盖：饿死头永远做不了施主 ⇒ 那些列只有单边支撑，系数不可辨
        pos = (ALL[sel] > 0).sum(0); neg = (ALL[sel] < 0).sum(0)
        one = int(((pos == 0) | (neg == 0)).sum())
        print(f"  设计覆盖：单边支撑的列 {one}/{G}"
              f"（这些头的系数与截距混淆，不要单独解读）"
              f"  每列非零方向数中位 {np.median(pos+neg):.0f}")
        if N < 2 * G:
            print(f"  ⚠ N={N} < 2G={2*G}：留出 R² 会偏低，读到接近 0 不能直接判死")
        bb = (-9, None)
        for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
            r2 = cv_r2(Xn, y, lam)
            r2s = cv_r2(Xn, np.random.default_rng(1).permutation(y), lam)
            print(f"    λ={lam:<7g} CV R² = {r2:+.4f}   打乱对照 {r2s:+.4f}")
            if r2 > bb[0]:
                bb = (r2, lam)
        r2, lam = bb
        u = np.linalg.solve(Xn.T @ Xn + lam * np.eye(G), Xn.T @ (y - y.mean())) / sx
        u -= u.mean()
        ua = []
        for sl in (slice(1, None, 2), slice(0, None, 2)):
            yh = Aq[:, sl].mean(1); yh = yh - yh.mean()
            v_ = np.linalg.solve(Xn.T @ Xn + lam * np.eye(G), Xn.T @ yh) / sx
            ua.append(v_ - v_.mean())
        rel = st.spearmanr(ua[0], ua[1])[0]
        print(f"  最好 λ={lam:g}  **CV R² = {r2:+.4f}**   "
              f"奇/偶问句各解一次 u 的 Spearman {rel:+.3f}")
        # **合解假设 u 跨篇共享**。逐篇各解一次并比对，是这个假设的直接检验；
        # 若逐篇 R² 明显高于合解，就说明 u 依赖文档、静态表不成立。
        ds = sorted(set(DOC[sel]))
        if len(ds) > 1:
            pr = []
            for d_ in ds:
                m_ = DOC[sel] == d_
                if m_.sum() < 20:
                    continue
                Xd = Xn[m_]; yd = y[m_]
                pr.append((d_, int(m_.sum()), cv_r2(Xd, yd, lam)))
            if pr:
                print("    逐篇 CV R²：" + "  ".join(
                    f"doc{d_}(n={n_}) {r_:+.3f}" for d_, n_, r_ in pr)
                    + f"   合解 {r2:+.3f}")
        summary.append((ep, N, r2, rel))
        if best_u is None or r2 > best_u[0]:
            best_u = (r2, ep, u)

    print(f"\n════ R²(ε) 汇总 ════")
    for ep, N, r2, rel in summary:
        print(f"  ε={ep:<8g} N={N:<5d} CV R² {r2:+.4f}   u 的奇偶信度 {rel:+.3f}")
    print(f"  读法：小 ε 高、大 ε 低 ⇒ 局部势能存在但曲率随搬动量增长；"
          f"全部 ≈0 ⇒ 该工作点无可学的一阶结构")
    print(f"  **判据（工程 go/no-go，不是定理）**：CV R² ≥ 0.20 才值得训控制器；"
          f"≥ 0.50 则势能参数化足够")

    r2, ep, u = best_u
    if rho is not None:
        ok = rho > 0
        print(f"\n════ 从配额空间翻译回分数空间 ════")
        print(f"  全局 top-B 下 R = ∂b/∂c = diag(ρ) − ρρᵀ/P（对称，null = span(1)，"
              f"有限差分对拍 r=+0.967）")
        print(f"  要让配额走到 d 需解 Rc = d ⇒ **c_h = d_h/ρ_h + κ，除以密度**。")
        print(f"  所以控制器要拟合的**不是** u 本身：")
        print(f"    ρ 与 |u| 的 Spearman {st.spearmanr(rho[ok], np.abs(u[ok]))[0]:+.3f}")
        c_t = np.where(ok, u / np.where(ok, rho, 1), 0.0); c_t -= c_t[ok].mean()
        print(f"    u 的极差   [{u.min():+.3e}, {u.max():+.3e}]")
        print(f"    u/ρ 的极差 [{c_t.min():+.3e}, {c_t.max():+.3e}]"
              f"   两者 Spearman {st.spearmanr(u[ok], c_t[ok])[0]:+.3f}")
        print(f"    ⇒ 若这个相关明显小于 1，说明「学 u 直接当 c」与「学 u/ρ」是"
              f"两个不同的目标，密度不是可忽略的常数。")


# ────────────────────────────── 主流程 ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--level", default="pair")
    ap.add_argument("--max_ctx", type=int, default=131072)
    ap.add_argument("--n_doc", type=int, default=2)
    ap.add_argument("--n_fact", type=int, default=8,
                    help="每篇插入多少条**互不相同**的事实并各问一次。"
                         "首跑单条（答案 12–16 tok）信度只有 48.7%%（抛硬币），"
                         "噪声 ∝ 1/√T ⇒ 这是最直接的放大办法。")
    ap.add_argument("--n_recv", type=int, default=4)
    ap.add_argument("--n_don", type=int, default=4)
    ap.add_argument("--ks", default="1,4,16",
                    help="每个动作转移多少个 KV 条目。**不要只用 k=1** —— "
                         "真实控制器每头挪动的量级是几十到几百条，k=1 既最接近"
                         "数值地板、又不对应任何实际决策粒度；报表会打印 |A| 随 k "
                         "的曲线，若不随 k 增长就说明触到了地板。")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interaction", action="store_true",
                    help="每个动作额外跑 2 次前向，量出交互项 "
                         "I = A − (g⁺ − g⁻)。**这是判断「可分性/势能表示」"
                         "成不成立的唯一办法** —— |I| ≪ |A| 才能把 A 写成 u_i − u_j。")
    ap.add_argument("--mode", default="pair", choices=["pair", "grad"],
                    help="pair=逐 (受主,施主) 对的优势；"
                         "grad=保预算随机稠密扰动 + 最小二乘反解逐头边际效用。"
                         "**首跑判决 pair 不可用**：单对挪 16 条只动 0.002 nats，"
                         "与问句噪声同量级（符号一致 58.3%%、逐动作可分 2.8%%），"
                         "且 |A| 不随 k 增长 ⇒ 已在地板上，加大 k 无用。")
    ap.add_argument("--n_dir", type=int, default=384, help="grad 模式的随机方向数")
    ap.add_argument("--mb", default="0.0025,0.01,0.04",
                    help="grad 模式的搬动量 ½‖Δb‖₁ / B，**逗号分隔可多个**。"
                         "太小则信噪比差，太大则二阶项 ½Δbᵀ H Δb 主导、线性模型"
                         "R² 自然低 —— 所以要看 R²(ε) 而不是单点。默认三档跨 16×。")
    ap.add_argument("--out", default="scratch_adv_probe.json")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]

    m = ModelKVzip(a.model, "retain", a.gate)
    L = m.config.num_hidden_layers
    H = m.config.num_key_value_heads
    texts = [d["context"] for d in load_fineweb("fineweb_10k_cat")][:a.n_doc]
    print(f"文档 {len(texts)} 篇  L={L} H={H}  ρ={a.ratio}  k∈{ks}", flush=True)

    recs = []
    for di, txt in enumerate(texts):
        ids = m.encode(txt)[0].tolist()
        rng = random.Random(a.seed * 1000 + di)
        ctx_ids, qas, meta = make_task(
            m, ids, a.max_ctx, a.window, a.n_fact, rng)
        # ---- 三道守卫，与 `scratch_ctrl_teacher.py` 逐条对齐 ----
        if len(ctx_ids) < a.chunk // 2:
            print(f"doc{di} 太短 ({len(ctx_ids)})，跳过"); continue
        if a.ratio * len(ctx_ids) <= a.window:
            # wrapper.py:273-275 在 ratio·clen < window 时把 chunk_ratio 置 0，
            # `_threshold(·,0)` 取 thres=max ⇒ valid 恒为全 False：保留集等于
            # 局部窗口、与分数无关。此时任何配额转移都是构造性无操作。
            print(f"doc{di}: clen={len(ctx_ids)} ≤ window/ratio="
                  f"{a.window/a.ratio:.0f}，chunk_ratio 会塌缩到 0，跳过")
            continue
        ctx_t = torch.tensor([ctx_ids], device=m.device)
        qas_t = [(torch.tensor([q_ + a_], device=m.device), len(a_)) for q_, a_ in qas]
        n_q = a.n_fact

        # **分数必须在驱逐发生时录下来**（与 `scratch_ctrl_teacher.py` 同一手法）。
        # 真机首跑证明：预填结束后 `kv.score` 只剩最后一段（实测 4096 = window），
        # 拿不到全长；而 `prune_chunk` 里 `torch.stack(self.score,0)[..., lo:hi]`
        # 正好是该 chunk 驱逐区间的分数，逐块拼起来与 `valid` 对齐。
        rec_s = []
        _orig_pc = RetainCache.prune_chunk

        def _pc(self, ratio, evict_range=tuple, level="pair"):
            lo, hi = evict_range
            sc_ = torch.stack(self.score, 0)[..., lo:hi]
            rec_s.append((lo, hi, (sc_[:, 0] if sc_.dim() == 4 else sc_).float().cpu()))
            return _orig_pc(self, ratio, evict_range, level)

        RetainCache.prune_chunk = _pc
        try:
            kv = m.prefill(ctx_t, prefill_chunk_size=a.chunk, do_score=True,
                           chunk_ratio=a.ratio, window_size=a.window, level=a.level)
        finally:
            RetainCache.prune_chunk = _orig_pc
        n_seen = kv._seen_tokens
        raw_valid = kv.valid                                # 形状见下
        if raw_valid is None:
            # `prune_chunk` 一次都没被调用 ⇒ 没有驱逐决策可控。教师那边对应
            # 「doc 没有触发驱逐，跳过」。不拦会在下一行 .clone() 抛 AttributeError。
            print(f"doc{di}: 没有触发驱逐（valid is None），跳过")
            del kv; torch.cuda.empty_cache(); continue
        # **两处必须实测而非假设，第二遍复查各抓到一个 bug：**
        # ① `self.valid` 是 `threshold(score)` 的输出，而 score 是 [L,B,H,n]
        #    ⇒ valid 也是 **[L,B,H,n]**（B=1），不是 [L,H,n]。按后者索引会取错头。
        # ② `RetainCache.prune_chunk` 用 `torch.cat` **累积**各 chunk 的 valid，
        #    而末尾的 local window 从不进 evict_range ⇒ **valid 短于 ctx_len**。
        #    `_get_valid` 靠右侧 pad(True) 补齐。所以 score 必须按 valid 的实际
        #    长度切，不能按 [start_idx:end_idx]。
        vshape = tuple(raw_valid.shape)
        assert raw_valid.dim() in (3, 4), vshape
        base_valid = (raw_valid[:, 0] if raw_valid.dim() == 4 else raw_valid).clone()
        n_ev = base_valid.shape[-1]                          # 实际被驱逐区长度
        # 逐块拼分数。`valid` 覆盖 [start_idx, end_idx)（真机实测 112877 =
        # end−start），而各 chunk 的 evict_range 连续铺满这个区间，所以按 lo 排序
        # 拼接后与 `valid` 逐位对齐。**若区间不连续这里会形状不符而中止，不会静默错位。**
        assert rec_s, "prune_chunk 一次都没触发"
        rec_s.sort(key=lambda x: x[0])
        sc = torch.cat([t for _, _, t in rec_s], dim=-1).to(base_valid.device)
        cov = sum(hi - lo for lo, hi, _ in rec_s)
        # 真机实测：`valid` 覆盖 [start,end) 全上下文（112877），而各 chunk 的
        # evict_range 只铺到 108781 —— 差值**恰为 4096 = window_size**。
        # 末尾这段是**永远保留的局部窗口**，从不进驱逐决策，因此
        # **不能参与转移**（拿它当施主等于动一个方法根本控制不了的集合）。
        # 处理方式：把 valid 与 score 都截到可驱逐区，并断言尾部确实全 True。
        tail = n_ev - cov
        assert tail >= 0 and cov == sum(hi - lo for lo, hi, _ in rec_s), (cov, n_ev)
        if tail:
            assert bool(base_valid[..., cov:].all()), \
                f"尾部 {tail} 列不是全保留，局部窗口假设不成立"
            assert tail == a.window or tail == a.window - kv.start_idx, \
                f"尾部长度 {tail} 既不等于 window={a.window} 也不等于 window−sink"
            base_valid = base_valid[..., :cov].contiguous()
            n_ev = cov
        assert sc.shape[-1] == n_ev, (sc.shape, n_ev)
        assert sc.shape == base_valid.shape, (
            f"score/valid 形状不匹配 sc={tuple(sc.shape)} valid={tuple(base_valid.shape)} "
            f"raw_valid={vshape} score_raw={tuple(kv.score.shape) if torch.is_tensor(kv.score) else ('list', len(kv.score), tuple(kv.score[0].shape))} "
            f"start={kv.start_idx} end={kv.end_idx} n_ev={n_ev}")
        B0 = int(base_valid.sum())

        # ---- 自检④：score 与 valid 的对齐 ----
        # `valid` 由 `prune_chunk` 逐块 `torch.cat` 而成，覆盖 [start_idx,
        # start_idx+n_ev)；若这个区间假设错了，`sc` 与 `valid` 会整体错位，
        # 而错位**不会报错**——只会让「最好的被驱逐者/最差的保留者」全选错。
        # 保留者的分数必须系统性高于被驱逐者，否则立刻中止。
        _mr = float(sc[base_valid].mean()); _me = float(sc[~base_valid].mean())
        assert _mr > _me, f"score/valid 错位：保留 {_mr:.4f} ≤ 驱逐 {_me:.4f}"
        print(f"  自检④ 保留者均分 {_mr:.4f} > 被驱逐者 {_me:.4f} ✓", flush=True)

        j0, j0q = answer_nll(m, kv, qas_t, n_seen)
        print(f"doc{di}: clen={len(ctx_ids)} 保留 {B0} "
              f"({B0/base_valid.numel():.3f})  基线 NLL {j0:.4f}  "
              f"答案 {meta['n_ans']} tok / target {meta['n_tgt']}  "
              f"事实 {meta['n_fact']} 条", flush=True)

        # ---- 自检 ①：零动作必须逐位复现基线 ----
        def _wb(vv):
            """把可驱逐区掩码写回完整形状：尾部局部窗口恒 True，batch 轴按需补。"""
            full = raw_valid.clone()
            tgt = full[:, 0] if full.dim() == 4 else full
            tgt[..., :n_ev] = vv.to(tgt.device)
            return full

        kv.valid = _wb(base_valid)
        j_null, _ = answer_nll(m, kv, qas_t, n_seen)
        assert j_null == j0, f"零动作不复现基线：{j_null} vs {j0}"
        print(f"  自检① 零动作 A = {j0 - j_null:+.3e}（须恰为 0）✓", flush=True)

        if a.mode == "grad":
            ev_s, rt_s = presort(base_valid, sc)
            # k_nom 由目标搬动量反推：Σ_pair k = ε·B ⇒ k_nom ≈ ε·B/(G/2)
            G = base_valid.shape[0] * base_valid.shape[1]
            offs, _c = [], 0
            for _lo, _hi, _t in rec_s:
                offs.append((_c, min(_c + (_hi - _lo), n_ev))); _c += _hi - _lo
            offs = [(x, y) for x, y in offs if y > x]
            rho, taus = frontier_density(base_valid, sc, offs)
            mbs = [float(x) for x in str(a.mb).split(",")]
            print(f"  grad 模式：G={G} 组，每档 {a.n_dir} 个方向，ε∈{mbs}", flush=True)
            print(f"  逐头前沿密度 ρ（{len(taus)} 个 chunk 各自的 τ 累加）："
                  f"中位 {np.median(rho):.1f} 极差 [{rho.min():.2f}, {rho.max():.1f}]"
                  f"  零密度头 {(rho<=0).sum()}/{G}", flush=True)
            print(f"  各 chunk 阈值 τ_c：{['%.4f' % t for t in taus]}", flush=True)
            for mb in mbs:
                k_nom = max(1, int(round(mb * B0 / (G / 2))))
                print(f"  --- ε={mb:g} ⇒ 目标搬动 {mb*B0:.0f}，每对 k≈{k_nom}", flush=True)
                for it in range(a.n_dir):
                    v, d = random_delta(base_valid, ev_s, rt_s, k_nom, rng)
                    assert int(v.sum()) == B0, f"预算不守恒 {int(v.sum())} vs {B0}"
                    kv.valid = _wb(v)
                    jj, jjq = answer_nll(m, kv, qas_t, n_seen)
                    recs.append(dict(doc=di, mode="grad", eps=mb, d=d.tolist(),
                                     rho=[float(x) for x in rho],
                                     taus=[float(x) for x in taus],
                                     mb=float(np.abs(d).sum() / 2),
                                     A=float((j0q - jjq).mean()),
                                     Aq=[float(x) for x in (j0q - jjq)]))
                    if (it + 1) % 64 == 0:
                        aa = np.array([r["A"] for r in recs
                                       if r.get("mode") == "grad" and r["eps"] == mb])
                        print(f"    {it+1}/{a.n_dir}  |A| 中位 "
                              f"{np.median(np.abs(aa)):.5f}", flush=True)
                        # **增量落盘**：整轮要一两个小时，只在末尾写一次意味着
                        # 中途完全不可读，也经不起一次崩溃。写临时文件再 replace，
                        # 读的一方永远看到完整 json。
                        _tmp = os.path.join(ROOT, a.out + ".part")
                        json.dump(recs, open(_tmp, "w")); os.replace(
                            _tmp, os.path.join(ROOT, a.out))
            kv.valid = raw_valid
            del kv
            torch.cuda.empty_cache()
            continue

        recv, don, rnd = pick_actions(base_valid, sc, a.n_recv, a.n_don, rng)
        acts = [(i, j, k, "heur") for i in recv for j in don for k in ks]
        acts += [(i, j, k, "rand") for i in rnd for j in don[:1] for k in ks]

        for (i, j, k, tag) in acts:
            if i == j:
                continue
            v, kk = apply_transfer(base_valid, sc, i, j, k)
            if kk == 0:
                continue
            assert int(v.sum()) == B0, f"预算不守恒 {int(v.sum())} vs {B0}"  # 自检②
            kv.valid = _wb(v)
            jj, jjq = answer_nll(m, kv, qas_t, n_seen)
            Aq = (j0q - jjq)            # J = −NLL ⇒ A = NLL0 − NLL'，逐问句
            rec = dict(doc=di, recv=list(i), don=list(j), k=kk, tag=tag,
                       A=float(Aq.mean()), Aq=[float(x) for x in Aq],
                       A_odd=float(Aq[1::2].mean()), A_even=float(Aq[0::2].mean()))
            if a.interaction:
                # **交互项 I_ij** —— 外部评审的核心数学批评：
                #     A_{i←j} = J(S∪{i}\{j}) − J(S)
                #             = [J(S∪{i})−J(S)] + [J(S∪{i}\{j})−J(S∪{i})]
                # 第二项条件于「i 已加入」，与条件于 S 的 −g⁻_j **一般不等**，
                # 因为 softmax 分母同时被两侧改动：Z → Z + e^{qk_i} − e^{qk_j}。
                # 所以 `A ≈ g⁺−g⁻` 是**近似**，可分性必须测不能假设。
                # 两次额外前向即可量出 I：
                v_add, _ = apply_one_side(base_valid, sc, i, kk, True)
                kv.valid = _wb(v_add)
                j_add, _ = answer_nll(m, kv, qas_t, n_seen)
                v_rem, _ = apply_one_side(base_valid, sc, j, kk, False)
                kv.valid = _wb(v_rem)
                j_rem, _ = answer_nll(m, kv, qas_t, n_seen)
                gp = j0 - j_add          # 加 i 的收益（NLL 降多少）
                gm = j_rem - j0          # 减 j 的代价（NLL 升多少）
                rec.update(gp=gp, gm=gm, I=float(Aq.mean()) - (gp - gm))
            recs.append(rec)
        kv.valid = raw_valid
        del kv
        torch.cuda.empty_cache()

    if not recs:
        print("没有任何动作被评估"); return
    if a.mode == "grad":
        report_grad(recs, a)
        json.dump(recs, open(os.path.join(ROOT, a.out), "w"))
        print(f"\n写出 {a.out}")
        return

    A = np.array([r["A"] for r in recs])
    Aq = np.array([r["Aq"] for r in recs])                    # [n_act, n_q]
    h1 = np.array([r["A_odd"] for r in recs])                 # 奇数问句
    h2 = np.array([r["A_even"] for r in recs])                # 偶数问句
    from scipy import stats as st
    print(f"\n=== {len(A)} 个动作 ===")
    print(f"  A 均值 {A.mean():+.5f}  sd {A.std():.5f}  为正 {np.mean(A>0):.1%}")
    print(f"  |A| 中位 {np.median(np.abs(A)):.5f}  最大 {np.abs(A).max():.5f}")

    # ---- 对照：|A| 是否随 k 增长。这同时是**数值噪声地板**的检验 ----
    # 掩码相同 ⇒ 前向逐位相同（自检①测到 A 恰为 0），所以没有"重复测量抖动"；
    # 但掩码不同会改变 flash-attn 的归约长度，仍有 rounding 差。若 |A| 在 k=1
    # 与 k=16 上一样大，说明测到的是地板而不是效应。
    print(f"\n=== |A| 随 k（若不增长 ⇒ 触到数值/离散地板，k=1 不可用）===")
    for kk in sorted({r["k"] for r in recs}):
        sub = np.array([r["A"] for r in recs if r["k"] == kk])
        print(f"  k={kk:<4d} n={len(sub):<4d} |A| 中位 {np.median(np.abs(sub)):.5f}"
              f"  均值 {sub.mean():+.5f}")

    # ---- 自检③：信度。**按问句奇偶交错对半**，不是按 token 位置 ----
    # 为什么不按 token 位置：前半/后半答案 token 不是同一潜在量的重复测量
    # （教师强制下后半条件于答案前缀，对缓存依赖更弱），所以那种分法的低一致率
    # 无法区分"估计量噪声大"与"真效应本来就随位置变"。按**独立事实**分则两半
    # 估的是同一个 Ā = E_q[·]，才是真正的重复测量。取奇/偶而非前/后，是为了让
    # 两半的平均问句位置相同，抵消拼接带来的顺序效应。
    print(f"\n=== 自检③ 信度（奇数问句 A vs 偶数问句 A，n_q={Aq.shape[1]}）===")
    r_half = st.pearsonr(h1, h2)[0]
    r_sb = 2 * r_half / (1 + r_half) if r_half > -1 else float("nan")
    print(f"  Pearson  {r_half:+.3f}   Spearman {st.spearmanr(h1,h2)[0]:+.3f}")
    print(f"  Spearman-Brown 校正到全量 {Aq.shape[1]} 问句: r = {r_sb:+.3f}")
    nz = (h1 != 0) & (h2 != 0)
    print(f"  符号一致率 {np.mean(np.sign(h1[nz])==np.sign(h2[nz])):.1%}（n={nz.sum()}）")

    # ---- 逐动作 bootstrap（对**问句**重采样）----
    rs = np.random.default_rng(0)
    nq = Aq.shape[1]
    bs = Aq[:, rs.integers(0, nq, size=(2000, nq))].mean(-1)   # [n_act, 2000]
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=1)
    pos, neg = (lo > 0), (hi < 0)
    print(f"\n=== 逐动作 95% bootstrap CI（重采样问句，2000 次）===")
    print(f"  有益 (CI 下界>0) {pos.sum()}/{len(A)} = {pos.mean():.1%}")
    print(f"  有害 (CI 上界<0) {neg.sum()}/{len(A)} = {neg.mean():.1%}")
    print(f"  不可分 (0∈CI)   {(~pos & ~neg).sum()}/{len(A)} = {(~pos & ~neg).mean():.1%}")
    print(f"  ⇒ 判据：**可分比例 ≥ 50% 且信度 r_sb ≥ 0.6、符号一致 ≥ 75%** 才可训练；")
    print(f"    否则标签噪声主导。注意反过来不成立 —— 低一致率只说明**这个估计量**")
    print(f"    方差过大，不说明真优势为零。")

    if a.interaction and "I" in recs[0]:
        I = np.array([r["I"] for r in recs])
        GP = np.array([r["gp"] for r in recs]); GM = np.array([r["gm"] for r in recs])
        appr = GP - GM
        print(f"\n=== 交互项 I = A − (g⁺ − g⁻)：可分性/势能表示成不成立 ===")
        print(f"  |A| 中位 {np.median(np.abs(A)):.5f}   |I| 中位 {np.median(np.abs(I)):.5f}"
              f"   **|I|/|A| 中位 {np.median(np.abs(I))/max(np.median(np.abs(A)),1e-12):.3f}**")
        print(f"  近似 (g⁺−g⁻) 与精确 A：Pearson {st.pearsonr(appr,A)[0]:+.3f}  "
              f"Spearman {st.spearmanr(appr,A)[0]:+.3f}  "
              f"符号一致 {np.mean(np.sign(appr)==np.sign(A)):.1%}")
        print(f"  ⇒ |I|/|A| 远小于 1 且符号一致率高 ⇒ 可写成势能 u_i−u_j；否则**不能**")
    hp = [r for r in recs if r["tag"] == "heur"]; rp = [r for r in recs if r["tag"] == "rand"]
    if hp and rp:
        print(f"\n=== 启发式受主 vs 随机受主（对照）===")
        print(f"  启发式 A 均值 {np.mean([r['A'] for r in hp]):+.5f} (n={len(hp)})")
        print(f"  随机   A 均值 {np.mean([r['A'] for r in rp]):+.5f} (n={len(rp)})")
    json.dump(recs, open(os.path.join(ROOT, a.out), "w"))
    print(f"\n写出 {a.out}")


if __name__ == "__main__":
    main()
