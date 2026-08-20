"""配额投影：**唯一实现**。生产（learned_ctrlcache）与单测（scratch_test_project）
共用本文件，不允许任何一方镜像复制。

**为什么必须唯一**：镜像实现会让"生产改了、测试没改"或"两边带同一个错"时测试仍然
全绿。本项目已被离散化坑掉两批实验，不能再留这个口子。

数学约定
--------
真实 cache 分配属于整数单纯形

    B = { b ∈ Z_{≥0}^{L·H} : Σ b = B_tot, 0 ≤ b_g ≤ n }

所以运行时执行的是 `b = Π_B(b⁰ + Δ)`，而 **`Π(x+y) ≠ Π(x)+Π(y)`** —— 连续空间的
正交分解不蕴含离散实现的分解。三种模式因此各自把约束写进投影，而不是共用一个
target 让投影去"自动消掉"多余分量：

    full   : 直接投影 b⁰+Δ，全局配平到 B_tot
    within : Δ^W_{l,h} = Δ_{l,h} − mean_h Δ_{l,·}；每层总量锁死基线，层内配平
             （显式去均值是必需的：零配额头上的 clamp(0,n) 破坏对称性，层常数分量
               会借 clamp 泄漏进层内再分配；旧写法在 Δ_{l,h}=c_l 单测上 20/20 失败）
    across : ① 28 维层总量先整数化并严格配平到 B_tot
             ② 层内按**基线比例**分配，逐层严格配平到该层目标
             （**不做 112 维全局配平** —— 那会让 ±1 落到任意 (层,头) 上，破坏
               "只改层总量"的因果不变量）

within / across 不要求相加等于 full：它们是**各自锁死一个约束以隔离一个通道**的
两个干预，不是加性分解。
"""
import os

import torch


def rebalance(bt, tgt, total, hi):
    """把整数向量 `bt` 配平到 `sum == total`，按 `tgt − bt` 的小数余数优先，界 [0, hi]。

    迭代而非单轮最大余数法：`tgt` 先被 clamp，clamp 造成的缺口可能远大于可调项数，
    而每项每轮只能移动 ±1（实测 Δb=±9999 时缺口 18、可减项只有 11，一轮补不完）。
    """
    diff = int(total) - int(bt.sum().item())
    while diff != 0:
        if diff > 0:
            room = (bt < hi).nonzero().flatten()
            if room.numel() == 0:
                break
            take = min(diff, room.numel())
            bt[room[torch.argsort((tgt - bt.float())[room], descending=True)[:take]]] += 1
            diff -= take
        else:
            room = (bt > 0).nonzero().flatten()
            if room.numel() == 0:
                break
            take = min(-diff, room.numel())
            bt[room[torch.argsort((tgt - bt.float())[room])[:take]]] -= 1
            diff += take
    return bt


def _boundary_pair(ss_asc, q):
    """→ (s_{(q)}, s_{(q+1)})，降序名次。`ss_asc` 升序、形状 [G, n]，`q` 形状 [G]。

    约定与 `scratch_test_reach.py` 逐字一致：`q=0` 时 `s_{(0)} = +inf`（上界约束
    vacuous），`q=n` 时 `s_{(n+1)} = −inf`（下界约束 vacuous）。
    """
    G, n = ss_asc.shape
    ar = torch.arange(G, device=ss_asc.device)
    inf = torch.full((G,), float("inf"), device=ss_asc.device, dtype=ss_asc.dtype)
    i_q = (n - q).clamp(0, n - 1)
    s_q = torch.where(q >= 1, ss_asc[ar, i_q], inf)
    i_q1 = (n - q - 1).clamp(0, n - 1)
    s_q1 = torch.where(q < n, ss_asc[ar, i_q1], -inf)
    return s_q, s_q1


def slack_of(q, ss_asc, a):
    """规范不变的可达性判据：>0 ⟺ 存在公共阈值 τ′ 使 `q` 在 `|Δs| ≤ a` 下可实现。"""
    s_q, s_q1 = _boundary_pair(ss_asc, q)
    return float((s_q + a).min() - (s_q1 - a).max())


def reachable_project(tgt, sc, Btot, alpha, n_tau=2048, certify=True, sigma=None):
    """把目标配额投影到**放宽的有界分数可达集** `Q_box` 上。

    **命名很重要，此前的 docstring 写错了。** 三个集合的严格关系是

        Q_real  ⊆  Q_box  ⊆  Q_F = { b ∈ Z_{≥0}^{G} : Σb = B, 0 ≤ b_g ≤ n }

    `Q_real` 是真实网络 `Δs_{h,i} = g·α·σ_h·tanh(f_θ(x,h,i))` 能实现的配额集；
    `Q_box` 只要求 `|Δs_{h,i}| ≤ a_h`，**允许每个 token 独立任取** —— 比真实网络强。
    本函数投影到 `Q_box`。所以「投影后收益塌掉」是关于 `Q_box` 的陈述，
    对 `Q_real` 只能作为**必要条件**方向使用；反之若收益保住，也**不能**直接说
    真实网络学得到（`Q_real` 更小）。

    动机：`q_floor ∉ Q_box`（74/74 chunk）**推不出**「地板那 +33.60 取不到」——
    可达集是 G−1 维里的一大块，地板目标只是其中一点。本函数找一个**近旁**的
    可达点，下游分数由评测给出。

    可达集的刻画：`|Δs_{h,i}| ≤ a_h` 时，公共阈值 `τ` 下头 `h` 的配额范围是

        q_min_h(τ) = #{i : s_{h,i} > τ + a_h}   ≤  q_h  ≤  q_max_h(τ) = #{i : s_{h,i} > τ − a_h}

    **可行 τ 的集合恰是一个区间**，而且端点有闭式：`Σq_max` 与 `Σq_min` 都是
    τ 的非增阶梯函数，于是

        τ 可行 ⟺ Σq_min(τ) ≤ B ≤ Σq_max(τ)
               ⟺ τ ∈ [ 第(B+1)大的 {s−a},  第B大的 {s+a} )

    并且总有 `τ_lo ≤ τ_hi`（因为 `s−a ≤ s+a` 逐元素成立）⇒ **可达集永不为空**。
    先闭式定位这个区间、再在**区间内**采样，取代了原先在整个分数范围上撒 1024 个
    均匀点的做法 —— 那种做法可能整个错过一个很窄的可行区间，进而走到「找不到解」
    的分支。旧代码在那个分支上**静默退回地板配额**，也就是把本实验悄悄换成它要
    对比的另一臂；这类静默回退是本项目反复栽跟头的同一个模式，现在改为直接抛错。

    **返回的不是严格 L1 最近点。** 区间内按 clamp 距离（真 L1 的下界）排序后只对
    前若干个 τ 做整数配平，所以正确叫法是**「可行 τ 区间内的近似最近 box-可达投影」**。
    要严格最近需要分支定界。`certify=True` 时用 `slack_of` 独立复核返回值确实可达。
    """
    G = tgt.numel()
    # **与 calib_scorer.delta 逐字对齐**：那边是 `f0 = score0[:,0].float()`，
    # `sig_h = f0.std(-1)`，随后在 delta() 里 `.clamp_min(1e-6)`。差一个 float()
    # 或差一个 clamp，这个理论关键实验建模的就不是真实控制器的界。
    X = sc.reshape(G, -1).float()
    # `sigma=None`（生产路径）时从 `sc` 现算，与 calib_scorer.delta 逐字一致。
    # **离线探针必须显式传 `sigma`**：teacher trace 里的 `s0` 只有近阈值候选
    # （每 (chunk,层,头) 256~768 个），在这个截断池上重算 std 会系统性地
    # 低估界（实测 stored/pool 比值 p90 达 26.6×），于是可达集被算得过小。
    # trace 里存的 `sig_h` 是建 trace 时用**整块**算的，才是对的那个。
    a = (float(alpha) * X.std(dim=-1).clamp_min(1e-6) if sigma is None
         else float(alpha) * sigma.reshape(-1).float().clamp_min(1e-6))
    assert a.numel() == G, f"sigma 形状不对: {a.numel()} != {G}"
    ss, _ = torch.sort(X, dim=-1)                       # 升序
    n_tok = ss.shape[1]

    # ---- 闭式定位可行 τ 区间 -------------------------------------------------
    hi_pool = (X + a[:, None]).reshape(-1)
    lo_pool = (X - a[:, None]).reshape(-1)
    N = hi_pool.numel()
    kB = min(max(int(Btot), 1), N)
    tau_hi = float(torch.topk(hi_pool, kB, largest=True).values[-1])      # τ < tau_hi
    kB1 = min(int(Btot) + 1, N)
    tau_lo = float(torch.topk(lo_pool, kB1, largest=True).values[-1])     # τ ≥ tau_lo
    if not (tau_lo < tau_hi):
        # 数学上不该发生（s−a ≤ s+a ⇒ τ_lo ≤ τ_hi）。发生就是实现有 bug，必须炸。
        raise RuntimeError(f"可行 τ 区间为空：tau_lo={tau_lo} tau_hi={tau_hi} B={Btot}")
    taus = torch.linspace(tau_lo, tau_hi, n_tau + 2,
                          device=X.device, dtype=X.dtype)[:-1]            # 半开，去掉右端

    # ---- 区间内扫描：先算 clamp 距离（真 L1 的下界），只对最好的几个做整数配平 ----
    qmax = n_tok - torch.searchsorted(ss, (taus[:, None] - a[None, :]).T.contiguous(),
                                      right=True).T
    qmin = n_tok - torch.searchsorted(ss, (taus[:, None] + a[None, :]).T.contiguous(),
                                      right=True).T
    feas = (qmin.sum(1) <= Btot) & (qmax.sum(1) >= Btot)
    if not bool(feas.any()):
        raise RuntimeError("闭式区间内竟无可行 τ —— 端点推导或 searchsorted 语义有误")
    qc = torch.minimum(torch.maximum(tgt[None, :], qmin), qmax)
    lb = (qc - tgt[None, :]).abs().sum(1).float()
    lb = torch.where(feas, lb, torch.full_like(lb, float("inf")))
    cand = torch.argsort(lb)[:16].tolist()

    best, best_d = None, None
    for ti in cand:
        if not bool(feas[ti]):
            continue
        q = qc[ti].clone()
        d = Btot - int(q.sum())
        if d != 0:
            room = (qmax[ti] - q) if d > 0 else (q - qmin[ti])
            order = torch.argsort((tgt - q).float(), descending=(d > 0)).tolist()
            for g_ in order:
                if d == 0:
                    break
                t = min(abs(d), int(room[g_]))
                q[g_] += t if d > 0 else -t
                d -= t if d > 0 else -t
        if d != 0:
            continue
        dist = float((q - tgt).abs().sum())
        if best_d is None or dist < best_d:
            best_d, best = dist, q.clone()
    if best is None:
        raise RuntimeError("可行 τ 存在但整数配平全部失败 —— rebalance 逻辑有误")
    if certify:
        sl = slack_of(best, ss, a)
        if not (sl > 0):
            raise RuntimeError(f"投影结果未通过独立可达性复核：slack={sl}")
    return best, best_d


def max_lift_quota(b0, sc, Btot, alpha, n_tau=2048, certify=True, sigma=None):
    """`Q_box` 内**最大化饿死头总抬升**的那个配额点，闭式 + 网格取 τ*。

    为什么值得单独跑：`scratch_probe_projdir.py` 已经算出「整个 `Q_box` 里最多只能
    把饿死头抬起地板意图的 2.25%」。那是个**关于集合的量**，但它只说了「能抬多少」，
    没说「这点抬升值多少分」。本函数构造出**那个最大抬升点**，交给评测回答后半句。

    与 `floorproj` 的区别很关键：`floorproj` 找的是**离地板目标 L1 最近**的可达点，
    实测它与地板位移**余弦仅 0.046**（是另一个干预），分数为零也解释不了什么。
    本函数直接**最大化地板所关心的那个量**，所以它是「可达集能做到的最好的
    抬饿死头」的真实上界点。

    构造（给定 τ，箱子是 `q_min(τ) ≤ q ≤ q_max(τ)`，S = 零配额头）：
        Σ_S q 的最大值 = min( Σ_S q_max,  B − Σ_{S^c} q_min )
    取到它时 S 尽量高、S^c 尽量低；**剩余预算按「尽量靠近 b⁰」还给 S^c**，
    这样这个点的语义是「在可达范围内尽力抬饿死头，其余尽量不动」。
    再对可行 τ 取使总抬升最大的那个（R(τ) 是两反向单调函数的 min ⇒ 单峰）。
    """
    G = b0.numel()
    X = sc.reshape(G, -1).float()
    a = (float(alpha) * X.std(dim=-1).clamp_min(1e-6) if sigma is None
         else float(alpha) * sigma.reshape(-1).float().clamp_min(1e-6))   # 见 reachable_project
    assert a.numel() == G, f"sigma 形状不对: {a.numel()} != {G}"
    ss, _ = torch.sort(X, dim=-1)
    n_tok = ss.shape[1]
    S = (b0 == 0)
    if not bool(S.any()):
        return None, 0.0
    hi_pool = (X + a[:, None]).reshape(-1); lo_pool = (X - a[:, None]).reshape(-1)
    N = hi_pool.numel()
    tau_hi = float(torch.topk(hi_pool, min(max(int(Btot), 1), N), largest=True).values[-1])
    tau_lo = float(torch.topk(lo_pool, min(int(Btot) + 1, N), largest=True).values[-1])
    if not (tau_lo < tau_hi):
        raise RuntimeError(f"可行 τ 区间为空 tau_lo={tau_lo} tau_hi={tau_hi}")
    taus = torch.linspace(tau_lo, tau_hi, n_tau + 2, device=X.device, dtype=X.dtype)[:-1]
    qmax = n_tok - torch.searchsorted(ss, (taus[:, None] - a[None, :]).T.contiguous(),
                                      right=True).T
    qmin = n_tok - torch.searchsorted(ss, (taus[:, None] + a[None, :]).T.contiguous(),
                                      right=True).T
    feas = (qmin.sum(1) <= Btot) & (qmax.sum(1) >= Btot)
    if not bool(feas.any()):
        raise RuntimeError("闭式区间内无可行 τ")
    capS = qmax[:, S].sum(1)
    budg = Btot - qmin[:, ~S].sum(1)
    R = torch.minimum(capS, budg)
    R = torch.where(feas, R, torch.full_like(R, -(1 << 30)))
    ti = int(torch.argmax(R))
    qmn, qmx = qmin[ti], qmax[ti]
    q = torch.empty_like(b0, dtype=torch.long)
    # S 尽量高，但总量不超过 min(Σ_S qmax, B − Σ_{S^c} qmin)
    tgtS = int(min(int(qmx[S].sum()), Btot - int(qmn[~S].sum())))
    qS = qmx[S].clone(); over = int(qS.sum()) - tgtS
    if over > 0:                       # 从 S 里按「离 qmin 余量大」的先削
        room = (qS - qmn[S])
        order = torch.argsort(room, descending=True).tolist()
        for j in order:
            if over == 0:
                break
            t = min(over, int(room[j]))
            qS[j] -= t; over -= t
        if over != 0:
            raise RuntimeError("S 侧削减未能配平")
    q[S] = qS
    # S^c 尽量靠近 b⁰，再用剩余预算配平
    qC = torch.minimum(torch.maximum(b0[~S].long(), qmn[~S]), qmx[~S])
    need = Btot - int(q[S].sum()) - int(qC.sum())
    if need != 0:
        room = (qmx[~S] - qC) if need > 0 else (qC - qmn[~S])
        order = torch.argsort(room, descending=True).tolist()
        for j in order:
            if need == 0:
                break
            t = min(abs(need), int(room[j]))
            qC[j] += t if need > 0 else -t
            need -= t if need > 0 else -t
        if need != 0:
            raise RuntimeError("S^c 侧配平失败")
    q[~S] = qC
    # 上面两处 `RuntimeError` 是**真正的 bug 探测器**，不是兜底：由可行性
    # `Σq_min ≤ B ≤ Σq_max` 可证两侧总能配平 ——
    #   S 侧：tgtS ≥ Σ_S q_min（因 B ≥ Σq_min），所以削减到得了；
    #   S^c 下界：B − tgtS ≥ Σ_{S^c} q_min（tgtS 两种取值分别验证）；
    #   S^c 上界：B − tgtS ≤ Σ_{S^c} q_max（因 B ≤ Σq_max 且 q_min ≤ q_max）。
    # 所以一旦抛错，就是实现错了，必须让作业崩掉而不是产出一个别的干预。
    lift = float((q[S] - b0[S].long()).clamp(min=0).sum())
    if certify:
        sl = slack_of(q, ss, a)
        if not (sl > 0):
            raise RuntimeError(f"maxlift 结果未通过可达性复核 slack={sl}")
        if int(q.sum()) != int(Btot):
            raise RuntimeError(f"预算不守恒 {int(q.sum())} != {Btot}")
    return q, lift


def project_quota(b0, delta, n, mode, n_layers, n_heads, sc=None, alpha_eff=None):
    """b0, delta: [L*H] float tensor；返回 [L*H] long tensor。

    `mode` ∈ {full, within, across}。非 full 模式下预算守恒是**构造性**的，
    函数末尾直接断言而不做兜底修补 —— 若将来投影出 bug，必须让它崩，而不是被
    一个通用修补循环悄悄"修好"同时破坏因果不变量（那正是本项目栽过的坑）。
    """
    assert mode in ("full", "within", "across", "floor", "floorproj",
                    "pathproj", "floorpath", "maxlift", "floorcov"), mode
    L, H = int(n_layers), int(n_heads)
    assert b0.numel() == L * H and delta.numel() == L * H
    Btot = int(b0.sum().item())
    tgt = (b0 + delta).clamp(0, n)

    if mode == "maxlift":
        # **可达集内最大抬升点**：补完集合级结论的后半句（那 2.25% 值多少分）。
        assert sc is not None and alpha_eff is not None, "maxlift 需要 sc 与 alpha_eff"
        q, lift = max_lift_quota(b0, sc, Btot, alpha_eff)
        if q is None:
            raise RuntimeError("maxlift：本 chunk 没有零配额头")
        project_quota._ml_lift = getattr(project_quota, "_ml_lift", 0.0) + lift
        project_quota._ml_n = getattr(project_quota, "_ml_n", 0) + 1
        return q

    if mode in ("floor", "floorproj", "pathproj", "floorpath", "floorcov"):
        # **防饿死对照**：完全不用 `delta` 的方向，只强制 b_g ≥ b_min，
        # 缺口按 (b⁰ − b_min)⁺ 的比例从富余头等量扣回，总预算不变。
        # 为什么必须有这个对照：`fastkvzip@pair` 在 ρ=0.2 有 41.3% 的头零配额，
        # 而 `adakv-layer` 的 `safeguard=0.2` 本身就是一个逐头地板 ——
        # 若我们的 +25.80 主要只是"别把头饿死"，整套配额校准理论就塌成一个
        # 已被 Ada-KV 覆盖的启发式。离线预检显示学到的正向配额只有 2.3% 流向
        # 饿死头（随机应为 41.3%），但**质量不等于效果**，必须真跑。
        bmin = float(os.environ.get("VARIKV_QUOTA_FLOOR", "0"))
        bmin = min(bmin, float(n))
        # **地板不可行时饱和到均匀分配，而不是崩。** 文档末尾的短 chunk 总预算可能
        # 小于 `b_min × 112`（实测 PrefSuf 有 Btot=3514 的 chunk，`b_min=32` 需要
        # 3584）。此时"每个头至少 b_min"在数学上无解，其**连续极限**就是均匀分配
        # `Btot/(L·H)`，所以取 `min(b_min, Btot//(L·H))`。
        # 这不是把断言改宽：断言原本就抓到了真实的不可行，只是原来的处理方式
        # （直接崩）让整个作业挂掉，而正确的降级是走到该约束的边界。
        bmin = min(bmin, float(Btot // (L * H)))
        t = torch.maximum(b0, torch.full_like(b0, bmin))
        if mode == "floorcov":
            # **覆盖率剂量轴**：固定每头抬到 `b_min`（配 b_min=1 就是每头 +1），
            # 只改**抬多少个头**。这是把「广度」当自变量的直接实验 ——
            # `maxlift` 已证明 214 单位堆在 9/60 个头上只值 +5.00，而 60 单位铺满
            # 60/60 值 +25.80；本模式在两者之间连线。
            # 选头规则**只用基线分数、不用学到的方向**：按 `s_max` 降序取前 k 个，
            # 即「离全局阈值最近、最便宜抬」的那些 —— 也正是可达集会选的那批，
            # 所以 f≈0.155 这一点可与 `maxlift` 的覆盖率直接对照。
            # f=1 时 `t` 与 floor 完全相同（下面的断言保证），故它是可检验的退化点。
            assert sc is not None, "floorcov 需要 score 张量"
            fr = float(os.environ.get("VARIKV_COV_FRAC", "1.0"))
            below = (b0 < bmin)
            nb = int(below.sum())
            k = int(round(fr * nb))
            if k < nb:
                # **选头顺序是一个独立的自变量**，必须能切换：`smax` 是「最便宜抬」
                # （也是可达集会选的那批），`index` 是与分数无关的任意顺序。
                # 两者若给出同一条曲线 ⇒ 起作用的是**覆盖率本身**；若不同 ⇒
                # 「选了哪些头」也携带信息，覆盖率不是唯一变量。
                cand = torch.nonzero(below).flatten()
                _ord = os.environ.get("VARIKV_COV_ORDER", "smax")
                assert _ord in ("smax", "index", "revindex"), f"未知 VARIKV_COV_ORDER={_ord}"
                if _ord == "smax":
                    smax = sc.reshape(b0.numel(), -1).float().max(dim=-1).values
                    pick = cand[torch.argsort(smax[cand], descending=True)[:k]]
                elif _ord == "index":
                    # 头编号 g = layer*H + head ⇒ **编号最小 = 最早的层**。
                    # 实测 f=0.15 时它 95% 落在 L0–L2，且给出 +25.00★（几乎全部效应），
                    # 而 smax 顺序（层中位 15、39% 落在 L≥20）只给 +0.80 ns。
                    pick = cand[:k]
                else:
                    # `revindex`：编号最大 = **最晚的层**。这是 `index` 的镜像对照 ——
                    # 用来分辨「早层特殊」与「连续一段就行」。两者头数相同、
                    # 每头抬升相同，只有层位置相反。
                    pick = cand[-k:] if k > 0 else cand[:0]
                t = b0.clone()
                t[pick] = bmin
            # k == nb 时落回上一行的 `t`，与 floor 逐位相同
        excess = float(t.sum() - Btot)                      # ≥ 0
        if excess > 0:
            room = (b0 - bmin).clamp(min=0)                 # 可扣回的量
            tot_room = float(room.sum())
            assert tot_room >= excess, "富余不足以填地板"
            t = t - room * (excess / tot_room)
        t = t.clamp(0, n)
        bt = rebalance(t.round().long().clamp(0, n), t, Btot, n)
        if mode in ("pathproj", "floorpath"):
            # **可达效用探针（便宜的一档）**：沿 baseline → floor 方向取 λ 处的目标，
            # 再投影回 `Q_box`。目的见 reachable_project 的 docstring —— `floorproj`
            # （λ=1）只回答「地板那一点附近」，而我们真正想估的是
            #     max_{q ∈ Q_box} J(q)
            # 扫 λ 得到这个上确界在**一条一维路径上**的下界。λ=0 退化为基线配额
            # （本就可达，投影是恒等），λ=1 退化为 floorproj。
            # `floorpath` = 只插值、**不投影**：这是「地板机制」的**剂量–反应**轴。
            # 为什么必须是这条轴而不是 `b_min` 扫描：改 `b_min` 会同时改变抬升量、
            # 哪些头是供给头、以及全局阈值，三者混在一起；而 λ 沿**同一方向**
            # 线性缩放同一个位移，头集合与方向都不变，是干净的剂量。
            # 它要回答的前提是：「地板的 +33.60 是不是来自把饿死头抬起来」。
            # 若收益随 λ 单调上升 ⇒ 前提成立，配合「整个 Q_box 最多只能抬 2.25%」
            # 就直接推出可达性是主因；若小 λ 就拿满收益 ⇒ 收益另有来源。
            lam = float(os.environ.get("VARIKV_PROJ_LAMBDA", "1.0"))
            tp = (1.0 - lam) * b0 + lam * bt.float()
            bt = rebalance(tp.round().long().clamp(0, n), tp, Btot, n)

        if mode in ("floorproj", "pathproj"):
            # 把地板目标投影到**放宽的有界分数可达集** `Q_box`（注意不是 `Q_real`，
            # 见 reachable_project 的 docstring）。**没有回退分支**：投影失败必须
            # 让作业崩掉，而不是悄悄退回地板配额 —— 那等于把这个判决实验换成它
            # 正要对比的另一臂。
            assert sc is not None, "floorproj 需要 score 张量"
            assert alpha_eff is not None, (
                "floorproj 的界必须从 controller/ckpt 构造性取得，不能手抄第二份")
            q, d = reachable_project(bt, sc, Btot, alpha=alpha_eff)
            project_quota._proj_l1 = getattr(project_quota, "_proj_l1", 0.0) + d
            project_quota._proj_n = getattr(project_quota, "_proj_n", 0) + 1
            project_quota._proj_alpha = alpha_eff
            bt = q
    elif mode == "full":
        bt = rebalance(tgt.round().long().clamp(0, n), tgt, Btot, n)
    else:
        b0m = b0.reshape(L, H)
        dm = delta.reshape(L, H)
        bt = torch.zeros(L, H, dtype=torch.long, device=b0.device)
        if mode == "within":
            dw = dm - dm.mean(1, keepdim=True)               # 显式去均值
            for l in range(L):
                base_l = int(b0m[l].sum().item())
                t_l = (b0m[l] + dw[l]).clamp(0, n)
                bt[l] = rebalance(t_l.round().long().clamp(0, n), t_l, base_l, n)
        else:                                                # across：两级整数投影
            d_l = b0m.sum(1) + dm.sum(1)
            Bl = rebalance(d_l.round().long().clamp(0, H * n), d_l, Btot, H * n)
            for l in range(L):
                base_l = float(b0m[l].sum())
                tot_l = int(Bl[l].item())
                t_l = (b0m[l] * (tot_l / base_l) if base_l > 0
                       else torch.full((H,), tot_l / H, device=b0.device,
                                       dtype=b0.dtype)).clamp(0, n)
                bt[l] = rebalance(t_l.round().long().clamp(0, n), t_l, tot_l, n)
        bt = bt.reshape(-1)
        # 构造性守恒：within 逐层锁基线、across 两级各自严格配平 ⇒ 总和必然相等。
        assert int(bt.sum().item()) == Btot, \
            f"{mode} 模式预算不守恒：{int(bt.sum().item())} != {Btot}（投影有 bug）"

    assert int(bt.sum().item()) == Btot, \
        f"配额投影后预算 {int(bt.sum().item())} != 基线 {Btot}"
    assert int(bt.min().item()) >= 0 and int(bt.max().item()) <= n
    return bt
