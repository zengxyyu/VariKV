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


def reachable_project(tgt, sc, b0, Btot, alpha=1.0, n_tau=1024):
    """把目标配额 `tgt` 投影到**当前参数化真正能表示的**配额集合上。

    为什么需要这个函数（这是本项目最容易被误读的一步）：
    `scratch_probe_reach.py` 证明了地板目标配额在 74/74 个 chunk 上都**不可表示**
    —— 但那只说明「**那一个**配额取不到」，**不等于**「地板拿到的 +33.60 取不到」。
    完全可能存在另一个**可达**配额，效用与地板相当。要把「表示能力不足」从
    「关于某个点」升级成「关于收益」，必须把地板目标投影回可达集，再真跑一遍。
    这个函数就是那个投影；下游分数由评测给出。

    可达集的刻画（与 slack 判据同一套代数，`scratch_test_reach.py` 有 11 个单测）：
    修正满足 `|Δs_{h,i}| ≤ a_h`（`a_h = α·σ_h`）时，存在公共阈值 `τ′` 使头 `h`
    恰留 `q_h` 个，当且仅当

        q_min_h(τ′) ≤ q_h ≤ q_max_h(τ′),
        q_max_h(τ) = #{i : s_{h,i} > τ − a_h},   q_min_h(τ) = #{i : s_{h,i} > τ + a_h}

    于是可达集 = ∪_{τ} { q : q_min(τ) ≤ q ≤ q_max(τ), Σq = B }。对每个候选 `τ` 把
    `tgt` clamp 进箱子再配平到 `B`，取 L1 距离最小的那个 τ。**注意这给的是投影，
    不是「最优可达配额」** —— 后者需要效用模型，而效用正是我们要测的东西。
    """
    G = tgt.numel()
    ss, _ = torch.sort(sc.reshape(G, -1), dim=-1)            # 升序，便于 searchsorted
    n_tok = ss.shape[1]
    a = (alpha * sc.reshape(G, -1).std(dim=-1)).clamp(min=0)
    lo = float((ss[:, 0] - a).min()); hi = float((ss[:, -1] + a).max())
    taus = torch.linspace(lo, hi, n_tau, device=ss.device, dtype=ss.dtype)

    # **两段式，纯粹是为了速度**：τ 扫描全向量化算出每个 τ 的箱子与一个
    # L1 下界（只 clamp、不配平），再只对下界最小的少数几个 τ 做代价高的整数配平。
    # 下界成立是因为配平只会把 q 推离 tgt，不会拉近 ⇒ 真 L1 ≥ clamp 后的 L1。
    X = taus[:, None]                                        # [T,1]
    qmax = n_tok - torch.searchsorted(ss, (X - a[None, :]).T.contiguous(),
                                      right=True).T          # [T,G]
    qmin = n_tok - torch.searchsorted(ss, (X + a[None, :]).T.contiguous(),
                                      right=True).T
    feas = (qmin.sum(1) <= Btot) & (qmax.sum(1) >= Btot)
    if not bool(feas.any()):
        return None, None
    qc = torch.minimum(torch.maximum(tgt[None, :], qmin), qmax)
    lb = (qc - tgt[None, :]).abs().sum(1).float()
    lb = torch.where(feas, lb, torch.full_like(lb, float("inf")))
    cand = torch.argsort(lb)[:8].tolist()

    best, best_d = None, None
    for ti in cand:
        if not bool(feas[ti]):
            continue
        q = qc[ti].clone()
        d = Btot - int(q.sum())
        if d != 0:
            room = (qmax[ti] - q) if d > 0 else (q - qmin[ti])
            # 优先动「clamp 把它推离 tgt 最多」的那些头，使配平尽量不再加大 L1
            order = torch.argsort((tgt - q).float(), descending=(d > 0)).tolist()
            for g in order:
                if d == 0:
                    break
                t = min(abs(d), int(room[g]))
                q[g] += t if d > 0 else -t
                d -= t if d > 0 else -t
        if d != 0:
            continue
        dist = float((q - tgt).abs().sum())
        if best_d is None or dist < best_d:
            best_d, best = dist, q.clone()
    return best, best_d


def project_quota(b0, delta, n, mode, n_layers, n_heads, sc=None):
    """b0, delta: [L*H] float tensor；返回 [L*H] long tensor。

    `mode` ∈ {full, within, across}。非 full 模式下预算守恒是**构造性**的，
    函数末尾直接断言而不做兜底修补 —— 若将来投影出 bug，必须让它崩，而不是被
    一个通用修补循环悄悄"修好"同时破坏因果不变量（那正是本项目栽过的坑）。
    """
    assert mode in ("full", "within", "across", "floor", "floorproj"), mode
    L, H = int(n_layers), int(n_heads)
    assert b0.numel() == L * H and delta.numel() == L * H
    Btot = int(b0.sum().item())
    tgt = (b0 + delta).clamp(0, n)

    if mode in ("floor", "floorproj"):
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
        excess = float(t.sum() - Btot)                      # ≥ 0
        if excess > 0:
            room = (b0 - bmin).clamp(min=0)                 # 可扣回的量
            tot_room = float(room.sum())
            assert tot_room >= excess, "富余不足以填地板"
            t = t - room * (excess / tot_room)
        t = t.clamp(0, n)
        bt = rebalance(t.round().long().clamp(0, n), t, Btot, n)
        if mode == "floorproj":
            # 把地板目标投影回**当前参数化可达**的配额集。若某个 chunk 上地板本来就
            # 可达，投影是恒等的（单测里验过）；不可行时退回地板本身并计数，
            # 因为静默退回会把「投影没起作用」伪装成「投影没有代价」。
            assert sc is not None, "floorproj 需要 score 张量"
            al = float(os.environ.get("VARIKV_PROJ_ALPHA", "1.0"))
            q, d = reachable_project(bt, sc, b0, Btot, alpha=al)
            if q is None:
                project_quota._proj_fail = getattr(project_quota, "_proj_fail", 0) + 1
            else:
                project_quota._proj_l1 = getattr(project_quota, "_proj_l1", 0.0) + d
                project_quota._proj_n = getattr(project_quota, "_proj_n", 0) + 1
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
