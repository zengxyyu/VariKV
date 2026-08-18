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


def project_quota(b0, delta, n, mode, n_layers, n_heads):
    """b0, delta: [L*H] float tensor；返回 [L*H] long tensor。

    `mode` ∈ {full, within, across}。非 full 模式下预算守恒是**构造性**的，
    函数末尾直接断言而不做兜底修补 —— 若将来投影出 bug，必须让它崩，而不是被
    一个通用修补循环悄悄"修好"同时破坏因果不变量（那正是本项目栽过的坑）。
    """
    assert mode in ("full", "within", "across"), mode
    L, H = int(n_layers), int(n_heads)
    assert b0.numel() == L * H and delta.numel() == L * H
    Btot = int(b0.sum().item())
    tgt = (b0 + delta).clamp(0, n)

    if mode == "full":
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
