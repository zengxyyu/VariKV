"""ProMeta 的训练目标：**集合预测**，不是逐 token 回归。

Teacher 给 `U* : [M,L,H,N]`（M 个真实未来对前缀 KV 的需求分布），
Student 给 `Û : [M,L,H,N]`。**两边的未来没有天然编号对应** —— teacher 的
第 1 个 question 与 student 的第 1 个 probe 没有理由是同一个未来模式。
所以损失必须对未来轴的**置换不变**：

    L_demand = min_{π∈S_M} (1/M) Σ_m  KL( U*_m ‖ Û_{π(m)} )

**为什么用前向 KL** `KL(U*‖Û)` 而不是反向：前向是 **mode-covering**（覆盖模式）
—— 只要某个真实未来在某处有质量而 Student 给 0，惩罚就发散。这正是
ProMeta 想要的：**宁可多覆盖一个可能未来，也不要漏掉它**。反向 KL 是
mode-seeking，会鼓励 Student 只押一个未来 —— 与方法主张相反。

    L = L_demand + λ_div · mean_{m≠n} cos²(q̂_m, q̂_n)

`λ_div` 只防坍缩，**不强制正交**（未来之间本来可以相关）。

⚠ **M≤8 时直接枚举全部 M! 个置换**（M=5 只有 120 个），不引 scipy 依赖，
也避免「匈牙利实现有 bug 但看不出来」。M>8 才需要 `linear_sum_assignment`。
"""
import itertools

import torch
import torch.nn.functional as F

EPS = 1e-12


def pairwise_kl(Us, Uh):
    """C[m,n] = mean_{l,h} KL(U*_m[l,h] ‖ Û_n[l,h])。两者最后一维都是概率分布。"""
    assert Us.shape == Uh.shape, (Us.shape, Uh.shape)
    M = Us.shape[0]
    logs = torch.log(Us.clamp_min(EPS))
    logh = torch.log(Uh.clamp_min(EPS))
    # [M,1,...] vs [1,M,...] → [M,M,L,H]
    kl = (Us[:, None] * (logs[:, None] - logh[None])).sum(-1)
    return kl.reshape(M, M, -1).mean(-1)


def match_loss(Us, Uh, return_perm=False):
    """置换不变的需求损失。M≤8 走全枚举。"""
    M = Us.shape[0]
    assert M <= 8, f"M={M} 太大，全枚举 {M}! 不现实；改用 scipy 匈牙利"
    C = pairwise_kl(Us, Uh)                       # [M,M]
    best, bestp = None, None
    for p in itertools.permutations(range(M)):
        v = C[torch.arange(M), torch.tensor(p)].mean()
        if best is None or v.item() < best.item():
            best, bestp = v, p
    return (best, bestp) if return_perm else best


def total_loss(Us, Uh, q, lam_div=0.1):
    from prometa.model import diversity_loss
    ld = match_loss(Us, Uh)
    dv = diversity_loss(q)
    return ld + lam_div * dv, {"demand": float(ld), "div": float(dv)}


def _selftest():
    torch.manual_seed(0)
    M, L, H, N = 5, 3, 2, 17
    Us = torch.softmax(torch.randn(M, L, H, N), -1)

    # ① 完全相同 ⇒ 损失 0，且最优置换是恒等
    v, p = match_loss(Us, Us.clone(), return_perm=True)
    assert v.item() < 1e-8, v.item()
    assert p == tuple(range(M)), p
    print(f"① 自匹配 loss={v.item():.2e}、置换={p}　PASS")

    # ② **置换不变**：打乱 Student 的未来轴，损失必须不变、置换必须跟着反过来
    perm = [3, 0, 4, 1, 2]
    v2, p2 = match_loss(Us, Us[perm].clone(), return_perm=True)
    assert v2.item() < 1e-8, v2.item()
    inv = [perm.index(i) for i in range(M)]
    assert list(p2) == inv, (p2, inv)
    print(f"② 置换不变：loss={v2.item():.2e}、恢复出的置换={p2}（期望 {tuple(inv)}）　PASS")

    # ③ KL ≥ 0，且不同分布严格 > 0
    Uh = torch.softmax(torch.randn(M, L, H, N), -1)
    C = pairwise_kl(Us, Uh)
    assert (C >= -1e-9).all() and C.min() > 1e-3, (C.min().item(),)
    print(f"③ KL≥0 且异分布严格为正（min {C.min().item():.4f}）　PASS")

    # ④ **前向 KL 是 mode-covering 的阴性对照**：Student 在 teacher 有质量处置 0
    #    应当被重罚，而 teacher 为 0 处 Student 放质量只受轻罚。
    a = torch.zeros(1, 1, 1, 4); a[..., 0] = 1.0
    b_drop = torch.tensor([[[[1e-9, 1 - 3e-9, 1e-9, 1e-9]]]])   # 漏掉 teacher 的模式
    b_extra = torch.tensor([[[[0.5, 0.5, 1e-9, 1e-9]]]])        # 多押一个
    kd = pairwise_kl(a, b_drop)[0, 0].item()
    ke = pairwise_kl(a, b_extra)[0, 0].item()
    assert kd > 10 * ke, (kd, ke)
    print(f"④ mode-covering：漏掉模式罚 {kd:.2f} ≫ 多押一个罚 {ke:.2f}　PASS")

    # ⑤ 梯度能回到 Student
    Uh = torch.softmax(torch.randn(M, L, H, N, requires_grad=True), -1)
    x = torch.randn(M, L, H, N, requires_grad=True)
    Uh = torch.softmax(x, -1)
    match_loss(Us, Uh).backward()
    assert x.grad is not None and x.grad.abs().max() > 0, "梯度没回到 Student"
    print(f"⑤ 梯度回到 Student（|g|max={x.grad.abs().max():.3e}）　PASS")

    # ⑥ total_loss 的两项都非零且可分
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prometa.model import ProMetaPredictor
    net = ProMetaPredictor(64, 8, L, H, n_future=M, d_proj=16, n_pool=2, d_lat=8)
    q = net(torch.randn(N, 64))
    keys = [torch.randn(1, H, N, 8) for _ in range(L)]
    Uh = ProMetaPredictor.demand(q, keys)
    tl, parts = total_loss(Us, Uh, q, lam_div=0.1)
    assert parts["demand"] > 0 and parts["div"] > 0
    tl.backward()
    dead = [n for n, p in net.named_parameters()
            if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, f"这些参数没梯度：{dead}"
    print(f"⑥ total_loss demand={parts['demand']:.4f} div={parts['div']:.4f}、"
          f"{len(list(net.parameters()))} 个张量全有梯度　PASS")
    print("\nprometa/train.py 自测 6 条全过")


if __name__ == "__main__":
    _selftest()
