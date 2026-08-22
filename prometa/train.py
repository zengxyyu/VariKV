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

⚠ **Ms≤8 时直接枚举全部单射**（Ms=Mt=5 只有 120 个），不引 scipy 依赖，
也避免「匈牙利实现有 bug 但看不出来」。Ms>8 才需要 `linear_sum_assignment`。
允许 `Ms > Mt`（Student 的 probe 比这条样本的真实未来多），见 `match_loss`。
"""
import itertools

import torch
import torch.nn.functional as F

EPS = 1e-12


def to_dist(U):
    """把教师效用 `U` 逐 (m,l,h) 归一化成概率分布。

    **必需，不是美化。** `prometa/teacher.py:future_utility` 给的是
    `max_{t,组内头} softmax_i(...)`，**行和不是 1**（真机实测行和 1.8–37.9）。
    `pairwise_kl` 的 `Σ U*·(log U* − log Û)` 只有在 `U*` 是分布时才是 KL。

    ⚠ 这一步**不改变同一 (m,l,h) 内的位置排序**（除以一个正常数），
    所以决策内容不变；改变的只是各未来在损失里的权重。真机上未来间总质量
    的 max/min 只有 **1.13×（Retr.KV 中位）/ 1.56×（En.QA 中位）**，
    所以归一化在本数据上是小改动 —— 但它是让目标函数**有定义**的那一步。
    """
    U = U if torch.is_tensor(U) else torch.as_tensor(U)
    return U / U.sum(-1, keepdim=True).clamp_min(EPS)


def pairwise_kl(Us, Uh):
    """C[m,n] = mean_{l,h} KL(U*_m[l,h] ‖ Û_n[l,h])。→ [Mt, Ms]。

    **两者最后一维都必须是概率分布**（教师侧走 `to_dist`）。这里显式断言，
    因为「传进来的其实不是分布」正是那种 loss 照降、量却不是 KL 的静默失败。
    """
    assert Us.shape[1:] == Uh.shape[1:], (Us.shape, Uh.shape)
    for nm, X in (("teacher", Us), ("student", Uh)):
        d = (X.sum(-1) - 1.0).abs().max().item()
        assert d < 1e-3, f"{nm} 最后一维不是概率分布（max|Σ−1| = {d:.3e}）；" \
                         f"教师侧请先过 `to_dist`"
    logs = torch.log(Us.clamp_min(EPS))
    logh = torch.log(Uh.clamp_min(EPS))
    # **不要写成 `(Us[:,None] * (logs[:,None] - logh[None])).sum(-1)`。**
    # 那会物化两个 [Mt,Ms,L,H,N] 的临时量：真机 M=5,L=28,H=4,N=16000 时
    # 每个 179 MB，前向后向合计约 1 GB，纯属浪费。
    # 拆成「熵项（无配对）− 交叉项（einsum 直接缩并）」，输出只有 [Mt,Ms,L,H]。
    ent = (Us * logs).sum(-1)                          # [Mt,L,H]
    cross = torch.einsum("mlhn,klhn->mklh", Us, logh)  # [Mt,Ms,L,H]
    kl = ent[:, None] - cross
    return kl.flatten(2).mean(-1)


def match_loss(Us, Uh, return_perm=False):
    """置换不变的需求损失。允许 **Student 的 probe 数 ≥ teacher 的未来数**。

        L = min_{单射 π: [Mt] → [Ms]}  (1/Mt) Σ_m KL(U*_m ‖ Û_{π(m)})

    `Ms > Mt` 时用**单射**而不是双射（DETR 式集合预测）：Student 可以携带比
    任何一条训练样本的未来数更多的 probe，多出来的那些本步不吃需求梯度，
    只由 `diversity_loss` 约束。这解除了「Student 的 M 必须等于教师任务的 M」
    这个本来毫无必要的耦合。

    ⚠ 枚举量是 `Ms!/(Ms−Mt)!`；`Ms≤8` 时最多 40320，可以接受。
    """
    Mt, Ms = Us.shape[0], Uh.shape[0]
    assert Ms >= Mt, f"Student 的 probe 数 {Ms} < teacher 的未来数 {Mt}"
    assert Ms <= 8, f"Ms={Ms} 太大，全枚举不现实；改用 scipy 匈牙利"
    assert Us.shape[1:] == Uh.shape[1:], (Us.shape, Uh.shape)
    C = pairwise_kl(Us, Uh)                       # [Mt, Ms]
    rows = torch.arange(Mt)
    best, bestp = None, None
    for p in itertools.permutations(range(Ms), Mt):
        v = C[rows, torch.tensor(p)].mean()
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
    b_extra = torch.tensor([[[[0.5, 0.5 - 2e-9, 1e-9, 1e-9]]]])  # 多押一个
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

    # ⑦ **Ms > Mt（单射匹配）**：多出来的 probe 不该改变最优损失
    Us4 = Us[:4]
    v_eq, p_eq = match_loss(Us4, Us4.clone(), return_perm=True)
    pad = torch.softmax(torch.randn(3, L, H, N), -1)
    Uh7 = torch.cat([pad[:2], Us4, pad[2:]], 0)                 # Ms=7 含全部 4 个真值
    v_in, p_in = match_loss(Us4, Uh7, return_perm=True)
    assert v_in.item() < 1e-8 and list(p_in) == [2, 3, 4, 5], (v_in.item(), p_in)
    print(f"⑦ Ms=7 > Mt=4 单射匹配：loss={v_in.item():.2e}、选中 {p_in}（期望 (2,3,4,5)）　PASS")

    # ⑧ `to_dist` 与非分布输入的守卫
    raw = torch.rand(3, 2, 2, 9) * 7 + 0.1
    D = to_dist(raw)
    assert (D.sum(-1) - 1).abs().max() < 1e-6
    rk = lambda x: torch.argsort(torch.argsort(-x, -1), -1)
    assert torch.equal(rk(raw), rk(D)), "to_dist 必须保序"
    try:
        pairwise_kl(raw, torch.softmax(torch.randn(3, 2, 2, 9), -1))
        raise SystemExit("守卫没生效：非分布输入被接受了")
    except AssertionError as ex:
        assert "概率分布" in str(ex), ex
    print("⑧ to_dist 归一且保序；非分布输入被断言拦住　PASS")

    # ⑨ **省显存的 einsum 写法与朴素写法逐位对拍**（同一个量两条实现）
    P = torch.softmax(torch.randn(4, 2, 3, 23), -1)
    Q = torch.softmax(torch.randn(6, 2, 3, 23), -1)
    lp, lq = torch.log(P.clamp_min(EPS)), torch.log(Q.clamp_min(EPS))
    naive = (P[:, None] * (lp[:, None] - lq[None])).sum(-1).flatten(2).mean(-1)
    e9 = float((naive - pairwise_kl(P, Q)).abs().max())
    assert e9 < 1e-5, e9
    print(f"⑨ pairwise_kl einsum 写法与朴素写法对拍 max|差| = {e9:.2e}　PASS")
    print("\nprometa/train.py 自测 9 条全过")


if __name__ == "__main__":
    _selftest()
