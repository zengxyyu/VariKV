"""ProMeta 的 Student：**只看前缀**，预测「未来记忆需求的分布」。

    x (prefix hidden states)  ──►  q̂_1 … q̂_M      （M 个前瞻潜在查询）
    q̂_m · k_i / √d  ──softmax over prefix──►  Û_{m,i}
    ρ_β(Û_{·,i})  ──►  R_i  ──►  TopB

**它不预测一个 token 分数。** 若写成 `û_i = f(k_i)`，方法就退化成又一个
learned future-importance predictor（LookaheadKV / TRIM-KV 已占据）。
ProMeta 的技术边界是**预测一个分布、再做风险敏感的不可逆决策**。

**三条与 RestoreKV 划清界限的构造性事实**：
① 这 M 个 probe **不是 append 到 LLM 序列里的 token**，不走 LLM forward、
   不经 LoRA，只是一个外置预测头的输出；
② 它们**不进入最终 cache**，不占任何预算 —— 最终 `C' ⊂ C_original`；
③ 因此 ProMeta 用**算力**改善选择，RestoreKV 用**一部分预算**合成补偿记忆。

**池化不用 mean。** 本项目已反复测到稀疏证据会被 mean pooling 稀释
（L1 那 4 个头只占全局抬升的 0.65% 却产出 74% 的收益），所以用少量
learned summary queries 做注意力池化。这 K 个 summary query 同样
**不进入 cache**，是预测器内部潜变量。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProMetaPredictor(nn.Module):
    """→ `q̂ : [M, L, Hkv, d]`，与 `key_cache[l][0,h]` 逐头点积即得 Û。

    参数量刻意压小：先把 hidden 投到 `d_proj`，再池化，再出 M 个低维隐变量
    `u_m`，最后由**共享**的 `A` 升到 head_dim 并加上**逐 (层,头) 的嵌入**。
    这样跨层头共享结构、只用嵌入区分身份 —— 与本项目 `chead` 的思路一致
    （逐头一个数就够，不需要逐头一整套权重）。
    """

    def __init__(self, hidden_dim, head_dim, n_layers, n_kv_heads,
                 n_future=5, d_proj=128, n_pool=4, d_lat=64):
        super().__init__()
        self.M, self.L, self.H, self.d = n_future, n_layers, n_kv_heads, head_dim
        self.proj = nn.Linear(hidden_dim, d_proj, bias=False)
        self.pool_q = nn.Parameter(torch.randn(n_pool, d_proj) * 0.02)
        self.trunk = nn.Sequential(
            nn.Linear(n_pool * d_proj, 4 * d_lat), nn.SiLU(),
            nn.Linear(4 * d_lat, n_future * d_lat),
        )
        self.A = nn.Linear(d_lat, head_dim, bias=False)      # 跨 (层,头) 共享
        self.head_emb = nn.Parameter(torch.zeros(n_layers, n_kv_heads, head_dim))
        # **probe 之间必须一开始就不同**，否则 diversity loss 要从完全对称的
        # 鞍点上把它们推开，实践中很容易全程坍缩（本项目在 8 个 writer 模块
        # 被填零后 GRU 退化那次已经吃过对称初始化的亏）。
        self.probe_bias = nn.Parameter(torch.randn(n_future, d_lat) * 0.5)
        nn.init.normal_(self.head_emb, std=0.02)
        self.d_lat = d_lat

    def latents(self, hidden):
        """hidden: [N, D] → u: [M, d_lat]。**只吃前缀，未来在此不可见。**"""
        assert hidden.dim() == 2, hidden.shape
        h = self.proj(hidden)                                  # [N,dp]
        a = torch.softmax(self.pool_q @ h.T / h.shape[-1] ** 0.5, dim=-1)  # [K,N]
        z = (a @ h).flatten()                                  # [K*dp]
        u = self.trunk(z).view(self.M, self.d_lat)
        return u + self.probe_bias

    def forward(self, hidden):
        """→ q̂: [M, L, Hkv, d]，已 L2 归一化（点积尺度由 1/√d 统一给）。"""
        u = self.latents(hidden)                               # [M,dl]
        base = self.A(u)                                       # [M,d]
        q = base[:, None, None, :] + self.head_emb[None]       # [M,L,H,d]
        return F.normalize(q, dim=-1)

    @staticmethod
    def demand(q, keys, n_prefix=None):
        """Û: [M, L, Hkv, N]。`keys[l]`: [1,Hkv,N,d]（post-RoPE，取自 cache）。

        softmax **只在前缀位置上归一化** —— 与 teacher 的口径逐字一致
        （`prometa/teacher.py`）。这定义的是「候选前缀记忆之间的归一化前瞻
        偏好」，**不是模型真实 forward 里的 attention 概率**，措辞不能混。
        """
        M, L, H, d = q.shape
        out = []
        for l in range(L):
            K = keys[l][0]                                     # [H,N,d]
            if n_prefix is not None:
                K = K[:, :n_prefix, :]
            out.append(torch.softmax(
                torch.einsum("mhd,hnd->mhn", q[:, l], K.to(q.dtype)) / d ** 0.5,
                dim=-1))
        return torch.stack(out, 1)                             # [M,L,H,N]


def diversity_loss(q):
    """防 probe 坍缩：`mean_{m≠n} cos²(q̂_m, q̂_n)`（q 已归一化）。

    **不强制正交** —— 未来之间本来就可能相关；只惩罚「全部相同」。
    """
    v = F.normalize(q.flatten(1), dim=-1)                      # [M, L*H*d]
    s = v @ v.T
    M = v.shape[0]
    off = ~torch.eye(M, dtype=torch.bool, device=v.device)
    return s[off].pow(2).mean()


def _selftest():
    torch.manual_seed(0)
    D, d, L, H, M, N = 256, 32, 3, 2, 5, 40
    net = ProMetaPredictor(D, d, L, H, n_future=M, d_proj=16, n_pool=2, d_lat=8)
    hid = torch.randn(N, D)
    q = net(hid)
    assert q.shape == (M, L, H, d), q.shape
    assert torch.allclose(q.norm(dim=-1), torch.ones(M, L, H), atol=1e-5)
    print(f"① 形状 {tuple(q.shape)}、已归一化　PASS　参数 {sum(p.numel() for p in net.parameters()):,}")

    keys = [torch.randn(1, H, N, d) for _ in range(L)]
    U = ProMetaPredictor.demand(q, keys)
    assert U.shape == (M, L, H, N), U.shape
    assert torch.allclose(U.sum(-1), torch.ones(M, L, H), atol=1e-5), "softmax 必须归一"
    print(f"② demand 形状 {tuple(U.shape)}、逐行和为 1　PASS")

    # ③ 手算对拍一格
    m, l, h = 2, 1, 0
    ref = torch.softmax(q[m, l, h] @ keys[l][0, h].T / d ** 0.5, -1)
    assert (ref - U[m, l, h]).abs().max() < 1e-6, (ref - U[m, l, h]).abs().max()
    print(f"③ 手算对拍 max|差| = {(ref - U[m,l,h]).abs().max():.2e}　PASS")

    # ④ **未来不可见**：改前缀 hidden 必须改 q̂；而 q̂ 不依赖 keys（只在 demand 里用）
    q2 = net(torch.randn(N, D))
    assert (q - q2).abs().max() > 1e-4, "q̂ 必须依赖前缀 hidden"
    print("④ q̂ 依赖前缀 hidden、且与 keys 解耦　PASS")

    # ⑤ probe 初始就不同（不是从对称鞍点出发）
    dv = diversity_loss(q).item()
    assert dv < 0.99, f"初始化就坍缩了：cos²={dv:.4f}"
    print(f"⑤ 初始多样性 cos² = {dv:.4f}（<0.99）　PASS")

    # ⑥ 阴性对照：把 probe_bias 清零 + trunk 输出常数 ⇒ 必须坍缩，diversity_loss≈1
    with torch.no_grad():
        net.probe_bias.zero_()
        for p in net.trunk[-1].parameters():
            p.zero_()
    qc = net(hid)
    dvc = diversity_loss(qc).item()
    assert dvc > 0.99, f"阴性对照没坍缩：cos²={dvc:.4f}"
    print(f"⑥ 阴性对照（强制对称）cos² = {dvc:.4f}（>0.99）　PASS")

    # ⑦ 梯度确实到达全部可训练张量（本项目吃过「loss 在降但 grad 恒零」的亏）
    net2 = ProMetaPredictor(D, d, L, H, n_future=M, d_proj=16, n_pool=2, d_lat=8)
    out = net2(hid)
    (out.sum() + diversity_loss(out)).backward()
    dead = [n for n, p in net2.named_parameters()
            if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, f"这些参数没拿到梯度：{dead}"
    print(f"⑦ {len(list(net2.parameters()))} 个参数张量全部拿到非零梯度　PASS")
    print("\nprometa/model.py 自测 7 条全过")


if __name__ == "__main__":
    _selftest()
