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
                 n_future=5, d_proj=128, n_pool=4, d_lat=64, n_atoms=8):
        super().__init__()
        self.M, self.L, self.H, self.d = n_future, n_layers, n_kv_heads, head_dim
        self.R = n_atoms
        self.proj = nn.Linear(hidden_dim, d_proj, bias=False)
        self.pool_q = nn.Parameter(torch.randn(n_pool, d_proj) * 0.02)
        self.trunk = nn.Sequential(
            nn.Linear(n_pool * d_proj, 4 * d_lat), nn.SiLU(),
            nn.Linear(4 * d_lat, n_future * d_lat),
        )
        self.A = nn.Linear(d_lat, head_dim, bias=False)      # 跨 (层,头,atom) 共享
        # ── 多 atom（2026-08-22，`scratch_prometa_arch.py` 定的） ────────────────
        # 教师是 `normalize(max_{t,g} softmax(q_{t,g}·K))`，**逐位置取 max 的包络**，
        # 其 log 是 k 的**凸分段线性**函数；单个 q̂ 只能给仿射函数 ⇒ 旧架构恰是
        # 教师族的 R=1 特例。实测（真实 `level=pair` 全局掩码 J@0.1，2 篇 val ×
        # 5 未来 × 112 个 (层,头)）：R=1 0.6414 → R=8 0.7204 → R=16 0.7360。
        # R=16 只比 R=8 多 +0.0156 却参数翻倍 ⇒ **取 R=8**。
        # atom 偏移必须**逐 (层,头)**：让它逐未来变而跨头共享（= 把 trunk 输出
        # 扩宽成 M×R×d_lat，多约 65 万参数）实测**一分不涨**（off_m 0.5041 vs
        # off_g 0.5039 vs R=1 0.5039）—— 因为教师的 atom 本来就是各 kv 组自己那
        # G 个 query 头的真实向量。
        self.head_emb = nn.Parameter(torch.zeros(n_atoms, n_layers, n_kv_heads, head_dim))
        # ── 可学温度（同一轮实验里代价最大的那个瓶颈） ──────────────────────
        # 旧代码无条件 `F.normalize(q)` ⇒ ‖q̂‖≡1 ⇒ logit 幅度被 ‖k‖/√d 钉死，
        # 而教师的 max 包络相当尖（行和 S 中位 20.17）。实测这一条**单独**值
        # 全局掩码 J +0.236（0.4055 → 0.6414），比加 atom 大 3 倍，也比**完全
        # 去掉归一化**（0.6352）略好。所以不是删约束，是把「方向」与「温度」
        # 拆开显式建模：`q = τ_{l,h} · v/‖v‖`。112 个参数。
        # **probe 之间必须一开始就不同**，否则 diversity loss 要从完全对称的
        # 鞍点上把它们推开，实践中很容易全程坍缩（本项目在 8 个 writer 模块
        # 被填零后 GRU 退化那次已经吃过对称初始化的亏）。
        self.probe_bias = nn.Parameter(torch.randn(n_future, d_lat) * 0.5)
        self.log_tau = nn.Parameter(torch.zeros(n_layers, n_kv_heads))   # τ=exp(·)，初始 1
        # ⚠ **atom 初始必须互不相同，且相对上下文项的权重必须是可控的。**
        # 首版沿用 `head_emb` 的 std=0.02，而 `A(u_m)` 的模远大于它 ⇒ 8 个 atom
        # 初始 cos≈0.996（自测⑦实测多样性 0.2463）。硬 `max_r` 只给 argmax 那个
        # atom 梯度 ⇒ 一个赢遍全场、其余 7 个**永远拿不到梯度**，R=8 静默退化成
        # R=1，而 loss 曲线看不出来（= 当年 8 个被填零的 writer 那一类错）。
        # 改成**两项各自单位化、相对权重 `atom_w` 可学**：随机 128 维下
        # cos(e_r,e_s)≈0、cos(c,e)≈0 ⇒ cos(v_r,v_s) = (1+w²·0)/(1+w²) = 1/2，
        # 恰好落在 hinge 的 margin 上，**与数据尺度无关**。
        # 代价：丢掉 ‖A(u_m)‖ 这个逐未来的幅度自由度（全局尖锐度仍由 τ 承担）。
        self.atom_w = nn.Parameter(torch.ones(n_atoms))
        nn.init.normal_(self.head_emb, std=1.0)
        self.d_lat = d_lat

    def pool(self, hidden):
        """hidden: [N, D] → z: [K, dp]。**离线（一次看全部位置）版本。**

        推理时分块预填走的是 `prometa/pool.py` 的在线版；两者**必须等价**，
        `prometa/model.py` 与 `prometa/pool.py` 的自测各自对拍过一次
        （同一个量两条实现必须对拍 —— 本项目铁律）。
        """
        assert hidden.dim() == 2, hidden.shape
        h = self.proj(hidden)                                  # [N,dp]
        a = torch.softmax(self.pool_q @ h.T / h.shape[-1] ** 0.5, dim=-1)  # [K,N]
        return a @ h                                           # [K,dp]

    def from_pooled(self, z):
        """z: [K, dp]（在线或离线池化的输出）→ q̂: **[M, R, L, Hkv, d]**。

        **这是部署路径的唯一入口** —— 分块预填时上下文摘要由
        `OnlineAttnPool` 增量维护，拿不到完整 hidden。

            v_{m,r,l,h} = Â(u_m) + w_r · Ê_{r,l,h}      （两项各自单位化）
            q̂_{m,r,l,h} = τ_{l,h} · v/‖v‖               （方向与温度分开）
        """
        u = self.trunk(z.flatten().to(self.probe_bias.dtype)).view(self.M, self.d_lat)
        u = u + self.probe_bias
        c = F.normalize(self.A(u), dim=-1)[:, None, None, None, :]    # [M,1,1,1,d]
        e = F.normalize(self.head_emb, dim=-1)[None]                  # [1,R,L,H,d]
        v = c + e * self.atom_w.abs()[None, :, None, None, None]      # [M,R,L,H,d]
        tau = self.log_tau.exp()[None, None, :, :, None]              # [1,1,L,H,1]
        return F.normalize(v, dim=-1) * tau

    def latents(self, hidden):
        """hidden: [N, D] → u: [M, d_lat]（离线路径，供训练与对拍用）。"""
        z = self.pool(hidden)
        return self.trunk(z.flatten()).view(self.M, self.d_lat) + self.probe_bias

    def forward(self, hidden):
        """→ q̂: [M, L, Hkv, d]。离线路径 = `from_pooled(pool(hidden))`。"""
        return self.from_pooled(self.pool(hidden))

    @staticmethod
    def demand_layer(q_l, K, ret_usage=False):
        """**唯一真源**：一层的需求分布。`q_l`: [M,R,H,d]，`K`: [H,n,d] → [M,H,n]。

            Û = normalize_i( max_r softmax_i( q_{m,r,h}·k_i / √d ) )

        与教师 `prometa/teacher.py:future_utility` **同族、同口径**：
        逐元素取 max、之后重新归一化，softmax 只在候选区间 `[lo,hi)` 上做。
        三个调用点（`demand` / `cache.prometa_scores` / 训练侧 `student_demand`）
        **必须都走这里** —— 本仓库栽过「同一个量两份实现」的第④类错。
        """
        d = q_l.shape[-1]
        p = torch.softmax(
            torch.einsum("mrhd,hnd->mrhn", q_l, K.to(q_l.dtype)) / d ** 0.5, dim=-1)
        u = p.amax(1)                                          # [M,H,n] 逐元素 max
        u = u / u.sum(-1, keepdim=True).clamp_min(1e-30)
        if ret_usage:
            # 硬 max 下**没被选中的 atom 梯度恒为 0**，会像当年被填零的 8 个 writer
            # 一样静默死掉。所以要能查「每个 atom 在多少位置上当过 max」。
            return u, torch.nn.functional.one_hot(
                p.argmax(1), q_l.shape[1]).float().mean((0, 1, 2))
        return u

    @staticmethod
    def demand(q, keys, n_prefix=None, ret_usage=False):
        """Û: [M, L, Hkv, N]。`q`: [M,R,L,H,d]；`keys[l]`: [1,Hkv,N,d]（post-RoPE）。"""
        L = q.shape[2]
        out, use = [], []
        for l in range(L):
            K = keys[l][0]                                     # [H,N,d]
            if n_prefix is not None:
                K = K[:, :n_prefix, :]
            r = ProMetaPredictor.demand_layer(q[:, :, l], K, ret_usage)
            if ret_usage:
                out.append(r[0]); use.append(r[1])
            else:
                out.append(r)
        U = torch.stack(out, 1)                                # [M,L,H,N]
        return (U, torch.stack(use, 0).mean(0)) if ret_usage else U


def atom_diversity_loss(q, margin=0.5):
    """**防 atom 坍缩**：`mean_{r≠s} max(cos(q̂_{m,r,l,h}, q̂_{m,s,l,h}) − margin, 0)²`。

    与 `diversity_loss`（防**未来**坍缩）是两件事，必须分开。硬 `max_r` 只把
    梯度给到 argmax 的那一个 atom，所以一个从没赢过的 atom **永远拿不到梯度**，
    R=8 会静默退化成 R=1，而 loss 曲线上看不出来。
    """
    v = F.normalize(q, dim=-1)                                 # [M,R,L,H,d]
    R = v.shape[1]
    if R < 2:
        return v.sum() * 0.0
    s = torch.einsum("mrlhd,mslhd->mlhrs", v, v)               # [M,L,H,R,R]
    off = ~torch.eye(R, dtype=torch.bool, device=v.device)
    return (s[..., off] - margin).clamp_min(0).pow(2).mean()


def diversity_loss(q, margin=0.5):
    """防 probe 坍缩：`mean_{m≠n} max(cos(q̂_m,q̂_n) − margin, 0)²`。

    ⚠ **撤回首版的 `mean cos²` 与它的说明（2026-08-22，外部复核指出，我同意）。**
    首版写「不强制正交」是**错的**，两处：
      ① 最小化 `cos²` 的唯一零点就是 `cos = 0` ⇒ 它**确实**在强制正交；
      ② `cos = −1`（反向 probe）也拿满罚 `cos²=1`，可是 `softmax(q·k)` 与
         `softmax(−q·k)` 给出的**排序几乎相反** —— 那是**最大**的多样性，
         不是坍缩。首版会主动把这种解推开。
    改成**单边 hinge、作用在有符号的 `cos` 上**：只惩罚「过度相似」，
    对正交与反向都零罚。`margin` 是唯一超参，默认 0.5。
    """
    v = F.normalize(q.flatten(1), dim=-1)                      # [M, L*H*d]
    s = v @ v.T
    M = v.shape[0]
    off = ~torch.eye(M, dtype=torch.bool, device=v.device)
    return (s[off] - margin).clamp_min(0).pow(2).mean()


def _selftest():
    torch.manual_seed(0)
    D, d, L, H, M, N, R = 256, 32, 3, 2, 5, 40, 4
    net = ProMetaPredictor(D, d, L, H, n_future=M, d_proj=16, n_pool=2, d_lat=8, n_atoms=R)
    hid = torch.randn(N, D)
    q = net(hid)
    assert q.shape == (M, R, L, H, d), q.shape
    tau = net.log_tau.exp()
    assert torch.allclose(q.norm(dim=-1), tau[None, None].expand(M, R, L, H), atol=1e-5)
    print(f"① 形状 {tuple(q.shape)}、‖q̂‖ = τ_{{l,h}}（初始 1.0）　PASS　"
          f"参数 {sum(p.numel() for p in net.parameters()):,}")

    keys = [torch.randn(1, H, N, d) for _ in range(L)]
    U = ProMetaPredictor.demand(q, keys)
    assert U.shape == (M, L, H, N), U.shape
    assert torch.allclose(U.sum(-1), torch.ones(M, L, H), atol=1e-5), "必须是分布"
    print(f"② demand 形状 {tuple(U.shape)}、逐行和为 1　PASS")

    # ③ 手算对拍一格：max over atoms 再归一化（与教师 future_utility 同族）
    m, l, h = 2, 1, 0
    p_ = torch.softmax(torch.einsum("rd,nd->rn", q[m, :, l, h], keys[l][0, h]) / d ** 0.5, -1)
    ref = p_.amax(0); ref = ref / ref.sum()
    e3 = (ref - U[m, l, h]).abs().max()
    assert e3 < 1e-6, e3
    print(f"③ 手算对拍（max_r 后重归一化）max|差| = {e3:.2e}　PASS")

    # ④ **R=1 必须退化回旧的纯 softmax**（新族包含旧族，这是可验证的）
    net1 = ProMetaPredictor(D, d, L, H, n_future=M, d_proj=16, n_pool=2, d_lat=8, n_atoms=1)
    q1 = net1(hid)
    U1 = ProMetaPredictor.demand(q1, keys)
    ref1 = torch.softmax(torch.einsum("md,nd->mn", q1[:, 0, 1, 0], keys[1][0, 0]) / d ** 0.5, -1)
    e4 = (ref1 - U1[:, 1, 0]).abs().max()
    assert e4 < 1e-6, e4
    print(f"④ R=1 逐位退化成单 softmax，max|差| = {e4:.2e}　PASS")

    # ⑤ **温度真的在起作用**：改 log_tau 必须改分布的尖锐度（旧架构做不到）
    with torch.no_grad():
        net1.log_tau.fill_(2.0)
    U1b = ProMetaPredictor.demand(net1(hid), keys)
    ent_a = -(U1 * U1.clamp_min(1e-12).log()).sum(-1).mean()
    ent_b = -(U1b * U1b.clamp_min(1e-12).log()).sum(-1).mean()
    assert ent_b < ent_a - 1e-3, (float(ent_a), float(ent_b))
    print(f"⑤ τ: 1.0→{2.0:.1f} 使熵 {ent_a:.4f}→{ent_b:.4f}（更尖）　PASS")

    # ⑥ q̂ 依赖前缀 hidden、与 keys 解耦
    assert (q - net(torch.randn(N, D))).abs().max() > 1e-4
    print("⑥ q̂ 依赖前缀 hidden、且与 keys 解耦　PASS")

    # ⑦ 两个多样性损失各司其职：未来间 / atom 间
    dv = diversity_loss(q).item()
    da = atom_diversity_loss(q).item()
    assert dv < 0.24 and da < 0.24, (dv, da)
    qc = q.clone(); qc[:, 1] = qc[:, 0]                     # 人为让 atom0/1 重合
    assert atom_diversity_loss(qc).item() > da + 1e-4
    qf = q.clone(); qf[1] = qf[0]                           # 人为让未来0/1 重合
    assert diversity_loss(qf).item() > dv + 1e-4
    # 反向 atom 必须零罚（首版 cos² 会罚满，那是最大多样性不是坍缩）
    two = torch.stack([q[0, 0], -q[0, 0]], 0)[None]         # [1,2,L,H,d]
    assert atom_diversity_loss(two).item() == 0.0
    print(f"⑦ 未来多样性 {dv:.4f} / atom 多样性 {da:.4f}；人为坍缩各自上升；"
          f"反向 atom 零罚　PASS")

    # ⑧ atom 利用率可查（硬 max 下没赢过的 atom 梯度恒零，会静默死掉）
    _, use = ProMetaPredictor.demand(q, keys, ret_usage=True)
    assert use.shape == (R,) and abs(float(use.sum()) - 1.0) < 1e-5
    print(f"⑧ atom 利用率 {[f'{float(x):.3f}' for x in use]}（和为 1）　PASS")

    # ⑨ 在线/离线池化对拍（同一个量两条实现必须对拍）
    import sys as _s, os as _o
    _s.path.insert(0, _o.path.dirname(_o.path.dirname(_o.path.abspath(__file__))))
    from prometa.pool import OnlineAttnPool
    op = OnlineAttnPool(net.pool_q)
    hp = net.proj(hid)
    for i in range(0, N, 7):
        op.update(hp[i:i + 7])
    e = (op.value() - net.pool(hid)).abs().max().item()
    e2 = (net.from_pooled(op.value()) - net(hid)).abs().max().item()
    assert e < 1e-5 and e2 < 1e-5, (e, e2)
    print(f"⑨ 在线/离线池化对拍 {e:.2e}；q̂ 对拍 {e2:.2e}　PASS")

    # ⑩ 梯度到达全部可训练张量（含 log_tau 与全部 atom）
    net2 = ProMetaPredictor(D, d, L, H, n_future=M, d_proj=16, n_pool=2, d_lat=8, n_atoms=R)
    q2 = net2(hid)
    U2 = ProMetaPredictor.demand(q2, keys)
    (U2.sum() + diversity_loss(q2) + atom_diversity_loss(q2)).backward()
    dead = [n for n, p in net2.named_parameters()
            if p.grad is None or p.grad.abs().max() == 0]
    assert not dead, f"这些参数没拿到梯度：{dead}"
    print(f"⑩ {len(list(net2.parameters()))} 个参数张量全部拿到非零梯度　PASS")
    print("\nprometa/model.py 自测 10 条全过")


if __name__ == "__main__":
    _selftest()
