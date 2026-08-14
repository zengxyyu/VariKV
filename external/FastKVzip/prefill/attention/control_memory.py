"""ControlMemory —— VariKV-B 最终版的历史控制状态。

与手工版（`ctrlcache.py`，git tag `varikv-b-handcrafted`）的关系：**范式相同、内容不同。**
两者都只修正驱逐分数、绝不进 attention 输出；区别是手工版把「历史有用的部分」
写死成 `xᵀĈx/‖x‖²` 的方向对齐度，而这里让模型自己学。

    M_t = Write_θ(M_{t-1}, R_t, E_t)                  ← 保留集与驱逐集分别写入
    r_i = Read_θ(x_i, M_{t-1})                        ← 当前候选去查历史
    Δs_i = α · σ_base(l,h) · tanh(MLP[x̃_i, r_i, z(s⁰_i)])
    s_i = s⁰_i + Δs_i

要学的量有精确定义。设 U 为 token 的未来效用，无历史的选择器最好只能预测
`E[U|X]`，有历史则是 `E[U|X,M]`，所以控制器要学的恰好是

    Δ*(x,m) = E[U|X=x,M=m] − E[U|X=x]

即**历史相对于当前信息的增量**，而不是重新学一个 importance scorer。
条件化不会增加最优 Bayes 风险，严格改善当且仅当 `I(U;M|X) > 0`。这就是 B 的核心科学假设：
**过去保留/丢弃了什么，改变了接下来该保留什么的边际价值。**

--------------------------------------------------------------------------------
四个设计决策（都不是随手定的）

1. **参数在 112 个 (层, kv头) 之间共享**，每组只有一个可学的初始状态 `M_init[l,h]`。
   每组一套参数是 112 倍规模，而且会退化成"每个头背下自己的规则"；共享逼它学通用规则，
   也让「有历史 vs 无历史」的对照不掺容量差异。

2. **`alpha` 初始化为 0 ⇒ Δs ≡ 0 ⇒ 逐位退化回基线。** 构造性保证，不是断言。
   而且这一条是必要的：手工版实测大幅扰动 FastKVzip 排序**本身**就有害
   （β=±1.5 时连 shuffle 对照都掉 4.6–5.8 分），所以修正必须从 0 出发、且有界。
   `tanh` 提供有界性。

3. **写入用带掩码的 cross-attention 而不是变长 gather。** 每组 K×n 个分数，
   K=8、n=16000 时 112 组共 14M，可忽略；而变长 gather 会把每头不同长度的
   `[Σ_h n_h, d]` 布局问题重新引进来——那是 `memcache.py` 当初最难的部分。

4. **三种 mode，其中 `shuffled` 才是强对照**：
   - `stateful`   : 正常
   - `memoryless` : M 永远停在 M_init（写入不执行）
   - `shuffled`   : 写入照常执行，但 retained/evicted 的**成员身份被随机置换**
     ⇒ 参数、计算量、状态动力学完全一致，只破坏「历史 ↔ 当前候选」的对应关系。
   `memoryless` 的可训练参数严格更少（writer 闲置），历史臂赢了也说不清是信息还是容量；
   `shuffled` 没有这个问题。手工版里同样的对照模式已经验证有效。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlMemory(nn.Module):
    def __init__(self, d_kv: int, n_layers: int, n_heads_kv: int,
                 n_slots: int = 8, d_m: int = 64, mode: str = "stateful"):
        super().__init__()
        assert mode in ("stateful", "memoryless", "shuffled")
        self.L, self.H, self.K, self.d_m = n_layers, n_heads_kv, n_slots, d_m
        self.mode = mode
        d_x = 2 * d_kv                                  # 候选特征 = [k_i ; v_i]

        # 每组一个可学初始状态（唯一的 per-group 参数）
        self.M_init = nn.Parameter(torch.randn(n_layers, n_heads_kv, n_slots, d_m) * 0.02)

        self.x_proj = nn.Linear(d_x, d_m)               # 候选 → d_m
        self.q_read = nn.Linear(d_m, d_m)               # 读：候选作 query
        self.k_ret = nn.Linear(d_m, d_m)                # 写：保留集
        self.v_ret = nn.Linear(d_m, d_m)
        self.k_evi = nn.Linear(d_m, d_m)                # 写：驱逐集
        self.v_evi = nn.Linear(d_m, d_m)
        self.q_slot = nn.Linear(d_m, d_m)               # 写：槽作 query
        self.mix = nn.Linear(2 * d_m, d_m)
        self.gru = nn.GRUCell(d_m, d_m)
        self.head = nn.Sequential(                      # 控制器
            nn.Linear(2 * d_m + 1, d_m), nn.GELU(), nn.Linear(d_m, 1))
        # **alpha=0 ⇒ 逐位退化回基线**。这是构造性的，验收里必须实测到。
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.alpha_on = nn.Parameter(torch.zeros(()))   # sigmoid(0)=0.5 起步太大，见下
        with torch.no_grad():
            self.alpha_on.fill_(-8.0)                   # sigmoid(-8)=3.4e-4 ≈ 0

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_on) * self.log_alpha.exp()

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ 状态
    def init_state(self, layer_idx: int, dtype=torch.float32):
        return self.M_init[layer_idx].to(dtype)          # [H,K,d_m]

    def feat(self, k, v):
        """k,v: [H,n,d_kv] → x̃ [H,n,d_m]（fp32；bf16 累积会掉精度）"""
        return self.x_proj(torch.cat([k, v], dim=-1).float())

    # ------------------------------------------------------------------ 读
    def read(self, M, x):
        """M [H,K,d_m], x̃ [H,n,d_m] → r [H,n,d_m]"""
        q = self.q_read(x)                                        # [H,n,d]
        att = torch.einsum("hnd,hkd->hnk", q, M) * self.d_m ** -0.5
        return torch.einsum("hnk,hkd->hnd", att.softmax(-1), M)

    # ------------------------------------------------------------------ 写
    def _pool(self, M, x, mask, kp, vp):
        """槽对 (被 mask 选中的) 候选做 cross-attention → [H,K,d_m]。

        用带掩码的 softmax 而不是变长 gather：每头保留/驱逐的条数不同，
        gather 会把 per-head 变长布局重新引进来。
        """
        q = self.q_slot(M)                                        # [H,K,d]
        att = torch.einsum("hkd,hnd->hkn", q, kp(x)) * self.d_m ** -0.5
        att = att.masked_fill(~mask[:, None, :], float("-inf"))
        att = att.softmax(-1)
        att = torch.nan_to_num(att, nan=0.0)                      # 空集合 ⇒ 全 -inf
        return torch.einsum("hkn,hnd->hkd", att, vp(x))

    def write(self, M, x, m_ret, m_evi, gen=None):
        """M [H,K,d_m], x̃ [H,n,d_m], m_ret/m_evi [H,n] bool → M' [H,K,d_m]"""
        if self.mode == "memoryless":
            return M                                              # 状态永不更新
        if self.mode == "shuffled":
            # **成员身份随机置换**：集合大小与统计量不变，只打断"历史↔候选"的对应
            n = x.shape[1]
            perm = torch.stack([torch.randperm(n, generator=gen)
                                for _ in range(x.shape[0])]).to(x.device)
            m_ret = torch.gather(m_ret, 1, perm)
            m_evi = torch.gather(m_evi, 1, perm)
        a_r = self._pool(M, x, m_ret, self.k_ret, self.v_ret)
        a_e = self._pool(M, x, m_evi, self.k_evi, self.v_evi)
        u = self.mix(torch.cat([a_r, a_e], dim=-1))               # [H,K,d]
        H, K, d = M.shape
        return self.gru(u.reshape(-1, d), M.reshape(-1, d)).view(H, K, d)

    # ------------------------------------------------------------ 控制器
    def delta(self, x, r, s0):
        """x̃/r [H,n,d_m], s0 [H,n] 原始基线分 → Δs [H,n]（**已含 α 与逐头尺度**）。

        s0 先在 (层,kv头) 内 z-score：`level="pair"` 是全局阈值化，不归一的输入会让
        控制器隐式学到各头的尺度差异而不是内容。
        """
        z = (s0 - s0.mean(-1, keepdim=True)) / s0.std(-1, keepdim=True).clamp_min(1e-6)
        raw = self.head(torch.cat([x, r, z[..., None]], dim=-1)).squeeze(-1)
        sb = s0.std(-1, keepdim=True).clamp_min(1e-6)             # 逐头尺度
        return self.alpha * sb * torch.tanh(raw)
