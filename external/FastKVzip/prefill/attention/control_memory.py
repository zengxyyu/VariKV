"""ControlMemory —— VariKV-B 最终版的历史控制状态。

与手工版（`ctrlcache.py`，git tag `varikv-b-handcrafted`）的关系：**范式相同、内容不同。**
两者都只修正驱逐分数、绝不进 attention 输出；区别是手工版把「历史有用的部分」
写死成 `xᵀĈx/‖x‖²` 的方向对齐度，而这里让模型自己学。

    M_t = Write_θ(M_{t-1}, R_t, E_t)                  ← 保留集与驱逐集分别写入
    r_i = Read_θ(x_i, M_{t-1})                        ← 当前候选去查历史
    Δs_i = α · σ_base(l,h) · tanh(MLP[x̃_i, r_i, z(s⁰_i)])
    s_i = s⁰_i + Δs_i

要学的量有精确定义。设 U 为 token 的未来效用，无历史的最优预测器是 `E[U|X]`，
有历史则是 `E[U|X,M]`，控制器要学的是二者之差——**历史相对于当前信息的增量**，
而不是重新学一个 importance scorer。平方损失下有精确恒等式

    R_X − R_{X,M} = E[ ( E[U|X,M] − E[U|X] )² ]  ≥ 0

所以**严格改善当且仅当条件均值发生变化**，即 `P(E[U|X,M] ≠ E[U|X]) > 0`。
注意这**不等价于** `I(U;M|X) > 0`：互信息为正也可能只体现在条件方差或更高阶矩上，
此时条件均值不变、平方风险毫无改善。此前把两者写成等价是数学错误。

还有一处必须讲清楚：**`s⁰`（FastKVzip 的门控分）并不等于 `E[U|X]`**，它只是一个
工程上的先验。所以学出来的残差并不是严格意义上的 `Δ*`。理论上比较的是理想预测器
`f₀*(X)` 与 `f_M*(X,M)`；工程上 memoryless 臂近似前者、stateful 臂近似后者，
**历史收益只能由 stateful − memoryless（或更强的 stateful − shuffled）来度量**，
不能由「比 FastKVzip 高」来度量。这恰好就是当前三臂设计。

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
                 n_slots: int = 8, d_m: int = 128, mode: str = "stateful",
                 alpha_max: float = 1.0):
        super().__init__()
        assert mode in ("stateful", "memoryless", "shuffled")
        self.L, self.H, self.K, self.d_m = n_layers, n_heads_kv, n_slots, d_m
        self.mode = mode
        d_x = 2 * d_kv                                  # 候选特征 = [k_i ; v_i]

        # 每组一个可学初始状态（唯一的 per-group 参数）
        self.M_init = nn.Parameter(torch.randn(n_layers, n_heads_kv, n_slots, d_m) * 0.02)

        self.x_proj = nn.Linear(d_x, d_m)               # 候选 → d_m（写入侧 & 头输入）
        # **读出 query 用独立于 x_proj 的投影**。共用一个会强迫同一张矩阵同时满足
        # "写入端摘要"和"读出端内积"两个目标；分开后 <q_read(x), r> 可以自由学成
        # 近似原空间的内积。合成正对照上，共用时"传一个 128 维方向"完全学不到。
        self.q_read = nn.Linear(d_x, d_m)
        self.k_ret = nn.Linear(d_m, d_m)                # 写：保留集
        self.v_ret = nn.Linear(d_m, d_m)
        self.k_evi = nn.Linear(d_m, d_m)                # 写：驱逐集
        self.v_evi = nn.Linear(d_m, d_m)
        self.q_slot = nn.Linear(d_m, d_m)               # 写：槽作 query
        # **除了学出来的注意力池化，还要一条非学习的均值通路。** 合成正对照测出：
        # 只有注意力池化时，"把被驱逐集合的均值方向传给下一个 chunk"学不出来——
        # 要得到均值就得先学会均匀注意，梯度路径太长。均值按构造给出后，
        # 方向信息不再依赖优化是否成功。
        self.mix = nn.Linear(4 * d_m, d_m)
        self.gru = nn.GRUCell(d_m, d_m)
        # **头必须含乘性交互**。要表达"候选与历史的匹配程度"就需要 x 与 r 的双线性项，
        # 而拼接后的 MLP 很难学出乘积——合成正对照上实测：只喂 [x,r,z] 时，即使注入
        # 一个可证明存在的历史信号，三臂全部停在 0.50 随机水平。加入 x⊙r 与 <x,r> 后
        # 线性层直接就能算加权内积。
        self.head = nn.Sequential(
            nn.Linear(3 * d_m + 3, d_m), nn.GELU(), nn.Linear(d_m, 1))
        # α **有上界**：α = α_max·sigmoid(a)。此前写成 sigmoid(a)·exp(b)，b 无界
        # ⇒ α 无界 ⇒ 「tanh 让修正有界」这句话不成立。而手工版实测**大幅扰动基线排序
        # 本身就有害**（β=±1.5 时连 shuffle 对照都掉 4.6–5.8 分），这个事实应当被编码
        # 进架构而不是指望优化器自觉。
        # 初值 sigmoid(-8)·α_max ≈ 3.4e-4·α_max，**接近 0 但不等于 0**；
        # 「α=0 ⇒ 逐位退化回基线」是一个关于 α=0 的命题，不是关于初始化的断言。
        self.alpha_max = float(alpha_max)
        self.alpha_on = nn.Parameter(torch.full((), -8.0))

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_on)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ 状态
    def init_state(self, layer_idx: int, dtype=torch.float32):
        return self.M_init[layer_idx].to(dtype)          # [H,K,d_m]

    def raw(self, k, v):
        """k,v: [H,n,d_kv] → [H,n,2*d_kv]（fp32；bf16 累积会掉精度）"""
        return torch.cat([k, v], dim=-1).float()

    def feat(self, x_raw):
        """[H,n,2*d_kv] → x̃ [H,n,d_m]"""
        return self.x_proj(x_raw)

    # ------------------------------------------------------------------ 读
    def read(self, M, x_raw):
        """M [H,K,d_m], x_raw [H,n,2*d_kv] → r [H,n,d_m]"""
        q = self.q_read(x_raw)                                    # [H,n,d]
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

    @staticmethod
    def _mean(x, mask):
        """被 mask 选中的候选的**朴素均值**，[H,1,d]。空集合给 0。"""
        w = mask.float()[..., None]
        return (x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1.0)

    def write(self, M, x, m_ret, m_evi, gen=None):
        """M [H,K,d_m], x̃ [H,n,d_m], m_ret/m_evi [H,n] bool → M' [H,K,d_m]"""
        if self.mode == "memoryless":
            return M                                              # 状态永不更新
        if self.mode == "shuffled":
            # **成员身份随机置换**：保住的是**条数与计算量**，不是统计量——换了成员，
            # 均值/池化/方向统计当然都变，这正是要破坏的东西。准确说法是
            # "preserves retained/evicted counts and compute, destroys content membership"。
            n = x.shape[1]
            # **generator 的 device 必须和 randperm 的 device 一致**：randperm 默认
            # 建 CPU 张量，传 CUDA generator 会直接报错。推理路径传的是 CPU generator
            # 所以一直没暴露，训练脚本传 CUDA generator 时才炸。
            gdev = gen.device if gen is not None else x.device
            perm = torch.stack([torch.randperm(n, generator=gen, device=gdev)
                                for _ in range(x.shape[0])]).to(x.device)
            m_ret = torch.gather(m_ret, 1, perm)
            m_evi = torch.gather(m_evi, 1, perm)
        a_r = self._pool(M, x, m_ret, self.k_ret, self.v_ret)
        a_e = self._pool(M, x, m_evi, self.k_evi, self.v_evi)
        mu_r = self._mean(x, m_ret)                               # [H,1,d] 广播到 K
        mu_e = self._mean(x, m_evi)
        K_ = M.shape[1]
        u = self.mix(torch.cat([a_r, a_e,
                                mu_r.expand(-1, K_, -1),
                                mu_e.expand(-1, K_, -1)], dim=-1))
        H, K, d = M.shape
        return self.gru(u.reshape(-1, d), M.reshape(-1, d)).view(H, K, d)

    # ------------------------------------------------------------ 控制器
    def delta(self, x, r, s0, q=None, margin=None):
        """x̃/r [H,n,d_m], s0 [H,n] 原始基线分 → Δs [H,n]（**已含 α 与逐头尺度**）。

        两个标量特征，缺一不可：

        - `z` = s0 在 (层,kv头) 内的 z-score。不归一的话控制器会隐式学各头的尺度差异
          而不是内容。
        - `margin` = (s0 − τ)/σ_global，**到全局淘汰阈值的距离**。`level="pair"` 是
          跨层跨头的全局阈值化，真正决定去留的是 `s0 − τ` 而不是头内排名：两个 token
          可以有完全相同的 z 却因为所在头整体分数高低不同而一个稳留、一个稳删。
          只喂 z 等于把决策边界的信息藏起来。τ 在 trace 里已经存了，代价为零。
        """
        z = (s0 - s0.mean(-1, keepdim=True)) / s0.std(-1, keepdim=True).clamp_min(1e-6)
        mg = torch.zeros_like(z) if margin is None else margin
        qq = x if q is None else q
        dot = (qq * r).sum(-1, keepdim=True) * self.d_m ** -0.5
        raw = self.head(torch.cat([x, r, qq * r, dot,
                                   z[..., None], mg[..., None]],
                                  dim=-1)).squeeze(-1)
        sb = s0.std(-1, keepdim=True).clamp_min(1e-6)             # 逐头尺度
        return self.alpha * sb * torch.tanh(raw)
