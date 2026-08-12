"""带计数的点质心 + 归一化感知读出 —— P1 的 E1b 估计器接到真实 harness。

**为什么需要这个文件（2026-08-12）。** `memcache_retain.py` 的残差读出与 P1 阶梯里
唯一有效的估计器（E1b，局部恢复 oracle 的 70%）**不是同一个数学结构**，差四处：

| | E1b | memcache_retain 的残差读出 |
|---|---|---|
| 簇的成员数 `n_j` | `D̂_E = Σ_j n_j e^{aᵀk̄_j}` 带 `log n_j` | **无**。一个概括 1400 个 token 的槽只算一票 |
| 归一化 | 与保留侧**共享** softmax ⇒ 幸存者权重被正确调回 | 槽内单独 softmax，真实那次 softmax 完全不受影响 |
| 修正幅度 | `1−λ(q) = D̂_E/(D_R+D̂_E)`，**随 query 变** | 每 (层, kv头) 一个**学出来的常数** σ(gate) |
| 容量 | 109 簇/head | 16 |

四条同时存在，任何一条都足以让下游结果为零。本文件只做**正确的代数**，
不含 encoder/decoder/门控/优化器 —— 免训练。

**代数（在同一 q/K/V 内精确成立）。** 设保留集 R、被驱逐集 E，
`a = q/√d`，`D_X = Σ_{i∈X} e^{aᵀk_i}`，`N_X = Σ_{i∈X} e^{aᵀk_i} v_i`。则

    o_all = (N_R + N_E)/(D_R + D_E) = λ·o_R + (1−λ)·o_E,    λ = D_R/(D_R+D_E)

用簇 `(k̄_j, v̄_j, n_j)` 近似 E 侧的两个和：

    r_j = aᵀk̄_j + log n_j,   L_E = logsumexp_j r_j,   o_E = softmax(r)·V̄
    L_R = log D_R  ←  flash_attn_varlen_func(..., return_attn_probs=True) 直接给
    L   = logaddexp(L_R, L_E),   λ = e^{L_R − L},   o = λ o_R + (1−λ) o_E

与「把质心当 pseudo-token 塞进同一次 softmax 并加 `log n_j` 偏置」**严格等价**，
但不需要 kernel 支持任意 logit bias，且可逐项对账。全程 log 空间，无溢出。

**三个刻意的设计选择，改动前先读。**

1. **只在 query / 生成时修正，不在分块预填时修正**（`q_len <= correct_max_q`）。
   这是**忠实复现 E1b**：阶梯里的修正是在 query 前向上用 o_proj 钩子施加的，
   预填走的是普通剪枝预填。也顺带避免 16000-token chunk 上的额外注意力开销。
2. **key 默认直接用缓存里的 post-RoPE key 做平均**（`rope_mode="post"`）。
   E1b 就是这么做的，第一轮必须严格复现。注意这与 `CLAUDE.md` 的 RoPE 一节
   有张力：post-RoPE key 的平均在一般情况下不是任何位置的合法 key（MemRoPE）。
   它在这里能用，是因为簇是**位置局部**的，簇内相位跨度有界。
   **后果：K 越大簇越窄 ⇒ 内容分辨率与相位一致性同时改善，容量曲线混淆两个效应。**
   `rope_mode="inv"` 是分开它们的对照臂（逆旋到无位置帧再按位置质心旋回）。
3. **簇 = 位置局部块，宽度 W 自适应倍增。** 块号 = pos // W；当占用块数将超过 K 时
   W 翻倍并把相邻两块合并（和与计数直接相加 ⇒ 合并无损）。这样无需预知上下文长度，
   收敛结果与 E1b 的固定宽度分块一致。

**因果性**：被驱逐的全是 context token，全部先于 query token，且 harness 每问一次
就 `slice()` 回滚，所以「簇只概括 i<t」自动成立。`assert_causal()` 把它钉死。
"""
import torch

from .kvcache import RetainCache


class CentroidRetainCache(RetainCache):
    """RetainCache + 带计数的点质心摘要 + 归一化感知的读出。"""

    def __init__(self, model, evict_range, n_clusters: int = 109,
                 rope_inv_freq=None, rope_mode: str = "post",
                 correct_max_q: int = 4096, enabled: bool = True):
        super().__init__(model, evict_range)
        self.head_dim = getattr(
            model.config, "head_dim",
            model.config.hidden_size // model.config.num_attention_heads)
        self.K = int(n_clusters)
        self.inv_freq = rope_inv_freq
        assert rope_mode in ("post", "inv")
        self.rope_mode = rope_mode
        if rope_mode == "inv":
            assert rope_inv_freq is not None, "rope_mode='inv' 需要 inv_freq"
        self.correct_max_q = int(correct_max_q)
        self.centroid_mode = bool(enabled) and self.K > 0
        # attn.py 用它决定是否在 flash 调用上要 softmax_lse（有代价，故按需）
        self.need_lse = self.centroid_mode

        H, K, d = self.n_heads_kv, self.K, self.head_dim
        dev = self.device
        z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float32)  # noqa: E731
        self.c_sum_k = [z(H, K, d) for _ in range(self.n_layers)]
        self.c_sum_v = [z(H, K, d) for _ in range(self.n_layers)]
        self.c_sum_p = [z(H, K) for _ in range(self.n_layers)]
        self.c_cnt = [z(H, K) for _ in range(self.n_layers)]
        self.W = 64                       # 块宽，按需倍增
        self._absorbed = 0                # 已吸收的 KV 条数（诊断 + 空记忆判据）
        self._max_pos = -1                # 已吸收的最大原始位置（因果性断言用）
        self._cache = [None] * self.n_layers   # 每层的 (k̄, v̄, log n) 缓存

    # -------------------------------------------------------------- 预算核算

    def budget(self):
        """返回本方法的字节核算，用于 matched-budget 对照。

        一条 exact KV = 2d scalars；一个带计数的质心 = 2d+1 scalars ⇒ 1.004×。
        所以「K 个质心」≈「K 条 exact KV」，两者可直接对比。
        """
        d = self.head_dim
        occ = int(sum(int((c > 0).sum()) for c in self.c_cnt))
        retained = int(self.valid.sum()) if self.valid is not None else 0
        return {"clusters_occupied": occ, "K_per_head": self.K,
                "scalars_centroid": occ * (2 * d + 1),
                "retained_kv": retained, "scalars_retained": retained * 2 * d,
                "absorbed_kv": self._absorbed,
                "overhead_frac": occ * (2 * d + 1) / max(retained * 2 * d, 1)}

    # -------------------------------------------------------------- 写入

    def _grow(self, need_blocks):
        """块数将超过 K ⇒ W 翻倍、相邻两块合并（和与计数直接相加 ⇒ 合并无损）。

        用 `index_add_(1, b//2, ·)` 而不是 `buf[:,0::2]+buf[:,1::2]` —— **K 为奇数时
        两个 stride 切片长度差 1**（K=109 ⇒ 55 vs 54），直接相加会崩。
        `pos//(2W) = (pos//W)//2`，所以按 b//2 归并与新的块宽严格一致。
        """
        while need_blocks > self.K:
            tgt = (torch.arange(self.K, device=self.device) // 2)
            for l in range(self.n_layers):
                for buf in (self.c_sum_k[l], self.c_sum_v[l],
                            self.c_cnt[l], self.c_sum_p[l]):
                    new = torch.zeros_like(buf)
                    new.index_add_(1, tgt, buf)
                    buf.copy_(new)
            self.W *= 2
            need_blocks = (need_blocks + 1) // 2
        self._cache = [None] * self.n_layers

    def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
        out = super().prune_chunk(ratio, evict_range, level)
        if not self.centroid_mode:
            return out
        lo, hi = evict_range
        s, e = lo - self.sink, hi - self.sink          # self.valid 坐标（不含 sink）
        if e <= s:
            return out
        # 最大原始位置 = e-1+sink ⇒ 需要的块数
        self._grow(((e - 1 + self.sink) // self.W) + 1)
        for l in range(self.n_layers):
            self._absorb_layer(l, s, e)
        self._cache = [None] * self.n_layers
        return out

    @torch.no_grad()
    def _absorb_layer(self, layer_idx, s, e):
        H = self.n_heads_kv
        vmask = self.valid[layer_idx][..., s:e]
        if vmask.dim() == 3:
            vmask = vmask.squeeze(0)
        k_all, v_all = self.key_cache[layer_idx], self.value_cache[layer_idx]
        for h in range(H):
            drop = (~vmask[h]).nonzero(as_tuple=True)[0]
            if drop.numel() == 0:
                continue
            pos = (drop + s + self.sink)               # 原始 token 位置（无记忆前缀）
            kk = k_all[0, h, pos].float()
            vv = v_all[0, h, pos].float()
            if self.rope_mode == "inv":
                from varikv.rope import cos_sin_at, inverse_rope
                cos, sin = cos_sin_at(self.inv_freq, pos.float().view(1, -1),
                                      dtype=kk.dtype)
                kk = inverse_rope(kk.view(1, 1, -1, kk.shape[-1]), cos, sin)[0, 0]
            b = (pos // self.W).clamp_(max=self.K - 1)
            self.c_sum_k[layer_idx][h].index_add_(0, b, kk)
            self.c_sum_v[layer_idx][h].index_add_(0, b, vv)
            self.c_sum_p[layer_idx][h].index_add_(0, b, pos.float())
            self.c_cnt[layer_idx][h].index_add_(
                0, b, torch.ones_like(b, dtype=torch.float32))
            self._absorbed += int(drop.numel())
            self._max_pos = max(self._max_pos, int(pos.max()))

    # -------------------------------------------------------------- 读出

    def _summary(self, layer_idx, dtype):
        """返回 (k̄ [H,K,d], v̄ [H,K,d], log n [H,K])，空簇的 log n = −inf。"""
        c = self._cache[layer_idx]
        if c is not None and c[0].dtype == dtype:
            return c
        cnt = self.c_cnt[layer_idx]
        occ = cnt > 0
        den = cnt.clamp_min(1.0).unsqueeze(-1)
        kbar = self.c_sum_k[layer_idx] / den
        vbar = self.c_sum_v[layer_idx] / den
        if self.rope_mode == "inv":
            from varikv.rope import cos_sin_at, apply_rope
            pbar = self.c_sum_p[layer_idx] / cnt.clamp_min(1.0)   # 位置质心
            cos, sin = cos_sin_at(self.inv_freq, pbar, dtype=kbar.dtype)
            kbar = apply_rope(kbar.unsqueeze(0), cos, sin)[0]
        # 空簇用**大负有限值**而不是 −inf：一个头若所有簇都空，全 −inf 的行
        # 经 softmax 会得到 NaN，再乘 (1−λ)=0 仍是 NaN，会污染整个输出（实测踩到）。
        # −1e30 下 softmax 权重精确为 0、logsumexp 有限、λ 精确为 1，语义与 −inf 相同。
        logn = torch.where(occ, cnt.clamp_min(1.0).log(),
                           torch.full_like(cnt, -1e30))
        c = (kbar.to(dtype), vbar.to(dtype), logn.to(torch.float32))
        self._cache[layer_idx] = c
        return c

    def assert_causal(self, q_abs_start: int):
        """簇只能概括 i < t。被驱逐的全在 context 内、query 全在其后 ⇒ 自动成立。"""
        assert self._max_pos < q_abs_start, (
            f"因果性违反：吸收了位置 {self._max_pos}，而 query 起始于 {q_abs_start}")

    @torch.no_grad()
    def memory_correct(self, query_states, layer_idx, o_R, lse):
        """o = λ·o_R + (1−λ)·o_E，返回与 o_R 同形状 [B, T, HQ*d]。

        query_states: [B, HQ, T, d]，post-RoPE、prepare 之前
        o_R:          [B, T, HQ*d]，flash 输出（已 view）
        lse:          [G, H*T]，flash 的 softmax_lse，= log Σ exp(aᵀk)，见 attn.py
        """
        if not self.centroid_mode or self._absorbed == 0:
            return o_R
        B, HQ, T, d = query_states.shape
        if T > self.correct_max_q:            # 分块预填不修正（忠实复现 E1b）
            return o_R
        H = self.n_heads_kv
        G = HQ // H
        kbar, vbar, logn = self._summary(layer_idx, torch.float32)
        q = query_states.view(B, H, G, T, d)[0].float() * (d ** -0.5)   # [H,G,T,d]
        o = o_R.view(B, T, H, G, d)[0].float()                          # [T,H,G,d]
        lse = lse.float().view(G, H, T)                                 # [G,H,T]
        r = torch.einsum("hgtd,hkd->hgtk", q, kbar) + logn[:, None, None, :]
        L_E = torch.logsumexp(r, -1)                                    # [H,G,T]
        o_E = torch.einsum("hgtk,hkd->hgtd", torch.softmax(r, -1), vbar)
        L_R = lse.permute(1, 0, 2)                                      # [H,G,T]
        L = torch.logaddexp(L_R, L_E)
        lam = torch.exp(L_R - L).unsqueeze(-1)                          # [H,G,T,1]
        out = lam * o.permute(1, 2, 0, 3) + (1.0 - lam) * o_E           # [H,G,T,d]
        assert torch.isfinite(out).all(), f"层 {layer_idx} 的质心修正出现非有限值"
        return out.permute(2, 0, 1, 3).reshape(B, T, HQ * d).to(o_R.dtype)
