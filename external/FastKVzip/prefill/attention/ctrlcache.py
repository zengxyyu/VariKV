"""ControlRetainCache —— 记忆只用来**修正驱逐分数**，不进 attention。

和 `memcache_retain.py` / `centroid.py` 的根本区别：那两个都把记忆变成 KV 送进 softmax，
这个**完全不碰 attention 输出**。记忆是一个旁路状态，只在压缩决策那一刻起作用：

    s'_i = s_base_i
         + β_tok · σ_base(l,h) · z_{l,h}(nov_i)      ← 头内重排
         + β_grp · σ_base(全局) · z_{(l,h)}(mean_i nov)  ← 跨头/层的预算再分配

时序（无同 chunk 泄漏）：用 **C_{t-1}** 算修正 → 阈值化定出 R_t/E_t → 再用 R_t/E_t 更新到 C_t。

--------------------------------------------------------------------------------
两项修正为什么用**不同的尺度**（这是设计要点，不是笔误）

`level="pair"` 是对所有 (层, kv头, token) **全局**排序阈值化。

- 头内重排项要和该头自己的分数分布可比 ⇒ 乘 **σ_base(l,h)**。它在头内零均值，
  所以它**几乎不能**系统性地把预算从一个头挪到另一个头。
- 要表达「这个头整体该多拿/少拿预算」，修正必须在**全局**分数尺度上 ⇒ 乘 **σ_base(全局)**。
  用逐头 σ 去做跨头搬运会被各头自己的尺度扭曲。

**因此 β_grp=0（默认）时，本模块只在检验 B 路线的「头内重排」子问题，不是它的全部能力。**
若第一批得到 null，不能直接宣判 memory-guided eviction 失败——还有 β_grp 这一半没测。

--------------------------------------------------------------------------------
对齐度的定义（无需低秩、无需特征分解）

    C_t = ρ·C_{t-1} + Σ_{i∈S_t} x_i x_iᵀ         （S_t 由 --ctrl_src 决定）
    Ĉ   = C / tr(C)                              （迹归一 ⇒ 尺度无关）
    align(x)   = xᵀĈx / ‖x‖²
    nov(x)     = −align(x)

**术语要精确**：`C` 是**未中心化的二阶矩 / Gram 状态**，不是协方差（没有减均值）；
`Ĉ` 迹归一后仍然 `Ĉ² ≠ Ĉ`，**不是投影算子**，所以 `xᵀĈx/‖x‖²` 只能叫
「与历史累积方向的对齐度」，**不能**解释成「投影到历史子空间的能量占比」。
写论文时这个区别是要被审的。

另一个隐含选择：历史 token 以 `‖x_i‖²` 加权进 C，所以这是 **energy-weighted** 而非
frequency-weighted 的方向统计。norm 波动大不大由 `self.norm_cv` 诊断记录，先测再决定
要不要改成单位化的 `Σ u_i u_iᵀ`。

尺寸：`feat=key|value` 时 d=128 ⇒ 每 (层,kv头) 16384 个 float，112 头共 **7.3 MB**；
`feat=keyvalue` 时 d=256 ⇒ **29 MB**，且矩阵运算量 4×。所以 keyvalue 若更好，
**不能**直接说「K+V 表示更强」——它同时拿到了 4 倍状态，正式消融要做等字节对照。

--------------------------------------------------------------------------------
预算：**不是构造性恒等，是经验性相等**

父类 `_threshold` 用的是 `score > score_sort[n]`，不是严格 `topk`。阈值处若有并列，
保留数会少于 n。连续分数下并列极罕见，实测应当逐样本完全相等——但那是**要被验证的经验
事实**，不是构造性保证。（这一条此前写成 "exact budget matching by construction"，过强，已收紧。）

--------------------------------------------------------------------------------
范围限制：本类仍派生自 `RetainCache`，它**逻辑上**用掩码压缩、物理上保留全部 KV。
所以现在能回答「memory-conditioned selection 有没有用」，**不能**用来声称峰值显存下降。
若 B 路线成立，再移植到 `EvictCache` / 真流式实现。
"""
from typing import Tuple

import torch

from .kvcache import RetainCache


class ControlRetainCache(RetainCache):
    def __init__(self, model, evict_range: Tuple[int, int], beta: float = 0.0,
                 beta_group: float = 0.0, rho: float = 1.0, src: str = "evicted",
                 feat: str = "key", rope_mode: str = "post", rope_inv_freq=None,
                 shuffle: bool = False, seed: int = 0):
        super().__init__(model, evict_range)
        self.beta, self.beta_group = float(beta), float(beta_group)
        self.rho, self.src = float(rho), src
        self.feat, self.rope_mode, self.shuffle = feat, rope_mode, bool(shuffle)
        assert src in ("evicted", "retained")
        assert feat in ("key", "value", "keyvalue")
        # 基类不提供这两个，和 centroid.py:86 取同一条路径
        self.head_dim = getattr(
            model.config, "head_dim",
            model.config.hidden_size // model.config.num_attention_heads)
        self.inv_freq = rope_inv_freq
        if rope_mode == "inv":
            assert rope_inv_freq is not None, "rope_mode='inv' 需要 inv_freq"
        d = self.head_dim * (2 if feat == "keyvalue" else 1)
        self.d_feat = d
        # float32 累积：bf16 累加二阶矩会掉精度（bug 列表里有先例）
        self.C = [torch.zeros(self.n_heads_kv, d, d, dtype=torch.float32,
                              device=self.device) for _ in range(self.n_layers)]
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        # 诊断量（供报告核对，不参与计算）
        self.n_absorbed = 0
        self.corr_std = []        # 每 chunk 的修正幅度
        self.norm_cv = []         # 每 chunk 的 ‖x‖ 变异系数

    @property
    def active(self) -> bool:
        return self.beta != 0.0 or self.beta_group != 0.0

    # ------------------------------------------------------------------ 特征
    def _x(self, layer_idx: int, pos: torch.Tensor) -> torch.Tensor:
        """→ [H, n, d_feat]，pos 是**缓存坐标**下的位置索引（与 centroid.py 同一约定）。"""
        k = self.key_cache[layer_idx][0][:, pos].float()        # [H,n,dh]
        if self.rope_mode == "inv":
            from varikv.rope import cos_sin_at, inverse_rope
            cos, sin = cos_sin_at(self.inv_freq, pos.float(), dtype=k.dtype)
            k = inverse_rope(k, cos, sin)
        if self.feat == "key":
            return k
        v = self.value_cache[layer_idx][0][:, pos].float()
        return v if self.feat == "value" else torch.cat([k, v], dim=-1)

    def _novelty(self, l: int, x: torch.Tensor):
        """x: [H,n,d] → nov [H,n]；C 还没积累过则返回 None。"""
        C = self.C[l]
        tr = C.diagonal(dim1=-2, dim2=-1).sum(-1)               # [H]
        if float(tr.max()) <= 0:
            return None
        Chat = C / tr.clamp_min(1e-30)[:, None, None]
        align = torch.einsum("hnd,hde,hne->hn", x, Chat, x)
        return -align / x.pow(2).sum(-1).clamp_min(1e-30)

    # ------------------------------------------------------------ 分数修正
    def _correction(self, score: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """score: [L,1,H,n] → 同形状的加性修正。"""
        L, _, H, n = score.shape
        pos = torch.arange(lo, hi, device=self.device)
        out = torch.zeros_like(score)
        sigma_glob = score.std().clamp_min(1e-6)                # 跨层跨头的全局尺度
        novs, feats = [], []
        for l in range(L):
            x = self._x(l, pos)
            feats.append(x)
            novs.append(self._novelty(l, x))
        if all(v is None for v in novs):
            return out                                          # 冷启动，无修正
        nrm = torch.cat([f.norm(dim=-1).flatten() for f in feats])
        self.norm_cv.append(float(nrm.std() / nrm.mean().clamp_min(1e-30)))

        # 跨头/层的组级 z-score 要一起算，所以先收集各头的平均 novelty
        gmean = torch.full((L, H), float("nan"), device=score.device)
        for l, nv in enumerate(novs):
            if nv is not None:
                gmean[l] = nv.mean(-1)
        ok = torch.isfinite(gmean)
        if self.beta_group != 0.0 and int(ok.sum()) > 1:
            mu, sd = gmean[ok].mean(), gmean[ok].std().clamp_min(1e-6)
            gz = torch.where(ok, (gmean - mu) / sd, torch.zeros_like(gmean))
        else:
            gz = torch.zeros_like(gmean)

        for l, nv in enumerate(novs):
            if nv is None:
                continue
            if self.shuffle:                                    # 随机对照：只打乱配对关系
                perm = torch.stack([torch.randperm(n, generator=self._gen)
                                    for _ in range(H)]).to(nv.device)
                nv = torch.gather(nv, 1, perm)
            z = (nv - nv.mean(-1, keepdim=True)) / nv.std(-1, keepdim=True).clamp_min(1e-6)
            sb = score[l, 0].std(-1, keepdim=True).clamp_min(1e-6)      # [H,1] 逐头尺度
            corr = self.beta * sb * z                                    # 头内重排
            if self.beta_group != 0.0:
                corr = corr + self.beta_group * sigma_glob * gz[l][:, None]  # 跨头分配
            out[l, 0] = corr
        self.corr_std.append(float(out.std()))
        return out

    # ------------------------------------------------------------ 状态更新
    @torch.no_grad()
    def _update_C(self, lo: int, hi: int, valid_new: torch.Tensor):
        """valid_new: [L,H,n]，本 chunk 的保留掩码。"""
        pos = torch.arange(lo, hi, device=self.device)
        for l in range(self.n_layers):
            # **衰减必须无条件执行**，否则某层本 chunk 无样本时就不衰减，
            # 实际递推变成 C_t = C_{t-1}，与 C_t = ρC_{t-1} + XᵀX 不符
            if self.rho != 1.0:
                self.C[l].mul_(self.rho)
            m = valid_new[l]
            take = m if self.src == "retained" else ~m
            if not bool(take.any()):
                continue
            # **每层只算一次特征**：此前是在 head 循环里调 `_x(...)[h]`，
            # 等于把 [H,n,d] 的 float 拷贝重算 H 遍（n=16000/H=4 时每 chunk 约 3.7 GB 的
            # 无谓分配）。GPT 的评审没抓到这条。
            x_all = self._x(l, pos)                             # [H,n,d]
            for h in range(self.n_heads_kv):
                idx = take[h].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                x = x_all[h][idx]                               # [n_sel,d]
                self.C[l][h].addmm_(x.T, x)
                if l == 0 and h == 0:
                    self.n_absorbed += int(idx.numel())

    # ------------------------------------------------------------------ 主体
    def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
        """复刻 RetainCache.prune_chunk，只在 threshold 之前插入修正。

        **不能调 super() 再补救**——修正必须发生在阈值化之前，而父类把两步写在一起。
        """
        lo, hi = evict_range
        score = torch.stack(self.score, dim=0)[..., lo:hi]       # [L,1,H,n]
        if self.active:
            score = score + self._correction(score, lo, hi)
        valid, thres = self.threshold(score, ratio, level)        # [L,H,n]

        if self.active:
            self._update_C(lo, hi, valid)

        if self.valid is None:
            self.valid = valid
        else:
            self.valid = torch.cat([self.valid, valid], dim=-1)
        r_ = self.valid.float().mean().item()
        self.flatten = True
        return thres, r_
