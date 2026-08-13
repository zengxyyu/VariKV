"""ControlRetainCache —— 记忆只用来**修正驱逐分数**，不进 attention。

和 `memcache_retain.py` / `centroid.py` 的根本区别：那两个都把记忆变成 KV 送进 softmax，
这个**完全不碰 attention 输出**。记忆是一个旁路状态，只在压缩决策那一刻起作用：

    s'_i = s_i^base + β · σ_base(l,h) · z_{l,h}(novelty_i)
    valid = threshold(s', ratio, level)

由此得到三个结构性好处（不是靠 assert，是靠构造）：

1. **预算恒等匹配**。`threshold` 按 ratio 取全局 top-n，改分数只改「留哪些」，不改「留几个」。
   不再需要 `M·H·L` 那套核算（bug 4 折腾了两版才收敛）。
2. **满缓存参考干净**。ratio=1.0 不进 `prune_chunk`，记忆连被调用的机会都没有——
   learned-memory 那轮「空记忆仍然注入、污染 `full__`」的问题在这里不可能发生。
3. **β=0 必须与基线逐字相同**，这是验收第一条。

--------------------------------------------------------------------------------
覆盖度的定义（无需低秩、无需特征分解）

    C_t = ρ·C_{t-1} + Σ_{i∈S_t} x_i x_iᵀ         （S_t 由 --ctrl_src 决定）
    Ĉ   = C / tr(C)                              （迹归一 ⇒ 尺度无关）
    cov(x)     = xᵀĈx / ‖x‖²                     （x 与历史累积方向的对齐度）
    novelty(x) = −cov(x)                          （对齐得越少越"新"）

d=128 ⇒ 每个 (层,kv头) 的 C 是 16384 个 float，112 个头共 7.3 MB，相对 9.7 GB 的
KV cache 可忽略。`xᵀĈx` 是 [n,d]@[d,d] 再逐行点积，O(nd²)，一个 chunk 约 29 GFLOP，
在 H100 上是毫秒级。**低秩只有在讲部署 footprint 时才需要，正确性上不需要。**

--------------------------------------------------------------------------------
两个开关是**欠定量**，必须当实验变量而不是默认值：

- `--ctrl_src evicted`：C 累积**被驱逐**的 token。novelty 高 = 与被扔掉的东西不像。
- `--ctrl_src retained`：C 累积**被保留**的 token。novelty 高 = 补充了缓存里缺的方向。
  这两个语义相反，谁对是经验问题。配合 β 的正负共四种组合。

`--ctrl_shuffle`：把 novelty 在每个 (层,kv头) 内随机置换后再用。**这是必须跑的对照**——
stage-1 测过随机驱逐打败所有有原则的准则，不做这个对照就分不清「覆盖信号有用」
和「任何同幅度扰动都会改变结果」。
"""
from typing import Tuple

import torch

from .kvcache import RetainCache


class ControlRetainCache(RetainCache):
    def __init__(self, model, evict_range: Tuple[int, int], beta: float = 0.0,
                 rho: float = 1.0, src: str = "evicted", feat: str = "key",
                 rope_mode: str = "post", rope_inv_freq=None,
                 shuffle: bool = False, seed: int = 0):
        super().__init__(model, evict_range)
        self.beta, self.rho, self.src = float(beta), float(rho), src
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
        # C: [L][H,d,d]，float32 累积（bf16 累积二阶矩会掉精度，这一条在 bug 列表里有先例）
        self.C = [torch.zeros(self.n_heads_kv, d, d, dtype=torch.float32,
                              device=self.device) for _ in range(self.n_layers)]
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        self.n_seen = 0                      # 已累积的 token 数，用于诊断
        self._corr_std = []                  # 每次修正的实际幅度，供报告核对

    # ------------------------------------------------------------------ 特征
    def _x(self, layer_idx: int, pos: torch.Tensor) -> torch.Tensor:
        """→ [H, n, d_feat]，pos 是**缓存坐标**下的一维位置索引。"""
        k = self.key_cache[layer_idx][0][:, pos].float()        # [H,n,dh]
        if self.rope_mode == "inv":
            # 与 centroid.py 同一套逆旋转；post-RoPE 的 key 带位置相位，
            # 但质心那轮实测 post 反而更好，所以默认 post，inv 只作对照臂。
            from varikv.rope import cos_sin_at, inverse_rope
            cos, sin = cos_sin_at(self.inv_freq, pos.float(), dtype=k.dtype)
            k = inverse_rope(k, cos, sin)
        if self.feat == "key":
            return k
        v = self.value_cache[layer_idx][0][:, pos].float()
        return v if self.feat == "value" else torch.cat([k, v], dim=-1)

    # ------------------------------------------------------------ 分数修正
    def _correction(self, score: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """score: [L,1,H,n] → 同形状的加性修正。"""
        L, _, H, n = score.shape
        pos = torch.arange(lo, hi, device=self.device)
        out = torch.zeros_like(score)
        for l in range(L):
            C = self.C[l]                                        # [H,d,d]
            tr = C.diagonal(dim1=-2, dim2=-1).sum(-1)            # [H]
            if float(tr.max()) <= 0:
                continue                                         # 还没积累过，无修正
            x = self._x(l, pos)                                  # [H,n,d]
            Chat = C / tr.clamp_min(1e-30)[:, None, None]        # tr=1
            cov = torch.einsum("hnd,hde,hne->hn", x, Chat, x)
            nov = -cov / (x.pow(2).sum(-1).clamp_min(1e-30))     # [H,n]，尺度无关
            if self.shuffle:                                     # 随机对照：打乱 novelty
                perm = torch.stack([torch.randperm(n, generator=self._gen)
                                    for _ in range(H)]).to(nov.device)
                nov = torch.gather(nov, 1, perm)
            # **按 (层,kv头) 内 z-score，再乘基线分的 std**：level="pair" 是全局阈值化，
            # 不归一就只是在层间搬预算（"永远对所有层求和"那条教训的同源问题）。
            z = (nov - nov.mean(-1, keepdim=True)) / nov.std(-1, keepdim=True).clamp_min(1e-6)
            sb = score[l, 0].std(-1, keepdim=True).clamp_min(1e-6)   # [H,1]
            out[l, 0] = self.beta * sb * z
            self._corr_std.append(float((self.beta * sb * z).std()))
        return out

    # ------------------------------------------------------------ 覆盖更新
    @torch.no_grad()
    def _update_C(self, lo: int, hi: int, valid_new: torch.Tensor):
        """valid_new: [L,H,n]，本 chunk 的保留掩码。"""
        for l in range(self.n_layers):
            m = valid_new[l]                                     # [H,n] bool
            take = m if self.src == "retained" else ~m
            if not bool(take.any()):
                continue
            self.C[l].mul_(self.rho)
            for h in range(self.n_heads_kv):
                idx = take[h].nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                x = self._x(l, idx + lo)[h]                      # [n_sel,d]
                self.C[l][h].addmm_(x.T, x)                      # C += XᵀX
                if l == 0 and h == 0:
                    self.n_seen += int(idx.numel())

    # ------------------------------------------------------------------ 主体
    def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
        """复刻 RetainCache.prune_chunk，只在 threshold 之前插入修正。

        **不能调 super() 再补救**——修正必须发生在阈值化之前，而父类把两步写在一起。
        """
        lo, hi = evict_range
        score = torch.stack(self.score, dim=0)[..., lo:hi]       # [L,1,H,n]
        if self.beta != 0.0:
            score = score + self._correction(score, lo, hi)
        valid, thres = self.threshold(score, ratio, level)        # [L,H,n]

        if self.beta != 0.0 or self.src == "retained":
            self._update_C(lo, hi, valid)

        if self.valid is None:
            self.valid = valid
        else:
            self.valid = torch.cat([self.valid, valid], dim=-1)
        r_ = self.valid.float().mean().item()
        self.flatten = True
        return thres, r_
