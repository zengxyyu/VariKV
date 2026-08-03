"""MomentKV 式的矩统计记忆 —— Stage-1 的**强** baseline。

为什么需要它（2026-08-03 文献检索的结论）：
MomentKV (arXiv 2606.01563) 已经 training-free 地保留了被驱逐 token 集合的
「count + key mean + value mean + value-key 协方差」，并给出闭式一阶修正。
因此「分布式记忆打赢点均值记忆」这件事**不再构成贡献** —— 真实门槛是
「打赢一个已经存了二阶矩、而且完全不用训练的方法」。
Stage-1 若只跟 point 档比，赢了也说明不了问题。

与 DistributionalMemory 的关系（这是对照的关键）：
    point   均值，写入率是可学习标量                     ≈ IndexMem 缩影
    moment  均值 + 二阶矩 + 计数，**零可训练参数**        ≈ MomentKV      ← 本文件
    dist    latent 高斯信念，KL 门控 + 精度加权            VariKV
moment 与 dist 都持有二阶信息，差别在于：moment 是 KV 原空间的**频率派矩估计**、
写入无门控、读出不含不确定性；dist 是 latent 空间的**贝叶斯信念**，
写入由 surprise 门控、读出方差感知。这样「二阶矩」这个因素被两边共有，
实验的自变量就收敛到「贝叶斯信念 vs 矩统计」，而不是「有没有二阶信息」。

诚实标注：这是在本仓库架构下的**近似复现**，不是 MomentKV 原文。
原文的一阶修正项 `C·q/√d` 依赖当前 query，而本架构把记忆读出成静态 effective KV
拼进 cache，静态张量无法表达 query 依赖项。因此这里保留 count/均值/协方差的
统计，但读出时只用到零阶项 + 协方差的各向同性近似。
论文里做最终对比时应当直接跑官方实现，不能拿这个当作 MomentKV 的成绩。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig


class MomentMemory(nn.Module):
    """按 key 相似度把被驱逐 KV 聚到 K 个槽，每槽维护 count / 均值 / 二阶矩。

    **无任何可训练参数** —— training-free，与 MomentKV 的定位一致。
    接口与 DistributionalMemory 保持一致（reset / absorb / read / read_precision），
    以便 cache.py 用同一条代码路径驱动。
    """

    def __init__(self, d_kv: int, cfg: MemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = "moment"
        self.d_kv = d_kv
        self.d_head = d_kv // 2
        self.K = cfg.num_slots
        self.d_z = cfg.d_latent          # 仅为接口兼容，moment 不用 latent

        self.count: Optional[torch.Tensor] = None      # [B,G,K]
        self.k_mean: Optional[torch.Tensor] = None     # [B,G,K,d_head]
        self.v_mean: Optional[torch.Tensor] = None
        self.kv_m2: Optional[torch.Tensor] = None      # value-key 二阶矩（对角近似）
        self.pos: Optional[torch.Tensor] = None        # 位置质心（RoPE 重旋用）
        self._pos_tau: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ 状态

    def reset(self, batch: int, groups: int, device=None, dtype=None):
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
        self.count = z(batch, groups, self.K)
        # 槽的 key 锚点随机初始化并归一化：全零锚点会让相似度退化成常数、
        # 所有证据挤进同一个槽。
        g = torch.Generator(device="cpu").manual_seed(0)
        anchor = torch.randn(self.K, self.d_head, generator=g).to(device=device, dtype=dtype)
        self.k_mean = F.normalize(anchor, dim=-1).expand(batch, groups, self.K, self.d_head).clone()
        self.v_mean = z(batch, groups, self.K, self.d_head)
        self.kv_m2 = z(batch, groups, self.K, self.d_head)
        self.pos = torch.zeros(batch, groups, self.K, device=device, dtype=torch.float32)
        self._pos_tau = torch.full((batch, groups, self.K), 1e-4,
                                   device=device, dtype=torch.float32)

    def detach_state(self):
        for n in ("count", "k_mean", "v_mean", "kv_m2", "pos", "_pos_tau"):
            t = getattr(self, n)
            if t is not None:
                setattr(self, n, t.detach())

    # ------------------------------------------------------------------ 写入

    def absorb(self, evidence: torch.Tensor, expected_attn: Optional[torch.Tensor] = None,
               positions: Optional[torch.Tensor] = None):
        """把一批被驱逐的 KV 并入矩统计。

        无门控、无 surprise —— 这正是它作为 baseline 的意义：
        所有被驱逐的证据一视同仁地进入均值，靠的是频率派的计数平均。

        返回 (伪 kl, 伪 free_energy) 以兼容接口；两者都是常数 0，
        因为 moment 档没有变分目标可训。
        """
        d = self.d_head
        k, v = evidence[..., :d], evidence[..., d:]              # [B,G,N,d]

        # 按 key 与槽锚点的余弦相似度做硬分配（argmax）。
        # 用硬分配而非 softmax：矩统计的语义是「这批 token 的均值」，
        # 软分配会把同一个 token 摊进所有槽，计数失去意义。
        sim = torch.einsum("bgnd,bgkd->bgnk", F.normalize(k, dim=-1),
                           F.normalize(self.k_mean, dim=-1))
        assign = sim.argmax(-1)                                   # [B,G,N]
        onehot = F.one_hot(assign, self.K).to(k.dtype)            # [B,G,N,K]

        n_new = onehot.sum(dim=2)                                 # [B,G,K]
        sum_k = torch.einsum("bgnk,bgnd->bgkd", onehot, k)
        sum_v = torch.einsum("bgnk,bgnd->bgkd", onehot, v)
        sum_kv = torch.einsum("bgnk,bgnd->bgkd", onehot, k * v)   # value-key 二阶矩(对角)

        # 与 dist 档共用同一个遗忘因子，保证两者的时间尺度可比
        gamma = self.cfg.precision_decay
        c_old = gamma * self.count
        c_new = c_old + n_new
        w_old = (c_old / c_new.clamp_min(1e-8)).unsqueeze(-1)
        w_new = (1.0 / c_new.clamp_min(1e-8)).unsqueeze(-1)

        self.k_mean = w_old * self.k_mean + w_new * sum_k
        self.v_mean = w_old * self.v_mean + w_new * sum_v
        self.kv_m2 = w_old * self.kv_m2 + w_new * sum_kv
        self.count = c_new

        if positions is not None:
            pos_i = positions.float().view(1, 1, -1, 1)           # [1,1,N,1]
            num = (onehot.float() * pos_i).sum(dim=2)             # [B,G,K]
            tau_old = gamma * self._pos_tau
            tau_new = tau_old + n_new.float()
            self.pos = (tau_old * self.pos + num) / tau_new.clamp_min(1e-8)
            self._pos_tau = tau_new

        zero = evidence.new_zeros(())
        return evidence.new_zeros(evidence.shape[:3]), zero

    # ------------------------------------------------------------------ 读出

    def read(self) -> torch.Tensor:
        """输出 effective KV [B,G,K*T,d_kv]。

        零阶项是 (k_mean, v_mean)。协方差的一阶修正 `C·q/√d` 依赖当前 query，
        静态 KV 表达不了，这里用各向同性近似：把中心化的二阶矩
        Cov = E[kv] − E[k]E[v] 以一个小系数并入 value，
        使「key 与 value 有强相关」的槽读出更强的 value 响应。
        """
        cov = self.kv_m2 - self.k_mean * self.v_mean
        v_eff = self.v_mean + cov / (self.d_head ** 0.5)
        eff = torch.cat([self.k_mean, v_eff], dim=-1)             # [B,G,K,d_kv]
        T = self.cfg.tokens_per_slot
        if T > 1:
            eff = eff.repeat_interleave(T, dim=2)
        return eff

    def read_precision(self) -> torch.Tensor:
        """用计数代替精度：概括的 token 越多，该槽的 key 响应越强。

        与 dist 档的 read_precision 语义对齐（都返回一个「可信度」标量供上层缩放），
        差别在于 moment 用频率派的计数，dist 用贝叶斯精度 exp(−logvar)。
        """
        prec = self.count
        return prec.repeat_interleave(self.cfg.tokens_per_slot, dim=-1)
