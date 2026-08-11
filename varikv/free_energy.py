"""吸收自由能 F_i = D_i + λ·KL_i —— 一个标量，两个决策（§11.1-11.2）。

    F_i 高 = 把 KV_i 压进记忆的代价大（失真大 或 信息新）→ 留精确
    F_i 低 = 冗余、可压                                   → 降级进记忆

    决策 A（驱逐）：按 F_i 排序，留 top-B 精确，其余降级
    决策 B（写入）：降级后的写入率由**同一个** KL_i 导出（见 memory.absorb）

两项的含义：
    D_i  失真项。**必须在注意力输出空间**（缺口①）：衡量把 KV_i 换成它的记忆重建
         之后，未来 query 的注意力输出会差多少。注意是对**未来 query 分布**取期望，
         不是用已实现的注意力 —— 后者会让方法塌回 H2O/SnapKV（§11.3 退化表第二行）。
    KL_i 惊讶项。KV_i 相对当前记忆混合先验的信息增益。去掉它就塌回 Expected Attention
         （退化表第一行）。这一项正是本方法区别于纯打分方法的地方。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import FreeEnergyConfig


class QueryStatistics:
    """未来 query 分布 p(q) 的在线估计（对角高斯）。

    用 EMA 维护已见 query 的均值与方差，作为「未来 query 长什么样」的代理。
    这是 Expected-Attention 式的做法：把重要性定义在 q 的分布上而非某几个已实现的 q，
    因此对尚未到来的查询有泛化能力。
    """

    def __init__(self, cfg: FreeEnergyConfig):
        self.cfg = cfg
        self.mean: Optional[torch.Tensor] = None   # [G, d_head]
        self.var: Optional[torch.Tensor] = None
        self._initialized = False

    def reset(self):
        self.mean = None
        self.var = None
        self._initialized = False

    @torch.no_grad()
    def update(self, q: torch.Tensor):
        """q [B, G, n_q_per_kv, T, d_head] 或 [B, G, T, d_head]"""
        if q.dim() == 5:
            q = q.flatten(2, 3)
        q = q.float()
        m = q.mean(dim=(0, 2))                      # [G, d_head]
        v = q.var(dim=(0, 2), unbiased=False)
        if not self._initialized:
            self.mean, self.var = m, v
            self._initialized = True
        else:
            r = self.cfg.query_stat_momentum
            self.mean = r * self.mean + (1 - r) * m
            self.var = r * self.var + (1 - r) * v

    def expected_attention(self, k: torch.Tensor) -> torch.Tensor:
        """E_{q~p(q)}[ softmax logit ] 的对数正态近似 → 归一化的期望注意力 ā_i。

        logit_i = q·k_i/√d 在 q~N(μ,Σ) 下服从 N(m_i, s_i²)：
            m_i = μ·k_i/√d,   s_i² = k_iᵀΣk_i/d
        对数正态给出 E[exp(logit_i)] = exp(m_i + s_i²/2)，
        再对 i 归一化即得期望注意力权重。

        k [B, G, N, d_head] → ā [B, G, N]
        """
        d = k.shape[-1]
        scale = 1.0 / (d ** 0.5)
        kf = k.float()
        if not self._initialized:
            # 还没见过任何 query：退化成均匀分布（不引入偏好）
            return torch.full(k.shape[:-1], 1.0 / max(k.shape[-2], 1),
                              device=k.device, dtype=torch.float32)
        m = torch.einsum("bgnd,gd->bgn", kf, self.mean) * scale
        s2 = torch.einsum("bgnd,gd->bgn", kf.pow(2), self.var) * (scale ** 2)
        logits = m + 0.5 * s2
        return torch.softmax(logits, dim=-1)


def _zscore(x: torch.Tensor) -> torch.Tensor:
    """chunk 内标准化（沿 token 维）。驱逐只用排序，故单调变换不损失信息。

    unbiased=False：默认的无偏 std 在 n=1 时返回 NaN，clamp_min 挡不住
    （NaN.clamp_min 还是 NaN）。见 memory.py 里 kl_z 处的详细说明。
    """
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(
        dim=-1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def distortion_term(
    v: torch.Tensor, v_hat: torch.Tensor, expected_attn: torch.Tensor
) -> torch.Tensor:
    """D_i = E_q[a_i(q)²] · ‖v_i − v̂_i‖²  —— 注意力输出空间的失真（缺口①）。

    推导：注意力输出 o = Σ_j a_j v_j。把 KV_i 换成记忆重建后，
    输出扰动的主导项是 a_i·(v_i − v̂_i)，其平方范数即上式。
    （k 的扰动只通过 softmax 二阶影响输出，量级更低，这里略去。）

    v/v_hat [B,G,N,d_head]，expected_attn [B,G,N] → D [B,G,N]
    """
    err = (v.float() - v_hat.float()).pow(2).sum(-1)
    return expected_attn.pow(2) * err


class FreeEnergyPredictor(nn.Module):
    """摊销 F 预测器（§11.4.1）。

    精确 F_i 要对每个 token 跑一遍 encoder→decoder 重建、再算期望注意力，
    在长上下文的每个 chunk 上都做太贵。这里训一个轻量网络直接预测 F_i ——
    完全类比 FastKVzip 用 gate 蒸馏 KVzip 的重建注意力分数（<1 H100 时）。

    额外的好处：驱逐是离散 top-k、不可微，而预测器靠**蒸馏**训练，
    梯度不必穿过驱逐操作，整条链路依然可训。
    """

    def __init__(self, d_kv: int, cfg: FreeEnergyConfig, d_latent: int = 0):
        super().__init__()
        h = cfg.predictor_hidden
        # 输入：concat(k, v) + 两个标量（期望注意力、相对位置）+ 记忆摘要。
        #
        # 记忆摘要是必要的：预测目标 F 含 KL_i，而 KL_i 依赖当前槽 (μ_k, σ_k²)。
        # 不给记忆信息，预测器只能学到「平均意义」的 KL，记忆演化时必然失准。
        # 但也不能给精确信息 —— 那要走 encode，而预测器存在的意义正是省掉 encode。
        # 折中：喂槽的低维池化统计（μ 的均值、logvar 的均值），成本接近零。
        self.d_summary = 2 * d_latent
        self.net = nn.Sequential(
            nn.Linear(d_kv + 2 + self.d_summary, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1),
        )

    def forward(self, evidence, expected_attn, rel_pos, mem_summary=None):
        """evidence [B,G,N,d_kv]，expected_attn/rel_pos [B,G,N]，
        mem_summary [B,G,2*d_latent] → F̂ [B,G,N]"""
        feats = [
            evidence,
            expected_attn.unsqueeze(-1).to(evidence.dtype),
            rel_pos.unsqueeze(-1).to(evidence.dtype),
        ]
        if self.d_summary:
            if mem_summary is None:
                mem_summary = evidence.new_zeros(
                    evidence.shape[0], evidence.shape[1], self.d_summary
                )
            n = evidence.shape[2]
            feats.append(mem_summary.unsqueeze(2).expand(-1, -1, n, -1).to(evidence.dtype))
        return self.net(torch.cat(feats, dim=-1)).squeeze(-1)


class FreeEnergyScorer(nn.Module):
    """把记忆模块和 query 统计组装成 F_i 的计算入口。"""

    def __init__(self, d_kv: int, cfg: FreeEnergyConfig, d_latent: int = 0):
        super().__init__()
        self.cfg = cfg
        self.query_stats = QueryStatistics(cfg)
        self.predictor = (
            FreeEnergyPredictor(d_kv, cfg, d_latent)
            if cfg.use_amortized_predictor else None
        )
        self._steps = 0
        # E‖v‖² 的 running 估计，用于把失真相对化。这是**数据集级**的尺度，
        # 不随 chunk 组成变化，因此不会像 z-score 那样破坏 F_i 的绝对语义。
        self.register_buffer("v_scale", torch.ones(1))
        self.register_buffer("v_init", torch.zeros(1))
        # 两项各自的 running 标准差，用于把它们放到可比的离散度上
        self.register_buffer("d_std", torch.ones(1))
        self.register_buffer("kl_std", torch.ones(1))
        self.register_buffer("std_init", torch.zeros(1))

    def reset(self):
        self.query_stats.reset()

    def exact(self, memory, k, v, rel_pos) -> Tuple[torch.Tensor, dict]:
        """精确 F_i = D_i + λ·KL_i。用作驱逐依据（warmup 期）和预测器的蒸馏标签。

        k/v [B,G,N,d_head] → F [B,G,N]
        """
        evidence = torch.cat([k, v], dim=-1)                 # [B,G,N,d_kv]

        # 失真项：把证据压进记忆再读出，看注意力输出差多少
        v_hat = memory.reconstruct(evidence)[..., v.shape[-1]:]
        a_bar = self.query_stats.expected_attention(k)
        D = distortion_term(v, v_hat, a_bar)

        # 惊讶项：相对当前记忆混合先验的信息增益（保留混合，逐槽 KL 加权）
        mu_q, logvar_q = memory.encode(evidence)
        w = memory.get_prior(mu_q)
        _, kl = memory.kl_to_mixture(mu_q, logvar_q, w)

        # 把两项归一化到可比尺度后相加。不做归一化的话，实测 D≈4e-3 而 KL≈1.6e3，
        # 相差 3.8e5 倍，F 会退化成纯 KL 打分，§11.1「一个标量统一失真与惊讶」名存实亡。
        if self.cfg.f_normalize == "running":
            with torch.no_grad():
                vs = v.float().pow(2).sum(-1).mean()
                if self.v_init.item() == 0:
                    self.v_scale.fill_(vs.clamp_min(1e-6)); self.v_init.fill_(1)
                else:
                    r = self.cfg.v_scale_momentum
                    self.v_scale.mul_(r).add_((1 - r) * vs)
            n = k.shape[-2]
            # D_raw = ā²‖Δv‖²。乘 N² 换成 (N·ā)²（N·ā 的均值恒为 1，故与序列长度无关），
            # 再除以 E‖v‖² 得到无量纲的相对失真。
            D_n = D * (n ** 2) / self.v_scale.clamp_min(1e-6)
            KL_n = kl.float() / memory.d_z          # per-dim，与 d_latent 解耦
            with torch.no_grad():
                ds = D_n.std(unbiased=False).clamp_min(1e-6)
                ks = KL_n.std(unbiased=False).clamp_min(1e-6)
                if self.std_init.item() == 0:
                    self.d_std.fill_(ds); self.kl_std.fill_(ks); self.std_init.fill_(1)
                else:
                    r = self.cfg.v_scale_momentum
                    self.d_std.mul_(r).add_((1 - r) * ds)
                    self.kl_std.mul_(r).add_((1 - r) * ks)
            F = D_n / self.d_std + self.cfg.lam * KL_n / self.kl_std
        else:
            F = _zscore(D) + self.cfg.lam * _zscore(kl.float())
        return F, {"D": D, "KL": kl, "expected_attn": a_bar, "evidence": evidence}

    @staticmethod
    def memory_summary(memory):
        """槽的低维池化统计 [B,G,2*d_z]，让预测器知道记忆当前的状态与确定度。"""
        if memory is None or memory.mu is None:
            return None
        return torch.cat([memory.mu.mean(dim=-2), memory.logvar.mean(dim=-2)], dim=-1)

    def predicted(self, evidence, expected_attn, rel_pos, mem_summary=None):
        return self.predictor(evidence, expected_attn, rel_pos, mem_summary)

    def score(self, memory, k, v, rel_pos, force_exact: bool = False):
        """驱逐时调用。返回 (F_used, aux)。

        训练：算精确 F（既用于早期驱逐，也作预测器的蒸馏标签）+ 跑预测器。
        推理：**只跑预测器**，完全跳过精确 F 的重建与期望注意力计算 ——
              这才是「摊销」的意义所在（§11.4.1）。若推理时仍算精确 F，
              轻量性就无从谈起，HANDOFF 红线 2 的效率故事直接塌掉。
        """
        if self.predictor is not None and not self.training and not force_exact:
            evidence = torch.cat([k, v], dim=-1)
            a_bar = self.query_stats.expected_attention(k)
            summ = self.memory_summary(memory)
            return self.predicted(evidence, a_bar, rel_pos, summ), {}

        want_exact = (
            force_exact
            or self.predictor is None
            or self._steps < self.cfg.exact_f_warmup_steps
        )
        F_exact, aux = self.exact(memory, k, v, rel_pos)
        if self.predictor is not None:
            F_pred = self.predicted(
                aux["evidence"], aux["expected_attn"], rel_pos,
                self.memory_summary(memory),
            )
            aux["F_exact"] = F_exact
            aux["F_pred"] = F_pred

            # 蒸馏目标 = **chunk 内的归一化秩**，不是 F 的值，也不是 z-score。
            #
            # 动机同前：驱逐只用 F 的排序，而 F 的绝对尺度随记忆状态剧烈漂移
            # （记忆还是初值时 μ_q≈μ_k、F≈0，吸收后 logvar 收缩、F 涨到 1e2）。
            # 所以目标必须是 chunk 内的相对量。
            #
            # 但 z-score 不够（2026-08-07 实测，这是自由能驱逐失效的根因）：
            # 它只修尺度、不修**形状**，而 chunk 内的 F 是灾难性重尾 ——
            # 96.4% 的 token |z|<0.1，99.4% <0.5，而 99.9% 分位是 27.0，
            # 峰度 702（正态=3）。0.16% 的 token 扛着全部方差。
            # 再叠加 Huber 对离群值的梯度截断，「恒输出 0」几乎就是最优解：
            # 实测 std(F_pred)=0.047 而 std(target)=1.0，
            # predictor_loss 0.0419 vs 恒输出 0 的 0.0421 —— 只赢 0.5%。
            # 预测器塌成常数 ⇒ 排序全是噪声 ⇒ ρ(pred,exact) 实测 **−0.28**
            # （连训练长度上都是负的），驱逐拿到的是反向信号。
            #
            # 秩变换是对症的：它同样是 chunk 内的单调变换（排序信息一字不丢），
            # 但天然有界、均匀、无离群值，重尾问题从根上消失。
            with torch.no_grad():
                f = F_exact.detach().float()
                n = f.shape[-1]
                order = f.argsort(dim=-1)
                ranks = torch.empty_like(f)
                ranks.scatter_(
                    -1, order,
                    torch.arange(n, device=f.device, dtype=f.dtype).expand_as(f),
                )
                target = ranks / max(n - 1, 1) * 2.0 - 1.0        # → 均匀分布于 [-1,1]
            # 目标已有界于 [-1,1]，Huber 在此区间内基本等同 MSE；保留它只为
            # 防预测器早期输出跑飞时的梯度爆炸。
            aux["predictor_loss"] = torch.nn.functional.smooth_l1_loss(
                F_pred.float(), target
            )
        # 注意两条路径的 F 处在不同尺度（精确值 vs 标准化值），但驱逐只用排序，
        # 单调变换不影响结果，所以 warmup 前后切换是安全的。
        return (F_exact if want_exact else aux["F_pred"].detach()), aux

    def step(self):
        self._steps += 1
