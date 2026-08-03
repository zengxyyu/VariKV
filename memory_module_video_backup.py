"""
基于变分自由能的视频记忆模块
理论基础：自由能原理（Friston 2010）+ 变分推断（VAE，Kingma 2014）

核心思想：
  记忆 = 概率分布（均值 + 方差），而非固定向量
  更新原则：KL 散度大（新信息多）才更新，KL 小（已经知道了）就跳过
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreeEnergyMemory(nn.Module):
    """
    变分自由能记忆模块

    参数：
        d_model       : 特征维度（InternVL3-8B 为 4096）
        num_slots     : 记忆槽数量 K（默认 16）
        tokens_per_slot: 每个槽读出时展开成多少个 token（默认 32）
    """

    def __init__(self, d_model: int = 4096, num_slots: int = 16, tokens_per_slot: int = 32):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.tokens_per_slot = tokens_per_slot

        # ── 记忆槽（概率分布形式）────────────────────────────────────────
        # mu     : 每个槽的均值，代表"记住了什么"
        # logvar : 每个槽的对数方差，代表"有多确定"
        #          logvar 小（比如 -4）→ 方差小 → 很确定，不容易被覆盖
        #          logvar 大（比如  0）→ 方差大 → 不确定，容易被更新
        #
        # 关键区分（修正原 bug）：
        #   记忆是「递推状态」，不是「参数」。
        #   - 可学习的初始状态 mem_mu_init / mem_logvar_init 由优化器训练（每个新视频从这里出发）
        #   - 运行时状态 mem_mu / mem_logvar 是普通张量，在前向过程中被 update_memory 递推更新，
        #     梯度可沿递推链回传 → 记忆真正端到端可训（这是本方法最核心的卖点）
        self.mem_mu_init     = nn.Parameter(torch.zeros(num_slots, d_model))
        self.mem_logvar_init = nn.Parameter(torch.full((num_slots, d_model), -2.0))
        # 初始化：logvar=-2 → 标准差≈0.37，初始不确定性适中

        # 运行时状态（在 reset() 中从可学习初始状态克隆得到，非 nn.Parameter）
        self.mem_mu     = None
        self.mem_logvar = None
        self.reset()

        # ── 压缩器：N 个视觉 token → 1 个证据向量 ──────────────────────
        # 用交叉注意力，让一个可学习的 query token 去"采访"所有视觉 token
        self.compress_attn  = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.compress_query = nn.Parameter(torch.randn(1, d_model))
        # compress_query 是可学习的，训练后会学会"问什么问题"

        # ── 识别网络（编码器）：证据 → 后验分布参数 ─────────────────────
        # 对应公式：q_φ(z|e_t) = N(μ_φ(e_t), σ²_φ(e_t))
        self.enc_mu     = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.enc_logvar = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

        # ── 解码器：用于计算重建损失（似然项）─────────────────────────────
        # 对应公式：p(e_t|z) 用 MSE 近似
        self.decoder = nn.Linear(d_model, d_model)

        # ── 读取器：把槽的均值向量展开成多个 token ─────────────────────
        # 每个槽 d_model 维 → tokens_per_slot × d_model 维 → reshape
        self.slot_to_tokens = nn.Linear(d_model, tokens_per_slot * d_model)

    # ══════════════════════════════════════════════════════════════════════
    # 步骤 1：压缩
    # ══════════════════════════════════════════════════════════════════════

    def compress(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        """
        把 N 个视觉 token 压缩成 1 个证据向量 e_t

        输入：vit_embeds  [B, N, d_model]   N 个视觉 token
        输出：            [B, d_model]       1 个压缩向量
        """
        B = vit_embeds.shape[0]
        # compress_query 扩展到 batch 维度
        q = self.compress_query.unsqueeze(0).expand(B, 1, -1)  # [B, 1, d]
        # 用 q 去 attend vit_embeds，提炼关键信息
        out, _ = self.compress_attn(q, vit_embeds, vit_embeds)  # [B, 1, d]
        return out.squeeze(1)  # [B, d]

    # ══════════════════════════════════════════════════════════════════════
    # 步骤 2：先验计算
    # ══════════════════════════════════════════════════════════════════════

    def get_prior(self, e_t: torch.Tensor):
        """
        根据当前证据 e_t，在记忆槽中找最相关的作为先验

        理论：p(z | M_{t-1}) = Σ_k w_k · N(μ_k, σ²_k)
              w_k 由 e_t 和 μ_k 的余弦相似度决定

        输入：e_t  [B, d]
        输出：mu_prior    [B, d]
              logvar_prior [B, d]
              weights      [B, K]  （每个槽的权重，用于后续更新）
        """
        # 计算 e_t 和每个记忆槽均值的余弦相似度
        # mem_mu: [K, d]，e_t: [B, d]
        sim = F.cosine_similarity(
            e_t.unsqueeze(1),            # [B, 1, d]
            self.mem_mu.unsqueeze(0),    # [1, K, d]
            dim=-1
        )  # [B, K]

        # temperature=3：让权重更集中（相关性高的槽权重更大）
        weights = F.softmax(sim * 3.0, dim=-1)  # [B, K]

        # 加权混合先验
        # einsum: (B,K) × (K,d) → (B,d)
        mu_prior     = torch.einsum('bk,kd->bd', weights, self.mem_mu)
        logvar_prior = torch.einsum('bk,kd->bd', weights, self.mem_logvar)

        return mu_prior, logvar_prior, weights

    # ══════════════════════════════════════════════════════════════════════
    # 步骤 3：后验推断 + 自由能计算
    # ══════════════════════════════════════════════════════════════════════

    def compute_free_energy(self, e_t, mu_prior, logvar_prior):
        """
        计算变分自由能 F = KL[q(z|e_t) || p(z|M)] - E_q[log p(e_t|z)]

        KL 项：衡量新证据和旧记忆的差距（惊讶度）
        重建项：衡量记忆对新证据的解释能力

        输入：
            e_t          [B, d]  当前证据
            mu_prior     [B, d]  先验均值
            logvar_prior [B, d]  先验对数方差
        输出：
            free_energy  标量   总自由能（用于辅助训练损失）
            mu_q         [B, d] 后验均值
            logvar_q     [B, d] 后验对数方差
            z            [B, d] 采样的潜变量
            kl           标量   KL 散度（用于决定更新幅度）
        """
        # ── 识别网络：q(z|e_t) = N(μ_q, σ²_q) ──────────────────────────
        mu_q     = self.enc_mu(e_t)                   # [B, d]
        logvar_q = self.enc_logvar(e_t).clamp(-4, 2)  # [B, d]，clamp 防止数值爆炸

        # ── 重参数化采样：z = μ_q + σ_q ⊙ ε，ε ~ N(0,I) ────────────────
        # 这使得梯度可以通过采样操作反向传播（VAE 的核心技巧）
        if self.training:
            std = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std)
            z = mu_q + std * eps
        else:
            z = mu_q  # 推理时直接用均值，更稳定

        # ── KL 散度：KL[q(z|e_t) || p(z|M)] ────────────────────────────
        # 公式（对角高斯的解析解）：
        # KL = 0.5 * Σ_i [ σ²_q,i/σ²_p,i + (μ_q,i-μ_p,i)²/σ²_p,i - 1 + log(σ²_p,i/σ²_q,i) ]
        # 用 logvar 改写（避免 exp 溢出）：
        # KL = 0.5 * Σ_i [ exp(logvar_q-logvar_p) + (μ_q-μ_p)²/exp(logvar_p) - 1 + logvar_p - logvar_q ]
        sigma_p_sq = logvar_prior.exp().clamp(min=1e-6)  # 防止除以零
        kl = 0.5 * (
            logvar_q.exp() / sigma_p_sq                # σ²_q / σ²_p
            + (mu_q - mu_prior).pow(2) / sigma_p_sq   # (μ_q-μ_p)² / σ²_p
            - 1                                         # -1
            + logvar_prior - logvar_q                  # log(σ²_p / σ²_q)
        ).sum(dim=-1).mean()  # 对维度 d 求和，对 batch 求均值 → 标量

        # ── 重建损失：-E_q[log p(e_t|z)] ─────────────────────────────────
        # 用 MSE 近似（假设 p(e_t|z) 是各向同性高斯）
        e_recon    = self.decoder(z)
        recon_loss = F.mse_loss(e_recon, e_t.detach())

        # ── 自由能 = KL + 重建损失 ────────────────────────────────────────
        free_energy = kl + recon_loss

        return free_energy, mu_q, logvar_q, z, kl

    # ══════════════════════════════════════════════════════════════════════
    # 步骤 4：记忆更新
    # ══════════════════════════════════════════════════════════════════════

    def update_memory(self, mu_q, logvar_q, weights, kl):
        """
        用后验更新记忆槽，更新幅度由 KL 自适应决定

        更新公式：
            η_k = σ(α · KL · w_k - β)   ← 自适应更新率
            μ_k ← (1-η_k)·μ_k + η_k·μ_q
            logσ²_k ← (1-η_k)·logσ²_k + η_k·logσ²_q

        直觉：
            KL 大且 w_k 大 → η_k 大 → 这个槽大幅更新
            KL 小或 w_k 小 → η_k 小 → 这个槽基本不变

        输入：
            mu_q     [B, d]  后验均值
            logvar_q [B, d]  后验对数方差
            weights  [B, K]  每个槽的相关性权重
            kl       标量    KL 散度值
        """
        # 对 batch 取均值（推理时 B=1，训练时 B>1）
        # 注意：不再 detach —— 写入的「内容」(mu_q/logvar_q) 必须带梯度，
        #      这样识别网络/压缩器才能从 LM 损失经由记忆递推链学到东西。
        mu_q_mean     = mu_q.mean(0)      # [d]
        logvar_q_mean = logvar_q.mean(0)  # [d]
        w = weights.mean(0)                # [K]

        # 自适应更新率：η_k = sigmoid(2·KL·w_k - 2)
        # α=2, β=2 → KL·w_k > 1 时才明显更新（防止噪声覆盖记忆）
        # 设计选择：对 KL 做 stop-gradient，把 η 当作「控制信号」（决定写入多少），
        #          而写入内容仍可回传。若想让梯度也流过门控幅度，去掉下面的 .detach() 即可。
        kl_val = kl.detach()
        eta = torch.sigmoid(2.0 * kl_val * w - 2.0)  # [K]，范围 (0,1)

        # 更新每个槽（普通张量运算，非 .data 赋值 → 梯度可沿递推链回传）
        # eta: [K] → [K,1] 广播到 [K,d]
        eta_expanded = eta.unsqueeze(-1)  # [K, 1]

        self.mem_mu = (
            (1 - eta_expanded) * self.mem_mu
            + eta_expanded * mu_q_mean.unsqueeze(0)   # [1,d] 广播
        )
        self.mem_logvar = (
            (1 - eta_expanded) * self.mem_logvar
            + eta_expanded * logvar_q_mean.unsqueeze(0)
        )

    # ══════════════════════════════════════════════════════════════════════
    # 步骤 5：读取记忆
    # ══════════════════════════════════════════════════════════════════════

    def read(self) -> torch.Tensor:
        """
        把所有记忆槽展开成 token 序列，送给 LLM

        设计：不确定性高（logvar 大）的槽权重低，
             确定的槽（logvar 小）权重高
        （让 LLM 更关注可信的记忆）

        输出：[num_slots * tokens_per_slot, d_model]
        """
        # 确定性 = -logvar 的均值（logvar 越小越确定）
        certainty = -self.mem_logvar.mean(dim=-1)   # [K]
        slot_weights = F.softmax(certainty, dim=0)  # [K]，加权用

        tokens_list = []
        for k in range(self.num_slots):
            # 每个槽：d_model 维向量 → tokens_per_slot 个 token
            slot_tokens = self.slot_to_tokens(self.mem_mu[k])          # [T*d]
            slot_tokens = slot_tokens.view(self.tokens_per_slot, self.d_model)  # [T, d]
            # 按确定性加权（确定的槽给 LLM 更强的信号）
            tokens_list.append(slot_tokens * slot_weights[k])

        return torch.cat(tokens_list, dim=0)  # [K*T, d]

    def reset(self):
        """处理新视频前，把运行状态重置为可学习的初始状态。

        用 clone() 得到新张量：既让首段更新从可学习初始状态出发（梯度可回传到
        mem_*_init，从而训练"初始记忆"），又不把递推更新写回初始参数本身。
        """
        self.mem_mu     = self.mem_mu_init.clone()
        self.mem_logvar = self.mem_logvar_init.clone()

    # ══════════════════════════════════════════════════════════════════════
    # 主接口
    # ══════════════════════════════════════════════════════════════════════

    def forward(self, vit_embeds: torch.Tensor):
        """
        处理一个视频段，更新记忆并返回记忆 token

        输入：vit_embeds  [B, N, d_model]  当前视频段的视觉 token
        输出：
            memory_tokens  [K*T, d_model]  记忆 token（送给 LLM）
            free_energy    标量            辅助训练损失
        """
        # 步骤 1：压缩 N 个视觉 token → 1 个证据向量
        e_t = self.compress(vit_embeds)                          # [B, d]

        # 步骤 2：从记忆槽计算先验
        mu_p, logvar_p, weights = self.get_prior(e_t)           # [B,d], [B,d], [B,K]

        # 步骤 3：计算自由能和后验
        F_val, mu_q, logvar_q, z, kl = self.compute_free_energy(e_t, mu_p, logvar_p)

        # 步骤 4：更新记忆
        self.update_memory(mu_q, logvar_q, weights, kl)

        # 步骤 5：读取记忆 token
        memory_tokens = self.read()                              # [K*T, d]

        return memory_tokens, F_val


# ══════════════════════════════════════════════════════════════════════════
# 如何集成到 InternVL3
# ══════════════════════════════════════════════════════════════════════════

"""
在 internvl_chat.py 的 __init__ 里加：

    self.memory = FreeEnergyMemory(
        d_model=self.config.llm_config.hidden_size,  # 4096
        num_slots=16,
        tokens_per_slot=32
    )

在 forward() 里，MLP Projector 输出之后加：

    # vit_embeds: [B, N, 4096]，已经过 MLP Projector

    if is_history_frame:
        # 历史帧：写入记忆
        _, free_energy = self.memory(vit_embeds)
        aux_loss = 0.01 * free_energy  # 辅助损失
    else:
        # 当前帧：读取记忆并拼接
        memory_tokens = self.memory.read()  # [K*T, d]
        B = vit_embeds.shape[0]
        mem = memory_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, K*T, d]
        vit_embeds = torch.cat([mem, vit_embeds], dim=1)    # [B, K*T+N, d]

    # 后续代码不变，vit_embeds 送进 LLM

训练总损失：
    total_loss = lm_loss + aux_loss
"""


# ══════════════════════════════════════════════════════════════════════════
# 快速测试（直接运行此文件）
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(42)

    # 模拟 InternVL3-8B 的参数
    d_model        = 4096
    num_slots      = 16
    tokens_per_slot = 32
    B              = 2   # batch size
    N              = 256 # 每帧的视觉 token 数

    model = FreeEnergyMemory(d_model, num_slots, tokens_per_slot)
    print(f'参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')

    # 模拟处理 10 个历史视频段
    print('\n处理历史段...')
    for t in range(10):
        # 模拟一段视频的视觉 token
        vit_embeds = torch.randn(B, N, d_model)
        memory_tokens, free_energy = model(vit_embeds)
        print(f'  段 {t+1}: free_energy={free_energy.item():.4f}, '
              f'memory_tokens shape={memory_tokens.shape}')

    # 最终读取记忆
    final_memory = model.read()
    print(f'\n最终记忆 token 形状：{final_memory.shape}')
    print(f'期望：[{num_slots * tokens_per_slot}, {d_model}]')
    # → [512, 4096]，这些 token 将被送给 LLM
