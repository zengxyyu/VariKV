

"""分布式记忆：吸收被驱逐的 KV。

槽存的是 latent 空间的高斯信念 (μ_k, σ²_k)，由非线性 decoder 解成 effective (k̂, v̂)，
使未来的 query 能像注意普通 KV 一样注意它（§11.4.2）。

维度约定
    G  = 组数。每个 (layer, kv_head) 是一组，各组有独立的槽状态，但模块参数跨组共享
         （HANDOFF 红线 2：模块必须轻量，否则「加速推理」的故事塌掉）。
    d_kv = 2 * d_head，即 concat(k, v)。
    证据      evidence  [B, G, N, d_kv]
    槽状态    mu/logvar [B, G, K, d_z]
    读出      eff_kv    [B, G, K*T, d_kv]

point 与 dist 两档共享**完全相同**的结构和参数量，唯一差别是精度项是否携带信息：
    dist  : τ_old = exp(-logvar) 随槽变化，τ_obs = exp(-logvar_q) 随观测变化，η 由 KL 导出
    point : τ_old ≡ τ_obs ≡ 1（方差恒定、不更新），η 由可学习标量导出
两者走同一个精度加权更新公式，因此实验中唯一的自变量就是「方差是否有用」。
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MemoryConfig


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    """对角高斯 KL( N(mu_q, e^logvar_q) || N(mu_p, e^logvar_p) )，在最后一维求和。

    KL = 0.5 * Σ_d [ logvar_p - logvar_q + (σ_q² + (μ_q-μ_p)²)/σ_p² - 1 ]
    """
    var_ratio = torch.exp(logvar_q - logvar_p)
    mahalanobis = (mu_q - mu_p).pow(2) * torch.exp(-logvar_p)
    return 0.5 * (logvar_p - logvar_q + var_ratio + mahalanobis - 1.0).sum(-1)


class DistributionalMemory(nn.Module):
    def __init__(self, d_kv: int, cfg: MemoryConfig, mode: str = "dist"):
        super().__init__()
        assert mode in ("point", "dist"), mode
        self.cfg = cfg
        self.mode = mode
        self.d_kv = d_kv
        self.K = cfg.num_slots
        self.d_z = cfg.d_latent

        h = cfg.d_hidden

        # --- 识别网络 q_φ(z | e)：证据 → latent 后验 ---
        self.encoder = nn.Sequential(
            nn.Linear(d_kv, h), nn.GELU(), nn.Linear(h, h), nn.GELU()
        )
        self.to_mu = nn.Linear(h, self.d_z)
        self.to_logvar = nn.Linear(h, self.d_z)

        # --- 解码器：latent → effective (k̂, v̂) ---
        # 缺口③：logvar 作为显式输入，让「有多确定」直接参与读出计算。
        # point 档下这一路输入是常数，参数量不变 → 对照干净。
        dec_in = self.d_z + (self.d_z if cfg.logvar_into_decoder else 0)
        if cfg.nonlinear_decoder:
            # 缺口②：非线性。线性 decoder + 单高斯先验会让整个模型退化成
            # 线性高斯系统（卡尔曼闭式可解），摊销识别网络就失去存在理由。
            self.decoder = nn.Sequential(
                nn.Linear(dec_in, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                nn.Linear(h, d_kv * cfg.tokens_per_slot),
            )
        else:
            self.decoder = nn.Linear(dec_in, d_kv * cfg.tokens_per_slot)

        # --- 槽初值：是可学习参数；运行时的槽是从它 clone 出来的普通张量 ---
        # （2026-07-22 修的 bug：槽本身若是 nn.Parameter 且用 .data= 更新，
        #   会脱离 autograd，「端到端训练的自由能记忆」这个说法就是假的。）
        self.slot_mu_init = nn.Parameter(torch.randn(self.K, self.d_z) * 0.02)
        self.slot_logvar_init = nn.Parameter(
            torch.full((self.K, self.d_z), cfg.logvar_init)
        )

        # point 档的固定写入率（可学习标量，≈IndexMem 缩影）
        self.point_gate_logit = nn.Parameter(torch.zeros(1))

        # 初始化成「后验 ≈ 先验」：μ_q≈0 对齐 slot_mu_init≈0，logvar_q≈logvar_init。
        # 这样训练从 KL≈0 起步。否则随机初始化的 to_mu 输出量级 ~1，
        # 而 KL 里的 (μ_q−μ_k)²/σ_p² 会把它放大 d_z 倍，初始 loss 直接爆到 1e2~1e3。
        nn.init.normal_(self.to_mu.weight, std=0.02)
        nn.init.zeros_(self.to_mu.bias)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, cfg.logvar_init)

        self.mu: Optional[torch.Tensor] = None
        self.logvar: Optional[torch.Tensor] = None
        # 每个槽所概括 token 的位置质心 [B,G,K]。读出时按它把 pre-RoPE 的 k̂
        # 重旋回一个有效位置（EPL 2409.14364 的 UPL 最优解即「该组的中心」）。
        self.pos: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ 状态

    def reset(self, batch: int, groups: int, device=None, dtype=None):
        """每条序列开始时调用，把槽恢复成学到的初值。"""
        device = device or self.slot_mu_init.device
        dtype = dtype or self.slot_mu_init.dtype
        self.mu = (
            self.slot_mu_init.to(device=device, dtype=dtype)
            .expand(batch, groups, self.K, self.d_z).clone()
        )
        self.logvar = (
            self.slot_logvar_init.to(device=device, dtype=dtype)
            .expand(batch, groups, self.K, self.d_z).clone()
        )
        self.pos = torch.zeros(batch, groups, self.K, device=device, dtype=torch.float32)
        # 累计的标量精度，用于位置质心的加权平均（与 μ 的更新同权重）
        self._pos_tau = torch.full(
            (batch, groups, self.K), 1e-4, device=device, dtype=torch.float32
        )

    def detach_state(self):
        """截断 BPTT：切断跨 chunk 的梯度，但保留数值（§11.4.3 数值稳定）。"""
        if self.mu is not None:
            self.mu = self.mu.detach()
            self.logvar = self.logvar.detach()
        if self.pos is not None:
            self.pos = self.pos.detach()
            self._pos_tau = self._pos_tau.detach()

    def _clamp(self, logvar):
        return logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max)

    # ------------------------------------------------------------------ 推断

    def encode(self, evidence: torch.Tensor):
        """证据 → latent 后验 q_φ(z|e)。evidence [B,G,N,d_kv] → mu/logvar [B,G,N,d_z]"""
        h = self.encoder(evidence)
        mu_q = self.to_mu(h)
        logvar_q = self._clamp(self.to_logvar(h))
        if self.mode == "point":
            # 点记忆：后验方差不携带信息（恒等于先验初值），
            # 但张量形状与 dist 完全一致，保证参数量与计算图对称。
            logvar_q = torch.full_like(logvar_q, self.cfg.logvar_init)
        return mu_q, logvar_q

    def get_prior(self, mu_q: torch.Tensor):
        """混合先验的分配权重 w_k。

        缺口①：**保留混合**。这里只返回 w_k，KL 在 kl_to_mixture 里逐槽算完再按
        w_k 加权求和；绝不先把 K 个高斯平均成单个高斯再算 KL —— 那样会抹掉多峰性，
        使模型塌回共轭可解的单高斯，变分/摊销的理由随之消失。

        mu_q [B,G,N,d_z] → w [B,G,N,K]
        """
        # 余弦相似度：对 latent 尺度不敏感，比内积稳定
        q = F.normalize(mu_q, dim=-1)                      # [B,G,N,d_z]
        m = F.normalize(self.mu, dim=-1)                   # [B,G,K,d_z]
        sim = torch.einsum("bgnd,bgkd->bgnk", q, m)
        return torch.softmax(sim / self.cfg.prior_temperature, dim=-1)

    def kl_to_mixture(self, mu_q, logvar_q, w):
        """KL 到混合先验的上界：Σ_k w_k · KL(q ‖ N(μ_k, σ_k²))。

        这是 KL(q ‖ Σ_k w_k p_k) 的一个闭式上界（Jensen），
        既保留了混合结构，又避免了对混合求 KL 的不可解性。

        返回 kl_per_slot [B,G,N,K] 和 kl [B,G,N]
        """
        mu_q_e = mu_q.unsqueeze(-2)          # [B,G,N,1,d_z]
        logvar_q_e = logvar_q.unsqueeze(-2)
        mu_p = self.mu.unsqueeze(2)          # [B,G,1,K,d_z]
        logvar_p = self.logvar.unsqueeze(2)
        kl_per_slot = gaussian_kl(mu_q_e, logvar_q_e, mu_p, logvar_p)  # [B,G,N,K]
        kl = (w * kl_per_slot).sum(-1)
        return kl_per_slot, kl

    def decode(self, z, logvar):
        """latent → effective KV。z/logvar [.., d_z] → [.., T*d_kv]"""
        if self.cfg.logvar_into_decoder:
            z = torch.cat([z, logvar], dim=-1)
        return self.decoder(z)

    def reconstruct(self, evidence: torch.Tensor) -> torch.Tensor:
        """把证据过一遍 encode→decode，得到「压进记忆后再读出」的重建。

        这是自由能失真项 D_i 的原料：D_i 衡量的正是把 KV_i 换成它的记忆重建
        之后，注意力输出会差多少（见 free_energy.py）。
        """
        mu_q, logvar_q = self.encode(evidence)
        out = self.decode(mu_q, logvar_q)                  # [B,G,N,T*d_kv]
        # 只取第一个 token 的重建来对齐原 KV 的形状
        return out[..., : self.d_kv]

    # ------------------------------------------------------------------ 写入

    def absorb(self, evidence: torch.Tensor, expected_attn: Optional[torch.Tensor] = None,
               positions: Optional[torch.Tensor] = None):
        """把一批被驱逐的 KV 写入记忆（决策 B）。

        expected_attn [B,G,N]：期望注意力 ā_i。传入后，ELBO 的重建项会按 ā 加权，
        与 F_i 的失真项使用**同一个失真定义**（见下方 free_energy 处的说明）。
        positions [N]：被吸收 KV 的原始绝对位置。用来把槽的位置质心一并更新，
        供读出时重旋（见 varikv/rope.py 的说明）。证据本身必须已是 pre-RoPE。

        多观测的贝叶斯更新：独立观测的**精度可加**，因此一个 chunk 内被驱逐的
        N 个 KV 可以并行聚合，无需逐个串行递归 —— 数学上与顺序写入等价。

            τ_new = τ_old + Σ_i η_ik · τ_obs_i
            μ_new = (τ_old·μ_old + Σ_i η_ik·τ_obs_i·μ_q_i) / τ_new

        低 σ²_old（= 高 τ_old）的槽自动抗覆盖 —— 这就是「确定的记忆不易被冲刷」，
        不需要额外机制。dist 与 point 走同一个公式，仅精度项是否为常数不同。

        返回 (kl [B,G,N], free_energy 标量) 供训练用。
        """
        mu_q, logvar_q = self.encode(evidence)
        w = self.get_prior(mu_q)                            # [B,G,N,K]
        _, kl = self.kl_to_mixture(mu_q, logvar_q, w)       # [B,G,N]

        if self.mode == "dist":
            # 写入率由 surprise 门控：KL 大（信息新）且该槽相关（w 大）→ 写得多。
            # KL 上 detach 是刻意的：η 作为**控制信号**而非可微路径，
            # 避免模型通过压低 KL 来偷懒关闭写入。写入内容 μ_q/logvar_q 仍带梯度。
            #
            # 用 **chunk 内标准化的 KL**，而不是它的绝对值。
            #
            # KL 的绝对量级会随记忆状态漂移好几个数量级（实测：记忆是初值时约 0.1，
            # 吸收几轮后涨到 1e3）。固定的 α/β 不可能同时覆盖这个范围 ——
            # 一端 sigmoid 恒为 0.12、另一端恒为 1.00，两头都 std≈0，
            # 门控从未工作在敏感区，dist 档实际退化成「无条件全量写入」。
            # 改用相对 surprise 后语义也更正确：写入率取决于「这个 KV 相对同批
            # 其他 KV 有多意外」，这是个不随记忆漂移的稳定信号。
            # w*K 把分配权重归一化到均值 1，避免 K 变化时需要重调 α。
            kl_d = kl.detach()
            kl_z = (kl_d - kl_d.mean(dim=-1, keepdim=True)) / kl_d.std(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            # 分配与强度**解耦**（缺口 B4）：
            #   η_i  = sigmoid(α·z(KL_i) − β)  标量，"这个观测总共写入多少"
            #   w_ik = softmax over k          分配比例，Σ_k w_ik = 1
            #   gate = w_ik · η_i              ⇒ Σ_k gate_ik = η_i ≤ 1
            # 把两者混进同一个 sigmoid 会让行和最大到 K（实测 0.66~16），
            # 即一个观测以全强度写入所有槽 —— 同一份信息的精度被重复计了 K 次，
            # 直接违反「独立观测的精度可加」这个贝叶斯更新赖以成立的前提。
            eta = torch.sigmoid(
                self.cfg.eta_alpha * kl_z - self.cfg.eta_beta
            )                                               # [B,G,N]
            gate = w * eta.unsqueeze(-1)                    # [B,G,N,K]
            tau_obs = torch.exp(-logvar_q)                  # [B,G,N,d_z]
            tau_old = torch.exp(-self.logvar)               # [B,G,K,d_z]
        else:
            # 点记忆：写入率是与内容无关的可学习标量，方差恒定。
            # 结构与 dist 完全对称（同为 w·η），差别只在 η 是否由 KL 导出。
            gate = w * torch.sigmoid(self.point_gate_logit)
            tau_obs = torch.ones_like(logvar_q)
            tau_old = torch.ones_like(self.logvar)

        # Σ_i η_ik · τ_obs_i  和  Σ_i η_ik · τ_obs_i · μ_q_i
        w_tau = gate.unsqueeze(-1) * tau_obs.unsqueeze(-2)          # [B,G,N,K,d_z]
        sum_tau = w_tau.sum(dim=2)                                  # [B,G,K,d_z]
        sum_tau_mu = (w_tau * mu_q.unsqueeze(-2)).sum(dim=2)        # [B,G,K,d_z]

        # 遗忘因子：给旧证据的精度打折，防止流式吸收下精度无界累加。
        # 否则 τ_old 会一路涨到上界，记忆变得过度自信而拒绝一切新写入 ——
        # 那样 stage1 的 update 型样本（事实被改写）必然做不对。
        gamma = self.cfg.precision_decay
        tau_old = gamma * tau_old
        tau_new = tau_old + sum_tau
        mu_new = (tau_old * self.mu + sum_tau_mu) / tau_new.clamp_min(1e-8)

        # 位置质心与 μ 用同一套精度权重更新，保证「槽的位置」确实是
        # 它所概括的那些 token 的加权中心。
        if positions is not None:
            pos_i = positions.float().view(1, 1, -1)                 # [1,1,N]
            tau_s_obs = tau_obs.mean(-1)                             # [B,G,N]
            wgt = gate.detach().float() * tau_s_obs.detach().float().unsqueeze(-1)
            num = (wgt * pos_i.unsqueeze(-1)).sum(dim=2)             # [B,G,K]
            den = wgt.sum(dim=2)                                     # [B,G,K]
            tau_p_old = gamma * self._pos_tau
            tau_p_new = tau_p_old + den
            self.pos = (tau_p_old * self.pos + num) / tau_p_new.clamp_min(1e-8)
            self._pos_tau = tau_p_new

        self.mu = mu_new
        if self.mode == "dist":
            # 方差随证据收缩：吸收得越多越确定。point 档不更新方差。
            self.logvar = self._clamp(-torch.log(tau_new.clamp_min(1e-8)))

        # 变分自由能（ELBO 意义）：重建失真 + KL。
        #
        # 缺口③（theory §9.7）：重建失真必须与 F_i 的失真项**同一个定义**，
        # 否则「自由能」这个词在代码里指了两个不同的量，
        # §11.1「一个标量统一两个决策」就是拼出来的，审稿人一眼能看穿。
        # 按 §9.2「自由能是率失真的变分上界」，失真应在注意力输出空间，
        # 即按期望注意力 ā 加权 —— 不被未来 query 注意的 KV 本来就不该占码率，
        # 这正是率失真「按重要性分配比特」的应有之义。
        #
        # 权重用 N·ā（均值归一化到 1）而非 ā²：保持「按重要性加权」的语义，
        # 同时让这一项的量级与普通 MSE 相当，不必为它单独调 free_energy_weight。
        # （F_i 里用 ā² 是因为那里要的是输出扰动的平方范数，且只用于排序、尺度无关。）
        recon = self.decode(mu_q, logvar_q)[..., : self.d_kv]
        sq_err = (recon - evidence).pow(2).mean(-1)                 # [B,G,N]
        if expected_attn is not None:
            n = sq_err.shape[-1]
            w_attn = (expected_attn.detach().float() * n).clamp(0.0, float(n))
            recon_err = (w_attn * sq_err.float()).mean()
        else:
            # recency 驱逐档没有 ā（不算自由能），退回未加权重建
            recon_err = sq_err.mean()
        # KL 取 per-dim 平均，使量级与 d_latent 解耦；否则换个 d_z 就得重调权重，
        # 且 d_z 较大时辅助项会盖过 lm_loss，把训练带向 posterior collapse。
        free_energy = recon_err + kl.mean() / self.d_z
        return kl, free_energy

    # ------------------------------------------------------------------ 读出

    def read(self) -> torch.Tensor:
        """槽 → effective KV，供未来 query 注意。返回 [B, G, K*T, d_kv]

        缺口③（方差进读出，两条路径）：
          1. 重参数化采样 z ~ N(μ, σ²) —— 不确定的槽读出带更大抖动，
             其贡献在期望意义下被自然削弱；
          2. logvar 直接喂进 decoder —— 让网络显式学到「高精度槽输出更强的 key」。
        若这两条都关掉，σ² 就只影响写入、不影响读出，「分布式」只用了一半功能，
        生死实验会因此假阴性 —— 这正是 HANDOFF 红线 1 说的「方差必须是功能性的」。
        """
        mu, logvar = self.mu, self.logvar
        if self.cfg.sample_on_read and self.training and self.mode == "dist":
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        out = self.decode(z, logvar)                        # [B,G,K,T*d_kv]
        B, G, K, _ = out.shape
        return out.reshape(B, G, K * self.cfg.tokens_per_slot, self.d_kv)

    def read_precision(self) -> torch.Tensor:
        """每个 effective KV 的精度标量 [B, G, K*T]，供上层做方差感知的加权。"""
        prec = torch.exp(-self.logvar).mean(-1)             # [B,G,K]
        return prec.repeat_interleave(self.cfg.tokens_per_slot, dim=-1)
