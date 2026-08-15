"""CalibScorer —— 把 memoryless 的 +4.27 拆开：它到底来自什么？

`memoryless` 臂在 Retr.KV @0.1 上稳定 +4.27 ± 0.23（3 训练种子）。我一直称它为
"学习残差打分器"，暗示它学到了 KV 的语义重要性。但它的输入里有

    z = (s⁰−μ_h)/σ_h ,  margin = (s⁰−τ)/σ_g ,  log(σ_h/σ_g) ,  以及逐 (层,头) 的 M_init

而 `level="pair"` 是**跨层跨头的全局阈值化**。若两个头的分数尺度相差一个量级，尺度大的
头会天然多拿预算——这是**校准问题，不是排序问题**：即使两个头**头内排序都完美**，
放到一起做全局 top-B 仍会分配失当。而上面那几个特征恰好把纠正它所需的信息全给了。

所以有一个必须证伪的竞争解释：

    +4.27 主要来自**跨层/头的分数尺度重校准**，而不是 KV 语义。

本模块用**逐 (层,kv头) 只有 2 个标量**的仿射变换来检验它。若 224 个参数就能拿到
接近 +4 的增益，那 637.8K 的 ControlMemory 是杀鸡用牛刀，而论文命题也随之变得干净：

    KV 驱逐分数在头内是有排序信息的，但在全局预算下**跨层/头失准**。

--------------------------------------------------------------------------------
四种消融，共用 `ControlMemory` 的鸭子类型接口（raw/feat/q_read/read/delta/write/
init_state/alpha/n_params），所以 trainer 与 `LearnedControlRetainCache` 一行都不用改。

| mode     | Δs / σ_h                               | 参数量 (L=28,H=4) | 分离什么 |
|----------|----------------------------------------|------------------|---------|
| `bias`   | `b_{lh}`                               | 112              | 纯预算再分配（头间平移） |
| `affine` | `a_{lh}·z + b_{lh}`                    | 224              | + 尺度重校准 |
| `scalar` | `MLP([z, margin, rs, e_{lh}])`         | ~10K             | 不看 KV，只看分数统计量 |
| `kv`     | `MLP([x̃, e_{lh}])`                    | ~40K             | 只看 KV，不看分数统计量 |

判读：
  `affine ≈ full` ⇒ 增益是校准，方法可以大幅简化，论文换命题；
  `scalar ≫ kv`   ⇒ 增益来自分数统计量的重整，与 token 内容无关；
  `kv ≫ scalar`   ⇒ KV 确实携带 gate 没抓到的 token 级效用，原故事成立。

**`per-(l,h)` 参数怎么进来**：`delta()` 拿不到层号（`ControlMemory` 是靠
`init_state(l)` 把 `M_init[l]` 塞进 state、再由 `read` 带出来的）。这里沿用同一手法——
`init_state(l)` 返回带层号的 state，`read` 把该层的参数取出来当 `r` 返回。这样接口
完全不变，也就不会因为改 trainer 而引入新的不可比因素。
"""
import torch
import torch.nn as nn


class CalibScorer(nn.Module):
    MODES = ("bias", "affine", "scalar", "kv")

    def __init__(self, d_kv: int, n_layers: int, n_heads_kv: int, arch: str = "affine",
                 n_slots: int = 8, d_m: int = 128, mode: str = "memoryless",
                 alpha_max: float = 1.0, alpha_init: float = 1.0, typed: bool = True,
                 d_emb: int = 16):
        super().__init__()
        assert arch in self.MODES, arch
        # mode 只为与 ControlMemory 的三臂接口兼容；本模块无记忆，三臂必然同解，
        # 所以只允许 memoryless，避免有人误读成"对照做过了"
        assert mode == "memoryless", "CalibScorer 无记忆，只跑 memoryless 臂"
        self.arch, self.mode = arch, mode
        self.L, self.H, self.d_m = n_layers, n_heads_kv, d_m
        d_x = 2 * d_kv

        if arch in ("bias", "affine"):
            # a 初始为 0 ⇒ Δs ≡ 0 ⇒ 与基线逐位相同（与 ControlMemory 同一条构造性保证）
            self.ab = nn.Parameter(torch.zeros(n_layers, n_heads_kv, 2))
        else:
            self.emb = nn.Parameter(torch.randn(n_layers, n_heads_kv, d_emb) * 0.02)
            d_in = (3 + d_emb) if arch == "scalar" else (d_m + d_emb)
            self.head = nn.Sequential(nn.Linear(d_in, d_m), nn.GELU(),
                                      nn.Linear(d_m, 1))
            if arch == "kv":
                self.x_proj = nn.Linear(d_x, d_m)
        self.alpha_max = float(alpha_max)
        _p = min(max(alpha_init, 1e-6), 0.999)
        self.alpha_on = nn.Parameter(
            torch.full((), float(torch.logit(torch.tensor(_p)))))

    # ------------------------------------------------- ControlMemory 的接口
    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_on)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def init_state(self, layer_idx: int, dtype=torch.float32):
        """把层号塞进 state —— 与 ControlMemory 用 `M_init[l]` 是同一手法。"""
        return (torch.tensor(int(layer_idx)), torch.zeros(1))

    def raw(self, k, v):
        return torch.cat([k, v], dim=-1).float()

    def feat(self, x_raw):
        return self.x_proj(x_raw) if self.arch == "kv" else x_raw

    def q_read(self, x_raw):
        return x_raw

    def read(self, state, x_raw):
        """返回该层的逐头参数，形状 [H, ·]。`delta` 靠它拿到 (l,h) 身份。"""
        l = int(state[0])
        return self.ab[l] if self.arch in ("bias", "affine") else self.emb[l]

    def write(self, state, x, m_ret, m_evi, gen=None):
        return state                                   # 无记忆

    # ------------------------------------------------------------ 控制器
    def delta(self, x, r, s0, q=None, margin=None, stats=None):
        if stats is None:
            mu_h = s0.mean(-1, keepdim=True)
            sig_h = s0.std(-1, keepdim=True).clamp_min(1e-6)
            sig_g = s0.std().clamp_min(1e-6)
        else:
            mu_h, sig_h, sig_g = stats
            mu_h = torch.as_tensor(mu_h, dtype=s0.dtype, device=s0.device).view(-1, 1)
            sig_h = torch.as_tensor(sig_h, dtype=s0.dtype,
                                    device=s0.device).view(-1, 1).clamp_min(1e-6)
            sig_g = torch.as_tensor(sig_g, dtype=s0.dtype, device=s0.device).clamp_min(1e-6)
        z = (s0 - mu_h) / sig_h
        r = r.to(s0.device)

        if self.arch == "bias":
            raw = r[:, 1:2].expand_as(z)                       # 只有平移
        elif self.arch == "affine":
            raw = r[:, 0:1] * z + r[:, 1:2]                    # 尺度 + 平移
        else:
            mg = torch.zeros_like(z) if margin is None else margin
            rs = (sig_h / sig_g).log().expand_as(z)
            e = r[:, None, :].expand(z.shape[0], z.shape[1], -1)
            feats = ([z[..., None], mg[..., None], rs[..., None], e]
                     if self.arch == "scalar" else [x, e])
            raw = self.head(torch.cat(feats, dim=-1)).squeeze(-1)
        # 与 ControlMemory 同一条输出规范：有界、逐头尺度、乘 α
        return self.alpha * sig_h * torch.tanh(raw)
