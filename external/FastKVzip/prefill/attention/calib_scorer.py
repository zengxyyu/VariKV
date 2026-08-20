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
    # 因子消融族（2026-08-17 加）：把 `scalar` 的三个标量输入拆开单独测。
    # **必须拆**，因为 `scalar ≫ affine` 同时改了两件事——函数类（线性→MLP）和
    # 输入集（只有 z → z+margin+rs），现在分不清胜因是非线性还是"知道全局阈值在哪"。
    # 四个变体共用同一个 MLP、同一 hidden、同一 embedding，唯一变量是输入集：
    #     sz    = MLP([z, e])              只有头内位置（+ 非线性）
    #     szr   = MLP([z, rs, e])          + 本头尺度／全局尺度
    #     szm   = MLP([z, margin, e])      + 到全局阈值的距离
    #     scalar= MLP([z, margin, rs, e])  全部（原样保留）
    #     szmr0 = MLP([z, margin, rs])     **去掉头身份** —— 若仍有效，说明存在
    #                                      与"我是哪个头"无关的普适修正律
    MODES = ("bias", "affine", "scalar", "kv", "k", "v",
             "sz", "szr", "szm", "szmr0", "chead")
    # 逐 arch 的标量输入清单，delta() 与 __init__ 共用同一张表，避免两处各改各的
    SCALAR_FEATS = {"sz": ("z",), "szr": ("z", "rs"), "szm": ("z", "mg"),
                    "scalar": ("z", "mg", "rs"), "szmr0": ("z", "mg", "rs")}
    NO_EMB = ("szmr0",)
    # **逐头常数臂（2026-08-21 加）**：输出与 token 无关的 `c_h`。
    # 为什么必须有它：`scale="global"` 下逐 token 残差**会破坏头内保序** ——
    # 实测三个种子最小 `ds'/ds` = −4.87/−4.41/−1.56，非单调状态 528/506/347 (共 1680)，
    # 最危险点全在 `A_h = σ_h/σ_g ≈ 0.003` 的低 σ 头上（放大 1/A ≈ 300×）。
    # 而头内常数平移使 `ds'/ds ≡ 1` —— **保序是构造性的，不需要任何探针**，
    # 于是等价定理的前提自动满足，该臂**可证地只做逐头配额重分配**。
    # 头级输入（每头三个标量，全部可从 stats 得到，不看 K/V）：
    #   rs  = log(σ_h/σ_g)        本头尺度 vs 全局尺度
    #   mgm = mean_i (s⁰_i − τ)/σ_g   本头到全局阈值的平均距离
    #   mgx = max_i  (s⁰_i − τ)/σ_g   本头**最好的那个 token** 离阈值多远
    # `mgx` 是可达性分析直接指出的量：越阈与否取决于缺口 `τ − s_max,h`。
    HEAD_FEATS = {"chead": ("rs", "mgm", "mgx")}

    def __init__(self, d_kv: int, n_layers: int, n_heads_kv: int, arch: str = "affine",
                 n_slots: int = 8, d_m: int = 128, mode: str = "memoryless",
                 alpha_max: float = 1.0, alpha_init: float = 1.0, typed: bool = True,
                 d_emb: int = 16, replace: bool = False, scale: str = "head"):
        super().__init__()
        assert arch in self.MODES, arch
        # **动作幅度的尺度**（2026-08-21 加）。原实现固定用 `sig_h`（该头自己的分数
        # 离散度），但**决策边界是全局 Top-B，活在 `sig_g` 的尺度上**。
        # 实测（`scratch_probe_cstar.py`，74 chunk）：让地板配额可达所需的统一界
        # `C* = 1.16·sig_g` 中位，而当前中位允许量只有 `0.137·sig_g` —— **差 7.3 倍**。
        # ⚠ 这不是「把界调大」：`VARIKV_CTRL_GAIN` 的增益扫描已证明单纯放大**有害**
        # （|g|=1 近最优，g=2 在两个 panel 上都显著变差），因为按 `sig_h` 放大会
        # **同比例**放大本来就可达的高 σ 头。`scale="global"` 是**重加权**：
        # 低 σ 头相对得到更多、高 σ 头相对更少，正是诊断指向的方向。
        # **存进 ckpt、不走环境变量** —— `--ctrlm_mode` 那次默认值把整批评测跑成
        # 另一个方法的教训：任何会改变方法本体的开关都必须随权重一起走。
        assert scale in ("head", "global"), scale
        self.scale = scale
        # mode 只为与 ControlMemory 的三臂接口兼容；本模块无记忆，三臂必然同解，
        # 所以只允许 memoryless，避免有人误读成"对照做过了"
        assert mode == "memoryless", "CalibScorer 无记忆，只跑 memoryless 臂"
        self.arch, self.mode = arch, mode
        # **replace=True：分数不再是 s⁰+Δs，而是 Δs 本身。**
        # 这一条决定方法的身份：若独立打分器追平残差，FastKVzip 的分数就不是必要组件，
        # 方法是"从 KV 学驱逐效用"——那正是 KVP (2602.10238) 的地盘；若残差明显更强，
        # 方法就是"对一个强先验做残差修正"，与 KVP / KVpop（都是 standalone）区分开。
        # replace 时输出**不经 tanh、不乘 σ_h、不乘 α**：那三样是"有界修正"的语义，
        # 对一个从头打的分没有意义，加上反而把分数压进 ±σ_h 的窄带。
        self.replace = bool(replace)
        self.L, self.H, self.d_m = n_layers, n_heads_kv, d_m
        d_x = 2 * d_kv

        if arch in ("bias", "affine"):
            # a 初始为 0 ⇒ Δs ≡ 0 ⇒ 与基线逐位相同（与 ControlMemory 同一条构造性保证）
            self.ab = nn.Parameter(torch.zeros(n_layers, n_heads_kv, 2))
        else:
            self.emb = nn.Parameter(torch.randn(n_layers, n_heads_kv, d_emb) * 0.02)
            if arch in self.HEAD_FEATS:
                d_in = len(self.HEAD_FEATS[arch]) + d_emb
            elif arch in self.SCALAR_FEATS:
                d_in = len(self.SCALAR_FEATS[arch]) + (0 if arch in self.NO_EMB else d_emb)
            else:
                d_in = d_m + d_emb
            self.head = nn.Sequential(nn.Linear(d_in, d_m), nn.GELU(),
                                      nn.Linear(d_m, 1))
            # k / v 单独测：`f(K)≈f(K,V)` ⇒ 学的是"将来是否容易被 query 匹配到"
            # （addressability）；`f(V)≈f(K,V)` ⇒ 学的是 payload 本身的重要性；
            # 只有 K+V 才行 ⇒ 效用依赖两者的交互。三个故事完全不同。
            if arch in ("kv", "k", "v"):
                self.x_proj = nn.Linear(d_x if arch == "kv" else d_kv, d_m)
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
        d = x_raw.shape[-1] // 2                       # raw = [k ; v]
        if self.arch == "kv":
            return self.x_proj(x_raw)
        if self.arch == "k":
            return self.x_proj(x_raw[..., :d])
        if self.arch == "v":
            return self.x_proj(x_raw[..., d:])
        return x_raw

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
            if self.arch in self.HEAD_FEATS:
                # **逐头一个标量**：先把逐 token 的 mg 汇成头级统计量，
                # MLP 只吃 [H, 3+d_emb]，输出 [H,1] 再广播回 [H,n]。
                # 广播是关键：同一头内所有 token 拿到**同一个**修正 ⇒ 保序恒成立。
                avail_h = {"rs": rs[:, :1], "mgm": mg.mean(-1, keepdim=True),
                           "mgx": mg.max(-1, keepdim=True).values}
                fh = [avail_h[k] for k in self.HEAD_FEATS[self.arch]] + [r]
                raw = self.head(torch.cat(fh, dim=-1)).expand_as(z)
            elif self.arch in self.SCALAR_FEATS:
                avail = {"z": z, "mg": mg, "rs": rs}
                feats = [avail[k][..., None] for k in self.SCALAR_FEATS[self.arch]]
                if self.arch not in self.NO_EMB:
                    feats.append(e)
            else:
                feats = [x, e]
            if self.arch not in self.HEAD_FEATS:
                # chead 分支上面已经算好 raw 并广播过了，不能再走这一行
                raw = self.head(torch.cat(feats, dim=-1)).squeeze(-1)
        if self.replace:
            return raw                                 # 独立打分器：分数本身，不设界
        # 与 ControlMemory 同一条输出规范：有界、乘 α。
        # `scale="head"` 用逐头 `sig_h`（原样，默认，逐位不变）；
        # `scale="global"` 用全局 `sig_g` —— 见 __init__ 里的推导。
        _sc = sig_h if getattr(self, "scale", "head") == "head" else sig_g
        return self.alpha * _sc * torch.tanh(raw)
