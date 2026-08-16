#!/usr/bin/env python3
"""VariKV v2 —— 自包含的最小实现，与 `ctrl_b_a1_s0.pt/memoryless.pt` **逐位等价**。

v2 是这个项目唯一一个可复现的正结果：Retr.KV @ratio 0.1 稳定 **+4.27 ± 0.23**
（3 个训练种子，全部 ★），@0.2 **+18.80★**。它做的事只有一件：

    在 FastKVzip 的驱逐分数上加一个**有界的、学出来的残差**，
    绝不进入 attention 输出，绝不改变保留条数。

        s'_i = s⁰_i + α · σ_h · tanh( MLP[ 特征_i ] )

`α=0 ⇒ Δs≡0 ⇒ 与基线逐位相同`，这是构造性保证而非断言。

--------------------------------------------------------------------------------
为什么能简化掉三分之一：v2 是 `memoryless` 臂，而 `ControlMemory.write()` 在
memoryless 下**第一行就原样返回状态**。于是 `M_init` / `D_init` 永不更新，
`read()` 是在一组**固定的、逐 (层, kv头) 学出来的码本**上做注意力。

    ⇒ v2 不是记忆，是**逐 (层,kv头) 的码本查表 + MLP**。

只在 `write`/`_pool` 里出现的模块因此永不被调用（实测 637,828 → 423,298，死代码
**33.6%**）：`k_ret` `v_ret` `k_evi` `v_evi` `q_slot` `mix` `gru` `dir_decay`。
本文件用 `strict=False` 加载原 ckpt，并**断言缺失键恰好等于上面这一串**——
少一个多一个都会崩，所以"哪些是死的"这件事由代码强制，不靠注释。

递归历史被三条独立证据判否（凸线性探针留一 / 三臂 TOST 等价 / 下游配对
bootstrap，见 `varikv_b_method.md` §9），所以 stateful / shuffled 两臂在这里
**不再实现**——它们是对照，不是方法。

--------------------------------------------------------------------------------
同时删掉的，以及为什么删得掉

| 删掉 | 理由 |
|---|---|
| `stateful`/`shuffled` 两臂与 `write()` | 上面那条；对照已完成，结论是否定 |
| `_write` 的逐头子采样 `n_write` | 只为 writer 的训练/部署分布一致而存在 |
| `train_mode` / `collect()` / `trace` | 教师采集是另一支脚本的事，不属于方法 |
| `replace` 分支 | 那是"独立打分器"消融（KVP 的地盘），不是 v2 |
| `rho_max` 预算门控 | v2 之后加的可部署门控，默认 1.0 即无操作；见文末如何加回 |
| `typed=False` 分支 | v2 的 ckpt 是 typed（`M_init` 第 3 维 = 2K），另一支从未用于正结果 |

--------------------------------------------------------------------------------
用法

    # 逐位等价性验收（不需要 GPU 上的模型，只加载 ckpt）
    .venv/bin/python varikv_v2.py verify

    # 评测：把 v2 接进 harness（与 eval_chunk.py 的 --ctrlm_ckpt 等价）
    #   在 eval_chunk.py 里把 kv_class 换成 V2RetainCache 并传入 scorer 即可，
    #   或直接用现成路径：eval_chunk.py ... --ctrlm_ckpt <ckpt>

    # 训练（需要教师 trace，格式见 run_doc 的 docstring）
    .venv/bin/python varikv_v2.py train --traces scratch_ctrl_traces_v2 \
        --epochs 40 --seed 0 --out varikv/v2_clean.pt
"""
import argparse
import os
import random
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
_P = os.path.join(ROOT, "external/FastKVzip/prefill")
if _P not in sys.path:
    sys.path.insert(0, _P)

# memoryless 下永不被调用的模块 —— `verify` 用它做强制断言
DEAD = ("k_ret", "v_ret", "k_evi", "v_evi", "q_slot", "mix", "gru", "dir_decay")


# ══════════════════════════════════════════════════════════════════ 模型
class V2Scorer(nn.Module):
    """逐 (层, kv头) 码本查表 + MLP → 有界的驱逐分数残差。

    参数名与 `ControlMemory` 完全一致，所以原 ckpt 可以直接
    `load_state_dict(..., strict=False)` 加载，等价性可验证。

    形状约定（全程 fp32；bf16 累积在这里会掉精度）：
        k, v      [H, n, d_kv]          某一层某个 chunk 的候选 KV
        x, q      [H, n, d_m]           内容投影 / 读出 query
        r_R, r_E  [H, n, d_m]           两型码本的读出
        s0, Δs    [H, n]                基线分 / 残差
    """

    def __init__(self, d_kv: int = 128, n_layers: int = 28, n_heads_kv: int = 4,
                 n_slots: int = 8, d_m: int = 128, alpha_max: float = 1.0,
                 alpha_init: float = 1.0):
        super().__init__()
        self.L, self.H, self.K, self.d_m = n_layers, n_heads_kv, n_slots, d_m
        d_x = 2 * d_kv                                  # 候选特征 = [k_i ; v_i]

        # ---- 码本：前 K 槽 = "像被保留的"，后 K 槽 = "像被驱逐的" ----
        # 分型的动机是表达力，不是拟人化：把所有槽拼起来做**一次** softmax，输出必然是
        # 槽向量的凸组合 `r = Σ_j a_j S_j`（a_j ≥ 0, Σa_j = 1）。而控制器需要的是
        # "像保留的" 与 "像驱逐的" 之间的**有符号对比**，它一般落在凸包之外（两个近似
        # 单位向量之差，范数 √2）。加 value 投影救不了：`r = W(Σ a_j S_j) + b`，括号里
        # 仍在凸包内。约束在**权重**上，所以只能两型各读各的，再把
        # `[r_R, r_E, r_E−r_R, ...]` 一起交给 MLP，由它学有符号组合。
        self.M_init = nn.Parameter(
            torch.randn(n_layers, n_heads_kv, 2 * n_slots, d_m) * 0.02)
        # `D_init` 在原版里是 EMA 通路的初值；memoryless 下它就是**每型各多一个可学槽**。
        # 保留它不是为了兼容，是因为 ckpt 里这 28,672 个参数确实参与前向。
        self.D_init = nn.Parameter(torch.zeros(n_layers, n_heads_kv, 2, d_m))
        # 类型嵌入：读出是对槽集合做注意力，**对槽的身份置换不变**，单看内容分不出
        # "这是保留过的方向"还是"被丢弃过的方向"。代价 2·d_m 个参数。
        self.dir_type = nn.Parameter(torch.randn(2, d_m) * 0.02)

        self.x_proj = nn.Linear(d_x, d_m)               # 内容投影（进 MLP）
        # **读出 query 必须独立于 x_proj。** 共用一张矩阵会强迫它同时满足"摘要"和
        # "内积检索"两个目标；分开后 <q_read(x), S_j> 可以自由学成近似原空间的内积。
        self.q_read = nn.Linear(d_x, d_m)

        # MLP 必须含**乘性交互**：要表达"候选与码本的匹配程度"就需要 x 与 r 的双线性项，
        # 拼接后的 MLP 很难自己学出乘积。给了 q⊙r 与 <q,r> 之后线性层直接能算加权内积。
        # 输入 = [x, r_R, r_E, r_E−r_R, q⊙r_R, q⊙r_E] (6·d_m) + [dot_R, dot_E, z, mg, rs]
        d_in = 6 * d_m + 5
        self.head = nn.Sequential(nn.Linear(d_in, d_m), nn.GELU(),
                                  nn.Linear(d_m, 1))
        # α **有上界**：α = α_max·sigmoid(a)。写成 sigmoid(a)·exp(b) 会让 α 无界，
        # "tanh 让修正有界"这句话就不成立了。而手工版实测**大幅扰动基线排序本身有害**
        # （β=±1.5 时连 shuffle 对照都掉 4.6–5.8 分），这个事实应当编码进架构。
        # v2 训练时 α 被**冻结在 1.0**：让它自学 40 epoch 只从 0.050 爬到 0.0555，
        # Δs 满幅仅为近阈值池内典型 |Δs⁰| 的 12%，只有 24% 的成对翻得动 —— 判据被
        # 构造性封顶。冻结是为了解除这个封顶，不是调参。
        self.alpha_max = float(alpha_max)
        _p = min(max(alpha_init, 1e-6), 0.999)
        self.alpha_on = nn.Parameter(
            torch.full((), float(torch.logit(torch.tensor(_p)))))

    @property
    def alpha(self):
        return self.alpha_max * torch.sigmoid(self.alpha_on)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ---------------------------------------------------------------- 前向
    def raw(self, k, v):
        """[H,n,d_kv]×2 → [H,n,2·d_kv]，fp32。"""
        return torch.cat([k, v], dim=-1).float()

    def banks(self, layer_idx: int):
        """该层的两组码本 [H, K+1, d_m]。memoryless 下它们与 chunk 无关，
        所以本可以整篇缓存一次；训练时参数每步都变，故仍逐次构造（只是 cat+add）。"""
        M, D = self.M_init[layer_idx], self.D_init[layer_idx]
        K = M.shape[1] // 2
        return (torch.cat([M[:, :K], D[:, 0:1] + self.dir_type[0]], dim=1),
                torch.cat([M[:, K:], D[:, 1:2] + self.dir_type[1]], dim=1))

    def read(self, layer_idx: int, q):
        """q [H,n,d_m] → (r_R, r_E)，各 [H,n,d_m]。两型**各读各的**，
        绝不先合成一个凸组合向量（见 `M_init` 处的说明）。"""
        S_R, S_E = self.banks(layer_idx)

        def _attn(S):
            a = torch.einsum("hnd,hkd->hnk", q, S) * self.d_m ** -0.5
            return torch.einsum("hnk,hkd->hnd", a.softmax(-1), S)

        return _attn(S_R), _attn(S_E)

    def delta(self, layer_idx: int, k, v, s0, margin, stats):
        """→ Δs [H,n]，**已含 α 与逐头尺度**。

        三个标量特征，缺一不可：

        - `z = (s⁰−μ_h)/σ_h` 逐 (层,kv头) 的 z-score。不归一的话 MLP 会隐式去学各头的
          尺度差异而不是内容。
        - `margin = (s⁰−τ)/σ_g` 到**全局**淘汰阈值的距离。`level="pair"` 是跨层跨头的
          全局阈值化，真正决定去留的是 `s⁰−τ` 而不是头内排名：两个 token 可以有完全
          相同的 z，却因所在头整体分数高低不同而一个稳留、一个稳删。只喂 z 等于把
          决策边界藏起来。
        - `log(σ_h/σ_g)` 本头尺度相对全局尺度。**输出被缩放到逐头单位、而决策边界是
          全局单位**，两者的换算率就是这个比值；不给的话头无从知道自己这一步"值多少
          全局 σ"，`margin` 也就用不起来。

        `stats=(μ_h, σ_h, σ_g)` **必须来自整块候选的全量统计**，不能从手上这批候选现算。
        训练侧拿到的是"近阈值 + 随机"的有偏子集，σ 被系统性低估，而 σ 是**直接乘在
        Δs 上的尺度** ⇒ 部署时的扰动幅度会大于训练时学到的幅度。上面刚说过大幅扰动
        基线排序本身有害，所以这不是小数点问题。
        """
        xr = self.raw(k, v)
        x = self.x_proj(xr)
        q = self.q_read(xr)
        rR, rE = self.read(layer_idx, q)

        mu_h, sig_h, sig_g = stats
        mu_h = torch.as_tensor(mu_h, dtype=s0.dtype, device=s0.device).view(-1, 1)
        sig_h = torch.as_tensor(sig_h, dtype=s0.dtype,
                                device=s0.device).view(-1, 1).clamp_min(1e-6)
        sig_g = torch.as_tensor(sig_g, dtype=s0.dtype,
                                device=s0.device).clamp_min(1e-6)
        z = (s0 - mu_h) / sig_h
        mg = torch.zeros_like(z) if margin is None else margin
        rs = (sig_h / sig_g).log().expand_as(z)
        sc = self.d_m ** -0.5
        raw = self.head(torch.cat(
            [x, rR, rE, rE - rR, q * rR, q * rE,
             (q * rR).sum(-1, keepdim=True) * sc,
             (q * rE).sum(-1, keepdim=True) * sc,
             z[..., None], mg[..., None], rs[..., None]], dim=-1)).squeeze(-1)
        return self.alpha * sig_h * torch.tanh(raw)

    # ---------------------------------------------------------------- 存取
    @classmethod
    def from_ckpt(cls, path, strict_dead: bool = True):
        """加载 `ControlMemory` 的 memoryless ckpt，并**断言缺失键恰好是死代码**。"""
        sd = torch.load(path, map_location="cpu")
        assert sd.get("mode") == "memoryless", \
            f"v2 是 memoryless 臂，收到 mode={sd.get('mode')}"
        assert sd.get("arch", "memory") == "memory", \
            f"这是 CalibScorer 的 ckpt（arch={sd['arch']}），不是 v2"
        m = cls(sd.get("d_kv", 128), sd["L"], sd["H"],
                n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128))
        r = m.load_state_dict(sd["state"], strict=False)
        assert not r.missing_keys, f"本类缺少 ckpt 里有的参数：{r.missing_keys}"
        if strict_dead:
            got = sorted({k.split(".")[0] for k in r.unexpected_keys})
            assert got == sorted(DEAD), \
                (f"未使用的键与预期的死代码不符。\n  预期 {sorted(DEAD)}\n  实得 {got}\n"
                 "  ⇒ 说明 memoryless 下的存活集合变了，简化的前提失效，必须重新核对")
        return m


# ══════════════════════════════════════════════════════════════════ 推理
from attention.kvcache import RetainCache                          # noqa: E402


class V2RetainCache(RetainCache):
    """把 v2 接进 FastKVzip 的分块预填。时序（无同 chunk 泄漏）：

        s⁰_t ──Δs_t──▶ s⁰+Δs ──threshold──▶ R_t / E_t

    三条结构性保证：
      1. **预算不变**：`threshold` 按 ratio 取全局 top-n，改分数只改"留哪些"不改
         "留几个"。注意父类用 `score > score_sort[n]` 而非严格 topk，阈值处并列时
         会少留 —— 所以 `retain_delta` 是**经验事实**而非构造性恒等，要实测。
      2. ratio=1.0 不进 `prune_chunk` ⇒ 满缓存参考天然干净。
      3. `alpha=0 ⇒ Δs≡0 ⇒ 与基线逐位相同`。

    仍派生自 `RetainCache`（逻辑掩码，物理保留全部 KV），所以能回答"选择改变有没有
    用"，**不能**声称峰值显存下降。
    """

    def __init__(self, model, evict_range: Tuple[int, int], scorer=None):
        super().__init__(model, evict_range)
        self.scorer = scorer                    # None ⇒ 纯基线
        self.flip_frac, self.retain_delta, self.delta_std = [], [], []

    @property
    def active(self) -> bool:
        return self.scorer is not None and float(self.scorer.alpha) != 0.0

    def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
        lo, hi = evict_range
        score0 = torch.stack(self.score, dim=0)[..., lo:hi]        # [L,1,H,n]
        score = score0

        if self.active:
            pos = torch.arange(lo, hi, device=self.device)
            with torch.no_grad():
                # τ 用**未修正**的分数算。这不是泄漏：τ 只依赖 s⁰ 与 ratio，
                # 推理时同样拿得到；用修正后的分数算 τ 才会造成自指。
                _, thr_g = self.threshold(score0, ratio, level)
                f0 = score0[:, 0].float()                   # [L,H,n]，**整块全量**
                gsig = f0.std().clamp_min(1e-6)
                mu_h, sig_h = f0.mean(-1), f0.std(-1)       # [L,H]
                delta = torch.zeros_like(score0)
                for l in range(self.n_layers):
                    k = self.key_cache[l][0][:, pos]
                    v = self.value_cache[l][0][:, pos]
                    s0l = score0[l, 0].float()
                    mg = None if thr_g is None else (s0l - thr_g) / gsig
                    delta[l, 0] = self.scorer.delta(
                        l, k, v, s0l, mg, (mu_h[l], sig_h[l], gsig))
                    # **每层用完即弃**：[H,n,d_m] 每层约 16 MB，28 层留着就是 450 MB
                    del k, v
            score = score0 + delta.to(score0.dtype)
            self.delta_std.append(float(delta.std()))

        valid, thres = self.threshold(score, ratio, level)          # [L,H,n]

        if self.active:                                             # 自包含诊断
            with torch.no_grad():
                v0, _ = self.threshold(score0, ratio, level)
                self.flip_frac.append(float((valid ^ v0).float().mean()))
                self.retain_delta.append(int(valid.sum()) - int(v0.sum()))

        self.valid = valid if self.valid is None else torch.cat(
            [self.valid, valid], dim=-1)
        self.flatten = True
        return thres, self.valid.float().mean().item()


# ══════════════════════════════════════════════════════════════════ 训练
def _pw(du, mode="linear"):
    """成对权重 = |ΔU| / median|ΔU|，截到 [0,5]。

    **按 |ΔU| 加权才是那个 regret 的可微代理**：固定预算 top-B 选择的 regret 恰好是
    被错换的成对的 `|U_i − U_j|` 之和。不加权等于把"两个几乎并列的候选排反"和
    "把最重要的和最没用的排反"惩罚成一样。截断是防长尾主导。
    """
    if mode == "none":
        return torch.ones_like(du)
    a = du.abs()
    return (a / a.median().clamp_min(1e-12)).clamp(0.0, 5.0)


def _pairs(sp, s0r, U, sigma, n_pairs, gen, per_head, pair_w="linear"):
    """成对 logistic 排序损失。`per_head=True` 时逐头采样（sp/U 为 [H,n]），
    否则在展平后的一维上采样（跨 (层,kv头) 的全局项）。

    `keep = |ΔU| > 1e-6`：近似并列的对不含排序信息，只会把噪声当信号。
    """
    if per_head:
        H, n = sp.shape
        i = torch.randint(0, n, (H, n_pairs), generator=gen, device=sp.device)
        j = torch.randint(0, n, (H, n_pairs), generator=gen, device=sp.device)
        du = torch.gather(U, 1, i) - torch.gather(U, 1, j)
        ds = (torch.gather(sp, 1, i) - torch.gather(sp, 1, j)) / sigma
        ds0 = (torch.gather(s0r, 1, i) - torch.gather(s0r, 1, j)) / sigma
    else:
        n = sp.numel()
        i = torch.randint(0, n, (n_pairs,), generator=gen, device=sp.device)
        j = torch.randint(0, n, (n_pairs,), generator=gen, device=sp.device)
        du = U[i] - U[j]
        ds = (sp[i] - sp[j]) / sigma
        ds0 = (s0r[i] - s0r[j]) / sigma
    keep = du.abs() > 1e-6
    if not bool(keep.any()):
        z = torch.zeros((), device=sp.device)
        return sp.sum() * 0.0, z, z
    lg, lg0 = ds * du.sign(), ds0 * du.sign()
    w = _pw(du, pair_w)
    loss = (w * F.softplus(-lg))[keep].sum() / w[keep].sum().clamp_min(1e-6)
    # **必须同时报 s⁰ 自己的准确率**：只报 acc(s') 会把"s⁰ 本来多好"和"修正加了多少"
    # 混在一起。真正的量是 acc(s') − acc(s⁰)。
    return loss, (lg[keep] > 0).float().mean(), (lg0[keep] > 0).float().mean()


def run_doc(m, doc, dev, n_pairs, gen, lam_global=1.0, skip_first=True,
            pair_w="linear"):
    """重放一篇文档的所有 chunk，返回 (loss, 头内Δacc, 全局Δacc)。

    教师 trace 的格式（由 `scratch_ctrl_teacher.py` 写出）：

        doc = {H, L, chunks: [ {gsig, layers: [ {k, v, s0, U, ret,
                                                 n_near, thres, mu_h, sig_h} ]} ]}

    每层的候选是 **[近阈值 n_near 个] ++ [随机若干个]** 拼接而成。排序损失**只用前
    n_near 个**：离阈值远的 token 无论 Δs 多大都翻不了，用它们做排序损失是在学一个
    恒真的排序（手工版实测 β=0.5 只翻转 0.895% 的条目）。后半段的随机子集原本是给
    writer 用的 —— memoryless 下无 writer，所以本实现**完全不读它**，`ret` 同理。

    **两级损失。** `level="pair"` 是跨 (层×kv头×token) 的全局阈值化，只在头内采样成对
    样本等于完全不监督"layer 23/head 2 的 token 该不该压过 layer 5/head 1 的"，而
    跨层/头的预算再分配很可能正是收益的主要来源。全局项能成立是因为教师的 U 已经是
    **W_O 投影后、再除以逐层残差 RMS** 的量，天然跨组可比；且全局项**不除以逐头 σ**，
    因为全局阈值比的就是原始分数。

    两项必须**分开聚合**：都塞进一个 list 再取均值的话，每个 chunk 有 L≈28 个头内项
    却只有 1 个全局项 ⇒ `lam_global=1` 的实际权重只有 1/28，"加了全局监督"名不副实。

    `skip_first=True` 沿用 v2 训练时的默认（当时是为了"第一个 chunk 没有历史可读"）。
    **memoryless 下没有历史，这一条纯属丢数据**；保留默认只为逐位复现 v2，做新实验
    时应显式关掉。
    """
    losses, gl, tl, ta, tg, c, gc = [], [], 0.0, 0.0, 0.0, 0, 0
    for ci, ch in enumerate(doc["chunks"]):
        g_sp, g_s0, g_U = [], [], []
        gsig = torch.tensor(float(ch["gsig"]), device=dev).clamp_min(1e-6)
        for l, pl in enumerate(ch["layers"]):
            k = pl["k"].to(dev).float()
            v = pl["v"].to(dev).float()
            s0 = pl["s0"].to(dev).float()
            U = pl["U"].to(dev).float()
            nn_ = pl["n_near"]
            thr = pl.get("thres", None)
            mg = None if thr is None else (s0 - float(thr)) / gsig
            st = (pl["mu_h"].to(dev).float(), pl["sig_h"].to(dev).float(), gsig)
            sp = s0 + m.delta(l, k, v, s0, mg, st)
            if not (skip_first and ci == 0):
                sig = s0.std(-1, keepdim=True).clamp_min(1e-6)
                lo_, a_, a0 = _pairs(sp[:, :nn_], s0[:, :nn_], U[:, :nn_],
                                     sig, n_pairs, gen, True, pair_w)
                losses.append(lo_)
                tl += float(lo_); ta += float(a_ - a0); c += 1
                g_sp.append(sp[:, :nn_].reshape(-1))
                g_s0.append(s0[:, :nn_].reshape(-1))
                g_U.append(U[:, :nn_].reshape(-1))
            del k, v
        if lam_global > 0 and g_sp:
            g0 = torch.cat(g_s0)
            lg_, ag, ag0 = _pairs(torch.cat(g_sp), g0, torch.cat(g_U),
                                  g0.std().clamp_min(1e-6), n_pairs, gen,
                                  False, pair_w)
            gl.append(lg_); tg += float(ag - ag0); gc += 1
    if not losses and not gl:
        return None, 0.0, 0.0, 0.0
    tot = torch.zeros((), device=dev)
    if losses:
        tot = tot + torch.stack(losses).mean()
    if gl:
        tot = tot + lam_global * torch.stack(gl).mean()
    return tot, tl / max(c, 1), ta / max(c, 1), tg / max(gc, 1)


def train(a):
    import glob
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(ROOT, a.traces, "*.pt")))
    assert files, f"{a.traces} 里没有 trace"
    docs = [torch.load(f, map_location="cpu") for f in files]
    L, H = docs[0]["L"], docs[0]["H"]
    # **划分种子与训练种子分开**：`--seed` 只该控制初始化/采样/顺序，若它同时决定
    # train/val 划分，跨种子比较就同时变了两样东西。
    idx = list(range(len(docs)))
    random.Random(a.split_seed).shuffle(idx)
    nv = max(1, int(len(docs) * a.val_frac))
    va = [docs[i] for i in idx[:nv]]
    tr = [docs[i] for i in idx[nv:]]
    print(f"trace {len(docs)} 篇 → 训练 {len(tr)} / 验证 {len(va)}　L={L} H={H}")

    torch.manual_seed(a.seed)
    m = V2Scorer(a.d_kv, L, H, n_slots=a.slots, d_m=a.dim,
                 alpha_init=a.alpha_init).to(dev)
    if a.freeze_alpha:
        m.alpha_on.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01)
    print(f"参数 {m.n_params()/1e3:.1f}K　alpha {float(m.alpha):.4f}"
          f"{'（冻结）' if a.freeze_alpha else ''}")

    for ep in range(a.epochs):
        m.train()
        g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + ep)
        order = list(range(len(tr)))
        random.Random(a.seed * 100 + ep).shuffle(order)
        el, ea, n = 0.0, 0.0, 0
        for di in order:
            out = run_doc(m, tr[di], dev, a.n_pairs, g, a.lam_global,
                          a.skip_first, a.pair_w)
            if out[0] is None:
                continue
            opt.zero_grad(set_to_none=True)
            out[0].backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            el += out[1]; ea += out[2]; n += 1
        m.eval()
        with torch.no_grad():
            gv = torch.Generator(device=dev).manual_seed(12345)
            vl = vh = vg = 0.0
            for d_ in va:
                o = run_doc(m, d_, dev, a.n_pairs, gv, a.lam_global,
                            a.skip_first, a.pair_w)
                vl += o[1]; vh += o[2]; vg += o[3]
            k = max(len(va), 1)
        print(f"  ep{ep} train loss {el/max(n,1):.4f} Δacc {ea/max(n,1):+.4f} | "
              f"val 头内Δacc {vh/k:+.4f} **全局Δacc {vg/k:+.4f}**", flush=True)

    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)) or ".", exist_ok=True)
    torch.save(dict(state=m.state_dict(), mode="memoryless", arch="memory",
                    slots=a.slots, dim=a.dim, d_kv=a.d_kv, L=L, H=H,
                    args=vars(a)), os.path.join(ROOT, a.out))
    print(f"已保存 {a.out}")
    # 训练侧 ranking 指标**与下游反相关**过（`ckpt_kl_v2a` 验证第一、下游最差），
    # 所以这条提醒必须跟着输出走，别让人拿验证曲线当结论。
    print("提醒：训练侧 Δacc 不是下游证据。历史上验证最好的 ckpt 下游最差。")


# ══════════════════════════════════════════════════════════════════ 验收
def verify(a):
    """与原 `ControlMemory` 的**逐位等价性**验收。

    简化只有在能证明是同一个函数时才可信。这里不比较架构描述，直接比较输出。
    """
    from attention.control_memory import ControlMemory

    print(f"ckpt: {a.ckpt}")
    v2 = V2Scorer.from_ckpt(os.path.join(ROOT, a.ckpt)).eval()
    sd = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu")
    cm = ControlMemory(sd.get("d_kv", 128), sd["L"], sd["H"],
                       n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                       mode="memoryless", typed=True)
    cm.load_state_dict(sd["state"])
    cm.eval()

    dead = sum(p.numel() for n_, p in cm.named_parameters()
               if n_.split(".")[0] in DEAD)
    print(f"\n[1] 参数：原版 {cm.n_params():,} → v2 {v2.n_params():,}"
          f"　删掉 {dead:,}（{dead/cm.n_params():.1%}，全部只在 write/_pool 里出现）")
    assert v2.n_params() + dead == cm.n_params()
    print(f"    死代码模块：{', '.join(DEAD)}")

    torch.manual_seed(0)
    H, n, d = sd["H"], 97, sd.get("d_kv", 128)
    worst, worst_l = 0.0, -1
    with torch.no_grad():
        for l in (0, sd["L"] // 2, sd["L"] - 1):
            k, v = torch.randn(H, n, d), torch.randn(H, n, d)
            s0 = torch.randn(H, n) * 3.0 + 1.0
            mu_h, sig_h = s0.mean(-1), s0.std(-1)
            sig_g = s0.std()
            mg = (s0 - 0.3) / sig_g
            st = (mu_h, sig_h, sig_g)
            # 原版路径：raw → feat/q_read/read → delta（与 learned_ctrlcache 一致）
            xr = cm.raw(k, v)
            d_cm = cm.delta(cm.feat(xr), cm.read(cm.init_state(l), xr), s0,
                            q=cm.q_read(xr), margin=mg, stats=st)
            d_v2 = v2.delta(l, k, v, s0, mg, st)
            e = float((d_cm - d_v2).abs().max())
            if e > worst:
                worst, worst_l = e, l
            print(f"[2] 层 {l:>2}: max|Δs_orig − Δs_v2| = {e:.3e}"
                  f"　(|Δs| 典型 {float(d_cm.abs().median()):.4f})")
    assert worst < 1e-6, f"层 {worst_l} 不等价，误差 {worst:.3e}"
    print(f"    ⇒ 最大误差 {worst:.3e} < 1e-6，**逐位等价**")

    # α=0 的构造性保证：这是一条关于 α=0 的命题，不是关于初始化的断言，所以显式置 0 测
    with torch.no_grad():
        v2.alpha_on.fill_(-1e9)
        k, v = torch.randn(H, n, d), torch.randn(H, n, d)
        s0 = torch.randn(H, n)
        z = v2.delta(0, k, v, s0, None,
                     (s0.mean(-1), s0.std(-1), s0.std())).abs().max()
    print(f"[3] α=0 ⇒ max|Δs| = {float(z):.3e}　⇒ 与基线逐位相同（构造性保证）")
    print("\n验收通过。")


def verify_cache(a):
    """cache 层的端到端等价性：同一批分数与 KV 下，`valid` 掩码必须**逐位相同**。

    `verify` 只证明了打分器是同一个函数；真正决定保留哪些 KV 的是 `prune_chunk` 里
    阈值化的**顺序与口径**（τ 用未修正分数算、统计量取整块全量、`score > thres` 而非
    严格 topk）。那一层重写错了，打分器再等价也没用。

    **不需要 GPU 也不需要模型权重**：`RetainCache.__init__` 只用到 `model.parameters()`
    的 device/dtype 与 `config` 的三个层数字段，所以拿一个 stub 就能跑，`score` /
    `key_cache` / `value_cache` 直接手工填随机张量。用小维度（L=4,H=2）是因为这里测的
    是代码路径不是权重。
    """
    from attention.control_memory import ControlMemory
    from attention.learned_ctrlcache import LearnedControlRetainCache

    L, H, d_kv, d_m, n_tot, sink = 4, 2, 16, 32, 300, 8
    lo, hi = sink, n_tot

    class _Cfg:
        num_hidden_layers, num_attention_heads, num_key_value_heads = L, H * 3, H
        hidden_size, head_dim = H * 3 * d_kv, d_kv

    class _Stub(nn.Module):
        config = _Cfg()

        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.zeros(1))

    stub = _Stub()
    torch.manual_seed(7)
    cm = ControlMemory(d_kv, L, H, n_slots=4, d_m=d_m, mode="memoryless",
                       typed=True, alpha_init=0.9)
    v2 = V2Scorer(d_kv, L, H, n_slots=4, d_m=d_m, alpha_init=0.9)
    r = v2.load_state_dict(cm.state_dict(), strict=False)
    assert not r.missing_keys and sorted({k.split(".")[0] for k in r.unexpected_keys}) \
        == sorted(DEAD)

    score = [torch.randn(1, H, n_tot) for _ in range(L)]
    keyc = [torch.randn(1, H, n_tot, d_kv) for _ in range(L)]
    valc = [torch.randn(1, H, n_tot, d_kv) for _ in range(L)]

    def build(cls, **kw):
        c = cls(stub, (sink, n_tot), **kw)
        c.score = [s.clone() for s in score]
        c.key_cache = [t.clone() for t in keyc]
        c.value_cache = [t.clone() for t in valc]
        c.valid = None
        return c

    print(f"stub: L={L} H={H} n={n_tot} sink={sink}　（CPU，无需模型权重）")
    ok = True
    for ratio in (0.5, 0.2, 0.1, 0.05):
        A = build(LearnedControlRetainCache, ctrl=cm)
        B = build(V2RetainCache, scorer=v2)
        tA, rA = A.prune_chunk(ratio, (lo, hi), "pair")
        tB, rB = B.prune_chunk(ratio, (lo, hi), "pair")
        same = bool(torch.equal(A.valid, B.valid))
        flip = float((A.valid ^ B.valid).float().mean())
        ok &= same and abs(tA - tB) < 1e-9
        print(f"  ratio {ratio:<5} valid 逐位相同 {same}　不一致比例 {flip:.1e}　"
              f"thres {tA:.6f} vs {tB:.6f}　保留率 {rA:.4f}/{rB:.4f}")
    # 基线对照：scorer=None 必须与不带 ctrl 的父类完全一致（"不启用就是基线"）
    C = build(V2RetainCache, scorer=None)
    D = build(RetainCache)
    tC, _ = C.prune_chunk(0.1, (lo, hi), "pair")
    tD, _ = D.prune_chunk(0.1, (lo, hi), "pair")
    base_ok = bool(torch.equal(C.valid, D.valid)) and abs(tC - tD) < 1e-9
    print(f"  scorer=None 与原生 RetainCache 逐位相同 {base_ok}")
    assert ok and base_ok, "cache 层不等价"
    print("\ncache 层验收通过。")


def main():
    ap = argparse.ArgumentParser(description="VariKV v2 最小实现")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="与原 ControlMemory 的逐位等价性验收")
    v.add_argument("--ckpt", default="varikv/ctrl_b_a1_s0.pt/memoryless.pt")
    v.set_defaults(fn=verify)

    c = sub.add_parser("verify-cache", help="cache 层 valid 掩码的等价性（CPU）")
    c.set_defaults(fn=verify_cache)

    t = sub.add_parser("train", help="训练（需要教师 trace）")
    t.add_argument("--traces", default="scratch_ctrl_traces_v2")
    t.add_argument("--out", default="varikv/v2_clean.pt")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--n_pairs", type=int, default=256)
    t.add_argument("--slots", type=int, default=8)
    t.add_argument("--dim", type=int, default=128)
    t.add_argument("--d_kv", type=int, default=128)
    t.add_argument("--seed", type=int, default=0,
                   help="只控制初始化 / pair 采样 / 训练顺序")
    t.add_argument("--split_seed", type=int, default=42, help="只控制 train/val 划分")
    t.add_argument("--val_frac", type=float, default=0.25)
    t.add_argument("--alpha_init", type=float, default=1.0)
    t.add_argument("--freeze_alpha", action="store_true", default=True)
    t.add_argument("--no_freeze_alpha", dest="freeze_alpha", action="store_false")
    t.add_argument("--pair_w", default="linear", choices=["linear", "none"])
    t.add_argument("--lam_global", type=float, default=1.0)
    t.add_argument("--skip_first", action="store_true", default=True,
                   help="跳过每篇第一个 chunk 的损失（v2 原默认；memoryless 下纯丢数据）")
    t.add_argument("--no_skip_first", dest="skip_first", action="store_false")
    t.set_defaults(fn=train)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
