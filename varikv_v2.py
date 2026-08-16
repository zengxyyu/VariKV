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
v2 的**完整训练配方**（从 `scratch_ctrl_teacher_ext.sh` / `scratch_ctrl_train_run.sh`
考出来的实际命令，不是脚本的 argparse 默认值 —— 两者不一样）

**教师**（产出 `scratch_ctrl_traces_v2`，30 篇，每篇约 359 MB）：

    scratch_ctrl_teacher.py --n_cat 30 --out scratch_ctrl_traces_v2

其余全取默认：`Qwen2.5-7B-Instruct-1M` / gate `fastkvzip` / ratio **0.1** /
chunk **16000** / window **4096** / level `pair` / max_ctx **131072** /
target_len 256 / n_qpos 16 / **n_keep 256**（近阈值候选）+ **n_rand 512**（随机，
只喂 writer，本实现不读）/ min_prunes 2 / task `continuation` /
utility **`full_single`（= U^full，满缓存单 token 移除损伤）**。

注意 `max_ctx 131072` 与 `n_short 0 / n_long 10` 是**非退化门槛**逼出来的，不是随手
调的：`ratio × clen ≤ window` 时 `wrapper.py:271-277` 会把 chunk_ratio 置 0，驱逐退化
成"只留局部窗口"，**任何改分数的方法都恒为 no-op**。门槛是 `clen > 4096/0.1 = 40,960`，
而 `fineweb_10k` 最长才约 31k ⇒ 全部退化，只有 `fineweb_10k_cat` 能用。

**训练**（3 个种子，只变初始化/采样/顺序）：

    scratch_ctrl_train.py --traces scratch_ctrl_traces_v2 --epochs 40 \
        --seed {0,1,2} --split_seed 42 --alpha_init 1.0 --freeze_alpha \
        --pair_w linear --lam_global 1.0 --out varikv/ctrl_b_a1_s{S}.pt

未覆盖的默认值即实际值：`--slots 8 --dim 128 --d_kv 128 --lr 3e-4 --n_pairs 256
--val_frac 0.25`，优化器 AdamW(weight_decay=0.01) + `clip_grad_norm_(1.0)`。
原脚本一次训练**三条臂**（stateful / memoryless / shuffled）各存一个 `{mode}.pt`；
**v2 就是其中的 `memoryless.pt`**。

**本文件的 argparse 默认值已经全部对齐上面这份配方**，所以 `train` 不带任何参数就是
v2 的配置。但要留意：这些默认值与**原脚本**的 argparse 默认值不同（原脚本是
`epochs 8 / alpha_init 0.05 / freeze_alpha False / traces scratch_ctrl_traces`），
因为本文件提炼的是"真正拿到 +4.27 的那次运行"，不是复刻早期脚本。

| | v2 实际用的 | 原脚本默认 | 本文件默认 |
|---|---|---|---|
| epochs | 40 | 8 | **40** |
| alpha_init | 1.0 | 0.05 | **1.0** |
| freeze_alpha | 是 | 否 | **是** |
| traces | `..._v2` | `scratch_ctrl_traces` | **`..._v2`** |
| lr / n_pairs / slots / dim | 3e-4 / 256 / 8 / 128 | 同左 | 同左 |
| split_seed / val_frac | 42 / 0.25 | 同左 | 同左 |
| pair_w / lam_global | linear / 1.0 | 同左 | 同左 |

--------------------------------------------------------------------------------
四条**已知的、有意为之的**差异 —— 等价性验收覆盖不到它们，所以写在这里

1. **`train` 的随机初始化与原版不同，因此不会复现出 `ctrl_b_a1_s0.pt` 的权重。**
   `ControlMemory.__init__` 在 `q_read` 与 `head` 之间还构造 5 个 Linear + 1 个
   `mix` + 1 个 `GRUCell`，它们**消耗随机数**；本类直接跳到 `head`，于是同一个
   `--seed` 下 `head` 的初值不同。刻意"烧掉"同样多的随机数只会得到一份 bug 兼容的
   代码，没有收益。**结论：本文件 `train` 出来的是一个新种子，不是对 v2 的重新推导。**
   要用 v2 本身，`V2Scorer.from_ckpt(...)` 加载既有 ckpt。
   而这一点在本项目尤其要当真：`一次训练不是一次测量`——v1 同一份代码三次重训跨度
   39 分，所以任何新训练都必须 n≥3 种子并报跨种子散布。
2. **trace 缺 `gsig`/`sig_h`/`thres` 时硬失败，而原版会 warn 并退回子集统计。**
   那个回退正是 `delta` 里点名的危险路径（子集 σ 低估 ⇒ 部署扰动大于训练时学到的
   幅度），静默用错的 σ 比崩掉糟得多。`scratch_ctrl_traces_v2` 三个字段齐全，
   所以对 v2 无影响。
3. **`rho_max` 预算门控没有移植。** 加回只需在 `V2RetainCache.prune_chunk` 里把
   `if self.active:` 改成 `if self.active and ratio <= self.rho_max:`，并在
   `__init__` 里存下它。留空是因为它是 v2 之后的追加，默认 1.0 时逐位无操作
   （ratio ∈ (0,1] 恒 ≤ 1）—— 对历史 +4.27 无影响，但不能宣称"对
   `LearnedControlRetainCache` 的任意构造参数都行为相同"。
4. **`prune()` 改成抛错**，而原版会静默给出基线结果。方向是更安全，见该方法的说明。

--------------------------------------------------------------------------------
用法

    # 四级等价性验收。前三级在 CPU 上跑、不需要模型权重；第四级要 GPU。
    .venv/bin/python varikv_v2.py verify         # 打分器：60 次随机微分测试
    .venv/bin/python varikv_v2.py verify-cache   # cache：valid 掩码逐位相同（stub）
    .venv/bin/python varikv_v2.py verify-train   # 训练：loss 与随机数流同步
    .venv/bin/python varikv_v2.py verify-real    # 真 7B：valid 与生成文本都逐位相同

    # 为什么第四级不能省：前三级喂的都是 fp32 随机张量，而真实路径上 score 与 KV 是
    # **bf16**，`delta` 内部一路 fp32、最后 `.to(score0.dtype)` 降回 bf16，**降精度
    # 发生在阈值化之前**。原则上存在"两边的 Δs 舍入到不同 bf16 值 ⇒ 阈值处翻转不同"
    # 的可能，只有真跑能排除。

    # 评测：训练会同时写出 `*_compat.pt`（ControlMemory 形状），直接喂现有 harness，
    # 这样新实现产出的数字与历史结果仍在同一条路径上可比：
    #   cd external/FastKVzip/prefill && eval_chunk.py ... --ctrlm_ckpt <*_compat.pt>
    # **注意只覆盖了 `prune_chunk`**：`eval.py` 走的不分块 `prune()` 路径上 v2
    # 一声不响地不生效，评测一律走 `eval_chunk.py`。

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
        self.d_kv = int(d_kv)          # 只为 to_compat_ckpt 记账，不参与前向
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
            # **两种合法情形**：原版 ControlMemory 的 ckpt 会多出恰好那 8 个死模块；
            # 本文件自己训出来的 ckpt 一个都不多。首版只允许前者，于是加载自己的产物
            # 必然断言失败 —— 一个只在"训练→再加载"这条闭环上才暴露的 bug。
            assert got in ([], sorted(DEAD)), \
                (f"未使用的键既不是空集也不是预期的死代码。\n  预期 {sorted(DEAD)} 或 []\n"
                 f"  实得 {got}\n"
                 "  ⇒ memoryless 下的存活集合变了，简化的前提失效，必须重新核对")
        return m

    def to_compat_ckpt(self, path):
        """导出成 `ControlMemory` 形状的 ckpt，好让**现有** `eval_chunk.py --ctrlm_ckpt`
        直接吃。

        为什么需要：`eval_chunk.py` 用 `strict=True` 加载进 `ControlMemory`，缺那 8 个
        死模块会直接报错。而所有下游数字都是那条路径产出的，不能因为换了实现就断掉
        与历史结果的可比性。死模块填零是安全的 —— `verify` 已经证明 memoryless 下它们
        不参与前向，`ControlMemory.write` 与 `LearnedControlRetainCache._write` 都在
        第一行短路。
        """
        from attention.control_memory import ControlMemory
        # **ckpt 格式里没有 alpha_max 这一栏**，而 `eval_chunk.py` 建 ControlMemory 时
        # 用的是默认 1.0。`alpha = alpha_max·sigmoid(alpha_on)`，所以 alpha_max≠1 会在
        # 导出时被**静默丢掉**，部署的修正幅度与训练时不同。宁可在这里崩。
        assert self.alpha_max == 1.0, (
            f"alpha_max={self.alpha_max}，但 ckpt 格式无此字段、eval_chunk 恒用 1.0"
            "　⇒ 导出会静默改变 alpha。要支持请先给 ckpt 加字段并改 eval_chunk")
        cm = ControlMemory(self.d_kv, self.L, self.H,
                           n_slots=self.K, d_m=self.d_m, mode="memoryless",
                           typed=True)
        for n_, p in cm.named_parameters():
            if n_.split(".")[0] in DEAD:
                torch.nn.init.zeros_(p)
        miss = cm.load_state_dict(self.state_dict(), strict=False)
        assert not miss.unexpected_keys, miss.unexpected_keys
        assert sorted({k.split(".")[0] for k in miss.missing_keys}) == sorted(DEAD)
        torch.save(dict(state=cm.state_dict(), mode="memoryless", arch="memory",
                        slots=self.K, dim=self.d_m, d_kv=self.d_kv,
                        L=self.L, H=self.H), path)
        return path


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

    **`prune()`（不分块路径）被显式改成抛错。** 原版 `LearnedControlRetainCache`
    只覆盖 `prune_chunk`，于是走 `eval.py` 时父类会给出一个看不出异常的**基线**结果
    —— 静默失败。这是本实现**有意**与原版不等价的第四处，方向是更安全。
    评测一律走 `eval_chunk.py`。
    """

    def __init__(self, model, evict_range: Tuple[int, int], scorer=None):
        super().__init__(model, evict_range)
        self.scorer = scorer                    # None ⇒ 纯基线
        self.flip_frac, self.retain_delta, self.delta_std = [], [], []

    @property
    def active(self) -> bool:
        return self.scorer is not None and float(self.scorer.alpha) != 0.0

    def prune(self, *_a, **_kw):
        """**不分块的那条路径不支持，直接报错而不是静默退回基线。**

        `eval.py:104` 与 `eval_mrcr.py:67` 是全树仅有的两个 `prune()` 调用点，都不在
        v2 的工作路径上。父类的实现会跑出一个**完全正常的基线结果**，看不出任何异常
        —— 这正是本项目反复吃亏的静默失败类（原版 `LearnedControlRetainCache` 也有
        这个洞，只是从没人踩到）。宁可炸。
        """
        raise RuntimeError(
            "V2RetainCache 只支持分块预填（prune_chunk / eval_chunk.py）。"
            "eval.py 走的不分块 prune() 上 v2 不生效，父类会静默给出基线结果，"
            "所以这里直接报错。")

    def prune_chunk(self, ratio: float, evict_range: Tuple[int, int] = None,
                    level: str = "pair"):
        # 参数顺序与 `wrapper.py:302` 的 `kv.prune_chunk(ratio, (lo,hi), level)` 对齐。
        # 原版这里的默认值写成 `evict_range=tuple`（一个类型对象），漏传时会报
        # "cannot unpack type"；改成 None 只是让错误信息可读，调用方从不漏传。
        assert evict_range is not None, "evict_range 必传，形如 (start_idx, end_idx)"
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
        assert "gsig" in ch, "trace 缺 gsig（旧版教师）——见下面对 sig_h 的同一条说明"
        gsig = torch.tensor(float(ch["gsig"]), device=dev).clamp_min(1e-6)
        for l, pl in enumerate(ch["layers"]):
            # **原版在缺 `gsig`/`sig_h` 时会 warn 并退回"从手上这批候选现算"。
            # 本实现改成硬失败**，因为那个回退恰好就是 `delta` 的 docstring 里点名的
            # 危险路径：近阈值子集的 σ 被系统性低估，而 σ 直接乘在 Δs 上。
            # 静默用错的 σ 比崩掉糟得多。
            for _f in ("mu_h", "sig_h", "thres"):
                assert _f in pl, (
                    f"trace 缺字段 {_f}——这是旧版教师产出的。请用 "
                    "scratch_ctrl_teacher.py 重新采集，不要退回子集统计")
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
    # **`doc*.pt` 而不是 `*.pt`**，与原版 `scratch_ctrl_train.py:236` 一致。
    # 教师目录里以后一旦多出 `stats.pt` / `merged.pt` 之类，`*.pt` 会把它们当文档读
    # 进来，然后在 `doc["chunks"]` 上抛 KeyError —— 或者更糟，形状恰好能走通。
    files = sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))
    assert files, f"{a.traces} 里没有 doc*.pt"
    docs = [torch.load(f, map_location="cpu") for f in files]
    L, H = docs[0]["L"], docs[0]["H"]
    # **d_kv 从 trace 推**，不要信 argparse 的默认值：不一致时只会在 `x_proj` 里
    # 抛一个看不懂的形状错误，而不是在这里说清楚。
    d_kv = int(docs[0]["chunks"][0]["layers"][0]["k"].shape[-1])
    assert d_kv == a.d_kv, (f"trace 的 d_kv={d_kv} 与 --d_kv={a.d_kv} 不符；"
                            f"请传 --d_kv {d_kv}")
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
    # **只存可序列化的标量。** `set_defaults(fn=train)` 会把函数对象放进
    # `vars(a)`，而 torch 2.6+ 的 `weights_only=True`（默认）会因此拒收
    # 整个 ckpt —— 存得下、读不出，最坏的一种失败。
    args = {k: v for k, v in vars(a).items()
            if isinstance(v, (int, float, str, bool, type(None)))}
    torch.save(dict(state=m.state_dict(), mode="memoryless", arch="memory",
                    slots=a.slots, dim=a.dim, d_kv=a.d_kv, L=L, H=H,
                    args=args), os.path.join(ROOT, a.out))
    # **必须用 splitext，不能 str.replace('.pt', ...)。** `--out varikv/v2clean`
    # （无后缀）时 replace 是恒等映射 ⇒ 兼容版**覆盖掉刚存好的主 ckpt**；而
    # `a/b.pt/c.pt` 这种路径会被替换到错误的目录里去。
    _base = os.path.join(ROOT, a.out)
    _root, _ext = os.path.splitext(_base)
    _compat = _root + "_compat" + (_ext or ".pt")
    assert _compat != _base
    m.to_compat_ckpt(_compat)
    print(f"已保存 {a.out}　+ 兼容版 *_compat.pt（可直接喂 eval_chunk --ctrlm_ckpt）")
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

    # **随机微分测试，不是挑几个点看看。** 首版只测了 3 层、且 `margin` 恒不为 None、
    # `stats` 恒为 [H] 形状 —— 那样测不到 `margin=None`（`mg = zeros_like(z)`）和
    # `stats` 传 [H,1] 时的 `.view(-1,1)` 这两条分支。等价性断言必须覆盖被调用到的
    # 全部分支，否则"逐位等价"只是对那几个点成立。
    torch.manual_seed(0)
    H, d = sd["H"], sd.get("d_kv", 128)
    worst, worst_cfg, cov = 0.0, None, {"mg=None": 0, "mg=有": 0, "stats2d": 0}
    with torch.no_grad():
        for t in range(60):
            l = int(torch.randint(0, sd["L"], ()))
            n = int(torch.randint(8, 200, ()))
            k, v = torch.randn(H, n, d), torch.randn(H, n, d)
            s0 = torch.randn(H, n) * float(torch.rand(()) * 5 + 0.1) + \
                float(torch.randn(()))
            two_d = bool(torch.rand(()) < 0.5)          # stats 传 [H] 还是 [H,1]
            mu_h = s0.mean(-1, keepdim=True) if two_d else s0.mean(-1)
            sig_h = s0.std(-1, keepdim=True) if two_d else s0.std(-1)
            sig_g = s0.std()
            st = (mu_h, sig_h, sig_g)
            use_mg = bool(torch.rand(()) < 0.5)
            mg = ((s0 - float(torch.randn(()))) / sig_g) if use_mg else None
            cov["mg=有" if use_mg else "mg=None"] += 1
            cov["stats2d"] += int(two_d)
            # 原版路径：raw → feat/q_read/read → delta，与 learned_ctrlcache.py:98-108
            # 逐字一致（那里 q_read 被调用两次——一次给 q、一次在 read 内部——本实现
            # 只算一次，同一个 Linear 同一个输入，输出相同）
            xr = cm.raw(k, v)
            d_cm = cm.delta(cm.feat(xr), cm.read(cm.init_state(l), xr), s0,
                            q=cm.q_read(xr), margin=mg, stats=st)
            d_v2 = v2.delta(l, k, v, s0, mg, st)
            e = float((d_cm - d_v2).abs().max())
            if e > worst:
                worst, worst_cfg = e, (l, n, use_mg, two_d)
    print(f"[2] 60 次随机微分测试（层/长度/尺度/margin 有无/stats 形状全随机）")
    print(f"    覆盖：margin=None {cov['mg=None']} 次、margin=有 {cov['mg=有']} 次、"
          f"stats 传 [H,1] {cov['stats2d']} 次")
    assert worst < 1e-6, f"不等价：{worst_cfg} 处误差 {worst:.3e}"
    print(f"    max|Δs_orig − Δs_v2| = {worst:.3e} < 1e-6　⇒ **逐位等价**")

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


def verify_train(a):
    """训练路径的等价性 —— `verify` / `verify-cache` 都没覆盖到的第三块。

    只证明打分器与 cache 等价是不够的：`run_doc` 重写时任何一处顺序改动都会改变
    **随机数流**（每个非跳过的层 2 次 `randint`，每个 chunk 再 2 次），从而改变采到的
    成对样本，训练轨迹就分岔了。这里让原版 `scratch_ctrl_train.run_doc` 与本文件的
    `run_doc` 在**同一篇 trace、同一个 generator 种子、同权重**下各跑一遍，比较
    loss / 头内Δacc / 全局Δacc。

    `shuf_gen` 在原版里是喂给 `write()` 的独立 generator；memoryless 下 `write` 第一行
    就返回，**不消耗任何随机数**，所以本实现不建它也不影响 `gen` 的流 —— 这个测试正是
    用来证实这一点的，而不是靠读代码断言。
    """
    import glob
    import scratch_ctrl_train as orig
    from attention.control_memory import ControlMemory

    files = sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))
    assert files, f"{a.traces} 里没有 doc*.pt"
    doc = torch.load(files[0], map_location="cpu")
    L, H = doc["L"], doc["H"]
    torch.manual_seed(3)
    cm = ControlMemory(128, L, H, n_slots=8, d_m=128, mode="memoryless",
                       typed=True, alpha_init=1.0)
    v2 = V2Scorer(128, L, H, n_slots=8, d_m=128, alpha_init=1.0)
    v2.load_state_dict(cm.state_dict(), strict=False)   # 共权重，只比代码路径

    print(f"trace {os.path.basename(files[0])}　L={L} H={H} "
          f"chunks={len(doc['chunks'])}")
    ok = True
    for skip in (True, False):
        gA = torch.Generator().manual_seed(999)
        gAs = torch.Generator().manual_seed(555)        # 原版的 shuffle 流
        gB = torch.Generator().manual_seed(999)
        with torch.no_grad():
            A = orig.run_doc(cm, doc, "cpu", 64, gA, train=False,
                             lam_global=1.0, skip_first_loss=skip,
                             shuf_gen=gAs, pair_w="linear", replace=False)
            B = run_doc(v2, doc, "cpu", 64, gB, lam_global=1.0,
                        skip_first=skip, pair_w="linear")
        dl = abs(float(A[0]) - float(B[0]))
        d1, d2 = abs(A[2] - B[2]), abs(A[3] - B[3])
        ok &= dl < 1e-9 and d1 < 1e-9 and d2 < 1e-9
        print(f"  skip_first={str(skip):<5} loss {float(A[0]):.8f} vs "
              f"{float(B[0]):.8f}　|Δ|={dl:.2e}　头内Δacc |Δ|={d1:.2e}　"
              f"全局Δacc |Δ|={d2:.2e}")
    # generator 是否被消耗了同样多的随机数 —— 若两边流不同步，后续 epoch 会分岔
    gA = torch.Generator().manual_seed(999)
    gB = torch.Generator().manual_seed(999)
    with torch.no_grad():
        orig.run_doc(cm, doc, "cpu", 64, gA, train=False, shuf_gen=
                     torch.Generator().manual_seed(555))
        run_doc(v2, doc, "cpu", 64, gB)
    same_stream = bool(torch.equal(torch.randint(0, 10**6, (32,), generator=gA),
                                   torch.randint(0, 10**6, (32,), generator=gB)))
    print(f"  跑完后两边 generator 的后续抽样一致：{same_stream}"
          f"　⇒ 随机数流同步，多 epoch 不会分岔")
    assert ok and same_stream, "训练路径不等价"
    print("\n训练路径验收通过。")


def verify_real(a):
    """真模型上的端到端等价性 —— 前三个验收都用 stub / 随机张量，这一个用 7B 真跑。

    `verify-cache` 已经证明"同样的输入 ⇒ 同样的 `valid`"，但它喂的是 fp32 随机张量。
    真实路径里 `score` 是 bf16、KV 是 bf16、n≈16000、层数 28，而 `delta` 内部一路
    fp32、最后 `delta.to(score0.dtype)` 降回 bf16。**降精度发生在阈值化之前**，所以
    原则上存在"两边算出的 Δs 在 bf16 下舍入到不同值 ⇒ 阈值处翻转不同"的可能。
    只有真跑才能排除。

    比较两样东西，都要求逐位相同：`kv.valid` 掩码，以及最终生成的答案字符串。
    """
    from attention.control_memory import ControlMemory
    from attention.learned_ctrlcache import LearnedControlRetainCache
    from data import DataWrapper, load_dataset_all
    from model import ModelKVzip
    from utils import set_gen_length

    os.chdir(_P)
    m = ModelKVzip(a.model, "retain", "fastkvzip")
    ds = DataWrapper(a.data, load_dataset_all(a.data, m.tokenizer), m)
    set_gen_length(a.data, m)
    sd = torch.load(os.path.join(ROOT, a.ckpt), map_location="cpu")
    cm = ControlMemory(sd.get("d_kv", 128), sd["L"], sd["H"],
                       n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                       mode="memoryless", typed=True)
    cm.load_state_dict(sd["state"])
    cm = cm.to(m.device).eval()
    v2 = V2Scorer.from_ckpt(os.path.join(ROOT, a.ckpt)).to(m.device).eval()

    # 两臂都走**同一条** dispatch（`wrapper.py:206 kv_type == "control_learned"`），
    # 只在 v2 那一遍把模块属性临时换成一个 shim —— wrapper 是在分支**内部**才
    # `from attention.learned_ctrlcache import ...`，所以打模块属性有效。这样两臂
    # 除了 cache 类之外，连预填的调用路径都逐字相同。
    import attention.learned_ctrlcache as _lc
    _real = _lc.LearnedControlRetainCache

    class _Shim(V2RetainCache):
        def __init__(self, model, evict_range, ctrl=None, **_kw):
            super().__init__(model, evict_range, scorer=v2)

    m.kv_type = "control_learned"
    m.ctrl_module = cm
    ok = True
    for si in range(a.num):
        outs = {}
        for name in ("orig", "v2"):
            _lc.LearnedControlRetainCache = _real if name == "orig" else _Shim
            kv = ds.prefill_context(si, prefill_chunk=a.chunk,
                                    window_size=a.window, chunk_ratio=a.ratio,
                                    level="pair")
            assert type(kv) is (_real if name == "orig" else _Shim), \
                f"dispatch 没走到预期的类：{type(kv)}"
            _, txt = ds.generate_answer(si, kv, prob=False)
            outs[name] = (kv.valid.clone().cpu(), str(txt))
            del kv
            torch.cuda.empty_cache()
        _lc.LearnedControlRetainCache = _real
        same_v = bool(torch.equal(outs["orig"][0], outs["v2"][0]))
        same_t = outs["orig"][1] == outs["v2"][1]
        ok &= same_v and same_t
        diff = float((outs["orig"][0] ^ outs["v2"][0]).float().mean())
        print(f"  样本 {si}: valid 逐位相同 {same_v}（不一致 {diff:.2e}）　"
              f"生成文本相同 {same_t}", flush=True)
    assert ok, "真模型上不等价"
    print("\n真模型端到端验收通过。")


def main():
    ap = argparse.ArgumentParser(description="VariKV v2 最小实现")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="与原 ControlMemory 的逐位等价性验收")
    v.add_argument("--ckpt", default="varikv/ctrl_b_a1_s0.pt/memoryless.pt")
    v.set_defaults(fn=verify)

    c = sub.add_parser("verify-cache", help="cache 层 valid 掩码的等价性（CPU）")
    c.set_defaults(fn=verify_cache)

    w = sub.add_parser("verify-train", help="训练路径与随机数流的等价性（CPU）")
    w.add_argument("--traces", default="scratch_ctrl_traces_v2")
    w.set_defaults(fn=verify_train)

    rr = sub.add_parser("verify-real", help="真 7B 模型上的端到端等价性（需 GPU）")
    rr.add_argument("--ckpt", default="varikv/ctrl_b_a1_s0.pt/memoryless.pt")
    rr.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    rr.add_argument("-d", "--data", default="scbench_kv")
    rr.add_argument("--num", type=int, default=2)
    rr.add_argument("--ratio", type=float, default=0.1)
    rr.add_argument("--chunk", type=int, default=16000)
    rr.add_argument("--window", type=int, default=4096)
    rr.set_defaults(fn=verify_real)

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
