# VariKV 方法说明书（从代码反写）

> 写于 **2026-08-11**。**这份文件描述的是 `varikv/` 里真正跑着的实现**，不是设计意图。
> `theory_distributional_memory.md` §9–§11 是设计文档，两者已经分叉过多次（RoPE、
> 混合 ELBO、失真定义、方差坍缩都是先在代码里修好、文档才跟上）。
> **两者冲突时以本文件为准**，每一节末尾给出 `文件:行` 供核对。

---

## 0. 一句话

冻结的 LLM 在长上下文 prefill 时会驱逐一部分 KV。VariKV 不把它们丢掉，而是**编码进一组
高斯槽**（每个槽是一个 latent 空间的对角高斯），用**贝叶斯精度加权**更新，读出时**解码回
若干条 effective KV**（或作为注意力输出的残差），从而补偿驱逐造成的信息损失。

---

## 1. 符号与实际维度

以 Qwen2.5-7B-Instruct-1M 为例（`L=28` 层，`H=4` 个 kv head，`d_head=128`）：

| 符号 | 含义 | 值 | 来源 |
|---|---|---|---|
| `G` | 记忆组数 = `L × H`，**每个 (层, kv头) 一份独立记忆** | 112 | `train.py:build` |
| `d_kv` | 记忆的输入/输出维度 = **`2 × d_head`**（k 与 v 拼接） | 256 | 同上，随模型变；Gemma3 的 `d_head=256` ⇒ `d_kv=512` |
| `d_z` | **latent 维度**，槽里存的高斯的维度 | 64 | `config.py:28 d_latent` |
| `h` | 编码器/解码器的 MLP 隐藏宽度，**与模型无关的超参** | 256 | `config.py:56 d_hidden` |
| `K` | 每组的槽数 | 16 | `config.py:27 num_slots` |
| `T` | 每个槽读出几条 effective KV | 1 | `config.py:89 tokens_per_slot` |
| `N` | 当前 chunk 里被驱逐的 KV 条数（每组各不相同，补零对齐） | ~1e4 | — |

**注意 `d_kv` 与 `h` 在这个模型上恰好都等于 256，纯属巧合**，ckpt 里的 `(256,256)` 因此有歧义。
换成 Gemma3 就会分开（512 vs 256）。

记忆状态（每组）：`μ ∈ R^{K×d_z}`、`logvar ∈ R^{K×d_z}`、`var_content ∈ R^{K×d_z}`、
位置质心 `pos ∈ R^K`、位置精度 `_pos_tau ∈ R^K`。**均值状态大小 = K·d_z = 1024 个数**。

---

## 2. 编码器和解码器到底是什么

它们是**两个很小的 MLP**，是整个方法里仅有的两个"网络"，共 0.33M 参数（骨干 LLM 全程冻结）。

**编码器 = 变分推断里的识别网络 `q_φ(z | e)`**（`memory.py:69-73`）：

```
e_i ∈ R^{d_kv}           证据 = 第 i 条被驱逐 KV 的 [k_i ; v_i] 拼接
h_i = GELU(W₂ GELU(W₁ e_i))              W₁: d_kv→h,  W₂: h→h
μ^q_i     = W_μ h_i                       W_μ: h→d_z
logvar^q_i = lo + (hi−lo)·σ(W_σ h_i)      软约束到 [−4, 4]
```

它把一条 256 维的 KV 压成一个 **64 维 latent 空间上的对角高斯** `N(μ^q_i, diag e^{logvar^q_i})`。
"压缩"和"给出不确定性"是同一步完成的——这就是它相对"直接存 KV 均值"的全部差别。

`logvar` 用软约束不用硬 clamp（`memory.py:157`）：硬 clamp 在边界梯度为 0，实测 60 步后
**99.3% 的槽 logvar 焊死在下界**，方差失去动态范围，`dist` 相对 `point` 的优势会被抹平。

**解码器 = 生成模型 `p_θ(e | z)`**（`memory.py:82-87`），把 latent 还原成 KV：

```
ê = W₃ GELU(W₂' GELU(W₁' [z ; logvar]))    输入 2·d_z=128 → h → h → T·d_kv
```

两点非默认设计：
- **`logvar` 显式进解码器**（`logvar_into_decoder=True`）：让"这个槽有多确定"直接参与读出，
  而不只影响写入。
- **非线性解码器**（`nonlinear_decoder=True`）：线性解码器 + 单高斯先验会让整个系统退化成
  线性高斯模型，**卡尔曼闭式可解**，摊销变分推断就失去存在理由。这是与 Kalman 系竞品
  （Memory by Design、Kalman Linear Attention）唯一的理论分界线，不能关。

---

## 3. 整体流程（每个 prefill chunk）

```
   chunk 前填 ──► 门控打分 ──► 驱逐决策
                                  │
                     保留的 KV ────┴──── 被驱逐的 KV  e_{1..N}
                          │                    │
                          │              ┌─────▼─────┐
                          │              │  absorb   │  §4
                          │              │ 写入记忆   │
                          │              └─────┬─────┘
                          │                    │
                          │              ┌─────▼─────┐
                          │              │   read    │  §5
                          │              │ 读出补偿   │
                          ▼                    ▼
                     ┌──────────────────────────────┐
                     │  注意力（两种接法，见 §5）      │
                     └──────────────────────────────┘
```

驱逐决策**不是我们做的**——用的是 FastKVzip released 的门控（`-g fastkvzip`）。
我们只接管"被扔掉的那部分怎么办"。§6 的自由能 `F_i` 是一个**可选**的替代驱逐准则，
stage-1 实测它打不过随机，目前不启用。

---

## 4. 写入（absorb）

### 4.1 混合先验的分配权重

槽构成一个 `K` 分量的混合先验。观测 `i` 归属各槽的权重用**余弦相似度 + softmax**
（`memory.py:182-195`）：

```
w_ik = softmax_k( ⟨ μ^q_i/‖μ^q_i‖ , μ_k/‖μ_k‖ ⟩ / τ_prior ),   τ_prior = 1
```

**绝不先把 K 个高斯平均成一个再算 KL** —— 那会抹掉多峰性，模型塌回共轭可解的单高斯。

### 4.2 到混合先验的 KL

```
KL_i = Σ_k w_ik · KL( q_i ‖ N(μ_k, diag e^{logvar_k}) )  +  [ log K − H(w_i) ]
       └──────────── 条件项（Jensen 上界）────────────┘     └── 分配项 ──┘
```

其中对角高斯 KL（`memory.py:gaussian_kl`）：

```
KL(q‖p) = ½ Σ_d [ logvar_p − logvar_q + e^{logvar_q−logvar_p} + (μ_q−μ_p)²e^{−logvar_p} − 1 ]
```

条件项是 `KL(q ‖ Σ_k w_k p_k)` 的闭式上界（蒙特卡洛验证过：真值 6.694 ≤ 上界 8.535）。
分配项 `log K − H(w) ≥ 0` 是 `KL(q(c|e) ‖ p(c))`，`p(c)` 取均匀——用 data-dependent 的
responsibility 就必然要付这个码率。**它是"K 越大越差"的头号嫌疑**：最小化它等于最大化
责任分布的熵，把写入摊平到所有槽上。开关 `config.py:72`。

### 4.3 写入门：分配与强度解耦

```
z_i  = ( KL_i − mean_chunk(KL) ) / std_chunk(KL)        chunk 内 z-score，detach
η_i  = σ( α·z_i − β ),   α=2.0, β=0.0                   标量：这个观测总共写多少
gate_ik = w_ik · η_i                        ⇒  Σ_k gate_ik = η_i ≤ 1
```

三处都是踩过坑才成这样的（`memory.py:280-311`）：

- **用 chunk 内 z-score 而不是 KL 绝对值**：KL 的量级随记忆演化跨 4 个数量级（实测
  0.05 → 1589），任何固定 (α,β) 都会饱和，门从未工作在敏感区。
- **`η` 与 `w` 解耦**：混进同一个 sigmoid 会让行和到达 K（实测 0.66–16.0），
  即一个观测以全强度写入所有槽——同一份信息的精度被重复计 K 次，
  直接违反"独立观测精度可加"这个贝叶斯更新赖以成立的前提。
- **`std` 必须 `unbiased=False`**：`torch.std` 在 n=1 时返回 NaN，而 `NaN.clamp_min` 仍是 NaN。
  上下文长度 ≡ 1 (mod chunk) 时最后一块只有 1 个 token，一次就把 57,344 个记忆元素全污染。

### 4.4 精度加权的贝叶斯更新

观测精度 `τ^obs_i = e^{−logvar^q_i}`，槽的旧精度 `τ^old_k = e^{−logvar_k}`：

```
τ_k ← γ·τ^old_k + s·Σ_i gate_ik · τ^obs_i                    γ = 0.95
μ_k ← ( γ·τ^old_k·μ^old_k + s·Σ_i gate_ik·τ^obs_i·μ^q_i ) / τ_k
```

- **遗忘因子 `γ=0.95`**：否则流式吸收下 `τ` 无界累加，记忆过度自信、拒绝一切新写入，
  "事实被改写"型样本必然做错。
- **有效样本量缩放 `s = min(1, n_eff_max / Σ_i gate_ik)`，`n_eff_max = 1`**：一个 chunk 驱逐的
  是**相邻且高度相关**的 KV，当成那么多独立观测是典型的伪重复。实测每次吸收每槽累计约 8，
  稳态 `τ = 8/(1−γ) = 160`，`1/τ ≈ 0.006`，方差被压死在下界。
  一阶二阶矩**同步缩放**，所以 `μ` 的更新完全不受影响，只压住精度增长。

### 4.5 槽方差 = 估计不确定性 + 内容离散度

```
S²_k ← γ·τ^old_k·(μ^old_k² + var_content_k) + s·Σ_i gate_ik·τ^obs_i·(μ^q_i)²
var_content_k = max( S²_k/τ_k − μ_k², 0 )
σ²_k = 1/τ_k + var_content_k
logvar_k = clamp( log σ²_k, −4, 4 )
```

只用 `1/τ` 是错的：那等于断言"观测越多越确定"，但一个槽概括几百个**内容各异**的 token 之后
理应更**不**可靠。实测只用 `1/τ` 时 98.3% 的槽 logvar 焊死在下界。
验证：内容一致 ⇒ logvar −3.05；内容分散 ⇒ −2.53。

### 4.6 位置质心与 RoPE

缓存里的 key 是 **post-RoPE** 的，而槽是加权平均，`R_p` 不对加法封闭：

```
α·R_p k + (1−α)·R_{p'} k' ≠ R_φ( α·k + (1−α)k' )
```

所以吸收前先**逆旋回位置无关的帧** `R_p^{-1} k`，读出时再旋到该槽的**位置质心**
（EPL 2409.14364 的 UPL 最优解）。质心用与 `μ` 相同的精度权重更新：

```
pos_k ← ( γ·τ^pos_k · pos_k + Σ_i gate_ik·τ̄^obs_i·p_i ) / ( γ·τ^pos_k + Σ_i gate_ik·τ̄^obs_i )
```

不做这一步的后果**不只是精度**：同一个槽与 query 的内积会随 query 位置在 −17…+13 之间乱摆，
且相位混合会**以与认识论无关的原因抬高 σ²**——那样方差就不再度量论文声称的东西。

---

## 5. 读出（两种接法）

### 5.1 KV 注入（旧，已被证伪）

```
ẑ_k = μ_k  (+ 训练时重参数化采样 μ_k + ε·e^{logvar_k/2})
[k̂_k ; v̂_k] = Decoder([ẑ_k ; logvar_k]),  再按 pos_k 正旋回 RoPE
```

把这 `K·T` 条 effective KV **当成额外的 token 塞进 cache**，参与后续每一次 softmax。
**实测在 11 个数据集上全线为负**（ratio 0.2 处 −3 到 −49），原因是它们会抢走注意力质量。

### 5.2 输出端门控残差（现行）

```
logits = q · k̂ᵀ / √d_head          只在 M 个槽内部归一化
m(q)   = softmax(logits) · v̂
o      = o_attn + σ(g_{layer,head}) · m(q)
```

`g` 每 (层, kv头) 一个标量，初值 −4（`σ ≈ 0.018`）。记忆**不进 cache、不参与真实 KV 的
那次 softmax**，所以不抢注意力质量；`g → −∞` 时精确退回基线。
（`memory.py:56`、`external/FastKVzip/prefill/attention/attn.py:149`、`memcache_retain.py:245`）

**已知缺陷**：`attn.py:149` 的调用**无条件**执行，没有"还没吸收过任何东西就跳过"的判断，
所以空记忆也在往输出里注入。见 CLAUDE.md 2026-08-11 节。

---

## 6. 自由能 `F_i`（可选的驱逐准则，当前不启用）

```
F_i = D_i + λ·KL_i,   λ = 0.3
D_i = ā_i² · ‖v_i − v̂_i‖²
ā_i = 期望注意力，对数正态近似：ā_i ∝ exp( μ_q·k_i/√d + k_iᵀΣ_q k_i/(2d) )
```

`D_i` 是"把 KV_i 换成记忆重建后，注意力输出的扰动平方范数"——注意力输出空间的失真，
与 `absorb` 的 ELBO 重建项**同一个失真定义**（这是"一个标量统一两个决策"成立的前提）。
`λ=0` 时 `F` 退化成 Expected Attention。

排序前两项各自除以自己的**running std**（`f_normalize="running"`）：只做量纲归一化不够，
排序由**离散度**决定——实测 `std(D_n)≈0.69` 恒定而 `std(KL_n)` 从 2e-4 长到 7e-2，
`F` 的排序 99% 由 `D` 决定，正是 Expected Attention 退化。

评测时 `F` 不精确计算，而是由一个**摊销预测器**给出（这是效率故事的核心）。
它蒸馏 `F` 的**组内归一化秩**而不是值——`F` 的分布峰度 702、96.4% 的 token 落在 |z|<0.1，
用值做目标时 Huber 会让"恒输出 0"接近最优（实测 loss 0.0419 vs 平凡解 0.0421）。

**现状**：修好预测器后 ρ(pred, exact) = 0.78，但用 `F` 驱逐**比 recency 更差**；
而在 stage-1 上**随机驱逐打赢全部十种准则**，说明该任务无法判别驱逐策略。
Error Certificates (2607.21475) 从理论上支持这个观察（确定性 top-k 下误差不可辨识）。

---

## 7. `point` 与 `dist` 的唯一差别

两档**结构、参数量、计算图完全相同**（已验证参数量相等），只有三处取值不同：

| | `dist` | `point` |
|---|---|---|
| 后验方差 | 网络输出 | 常数 `logvar_init` |
| 写入强度 `η` | `σ(α·z(KL) − β)`，由 surprise 决定 | 可学习标量 `σ(point_gate_logit)` |
| `τ^obs`, `τ^old` | `e^{−logvar}` | 全 1 |

这保证消融的自变量只有"方差是否携带信息"。**实测 11 个数据集里 `dist` 输给 `point` 10 次。**

---

## 8. 参数量与容量核算

```
encoder   d_kv·h + h·h            = 256·256 + 256·256   = 131,072
to_mu/σ   2·h·d_z                 = 2·256·64            =  32,768
decoder   2·d_z·h + h·h + h·(T·d_kv) = 128·256+256·256+256·256 = 163,840
slot init 2·K·d_z                 = 2·16·64             =   2,048   ← 唯一依赖 K
residual_gate  G                  = 112
                                                   合计 ≈ 331k
```

**只有 0.6% 的参数依赖 K**，编码器/解码器/门都与 K 无关。

容量对比（每 (层, kv头)，均值状态）：

| | 状态大小 |
|---|---|
| 我们（μ，K=16, d_z=64） | **1,024** |
| 我们（μ+logvar+var_content） | 3,072 |
| RetentiveKV（`S ∈ R^{d×d}`, d=128） | 16,384 |

2026-08-11 探针实测：这 16 个槽最多只能修补被驱逐造成的注意力输出损伤的 **11–15%**，
且信号集中在最后 3 层（layer 26 的 R_opt 31.3%，其余 22 层 < 1%）。

---

## 9. latent 还能怎么改（按改动成本排序）

| 方案 | 改什么 | 状态大小 | 代价 | 保住 O(1)？ |
|---|---|---|---|---|
| **加大 `d_z`** | `config.py:28`，64 → 256 | ×4 | 编码/解码参数 ×4（0.33M→1.3M，仍极小）；**不触碰任何 K 相关逻辑** | ✔ |
| 加大 `T` | `tokens_per_slot` 1 → 4 | 读出宽度 ×4 | 解码器末层 ×4；KV 注入下会占预算，残差下不占 | ✔ |
| 加大 `K` | `num_slots` | ×K | 需先解决 assignment-KL；分配矩阵 `[N,K]` 显存需分块 | ✔ |
| `K` 随长度（√N / 1/c） | 新增调度 + K-无关的槽初值 | 随 N | 上面全部 + 初值要改成 `μ_base + f(k/K)` | ✘ |
| **换成矩阵值联想记忆** | 重写 absorb/read | ~d×d | 最大；分布式故事要重讲（矩阵均值+对角高斯） | ✔ |

**`d_z` 是成本最低的那个旋钮**：状态大小与编码器容量同时上去，而分配 softmax、
assignment-KL、精度累积那套 K 相关的逻辑一行都不用动。
残差读出下记忆不进 cache，所以**这些扩容在 KV 预算上都是免费的**，只花算力。

---

## 10. 已知与理论的偏离

- **驱逐粒度有三种实现，别混淆**（2026-08-11 核实）：

  | 实现 | 用在哪 | 驱逐决策 | 物理布局 |
  |---|---|---|---|
  | `varikv/cache.py` | stage 1（1.5B 合成针）、stage 2a（fineweb LM） | **逐 token**：一个 token 的 KV 在所有层/头上同时保留或降级 | 矩形 |
  | `memcache_retain.py` | **Stage 2b 现行默认**（`--varikv_kv_type` 默认值），全部 `gap_*` / `res` ckpt 的训练与评测 | **逐 (层, kv-head)**：`level="pair"` 在所有层所有头上全局取阈值，各头保留数不同（`:143` 的 `drops = [(~vmask[h]).nonzero() ...]`） | **矩形 + `valid` 掩码，物理不删**；吸收时各头驱逐数不同，补零凑矩形并用 `valid` 把 padding 排除出统计 |
  | `memcache.py` | 早期版本，现未使用 | 逐 (层, kv-head) | **变长扁平** `[Σ_heads len_k_head, dim]`，物理删除，需 `pos_track` |

  要点：**逐 head 的驱逐语义**与**变长的物理布局**是两件独立的事。`RetainCache` 拿到前者
  却不需要后者，因为它不真删只掩码——这正是 `memcache_retain.py` 简单得多的原因
  （矩形下"原始位置 = 序列下标"，整套 `pos_track` 机器都省掉）。
- **`D_i` 只算 v 的扰动**，k 的扰动只通过 softmax 二阶影响输出，略去。
- **query 统计只取第 0 层的 `q_proj`** 作为所有层的代理，避免多跑一次前向。
- **"贝叶斯 surprise"用词不严格**：代码算的是 `KL(q(z|e) ‖ p(z|M))`（观测与记忆的分歧），
  而 Itti & Baldi 的定义是 `KL(后验‖先验)`（记忆改变了多少）。相关但不同。
- **`B > 1` 不支持**（有断言）；位置跟踪、逐层状态切片、padding 都假设 B=1。

---

## 11. 代码索引

| 内容 | 位置 |
|---|---|
| 编码器 / 解码器 / 槽初值 | `varikv/memory.py:69-98` |
| `reset` / `detach_state` | `varikv/memory.py:120-151` |
| `encode`（识别网络） | `varikv/memory.py:171` |
| `get_prior`（分配权重） | `varikv/memory.py:182` |
| `kl_to_mixture`（含分配项） | `varikv/memory.py:197` |
| `absorb`（写入全过程） | `varikv/memory.py:241-420` |
| `read` / `read_precision` | `varikv/memory.py` 尾部 |
| 残差门 `residual_gate` | `varikv/memory.py:56` |
| 残差读出的挂钩点 | `external/FastKVzip/prefill/attention/attn.py:149` |
| 残差读出的实现 | `external/FastKVzip/prefill/attention/memcache_retain.py:245` |
| `F_i` / `D_i` / 期望注意力 / 预测器 | `varikv/free_energy.py` |
| RoPE 逆旋与重旋 | `varikv/rope.py` |
| 全部超参 | `varikv/config.py` |
