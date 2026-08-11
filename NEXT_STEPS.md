# VariKV 下一步：优化路线与待办

> **v3**，2026-08-11（当天三轮外部评审后修订）。
> 目的：把"接下来做什么、为什么、做到什么程度算过关"固定下来，避免忘记或反复重推。
> 相关：`CLAUDE.md`（实验记录）、`varikv_method_spec.md`（方法的代码级公式）、
> `kv_inference_acceleration_2026.md`（竞品全景）。
> **v3 的主要变化**：`1−λ` 从"单一生死判据"降级为三因子分解的一项；新增 joint-Gaussian
> MGF 这条方向（它让方差解析地进入 attention 等式）；把实验重排成一条**免训练优先**的阶梯，
> 消除 v2 里"表示/代数/目标同时改"的混淆。

---

## 0. 研究问题（拆成两级，不要混在一起）

**H1 — 表示假设**（先做，可独立成为技术贡献）：

> 在同预算、同驱逐、同修正代数下，**被驱逐 key/value 几何的分布式摘要**能否比
> 点质心或一阶矩摘要更准确地重建遗漏的 softmax 质量与方向？

**H2 — 不确定性假设**（仅在 H1 成立后做）：

> 给定同一个分布式估计器，**重建的预测不确定性**能否改善"何时/多大程度信任残差"？

当前 16 高斯槽 + 加性残差降级为 **prototype-memory baseline**（见 §11 叙事进阶）。
不要再问"怎么把它调好"。

---

## 1. 精确恒等式与三因子分解

设保留集 `R`、被驱逐集 `E`，`D_X(q)=Σ_{i∈X}e^{q·k_i/√d}`，`N_X(q)=Σ_{i∈X}e^{q·k_i/√d}v_i`，
`o_X=N_X/D_X`。**无需任何核近似**：

```
o_full = λ·o_R + (1−λ)·o_E,        λ = D_R/(D_R+D_E)
Δo     = o_full − o_R = (1−λ)·(o_E − o_R)
```

### 1.1 三因子分解 —— 不要只看质量

```
G(q) = ‖Δo(q)‖ = M(q) · C(q)

M(q) = 1 − λ(q) = D_E/(D_R+D_E)        遗漏的 softmax 质量
C(q) = ‖o_E(q) − o_R(q)‖               驱逐-保留方向/内容反差
```

**`M` 小不等于影响小。** 反例：`o_R=[1,0]`、`o_E=[−10,0]`、`M=0.03` ⇒ `Δo=[−0.33,0]`，
相对 `‖o_R‖` 是 33%。**MomentKV 的核心观察正是这一点**：性能下降主要来自保留集与驱逐集的
**方向不匹配**，即便残余质量很小也可能造成不成比例的输出方向变化。

`M` 提供的是一个**有效但不充分**的上界（`o_E`、`o_R` 都是 value 的凸组合）：

```
‖Δo‖ ≤ 2·M·max_i‖v_i‖
```

所以 `M` 既不是 kill switch，也不是无关量。**不要重复 `gap MSE` 那次的错误——
不要让任何单一指标解释一切。**

### 1.2 还要再乘一层：任务敏感度

局部 `G` 大也不一定影响最终 logits。定义

```
T(t) = KL( p_full(·|x≤t) ‖ p_FastKVzip(·|x≤t) )
```

真正值得记忆去学的 query 是 **`M` 非平凡 ∧ `C` 高 ∧ `T` 高**。
研究对象是 **Mass × Contrast × Task-sensitivity** 这个三层结构。

### 1.3 关于"加性残差是否错"——精确表述

**形式本身没错**：定义 `m*(q)=(1−λ)(o_E−o_R)` 则 `o_full=o_R+m*` 精确成立。
错的是我们那个具体的 `m(q)`：不知道 `D_E`、不知道 `D_R`、门 `σ(g_{l,h})` 是 query 无关常数、
读出落在 16 个学习向量的**凸包**内（非子空间）。

论文写法：*post-hoc additive correction hides the softmax normalization structure and forces a
compact memory to implicitly approximate a retained-cache-dependent residual function.*
**不要**写 "additive residual is mathematically incorrect"。

---

## 2. 两条候选表示：核近似 vs 高斯矩

### 2.A 联想状态 `(S,z)`（核近似路线）

```
N_E(q) ≈ φ(q)ᵀS,  S = Σ_{i∈E}φ(k_i)v_iᵀ
D_E(q) ≈ φ(q)ᵀz,  z = Σ_{i∈E}φ(k_i)
```

**`exp(qᵀk)` 没有有限维精确特征映射**，所以只能写"**在指数核的正定特征近似下**，遗漏的
分子与分母允许一个固定大小的联想充分状态 `(S,z)`"。**不要**写 "sufficient statistics for
softmax attention"。

### 2.B joint-Gaussian 指数矩（新增，v3；更贴合"分布式记忆"的初衷）

把 `E` 划分为若干 cluster。设某 cluster 内 `(k,v)` 联合高斯，`a = q/√d`。则**解析地**：

```
E[e^{aᵀk}]     = exp( aᵀμ_k + ½aᵀΣ_kk a )
E[v·e^{aᵀk}]   = ( μ_v + Σ_vk a )·exp( aᵀμ_k + ½aᵀΣ_kk a )
```

（第二式可由指数倾斜/Esscher 测度验证：倾斜后 `k` 的均值移到 `μ_k+Σ_kk a`，
代入 `E[v|k]=μ_v+Σ_vk Σ_kk^{-1}(k−μ_k)` 即得。两式均已核对正确。）

于是每个 cluster `c`（含 `n_c` 个 token）：

```
D̂_c = n_c·exp( aᵀμ_k + ½aᵀΣ_kk a )
N̂_c = D̂_c·( μ_v + Σ_vk a )
D̂_E = Σ_c D̂_c ,   N̂_E = Σ_c N̂_c
ô   = (N_R + N̂_E) / (D_R + D̂_E)
```

**为什么这条路线值得优先试**：

| 需求 | 由谁满足 |
|---|---|
| 正确的 softmax 代数（分子 + 分母） | `ô` 的形式 |
| query 依赖 | `a = q/√d` 出现在指数里 |
| key-value 关联（非无条件平均） | `Σ_vk` |
| **方差解析地进入 attention 等式** | `Σ_kk` 进分母、`Σ_vk` 进分子 |
| 干净的 dist vs point 消融 | **只设 `Σ_kk=0, Σ_vk=0`**，其余一切不变 |
| 可以完全免训练 | 全是流式统计量 |

这比"高斯 latent 原型"强得多：随机变量从 `z_latent` 换成 `(k,v)|cluster`，
方差第一次有数学上的作用而不是"存着但不知为何有用"。

**三个必须先解决的问题**：

1. **MGF 对分布误设是指数敏感的。** `exp(½aᵀΣ_kk a)` 若 cluster 内 key 实际重尾而非高斯，
   会**严重高估** `D̂_E`，进而错误压低保留贡献。**必须逐 cluster 比较 MGF 估计与真实 `D_E`，
   看误差分布而非均值**（见 P1 阶梯）。缓解手段：截断、逐 cluster 校正因子、或用少量保留样本
   估经验 MGF。
2. **谁定义 cluster？** 这是未定的设计变量：顺序分块 / key 上 k-means / 注意力引导。
   cluster 数即预算旋钮。实现前必须定下来（ResKV 用的是 cluster + `log c_j` 恢复群体质量）。
3. **协方差的存储必须压。** `Σ_kk` 全矩阵是 128×128/cluster，太大。第一版：
   `Σ_kk = diag(s_k²)`（128 floats）；`Σ_vk = UVᵀ`，`U,V ∈ R^{128×r_c}`，`r_c = 4~8`，
   于是 `Σ_vk a = U(Vᵀa)`，成本 `O(d·r_c)`。

### 2.C 一个必须提前分清的概念

`Σ_kk` 是**族内离散度（aleatoric）**，不是**我们对 `μ_k` 的估计不确定性（epistemic）**。
前者进入均值估计器（上面的公式）是正确的；**后者才能用来做 confidence gate**，需要另外估
`Var[N̂_E], Var[D̂_E]`。**这正是旧方法最容易混淆的地方，未来论文必须写清。**

---

## 3. P0 —— 诊断（半天到一天，不写新方法）

### P0-A 修 empty-memory 注入 bug 【无需讨论，先做】

`attn.py:149` 无条件调用 `memory_residual`，空记忆照样注入。证据：同一 ckpt 跨独立 job 的
ratio-1.0 分数逐字相同、不同 ckpt 之间不同（68.20 / 66.80 / 68.60 / 67.20 / 67.80 / 70.40）。
修法：`if self._absorbed_upto == 0: return 0`。

### P0-B Mass–Contrast–Gap–Sensitivity 探针

先只在 **`scbench_kv` @ ratio 0.1**、20–30 个样本上做。逐
`(layer, kv_head, query_head, query_position)` 记录：

```
M = 1−λ ,   C = ‖o_E − o_R‖ ,   G = M·C ,   T = KL(p_full ‖ p_FastKVzip)
```

**四条实现要求，缺一不可：**

1. **用 logsumexp 算，不要算 `exp(lse)`。** 超长上下文下会溢出且不必要：
   ```
   L_R = LSE(s_R), L_E = LSE(s_E), L_F = logaddexp(L_R, L_E)
   M = exp(L_E − L_F)
   ```
2. **必须逐 query-head，不能只逐 kv-head。** Qwen2.5-7B 是 GQA，28 个 query head 共享
   4 个 kv head（7:1）。`D_E(q)` 是 **query-head 依赖**的；把 7 个 q-head 平均掉可能淹掉
   强受影响的那个。记录到 `(l, h_kv, h_q, t)` 再聚合。
3. **必须用模型实际的 attention score**：`s_i = q·k_i/√d + mask/bias`，包含 scale、causal
   有效性、**post-RoPE** 的 q/k、GQA head 映射，以及保留集的**实际**定义
   （FastKVzip 的 local window + sink 保护，即 `self.valid`）。
4. **不要只报均值。** `M` 可能极度重尾：99% 的 query 是 0.001、1% 是 0.7，均值 0.008 看着
   微不足道，但 SCBench 检索的成败可能恰在那 1%。必须报
   **median / P90 / P95 / P99 / max + 逐层 heatmap + 条件分布 `E[M | T ∈ top 10%]`**。

**三张图比任何单一均值都重要：**

| 图 | x | y | 颜色 |
|---|---|---|---|
| 1 | `M` | `C` | `T` |
| 2 | `M` | `G` | — |
| 3 | `G` | `T` | — |

图 1 直接回答：**FastKVzip 删掉的是"低质量 + 低反差"的垃圾，还是"低质量但高反差"的信息？**

### P0-C SVD 谱（diagnostic，不作判据）

对 `Y = O_full − O_R ∈ R^{T×d}` 逐 (层, kv-head) 做 SVD，报
`E(r)=Σ_{j≤r}σ_j²/Σσ_j²`，`r=1,4,8,16,32,64,128`，出 heatmap。

**解释规则（易用错）**：`E(16)` 低 ⇒ 秩是硬约束、当前槽结构必死；
`E(16)` 高 ⇒ **推不出任何结论**，因为读出落在 16 顶点凸包内，比"秩≤16 的子空间"约束强得多。

---

## 4. P1 —— 免训练的局部统计 oracle（不跑 LLM 下游）

**先证明统计表示本身有信号，再决定要不要训练。** 顺序不能反。
如果解析统计量就能 work，我们根本不需要那 0.33M 的 MLP —— 对一篇 inference 论文这是**优势**。
MomentKV 已证明免训练矩摘要是强 baseline。

在真实被驱逐集合上，逐级降级近似，比较 `D̂_E`、`N̂_E`、`ô` 的误差：

| 级别 | 状态 | 说明 |
|---|---|---|
| E0 | 完整 `E` 的精确 `(N_E,D_E)` | **sanity check**，必须数值上恢复 `o_full` |
| E1 | cluster 质心 `(n, μ_k, μ_v)` | 点摘要下界 |
| E2 | MomentKV 式一阶校正（+ `Σ_vk`） | 已发表的免训练 baseline |
| E3 | **高斯指数矩，仅 `Σ_kk`（对角）** | 测"key 离散度对分母重建是否有价值" |
| E4 | **高斯指数矩，`Σ_kk` + 低秩 `Σ_vk`** | 完整 §2.B |
| E5 | 低秩联想 `(S,z)`，`r=8/16/32/64` | 另一条表示路线 |

**评估必须在正确的 query 上做**：按 P0-B 的 `T` 取 top 分位，否则会在无关 query 上优化重建。

**同时必须输出 MGF 误设诊断**（见 §2.B 风险 1）：逐 cluster 的 `D̂_E/D_E` 比值分布、
以及它与 cluster 内 key 峰度的关系。

**这一步的产出**：需要几阶统计量、需要多大 `r`/`r_c`、高斯假设是否站得住。**全部免训练。**

---

## 5. P2 —— 下游验证（仍然免训练优先）

取 P1 局部 oracle 最好的 1–2 个表示，用正确代数

```
ô = (N_R + N̂_E) / (D_R + D̂_E)
```

**第一版最好完全免训练**，只跑 **`scbench_kv` @ ratio 0.1**（基线 32.60，满缓存 68.20，
headroom 35.6 分，且未触地板 —— 这是最好的 recovery stress test）。

判据：

| 结果 | 含义 |
|---|---|
| 32.6 → **40+** | 值得认真继续，再补 0.2 / 0.3 |
| 32.6 → 33 | **不要急着叠 Bayesian gate**，先查表示或代数 |

**ratio 说明**：不要用 0.05 —— 我们实测 `scbench_kv` 在那里基线只剩 **2.00 分**（地板，
指标退化）。区间是 **0.3 / 0.2 / 0.1**，早期甚至只跑 0.1。

---

## 6. P3 —— Teacher KL（仅在 P2 有信号后）

### 6.1 目标

```
d_t = KL( p_full(·|x≤t) ‖ p_FastKVzip(·|x≤t) )       离线预计算，固定不变
采样 = 50% top-d_t + 50% uniform
L    = Σ_t KL( p_full,t ‖ p_memory,t )
```

- **`d_t` 必须离线用 full vs FastKVzip-only 算**，不能让 student 自己选 ⇒ 否则 moving
  target / selection bias。
- **保留 50% uniform**，避免只学极端 failure query 而牺牲整体行为。
- 词表 152k，不能算全部位置的 logits：先采 ~2048 候选位置、只在这些位置跑 `lm_head`
  （两个模型各一次），再取 top-k。

对比现状：现在是文档尾部固定 **128** 个位置（`--target_len 128`）+ CE against ground truth。

### 6.2 必须同时跑 `当前 slots + teacher KL`

**v2 的执行漏洞**：直接从"slots + 旧目标"跳到"`(S,z)` + teacher KL"，同时改了
①表示 ②读出代数 ③训练目标。结果若从 45 变 55，无法归因。
所以必须补一个只改目标的 cell（见 §7 阶梯的 M2）。

---

## 7. P4 —— 不确定性（仅在 H1 成立后）

### 7.1 从方差到门：需要一层先验

delta method 只给预测协方差（`D_T=D_R+D_E`）：

```
∂f/∂N_E = I/D_T ,  ∂f/∂D_E = −o_full/D_T ,  Cov[o] ≈ J_f Σ_X J_fᵀ
```

平方损失下的最优估计是 `E[o_full|M,q]`，**不自动**等于 `o_R+g(σ)(ô−o_R)`。要得到收缩门必须
再设 `δ̂=δ+ε`，`ε~N(0,σ²_ε)`，先验 `δ~N(0,σ²_δ)`：

```
g*(q) = σ²_signal(q) / ( σ²_signal(q) + σ²_noise(q) )
```

`g*→0` 时自然退回 FastKVzip 基线。

**注意 `σ²_noise` 必须是 epistemic（对 `N̂_E,D̂_E` 的估计不确定性），不是 `Σ_kk`（族内离散度）。**
后者已经进了均值估计器，再拿去当 confidence 是概念错误。

### 7.2 标定实验 —— H2 的生死判据，必须做在下游涨分之前

逐 query 计算 `u(q)` 与真实误差 `e(q)=‖o_full−ô‖²`，按 `u` 分 10 bin：

```
应看到:  E[e|bin_1] < … < E[e|bin_10]
报告:    Spearman ρ、NLL、calibration curve、risk-coverage curve
```

**若 `u(q)` 与真实误差不相关 ⇒ 立即停止 H2**，不靠下游偶然涨分维持。

### 7.3 不确定性还可服务另两处

- **写入**：不再用 `KL(q(z|e)‖p(z))` 当 surprise，改问"这条被驱逐 KV 能让残差估计的后验方差
  下降多少"，即 `ΔU_i = U(M) − U(M⊕e_i)` —— 真正的 information gain。
- **预算**：某 `(l,h)` 的残差重构不确定性长期很高 ⇒ 不适合激进压缩 ⇒ 增大 exact-cache 预算。
  比当前 `K=16 ∀(l,h)` 合理得多（`R_opt` 显示 layer 26 = 31.3% 而 22 个层 <0.1%）。

---

## 8. 实验阶梯（消除混淆，每一步只动一个变量）

| 级 | 配置 | 训练 | 状态 |
|---|---|---|---|
| **M0** | FastKVzip（`o_R`） | — | 已有 |
| **M1** | 16 槽 + 加性残差（现状） | 旧目标 | 已有，全部为负/为零 |
| **M2** | 同 M1，**只换 teacher KL** | KL | **待做** —— 隔离"目标"变量 |
| **M3** | cluster 点摘要 `(n,μ_k,μ_v)` + **正确代数** | **免训练** | **待做** —— 隔离"代数+表示" |
| **M4** | 高斯指数矩（`Σ_kk`，再加 `Σ_vk`） | **免训练** | **待做** —— H1 的主张 |
| **M5** | 低秩联想 `(S,z)` | 免训练/训练 | 另一条表示路线 |
| **M6** | M4/M5 最优者 + teacher KL | KL | 只在 M3/M4 有信号后 |
| **M7** | + epistemic 不确定性与收缩门 | KL | 只在 H1 成立后 |

---

## 9. 明确停止的事项

| 停止 | 理由 |
|---|---|
| `d_z` 64→256 | 增加原型编码精度，缺的是关联的数量与结构。把 800 人合照的"平均脸"从 64 维提到 256 维，不会重新得到那 800 张脸 |
| `K` 16→64/128 扫描 | 已实测 K 增大更差；不回答核心科学问题 |
| 门的 α/β/lr/init 扫描 | `R_opt` 探针已证明门在有信号的层接近 oracle 最优（layer 26 达成 0.700 vs 下界 0.687） |
| `gap` MSE 作主目标 | off-policy 且跨层 target 会 stale；保留为 auxiliary/diagnostic |
| ratio 0.75/0.5/0.4 上的主实验 | 基线近乎无损 |
| ratio 0.05（`scbench_kv`） | 基线只剩 2.00 分，地板效应 |
| 试图"修好"当前高斯槽 | 降级为 prototype baseline，见 §11 |
| **先训练再验证表示** | 顺序必须反过来：免训练 oracle 先行 |

---

## 10. 竞品与 novelty 约束

| 工作 | 已占据 | 代码 |
|---|---|---|
| **QEvict** (2608.05326, **08-05，最新**) | **"Future Missed Mass" 与 "Global LIR" 两个诊断量**（度量被丢弃状态上的未来注意力、以及历史静默区的重新激活）；importance drift 现象；三层可恢复驱逐 | 未核实 |
| **ResKV** (2607.29591, 07-31) | 分子/分母残差公式、残差条目与主缓存**进同一 softmax**、逐 layer/head 自适应分配、query 相关动态门；residual entries 存 cluster `(k̄_j,v̄_j,c_j)` 并用 `log c_j` 恢复群体质量 | 未见 |
| **MomentKV** (2606.01563) | count / key 均值 / value 均值 / **value-key 协方差**、一阶校正、**免训练**；**"directional mismatch 比 mass 更要紧"这一观察** | 未核实 |
| **Tensor Cache** (2605.22884) | exact 滑窗 + 驱逐 KV 写入固定大小外积矩阵、learned gate | 未见 |
| **IndexMem** (ICML'26) | learned indexer + fast-weight 矩阵 + stabilizer + 残差读出 | 未见 |
| **Still** (2606.07878) | 冻结骨干 + 小 compactor + 单次前向，8×–200× | 未见 |
| **Attention Matching** (2602.16284) | 免训练闭式匹配 attention 输出/质量，最高约 50× | **有（MIT 许可）** |

**四个后果：**

1. **`1−λ` 这个量已被 QEvict 命名为 "Future Missed Mass"** ⇒ 我们测它是**复现别人的诊断**，
   不是贡献；论文必须引用，且**不要用那个名字**。
2. **"evicted KV → matrix associative memory" 不是新点**（Tensor Cache / IndexMem）。
3. **"uncertainty-aware KV cache" 这个词也不够**（RetentiveKV / InfoKV 已用熵）。
4. **除 Attention Matching 外都没有代码** ⇒ 对照只能是概念覆盖或自行重实现；
   `varikv/moment.py` 是近似复现，**不能**当作 MomentKV 的正式对照。

### 唯一可能独占的 claim（按 H1/H2 两级）

> **H1**: Under a fixed residual budget, a *distributional* summary of evicted key–value
> geometry — entering the softmax numerator and denominator through analytic exponential
> moments — reconstructs omitted attention mass and direction more accurately than point or
> first-order moment summaries.
>
> **H2**: The *epistemic* uncertainty of that reconstruction, conditioned on the current query,
> yields a principled confidence-aware recovery rule.

与 ResKV 的分界必须写清：**ResKV 问"什么时候残差可能有用"**（main-cache attention sharpness
启发式）；**我们问"重构出来的残差本身有多不确定"**。这是两个不同的问题。

### 必须证明的四件事，缺一不可

1. `u(q)` 能预测 `‖ô−o_full‖`（相关性）
2. `u(q)` 是 **calibrated** 的
3. 同一均值估计器下 **Point + uncertainty > Point**
4. **收益抵得过开销**：相近 persistent memory 下 quality / latency / memory 的 Pareto 更好

只满足前三个但慢 2×，对 inference 论文仍然危险。

---

## 11. 系统核算与叙事

### 11.1 系统核算（不能回避）

换成矩阵/协方差状态后**不能再宣称"0.33M 参数 / 极小状态"**。必须报告
`persistent bytes / write FLOPs / read FLOPs / decode latency / throughput`。

参考量级：当前 16 槽状态 `112 × 3072` floats ≈ 1.38 MB(fp32)；
`r=32` 的 `S` 是 `32×128=4096`/组，量级相当（较公平的对照）；
`r=128` 时 `112×128×128` bf16 ≈ 3.67 MB，加 variance/二阶矩至少翻倍。
高斯路线：`diag(Σ_kk)` 128 floats/cluster + `Σ_vk` 低秩 `2×128×r_c`。

另注：现有残差读出已使评测**变慢 25%**（`scbench_kv` 实测 199.5 → 248–257 s/样本），
它在每层每个解码步都要做一次小 softmax。换表示后必须重测。

### 11.2 叙事进阶（把失败的工作变成资产）

不要删高斯槽代码，它应作为论文里的 prototype baseline：

```
Discard → Prototype → Point Statistical → Distributional Statistical → + Uncertainty
（丢弃）   （原型聚合）    （质心+正确代数）      （高斯指数矩）              （置信收缩）
```

名字可留：**VariKV: Uncertainty-Calibrated Distributional Residuals for KV Cache Compression**。

### 11.3 认知科学（控制在全文 10–15%，写明是 computational analogy）

| 计算侧 | 认知侧 |
|---|---|
| exact cache（少量高重要 token，保留个体身份） | pattern-separated episodic memory |
| 残差统计记忆（大量被逐经验累积成 `μ,Σ,association`） | overlapping statistical / gist memory |
| `u(q)` 引导的读出信任度 | confidence-guided retrieval |

可写：*point prototypes preserve gist but discard relational variability; distributional
summaries retain the within-gist covariance required for cue-dependent reconstruction.*
**不要**写 "brain has uncertainty → therefore our architecture"，也不要把 `Σ_kk` 直接叫
cognitive uncertainty。CLS 文献支持 episodic ↔ statistical 的互补性，但那是启发不是证明。

---

## 12. 已被证伪的说法（不要再重新提出）

- ~~FastKVzip 用 last-128 token LM loss 训门控~~ —— 它是 `hidden states + KVzip 重建注意力
  分数 → BCE`（`CLAUDE.md:237`；本地 `attention/score.py:70` 是 `attn_weights.amax(dim=(-3,-2))`）。
  我们只共用它的 FineWeb-Edu **数据**。
- ~~K=64 更差是因为 `log K` 多罚 1.39 nat~~ —— `log K` 是 chunk 内常数，被 z-score 抵消。
  但 `−H(w)` 逐 token 变化、**不会**被抵消，导致"归属越明确 ⇒ surprise 越大 ⇒ 写得越多"的倒置。
- ~~训练/评测长度不匹配~~ —— 训练上下文覆盖 112k–128k（日志分块数 `num 2/7/8`），与评测同配置。
- ~~Attention Matching 两秒解完整个流程~~ —— bias 2.2s + value 1.8s 是闭式，key 选择
  （3s / OMP 565s）与 query 生成（8–139s）另算，完整约 150s。
- ~~SVD `E(16)` 高 ⇒ 槽结构表达力够~~ —— 推不出，读出在凸包内而非子空间。
- ~~加性残差在数学上错误~~ —— 形式本身没错，见 §1.3。
- ~~`(S,z)` 是 softmax attention 的精确充分统计量~~ —— 仅在核近似下成立，见 §2.A。
- ~~delta method 直接给出 Bayes 最优门~~ —— 需额外先验/收缩模型，见 §7.1。
- ~~`E[1−λ]` 小就能判死整个方向~~ —— **`M` 小不等于影响小**；`‖Δo‖ = M·C`，
  MomentKV 已指出小质量 + 高方向反差仍可造成大影响。`M` 只给上界 `‖Δo‖ ≤ 2M·max‖v‖`。
- ~~"FastKVzip 删掉的都是真没用的 token"~~ —— **不能默认**。它的门预测的是 task-agnostic
  的 KVzip 重建重要性，不是未来 query 的真实注意力；QEvict 明确记录了 importance drift。

**旁证**：另有 *Protection Is (Nearly) All You Need: Structural Protection Dominates Scoring
in Globally Capped KV Eviction* (2605.18053)，与我们 stage-1 "随机驱逐打赢全部十种准则"
的结果同向，可作为引用（未核实全文）。

---

## 13. 生死树

| 观察 | 结论 |
|---|---|
| `M` 小、`C` 小、`G` 小，且在 FastKVzip 失败的 query 上也一样 | 被驱逐内容确实没有残差信号 ⇒ **停止吸收路线** |
| `M` 小但 `C` 大，`G` 在关键 query 上显著 | MomentKV 式方向缺口成立 ⇒ **继续统计/分布重建** |
| 质心差、高斯矩明显更好 | **最理想** ⇒ distributional 主张被真正救活 |
| 高斯矩 ≈ 质心，但 `(S,z)` 明显更好 | 高斯方向不成立 ⇒ 转联想矩阵 |
| 局部估计都很好但下游不涨 | 局部重建与端到端效用错配 ⇒ teacher KL / trajectory 是主问题 |
| teacher KL 后仍不涨 | 在 FastKVzip + 冻结 Qwen 这个设定下，整个后驱逐重建方向应停 |

---

## 14. 立即可执行顺序

```
1. 修 empty-memory 注入 bug                                    ~30 min（阻塞项）
2. Mass–Contrast–Gap–Sensitivity 探针
   scbench_kv @ ratio 0.1，20–30 样本，逐 (l,h_kv,h_q,t)
   logsumexp 算 M；报 median/P90/P95/P99/max + heatmap + E[M|T top10%]
   出三张联合图                                                 ~2 h  ★决定后面走哪条
3. SVD 谱 heatmap                                              ~30 min
4. 免训练局部统计 oracle：E0→E5 阶梯 + MGF 误设诊断              ~半天
   └ 产出：需要几阶统计量、需要多大 r/r_c、高斯假设是否成立
5. 取最优 1–2 个表示，免训练跑 scbench_kv @ ratio 0.1           ~1 GPU-h
   └ 32.6 → 40+ 才继续；→33 就先查表示/代数
6. M2（slots + teacher KL）与 M6（最优表示 + teacher KL）        各 ~1 h 训练
7. 只有 H1 成立后才做 M7（epistemic 不确定性 + 标定实验）
8. 最后才扩 LongBench/RULER、多模型、系统效率
```

评测成本可压：比例从 5 个砍到 1–3 个直接省一半以上（固定开销只有一份满缓存前填）。
实测参考：`scbench_kv` 单档 5 比例 100 条约 7 小时，其中基线本身占 5.5 小时（记忆额外 +25%）。
