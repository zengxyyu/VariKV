# VariKV 下一步：优化路线与待办

> **v5（冻结版）**，2026-08-11（当天五轮外部评审后修订）。
> **这份文件到此冻结为 roadmap + 可被数据推翻的 hypotheses，不再继续纸上迭代。**
> 继续改的边际价值已低于跑第一批真实诊断。
> 目的：把"接下来做什么、为什么、做到什么程度算过关"固定下来，避免忘记或反复重推。
> 相关：`CLAUDE.md`（实验记录）、`varikv_method_spec.md`（方法的代码级公式）、
> `kv_inference_acceleration_2026.md`（竞品全景）。
>
> **v5 的五处关键修订**：
> 1. **一切损伤量必须经 `W_O` 投影**。value 空间的 `‖Δo_h‖` 不能跨 head/layer 比较。
>    **回溯后果：我们 `R_opt` 探针的"信号只在 layer 24/26/27"结论需带投影重算**（§1.6）。
> 2. **`G` 与 `B` 的相关性不能 flatten**：784 个 `(l,h)` 点共享同一个 `B_t`，是伪重复（§4.2）。
> 3. **固定 K 与 position-local clustering 在 `N→∞` 时冲突**，必须做 context-scaling oracle（§2.B.5）。
> 4. **equal-bytes 要按实际 dtype 的运行时字节算**，不是 scalar 计数；且写入 FLOPs 与读出分开（§3.2）。
> 5. **COBS (2607.09052) 已用累积量展开 + 每块二阶统计量估计 attention mass** ⇒
>    我们剩下的新意收窄到**分子的联合 key–value 指数矩** + 等预算框架（§10）。
>
> v4 的五点（局部/全局分离、gap target 措辞修正、H1 改成率-失真、MGF 诊断改到 query 投影、
> H2 前的偏差-方差检查）全部保留。

---

## 0'. **P0 已执行完毕（2026-08-11 夜），三个前提被数据改写**

完整结果见 **`P0_FINDINGS.md`**。对本路线图影响最大的三条：

| v5 的判断 | P0 实测 |
|---|---|
| "若 `E[1−λ]` 极小则方向危险" | **`M` = 0.316（P90 0.770），质量充足，此分支作废** |
| Q1 局部损伤是否因果？ | **是。** 恢复 784 个头里的 5 个 ⇒ `B` 降 **17.9%**；低-`G` 对照 −0.5% |
| Q3 高斯簇随长度恶化？ | **不恶化，反而变好。** 128k 上 `r_MGF` 中位 0.973–0.988 ⇒ `O(1)` 预算故事保住 |

**另外两条方法论结论，后续必须遵守：**

1. **不要再用相关性判断"某个局部量重不重要"。** token 级 Spearman 只有 ~0.03、末 token 上是负的，
   而 intervention 给出 −17.9%。相关被 token 类型等因素混淆，**一律用 intervention**。
2. **跨头相消已确认且严重**：`‖Σ_h W_O Δo_h‖ / Σ_h‖W_O Δo_h‖` = **0.253**。
   75% 的逐头损伤在层内自我抵消，而这一族方法（含我们）全是逐头独立修正的。
   这意味着 §8 阶梯里的每一档都应该**顺便报层级联合误差**，而不只是逐头误差。

**另外两条已决出的（对 §8 阶梯直接有影响）：**

3. **局部可修的上界 = `B[last]` −55.7% / `B[mean]` −79%**（恢复全部 784 个头的 oracle）。
   其余 ~45% 是**继承漂移**，逐层局部修正原理上碰不到。**这是 H1 的靶子上限，
   任何有界记忆只能逼近它，不可能超过。**
4. **部分恢复是危险的，层内必须"全修或不修"。** top-k 曲线 5→80 几乎平（−19%/−17%/−24.5%），
   而且逐样本看，`top80` 在某些样本上把 `B` 从 3.32 推到 **6.44**（更差），`all` 则始终安全。
   原因就是 75% 的跨头相消：**打破相消平衡比不修更糟。**
   ⇒ **§3 的 equal-bytes 预算不能按 head 分配到"只修一部分 head"的配置上**，
   这条约束对 ResKV / IndexMem / 我们同样成立，而文献里没人讨论。

5. **`ε_MGF` 的尾部可以便宜地修**：簇内跨 query std / 跨簇 std = **0.15**
   ⇒ `ε` 基本是簇属性 ⇒ **每簇存一个 log 校正标量即可**（1 scalar/簇）。
   记进 §3.1 的预算表：高斯簇 1409 → 1410 scalars，可忽略。

---

## 0. 研究问题

**H1 — 固定预算下的表示假设**（先做，可独立成为技术贡献）：

> 在**相同比特预算、相同驱逐、相同修正代数**下，是把预算花在**更多点原型**上更好，
> 还是花在**更少但带协方差的分布式摘要**上更好？

这是一个**率-失真问题**，与本项目最初的率-失真动机接得上，比"dist vs point"强得多。

**H2 — 不确定性假设**（仅在 H1 成立后做）：

> 给定同一个估计器，**重建误差的可预测性**能否改善"何时/多大程度信任残差"？
> 注意：**不一定是贝叶斯后验方差**——见 §7.1 的偏差-方差前置检查。

当前 16 高斯槽 + 加性残差降级为 **prototype-memory baseline**（见 §11）。不要再问"怎么把它调好"。

---

## 1. 两个必须分开的对象

### 1.1 局部反事实恒等式（单一轨迹内，精确成立）

在**同一次前向**的某个 (layer, head) 上，取该层自己的 `q, K, V`，用 FastKVzip 的 mask 把
`K,V` 划分为 `R`（保留）与 `E`（被驱逐）。记 `D_X=Σ_{i∈X}e^{q·k_i/√d}`，
`N_X=Σ_{i∈X}e^{q·k_i/√d}v_i`，`o_X=N_X/D_X`，则**无需任何近似**：

```
o_all = λ·o_R + (1−λ)·o_E,        λ = D_R/(D_R+D_E)
Δo    = o_all − o_R = (1−λ)·(o_E − o_R)
G = ‖Δo‖ = M·C,    M = 1−λ（遗漏质量）,   C = ‖o_E−o_R‖（驱逐-保留反差）
```

**成立条件是"同一个 q、同一组 K/V、`R∪E` = 该层全部"。**

**这一点在我们的代码里已经实现了。** `memcache_retain.py:_attn_gap` 的两次 attention 用
**同一个 cache、同一个 query，只差一个 mask**——因为 `RetainCache` 物理上什么都不删，
全量 attention 可以直接算。这是选它作基类的意外收益，探针基础设施基本现成。

**`M` 小不等于影响小。** 反例：`o_R=[1,0]`、`o_E=[−10,0]`、`M=0.03` ⇒ `Δo=[−0.33,0]`，
相对 `‖o_R‖` 是 33%。MomentKV 的核心观察正是这个方向不匹配。`M` 只给上界：

```
‖Δo‖ ≤ 2·M·max_i‖v_i‖        （o_E、o_R 都是 value 的凸组合）
```

所以 `M` 既非 kill switch，也非无关量。**不要重复 gap MSE 那次的错误——不让任何单一指标解释一切。**

### 1.2 全局轨迹分歧（两次真实前向，恒等式**不**成立）

真实的 full 与 FastKVzip 是两条不同轨迹。到深层：

```
q_l^full ≠ q_l^pruned ,   K_l^full ≠ K_l^pruned ,   V_l^full ≠ V_l^pruned
```

所以**不能**写 `o_l^full = λ·o_l^pruned + (1−λ)·o_E`。只能测

```
B(t) = KL( p_full(·|x≤t) ‖ p_FastKVzip(·|x≤t) )      全局行为分歧
```

它包含当前层驱逐 + 之前所有层驱逐 + 轨迹漂移 + 非线性放大 + 下游补偿。
**两个探针绝不能混。**

### 1.3 `B(t)` 不是 layer/head 级的任务敏感度

`B(t)` 是 **token 级的可观测行为损伤**。给每个 `(l,h_kv,h_q,t)` 都涂同一个 `B(t)`，
只是在说"这个时间点模型坏得多不多"，**不是**"这个 head 的这个 `Δo` 造成了多少坏处"。

真正的敏感度需要 Jacobian（小扰动下 `KL ≈ ½Δo_lᵀ J_lᵀ F J_l Δo_l`）或 **head intervention**
（在压缩轨迹上单独恢复某个 head 的 `o_all`，看最终 `ΔKL`）。第一阶段**只报 `B(t)`**，
名字就叫 behavioral divergence，**不要叫 task sensitivity**；intervention 留到 §12.4。

### 1.6 **一切损伤量必须经 `W_O` 投影**（v5 新增，最重要的一条）

单个 head 产生 `o_h ∈ R^{d_head}`，但真正进入残差流的是拼接后过输出投影：

```
Δy = W_O · concat(Δo_1, …, Δo_H) = Σ_h W_O^{(h)} Δo_h
```

所以 **value 空间的 `‖Δo_h‖` 不能跨 head、跨 layer 比较**——某个 head 的 `‖Δo_h‖` 可以很大，
而它对应的 `W_O^{(h)}` 块奇异值很小，实际贡献微乎其微；反之亦然。

必须额外记录两个量：

```
G_proj(l,h,t)  = ‖ W_O^{(h)} Δo_{l,h,t} ‖              逐 head 投影后的损伤
G_layer(l,t)   = ‖ W_O concat_h(Δo_{l,h,t}) ‖          整层进入残差流的扰动
```

`G_layer` 单独重要，因为**跨头扰动可以互相增强也可以互相抵消**，所以
`Σ_h ‖W_O^{(h)}Δo_h‖` 绝不等于 `‖Σ_h W_O^{(h)}Δo_h‖`。

**回溯后果：我们 `R_opt` 探针的结论需要带投影重算。** `scratch_probe_gap_target.py` 算的是
**value 空间**的 MSE，正是未投影的量。所以"信号只在 layer 24 / 26 / 27，其余 22 层 <1%"
这个今天被反复引用的分布**可能是投影缺失造成的假象**。重算前不要再拿它当依据。

**一个设计层面的推论**：若实测 `‖Σ_h W_O^{(h)}Δo_h‖ ≪ Σ_h ‖W_O^{(h)}Δo_h‖`（跨头显著相消），
那么**逐头独立修正是在跟相消作斗争**——每个头都"修对了"却可能让层输出更差。
所有这类方法（ResKV / IndexMem / 我们）都是逐头独立的，这是一个未被讨论的共同弱点。

### 1.4 修正："gap target 是 off-policy" 这个说法不准确

`_attn_gap` 的两项来自同一轨迹、同一 cache，**所以它是"同状态反事实"（same-state
counterfactual）目标**。
（**术语**：不要用 on-policy / off-policy，那是 RL 词汇，容易纠缠。
写 *same-state counterfactual attention target* 或 *trajectory-conditioned local counterfactual*。）
真正的问题是另外两条：

1. **轨迹已经漂移**：修好 `Δo_l` 只恢复"该层若能看到自己的被驱逐 KV 会输出什么"，
   **不恢复** `o_l^full`，因为进入该层的隐状态已经不是 full 的。局部恢复 ≠ 全局恢复。
2. **顺序修正使后续 target 过期**：layer 24 一旦被修正，layer 26 那个在"未修正的 layer 24
   轨迹"上算出来的 target 就不再正确。**逐层局部最优之和 ≠ 全局最优。**

### 1.5 关于"加性残差是否错"

**形式本身没错**：定义 `m*(q)=(1−λ)(o_E−o_R)` 则 `o_all=o_R+m*` 精确成立。
错的是我们那个具体的 `m(q)`：不知道 `D_E`/`D_R`、门是 query 无关常数、读出落在 16 个学习向量的
**凸包**内。论文写法：*post-hoc additive correction hides the softmax normalization structure
and forces a compact memory to implicitly approximate a retained-cache-dependent residual
function.* **不要**写 "mathematically incorrect"。

---

## 2. 两条候选表示

### 2.A 联想状态 `(S,z)`（核近似路线）

```
N_E(q) ≈ φ(q)ᵀS,  S=Σφ(k_i)v_iᵀ ;   D_E(q) ≈ φ(q)ᵀz,  z=Σφ(k_i)
```

`exp(qᵀk)` **没有有限维精确特征映射**，所以只能写"在指数核的正定特征近似下允许一个固定大小的
联想充分状态"。**不要**写 "sufficient statistics for softmax attention"。

### 2.B joint-Gaussian 指数矩（更贴合"分布式记忆"的初衷）

把 `E` 划分为 cluster。令 `a=q/√d`。若 cluster 内 `(k,v)` 联合高斯，则**解析地**：

```
E[e^{aᵀk}]   = exp( aᵀμ_k + ½aᵀΣ_kk a )
E[v·e^{aᵀk}] = ( μ_v + Σ_vk a )·exp( aᵀμ_k + ½aᵀΣ_kk a )
```

（第二式经指数倾斜/Esscher 测度验证：倾斜后 `k` 均值移到 `μ_k+Σ_kk a`，代入
`E[v|k]=μ_v+Σ_vk Σ_kk^{-1}(k−μ_k)` 即得。两式已核对。）

```
D̂_c = n_c·exp(aᵀμ_k + ½aᵀΣ_kk a) ,   N̂_c = D̂_c·(μ_v + Σ_vk a)
ô   = (N_R + Σ_c N̂_c) / (D_R + Σ_c D̂_c)
```

**为什么值得优先试**：正确的 softmax 代数（分子+分母）；query 依赖（`a` 在指数里）；
key-value 关联来自 `Σ_vk` 而非无条件平均；**方差解析地进入 attention 等式**
（`Σ_kk` 进分母、`Σ_vk` 进分子）；**可以完全免训练**。

#### 2.B.1 参数化必须用回归形式，保证联合协方差合法

独立估 `diag(Σ_kk)` 和任意低秩 `Σ_vk`，隐含的联合协方差可能不半正定。改用

```
v = μ_v + B(k − μ_k) + ε      ⇒  Σ_vk = B·Σ_kk（自动一致）
E[v e^{aᵀk}] = ( μ_v + B·Σ_kk·a )·exp( aᵀμ_k + ½aᵀΣ_kk a )
```

第一版：`Σ_kk = diag(s_k²)`（128 floats）、`B = UVᵀ` 低秩（`r_c=4~8`）。
语义也更好："key 的偏移预测 value 的偏移"——这本身就是关联结构。

#### 2.B.2 MGF 误设诊断：必须在 query 投影后的一维分布上做

MGF 只看到标量 `δ_i = aᵀ(k_i − μ_k)`。所以正确的诊断量是

```
ε_MGF = log E_emp[e^δ] − ½·Var(δ)
```

由累积量展开（中心化后 `κ₁=0`）：

```
log E[e^δ] = κ₂/2 + κ₃/6 + κ₄/24 + …     ⇒  ε_MGF = κ₃/6 + κ₄/24 + …
```

**这直接回答"高斯失败是否因为高阶累积量"**，比"原始 key 峰度"强得多——
key 分布在 128 维里可以很不高斯，而沿特定方向 `a` 近乎高斯。

同时报归一化版本，看图更直观：

```
r_MGF = exp(ε_MGF) = E_emp[e^δ] / e^{Var(δ)/2}
```

`r=1` 精确；`r=2` 表示高斯把该 cluster 的分母**低估 2 倍**；`r=0.5` 高估 2 倍。

若 `ε_MGF ≈ 0` ⇒ 二阶分布式摘要成立。若 `ε_MGF ≫ 0` 但 `κ₃/κ₄` 能显著修复 ⇒
研究问题可升级为 **"重建被驱逐的指数注意力统计量需要几阶矩？"**（截断累积量展开）。
**先测，不要先实现。**

#### 2.B.3 RoPE 使 cluster 定义成为一级理论变量，不是工程细节

MGF 里的 `k` 必须是**实际进入 score 的 key**，即 post-RoPE `k_p = R_p k⁰`。两种做法都有坑：

- **存 post-RoPE key**：公式对应真实 score，但若 cluster 横跨很多位置，`Σ_kk` 同时描述
  语义变化**和 RoPE 相位变化**，而相位变化是旋转的、根本不高斯 ⇒ 严重破坏假设。
- **逆旋到位置无关帧**：query 时 `qᵀR_p k⁰` 仍依赖每个被驱逐 token 的 `p`，所以还必须保存
  位置分布 `p(k⁰,p)` 或做位置条件 cluster。

**第一选择：position-local clusters**（每个 cluster 只覆盖有限位置区间）。减少相位散布、
流式实现简单、无需在线 k-means，也更接近时间性记忆巩固。之后才试 key 空间聚类。

**一个需要注意的细化**（我们自己的补充）：即便位置区间宽度 `W` 有限，RoPE 各频率的相位散布是
`W·θ_j`——**低频维可容忍宽 cluster，高频维即使很小的 `W` 也会绕多圈**。所以可能需要
**按 RoPE 频段分开处理**（高频维在平均后基本是噪声）。这也是一条设计线索。

**但"高频/低频"必须从模型实际读，不能凭维度下标口头判断。** Qwen2.5-7B-**1M** 极可能带
long-context RoPE scaling（YaRN 类），角频率 `θ_j` 与标准 RoPE 不同。
代码里已经存了 `m.varikv_inv_freq = rotary_emb.inv_freq`，分析时直接用它。

**后续可做的漂亮变体**：用 `|ε_MGF|` 过大来触发 cluster 分裂，即
**adaptive cluster splitting by exponential-moment error** —— 目标对齐的是注意力指数矩误差，
而不是欧氏失真。**不要第一版就加。**

#### 2.B.5 **固定 K 与 position-local clustering 在长上下文下冲突**（v5 新增）

两个需求在 `N→∞` 时不可兼得：

- 要 `O(1)` 显存 ⇒ cluster 数 `K` 固定
- 要减少 RoPE 相位混合 ⇒ cluster 的位置跨度 `W` 要小

但固定 `K` 时 `W ~ N/K`：`N=128k, K=16 ⇒ W≈8000`；`N=1M ⇒ W≈62500`。
固定 `W` 则 `K=O(N)`，固定预算的故事就没了。而 `ε_MGF` 很可能**随 `W` 系统性恶化**。

**所以 P1 不能只扫 `W` 挑一个最好的，必须做 context-scaling oracle**：
固定同一个 `K`，在 `N ∈ {16k, 32k, 64k, 128k}` 上测 `ε_MGF(N)` 与 `D_E` 重建误差。
**若误差随 `N` 快速恶化 ⇒ 固定 K 的高斯 cluster 表示本身不具备长度可扩展性**，
这比任何单点 128k 实验都重要。

若真出现，四条出路：

| 路线 | 做法 | 代价 |
|---|---|---|
| A. 固定 K + 在线合并 | 新 cluster 来了就合并两个，**合并准则用 `Δε_MGF` 或期望注意力重建失真**，不用欧氏距离 | 合并准则本身可能成为方法的一部分 |
| B. 对数时间分箱 | age bins `1,2,4,8,…` ⇒ `O(log N)` | 不再严格 `O(1)`，但 KV Means 已提示固定状态在超长上下文可能不够，别为了 O(1) 硬撑 |
| C. 位置感知充分统计量 | 保存与 RoPE 频率对齐的统计量 | 理论最漂亮、工程最复杂，**第一版不做** |
| **D. 更多但更穷的 cluster**（我们自己补的） | 总字节固定，`N` 增大时用**更多、每个更简陋**的 cluster（如降 `r_c` 或去掉 `Σ_kk`） | **这正是 H1 的率-失真权衡沿上下文长度轴的应用**，且保住 `O(1)` 总预算 |

路线 D 最自然：它把"固定预算下存原型还是存分布"这个问题从"给定 N"推广到"沿 N 变化"。

#### 2.B.6 有限样本措辞

`D̂_c = n_c·E_Gaussian[e^{aᵀk}]` **不是**对 cluster 内有限 token 集的精确恒等式，
而是"把有限经验分布视为某高斯总体的样本，再用拟合分布的期望近似经验指数和"。
论文写法：*Under a joint-Gaussian approximation to the within-cluster empirical key–value
distribution, the omitted numerator and denominator admit closed-form exponential-moment
estimates.* **不要**写 "Gaussian modeling analytically reconstructs `D_c`"。
（与 §2.A 的 "sufficient statistics" 措辞问题同类。）

#### 2.B.4 `Σ_kk` 是 aleatoric，不是 epistemic

`Σ_kk` 是**族内离散度**，进入均值估计器是正确的；**不能**直接拿去做 confidence gate。
后者需要 `Var[N̂_E], Var[D̂_E]`——而且见 §7.1，那也可能不是对的工具。

---

## 3. 固定比特预算：H1 的核心

### 3.1 预算不匹配的算术

每 cluster 的状态量（`d=128`）：

| 表示 | 内容 | scalars |
|---|---|---|
| 点质心 | `n, μ_k, μ_v` | 1+128+128 = **257** |
| 高斯 | 上面 + `diag(Σ_kk)` 128 + `B=UVᵀ`（`r_c=4`）2·128·4=1024 | **1409** |

**1 个高斯 cluster ≈ 5.5 个点 cluster 的存储。** 所以"16 高斯 vs 16 点"里高斯赢**不能**证明
分布式更好——审稿人会直接说 *You simply gave the distributional representation ~5× more state.*

**但 5.5× 是 scalar 计数，真实差距可能 >10×。必须按实际运行时字节和 dtype 算：**

| 状态项 | 点原型 dtype | 高斯 dtype | 说明 |
|---|---|---|---|
| `count` | int32/fp32 | int32/fp32 | |
| `μ_k` | bf16 可行 | **可能需 fp32** | |
| `μ_v` | bf16 可行 | bf16 可行 | |
| **方差累加器** | — | **必须 fp32** | 流式 `E[x²]−E[x]²` 在 bf16 下有严重相消误差 |
| 低秩 `B = UVᵀ` | — | bf16/fp32 | |
| cluster 元数据 | 位置边界 / age / assignment | 同 | **两边都要算进去，别漏** |

所以 equal-bytes 表必须逐项列 dtype 与字节数，否则审稿人仍可攻击
*equal scalars is not equal memory*。

### 3.2 必须做两种比较

| 比较 | 设置 | 回答什么 |
|---|---|---|
| **same-K**（机制消融） | `K_dist = K_point` | 相同分组下，高阶矩本身是否有价值 |
| **equal-bytes**（真正的 H1） | 如 8 高斯 vs ~44 点 | **同样比特，存更多原型还是更富的分布？** |

再加 **equal-FLOPs，而且读写必须分开报**：

- **读出（decode 侧）**：MGF 除 `aᵀμ_k` 还要 `aᵀΣ_kk a` 与 `U(Vᵀa)`。
- **写入（prefill 侧）**：这是流式记忆，点原型写入约 `O(d)`，而高斯要维护 `μ, Σ, B`，
  约 `O(d·r_c)` 甚至更多。**v4 只强调了读出 FLOPs，这是漏项。**

系统表至少分 `prefill/write FLOPs` 与 `decode/read FLOPs` 两栏。
inference 论文最终必须回答"同 memory / 同 latency 下谁最好"。ResKV 明确在**固定 KV 预算**下
比较并报峰值显存与解码吞吐，这一维绕不过去。

### 3.3 这个问题的双刃性（必须写在决策树里）

如果 equal-bytes 下**"更多点原型"赢**，那么正确答案就是 **ResKV**（它存的正是 cluster
`(k̄_j, v̄_j, c_j)`）——**我们整个分布式前提在公平记账下输给已发表方法**。这是真实风险。

### 3.4 统一指标：Headroom Recovery Ratio

```
HRR = ( A_method − A_FKV ) / ( A_full − A_FKV )
```

例：full 68.20、FKV@0.1 32.60、method 40.0 ⇒ HRR ≈ 20.8%。跨 ratio 可比。
**分母小时不稳定，只在 headroom 充足的设置上报。**

---

## 4. P0 —— 诊断（不写新方法）

### P0-A 修 empty-memory 注入 bug 【先做，无需讨论】

`attn.py:149` 无条件调用 `memory_residual`，空记忆照样注入。证据：同一 ckpt 跨独立 job 的
ratio-1.0 分数逐字相同、不同 ckpt 不同（68.20/66.80/68.60/67.20/67.80/70.40）。
修法：`if self._absorbed_upto == 0: return 0`。

### P0-B Local Damage × Global Divergence 探针

`scbench_kv` @ ratio 0.1，20–30 样本。

**局部**（在**同一轨迹**内，复用 `_attn_gap` 的结构）逐 `(l, h_kv, h_q, t)` 记录**五个量**：

```
M, C, G = M·C          （value 空间，保留：有明确的 attention-level 解释）
G_proj  = ‖W_O^{(h)}Δo_h‖        ← v5 新增，跨 head/layer 唯一可比的量
G_layer = ‖W_O concat_h(Δo_h)‖   ← v5 新增，含跨头相消
```

**全局**（两次真实前向）记录 `B(t)`。

**首要要看的是 `G_proj` 与 `G_layer`，不是 `G`。** 若
`corr(G_proj, B) ≫ corr(G_raw, B)`，那我们之前很多 layer/head 探针的"奇怪结论"就有解释了。

#### 伪重复：`G` 与 `B` 的相关性绝不能 flatten

对一个 token `t`，`B_t` 会被 28 层 × 28 q-head = **784 个点共享**。直接 flatten
`(G_lht, B_t)` 报 Pearson/Spearman 是严重的伪重复，有效样本量被夸大近三个数量级。
（同一类错误我们在 `absorb` 的 `n_eff_max` 上已经踩过一次。）

正确做法两层：

1. **query 级相关**：先聚合成每 token 一个观测，如 `G_t^max = max_{l,h} G_proj`、
   `G_t^sum`、`G_t^{layermax}`，再算 `corr(G_t^agg, B_t)`。
2. **head 级论断**：不用相关系数，用条件对比
   `E[G_proj(l,h,·) | B ∈ top10%]` vs `E[G_proj(l,h,·) | B ∈ bottom50%]`。
   真正的因果性交给 intervention（见 P0-B'）。

### P0-B' Top-head intervention 【提前到这里做，不要拖到最后】

这是最便宜的因果检验，`P0-B` 一出结果就做，只挑少量 head：

- **正向**：在**压缩轨迹**上，把某 head 的输出换成它的 `o_all`，看最终 `ΔKL`。
- **反向**（我们自己补的）：在**满缓存轨迹**上，把某 head 换成 `o_R`，看最终 `ΔKL`。
  这一向隔离了该 head 驱逐的因果效应，不含压缩轨迹累积漂移的混淆。两向合起来给出上下界。

挑选：top-`G_proj` 的 3–5 个、top-`G_layer` 的层、以及低 `G` 的对照。

判读：`G` 大但 restore 后 `B` 几乎不变 ⇒ 局部损伤不是任务关键；`G` 中等但 restore 后
`B` 掉 60% ⇒ 真正重要。**这比纯相关快得多，也硬得多。**

**四条实现要求：**

1. **用 logsumexp，不要 `exp(lse)`**（会溢出）：
   `L_R=LSE(s_R)`, `L_E=LSE(s_E)`, `L_F=logaddexp(L_R,L_E)`, `M=exp(L_E−L_F)`。
2. **必须逐 query-head。** GQA 是 7:1（28 q-head / 4 kv-head），`D_E(q)` 是 query-head
   依赖的；把 7 个 q-head 平均掉会淹掉受影响的那个。
3. **必须用模型实际的 score**：`q·k/√d` + causal + **post-RoPE** q/k + GQA 映射 +
   保留集的**实际**定义（FastKVzip 的 local window / sink 保护，即 `self.valid`）。
4. **不要只报均值。** `M` 可能极度重尾（99% 是 0.001、1% 是 0.7，均值 0.008）。报
   **median / P90 / P95 / P99 / max + 逐层 heatmap + `E[M | B ∈ top 10%]`**。

**三张图比任何均值都重要：** ①`M` vs `C`，色 = `B`；②`M` vs `G`；③`G` vs `B`。
图①直接回答：**FastKVzip 删掉的是"低质量+低反差"的垃圾，还是"低质量但高反差"的信息？**

### P0-C 继承漂移 vs 局部损伤（我们自己加的一项）

逐层同时测：(a) 进入该层的隐状态在 full 与 pruned 之间的分歧；(b) 该层的局部损伤 `G_proj`。

**测什么**（不要只测 `‖h^F−h^P‖`，Transformer 隐状态的 norm/旋转变动很大）：

```
相对漂移     D_h(l,t) = ‖h_l^F − h_l^P‖ / (‖h_l^F‖ + ε)
方向         cos(h_l^F, h_l^P)
post-RMSNorm 上面两项在 RMSNorm(h_l) 上再算一遍   ← 这才直接决定 q,k,v
query 侧      ‖q_l^F − q_l^P‖ / ‖q_l^F‖           ← 比 raw hidden 更贴 attention 机制
```

**措辞要准确**：若继承漂移在浅层已占主导，结论**不是**"局部修正必败"——一个足够强的记忆
理论上仍可能把 `f(h_l^P, M)` 拉回更好的区域，数学上并非不可能。准确说法是
**"用独立定义的逐层局部 target 去恢复完整轨迹会变得越来越 ill-conditioned"**，
真正的病根是 §1.4 第 2 条：上游一旦修正，下游 target 就过期。

这条也直接关系到 `R_opt` 那个"信号只在 layer 24/26/27"的发现——在漂移累积 + 缺 `W_O` 投影
（§1.6）双重视角下，那个分布很可疑，需要重算。

### P0-D SVD 谱（diagnostic，不作判据）

对 `Y = O_all − O_R` 逐 (层, kv-head) 做 SVD，报 `E(r)`，`r=1,4,8,16,32,64,128`，出 heatmap。
**`E(16)` 低 ⇒ 秩是硬约束、槽结构必死；`E(16)` 高 ⇒ 推不出任何结论**（读出在 16 顶点凸包内，
比"秩≤16 子空间"约束强得多）。

---

## 5. P1 —— 免训练的局部统计 oracle（不跑 LLM 下游）

**先证明统计表示本身有信号，再决定要不要训练。** 若解析统计量就够，那 0.33M 的 MLP 根本不需要
——对 inference 论文这是**优势**。MomentKV 已证明免训练矩摘要是强 baseline。

在真实被驱逐集合上比较 `D̂_E`、`N̂_E`、`ô` 的误差：

| 级别 | 状态 | 目的 |
|---|---|---|
| **E0** | 完整 `E` 的精确 `(N_E,D_E)` | sanity ceiling，必须数值恢复 `o_all` |
| **E1** | 点质心，**same-K** | 原型 baseline（机制消融） |
| **E1b** | 点质心，**equal-bytes**（更多 cluster） | **真正的固定预算 baseline —— 没有它 H1 不完整** |
| **E2** | MomentKV 式一阶校正读出 | 已发表的免训练统计原则 |
| **E3** | 高斯，仅 `diag(Σ_kk)`（只改分母） | `Σ_kk` 的价值 |
| **E4** | 高斯，`Σ_kk` + 低秩 `B`（分子分母都改） | 完整 §2.B |
| **E5** | 低秩 `(S,z)`，**equal-bytes** | 联想路线的对照 |

**评估必须在正确的 query 上做**：按 `B` 取 top 分位，否则会在无关 query 上优化重建。

**同时必须输出：** ①`ε_MGF` 的分布及其与 `κ₃/κ₄` 的关系；②逐 cluster 的 `D̂_E/D_E` 比值分布；
③位置区间宽度 `W` 与 `ε_MGF` 的关系（验证 position-local clustering 是否够）。

**产出**：需要几阶统计量、需要多大 `r_c`、高斯假设是否站得住、cluster 该怎么切。**全部免训练。**

---

## 6. P2 —— 下游验证（仍免训练优先）

取 P1 最好的 1–2 个表示，用正确代数 `ô=(N_R+N̂_E)/(D_R+D̂_E)`，
**第一版完全免训练**，只跑 **`scbench_kv` @ ratio 0.1**（基线 32.60、满缓存 68.20、
headroom 35.6 分且未触地板 —— 最好的 recovery stress test）。

| 结果 | 含义 |
|---|---|
| 32.6 → **40+**（HRR ≳ 21%） | 值得继续，再补 0.2 / 0.3 |
| 32.6 → 33 | **不要叠 Bayesian gate**，先查表示或代数 |

**不要用 ratio 0.05** —— 实测 `scbench_kv` 在那里基线只剩 **2.00 分**（地板）。
区间是 **0.3 / 0.2 / 0.1**，早期只跑 0.1。
`40+` 是**内部研究管理阈值，不写进论文**；论文报 paired CI + HRR + 效率 Pareto。

---

## 7. P3 —— 不确定性（仅在 H1 成立后）

### 7.1 前置检查：偏差-方差分解（这一步决定 H2 该做什么）

cluster 有 800 个 token 时，`μ, Σ` 估计得非常精确 ⇒ **epistemic 方差 ≈ 0**。但若真实投影 logit
分布是多峰/重尾/偏斜，高斯 MGF 仍可能**错得巨大**。即

```
方差小，偏差大
```

此时贝叶斯门会看到 `σ²_epistemic≈0` ⇒ `g*≈1` ⇒ **极度自信却系统性错误**，比没有不确定性更危险。

所以先分解

```
δ̂ − δ = ( E[δ̂] − δ )  +  ( δ̂ − E[δ̂] )
          └ 近似偏差 ┘      └ 估计方差 ┘
```

用对被驱逐 token 的 bootstrap/子采样估 `Var[δ̂]`，真实误差直接算。
**若 bias² ≫ variance ⇒ H2 应该做经验误差标定（学一个误差预测器），而不是贝叶斯后验方差。**

### 7.2 若走贝叶斯路线：delta method 只给协方差，收缩门需要额外先验

```
∂f/∂N_E = I/D_T ,  ∂f/∂D_E = −o_all/D_T ,  Cov[o] ≈ J_f Σ_X J_fᵀ,  D_T=D_R+D_E
```

平方损失最优是 `E[o_all|M,q]`，**不自动**等于 `o_R+g(σ)(ô−o_R)`。设 `δ̂=δ+ε`，
`ε~N(0,σ²_ε)`，先验 `δ~N(0,σ²_δ)`：

```
g*(q) = σ²_signal(q) / ( σ²_signal(q) + σ²_noise(q) )
```

`g*→0` 时自然退回 FastKVzip 基线。

### 7.3 标定实验 —— H2 的生死判据，必须做在下游涨分之前

按 `u(q)` 分 10 bin，应看到 `E[e|bin_1] < … < E[e|bin_10]`；报 Spearman ρ、NLL、
calibration curve、risk-coverage curve。**若 `u(q)` 与真实误差不相关 ⇒ 立即停止 H2。**

### 7.4 不确定性还可服务另两处

- **写入**：改问"这条被驱逐 KV 能让残差估计误差下降多少"（information gain），
  而不是 `KL(q(z|e)‖p(z))`。
- **预算**：某 `(l,h)` 长期重构不确定性高 ⇒ 不适合激进压缩 ⇒ 增大 exact-cache 预算。
  比 `K=16 ∀(l,h)` 合理得多（`R_opt`：layer 26 = 31.3%，22 个层 <0.1%）。

---

## 8. 实验阶梯（每一步只动一个变量）

| 级 | 配置 | 训练 | 隔离的变量 |
|---|---|---|---|
| **M0** | FastKVzip（`o_R`） | — | 基线（已有） |
| **M1** | 16 槽 + 加性残差（现状） | 旧目标 | 已有，全为负/零 |
| **M2** | 同 M1，**只换 teacher KL** | KL | **训练目标** |
| **M3a** | 点 cluster `(n,μ_k,μ_v)` + **加性**修正 | 免训练 | **表示** |
| **M3b** | **同一 state**，改用 `(N_R+N̂_E)/(D_R+D̂_E)` | 免训练 | **归一化感知代数** |
| **M4** | 高斯矩（`Σ_kk`，再加 `B`），same-K | 免训练 | **高阶矩本身** |
| **M4b** | 高斯矩 vs **equal-bytes 点原型** | 免训练 | **固定预算下的率-失真（H1 正题）** |
| **M5** | 低秩 `(S,z)`，equal-bytes | 免训练 | 另一条表示路线 |
| **M6** | M4/M5 最优者 + teacher KL | KL | 学习能否再提升 |
| **M7** | + 误差预测/收缩门 | KL | H2 |

`M3a→M3b` 是纯代数消融；`M3b→M4` 是纯矩消融；`M4→M4b` 才是 H1 的正题。

---

## 9. 明确停止的事项

| 停止 | 理由 |
|---|---|
| `d_z` 64→256 | 增加原型编码精度，缺的是关联的数量与结构 |
| `K` 16→64/128 扫描 | 已实测 K 增大更差；不回答核心问题 |
| 门的 α/β/lr/init 扫描 | `R_opt` 已证明门在有信号的层接近 oracle 最优（0.700 vs 下界 0.687） |
| `gap` MSE 作主目标 | 见 §1.4 的两条精确原因；保留为 auxiliary/diagnostic |
| ratio 0.75/0.5/0.4 的主实验 | 基线近乎无损 |
| ratio 0.05（`scbench_kv`） | 基线只剩 2.00 分，地板效应 |
| 试图"修好"当前高斯槽 | 降级为 prototype baseline |
| **先训练再验证表示** | 顺序必须反过来 |
| **只做 same-K 的 dist vs point** | 不构成固定预算 claim，见 §3 |

---

## 10. 竞品与 novelty 约束

| 工作 | 已占据 | 代码 |
|---|---|---|
| **QEvict** (2608.05326, 08-05) | **"Future Missed Mass"、"Global LIR" 两个诊断量**；importance drift；三层可恢复驱逐 | 未核实 |
| **ResKV** (2607.29591, 07-31) | 分子/分母残差、残差条目**进同一 softmax**、逐 layer/head 自适应分配、query 相关动态门；residual entries 存 cluster `(k̄,v̄,c)` 并用 `log c` 恢复群体质量 | 未见 |
| **MomentKV** (2606.01563) | count / key 均值 / value 均值 / **value-key 协方差**、一阶校正、免训练；**"directional mismatch 比 mass 更要紧"** | 未核实 |
| **Tensor Cache** (2605.22884) | exact 滑窗 + 驱逐 KV 写入固定大小外积矩阵、learned gate | 未见 |
| **IndexMem** (ICML'26) | learned indexer + fast-weight 矩阵 + stabilizer + 残差读出 | 未见 |
| **Still** (2606.07878) | 冻结骨干 + 小 compactor + 单次前向，8×–200× | 未见 |
| **COBS** (2607.09052, 07-10) | **累积量展开近似 attention mass**；指出已有方法困于一阶；**每块存压缩的二阶统计量**。32k RULER：NSA 0.2999 → **0.8195**（补上 dense 0.9040 差距的 86%），KV 读流量 1.21× NSA。**只做分母/质量，用于 block selection，不重建 value** | 未见 |
| **MGF-softmax（同态加密）** (2602.01621) | 用 MGF 近似 softmax 分母 | — |
| **Attention Matching** (2602.16284) | 免训练闭式匹配 attention 输出/质量，最高约 50× | **有（MIT）** |

**四个后果：**

1. **`1−λ` 已被 QEvict 命名 "Future Missed Mass"** ⇒ 测它是复现别人的诊断；必须引用，**别用那个名字**。
2. **"evicted KV → matrix associative memory" 不是新点**（Tensor Cache / IndexMem）。
3. **"uncertainty-aware KV cache" 也不够**（RetentiveKV / InfoKV 已用熵）。
4. **`ε_MGF` 的累积量推导和"二阶统计量估分母"也不是新点** —— **COBS 一个月前就做了**
   （2026-07-10）。而且"用 MGF 近似 softmax 分母"这个想法本身在同态加密领域也已出现
   （2602.01621）。**所以绝不能把"我们首次用 MGF/累积量处理 softmax"当作 novelty。**
5. **除 Attention Matching 外都无代码** ⇒ 对照只能概念覆盖或自行重实现。

**COBS 划定了我们剩余空间的边界，同时也支持这条路线**：它证明二阶块统计量能在 32k RULER 上
补回 86% 的差距——机器是有效的，只要用在对的目标上。它没做的是**分子**：
`Σ_{i∈E} e^{q·k_i} v_i`，以及 **key→value 关联**。

**两处公平性表述必须收紧：**

- **MomentKV 的统计量反过来参与它自己的驱逐决策**（使被摘要的驱逐集更规则）。我们固定
  FastKVzip 的 scorer，所以只能比 **MomentKV 式的读出/统计重建**，
  **不能宣称"我们比完整 MomentKV 更强"**。
- **Protection 那篇**（2605.18053）的真实结论更细：在 globally capped harness 下
  prompt-boundary 保护带来主要恢复；加入结构保护后某些 scoring 差异变小，但 faithful per-head
  方法仍有额外收益。所以可写 *structural protection can dominate score choice in some globally
  capped regimes*，**不能**写 *eviction scoring generally does not matter*。
  （我们 stage-1 "随机驱逐打赢十种准则"只是同向旁证，不是同一 claim。）

### 唯一可能独占的 claim

> **H1**: At a fixed residual-memory budget, does modeling **joint key–value exponential moments**
> — i.e. `E[v e^{aᵀk}] = (μ_v + BΣ_kk a)·exp(aᵀμ_k + ½aᵀΣ_kk a)`, which reconstructs the omitted
> **numerator** and the key→value association, not only the mass — improve reconstruction over
> allocating the same bytes to additional point prototypes?

**"joint key–value" 这四个字是必须的**：COBS 已占 key 侧累积量（分母/质量），
ResKV 已占分子/分母残差，MomentKV 已占 key-value 协方差的一阶校正。
我们的组合必须真正产生一个**解析上不同的估计器**，而 `BΣ_kk a` 那一项是唯一的候选。
>
> **H2**: Is the reconstruction error *predictable* (empirically calibrated, not necessarily
> Bayesian-posterior), and does that prediction improve when/how much the residual is trusted?

与 ResKV 的分界：**ResKV 问"什么时候残差可能有用"**（main-cache attention sharpness 启发式）；
**我们问"重构出来的残差本身有多可信"**。

### 必须证明的四件事，缺一不可

1. `u(q)` 能预测 `‖ô−o_all‖`；2. `u(q)` 是 calibrated 的；3. 同一均值估计器下加不确定性更好；
4. **收益抵得过开销**（相近 persistent memory 下 quality/latency/memory 的 Pareto 更好）。
只满足前三个但慢 2×，对 inference 论文仍然危险。

---

## 11. 系统核算与叙事

### 11.1 系统核算

换表示后**不能再宣称"0.33M 参数 / 极小状态"**。必须报
`persistent bytes / write FLOPs / read FLOPs / decode latency / throughput`。

参考：当前 16 槽 `112×3072` floats ≈ 1.38 MB(fp32)；`r=32` 的 `S` 是 4096/组（量级相当）；
`r=128` 时 `112×128×128` bf16 ≈ 3.67 MB，加二阶矩至少翻倍；
高斯路线每 cluster 1409 scalars（见 §3.1）。

现有残差读出已使评测**慢 25%**（199.5 → 248–257 s/样本）。换表示后必须重测。

### 11.2 叙事进阶（把失败的工作变成资产）

```
Discard → Prototype → Point Statistical → Distributional Statistical → + Calibrated Confidence
（丢弃）   （原型聚合）    （质心+正确代数）      （固定预算下的高斯矩）        （误差预测收缩）
```

名字可留：**VariKV: Fixed-Budget Distributional Residuals for KV Cache Compression**。

### 11.3 认知科学（全文 10–15%，写明是 computational analogy）

固定预算下的 **specificity vs abstraction** 权衡才是真正的接口：

| 计算侧 | 认知侧 |
|---|---|
| 更多点原型 | 更多 pattern-separated episodic traces（specificity） |
| 更少但带协方差的分布 | overlapping statistical / gist memory（relational abstraction） |
| `u(q)` 引导的读出信任 | confidence-guided retrieval |

可写：*point prototypes preserve gist but discard relational variability; distributional
summaries retain the within-gist covariance required for cue-dependent reconstruction.*
**数学由固定预算率-失真支撑，认知科学只提供 intuition。**
不要写 "brain has uncertainty → therefore our architecture"，也不要把 `Σ_kk` 叫 cognitive uncertainty。

---

## 12. 已被证伪或需收紧的说法

- ~~FastKVzip 用 last-128 LM loss 训门控~~ —— 是 `hidden states + KVzip 重建注意力分数 → BCE`
  （`CLAUDE.md:237`；`attention/score.py:70` 是 `attn_weights.amax(dim=(-3,-2))`）。只共用它的**数据**。
- ~~K=64 更差因为 `log K` 多罚~~ —— `log K` 是 chunk 内常数、被 z-score 抵消；但 `−H(w)` 不会，
  导致"归属越明确 ⇒ surprise 越大 ⇒ 写得越多"的倒置。
- ~~训练/评测长度不匹配~~ —— 训练覆盖 112k–128k，与评测同配置。
- ~~Attention Matching 两秒解完整个流程~~ —— bias 2.2s + value 1.8s 闭式，key 选择
  （3s / OMP 565s）与 query 生成（8–139s）另算，完整约 150s。
- ~~SVD `E(16)` 高 ⇒ 槽结构够用~~ —— 推不出，读出在凸包内。
- ~~加性残差数学上错误~~ —— 形式没错，见 §1.5。
- ~~`(S,z)` 是精确充分统计量~~ —— 仅在核近似下，见 §2.A。
- ~~delta method 直接给 Bayes 最优门~~ —— 需额外先验，见 §7.2。
- ~~`E[1−λ]` 小就能判死方向~~ —— `‖Δo‖=M·C`，小质量+高反差仍可致大影响。
- ~~"FastKVzip 删掉的都是真没用的"~~ —— 不能默认；QEvict 记录了 importance drift。
- ~~**"gap target 是 off-policy"**~~ —— **不准确**。`_attn_gap` 两项同轨迹同 cache 只差 mask，
  是 on-policy 的局部反事实。真正的问题是：①轨迹已漂移，局部恢复 ≠ 全局恢复；
  ②顺序修正使后续层的 target 过期。见 §1.4。
- ~~`T(t)=KL(p_full‖p_FKV)` 是 layer/head 的任务敏感度~~ —— 它是 **token 级全局行为分歧**，
  改名 `B(t)`；真正的敏感度要 Jacobian 或 head intervention。见 §1.3。
- ~~value 空间的 `‖Δo_h‖` 可以跨 head/layer 比较~~ —— **不能**，必须经 `W_O` 投影。
  **`R_opt` 探针的"信号只在 layer 24/26/27"结论因此需要重算**。见 §1.6。
- ~~可以把 `(G_lht, B_t)` flatten 起来算相关~~ —— 784 个 `(l,h)` 点共享一个 `B_t`，
  伪重复。先聚合到 token 级。见 P0-B。
- ~~"`ε_MGF` / 累积量展开是我们的新推导"~~ —— **COBS (2607.09052) 一个月前就做了**
  （二阶块统计量估 attention mass）；MGF 近似 softmax 分母在同态加密领域也已出现。
- ~~`D̂_c = n_c·E_Gaussian[e^{aᵀk}]` 是解析恒等式~~ —— 是对**有限经验分布**的高斯近似，
  措辞见 §2.B.6。
- ~~继承漂移大 ⇒ 逐层局部修正"必败"~~ —— 太绝对。准确说法是"用独立定义的局部 target
  恢复完整轨迹会越来越 ill-conditioned"，病根是上游修正使下游 target 过期。
- ~~固定 K 与 position-local clustering 可以兼得~~ —— `N→∞` 时冲突（`W~N/K`）。见 §2.B.5。

### 12.4 待办：真正的 head 因果敏感度

对 top-`G` 的 head 做 intervention：在压缩轨迹上把某个 head 的输出换成它的 `o_all`，
看最终 `ΔKL_{l,h}`。比给所有 head 涂同一个 `B(t)` 可信得多。第一版只做少量 top head。

---

## 13. 生死树

| 观察 | 结论 |
|---|---|
| ~~`M` 小、`C` 小、`G` 小~~ | **已实测：`M`=0.316、`C`=5.44，不成立，此分支作废** |
| **top-head intervention 有显著收益** | **已实测：−17.9%（5/784 个头）⇒ 局部损伤是因果的，路线继续** |
| `M` 小但 `C` 大、`G` 在关键 query 显著 | MomentKV 式方向缺口成立 ⇒ **继续统计重建** |
| 继承漂移在浅层已占主导（P0-C） | 逐层局部修正必败 ⇒ 修正必须前移，或整个 per-layer 残差范式需重设 |
| `ε_MGF ≈ 0`，高斯矩 > **equal-bytes 点原型** | **最理想** ⇒ H1 成立，distributional 主张真正救活 |
| 高斯矩 > same-K 点原型，但 ≤ **equal-bytes** 点原型 | **协方差不值那些比特** ⇒ 正确答案是 ResKV 式更多 cluster，我们的前提在公平记账下失败 |
| `ε_MGF ≫ 0` 且 `κ₃/κ₄` 能修 | 升级为"需要几阶矩"的研究问题 |
| 高斯矩 ≈ 质心，`(S,z)` 明显更好 | 高斯方向不成立 ⇒ 转联想矩阵 |
| 局部估计都好但下游不涨 | 局部重建与端到端效用错配 ⇒ teacher KL / trajectory 是主问题 |
| teacher KL 后仍不涨 | 在 FastKVzip + 冻结 Qwen 这个设定下，整个后驱逐重建方向应停 |
| bias² ≫ variance（§7.1） | H2 改做经验误差标定，不做贝叶斯后验 |
| **`corr(G_proj,B) ≫ corr(G_raw,B)`** | 之前所有 value 空间的 layer/head 诊断都要重算（含 `R_opt`） |
| **跨头显著相消**（`‖Σ_h W_O Δo_h‖ ≪ Σ_h‖W_O Δo_h‖`） | 逐头独立修正在对抗相消 ⇒ 需要层级联合修正，这是所有同类方法的共同弱点 |
| **`ε_MGF(N)` 随 N 快速恶化** | 固定 K 的高斯 cluster 无长度可扩展性 ⇒ 走 §2.B.5 的路线 D（更多更穷的 cluster） |
| **intervention：`G` 大但 restore 后 `B` 不变** | 局部损伤非任务关键 ⇒ **整个后驱逐重建方向很危险** |

---

## 14. 立即可执行顺序

**未来 24 小时压成四步，做完再决定是否正式实现 E1–E5：**

```
1. 修 empty-memory 注入 bug                                     ~30 min（阻塞项）
2. Local counterfactual probe（复用 _attn_gap 结构）
   scbench_kv @0.1，20–30 样本，逐 (l,h_kv,h_q,t)
   记录 M, C, G, **G_proj, G_layer**；全局 B(t)
   logsumexp 算 M；报 median/P90/P95/P99/max + heatmap
   相关性**先聚合到 token 级**，不 flatten                        ~2–3 h
3. **Top-head intervention**（正向 + 反向，各 3–5 个 head）        ~1 h  ★最硬的因果检验
4. MGF feasibility probe：只算 ε_MGF / r_MGF，不实现方法
   看 W、N、RoPE 频段对它的影响（含 context-scaling）             ~2 h
```

**这四步要回答三个问题，每个都是潜在的 NO-GO：**

| Q | 问题 | 若答案为否 |
|---|---|---|
| **Q1** | FastKVzip 的失败是否真的伴随 `G_proj > 0`，且 intervention restore 有因果收益？ | **整个后驱逐重建方向很危险** |
| **Q2** | 在 relevant query 上 `r_MGF ≈ 1` 吗？ | **高斯 H1 很危险** |
| **Q3** | 这个近似在 `16k → 128k` 上稳定吗？ | **固定 K 的高斯残差不适合长上下文设定** |

**之后**（仅在上面过关时）：

```
5. 算清 equal-bytes 预算表（含 dtype 与元数据），定 K_dist 与 K_point   ~30 min（纸上）
6. 免训练局部统计 oracle E0→E5（含 E1b equal-bytes）                  ~半天
7. 最优 1–2 个表示，免训练跑 scbench_kv @0.1；HRR ≳ 21% 才继续         ~1 GPU-h
8. M2 / M6 的 teacher KL                                             各 ~1 h 训练
9. 只有 H1 成立才做 M7（§7.1 偏差-方差前置检查 → §7.3 标定）
10. 最后才扩 LongBench/RULER、多模型、系统效率（读写 FLOPs 分开报）
```

评测成本可压：比例从 5 个砍到 1–3 个省一半以上（固定开销只有一份满缓存前填）。
实测：`scbench_kv` 单档 5 比例 100 条约 7 小时，其中基线本身占 5.5 小时（记忆额外 +25%）。
