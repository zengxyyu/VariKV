# VariKV 下一步：优化路线与待办

> 写于 **2026-08-11**，v2（当天两轮外部评审后修订）。
> 目的：把"接下来做什么、为什么、做到什么程度算过关"固定下来，避免忘记或反复重推。
> 相关：`CLAUDE.md`（实验记录）、`varikv_method_spec.md`（方法的代码级公式）、
> `kv_inference_acceleration_2026.md`（竞品全景）。

---

## 0. 一句话现状与研究问题的重新定义

当前 **16 高斯槽 + 加性残差读出**这条路已被实验否定。但不要再问"怎么把它调好"，
问题应重新定义为：

> **在固定的 exact cache 之外，用一个有界状态去估计被驱逐集合对 softmax attention 的
> 遗漏统计量，并量化这个估计对当前 query 的可信度。**

这样高斯槽、KL surprise、精度平均都不再是必须保留的核心，它们降级为
**prototype-memory baseline**（见 §8 的叙事进阶）。

---

## 1. 精确恒等式：所有后续设计的起点

设保留集 `R`、被驱逐集 `E`，`D_X(q)=Σ_{i∈X}e^{q·k_i/√d}`，`N_X(q)=Σ_{i∈X}e^{q·k_i/√d}v_i`，
`o_X = N_X/D_X`。则**无需任何核近似**：

```
o_full(q) = λ(q)·o_R(q) + (1−λ(q))·o_E(q),        λ(q) = D_R(q) / (D_R(q)+D_E(q))
Δo(q)     = o_full − o_R = (1−λ(q))·[o_E(q) − o_R(q)]
```

### 1.1 由此得到的第一个、也是最该先测的量

```
‖o_full − o_R‖ = (1−λ) · ‖o_E − o_R‖
```

**`(1−λ) = D_E/(D_R+D_E)` 是全部可恢复信号的直接乘数。**
而 FastKVzip 驱逐的恰是**低注意力**token，所以即使 `|E|` 占 70%，`D_E` 仍可能极小。

> **若实测 `E[1−λ] ≈ 0.03`，则无论用什么记忆、什么目标、多大容量，可修正的量级本身只有 3%。**
> 这将是对 headroom 表最根本的机制性解释。见 P0-1。

### 1.2 关于"加性残差是否错"——精确表述

**加性形式本身没错**：可以定义 `m*(q)=(1−λ)(o_E−o_R)`，则 `o_full=o_R+m*` 精确成立。
错的是我们那个具体的 `m(q)`：不知道 `D_E`、不知道 `D_R`、门 `σ(g_{l,h})` 是 query 无关常数、
且读出落在 16 个学习向量的**凸包**内（非子空间）。

论文里应写：**post-hoc additive correction hides the softmax normalization structure and
forces a compact memory to implicitly approximate a retained-cache-dependent residual
function.** 不要写 "additive residual is mathematically incorrect"。

### 1.3 关于 `(S,z)` 的正确表述

在指数核的正定特征近似 `e^{q·k}≈φ(q)ᵀφ(k)` 下：

```
N_E(q) ≈ φ(q)ᵀS,  S = Σ_{i∈E}φ(k_i)v_iᵀ
D_E(q) ≈ φ(q)ᵀz,  z = Σ_{i∈E}φ(k_i)
```

**`exp(qᵀk)` 没有有限维精确特征映射**，所以只能写"**在核近似下**遗漏的分子与分母允许一个
固定大小的联想充分状态 `(S,z)`"。**不要写 "(S,z) are sufficient statistics for softmax
attention"** —— 会被直接圈出来。

---

## 2. P0 —— 诊断阶段（半天到一天，不写新方法）

按此顺序，每一步都可能提前终止后面的步骤。

### P0-1 测 `1−λ` 的分布 【最便宜、最靠前、可能直接定生死】

逐 (层, kv-head, query)、逐 ratio ∈ {0.3, 0.2, 0.1} 统计 `1−λ = D_E/(D_R+D_E)` 的分布
（均值、分位数、按层出 heatmap）。

`D_R` 需要保留集的 softmax 分母：代码走 `flash_attn_varlen_func`（`attn.py:108/276`），
**需核实**本版本能否返回 `softmax_lse`（`return_attn_probs=True` 通常返回 `(out, lse, …)`，
则 `D_R=exp(lse)`）；拿不到就在诊断脚本里对保留集单独重算 logsumexp（少量样本，成本可接受）。
`D_E` 用真实被驱逐 KV 精确算，诊断阶段不用核近似。

### P0-2 修 empty-memory 注入 bug 【阻塞后续所有实验】

`attn.py:149` 无条件调用 `memory_residual`，空记忆照样注入。证据：同一 ckpt 跨独立 job 的
ratio-1.0 分数逐字相同、不同 ckpt 之间不同（68.20 / 66.80 / 68.60 / 67.20 / 67.80 / 70.40）。
修法：`if self._absorbed_upto == 0: return 0`。

不修的话审稿人可以说"gains/losses 部分来自注入无条件学习向量，而非恢复被驱逐信息"，
而我们已有证据证明它确实产生非零输出。

### P0-3 分解诊断：收缩项 vs 新增内容项

```
Δo = (1−λ)·o_E  −  (1−λ)·o_R
     └ 新增内容 ┘   └ 重归一化/收缩 ┘
```

量两项的**范数**与**夹角**（只看范数会被相消误导）。若收缩项可比或更大，
则"只能加不能缩"的旧读出被**定量**证明不足——这比纯理论推导有说服力。

### P0-4 SVD 谱（降级为 diagnostic）

对 `Y = O_full − O_R ∈ R^{T×d}` 逐 (层, kv-head) 做 SVD，报
`E(r)=Σ_{j≤r}σ_j²/Σσ_j²`，`r=1,4,8,16,32,64,128`，**出 heatmap**。

**解释规则（易用错）**：`E(16)` 低 ⇒ 秩是硬约束、当前槽结构必死；
`E(16)` 高 ⇒ **推不出任何结论**，因为读出落在 16 顶点凸包内，比"秩≤16 的子空间"约束强得多。

### P0-5 统计阶数消融（oracle，不用任何压缩记忆）

先用**完整** `E` 通过 `(N_E, D_E)` 重构，数值上应精确恢复 `o_full` —— 这是 sanity check。
然后逐级降级近似，测 `(N̂_E, D̂_E)` 的误差：

```
仅均值 → 一阶矩 → 加协方差（MomentKV 式）→ 低秩联想状态 (S,z)，r=8/16/32/64
```

**目的：在写任何新方法之前，知道到底需要几阶统计量、需要多大 r。**

---

## 3. P1 —— Point Associative Residual（正确代数 + 正确目标）

**这一步完全不要 uncertainty。** 只回答一个问题：*正确的 attention 代数 + 联想状态是否有效？*

### 3.1 状态与读出

```
写:  S ← γS + Σ_{i∈E} η_i φ(k_i)v_iᵀ
     z ← γz + Σ_{i∈E} η_i φ(k_i)
读:  N̂_E = φ(q)ᵀS,   D̂_E = φ(q)ᵀz
用:  ô = (N_R + N̂_E) / (D_R + D̂_E)        ← 不是 o_R + m(q)
```

- **必须带 `z`**：没有 `z` 就估不出 `D_E`，退回"只能加不能缩"，把刚诊断出的病带进新实验。
- `φ(k)` 顺序试：`Wk`（线性）→ `ELU(Wk)+1`（正定）。**不要**用 generic Performer 随机特征——
  骨干是冻结的 Qwen，固定 `φ` 可能对它的 query/key 几何很差。
- **`r` 要按显存预算匹配，不要直接上 128。** 参考：当前状态是 `K·d_z·3 = 3072` floats/组；
  `r=32` 时 `S` 是 `32×128 = 4096` floats/组，量级相当，是较公平的对照。
  `r=128` 时全模型 `112×128×128` bf16 ≈ 3.67 MB，**必须重新做系统核算**（见 §7）。

### 3.2 训练目标：eviction-sensitive teacher KL

```
d_t = KL( p_full(·|x≤t) ‖ p_FastKVzip(·|x≤t) )          离线预计算，固定不变
采样 = 50% top-d_t  +  50% uniform
L    = Σ_t KL( p_full,t ‖ p_memory,t )
```

三个要点：
- **`d_t` 必须离线用 full vs FastKVzip-only 算，不能用 student 自己选** —— 否则 moving
  target / selection bias。
- **保留 50% uniform**，避免只学极端 failure query 而牺牲整体行为。
- 词表 152k，不能算全部位置的 logits：先采 ~2048 候选位置、只在这些位置跑 `lm_head`
  （两个模型各一次），再取 top-k。

对比现状：现在是文档尾部固定 **128** 个位置（`--target_len 128`）+ CE against ground truth；
改后是数百个**高信息**位置 + KL against full-cache teacher。

### 3.3 评测区间 —— 修正外部建议

**`ratio = 0.3 / 0.2 / 0.1`，不是 0.3/0.1/0.05。**
我们实测 `scbench_kv` 在 **ratio 0.05 处基线只剩 2.00 分**（地板，指标退化）；
有信息量的是 0.2（基线 45.20）与 0.1（32.60），0.3（65.40）作锚点。
**第一轮只跑 `scbench_kv`，绝不跑 11 个数据集。**

### 3.4 GO / NO-GO（研究管理阈值，非硬数学）

在 0.2 / 0.1 这两个基线确实下降的点上，deterministic point associative 应当
**稳定且配对显著地恢复所丢失任务性能的 20–30%**。
（0.2 处丢 23 分 ⇒ 需要 +5~7 分；0.1 处丢 35.6 分 ⇒ 需要 +7~11 分。）

**若连 point mean estimator 都几乎恢复不了 ⇒ 立刻停止 uncertainty 路线**，
再精致的 posterior variance 也没有意义。

---

## 4. P2 —— Distributional，只在 P1 成功后做

只改一个变量：`(S,z)` → `p(S,z)`。feature / write / mean estimator / 数据 / 优化器全部不变。

### 4.1 状态

```
S_2 ← γS_2 + Σ η_i [φ(k_i)v_iᵀ]^⊙2 ,   z_2 ← γz_2 + Σ η_i φ(k_i)^⊙2
⇒  Var[N̂_E(q)], Var[D̂_E(q)]
```

### 4.2 从方差到门：需要一层先验，delta method 不够

delta method 给预测协方差（`D_T = D_R+D_E`）：

```
∂f/∂N_E = I/D_T ,   ∂f/∂D_E = −o_full/D_T
Cov[o(q)] ≈ J_f Σ_X J_fᵀ
```

但平方损失下的最优估计是 `E[o_full|M,q]`，**不自动**等于 `o_R+g(σ)(ô−o_R)`。
要得到收缩门必须再设：`δ̂=δ+ε`，`ε~N(0,σ²_ε)`，先验 `δ~N(0,σ²_δ)`，于是

```
g*(q) = σ²_signal(q) / ( σ²_signal(q) + σ²_noise(q) )
```

`g*→0` 时自然退回 FastKVzip 基线。这比现在学出来的常数 `σ(g_{l,h})` 理论上强得多。

### 4.3 不确定性标定实验 —— distributional 主张的生死判据

**这是过去 VariKV 完全没有的实验，必须做在"下游涨分"之前。**

逐 query 计算预测不确定性 `u(q)` 与真实重构误差 `e(q)=‖o_full−ô‖²`，按 `u` 分 10 个 bin：

```
应当看到:  E[e|bin_1] < E[e|bin_2] < … < E[e|bin_10]
报告:      Spearman ρ、NLL、calibration curve、risk-coverage curve
```

**若 `u(q)` 与真实误差不相关 ⇒ 立即停止 distributional 路线**，不要靠下游偶然涨分维持。

### 4.4 干净的 dist vs point 定义

现在的 dist/point 一次变三件事（方差、写入门、精度更新），不是单变量。未来必须：

| | mean state | 额外 | uncertainty 作用于 |
|---|---|---|---|
| Point | `μ_S, μ_z` | — | — |
| Dist | **同一个** `μ_S, μ_z` | `Σ_S, Σ_z` | 只作用于 `g(q)` 或预算分配 |

这样才真正检验"**给定同一个均值估计器，不确定性是否带来价值**"。

### 4.5 不确定性还可服务另两处（让贡献更完整）

- **写入**：不再用 `KL(q(z|e)‖p(z))` 当 surprise，改问"这条被驱逐 KV 能让残差估计的后验方差
  下降多少"，即 `ΔU_i = U(M) − U(M⊕e_i)` —— 这才是真正的 information gain。
- **预算**：某 `(l,h)` 的残差重构不确定性长期很高 ⇒ 该 head 不适合激进压缩 ⇒ 增大 exact-cache
  预算。反之可多驱逐。比当前 `K=16 ∀(l,h)` 合理得多（我们自己的 `R_opt` 显示 layer 26 = 31.3%
  而 22 个层 <0.1%）。

---

## 5. 明确停止的事项

| 停止 | 理由 |
|---|---|
| `d_z` 64→256 | 增加原型编码精度，缺的是关联的数量与结构。把 800 人合照的"平均脸"从 64 维提到 256 维，不会重新得到那 800 张脸 |
| `K` 16→64/128 扫描 | 已实测 K 增大更差；不回答核心科学问题 |
| 门的 α/β/lr/init 扫描 | `R_opt` 探针已证明门在有信号的层接近 oracle 最优（layer 26 达成 0.700 vs 下界 0.687） |
| `gap` MSE 作主目标 | off-policy 且跨层 target 会 stale；保留为 auxiliary/diagnostic |
| ratio 0.75/0.5/0.4 上的主实验 | 基线近乎无损，测的只是"记忆会不会引入噪声" |
| ratio 0.05 上的主实验（`scbench_kv`） | 基线只剩 2.00 分，指标已退化 |
| 试图"修好"当前高斯槽 | 降级为 prototype baseline，见 §8 |

---

## 6. 竞品与 novelty 约束

**"evicted KV → matrix associative memory" 已经不是新点。**

| 工作 | 已占据 | 代码 |
|---|---|---|
| **ResKV** (2607.29591, 07-31) | 分子/分母残差公式、残差条目与主缓存**进同一个 softmax**（原文 "rather than acting as a post-hoc correction"）、逐 layer/head 自适应残差分配、**decode 时 query 相关动态门**。其 residual entries 存 cluster `(k̄_j, v̄_j, c_j)`，用 `log c_j` 恢复群体 softmax 质量；门基于 main-cache attention sharpness | 未见 |
| **Tensor Cache** (2605.22884) | exact 滑窗 + 驱逐 KV 写入固定大小外积矩阵 `A ← λA + Σ k⊗v`、矩阵乘法读出、learned gate | 未见 |
| **IndexMem** (ICML'26) | learned indexer + fast-weight 矩阵 `M` + stabilizer `b` + 归一化联想检索 + 残差读出 | 未见 |
| **MomentKV** (2606.01563) | count / key 均值 / value 均值 / **value-key 协方差**，一阶校正，**免训练** | 未核实 |
| **Still** (2606.07878) | 冻结骨干 + 小 compactor + 单次前向，8×–200× | 未见 |
| **RetentiveKV** (ACL'26 Findings) | 熵引导的连续状态记忆（多模态为主） | 未见 |
| **InfoKV** (2606.26875) | 预测不确定性/熵用于 KV 重要性 | 未核实 |

**两个后果**：
1. **矩阵记忆是 substrate/baseline，不是贡献**。"uncertainty-aware KV cache" 这个词也已经不够。
2. **四个最近竞品都没有代码** ⇒ 只能做概念覆盖或自行重实现；`varikv/moment.py` 是近似复现，
   不能当正式对照（论文若要比 MomentKV 必须跑官方实现）。

### 唯一可能独占的 claim

> Existing residual-memory approaches estimate omitted content but do not quantify the
> **reliability** of the reconstructed attention contribution. We formulate KV eviction
> recovery as probabilistic estimation of the omitted softmax **numerator and denominator**,
> yielding **query-conditioned predictive uncertainty** and a principled confidence-aware
> recovery rule.

与 ResKV 的分界必须说清：**ResKV 问"什么时候残差可能有用"**（attention sharpness 启发式）；
**我们问"重构出来的残差本身有多不确定"**（`p(N_E,D_E|q,M)`）。这是两个不同的问题。

### 必须证明的四件事，缺一不可

1. `u(q)` 能预测 `‖ô−o_full‖`（相关性）
2. `u(q)` 是 **calibrated** 的（高置信度确实误差小）
3. 同一均值估计器下，**Point + uncertainty > Point**
4. **收益抵得过开销**：相近 persistent memory 下 quality / latency / memory 的 Pareto 更好

只满足前三个但慢 2×，对一篇 inference 论文仍然危险。

---

## 7. 系统核算（不能再回避）

一旦换成矩阵状态，**不能再宣称"0.33M 参数 / 极小状态"**。必须报告：

```
persistent bytes、write FLOPs、read FLOPs、decode latency、throughput
```

参考量级：`r=128` 时 `112 组 × 128×128` bf16 ≈ **3.67 MB**（再加 variance / 二阶矩 / z 至少翻倍）；
当前 16 槽状态是 `112 × 3072` floats ≈ 1.38 MB（fp32）。
ResKV 明确在**相同 KV budget** 下比较并报告峰值显存与长上下文吞吐，这是绕不开的系统基线。

另注：现有残差读出已使评测**变慢 25%**（`scbench_kv` 实测 199.5 s/样本 → 248–257 s/样本），
它在每层、每个解码步都要对 M 个槽做一次 softmax。换矩阵后这个比例会变，必须重测。

---

## 8. 叙事进阶（把失败的工作变成资产）

不要删高斯槽代码。它应作为论文里的 **prototype-memory baseline**，于是得到：

```
Discard  →  Prototype  →  Associative Point  →  Associative Distributional
（丢弃）    （原型聚合）      （联想点估计）        （联想分布估计）
```

比现在 `point slot → dist slot` 的科学故事强得多，而且每一步都有我们自己的实测支撑。

名字可保留：**VariKV: Uncertainty-Calibrated Associative Residuals for KV Cache Compression**
（Vari 从 "Variational Gaussian slots" 改指 "Variance-aware / Variational Residual"）。

---

## 9. 认知科学的接法（控制在全文 10–15%）

作为**设计原则**而非证明，三个映射就够：

| 计算侧 | 认知侧 |
|---|---|
| exact cache（少量高重要 token，保留个体身份） | pattern-separated episodic memory |
| associative residual state（大量被逐经验累积成 `S,z`，失去个体 trace） | overlapping statistical / gist memory |
| `u(q)` 引导的读出信任度 | confidence-guided retrieval |

预算问题可写成 `min 记忆开销 + β·(interference / 预测不确定性)`，这比"KL surprise"自然。
CLS 文献支持 "pattern-separated episodic ↔ distributed statistical integration" 的互补性，
但**必须写明这是 computational analogy，不是神经生物学等价**。
不要写 "brain has uncertainty → therefore our architecture"，也不要把 `σ²_latent` 直接叫
cognitive uncertainty。

---

## 10. 已被证伪的说法（不要再重新提出）

- ~~FastKVzip 用 last-128 token LM loss 训练门控~~ —— 它是 `hidden states + KVzip 重建注意力
  分数 → BCE`（`CLAUDE.md:237`；本地 `attention/score.py:70` 是 `attn_weights.amax(dim=(-3,-2))`）。
  我们只共用它的 FineWeb-Edu **数据**。
- ~~K=64 更差是因为 `log K` 多罚 1.39 nat~~ —— `log K` 是 chunk 内常数，被 z-score 抵消。
  但 `−H(w)` 逐 token 变化、**不会**被抵消，导致"归属越明确 ⇒ surprise 越大 ⇒ 写得越多"的语义倒置。
- ~~训练/评测长度不匹配~~ —— 已核实训练上下文覆盖 112k–128k（日志分块数 `num 2/7/8`），
  与评测同 chunk/window。
- ~~Attention Matching 两秒闭式解完整个流程~~ —— bias 2.2s + value 1.8s 是闭式，
  key 选择（3s / OMP 565s）与 query 生成（8–139s）另算，完整约 150s。
- ~~SVD `E(16)` 高 ⇒ 槽结构表达力够~~ —— 推不出，读出在凸包内而非子空间。
- ~~加性残差在数学上错误~~ —— 形式本身没错，见 §1.2 的精确表述。
- ~~`(S,z)` 是 softmax attention 的精确充分统计量~~ —— 仅在核近似下成立，见 §1.3。
- ~~delta method 直接给出 Bayes 最优门~~ —— 需额外的先验/收缩模型，见 §4.2。

---

## 11. 立即可执行顺序

```
P0-1  测 1−λ = D_E/(D_R+D_E) 的分布（逐层 heatmap，ratio 0.3/0.2/0.1）   ~1 h  ★最靠前
      └ 若 E[1−λ] 极小 ⇒ 可恢复量级本身就微不足道，整个方向需重新评估
P0-2  修 empty-memory 注入                                              ~30 min（阻塞项）
P0-3  分解诊断：收缩项 vs 新增内容项（范数 + 夹角）                        ~1 h
P0-4  SVD 谱 heatmap                                                    ~30 min
P0-5  统计阶数消融 oracle（均值/一阶/协方差/低秩 r=8..64）                 ~2 h
      └ 输出：需要几阶统计量、需要多大 r
P1-1  Point associative (S,z) + 正确代数 (N_R+N̂_E)/(D_R+D̂_E)            ~1 天实现
P1-2  eviction-sensitive teacher KL 训练                                ~1 h 训练
P1-3  只跑 scbench_kv @ ratio 0.3/0.2/0.1                              ~3 GPU-h
      └ GO 条件见 §3.4；不达标则停止 uncertainty 路线
P2    p(S,z) + 标定实验（§4.3 是生死判据）                                仅在 P1 成功后
P3    LongBench/RULER 扩展、多模型、系统效率                              最后
```

评测成本可压：比例从 5 个砍到 2–3 个直接省一半（固定开销只有一份满缓存前填）。
实测参考：`scbench_kv` 单档 5 比例 100 条约 7 小时，其中基线本身占 5.5 小时（记忆额外 +25%）。
