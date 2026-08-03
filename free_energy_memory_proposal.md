

# 基于变分自由能的视频 VLM 记忆更新机制

## 前置：底座模型 InternVL3-8B

### 整体架构

```
视频帧输入
    ↓
InternViT-300M（视觉编码器，304M 参数）
    ↓  每帧 → 256 个 patch token，维度 1024
Dynamic Tiling（动态分块）
    ↓  高分辨率帧 → 切成最多 12 个 448×448 的 tile
MLP Projector（两层线性层 + GELU）
    ↓  视觉维度 1024 → LLM 维度 4096
InternLM2.5-7B（语言模型，7.61B 参数）
    ↓
文字输出
```

**总参数量**：ViT 304M + MLP 27.5M + LLM 7.61B ≈ **7.94B**

---

### 三个核心组件

**① InternViT-300M（视觉编码器）**

- 非 CLIP，是 Shanghai AI Lab 自研训练的 ViT
- 参数量 304M，比 EVA-CLIP ViT-G（1.8B）小 6 倍但性能接近
- 每帧输出 256 个 token（14×14 patch grid）
- 关键创新：**Variable Visual Position Encoding（V2PE）**
  - 不同帧、不同 tile 内的 token 有独立位置编码
  - 比标准 RoPE 更适合视频的时序建模

**② Dynamic Tiling（动态分块）**

| 方式      | 处理方法                       | 问题                      |
| --------- | ------------------------------ | ------------------------- |
| 普通 VLM  | 强行 resize 到 224×224         | 丢失高分辨率细节          |
| InternVL3 | 切成最多 12 个 448×448 的 tile | 保留细节，但 token 数增多 |

对视频的影响：

```
每帧 1 个 tile  →  256 token/帧  （省显存模式）
每帧 12 个 tile → 3072 token/帧  （高分辨率模式）
```

**③ MLP Projector（视觉-语言桥梁）**

- 两层线性层 + GELU 激活
- 把视觉 token 从 1024 维映射到 4096 维（InternLM 的隐藏层维度）
- 是视觉编码器和语言模型之间**唯一的连接点**
- **本文记忆模块插入此处之后**

---

### 处理长视频的现状（本文要解决的问题）

```
InternVL3 原始流程：
  均匀采样 N 帧（N=16 或 32）
  → 每帧 256 token
  → 全部拼接送给 LLM
  → LLM 上下文约 4096 token，最多容纳 16 帧

问题：
  100 帧视频只能看 16 帧 → 丢失 84% 的内容
  无任何记忆机制 → 历史信息全部丢弃
```

**这就是本文的切入点：在 MLP Projector 之后加入记忆模块，让模型能处理任意长度的视频。**

---

## 0. 核心问题

**视频 VLM 的根本矛盾：**

$$\underbrace{100 \text{ 帧} \times 256 \text{ tokens/帧}}_{\text{25,600 个视觉 token}} \gg \underbrace{4096 \text{ tokens}}_{\text{LLM 上下文上限}}$$

现有记忆方法的共同缺陷：

| 方法                 | 记忆更新方式          | 缺陷                 |
| -------------------- | --------------------- | -------------------- |
| MA-LMM (CVPR'24)     | FIFO 队列，无条件写入 | 不知道什么重要       |
| ∞-Video (ICML'25)    | 时间衰减加权          | 衰减权重手工设计     |
| MemStream (arXiv'26) | 稀疏 KV 选择          | 无端到端学习         |
| ReWind (CVPR'25)     | 指令引导固定 token    | 容量固定，无不确定性 |

**所有方法的共同假设：新视频段 → 直接写入记忆**

本文的核心主张：**记忆更新应该是一个有原则的信念修正过程，而非无条件写入。**

---

## 1. 理论动机

### 1.1 认知科学依据

**记忆再巩固理论（Memory Reconsolidation）**

- Nader et al., _Science_ 2000
- 核心发现：记忆每次被提取时都会变得「不稳定」，可以被修改后重新存储
- **对 AI 的启示**：记忆不是静态存储，而是被「证据」持续修正的动态状态

**贝叶斯脑假说（Bayesian Brain Hypothesis）**

- Knill & Pouget, _Trends in Neurosciences_ 2004
- 大脑维护世界状态的概率分布，感知新信息时做贝叶斯更新

**预测编码（Predictive Coding）**

- Friston, _Nature Reviews Neuroscience_ 2010
- 大脑持续预测感知输入，用预测误差更新内部模型
- **自由能原理**：所有认知过程都是在最小化「惊讶度」的上界——自由能

### 1.2 数学依据：为什么不能直接用贝叶斯

理想的记忆更新是贝叶斯后验：

$$p(M_t \mid e_t) = \frac{p(e_t \mid M_t) \cdot p(M_t \mid M_{t-1})}{p(e_t)}$$

**问题**：$p(e_t) = \int p(e_t \mid M) \, p(M \mid M_{t-1}) \, dM$ 在连续高维空间中**不可计算**。

**解决方案**：变分推断——用可计算的分布 $q_\phi(z \mid e_t)$ 近似真实后验。

---

## 2. 变分自由能框架

### 2.1 记忆的概率表示

**关键设计**：记忆不是一个向量，而是一个**概率分布**。

$$M_t = \{(\mu_k, \sigma_k^2)\}_{k=1}^{K}$$

其中：

- $K$：记忆槽数量（超参数，如 $K=16$）
- $\mu_k \in \mathbb{R}^d$：第 $k$ 个槽存储的内容（均值）
- $\sigma_k^2 \in \mathbb{R}^d$：第 $k$ 个槽的不确定性（方差）

**直觉**：
$$\sigma_k^2 \text{ 小} \Rightarrow \text{这条记忆很确定，不轻易被覆盖}$$
$$\sigma_k^2 \text{ 大} \Rightarrow \text{这条记忆不确定，容易被新证据更新}$$

### 2.2 变分自由能定义

设当前视频段的证据为 $e_t$（视觉特征的压缩表示），潜在记忆状态为 $z$。

**变分自由能**：

$$\mathcal{F}(e_t, M_{t-1}) = \underbrace{KL\bigl[q_\phi(z \mid e_t) \,\|\, p(z \mid M_{t-1})\bigr]}_{\text{新证据与旧记忆的「惊讶度」}} - \underbrace{\mathbb{E}_{q_\phi}\bigl[\log p(e_t \mid z)\bigr]}_{\text{记忆对新证据的解释能力}}$$

等价地，这是 ELBO（证据下界）的负值：

$$\mathcal{F} = -\mathcal{L}_{\text{ELBO}}, \quad \mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}[\log p(e_t \mid z)] - KL[q_\phi(z \mid e_t) \,\|\, p(z \mid M_{t-1})]$$

**最小化自由能 = 最大化 ELBO**

### 2.3 各项的具体含义

**先验** $p(z \mid M_{t-1})$：旧记忆对当前状态的预测

$$p(z \mid M_{t-1}) = \sum_{k=1}^{K} w_k \cdot \mathcal{N}(z \mid \mu_k, \text{diag}(\sigma_k^2))$$

$$w_k = \text{softmax}_k\!\left(\frac{\mu_k^\top e_t}{\|\mu_k\| \|e_t\|}\right) \quad \text{（相关性加权）}$$

**后验**（识别模型）$q_\phi(z \mid e_t)$：看到新证据后更新的信念

$$q_\phi(z \mid e_t) = \mathcal{N}(z \mid \mu_\phi(e_t),\, \text{diag}(\sigma_\phi^2(e_t)))$$

其中 $\mu_\phi, \sigma_\phi^2$ 是可学习的神经网络（编码器）。

**似然** $p(e_t \mid z)$：记忆对证据的解释能力

$$p(e_t \mid z) = \mathcal{N}(e_t \mid f_\theta(z), \sigma_r^2 I)$$

其中 $f_\theta$ 是解码器网络。

### 2.4 KL 散度的解析形式

对于两个对角高斯分布，KL 有解析解：

$$
KL\bigl[\mathcal{N}(\mu_q, \sigma_q^2) \,\|\, \mathcal{N}(\mu_p, \sigma_p^2)\bigr]
= \frac{1}{2} \sum_{i=1}^{d} \left[
\frac{\sigma_{q,i}^2}{\sigma_{p,i}^2}
+ \frac{(\mu_{q,i} - \mu_{p,i})^2}{\sigma_{p,i}^2}
- 1
+ \ln \frac{\sigma_{p,i}^2}{\sigma_{q,i}^2}
\right]
$$

**KL 的物理意义**：

- $KL \approx 0$：新证据和旧记忆完全一致，无需大幅更新
- $KL \gg 0$：新证据和旧记忆显著不同，携带真正的新信息

### 2.5 重参数化技巧（使梯度可传播）

直接从 $q_\phi(z \mid e_t)$ 采样不可微，使用重参数化：

$$z = \mu_\phi(e_t) + \sigma_\phi(e_t) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

这使得梯度可以通过采样操作反向传播，实现端到端训练。

---

## 3. 记忆更新规则

### 3.1 自适应更新率

$$\eta_k = \sigma\!\left(\alpha \cdot \underbrace{KL}_{\text{信息量}} \cdot \underbrace{w_k}_{\text{相关性}} - \beta\right)$$

其中 $\sigma$ 是 sigmoid 函数，$\alpha, \beta$ 是超参数。

**含义**：

- $KL$ 大且 $w_k$ 大（新信息多且和该槽相关）→ $\eta_k$ 大 → 大幅更新
- $KL$ 小或 $w_k$ 小 → $\eta_k$ 小 → 保持原有记忆

### 3.2 记忆槽更新公式

$$\mu_k^{(t)} \leftarrow (1 - \eta_k) \cdot \mu_k^{(t-1)} + \eta_k \cdot \mu_\phi(e_t)$$

$$\ln \sigma_k^{2(t)} \leftarrow (1 - \eta_k) \cdot \ln \sigma_k^{2(t-1)} + \eta_k \cdot \ln \sigma_\phi^2(e_t)$$

**使用对数方差**的好处：保证方差始终为正，数值稳定。

### 3.3 与 FIFO 的对比

| 性质 | FIFO（MA-LMM）| 自由能更新（本文）|

|------|--------------|-----------------|
| 为什么更新？ | 无原则，来一段存一段 | KL 量化了信息增益 |
| 更新多少？ | 固定（完全替换）| 自适应（$\eta_k \in (0,1)$）|
| 冲突时怎么办？ | 直接覆盖 | KL 大 → 大更新，保留高置信记忆 |
| 不确定性建模 | 无 | $\sigma_k^2$ 显式建模 |

---

## 4. 完整系统框架

### 4.1 整体流程

```
视频流（T 段）
    │
    ▼  对每段 t = 1, ..., T-1（历史段）
┌─────────────────────────────────────────┐
│  1. 视觉编码                             │
│     frames_t → InternViT → patch tokens  │
│     MLP Projector → v_t ∈ R^{N×d}       │
│                                          │
│  2. 证据压缩                             │
│     e_t = Compress(v_t) ∈ R^d           │
│     （可学习注意力压缩器）                │
│                                          │
│  3. 先验计算                             │
│     p(z|M_{t-1}) = Σ_k w_k N(μ_k,σ_k²) │
│                                          │
│  4. 后验推断（识别网络）                  │
│     q_φ(z|e_t) = N(μ_φ(e_t), σ_φ²(e_t))│
│                                          │
│  5. 自由能计算                           │
│     F = KL[q||p] - E_q[log p(e_t|z)]    │
│                                          │
│  6. 记忆更新                             │
│     η_k = σ(α·KL·w_k - β)              │
│     μ_k, σ_k² ← Bayesian update        │
└─────────────────────────────────────────┘
    │
    ▼  对最后一段（当前段）
┌─────────────────────────────────────────┐
│  7. 记忆读取                             │
│     memory_tokens ∈ R^{K×T_slot×d}     │
│     （K 个槽，每槽展开成 T_slot 个 token）│
│                                          │
│  8. LLM 推理                             │
│     Input = [memory_tokens; v_current; question] │
│     Output = answer                      │
└─────────────────────────────────────────┘
```

### 4.2 训练目标

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{LM}}}_{\text{语言建模损失}} + \lambda \cdot \underbrace{\mathcal{F}}_{\text{自由能辅助损失}}$$

- $\mathcal{L}_{\text{LM}}$：标准的 next-token prediction loss（交叉熵）
- $\mathcal{F}$：自由能损失，让记忆学会预测未来视频内容
- $\lambda$：权衡系数（初始设为 0.01，可调）

**自由能损失的作用**：即使最终答案监督信号稀疏（视频很长，只有最后一步有标签），自由能损失在每一段都提供监督，让记忆持续学习。

---

## 5. 与底座模型的集成

**底座选择**：InternVL3-8B（当前 8B 以下视频 VLM SOTA）

**需要修改的代码位置**：

```
InternVL/internvl_chat/model/internvl_chat.py
  → forward() 方法
  → MLP Projector 输出之后，LLM 输入之前
```

**可训练参数**（其余冻结）：

- 记忆压缩器（注意力模块，~10M 参数）
- 识别网络 $\mu_\phi, \sigma_\phi^2$（两层 MLP，~20M 参数）
- 解码器 $f_\theta$（线性层，~5M 参数）
- 记忆槽参数 $\{\mu_k, \sigma_k^2\}$
- InternVL3-8B LLM 的 LoRA 层

**估计额外参数量**：~40M，占底座 8B 的 0.5%

---

## 6. 与现有工作的关系

### 6.1 理论层面

| 理论来源                           | 本文的使用方式         |
| ---------------------------------- | ---------------------- |
| 自由能原理（Friston 2010）         | 记忆更新的整体优化目标 |
| VAE（Kingma & Welling, ICLR 2014） | 变分推断的具体实现框架 |
| 贝叶斯滤波（Kalman 1960）          | 先验→后验的更新结构    |
| 记忆再巩固（Nader et al., 2000）   | 方法设计的认知科学动机 |

### 6.2 最新相关工作（Bayesian + Memory）

近期已有工作将贝叶斯方法与记忆结合，但均与本文有本质区别：

**① EM-LLM（ICLR 2025）— 最接近**

> "Human-inspired Episodic Memory for Infinite Context LLMs"，Huawei Noah's Ark / UCL

- 用贝叶斯惊讶度 $-\log p(x_t \mid x_{<t})$ 检测记忆分段边界
- 超过自适应阈值 $T = \mu + \gamma\sigma$ 时触发新的情景记忆片段
- **区别**：只用惊讶度做分段触发器，无概率分布形式的记忆槽，无变分推断更新，针对文本 LLM 而非视频

**② Titans（NeurIPS 2025）— 惊讶度驱动**

> "Learning to Memorize at Test Time"，Google Research

- 记忆更新量正比于预测误差的梯度幅度（惊讶度代理）
- 遗忘门控制旧记忆衰减
- **区别**：无显式概率先验/后验结构，无变分推断，针对通用序列建模，Training-Free 无法端到端优化记忆选择策略

**③ LARIMAR（ICML 2024）— Bayesian 知识编辑**

> "Large Language Models with Episodic Memory Control"，IBM Research

- 将 LLM 知识更新建模为广义伪逆记忆操作（等价于一种 Bayesian 最小二乘更新）
- 支持一次性写入新事实，无需微调
- **区别**：针对静态知识编辑任务，不处理连续视觉流，无不确定性建模（无 $\sigma^2$）

**④ MESU（Nature Communications 2025）— 权重空间记忆**

> "Bayesian Continual Learning and Forgetting in Neural Networks"

- 将网络权重建模为概率分布，用 Bayesian 更新规则控制参数的学习速率
- 不确定性低的参数（已巩固知识）更新慢，不确定性高的参数保持可塑
- **区别**：记忆存在权重空间（参数后验），不是外部记忆槽；不处理视频

**空白确认**：目前没有工作同时满足：

```
(a) 完整变分生成模型（显式先验/后验/似然）
(b) 外部概率记忆槽（μ + σ²）的在线更新
(c) 流式视频 VLM 场景
```

本文填补这一空白。

### 6.3 视频记忆方法对比

**和 MA-LMM 的区别**：

- MA-LMM 写入的是「视觉特征的压缩」（低层，确定性向量）
- 本文写入的是「变分后验的均值」（高层语义，概率分布）

**和 ∞-Video 的区别**：

- ∞-Video 的更新权重由时间距离决定（手工设计，非自适应）
- 本文的更新权重由 KL 散度决定（自适应，端到端可学习）

**和 MemStream 的区别**：

- MemStream 用 Training-Free 的 MoE 检索选择关键帧（无法学习什么重要）
- 本文通过变分推断自动学习记忆写入策略

**和 Titans 的区别**：

- Titans 是启发式的梯度幅度加权，缺乏贝叶斯先验/后验的理论框架
- 本文有完整的 ELBO 目标，理论更严谨，且专门针对视频 VLM

### 6.4 综合对比表

| 论文     | 会议           |   Bayesian 框架   |  概率记忆槽   |  视频/多模态  | 端到端训练 |
| -------- | -------------- | :---------------: | :-----------: | :-----------: | :--------: |
| EM-LLM   | ICLR'25        |    惊讶度触发     |       ✗       |   ✗（文本）   |     ✗      |
| Titans   | NeurIPS'25     |   预测误差启发    |       ✗       | ✗（通用序列） |     ✗      |
| LARIMAR  | ICML'24        | Bayesian 最小二乘 |       ✗       | ✗（知识编辑） |     ✗      |
| MESU     | NatComm'25     |     权重后验      | ✗（权重空间） |       ✗       |     ✓      |
| MA-LMM   | CVPR'24        |         ✗         |       ✗       |       ✓       |     ✓      |
| **本文** | **NeurIPS'26** |   **完整 ELBO**   | **✓（μ+σ²）** | **✓（视频）** |   **✓**    |

---

## 7. 预期实验设计

### 7.1 Benchmark

| Benchmark          | 测试能力          | 当前 SOTA            |
| ------------------ | ----------------- | -------------------- |
| VideoMME（无字幕） | 综合视频理解      | Gemini 2.5 Pro 84.8% |
| LVBench            | 长视频专项（>1h） | MemStream            |
| MLVU               | 多任务长视频      | —                    |
| EgoSchema          | 第一人称长视频    | —                    |

### 7.2 消融实验

| 变体                            | 目的                   |
| ------------------------------- | ---------------------- |
| w/o 自由能（换回 FIFO）         | 验证自由能更新的必要性 |
| w/o 不确定性（$\sigma^2$ 固定） | 验证概率记忆的作用     |
| w/o KL 门控（$\eta$ 固定）      | 验证自适应更新率的作用 |
| 不同槽数 $K$（8/16/32）         | 超参数敏感性           |

### 7.3 算力估计（8× H200）

| 阶段                                | 估计时间 |
| ----------------------------------- | -------- |
| 跑通 InternVL3 baseline（VideoMME） | 0.5 天   |
| 接入记忆模块，LoRA 微调 1 轮        | <1 天    |
| 全部 benchmark 评测                 | 0.5 天   |
| 完整消融实验                        | 2–3 天   |

---

## 8. 开放问题（组会讨论）

1. **记忆读取策略**：所有槽都读出还是只读相关槽？
   - 全读：LLM 自己选择，但 token 数多
   - 选择读：减少 token，但需要额外的检索模块

2. **反思与自由能的结合**：
   - 当前框架中证据 $e_t$ 是视觉特征的压缩
   - 是否应该先用 LLM 生成语言反思，再把反思 embedding 作为 $e_t$？
   - 优点：更高层的语义；缺点：推理时额外调用 LLM，速度慢

3. **训练数据**：
   - 长视频 QA 数据稀缺（LVBench 只有训练集有限）
   - 是否用短视频合成长视频数据？（LongVPO 的思路）

4. **端到端 vs. 分阶段训练**：
   - 端到端：记忆模块和 LLM 联合优化，理论最优
   - 分阶段：先训记忆模块，再微调 LLM，工程更简单

---

## 附：关键符号表

| 符号                                    | 含义                          |
| --------------------------------------- | ----------------------------- |
| $M_t = \{(\mu_k, \sigma_k^2)\}_{k=1}^K$ | $t$ 时刻的记忆状态            |
| $e_t$                                   | 当前视频段的证据向量          |
| $z$                                     | 潜在记忆状态变量              |
| $p(z \mid M_{t-1})$                     | 先验分布（旧记忆的预测）      |
| $q_\phi(z \mid e_t)$                    | 后验近似（识别网络输出）      |
| $p(e_t \mid z)$                         | 似然函数（解码器）            |
| $\mathcal{F}$                           | 变分自由能                    |
| $\mathcal{L}_{\text{ELBO}}$             | 证据下界（$= -\mathcal{F}$）  |
| $KL[\cdot \| \cdot]$                    | KL 散度                       |
| $\eta_k$                                | 第 $k$ 槽的自适应更新率       |
| $K$                                     | 记忆槽数量                    |
| $d$                                     | 特征维度（InternVL3 为 4096） |

---

_文档版本：v1.0 | 2026-03-23 | 待组会讨论后更新_

