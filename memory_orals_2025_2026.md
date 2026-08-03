# 三大顶会 + 相关会议「记忆」方向 Oral 论文清单（2025–2026）

> 整理时间：2026-07-18 | 面向 NeurIPS 2026 投稿定位
> 范围：NeurIPS / ICML / ICLR 2025 与 2026 的 **Oral**（不含 spotlight / poster），主题与「记忆」相关

---
## 可信度标记（本文件的核心约定）

因为你特别要求"信息正确",本文件对每条都标注确认级别，**请按标记决定引用前是否需要自查**：

| 标记 | 含义 |
|------|------|
| ✅✅ | **我本轮亲自 fetch 了官方 oral 页 / 作者 repo / 会议录并复核** |
| ✅ | 由检索确认，且经**双源交叉验证**（papercopilot OpenReview 镜像 + 官方 iclr.cc/icml.cc/neurips.cc oral 页），但我未逐条亲自打开 |
| ⚠️ | **oral 状态存疑**：被检索标为 oral 但找不到官方 `/oral/` 页，**当 poster/spotlight 处理，不要引成 oral** |
| ❌ | **已确认不是 oral**（是 poster），列出以防误引 |

**未核实项（全文件适用）**：机构信息基本未逐一核实（OpenReview 被 Cloudflare 挡，多数按作者名推断，未写入本文件）；作者名只在能确认处给出。**正式引用前请打开对应 OpenReview / 会议录页确认作者与机构。**

---

## ⚠️ 首要结论：顶会 oral 的「记忆」≠ 你做的「视频记忆」

扫完 2025–2026 四届（NeurIPS'25、ICLR'25、ICML'25、ICLR'26、ICML'26）的全部 oral 后，**没有一篇是 video-VLM 记忆的 oral**。

视频记忆方向（FluxMem / SelectStream / MM-Mem 等，见 `video_vlm_survey.md`）**主要发在 CVPR 和 arXiv，在三大顶会拿不到 oral**。三大顶会 oral 认可的"记忆"是另外五种、更偏底层/理论的口味：

1. **状态空间 / 线性 RNN 记忆**（Mamba 血脉）— 数量最多，每届都有
2. **KV-Cache 压缩**（把 KV 当记忆裁剪）
3. **联想记忆 / Hopfield 理论**（证明注意力 = 联想记忆检索）
4. **检索即记忆**（RAG、retriever 训练）
5. **持续学习 / 遗忘**（replay buffer、catastrophic forgetting）

**对项目的含义**：若目标是 NeurIPS 2026 **oral**（而非仅 accept），"在 InternVL 上加视频记忆模块刷高 benchmark"这个故事历史上没拿过 oral。Oral 级记忆论文要么有**理论深度**（证明了什么），要么提出**新序列建模原语**。我们的**变分自由能角度是唯一能往理论深度靠的抓手** —— 论文重心应放在"为什么变分表述在原理上更对"，而非堆 SOTA 数字。

---

## 与本项目最相关的精读短名单（跨会议挑出）

如果只读三篇，读这三篇：

1. **MemAgent**（ICLR 2026 Oral，纯文本）✅✅ — 固定大小 token 记忆 + 跨轮覆写，**结构与我们"固定 slot + 更新"最像**
2. **RAP / Retrieval-Augmented Perception**（ICML 2025 Oral，多模态）✅ — 视觉 RAG，把图像 crop 当外部记忆，**多模态里最相关**
3. **In-Context Denoising = Associative Memory**（ICML 2025 Oral，理论）✅✅ — 证明注意力 = 在 Hopfield 能量面走一步梯度；**要讲"记忆的原理"绕不开这类**

其他多模态/具身相关的 oral（次相关）：Dynam3D（NeurIPS'25，VLM 导航空间记忆）、Latent Particle World Models（ICLR'26，object-centric 世界模型）、RoboMME（ICML'26，机器人记忆 benchmark）。

---

## NeurIPS 2025

**核实说明**：该会 OpenReview 被反爬挡，oral 状态靠 neurips.cc 官方页 + papers.cool Oral 分组 + 作者 repo 交叉确认。全 77 篇 oral 中逐条核到约 50 篇，尾部可能有遗漏（低概率）。

### 核心记忆 oral

| # | 论文 | 确认 | 口味 | 一句话 | 链接 |
|---|------|------|------|--------|------|
| 1 | **KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction** | ✅✅ | KV-Cache 🎯 | query 无关的 KV 缓存压缩，靠上下文重建判断重要性，KV 显存降 3–4× | [arXiv 2505.23416](https://arxiv.org/abs/2505.23416) · [repo](https://github.com/snu-mllab/KVzip) |
| 2 | **Memory Mosaics at scale** | ✅ | LLM 架构 | 联想记忆网络堆到 llama-8B 规模，in-context 新任务超 Transformer | — |
| 3 | **Dynam3D: Dynamic Layered 3D Tokens Empower VLM for Vision-and-Language Navigation** | ✅ | 视觉/具身 🎯 | 动态 3D token 给 VLM 导航用的长期空间记忆 | [arXiv 2505.11383](https://arxiv.org/abs/2505.11383) |

- KVzip 作者：Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song。repo 原话："🎉 KVzip has been accepted at NeurIPS 2025 as an **Oral Presentation**"（我亲自核实）。
- Memory Mosaics 作者：Jianyu Zhang, Léon Bottou。
- Dynam3D 作者：Zihan Wang, Seungjun Lee, Gim Hee Lee。

### 确认 oral，但「记忆」是次要属性

- **Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free** ✅ — 去注意力沉降，利好长上下文
- **Learning long range dependencies through time reversal symmetry breaking** ✅ — 训练递归系统持有长程记忆（理论）[arXiv 2506.05259](https://arxiv.org/abs/2506.05259)
- **Class-wise Balancing Data Replay for Federated Class-Incremental Learning** ✅ — replay 记忆（持续学习）
- **Learning Linear Attention in Polynomial Time** ✅ — 线性注意力 ≈ fast-weight/联想记忆，但本质是学习理论

### ❌ 明确不是 oral（防误引）

- **Titans: Learning to Memorize at Test Time** — ❌ **是 Poster**（[neurips.cc poster 119639](https://neurips.cc/virtual/2025/poster/119639)）。名气大，容易误当 oral。
- **A-Mem: Agentic Memory for LLM Agents** — ❌ Poster。

---

## ICLR 2025

**核实说明**：该会数据最干净 —— papercopilot 全表 213 篇 oral（数目与官方一致）逐条扫过，再与 iclr.cc 虚拟 oral 页交叉确认。Bucket B 为空（每篇候选都能双源确认 oral）。

### 核心记忆 oral

| 论文 | 确认 | 口味 | 一句话 | 链接 |
|------|------|------|--------|------|
| **Oscillatory State-Space Models (LinOSS)** | ✅ | SSM 记忆 | 振荡型 SSM，稳定长程依赖 | [OpenReview GRMfXcAAFh](https://openreview.net/forum?id=GRMfXcAAFh) |
| **Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues** | ✅ | Mamba 记忆 | 用负特征值扩展线性 RNN 递归状态表达力，解锁状态追踪 | [OpenReview UvTo3tVBk2](https://openreview.net/forum?id=UvTo3tVBk2) |
| **Retrieval Head Mechanistically Explains Long-Context Factuality** | ✅ | LLM 机理 | 定位"检索头"——从长上下文复制信息的 in-context 记忆机制 | [OpenReview EytBpUGB1Z](https://openreview.net/forum?id=EytBpUGB1Z) |
| **FlexPrefill: Context-Aware Sparse Attention for Efficient Long-Sequence Inference** | ✅ | 长上下文/KV | 自适应稀疏注意力 | [OpenReview OfjIlbelrT](https://openreview.net/forum?id=OfjIlbelrT) |
| **Inference Scaling for Long-Context Retrieval Augmented Generation** | ✅ | 检索即记忆 | 长上下文 RAG 的推理算力扩展（Google） | [OpenReview FSjIrOm1vz](https://openreview.net/forum?id=FSjIrOm1vz) |
| **Prioritized Generative Replay** | ✅ | RL 🎯 | 带优先级的生成式 replay 缓冲（记忆） | [OpenReview 5IkDAfabuo](https://openreview.net/forum?id=5IkDAfabuo) |
| **REGENT: A Retrieval-Augmented Generalist Agent** | ✅ | RL/agent 🎯 | 检索过去 demo/轨迹当记忆的 in-context 通才 agent | [OpenReview NxyfSW6mLK](https://openreview.net/forum?id=NxyfSW6mLK) |
| **Open-World RL over Long Short-Term Imagination (LS-Imagine)** | ✅ | RL 🎯 | 带长短期记忆的世界模型，延长想象视野 | [OpenReview vzItLaEoDa](https://openreview.net/forum?id=vzItLaEoDa) · [oral 页](https://iclr.cc/virtual/2025/oral/31740) |

- LinOSS 作者：T. Konstantin Rusch, Daniela Rus。
- Negative Eigenvalues 作者：Grazzi, Siems, Zela, Franke, Hutter, Pontil。

### 记忆邻近（确认 oral，但偏"参数化知识/编辑"）

- **AlphaEdit: Null-Space Constrained Knowledge Editing** ✅ — 编辑存在权重里的事实 = 参数记忆编辑 [OpenReview HvSytvg3Jh](https://openreview.net/forum?id=HvSytvg3Jh)
- **Synthetic Continued Pretraining (EntiGraph)** ✅ — 把小语料内化进参数记忆
- **Knowledge Entropy Decay during LM Pretraining** ✅ — 参数化知识记忆的动力学
- **Differential Transformer** ✅（Furu Wei 组）— 差分注意力消噪，改善从 KV 检索长上下文 [OpenReview OvoCm1gGhN](https://openreview.net/forum?id=OvoCm1gGhN)

---

## ICML 2025

**核实说明**：papercopilot 全表 121 篇 oral 逐条扫过 + icml.cc 交叉确认。

### 核心记忆 oral

| 论文 | 确认 | 口味 | 一句话 | 链接 |
|------|------|------|--------|------|
| **In-Context Denoising with One-Layer Transformers: Connections between Attention and Associative Memory Retrieval** | ✅✅ | 联想记忆理论 🎯 | 证明训练好的注意力层 = 在 context-aware 稠密联想记忆（现代 Hopfield）能量面上走一步梯度；context token 是联想记忆，query 是初始态 | [arXiv 2502.05164](https://arxiv.org/abs/2502.05164) · [MLR v267/smart25a](https://proceedings.mlr.press/v267/smart25a.html) |
| **Retrieval-Augmented Perception: High-resolution Image Perception Meets Visual RAG (RAP)** | ✅ | 多模态 🎯 | 视觉 RAG 检索/融合图像 crop 当外部记忆，服务高分辨率 MLLM | [OpenReview X9vBykZVYg](https://openreview.net/forum?id=X9vBykZVYg) |
| **Navigating Semantic Drift in Task-Agnostic Class-Incremental Learning** | ✅ | 视觉/持续学习 | 语义漂移校准，保留旧类知识（抗遗忘） | [OpenReview M6L7Eaw9BW](https://openreview.net/forum?id=M6L7Eaw9BW) |
| **Learning Dynamics in Continual Pre-Training for LLMs** | ✅ | LLM/持续 | 建模持续预训练中的遗忘 vs 获取 | [OpenReview Vk1rNMl0J1](https://openreview.net/forum?id=Vk1rNMl0J1) |

- In-Context Denoising 作者：Matthew Smart, Alberto Bietti, Anirvan M. Sengupta。**已亲自核实 oral**（MLR 会议录 + [官方 slides PDF](https://icml.cc/media/icml-2025/Slides/47245.pdf) + 2025-07-17 Vancouver 报告记录）。注：该论文同时有 [poster 页 45913](https://icml.cc/virtual/2025/poster/45913)，这是**正常现象**（ICML oral 论文也会安排 poster 场次），不否定 oral 状态。

### 记忆邻近（确认 oral）

- **Learning dynamics in linear recurrent neural networks** ✅ — 线性 RNN（递归记忆）学习动力学理论 [OpenReview KGOcrIWYnx](https://openreview.net/forum?id=KGOcrIWYnx)
- **Mixture of Lookup Experts** ✅ — 把 MoE 专家重参数化为推理期查找表（查找/表记忆） [OpenReview wUEp13rqXP](https://openreview.net/forum?id=wUEp13rqXP)

---

## ICLR 2026

**核实说明**：oral 全公开（223 篇）。数据源为 iclr.cc 官方 oral 页 + 策展 repo `XinyuLiuCs/iclr2026-oral-papers`。我亲自打开 iclr.cc oral 页确认了 MemAgent。

### 核心记忆 oral

| 论文 | 确认 | 口味 | 一句话 | 链接 |
|------|------|------|--------|------|
| **MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent** | ✅✅ | LLM/agent 🎯 | RL 训练的 agent 分块读长文档 + 维护固定大小 token 记忆并跨轮覆写 | [oral 页 10007826](https://iclr.cc/virtual/2026/oral/10007826) · [OpenReview k5nIOvYGCL](https://openreview.net/forum?id=k5nIOvYGCL) |
| **Mamba-3: Improved Sequence Modeling using State Space Principles** | ✅ | SSM 记忆 | 改进 SSM 递归（复数状态、MIMO），隐状态即压缩记忆 | [oral 页 10010353](https://iclr.cc/virtual/2026/oral/10010353) · [OpenReview HwCvaJOiCj](https://openreview.net/forum?id=HwCvaJOiCj) |
| **ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models** | ✅ | KV-Cache | 按推理"思维"自适应压缩 KV 缓存 | [OpenReview M3CeHnZKNC](https://openreview.net/forum?id=M3CeHnZKNC) |
| **Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling** | ✅ | 视觉/RL 🎯 | object-centric 隐世界模型（场景状态记为粒子/关键点） | [OpenReview lTaPtGiUUc](https://openreview.net/forum?id=lTaPtGiUUc) · [repo](https://github.com/taldatech/lpwm) |
| **Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training** | ✅ | 检索即记忆 | value-based（RL）训练 retriever 做多步长上下文检索 | [OpenReview MS9nWFY7LG](https://openreview.net/forum?id=MS9nWFY7LG) |
| **Revela: Dense Retriever Learning via Language Modeling** | ✅ | 检索即记忆 | 通过语言建模自监督训练稠密 retriever | [oral 页 10008351](https://iclr.cc/virtual/2026/oral/10008351) · [OpenReview e7pAjJZJWb](https://openreview.net/forum?id=e7pAjJZJWb) |
| **To Infinity and Beyond: Tool-Use Unlocks Length Generalization in SSMs** | ✅ | SSM 理论 | 研究/扩展 SSM 有限状态记忆的长度泛化 | [OpenReview sSfep4udCb](https://openreview.net/forum?id=sSfep4udCb) |
| **From Markov to Laplace: How Mamba In-Context Learns Markov Chains** | ✅ | SSM 理论 | 分析 Mamba 状态记忆如何编码 in-context 统计 | [OpenReview kmK3WSCOCT](https://openreview.net/forum?id=kmK3WSCOCT) |

- MemAgent 作者（iclr.cc 页亲自读取）：Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, Hao Zhou。Oral 时间 Thu, Apr 23, 2026。

### 记忆邻近（确认 oral，但记忆很边缘）

- **Hubble: a Model Suite to Advance the Study of LLM Memorization** ✅（ZfdnZhOP0k）— 研究记忆化（memorization），非记忆模块
- **WoW!: World Models in a Closed-Loop World** ✅（yDmb7xAfeb）
- **LoongRL** ✅（o29E01Q6bv）— 长上下文 RL，无专门记忆机制

---

## ICML 2026

**⚠️ 覆盖度警告**：oral 已公开（168 篇），但官方页抓取会截断，我方只**逐条核到约 80/168 篇**。因此下表是**下限，非全集** —— 未枚举的 ~88 篇里可能还有记忆 oral。

### 核心记忆 oral（已确认）

| 论文 | 确认 | 口味 | 一句话 | 链接 |
|------|------|------|--------|------|
| **RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies** | ✅ | RL/机器人 🎯 | 机器人 VLA 策略的时序/空间/物体/程序记忆 benchmark，做了 14 个记忆增强 π0.5 变体 | [oral 页 71077](https://icml.cc/virtual/2026/oral/71077) · [arXiv 2603.04639](https://arxiv.org/abs/2603.04639) |
| **AI Engram: In Search of Memory Traces in Artificial Intelligence** | ✅ | 可解释性 | 寻找训练模型内部的"记忆痕迹"（engram） | [oral 页 71045](https://icml.cc/virtual/2026/oral/71045) |
| **MuonSSM: Orthogonalizing State Space Models for Sequence Modeling** | ✅ | SSM 记忆 | 正交化治 SSM 长程记忆退化 | [oral 页 71058](https://icml.cc/virtual/2026/oral/71058) |
| **TG-RAG: A Retrieval-Augmented Framework for Reasoning Guidance in Specialized Domains** | ✅ | 检索即记忆 | 动态中断-检索-生成 RAG | [oral 页 71061](https://icml.cc/virtual/2026/oral/71061) |

记忆邻近：**Multimodal Nested Learning for Decoupled and Coordinated Optimization** ✅（[oral 71041](https://icml.cc/virtual/2026/oral/71041)）—— nested learning 是一种多时间尺度记忆/持续学习范式，但标题未突出记忆。

### ⚠️ oral 状态存疑（被检索误标，找不到官方 oral 页 —— 不要引成 ICML 2026 oral）

- **ATLAS: Learning to Optimally Memorize the Context at Test Time**（Behrouz et al., Google；Titans 血脉）—— test-time 长期记忆模块，很相关，但**oral 状态未确认** [OpenReview dpPW15y3n8](https://openreview.net/forum?id=dpPW15y3n8) · [arXiv 2505.23735](https://arxiv.org/abs/2505.23735)
- **Expressivity-Efficiency Tradeoffs for Hybrid Sequence Models**（UW-Madison）—— 出现在 papers.cool ICML Oral 分组但 icml.cc 无法印证，**倾向 oral 但未确认** [arXiv 2603.08859](https://arxiv.org/abs/2603.08859)
- **Semantic Integrity Matters (ShotKV): ... KV Cache Compression** —— arXiv 注释仅写 "ICML 2026" 无 oral/poster 标签，找不到官方页，**存疑、大概率 poster** [arXiv 2502.01941](https://arxiv.org/abs/2502.01941)

### ❌ 明确不是 oral（2026-07 复核更正）

- **Predicting Future KV Utility (LU-KV): Global Combinatorial Optimization for Task-Agnostic KV Cache Eviction** —— ❌ **Poster**（[poster 65241](https://icml.cc/virtual/2026/poster/65241)）。*原列为"存疑"，已核实为 poster。*
- **STAR-KV: Low-Rank KV Cache Compression via Soft Thresholding** —— ❌ **Spotlight**（非 oral；[页 61958](https://icml.cc/virtual/2026/poster/61958)）。*原误标为 Poster，实为 Spotlight，两者都不是 oral。*

---

## 汇总：按「口味」横向看（帮助定位自己）

| 口味 | 代表 oral（跨会议） | 与我们的关系 |
|------|--------------------|-------------|
| **SSM / 线性 RNN 记忆** | LinOSS、Negative-Eigenvalue RNN、Mamba-3、MuonSSM、多篇理论 | 数量最多，是 oral 主流；我们不在这条线 |
| **

** | KVzip、ThinKV | training-free 主流；我们是 trained，正交 |

| **联想记忆 / Hopfield 理论** | In-Context Denoising | **讲"记忆原理"必读** |
| **检索即记忆** | RAG scaling、REGENT、Q-RAG、Revela、RAP、TG-RAG | 记忆=外部数据库+检索，与我们的"隐分布记忆"哲学不同 |
| **持续学习 / 遗忘** | Class-wise Replay、Semantic Drift、Prioritized Generative Replay | 记忆=抗遗忘，相关但侧重不同 |
| **世界模型 / 具身记忆** | LS-Imagine、Latent Particle World Models、Dynam3D、RoboMME | 多模态/具身，次相关 |
| **视频 VLM 记忆** | **（无 oral）** | ← 我们所在，顶会 oral 空白，主战场在 CVPR/arXiv |

---

## 专项判断：KV-Cache 压缩这个方向（2026-07-19 复核）

**一句话**：发文量极高、但顶会 oral 近乎为零、且正被架构变革封顶。

### 产量 vs oral 的强烈反差
- arXiv 上「KV cache + compression」约 **546 篇**，2607 批次仍在每周出新（如 FreqDepthKV 2607.06519）。
- 每届顶会**接收几十篇**（ICLR'26 总接收 5355、ICML'26 总接收 6352，KV 类各"轻松几十篇"）。
- 但**确认的 KV 压缩 oral 只有 2 篇**：**KVzip**（NeurIPS'25，[oral 118742](https://nips.cc/virtual/2025/oral/118742)）、**ThinKV**（ICLR'26，[oral 10009981](https://iclr.cc/virtual/2026/oral/10009981)）。ICML'26 可枚举部分（~25–82/168）为 0；**未枚举部分无法排除，但方向不改**：几十篇接收 → 0–2 篇 oral。
- **解读**：这个方向是"在已知框架里换判据"（换 eviction 分数/量化位宽/低秩），工程增量大、思想增量小 → 接收容易、oral 极难。

### 饱和信号（多个，齐全）
- 综述：ACL'26《System-Aware KV Cache Optimization》综述，分类已标准化成 5 桶。
- 打脸论文：《The Pitfalls of KV Cache Compression》（ACL'26, [2510.00231](https://arxiv.org/abs/2510.00231)）—— 压缩致丢指令、泄露系统提示。
- 触顶：量化路线（Google TurboQuant）误差"逼近香农下界"，可压空间将尽。

### 结构性威胁：线性/混合架构（关键）
- KV 缓存是 **softmax 注意力**专属。Qwen3.5（Gated DeltaNet 混合，约 25% 层才留 KV）、Mamba-3（ICLR'26, [2603.15569](https://arxiv.org/abs/2603.15569)）把历史压进**固定大小隐状态，不随序列增长** → "缓存太大"这个问题本身在缩小。
- **未死**：混合架构仍留全注意力层；线性状态**检索能力弱**（Mamba-3 的 MIMO 就在修这个）；存量纯 softmax 模型（Llama、Qwen2.5/3、**几乎所有视频 VLM**）不会消失。研究界已出《HybridKV》（ACL'26, [2604.05887](https://arxiv.org/abs/2604.05887)）反应。

### 视频侧仍成立，但反攻已至
- 长视频 KV 爆炸是实打实的（有工作实测 LLaVA-OV-7B 1000 帧/batch 256 → KV 缓存约 720GB）；当前视频 VLM（Qwen2.5/3-VL、InternVL）全是纯 softmax。
- 但线性化压力也进入视频：Stanford《Linear Scaling Video VLMs》（[2605.31598](https://arxiv.org/abs/2605.31598)）已在做无 KV 缓存的长视频。

### 对本项目的含义
- 不建议挤进 KV 压缩的**拥挤核心**（几十抢零 oral + 饱和 + 被架构威胁 + 工业界主场）。
- 我们的 `FreeEnergyMemory` 是**学出来的、固定大小分布式记忆**，恰好站在"把历史压进固定状态"这股浪潮**这一侧**，而非被冲走的一侧——这比"又一个 KV 压缩"是更强的 motivation。

---

## 覆盖度与诚实边界（务必读）

1. **NeurIPS 2025**：77 篇 oral 逐条核到约 50 篇；OpenReview 反爬，oral 状态靠官方页+repo 交叉确认，非 OpenReview decision 字段。尾部可能漏低概率记忆 oral。
2. **ICLR/ICML 2025**：oral 状态**双源确认，可信度最高**；残余风险是"标题不含记忆关键词、只有摘要才看得出记忆相关"的论文可能被漏（topic recall 风险，非 oral-status 风险）。
3. **ICLR 2026**：oral 全公开，MemAgent 我亲自核实；其余靠策展 repo + 抽样官方页确认。
4. **ICML 2026**：**只核到 ~80/168 篇，是下限**。ATLAS 等 4 篇 KV/记忆论文的 oral 状态未确认，已单列 ⚠️。
5. **机构与部分作者未逐一核实** —— 正式引用前请打开 OpenReview / 会议录页。
6. 官方 OpenReview 页普遍被 Cloudflare 挡，本文件链接可用但需人工过验证页。

---

*生成：2026-07-18 | 数据来自 neurips.cc / iclr.cc / icml.cc 官方 oral 页、papercopilot OpenReview 镜像、papers.cool Oral 分组、作者 repo；旗舰条目经本人二次 fetch 复核（✅✅）*
