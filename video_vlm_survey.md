# Video VLM 文献调研报告

> 初次整理：2026-03-21 | **最近更新：2026-07-16**（补充 2026 年 2–7 月工作）
> 面向 NeurIPS 2026 投稿准备

---

## ⚠️ 本次更新要点（2026-07-16）

补完 3 月至今的四个月后，有三个发现直接影响项目定位，建议先读第 9 节：

1. **2026 年的记忆工作压倒性地转向 Training-Free。** 新增的十几篇里绝大多数不训练记忆，靠 KV-Cache 压缩 + 检索。我们端到端训练的 `nn.Parameter` 记忆是逆流而行 —— 这既是差异化，也是审稿人必问的问题。
2. **「惊讶度门控写入」已经不新了。** WeaveTime（CVPR'26）用 uncertainty 触发检索，SelectStream 用 surprise-driven 窗口决定何时写。二者都是启发式信号 —— 我们的 KL 是它的原理化版本，但**门控这个"想法"本身已被占坑**，卖点必须收缩到变分表述上。
3. **底座 InternVL3-8B 已经落后两代，且 32K 上下文 vs Qwen3.5 的 256K。** 这动摇的是项目前提本身：底座原生装得下 256K token 时，记忆模块的必要性论证要重写。

**好消息**：检索到 7 月为止，**没有找到任何用变分自由能 / ELBO 来门控视频记忆写入的工作**。最接近的是 MM-Mem 的语义信息瓶颈（ACL'26）。这个坑还空着。

**本节可信度**：所有标 ✅ 的论文我都逐条 fetch 了 arXiv 页面核实（标题/作者/日期/venue/摘要）。**机构信息基本未经核实**（arXiv 摘要页不列 affiliation，我按作者名推断的一律标注）。**benchmark 数字多为摘要级**，填表前需要查 PDF。

---

## 0. VLM 基础架构（必读前置）

你做过 LLM 的注意力修改，VLM 只是在 LLM 前面多了一个视觉处理模块：

```
视频输入
   ↓
① 帧采样（每秒取 N 帧）
   ↓
② 视觉编码器 ViT（把每帧图像变成 patch tokens）
   每帧 ≈ 256 个 token
   ↓
③ Projector（把视觉 token 映射到 LLM 的 embedding 空间）
   两种：MLP（简单直接）或 Q-Former（先压缩再映射）
   ↓
④ LLM（你熟悉的部分：LLaMA / Qwen / InternLM）
   ↓
⑤ 文字输出（回答问题）
```

**视频的核心矛盾：**
> 100 帧 × 256 tokens/帧 = 25,600 个视觉 token → LLM 上下文装不下，且计算量爆炸

**整个视频 VLM 领域的所有工作，本质上都在解决这一个问题。**
方法分两大类：压缩 token，或者用记忆机制绕开它。

> **2026 更新**：这个矛盾正在被底座本身稀释。Qwen3.5 原生 256K 上下文（可扩到 1M），
> 400 秒 720p @1fps 直接塞得下。「装不下」的前提在变弱，
> 但「装得下 ≠ 用得好」—— 见第 8 节 RIVER-bench 的预算-精度标度律。

---

## 第一类：基础视频 VLM（奠基工作，无专门记忆机制）

### 1. LLaVA — NeurIPS 2023 Oral ⭐

**机构**：UW Madison + Microsoft Research
**论文**：Visual Instruction Tuning

| 项目 | 内容 |
|------|------|
| 视觉编码器 | CLIP ViT-L/14 |
| Projector | 简单线性层（后改为 MLP） |
| LLM | LLaMA / Vicuna-13B |
| 核心创新 | 用 GPT-4 生成 158K 条指令微调数据，将视觉理解转化为指令跟随任务 |
| 局限 | 只处理静态图像，无视频；token 数量固定 |

**意义**：确立了「视觉编码器 + Projector + LLM」三段式范式，后续几乎所有工作都在此框架上扩展。

---

### 2. Video-LLaMA — EMNLP 2023

**机构**：阿里达摩院

| 项目 | 内容 |
|------|------|
| 视觉编码器 | EVA-CLIP ViT-G |
| Projector | Q-Former（来自 BLIP-2，先压缩再映射） |
| LLM | LLaMA-2 / Vicuna |
| 核心创新 | 首个同时支持视频+音频的多模态 LLM；Video Q-Former 专门处理时序帧 |
| 局限 | Q-Former 压缩过激，视频细节丢失；只能处理短视频（<3 分钟） |

---

### 3. Video-ChatGPT — ACL 2024

**机构**：MBZUAI

| 项目 | 内容 |
|------|------|
| 核心创新 | 时空特征聚合：所有帧做时间维度平均 + 空间维度平均，得两个全局特征向量拼接输入 LLM |
| 局限 | 特征平均丢失大量局部细节，时序信息几乎消失 |

---

### 4. LLaVA-Video — arXiv 2024

**机构**：NTU + ByteDance

| 项目 | 内容 |
|------|------|
| 核心创新 | 构建高质量视频指令微调数据集 LLaVA-Video-178K；AnyRes 策略处理不同分辨率 |
| 局限 | 均匀帧采样，长视频靠增大帧数硬撑，无显式记忆机制 |

---

## 第二类：记忆增强型长视频 VLM（2024–2025 奠基）

### 5. MC-ViT — ICML 2024 ⭐⭐

**机构**：Google DeepMind
**论文**：Memory Consolidation Enables Long-Context Video Understanding

| 项目 | 内容 |
|------|------|
| 记忆机制 | 非参数化记忆合并（KMeans / Coreset 选取历史 token） |
| 核心创新 | 受神经科学记忆重构理论启发：将历史段压缩为 K 个代表性 token 作为「记忆」，与当前帧做 Cross-Attention |
| 局限 | 纯视觉模型，无语言模型；记忆是非参数的，无法学习「什么值得记」 |

> **2026 状态**：**未找到 MC-ViT 后续工作**。这条线似乎断了。

---

### 6. MA-LMM — CVPR 2024 ⭐⭐

**机构**：Meta AI

| 项目 | 内容 |
|------|------|
| 记忆机制 | 显式 Memory Bank（先进先出队列） |
| 局限 | Memory Bank 随视频变长线性增长；无遗忘机制；写入策略简单（FIFO） |

> **2026 状态**：血脉延续到 **FluxMem（CVPR'26）**——一作 Bo He 正是 MA-LMM 一作。
> 但注意这个转向：**从"训练的记忆库"退回到"训练无关的压缩"**。
> 同一批人主动放弃了训练记忆，这个信号值得我们警惕并在 rebuttal 里准备好回答。

---

### 7. ∞-Video — ICML 2025 ⭐⭐⭐

**机构**：Google DeepMind

| 项目 | 内容 |
|------|------|
| 记忆机制 | 连续时间长期记忆（LTM）合并，训练无关 |
| 局限 | 衰减权重手工设计，非自适应；「重要性」无法从任务角度学习 |

> **2026 状态**：未找到后续工作。

---

### 8. ReWind — CVPR 2025 ⭐⭐

| 项目 | 内容 |
|------|------|
| 记忆机制 | 可学习记忆 token（Instructed Learnable Memory），由问题文本指导选择性读取 |
| 局限 | Memory token 数量固定；EVA-02 + LLaMA-2 底座已过时 |

**与 MA-LMM 的关键区别**：MA-LMM 无条件写入；ReWind 指令驱动选择性存储。

---

### 9. LongVPO — NeurIPS 2025 ⭐⭐⭐ / 10. Eagle 2.5 — NeurIPS 2025 ⭐⭐

- **LongVPO**（南大 MCG）：不改架构改训练，用 DPO 让模型从稀疏帧推断；16K 合成数据超 SOTA。
- **Eagle 2.5**（NVIDIA）：ADS + IAP 工程优化；8B 在 512 帧下 VideoMME 72.4%。

---

### 11. MemStream — arXiv 2602.18434 ✅

**论文**：Going Down Memory Lane: Scaling Tokens for Video Stream Understanding with Dynamic KV-Cache Memory
**作者**：Vatsal Agarwal, Saksham Suri, Matthew Gwilliam, Pulkit Kumar, Abhinav Shrivastava（UMD Shrivastava 组，*机构未核实*）
**链接**：https://arxiv.org/abs/2602.18434 | 2026-02-20

| 项目 | 内容 |
|------|------|
| 基础模型 | Qwen2.5-VL-7B |
| 记忆机制 | 动态稀疏 KV-Cache + 训练无关 MoE 检索 |
| 性能 | **LVBench +8.5%、CG-Bench +8.0%、VideoMME(Long) +2.4%（均 vs ReKV）** |
| 局限 | Training-Free，依赖外部专家模型检索，无学习到的记忆状态 |

**一个值得我们注意的发现**：他们指出现有编码器中 **query-frame 相似度随时间递增**，导致检索系统性偏向后期帧。
如果我们的 `get_prior` 用余弦相似度算 slot 权重，可能有同样的时间偏置 —— 值得做个诊断实验。

---

## 第三类：2026 新工作 ⭐（本次新增，与我们直接竞争）

> 检索范围：CVPR/ICLR/ICML/ACL/AAAI 2026 + arXiv 2602–2607。
> 数量太多，按"记忆是怎么来的"分三簇。

### 3A. KV-Cache 即记忆（训练无关，当前主流）

| 论文 | 出处 | 机制 | 数字 |
|------|------|------|------|
| **FluxMem** ✅<br>[2603.02096](https://arxiv.org/abs/2603.02096) | **CVPR 2026** | 两阶段：Temporal Adjacency Selection（帧间去冗余）+ Spatial Domain Consolidation（帧内合并）；**压缩率自适应场景**，非手工设定。短/中/长期三级记忆按"到 query 的时间距离"组织 | StreamingBench 76.4、OVO-Bench 67.2、MLVU 73.1；延迟 −69.9%、峰值显存 −34.5%、visual token −65% |
| **FlexMem** ✅<br>[2603.29252](https://arxiv.org/abs/2603.29252) | **CVPR 2026** | 视觉 KV cache 作记忆源；双通路压缩负责"写"，自适应读取策略负责"读"（含流式）。模拟人类"持续观看 + 回忆相关片段" | 单张 3090 处理 1000+ 帧；称在部分 benchmark 上比肩/超过 GPT-4o、Gemini-1.5-Pro。**摘要无具体数字** |
| **HERMES** ✅<br>[2601.14724](https://arxiv.org/abs/2601.14724) | **ACL 2026 Main** | 机理性注意力分析 → 把 KV cache 重新解释为多粒度层级记忆，复用紧凑 cache | TTFT 快 10×、video token −68%、流式 +11.4% |
| **CausalMem** ✅<br>[2606.25658](https://arxiv.org/abs/2606.25658) | arXiv 06-24 | **固定预算**记忆库；在线 SVD 语义基底估计 token 冗余并驱逐 | VideoMME 60.0、MLVU 70.9、LongVideoBench 49.5、LVBench 57.5；>20× 压缩，小时视频 ~82MB |
| **SAVEMem** ✅<br>[2605.07897](https://arxiv.org/abs/2605.07897) | arXiv 05 | 三层流式记忆 + **伪问题库**作语义先验决定保留什么 | OVO-Bench 52.27→62.69；128 帧显存 −48% |
| **FreshMem** ✅<br>[2602.01683](https://arxiv.org/abs/2602.01683) | arXiv 02 | 脑启发频域-空域混合：溢出帧转频率系数+残差，配合空间缩略图 | StreamingBench +5.20%、OVO-Bench +2.34% |
| **CurveStream** ✅<br>[2603.19571](https://arxiv.org/abs/2603.19571) | arXiv 03 | **曲率分数**检测特征轨迹的语义突变，K-Sigma 动态阈值分主/次记忆 | StreamingBench +10.69%、OVOBench +13.58% |
| **Vista** ✅<br>[2602.08448](https://arxiv.org/abs/2602.08448) | **AAAI 2026** | 场景分割 → 压缩（紧凑 token 留 GPU，全帧卸载 CPU）→ query 时选择性召回 | 摘要无数字 |
| **OmniMem** ✅<br>[2606.07577](https://arxiv.org/abs/2606.07577) | arXiv 06 | 音视频**模态感知预算分配** + 扰动感知 KV 选择。Video-SALMONN 2+ / Qwen2.5-Omni | 等预算下 +2–4%，微调后再 +1–2% |

**这一簇的共同软肋（也是我们的切入口）**：压缩准则全部是**视觉/统计冗余**驱动（相邻相似、SVD 基底、曲率突变），
没有一个是从**任务目标**学出来的。"什么值得记"依然是手工设计的代理指标。

---

### 3B. 学习/隐状态记忆（**与我们最直接竞争**）

#### ⚠️ SelectStream — What Should a Streaming Video Model Remember? ✅ **最近竞争者**

**链接**：https://arxiv.org/abs/2606.16353 | 2026-06-15
**作者**：Haonan Ge, Yiwei Wang, Hang Wu, Yujun Cai

| 项目 | 内容 |
|------|------|
| 框架 | 冻结 VLM；当前观测直接可见，历史**只通过紧凑的 query-conditioned 证据预算**以 latent token 注入（不回放帧） |
| 三大机制 | ① **Surprise-driven adaptive windowing**（何时写）② Priority-preserving consolidation（保留什么 = 遗忘）③ Query-conditioned graph reasoning（怎么读），固定容量隐记忆图 |
| 性能 | **StreamingBench 82.67、OVO-Bench 67.03、离线均值 74.4** |

**为什么这篇最危险**：它的三段式结构（何时写 / 保留什么 / 怎么读）和我们的
（KL 门控写入 / logvar 抗覆盖 / 确定性加权读出）**是一一对应的**。
区别只在：它的 surprise 是启发式的、记忆不训练；我们的 KL 有原理、端到端训。
**这正好是我们的 claim，但也意味着我们必须用实验证明"原理化"真的带来增益** —— 光讲故事不够了。

#### WeaveTime — Stream from Earlier Frames into Emergent Memory ✅

**链接**：https://arxiv.org/abs/2602.22142 | **CVPR 2026**
**作者**：Yulin Zhang, Cheng Shi, Sibei Yang

诊断出两个失败模式：**时序顺序歧义** + **过去/当前聚焦盲区**（Video-LLM 把视频当成无序证据袋）。
方案：轻量 Temporal Reconstruction 目标（少量微调，不需专门流式数据）+ **Past-Current Dynamic Focus Cache**，
**uncertainty 触发**的粗到细检索扩展 —— 只在不确定时才去翻历史。模型无关，不改架构。

> **这篇直接占了"不确定性触发"的坑，而且是 CVPR。必引，且必须说清我们和它的区别**：
> 它用不确定性决定**读**（何时检索历史），我们用 KL 决定**写**（何时更新记忆）。这个区分要写进 related work。

#### MM-Mem — From Verbatim to Gist ✅ **理论上最接近**

**链接**：https://arxiv.org/abs/2603.01455 | **ACL 2026 Main**（v1 03-02，v3 04-21）
**作者**：Niu Lian, Yuting Wang, Hanshu Yao, Jinpeng Wang, Bin Chen, Yaowei Wang, Min Zhang, Shu-Tao Xia

| 项目 | 内容 |
|------|------|
| 理论基础 | **Fuzzy-Trace Theory** 双痕迹（gist / verbatim） |
| 记忆层级 | Sensory Buffer → Episodic Stream → Symbolic Schema，逐级把细粒度感知痕迹蒸馏为高层语义图式 |
| **控制目标** | **Semantic Information Bottleneck**：平衡记忆压缩与任务相关信息保留 |
| 优化 | **SIB-GRPO**（RL），推理时熵驱动的自顶向下检索 |

> **这是我们理论上最近的邻居**：同样是"用一个信息论目标来控制记忆写入什么"。
> 差异必须讲清楚：**IB vs 自由能/ELBO**；**RL 优化 vs 端到端可微**；
> 它的记忆是符号/文本层级的，我们的是连续分布 slot。
> 另外它也是认知科学启发（Fuzzy-Trace vs 我们的 Friston 自由能原理）——
> **"认知科学启发的记忆"这个叙事已经拥挤，别把它当卖点。**

#### 其他

- **StreamOV** ✅ [2605.25621](https://arxiv.org/abs/2605.25621) — 证据引导的长短期记忆压缩音视频历史到固定预算 + 隐状态驱动的回应触发。**训练式**。附带 SOVBench。
- **Light-Omni** ✅ [2607.05511](https://arxiv.org/abs/2607.05511)（07-06，最新） — **双上下文状态**：有限尺寸全局多模态脚本（分层归并自 episodic memory）+ **参数化隐状态**驱动自主动作，绕开迭代推理。比 M3-Agent **+2.4%，快 12.1×，显存效率 2.6×**。
- **ROMA** ✅ [2601.10323](https://arxiv.org/abs/2601.10323) — 训练式实时全模态助手，"speak head" 解耦回应发起与生成。摘要未描述记忆机制，与我们相关性弱。

---

### 3C. Agentic / 图记忆（另一条路线）

| 论文 | 链接 | 要点 |
|------|------|------|
| **Homer** ✅ | [2607.02588](https://arxiv.org/abs/2607.02588) | 三级在线记忆（感知→复现实体→带时序/因果边的事件）+ 多轮 agentic 检索与验证。训练无关。**M3-Bench-robot +5.5、web +10.8、VideoMME-Long +4.4** |
| **MemDreamer** ✅ | [2606.07512](https://arxiv.org/abs/2606.07512) | 三层自顶向下图记忆增量构建 + Observation-Reason-Action 循环。**4 个 benchmark SOTA；仅用 2% 全上下文摄入即 +12.5 分；距人类专家 3.7 分** |
| **StreamMeCo** ✅ | [2604.09000](https://arxiv.org/abs/2604.09000) | 记忆图连通性：孤立节点 minmax 采样 + 连通节点边感知剪枝（**显式遗忘**）+ **时间衰减检索**。**70% 压缩 → 检索快 1.87×，精度反 +1.0%** |
| **EGAgent** ✅ | [2601.18157](https://arxiv.org/abs/2601.18157) | 实体场景图 + 规划 agent 结构化图搜索。面向智能眼镜的天/周级第一人称流。**EgoLifeQA 57.5%（SOTA）、VideoMME(Long) 74.1%** |
| **EventMemAgent** ✅ | [2602.15329](https://arxiv.org/abs/2602.15329) | 短期：事件边界检测 + **事件粒度蓄水池采样**；长期：逐事件归档。**Agentic RL 训练** |
| **PyraVid** ✅ | [2605.17065](https://arxiv.org/abs/2605.17065) | 基于 Event Segmentation Theory 的粗到细金字塔，结构引导扩展 + 剪枝 |
| **GOPAgen** ✅ | [2606.06532](https://arxiv.org/abs/2606.06532) | **编解码层级**：在 Group-of-Pictures 上做运动 agent，运动向量数据库。MotionBench、EgoSchema |

**评价**：这条路线走的是"记忆 = 可检索的结构化数据库 + agent 去查"，
和我们"记忆 = 连续隐分布"是**正交的哲学**。好消息是不直接冲突；
坏消息是它们数字很漂亮（MemDreamer 距人类 3.7 分），reviewer 会问为什么不走这条路。

---

## 第四类：Token 压缩（无显式记忆，关注效率）

| 论文 | 出处 | 核心 |
|------|------|------|
| **Video-XL** | CVPR 2025 | Visual Summarization Token（VST），16× 压缩，单 A100 千帧 |
| **DyCoke** | CVPR 2025 | 两阶段无训练：时间维合并相似 token + decode 阶段动态剪枝 |
| **HICom** | CVPR 2025 | 混合层级指令注入，语义/指令驱动压缩 |
| **MR.Video** | NeurIPS 2025 | MapReduce 框架，纯 API（Gemini+GPT-4o），LVBench +10% |
| **VideoChat-Flash** ✅ | **ICLR 2026** | [2501.00574](https://arxiv.org/abs/2501.00574)，OpenGVLab。**HiCo** 层级压缩 ~1/50、16 token/帧；短到长多阶段训练 + LongVid 数据集。**10k 帧 NIAH 99.1%，支持 3 小时视频**。训练式 |

> **Video-XL 线**：最新仍是 Video-XL-2（2506.19225, 2025-06），**未找到 Video-XL-3**。

---

## 综合对比表

| 论文 | 会议/年 | 底座 | 有记忆 | 需训练 | 核心方法 |
|------|--------|------|-------|-------|---------|
| LLaVA | NeurIPS'23 | LLaMA | ✗ | ✓ | 指令微调范式 |
| Video-LLaMA | EMNLP'23 | LLaMA-2 | ✗ | ✓ | Q-Former + 音频 |
| MC-ViT | ICML'24 | — | ✓ | △ | KMeans 记忆合并 |
| MA-LMM | CVPR'24 | Vicuna | ✓ | ✓ | FIFO Memory Bank |
| ∞-Video | ICML'25 | 任意 | ✓ | ✗ | 连续时间 LTM |
| ReWind | CVPR'25 | LLaMA-2 | ✓ | ✓ | 指令引导记忆 |
| Video-XL | CVPR'25 | InternVL系 | △ | ✓ | VST 压缩 |
| LongVPO | NeurIPS'25 | InternLM | ✗ | ✓ | DPO 训练策略 |
| MR.Video | NeurIPS'25 | Gemini/GPT | ✓ | ✗ | MapReduce |
| **MemStream** | arXiv'26.02 | Qwen2.5-VL | ✓ | ✗ | 稀疏KV+MoE检索 |
| **HERMES** | **ACL'26** | — | ✓ | ✗ | KV=层级记忆 |
| **VideoChat-Flash** | **ICLR'26** | Qwen2/2.5 | △ | ✓ | HiCo 层级压缩 |
| **WeaveTime** | **CVPR'26** | 任意 | ✓ | △微调 | **不确定性触发检索** |
| **FluxMem** | **CVPR'26** | — | ✓ | ✗ | 自适应层级压缩 |
| **FlexMem** | **CVPR'26** | LLaVA-Video | ✓ | ✗ | KV cache 即记忆 |
| **Vista** | **AAAI'26** | 任意 | ✓ | — | 场景压缩+CPU卸载 |
| **MM-Mem** | **ACL'26** | — | ✓ | ✓ RL | **语义信息瓶颈** |
| **SelectStream** | arXiv'26.06 | 冻结VLM | ✓ | ✗ | **惊讶度门控+图推理** |
| **CausalMem** | arXiv'26.06 | Qwen2.5-VL | ✓ | ✗ | 固定预算+SVD基底 |
| **Homer** | arXiv'26.07 | — | ✓ | ✗ | 三级图记忆+agent |
| **MemDreamer** | arXiv'26.06 | — | ✓ | — | 图记忆+ORA循环 |
| **Light-Omni** | arXiv'26.07 | — | ✓ | — | 双上下文+参数隐状态 |
| **我们** | 目标 NeurIPS'26 | InternVL3-8B？ | ✓ | ✓ **端到端** | **变分自由能门控** |

**一眼看出的问题**：最后一列往上扫，「需训练」这一栏在 2026 年几乎全是 ✗。我们是少数派。

---

## 主流 Benchmark 一览（2026-07 大幅更新）

### ⚠️ 首要发现：官方榜单全部停更

| 榜单 | 最后更新 | 状态 |
|------|---------|------|
| [Video-MME 官方](https://video-mme.github.io/home_page.html) | 2025-09-28 | **停更**，榜首 video-SALMONN 2+（76.4 长视频带字幕） |
| [LVBench 官方](https://lvbench.github.io/) | 2025-05-29 | **停更**，榜首 Deep Video Discovery 74.2% |
| [CG-Bench 官方](https://cg-bench.github.io/leaderboard/) | 无日期，2024 代模型 | **停更** |
| paperswithcode | — | **已关停**，302 跳转到 HuggingFace papers |
| EgoSchema | — | 无维护榜单，2026 年基本处于失管状态 |

**含义**：**不要再引用"当前 SOTA = XX"这种说法** —— 没有权威来源了。
可靠数字只能来自各家 tech report 和 2026 年论文自报。我们写论文时对比对象要自己复现或引原文。

### 可信数字（厂商技术报告）

[Qwen3.5-Omni Technical Report, arXiv:2604.15804](https://arxiv.org/html/2604.15804v1)（2026-04-17）Table 6：

| Benchmark | Omni-Flash | **Omni-Plus** |
|---|---|---|
| VideoMME (w/o sub) | 77.0 | **81.9** |
| MLVU (M-Avg) | 81.9 | **86.8** |
| LVBench | 65.7 | **71.2** |
| MVBench | 70.8 | **79.0** |

> 注意荒谬之处：Omni-Plus 的 LVBench 71.2 若提交能排官方榜第二 —— 而那个榜自 2025-05 就没更新过。

### 2026 新 benchmark（**对我们特别重要**）

| Benchmark | 链接 | 为什么重要 |
|-----------|------|-----------|
| **EGOSTREAM** ✅ | [2605.31557](https://arxiv.org/abs/2605.31557) | **流式情景记忆诊断基准**。2,250 问题 × 7 个认知维度（细节/空间/时序/事件/社交/因果/前瞻记忆）→ 8,528 次 recall-conditioned 评测。首创 **Answer Validity Window (AVW)**：区分"模型真忘了" vs "场景本身变了"。**最强方法天花板仅 ~45%** |
| **M³Eval** ✅ | [2606.05008](https://arxiv.org/abs/2606.05008) | 认知科学驱动的**记忆维度**评测：保持什么、保真度、抗干扰。发现：模型无法为并发视频流维持独立表征；干扰模式与人类不同；**空间记忆比时序记忆可靠**；符号记忆能力弱 |
| **RIVER-bench** ✅ | [2606.20726](https://arxiv.org/abs/2606.20726) | 按**到答案的时间距离**分层（~23s/44s/578s/2358s），15 档帧预算，~155k 预测 × 10 模型 |
| **Video-MME-v2** ✅ | [2604.05015](https://arxiv.org/abs/2604.05015) | 3,300 人工小时标注。指出"层级瓶颈"：视觉聚合与时序建模的错误会向上传播限制高层推理 |
| **EC-Bench** ✅ | [2603.29943](https://arxiv.org/abs/2603.29943) | 152 个 >30min 未剪辑视频。22 个 MLLM 最好仅 **29.98% 枚举 / 23.74% 计数**，人类 78.57%/82.97% |
| **LVOmniBench** ✅ | [2603.19217](https://huggingface.co/papers/2603.19217) | 长音视频，275 视频 10–90min |

> **建议换评测**：EGOSTREAM 和 M³Eval 是**为"记忆"这个 claim 量身定做的**，
> 比在 VideoMME 上刷 +1.2% 有说服力得多。VideoMME 已经饱和到 82%，
> 我们一个 8B 模型在上面很难讲出故事；而 EGOSTREAM 天花板 45%，
> **空间大、且直接测我们声称的能力**。这可能是本次调研最有行动价值的一条。

---

## 领域演进脉络（更新至 2026-07）

```
2023：建立范式
  LLaVA（NeurIPS'23 Oral）→ 三段式 VLM 成为标准

2023-2024：视频扩展
  Video-LLaMA / VideoChat2 / Video-ChatGPT → 长视频处理很差

2024：记忆机制兴起
  MC-ViT（ICML'24）→ 第一个系统性记忆合并（纯视觉）※ 此线已断
  MA-LMM（CVPR'24）→ 第一个完整 VLM 的 Memory Bank

2025：百花齐放
  ∞-Video（ICML'25）→ 无需训练的连续时间记忆
  ReWind（CVPR'25）→ 指令引导的可学习记忆
  DyCoke / HICom / Video-XL（CVPR'25）→ Token 压缩路线
  LongVPO / Eagle 2.5（NeurIPS'25）→ 训练策略 & 强底座路线
  ReKV（ICLR'25）→ KV-Cache 检索路线起点
  StreamingVLM（ICLR'26，2510 arXiv）→ 流式强基线

2026 上半年：三条线同时爆发 ← 本次新增
  ① KV-Cache 即记忆（主流，全部 training-free）
     ReKV → StreamKV → MemStream（2602）
     FluxMem / FlexMem（CVPR'26）、HERMES（ACL'26）、Vista（AAAI'26）
     CausalMem / SAVEMem / FreshMem / CurveStream（arXiv）
  ② 学习/隐状态记忆（人少，我们在这）
     WeaveTime（CVPR'26，不确定性触发读）
     MM-Mem（ACL'26，语义信息瓶颈 + RL）
     SelectStream（2606，惊讶度门控写）
     Light-Omni（2607，参数化隐状态）
  ③ Agentic / 图记忆（数字最漂亮）
     EGAgent / MemDreamer / Homer / StreamMeCo / EventMemAgent

  底座剧变：Qwen3.5（2602）原生多模态 + 256K 上下文
           → 「装不下」这个前提本身在松动
```

---

## 给自己的研究定位（**本次大幅改写**）

### 原来的定位站不住了

3 月版写的是：

> 「所有现有工作都在注意力机制的外部做文章…没有人直接改注意力机制的计算结构 —— 这是 LongLoRA 背景最直接的迁移点。」

这句话现在有两个问题：
1. **HERMES（ACL'26）做了机理性注意力分析**，把 KV cache 重新解释为层级记忆 —— 已经在往"注意力内部"走。
2. 更重要的是，**Qwen3.5 已经把注意力换成了 75% Gated DeltaNet 线性注意力 + 25% softmax 的混合架构**。
   底座自己在改注意力，而且改得比我们激进。「改注意力」不再是一个空白 niche。

### 现在真正的空白在哪

检索到 2026-07 为止，**没有任何工作用变分自由能 / ELBO 来门控视频记忆写入**。最接近的三个：

| 工作 | 它做的 | 与我们的差距 |
|------|--------|-------------|
| SelectStream（2606） | surprise-driven 决定何时写 | **启发式** surprise，无概率语义；记忆不训练 |
| WeaveTime（CVPR'26） | uncertainty 触发何时**读** | 门控的是读不是写；uncertainty 是启发式 |
| MM-Mem（ACL'26） | **语义信息瓶颈**控制记忆构建 | IB 而非自由能；RL 优化而非端到端可微；符号记忆而非连续分布 |

**所以我们的 claim 必须收缩到这个精确表述**：

> 不是"我们首次用不确定性门控记忆"（**假的，被 SelectStream / WeaveTime 占了**），
> 而是"我们首次把记忆写入表述为**变分推断问题**，使得门控信号 KL 是从一个
> **有原理的概率目标中导出**、而非手工设计，且记忆状态是**分布值的、端到端可微的**"。

### 三个必须准备好的 reviewer 问题

**Q1：2026 年大家都 training-free 了，你为什么要训练？**
→ 最好的弹药是 **RIVER-bench（2606.20726）** 的标度律：
StreamingVLM 的预算指数 α(1000s)=1.26 vs Qwen3-VL 基座 0.17（**7.4×**）；
1000 秒距离下 10× 预算给 StreamingVLM 带来 +29 分，给基座只有 +4 分。
其结论是：**记忆响应度来自"记忆专用训练"，而非模型容量**。这是支持我们训练路线的最强公开证据。
⚠️ 但注意：**该文单作者（Yixian Tian）、无机构、自报 R² 0.05–0.75**，质量未经同行评议，引用时要小心。

**Q2：底座 Qwen3.5 原生 256K，能直接装下 400 秒 720p，还要记忆模块干嘛？**
→ 这是**最致命的问题**，目前我们没有好答案。可能的方向：
装得下 ≠ 用得好（EC-Bench：>30min 视频最强模型枚举仅 29.98% vs 人类 78.57%），
以及 O(n²) 成本 —— 但 Qwen3.5 用的是线性注意力，这个成本论证也在削弱。**需要专门想。**

**Q3：MA-LMM 原班人马（Bo He）在 FluxMem 里都放弃训练记忆改做 training-free 压缩了，你凭什么？**
→ 需要正面回答，不能回避。

### ⚠️ 底座决策：InternVL3-8B 建议重新考虑

核实结果（[InternVL 官方 repo](https://github.com/OpenGVLab/InternVL)）：

- **InternVL3.5（2025-08-26）是最新版，不存在 InternVL4。**
- **该 repo 的 news 里 2026 年零条目** —— 最后更新是 2025-08-30。OpenGVLab 2026 年唯一动作是 InternVL-U（4B 统一理解+生成），是支线不是底座继任。
- **InternVL3.5 上下文 32K；Qwen3.5 / Qwen3-VL 是 256K（可扩 1M）** —— 差 8 倍。

我们的 `memory_module.py` 现在硬编码 `d_model=4096` 对齐 InternVL3-8B。
**如果继续用 InternVL3-8B，NeurIPS'26 审稿人大概率会说底座过时**（2025-04 发布，届时超一年，且原作者组已停更）。
2026 年的竞品几乎全在 Qwen2.5-VL / Qwen3-VL 上做。

**建议**：认真评估切到 Qwen3-VL-8B（256K，2025-11）。
`d_model` 是构造参数不是硬编码常量，切换成本主要在集成点而非模块本身。
但这是你的决定 —— 如果有算力/数据/熟悉度上的理由坚持 InternVL3，也合理，只是要在论文里主动解释。

### 优先精读列表（更新）

1. **SelectStream**（2606.16353）—— **最像我们的工作**，三段式结构一一对应，必须逐段对比
2. **MM-Mem**（ACL'26, 2603.01455）—— 理论上最近的邻居（信息论目标控记忆），必须讲清 IB vs 自由能
3. **WeaveTime**（CVPR'26, 2602.22142）—— 占了"不确定性触发"的坑，且是 CVPR，必引
4. **FluxMem**（CVPR'26, 2603.02096）—— MA-LMM 血脉，且是 CVPR 正面竞品
5. **EGOSTREAM**（2605.31557）+ **M³Eval**（2606.05008）—— 考虑作为主评测
6. **RIVER-bench**（2606.20726）—— 支持训练路线的弹药（但注意质量存疑）
7. **MemStream**（2602.18434）—— 注意它发现的 query-frame 相似度时间偏置，可能影响我们的 `get_prior`

### 算力参考（8× H200）

| 实验类型 | 时间估计 |
|---------|---------|
| Qwen2.5-VL-7B 全量微调 | 1-2 天/轮 |
| InternVL3-8B LoRA 微调 | < 1 天/轮 |
| 全部 benchmark 评测 | 0.5-1 天 |
| 注意力机制改造 + 微调 | 2-3 天/轮 |

---

## 未核实 / 待查清单

以下在搜索结果中出现但**未 fetch 核实**，引用前必须自查（搜索引擎可能编造标题/ID）：

- HPP (2606.21734)、Q-Fold (2606.12125)、DynaTok (2605.19322)
- "Watch, Remember, Reason" (2606.07433)、"Think-as-You-See" (2603.02872)
- "Video Streaming Thinking" (2603.12262)、"Thinking in Streaming Video" (2603.12938)
- WAT (2603.13412)、QueryStream (ICLR'26, OpenReview 738HjJEbml)
- **"VideoARM" 与 "SlotMemory: Object-centric KV Memory (2605.31033)"** —— 高度怀疑是搜索引擎幻觉，**不要引用**
- **ExtremeWhenBench** —— 被 RIVER-bench 引用但搜不到，存疑
- 各家自报的 Video-MME 分数（Seed 2.1 Pro 0.892 等）—— 仅聚合站数据，**源头 blog 无数字**，且字幕条件不明

**有用的活跃索引**（已确认存在）：https://github.com/sotayang/Awesome-Streaming-Video-Understanding

---

*3 月版基于三大顶会整理 | 2026-07-16 更新：补 CVPR/ICLR/ICML/ACL/AAAI 2026 + arXiv 2602–2607*
*✅ = 已逐条 fetch arXiv 页面核实标题/作者/日期/venue/摘要 | 机构信息多为按作者名推断，未核实*
