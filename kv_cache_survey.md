# Training-Free KV Cache 压缩综述

> 整理时间：2026-07-22 | 面向 ICLR 2027 投稿方向调研（training-free KV，纯文本 LLM）
> 范围：不训练、纯推理时压缩 KV 缓存的方法。数据基于逐会核实的 17 篇顶会论文 + 前沿扫描。

---

## 0. 领域全景：所有工作在解决同一个问题

**KV cache 随序列变长而线性膨胀 → 显存扛不住、解码变慢。**
training-free 这一支：**不训练、纯推理时，把缓存压小，同时尽量不掉精度。**

所有方法的差别只在「用什么手段压、按什么标准压」。按手段分 **5 大家族** + 2 个元层面。

**可信度**：本文件所列顶会 oral/poster 状态均在前序对话中逐会核实（官方 /oral/ 或 /poster/ 页 + OpenReview）；具体机制描述为摘要级，深读以原文为准。

---

## 家族一：Eviction（驱逐）—— 扔掉不重要的 token

**核心问题：哪些 token 可以扔？** 最大、最卷的一类。

| 方法 | 出处 | 做法 |
|------|------|------|
| StreamingLLM / H2O / SnapKV / TOVA | 奠基（2023-24） | 按注意力分数扔；留"注意力沉降"（开头 token）+ 最近窗口。**通用打分现已做死** |
| **KVzip** ⭐ | **NeurIPS'25 ORAL** | **query 无关**：按"该 KV 能否重建原始上下文"打分 → 一次压缩多 query 复用。arXiv 2505.23416 |
| Ada-KV | NeurIPS'25 poster | 第一个**按注意力头**自适应分配预算，带理论误差界。arXiv 2407.11550 |
| ManifoldKV | ICML'26 poster | **不用注意力矩阵**：按 token 到 key 质心的欧氏距离（角度+半径）判离群 |
| ChunkKV | NeurIPS'25 poster | 不扔孤立 token，而是**整块语义 chunk** 一起留/扔。arXiv 2502.00299 |
| R-KV | NeurIPS'25 poster | 推理模型：（重要性 − 冗余）联合打分，惩罚近重复 key |
| RPC | NeurIPS'25 poster | 推理路径的语义稀疏性，周期性压缩 |
| RocketKV | ICML'25 poster | 两阶段：粗粒度永久驱逐 + 细粒度 top-k。NVlabs |

---

## 家族二：Quantization（量化）—— 不扔，但每个数存得更省

**核心问题：32 位存不下，能否压成 4/2/1.5 位？** 保留所有 token，降数值精度。偏信号处理/系统，数学味重。

| 方法 | 出处 | 做法 |
|------|------|------|
| **TurboQuant** | **ICLR'26 poster**（注：非 oral，Google/DeepMind+NYU） | 在线向量量化，MSE 量化器 + 1-bit QJL 残差校正，证明**逼近信息论失真下界**，~3.5 bit 中性。arXiv 2504.19874 |
| KIVI / KVQuant | 奠基 | 2-bit KV 量化 |
| PolarQuant | **AISTATS'26**（非三大会） | 极坐标量化 |

> ⚠️ **量化路线空间将尽**：TurboQuant 已"逼近香农下界"，可压余量很小。

---

## 家族三：低秩 / 跨层共享（"分层"的一种）⭐ 相对有空间

**核心洞察：不同层之间的 KV 高度冗余，不用每层都存完整一份。**

| 方法 | 出处 | 做法 |
|------|------|------|
| **xKV** | ICML'26 poster | 发现**相邻层的 KV 主奇异向量对齐** → 一组层合并成一个**共享低秩子空间**。arXiv 2503.18893 |
| **KVTC**（KV Transform Coding） | ICLR'26 poster（NVIDIA） | 借媒体压缩：PCA 去相关 + 自适应量化 + 熵编码，20–40×。arXiv 2511.01815 |
| STAR-KV | ICML'26（press 称 spotlight，未官方证实） | 可微软阈值，按头/块自适应选秩 + 低秩感知量化。arXiv 2606.08382 |
| SALS | NeurIPS'25 poster | 低秩隐空间里做稀疏 token 选择（RoPE-free QK 匹配）。arXiv 2510.24273 |

> **"分层"有两种，别混淆**：
> - **跨层共享（cross-layer）**：上表这类，压层与层之间的冗余。
> - **分层记忆（hierarchical/temporal tiers）**：按新/旧、重要/次要把缓存分桶管理（如 Cambrian-S 全局/滑窗/情景三层、HERMES 层级记忆）——更偏"记忆结构"。

---

## 家族四：Merging（合并）—— 把多个 token 融成一个

不是非黑即白扔/留，而是合并相似 KV，信息不完全丢。

| 方法 | 出处 | 做法 |
|------|------|------|
| **Fast KV Compaction** | ICML'26 poster（MIT / Yoon Kim） | **闭式**构造紧凑 KV，逐头复现原注意力输出，几分钟、不训练、100×。arXiv 2602.16284 |
| Token merging 类 | 各处 | 相邻相似 token 合并 |

---

## 家族五：新场景 / 新设定 ⭐ —— 换战场，而非新手段

近年真正增长点：把 KV 压缩搬到新场景，老方法假设失效。

| 场景 | 代表 | 状态 |
|------|------|------|
| **推理模型 KV**（长输出而非长输入） | **ThinKV** ⭐（ICLR'26 ORAL）、R-KV、RPC、ForesightKV、ReasonAlloc | **9 个月从新鲜到饱和** |
| **混合/线性注意力 + MoE 模型 KV** | MoE-nD、PiKV、HySparse、NLL-guided SWA | **最空 ← 最佳空位** |
| 对话 / agentic | EpiCache（长对话情景）、CodeComp（代码） | 中等 |
| 多模态 KV | RetentiveKV | 中等 |

**ThinKV**（ICLR'26 ORAL，arXiv 2510.01290）：把 CoT 分解成"思维类型"，按思维重要性分配**逐 token 精度（量化）+ 渐进驱逐** → <5% 缓存近无损。**是"量化+驱逐"混合 + 推理场景的结合。**

---

## 元层面：不压缩，但标志领域成熟

- **批判/打脸**：《The Pitfalls of KV Cache Compression》（ACL'26 Long，arXiv 2510.00231）—— 压缩会悄悄**丢指令、泄露系统提示**；另有安全分析（压缩削弱越狱防御）。
- **综述 + 榜单**：ACL'26 系统综述；**KVPress** 榜单收 20+ 方法成标准基线 → 新方法要跟 20+ 个比。

---

## 总结表 + 拥挤度

| 家族 | 在做什么 | 代表（顶会） | 拥挤度 |
|------|---------|-------------|--------|
| Eviction 驱逐 | 扔不重要 token | KVzip⭐oral、Ada-KV、ManifoldKV、ChunkKV | 极挤（通用打分已死） |
| Quantization 量化 | 降数值精度 | TurboQuant、ThinKV⭐oral | 挤，逼近理论极限 |
| **低秩/跨层** | 层间冗余压缩 | xKV、KVTC、STAR-KV、SALS | **中等，相对有空间** |
| Merging 合并 | 相似 token 融合 | Fast KV Compaction | 中等 |
| **新场景** | 换战场（推理/MoE/对话） | ThinKV、MoE-nD、EpiCache | **混合/MoE 最空** |
| 元层面 | 批判/综述/榜单 | Pitfalls(ACL)、KVPress | 饱和信号 |

---

## 顶会 oral 情况（决定"有没有分量"）

**五个顶会（NeurIPS'25、ICML'25、ICLR'26、ICML'26、ACL'26）training-free KV 共 17 篇，oral 仅 2 篇**：
- **KVzip**（NeurIPS'25）—— 重建式驱逐
- **ThinKV**（ICLR'26）—— 思维自适应量化+驱逐

**含义**：两篇 oral 都给了"重要性打分/驱逐"这一类；量化、低秩全卡 poster。**training-free KV 想拿 oral 近乎不可能，是 poster 级方向。** 若目标"稳中一篇"则可行。

---

## 给自己的方向判断

- ❌ 再做"更聪明的 eviction 打分器"刷 LongBench/RULER = 死路（要跟 KVPress 榜 20+ 方法比）。
- ✅ **最佳空位：混合/线性注意力 + MoE 模型的 KV 压缩/预算分配**（Qwen3.5、DeepSeek-V4 类）——新架构打破 dense 注意力假设，只有三四篇，vLLM 有真实部署需求。
- ✅ 次优：以"保住某能力"（多指令/ICL/安全）为主贡献，蹭 ACL'26 批判那波。

---

*生成：2026-07-22 | oral/poster 状态经逐会核实；机制为摘要级，深读以原文为准*
