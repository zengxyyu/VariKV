# 方向定位：变分/分布式记忆 × KV Cache 压缩

> 整理时间：2026-07-23（2026-07-24 定案更新） | 面向 ICLR 2027 投稿（正文截稿约 2026-09-24）
> 目标：把「自由能分布式记忆」的思想引入 KV cache 压缩这个现代瓶颈
> 核心结论：**方向在顶会里空着，但四面紧贴；新颖性全押在「分布式存储(μ,σ²) + 自由能门控写入」这个配对上。**

## ✅ 定案（2026-07-24）
- **走方案 B**（分布式记忆吸收被驱逐 KV），投 **ICLR 2027**（约 10–20% 中稿概率，见 §7）。
- **代码基座 = Fast KVzip**（github.com/Janghyun1230/FastKVzip，已验证可复现）——提供冻结 LLM + 驱逐管线 + 评测。你把分布式记忆接上去吸收它驱逐的 KV。
- **两个最直接先例 IndexMem(2605.25475) / Tensor Cache(2605.22884) 均无公开代码**（2026-07-24 核实）→ 只能当**对标论文**，不能 fork。
- 原始视频版代码已备份为 `memory_module_video_backup.py`；在 `memory_module.py` 上移植到 KV。
- 第一步执行清单见 §8。

**可信度**：本文件竞品/venue 均经多路 fetch 核实（标注 [VF]=亲自抓取原文，[SLO]=仅搜索列表所见）。覆盖边界见文末——**网页搜索不索引顶会全文，"没搜到"≠"不存在"；临投稿写 related work 前需再做一次全文级普查。**

---

## 0. 一句话定位（论文可直接用）

> 变分分布式记忆（Kanerva Machine 1804.01756、Variational Memory Addressing NIPS'17）理论上更优，但**从未被用于 KV/长上下文压缩这个现代瓶颈**；而现代 KV 压缩（~55 篇顶会工作）**无一例外退回了确定性点记忆**。我们把前者的原理带进后者的战场，并证明**不确定性在这里是功能性的**。

这个叙事的价值：**有理论血统撑腰（不显空想）+ 坑确实空（有新意）**。避免写"我首创变分记忆"——2017/2020/Kanerva 会打脸。

---

## 1. 两条候选路径（A / B，都未定，各存档）

### 路径 A：自由能重要性打分（改造 Fast KVzip 的 gate）
- **做法**：fork Fast KVzip，把它 BCE 蒸馏出的**点估计 gate**，换成预测「自由能(重建+KL) + 不确定性」的 gate。重要性高、或不确定性高的 KV 保留。
- **优点**：最贴可复现地基（Fast KVzip repo 已验证可复现，见 `kv_cache_survey.md` 判定），代码复用多，9 周最现实。
- **缺点/被占**：差异只在「打分目标」，审稿人易贬为"又一个 gate"。CapKV（信息论打分）、RetentiveKV（熵）已占相邻地。
- **判定**：**部分被占**。空着的是「自由能(ELBO=KL+重建)导出打分 + 不确定性分布、纯文本 KV eviction 的训练式 gate」。

### 路径 B：分布式记忆吸收被驱逐的 KV（复用 `memory_module.py`）⭐ 助手倾向
- **做法**：最近的 KV 保持精确；**被驱逐的 KV 不丢，写进一组分布式 slot (μ,σ²)，写入规则用 KL 门控自由能**（确定的槽=低方差=抗覆盖）；用注意力从 slot 读回。
- **优点**：
  1. 差异是**架构级**（存储形态：分布 vs 点；写入：自由能 vs 启发式），比 A 更硬。
  2. **几乎原样复用你的 `memory_module.py`**——输入从"视频段"换成"被驱逐的 KV"，整套机器搬过来。9 周巨大优势。
  3. RetentiveKV 号称 uncertainty-aware，但它的不确定性是**选哪个 token 扔的熵**，**不存方差**——"它离你只差一个设计决定：存方差而不是只用熵"。正好衬托你。
- **缺点/被占**："吸收被驱逐 KV 进有界记忆"被 Infini-attention / IndexMem / Tensor Cache 占了（但全是点记忆）。
- **判定**：**空，但四面紧贴**。空着的是「分布式 slot(μ,σ²) + 自由能门控写入」这个**配对**——每个单独零件都被占，novelty 全在配对上。

### A vs B 快速对比
| 维度 | A（自由能 gate） | B（分布式记忆吸收 KV） |
|------|------------------|------------------------|
| 新意强度 | 目标函数级（弱） | 架构级（强） |
| 复用你的代码 | 少（借 Fast KVzip） | **多（几乎原样搬 memory_module.py）** |
| 地基可复现性 | **高（Fast KVzip 已验证）** | 中（需自己搭吸收/读回管线） |
| 最强对标 | CapKV、RetentiveKV、Fast KVzip | Infini-attention、IndexMem、Tensor Cache |
| 9 周可行性 | 高 | 中-高 |

---

## 1.5 方案 B 的范式：训练时 / 推理时 + 效率来源（2026-07-23 补）

### 与 HiCI 的根本范式差异
- **HiCI = 训练期方法**：推理时回到 full attention，推理过程不变 → 提升的是**质量/能力**，不涉及推理加速。
- **方案 B = 推理期方法**：记忆模块在推理时**持续运行、改变推理计算** → 目的是**推理效率**。
- 做 KV cache（为效率）必须从"训练期方法"转到"推理期方法"，这是真正的范式转变。

### 训练时
冻结 LLM；喂长文本；模拟"驱逐"——旧 KV 被赶出精确缓存 → 记忆模块压进分布式 slot(μ,σ²)，自由能门控写入；LLM 在「最近精确 KV + slot 读出 token」下预测下一词；`total = lm_loss + λ·free_energy`；反向只更新**记忆模块**，LLM 不动。要训练的：`compress`/`enc_mu`/`enc_logvar`/`decoder`/`slot_to_tokens`/`mem_*_init`。

### 推理时
最近一小窗 KV 精确保留；超窗的旧 KV **不丢，被吸收进固定大小 slot**；每个新 token，LLM 注意力扫「最近窗口精确 KV + K 个 slot 读出 token」——**大小固定，不随上下文膨胀**。

### 效率来源分析（审稿人第一问：省了什么、代价是什么）
- 普通长上下文：缓存 O(n)、每 token 注意力 O(n)、显存 O(n)。
- 方案 B：窗口 W + K slot 全固定 → 每 token O(W+K·T)=**常数**，显存**常数**。
- **加速真实，但是"有界记忆"式加速**：长上下文/流式下 O(1) 完胜 O(n)；**短上下文下模块固定开销可能反而拖慢** → 主场是长/流式。
- **两条腿分工**：「有界」给效率（与 Infini-attention 共享）；「分布式(μ,σ²)」补回压缩丢的质量（你的 novelty）。→ 这正是**方差消融**要证明的：分布式真补回了质量，否则只是更慢版 Infini-attention。
- **硬约束**：模块必须**轻**，否则固定开销吃掉有界记忆省下的钱，加速故事塌。

### 范式先例（成立、有据）
IndexMem（ICML'26，最直接：训 indexer + 压被驱逐 KV 进 latent 记忆，backbone 冻结）、Tensor Cache（MIT，被驱逐 KV→固定联想记忆，训门控）、Infini-attention（同范式但连 LLM 一起继续预训练，更重）、Larimar（训记忆模块+冻 LLM，但用于知识编辑）、更早血统 Gist/AutoCompressor/ICAE/500xCompressor/Cartridges（训模块把上下文压成少量 token，2023 起）。→ 范式已验证，方案 B = 把它们的点值记忆换成分布式+自由能。

---

## 2. 竞品与祖先地图（三层，related work 骨架）

### ① 理论祖先（搜 KV 永远漏，但审稿人认，**必引**）
| 论文 | venue | 关系 |
|------|-------|------|
| **Kanerva Machine** | ICLR 2018 (1804.01756) | 变分分布式记忆直系祖先；VAE+记忆，贝叶斯在线压缩更新。**突出引用** |
| **Variational Memory Addressing** | NIPS 2017 | 对外部记忆做**摊销变分推断**，混合/多峰读取。你 amortized 后验的源头 |
| **Learning to Learn Variational Semantic Memory** | NeurIPS 2020 | 记忆存分布 + 层次贝叶斯变分召回 |
| **Compressive Transformer** | ICLR 2020 | 有界压缩记忆吸收被驱逐激活（点值）。路径 B 祖先 |
| **Kanerva++** | ICLR 2021 | 可微分块分配潜记忆 |

### ② 最危险的近邻（**必引必区分**）
| 论文 | venue | 差异点（你的话术） |
|------|-------|-------------------|
| **Titans** | NeurIPS 2025 poster | surprise 门控写入最像你；但 surprise=梯度范数、记忆点值、无 ELBO |
| **Larimar** | ICML 2024 | Kanerva 记忆挂 LLM，但**故意去掉分布式**改确定性最小二乘。你正好保留 → 最佳对照 |
| **B'MOJO** | NeurIPS 2024 | eidetic+fading 有界记忆吸收过去状态；确定性，无 μσ² |
| **IndexMem** | ICML 2026 poster | 学习式 latent 记忆吸收被驱逐 KV + 残差回注；点值，用注意力蒸馏 KL（非 ELBO） |
| **Tensor Cache** | arXiv 2605.22884 (MIT) | 被驱逐 KV → 固定大小联想记忆矩阵，学习门控端到端；点值 |
| **InfiniPot-V** | NeurIPS 2025 poster | 流式**视频**有界记忆硬上限；与你视频线也相关；确定性 buffer |
| **RetentiveKV** | arXiv 2605.04075 | 多模态 uncertainty-aware，但不确定性=选择用的熵、**不存方差**；非 ELBO |
| **CapKV** | arXiv 2604.25975 | 信息瓶颈/互信息闭式打分；线性高斯代理；点估计、training-free。理论最近邻 |
| **Dense AM for Gaussians** | ICLR 2026 (2509.23162) | slot 真存高斯 + Wasserstein 检索；但无自由能门控、非 KV |

### ③ 顶会 KV 全景（~55 篇逐条核实的结论）
**NeurIPS/ICML/ICLR 2024–2026 的 KV 工作，无一做变分/分布式/自由能记忆。** 全是确定性点估计（注意力质量/冗余/投影/CUR/扰动界/RL/学习预测器）、低秩SVD、或量化。沾边的仅：CommVQ（EM 拟合码本）、SparseAR（注意力熵门控）、TurboQuant（率失真界）。唯一真分布式的 Expected Attention 是 **arXiv-only、未进顶会**。完整清单见 `kv_cache_survey.md`。

---

## 2.7 驱逐方法全景 + 三占位定位 + 决策A/B统一（2026-07-30）

### KV 压缩有两个可分开的决策
- **决策 A：哪些 KV 该驱逐**（打分/选择）—— KVzip/FastKVzip 的战场。
- **决策 B：被驱逐的 KV 怎么处理**（丢弃 vs 吸收）—— 你的战场。

### 决策 A 的六类打分信号（全核实，method 用）
| 类别 | 代表 | 信号 | query 相关 |
|---|---|---|---|
| 位置/新近 | StreamingLLM (2309.17453) | 留 sink+最近窗口，中间全扔 | 无关 |
| 注意力分数 | H2O(2306.14048)/SnapKV(2404.14469)/TOVA(2401.06104)/Keyformer(2403.09054) | 收到多少注意力 | 相关 |
| 注意力×值范数 | VATP (2406.12335) | 注意力 × value 范数 | 混合 |
| 范数 | L2-norm (2406.11430) | key 的 L2 范数小→留 | 无关 |
| 重建 | KVzip (2505.23416) | 重建上下文需不需要它 | 无关 |
| 预期注意力 | Expected Attention (2510.00636) | 建模未来 query，闭式算预期注意力 | 无关(估计) |

预算分配元策略：PyramidKV(层间)、Ada-KV(头间)、IndexMem(学习式)。实用点：H2O/SnapKV/TOVA 需注意力矩阵，**FlashAttention 不产生它** → L2/Expected/KVzip 才转向不需注意力矩阵的信号（你用 FlashAttention 要注意）。

### 关键发现：吸收型方法的驱逐几乎没打分
核实结论：**Infini-attention = 纯新近、全部压进记忆、无打分；Tensor Cache = 纯 FIFO、无打分**（学习只在"怎么写"）；**IndexMem = 唯一例外，学习式 indexer 打分**。→ "用学习式有原理信号决定哪些降级进记忆"这块在吸收型方法里**几乎是空的**。

### 你的三占位定位（没人做全）
| | 吸收 | 分布式(μσ²) | 打分驱逐 |
|---|---|---|---|
| Infini-attn/Tensor Cache | ✓ | ✗ 点 | ✗ FIFO |
| IndexMem | ✓ | ✗ 点 | ✓ 确定性 indexer |
| **你** | ✓ | ✓ | ✓ **自由能统一** |

### 已选定：选项 3 —— 决策 A/B 用同一自由能信号统一
- 一个自由能标量 $F_i$ 同时决定"驱逐哪个"(A) 和"写入多少"(B)。
- 比 Infini/Tensor Cache 多"有原理的降级选择"，比 IndexMem 多"分布式 + A/B 统一"。
- **完整方法设计见 `theory_distributional_memory.md` §11**（$F_i=D_i+λ\text{KL}_i$、退化特例表、摊销 $F$ 预测器、分阶段去风险、四档杀手对比）。
- **不要去卷决策 A 的打分精度**（H2O/SnapKV/KVzip 的拥挤战场）——把驱逐纳入自由能框架，占那块空地。

---

## 3. 生死实验（A/B 通用，决定成败）

三路核查都独立指向**同一个致命风险**，必须正面解决：

> 审稿人必问：你那套 (μ,σ²)+自由能机器，凭什么比"点记忆+学习标量门控"（IndexMem/Tensor Cache）更好？否则就是"给 IndexMem 刷了层贝叶斯油漆"。

**⚠️ 这与视频模块里未解决的理论风险同源**（`get_prior` 把混合塌成单高斯，见 `theory_distributional_memory.md`）——分布式/多峰必须是**功能性的**、能在实验证明更好。

**两个生死实验：**
1. **方差消融**：把 (μ,σ²) 退化成点记忆（去方差、门控换学习标量），看性能掉不掉。掉→你赢；不掉→贝叶斯油漆。**在带干扰/长程保持场景测最有区分度。**
2. **头对头**：同设置，你的自由能门控 vs Tensor Cache/IndexMem 的学习标量门控。

---

## 4. 一个必须清醒接受的前提

**"把你的 idea 和 KV 结合" = 自动离开 training-free。** 因为"识别网络 + 变分"本身就意味着要训练。KVzip 那种纯 training-free 方法没有地方放你的变分机器。**Fast KVzip 是关键的桥，正因为它是 KVzip 家族第一个"训练一个东西(gate)"的工作。** 你若之前坚持 training-free，走这条路需明确知道自己在改选择（能训练的模块往往更有分量，不一定是坏事）。

---

## 5. 覆盖边界（诚实，勿当"已查全"）
1. 网页搜索**不索引顶会全文**；"没搜到"≠"不存在"。穷尽需 grep `papers.nips.cc`/`proceedings.mlr.press` 全文。
2. **OpenReview 全程被反爬墙挡**，靠 proceedings 镜像/虚拟站绕过；少数条目靠 arXiv 声明 venue。
3. **NeurIPS 2026 未出结果**；未来日期 arXiv ID 全部排除（宁漏勿误）。
4. 仅覆盖**主会**，workshop / ACL / CVPR 等未纳入。
5. 这条 lane **每月在动**——临投稿前必须再扫一次，并设 arXiv 提醒（"KV cache" + "variational"/"free energy"/"Bayesian"）。

---

## 7. 中稿概率（ICLR 2027，诚实）
- **约 10–20%**（单人、早期、~2 个月、拥挤方向）。主要风险：执行时间 + 二元生死门（方差消融）。
- 加分：有真理论（KV 论文罕见）；方向已验证空着。天花板 = poster（KV oral 极罕见，全扫仅 2 篇）。
- 二元生死门（方差消融）通过率估 ~40–60%，真不确定。**这一门不过 → 拒。所以先跑它。**
- 改投 ICML 2027(1月)/NeurIPS 2027(5月) 可把概率约翻倍——但用户已定 ICLR 2027。

---

## 8. 第一步执行清单（方案 B，按顺序）

### 阶段 0：环境 + 复现地基（第 1 周，目标：跑通、拿到可信 baseline 数字）
- [ ] **配环境**：CUDA 12.8、`torch==2.7.0`(cu128)、`flash-attn==2.7.3`。单张 H200 起步。
- [ ] **clone Fast KVzip**：github.com/Janghyun1230/FastKVzip；让 `Jang-Hyun/Fast-KVzip` 的 gate 自动下载（无需自训）。
- [ ] **零训练复现招牌数字**：`qwen3-8b` 上跑 `prefill/eval_chunk.py -d all` + `results.parse`；数学任务 `math/run_math.py` aime24。**数字对得上论文 → 地基可信。**
- [ ] 读懂它代码里"驱逐发生在哪一步"（哪个文件/函数把旧 KV 扔掉）——这是你要插入记忆的锚点。
- [ ] 邮件问作者**代码 license**（repo 无 LICENSE 文件，发论文前要清）。

### 阶段 1：生死实验——方差消融（第 2–3 周，决定要不要继续）⛔ GO/NO-GO
目的：在**最小设置**下验证"分布式(存方差)比点记忆(不存方差)在同预算下更好"。不必先建完整系统。
- [ ] 设计一个**受控长上下文召回任务**（带干扰的 needle/多 key 检索——分布式该赢的场景）。
- [ ] 造两个版本对比：
  - **点版**：被驱逐 KV 压进一个记忆，但**去掉方差**、更新率固定/学习标量（≈IndexMem 缩影）。
  - **分布版**：同结构，但 slot 存 (μ,σ²)、方差控抗覆盖 + 精度加权。
- [ ] 同预算(同 K)下比召回精度。**分布版明显更好 → GO；不掉/持平 → NO-GO，回头想或换角度。**
- [ ] 这一步**便宜**（不用大规模训练、不用全 benchmark），1–2 周能出信号。**是全项目最高杠杆的一步。**

### 阶段 2：搭方案 B 完整版（第 4–7 周，仅在阶段 1 GO 后）
- [ ] 把 `memory_module.py`（已备份视频版）移植成"吸收被驱逐 KV"模块，接进 Fast KVzip。
- [ ] **移植时补齐四个理论缺口**（见 `theory_distributional_memory.md` §9.7）：
  1. `get_prior` 保留混合（逐 slot KL，勿平均）
  2. `decoder` 非线性 或 靠混合非共轭
  3. 重建似然向注意力输出空间靠拢
  4. `read` 让方差进入读出（采样 或 方差感知 token）
- [ ] loss：`lm_loss + λ·free_energy`，冻结 LLM，只训记忆模块。
- [ ] 保持模块**轻量**（否则短上下文被固定开销拖慢，加速故事塌）。

### 阶段 3：对比实验 + 写作（第 8–9 周）
- [ ] 主对比：同预算/Pareto 前沿打 Fast KVzip(丢弃) / IndexMem类(点记忆吸收) baseline。
- [ ] benchmark：RULER、LongBench、SCBench（Fast KVzip 已带脚本）。
- [ ] 效率报告：延迟、峰值显存 vs 上下文长度（证明 O(1) vs O(n)）。
- [ ] 理论章节：用 `theory_distributional_memory.md` §9 的率失真+自由能推导。
- [ ] 投稿前重扫竞品（这条 lane 每月在动）。

### 贯穿始终的两条红线
1. **方差必须是功能性的**（阶段 1 就是在赌这个）——否则"贝叶斯油漆"。
2. **模块必须轻**——否则加速故事不成立。

---

*生成：2026-07-23（2026-07-24 定案+执行清单更新）| 基于三路顶会系统检索 + 多路 arXiv 深挖，均 fetch 核实；配套：`kv_cache_survey.md`（KV 综述+Fast KVzip 复现判定）、`theory_distributional_memory.md`（理论 + §9 KV 实例化严格推演）、`memory_module.py`（核心模块，视频版备份于 `memory_module_video_backup.py`）*
