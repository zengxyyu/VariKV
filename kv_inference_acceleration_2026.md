# KV Cache 推理加速：2026 顶会全景与我们的定位

> 整理时间：**2026-08-11** ｜ 面向 ICLR 2027 投稿的竞品与定位调研
> 范围：**不限 training-free**。已有的 `kv_cache_survey.md`（2026-07-22）只覆盖免训练方法，
> 而本项目真正的竞争在**训练型紧凑表示**那一支，那份文件看不到它们——这是新写一份的原因。
> 两份文件互补：家族划分沿用旧文件，本文件补齐训练型、2026 新会期、开源状态与逐维度差异。

## 核实级别标注（重要）

本文件每条结论都带来源级别，**不要把不同级别的信息当同一回事使用**：

| 标记 | 含义 |
|---|---|
| ✔ | 2026-08-11 当天读过论文正文 / 官方会议页 / 代码仓库后写下 |
| ○ | 本仓库早前（2026-08-03 / 08-09 文献扫描）核实过，见 `CLAUDE.md` 对应小节 |
| △ | **未核实**，来自通用背景知识。引用前必须自己查一遍，尤其是仓库 URL 和会议档位 |

一个当天踩到的坑，写在最前面：**不要用 neurips.cc / iclr.cc 的 URL 路径推断报告档位。**
KVzip 的页面是 `neurips.cc/virtual/2025/poster/118741`，但页面标签是 **"2025 Oral Poster"**，
它是 oral。oral 论文的 URL 路径同样是 `/poster/`。要读页面上的格式标签，或看作者仓库的标题。

---

## 1. TL;DR — 五条对本项目有直接后果的结论

1. **我们的架构没问题，训练目标错了。** `Still`（2606.07878）与我们的设计几乎逐项相同——冻结骨干、
   每层一个小编码器、单次前向摊销压缩、紧凑 KV 进 softmax——它做到 8×–200×，我们 3× 就崩。
   唯一实质差别是它用**满上下文 teacher 对压缩缓存 student 的前向 KL、只在答案端 token 上**监督。✔
2. **训练/评测长度不匹配是文献认定的致命项**（Still 原文：8k 训练的压缩器部署到 128k 会**掉到
   "无上下文基线"之下**）——**但经 2
   我们当前的 ckpt 不成立，不要拿它当借口。**
   我们的训练数据是 FastKVzip 门控训练那份 FineWeb-Edu 的**逐字复制**（`load_fineweb("fineweb_10k")[:29]`
   + `("fineweb_10k_cat")[:5]`，34 篇、0.98M token、长度 **10,444–119,319**），训练日志的分块数分布
   （chunk=16000 时 `num 2` × 397 / `num 7` × 132 / `num 8` × 88）证明训练上下文实际覆盖到
   **112k–128k**，没有被截短。"8k 训练"只适用于最早的 `varikv/ckpt_stage2b/`，
   与现在评测的 `ckpt_stage2b_res` / `ckpt_gap_*` 无关。✔
   **真正残留的不匹配是任务形态，不是长度**：训练是"文档尾部 128 token 的续写"，评测是"问题在末尾、
   针在任意位置的检索"。这才是该攻的点，见第 1 条的目标函数。
3. **固定容量是已知的错误设计。** Still 的保留缓存是 **1/c 线性、明确不是 O(1)**；KV Means 报告固定
   状态需要 **√N** 增长才能在超长上下文上竞争。我们是固定 16 槽/head 处理 169k（ratio 0.3 时约 800:1）。✔○
4. **战场在 10×–100×，不在 3×。** 所有报告增益的工作都在激进压缩区；而我们自己的 headroom 表
   （`CLAUDE.md` 2026-08-11 节）显示 11 个数据集里只有 `scbench_kv` 在论文区间内有真空间。✔
5. **"不确定性感知地吸收驱逐 KV"已被占。** RetentiveKV 用注意力熵、MomentKV 用矩统计、IndexMem 用
   fast-weight，且**全部在输出端残差融合**。我们剩下的差异只有"高斯 (μ,σ²) + KL 门控 + 摊销变分"，
   而自家 11 数据集数据显示 `dist` 输给 `point` 10 次。✔

---

## 2. 我们 vs 五个最近的亲戚（逐维度）

| 维度 | 我们 (VariKV) | Still ✔ | Attention Matching ✔ | IndexMem ○ | RetentiveKV ✔ | MomentKV ✔○ |
|---|---|---|---|---|---|---|
| 骨干 | 冻结 | 冻结 | 免训练 | 冻结 | 冻结 | 免训练 |
| 压缩表示 | K 个高斯槽 (μ,σ²) | 紧凑 KV（Perceiver 输出） | 紧凑 KV（闭式拟合） | fast-weight 矩阵 | 矩阵状态 S_t | 矩统计（count/均值/协方差） |
| 读回方式 | 进 softmax（旧）/ 输出残差（新） | **替换** prefix KV，进 softmax | **替换** 原条目 + logit 偏置 | 输出残差 + 门 | 输出残差 | 输出端事后校正 |
| 监督信号 | ① 网页续写 LM loss ② 注意力残差 MSE | **答案端 forward KL 蒸馏** | 闭式最小二乘（无监督训练） | LongAlpaca SFT | 无（免训练） | 无（免训练） |
| 容量随长度 | **固定 K=16** | 1/c 线性 | 按预算比例 | 固定矩阵 | 固定矩阵 | 固定 |
| 压缩比 | 3× 就崩 | 8×–200× | 2×–100× | 4×–10× 区间报增益 | 5× | — |
| 不确定性 | 高斯方差 + KL 门 | 无 | 无 | 无 | 注意力**熵**门 | 二阶矩（频率派） |

**读法**：我们和它们的差别集中在两栏——**监督信号**和**容量随长度**。而"不确定性"这一栏
是我们唯一独有的东西，也正是自家消融里最不出彩的一项。

---

## 3. 家族 A：学习型 / 优化型紧凑 KV —— **我们真正的赛道，也是当前最强结果**

这一支的共同点：**合成出一小组 KV（或等价物）替换原缓存**，而不是从原 token 里挑子集。

### Still: Amortized KV Cache Compaction in a Single Forward Pass ✔
- arXiv **2606.07878**（2026-06-05），**preprint**，未见代码
- 技术：每层一个小 **Perceiver**——可学习 latent query 交叉注意整层 KV，经自注意+FFN 精炼，
  共享线性头投出紧凑 K/V。**推理时单次前向，无逐上下文优化、无梯度步、无闭式求解**
- 目标：**前向 KL(满上下文 teacher ‖ 紧凑缓存 student)，只在答案端 token 上加 mask**；
  四领域 QA 约 12 万条，AdamW 4e-5，3 个 seed
- 结果：8×–200×；RULER 上比 KV-Distill 高 **8–22 分**；对比 H2O / SnapKV / StreamingLLM /
  Attention Matching / KV-Distill；模型 Qwen3 4B–32B + 30B-A3B MoE、Gemma-3 4B；
  基准 RULER 16k–128k、HELMET multi_lexsum、LongBench v1/v2、QuALITY
- 自述限制：**不无损**；**强依赖训练 horizon**（8k 训练→128k 部署会掉到无上下文基线之下）；
  保留缓存 **1/c 线性、不是 O(1)**
- **与我们的关键差异：只有训练目标和容量调度。** 这是本项目最应该逐行对齐的一篇

### Fast KV Compaction via Attention Matching ✔
- arXiv **2602.16284**（2026-02-18，v2 05-26），MIT；**代码 [github.com/adamzweiger/compaction](https://github.com/adamzweiger/compaction)，MIT 许可**（含评测代码，无训练代码）
- 技术：**完全免训练闭式解**，逐 (layer, KV head)。拟合紧凑 K、逐 token 注意力质量偏置 β、紧凑 V，
  目标是在参考 query 上匹配**注意力输出**与**注意力质量**：β 用 NNLS、V 用 OLS、key 选择用
  最高聚合注意力或 OMP 贪心
- 集成：紧凑 KV **替换原条目进 softmax**，β 作为 logit 偏置（**不是**残差）
- 结果：10% 缓存 QASPER F1 **0.428 vs 基线 0.104**（Qwen3-4B）；1% 缓存 LongBench v2 **61.7%**；
  20×–100× 打赢 H2O+ / SnapKV / PyramidKV / **KVzip** / KVMerger；约 150 s/上下文，比 Cartridges 快 ~100×
- 自述限制：紧凑 key 被限制为**原 key 的子集**，限制极端压缩；*"directly learning compact keys
  could improve extreme compression"*；100× 时 Cartridges 在 LongHealth 上反超
- **与我们的关键差异：这是我们 `--obj gap` 的同一个优化问题，它 2 秒闭式解完。** 我们用 SGD 穿过
  瓶颈编码器加一个门，得到 loss 0.003 和一个关掉的门 ⇒ 强烈指向失败在参数化/优化，不在目标方向。
  它的限制那一句，是留给我们的唯一口子

### Cartridges: long context representations via self-study ✔
- arXiv **2506.06266**，**ICLR 2026**；**代码 [github.com/HazyResearch/cartridges](https://github.com/HazyResearch/cartridges)**
- 技术：为某个语料**离线训练一个小 KV cache 前缀**（"cartridge"），用 self-study 上下文蒸馏
  （把语料切块、让 LLM 自问自答生成合成 QA）
- 结果：约 **38.6× 省显存、26.4× 提吞吐**，长上下文基准上匹配 in-context 表现
- **与我们的关键差异：逐语料训练、可复用，不是逐上下文在线摊销。** 它是"离线换在线"的路线；
  我们和 Still 是"一次训练、任意上下文单次前向"的路线。两条路线的成本模型不同，别混着比

### KV-Distill △
- Still 的主要对照基线（被其高 8–22 分）。细节未核实

---

## 4. 家族 B：压缩式记忆吸收驱逐 KV —— **我们自认的家族，已相当拥挤**

### IndexMem ○
- arXiv **2605.25475**，**ICML 2026 poster**（`icml.cc/virtual/2026/poster/63943`）；**未找到代码**
- 技术：可学习 indexer 预测 KV 重要性 + 轻量潜在记忆把驱逐 token 压成在线更新的紧凑状态 +
  **残差读出**补偿被驱逐掉的注意力贡献。`o = o_attn + g(q)·m(q)`，`g→0` 精确回退基线
- 结果：RULER 4K/16K 上（Qwen/Mistral/Llama）激进驱逐下最高 **+25 分**
- **与我们的关键差异：三件套（打分器/潜在记忆/冻结骨干）与我们相同**，记忆是 fast-weight 矩阵而非
  高斯槽，且增益只在 75–90% 驱逐率出现。它是我们最早认定的"正确版本"，现被 Still 取代为更近的亲戚

### RetentiveKV ✔
- arXiv **2605.04075**（2026-04-14），**preprint**，未提代码
- 技术：`S_t = H_t ⊙ S_{t-1} + A_t ⊙ (k_t^T v_t)`，矩阵值状态吸收驱逐 KV；双状态（视觉主导 / 回忆导向）；
  读回 `O_t = Attn_local + Norm(q_t S_V + γ_t q_t S_T)`，**输出端残差**
- "不确定性感知"= **跨模态注意力熵**：保留分 `R = λα + (1−λ)H`，门 `γ_t = σ(W_r H_t + b_r)`。
  **没有方差/协方差、没有 KL**
- 结果：5× 缓存缩减 + 1.5× 解码加速；LLaVA-1.5-7B / Qwen3-VL-4B/8B；8 个多模态基准
- **与我们的关键差异：占掉了"uncertainty-aware 吸收驱逐 KV"这个 framing**，但用熵而非贝叶斯后验。
  我们若继续做，必须把"高斯+KL 优于熵"当成要证明的命题，而不是当作默认更优

### MomentKV ✔○
- arXiv **2606.01563**（2026-06-01），**preprint**，训练无关
- 技术：对驱逐集合维护 count / key 均值 / value 均值 / **value-key 协方差**；核心论点是保留集与
  驱逐集之间的**方向性不匹配**（驱逐 token 常与保留 token 近正交）；统计量**同时**用于
  (a) 引导驱逐向"已被摘要良好覆盖"的 token，(b) 驱逐后对注意力输出做闭式一阶校正
- **与我们的关键差异：它已经实现了我们 §11 的"一个量同时决定两个决策"，而且免训练。**
  这是 Stage-1 tier 3 的原型；本仓库 `varikv/moment.py` 只是近似复现，论文对比必须跑官方实现

### KV Means ○
- arXiv **2605.09877**；**代码 [github.com/featherless-ai/KVM-paper](https://github.com/featherless-ai/KVM-paper)**（模型+训练+Triton 核+lm_eval，checkpoint 在 HF `recursal/key-value-means`）
- **唯一和我们同侧的（记忆作为额外 KV 进 softmax，而非输出融合）**，因此它的结论对我们最直接：
  **固定大小状态"在极长上下文上吃力"，需要 √N 增长的可扩状态**。注意迁移限制：它从零训练 120M/350M

### Tensor Cache ○
- arXiv **2605.22884**，未见代码。FIFO 驱逐 + 把驱逐对转成紧凑联想记忆更新；
  `y = y_local + σ(g)·m_t` 输出残差；端到端训练；32k 上 NLL 5.14 vs 满 KV 6.00，打赢 Infini-attention

### Infini-attention / LESS ○△
- Infini-attention **2404.07143**：范式祖先，但 **Google 未放代码，HuggingFace 的复现明确标题为
  "A failed experiment"**。引用其为思想源头可以，把它的结果当既成事实不行
- LESS △：稀疏保留 + 低秩循环缓存的组合，细节未核实

---

## 5. 家族 C：驱逐 / 选择（我们的基线所在）

| 方法 | 会议/状态 | 关键技术 | 开源 |
|---|---|---|---|
| **KVzip** | **NeurIPS 2025 Oral** ✔（页面标 "Oral Poster"） | query-agnostic：用 LLM 自身**重建上下文**的能力给 KV 打分，3–4× 缩减、~2× 解码加速 | ✔ [snu-mllab/KVzip](https://github.com/snu-mllab/KVzip)，本仓库 `external/KVzip/` |
| **Fast KVzip**（我们的 base） | **仅 preprint** ✔ arXiv **2601.17668**（2026-01-27，Kim/Han/Yun） | 前向里嵌入轻量**门控**，prefill 与 decode 都能直接判 KV 重要性，去掉 KVzip 的运行时打分开销 | ✔ [Janghyun1230/FastKVzip](https://github.com/Janghyun1230/FastKVzip)，**无 LICENSE 文件**，本仓库 `external/FastKVzip/` |
| ReST-KV | **ICLR 2026** ✔ | 逐层输出重建 + 时空平滑 | ✔ [an-yongqi/rest-kv](https://github.com/an-yongqi/rest-kv) |
| Learning to Evict from KV Cache | **ICML 2026 poster** ✔ | 学习型驱逐策略（arXiv 2602.10238） | △ |
| LKV | preprint ✔ | 端到端学 head-wise 预算 + token 选择（2605.06676），与 FastKVzip 学习门最接近 | △ |
| CAOTE / Norm-Guided ℓ2 | **ICLR 2026 Workshop** ✔ | 按**注意力输出误差**打分驱逐 / 按 key 的 ℓ2 范数打分 | △ |
| H2O / SnapKV / PyramidKV / StreamingLLM | NeurIPS'23 / NeurIPS'24 / — / ICLR'24 △ | 累积注意力、窗口观测、逐层金字塔预算、attention sink + 滑窗 | △ 均有公开实现，URL 未今日核实 |
| DuoAttention / Expected Attention + AdaKV | ICLR'25 / — △ | 检索头 vs 流式头二分；对未来 query 边缘化求期望注意力（**+ 对旋转矩阵求期望**，见下） | △ |
| OBCache / Epiphany-Aware / AhaKV | preprint ✔（题录） | 最优脑剪枝式、无需注意力矩阵、整体注意力驱动 | △ |

**CAOTE 值得单独记**：它用"注意力输出误差"作为驱逐准则，那**正是我们 `D_i` 项的定义**。
如果我们要主张 `D_i` 有新意，必须先和它划清界限。○ 同理，FastKVzip 自带的 `expect` 门
（`prefill/attention/baseline.py:143-161`）已经在对未来 query 位置**边缘化旋转矩阵**
`R̄ = (1/T)Σ_j R_{t+j}`，是我们 RoPE 方案的现成对照。

---

## 6. 家族 D：量化，以及"驱逐+量化"联合的率-失真路线

**这一支对我们的意义是负面的：它把"率-失真"这个理论框架先占了。**

| 方法 | 会议/状态 | 关键技术 | 开源 |
|---|---|---|---|
| **ThinKV** | **ICLR 2026 Oral** ✔（`iclr.cc/virtual/2026/oral/10009981`） | **thought-adaptive 混合量化+驱逐**：注意力稀疏性揭示 CoT 里不同"思维类型"重要性不同 → 按思维重要性分配 token 精度，并随推理推进渐进驱逐次要思维的 token。Triton 写的 kernel 扩展 PagedAttention 复用驱逐 slot，免 compaction 开销。**<5% 原缓存近无损**，最高 **5.8× 吞吐**；DeepSeek-R1-Distill / GPT-OSS / AceReason | **未找到代码** ✔ |
| **TurboQuant** | **ICLR 2026 poster** ✔ | 数据无关（无需校准集）近最优 KV 量化，~3 bit、5–6× 缩减 | 官方无；第三方 [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) ✔ |
| **RDKV** | preprint ✔ **2605.08317**（ETH+清华，2026-05-08） | **把 KV 压缩直接写成率-失真问题**：驱逐与量化是同一 bit 分配方案的两个端点。给定 bit 预算，为 V 的每个 token、K 的每个 channel 分配位宽以最小化加权失真 | △ |
| RateQuant | preprint ✔ 2605.06675 | 率-失真理论导出的最优混合精度 KV 量化 | △ |
| Spherical KV | preprint ✔ 2605.18856 | 角度域注意力 + "率-失真保留" | △ |
| HqeKV | OpenReview ✔ | 量化与驱逐混合 | △ |
| KIVI | ICML 2024 △ | 2-bit KV 量化，key 按 channel、value 按 token | △ |

**后果**：`theory_distributional_memory.md` §11 里"用率-失真统一驱逐"的叙事**不能再当新东西讲**。
RDKV 已经把率-失真的 bit 分配版本做完，CapKV（2604.25975 ○）也占了"用一个信息论目标统一驱逐"。
我们能留的只有"失真项由**贝叶斯后验方差**给出"这一层，而这需要方差先被证明有用。

---

## 7. 家族 E / F / G：合并低秩、系统、理论

**E 合并/低秩/编码** △○：KVMerger、KVSlimmer（非对称 KV 合并，2603.00907）、CSKV（NeurIPS'24，
通道收缩，−80% 显存）、Palu/FDC（降维）、Lexico（通用字典稀疏编码）、
**KV Cache Transform Coding（ICLR 2026 ✔）**。

**F 系统/卸载** ○△（来自 ACL 2026 综述的分类）：Mooncake（FAST'25 **最佳论文**）、
FlashInfer（MLSys'25 **杰出论文**）、NEO（MLSys'25 spotlight）、LMCache、ShadowKV（ICML'25）、
RocketKV（ICML'25）、PQCache（SIGMOD'25）、InfiniPot-V、TinyServe（MM'25 Oral）。
**这一支和我们不冲突**，但它说明"KV 优化"作为领域已经系统化、成熟。

**G 理论/诊断 —— 有一篇为我们平反：**

### Error Certificates for KV-Cache Eviction via Randomized Design ✔
- arXiv **2607.21475v2**（2026-07-25）
- **定理 1（不可辨识）**：确定性 top-k 驱逐下，任何只用保留状态计算的估计量都**无法一致估计驱逐误差**
  ——被驱逐的 value 可以被任意改动而服务端保留的一切不变，真实误差却任意增大
- **定理 2**：已知包含概率的 **Poisson 随机化设计**恢复可辨识性；Hájek 校正就是一个 logit 偏置
  （给保留 logit 加 `log(1/π_i)`，让 softmax 分母完成重归一化）
- **精度不打折**：25% 预算下中位相对误差 **0.0317（Poisson+Hájek）vs 0.0447（top-k）vs 0.2386（均匀采样）**；
  question-aware LongBench 上确定性与 Poisson 差距 ≤0.7 分
- 模型 Qwen2.5-1.5B/7B/32B、Llama-3.1-8B、Mistral-7B
- **对我们的意义：这是 Stage-1"随机驱逐打赢所有原则性准则（含 Expected Attention）"的已发表对应物。**
  那一行不是坏实验，随机化选择确实有竞争力、且打分排序有形式化的可辨识性问题。以后引用它

### When Does Value-Aware KV Eviction Help? ✔
- arXiv **2605.08234**（2026-05-07）。"fixed-contract" 诊断：固定住 prefill 注意力张量、query 域、
  观测窗、预算、分配规则、投影方法，**只换排序标量**，才能把失败归因到具体阶段
- **压缩是非单调的**："压缩后的缓存可能低于、等于或**高于** FullKV"
- 增益集中在 **positive-margin cell**（FullKV 打赢基线的格子）：**72.6% vs 其他 32.4%**
- **对我们的意义：它把我们 2026-08-11 建的 headroom 表形式化了。**"只有基线真丢了分的地方才有可能赚回来"
  现在是可引用的结论，不只是我们的推断

### VASE ✔（题录）
- 2606.03928，value-aware **随机**驱逐；观察到 value 幅度分布强偏斜、少数 token 幅度异常大
  ——与我们 F 分布重尾（kurtosis 702）的发现同类

**综述**：*Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization*，
**ACL 2026 Findings**，arXiv **2607.08057** ✔，三维分类（temporal 调度 / spatial 放置 / structural 表示）。
跟踪列表：[jjiantong/Awesome-KV-Cache-Optimization](https://github.com/jjiantong/Awesome-KV-Cache-Optimization)、
[October2001/Awesome-KV-Cache-Compression](https://github.com/October2001/Awesome-KV-Cache-Compression)。

---

## 8. 顶会档位汇总（只列当天核实过的）

| 论文 | 档位 | 依据 |
|---|---|---|
| **ThinKV** | **ICLR 2026 Oral** | `iclr.cc/virtual/2026/oral/10009981` ✔ |
| **KVzip** | **NeurIPS 2025 Oral** | 页面标签 "2025 Oral Poster"；仓库标题 `[NeurIPS'25 Oral]` ✔ |
| Cartridges | ICLR 2026（档位未定） | 在 ICLR 2026 proceedings ✔ |
| TurboQuant / KV Cache Transform Coding / ReST-KV | ICLR 2026（poster/conference） | ✔ |
| IndexMem / Learning to Evict | ICML 2026 poster | `/poster/63943`、`/poster/66783` ✔ |
| CAOTE / Norm-Guided ℓ2 | ICLR 2026 **Workshop** | 非主会 ✔ |
| Fast KVzip（我们的 base） | **preprint，未发表** | arXiv 2601.17668 ✔ |
| Still / AM / RetentiveKV / MomentKV / RDKV / Error Certificates | preprint | ✔ |

**查不到的必须说清**：ICML 2026 本题目的完整 oral 名单我没能检索到可核实页面；
NeurIPS 2026 决定在 2026-08 应该还没出。**因此"2026 顶会 KV 压缩 oral"目前可核实的只有 ThinKV 一篇**，
而它打的是 **decode 侧 reasoning 战场**（对应 FastKVzip 论文 Figure 13），本项目从未涉足。

---

## 9. 开源状态汇总

| 已核实**有**代码 ✔ | 已核实**无** / 未找到 ✔ | 未核实 △ |
|---|---|---|
| Attention Matching（MIT，含评测） | Still、ThinKV、IndexMem、RetentiveKV、Tensor Cache | H2O / SnapKV / PyramidKV / StreamingLLM / DuoAttention / KIVI 等经典（有公开实现，URL 未今日核实） |
| Cartridges（HazyResearch） | Infini-attention（Google 未放，HF 复现失败） | RDKV / RateQuant / MomentKV / LKV / CAOTE |
| KVzip、Fast KVzip（**无 LICENSE**） | TurboQuant（官方无，仅第三方实现） | |
| KV Means（含 Triton 核 + HF ckpt） | | |
| ReST-KV | | |

**对我们最有用的三个可跑仓库**：Attention Matching（同一目标的闭式上界，直接量我们差多远）、
KV Means（唯一同侧的 KV-into-softmax 方法，且报告固定容量的失败）、Cartridges（训练型上界）。

**法务提醒**：`external/FastKVzip/` **没有 LICENSE 文件**，且论文仍是 preprint。
任何基于它的公开发布，需先与作者确认代码许可。

---

## 10. 与我们的关键差异 —— 逐维度结论

| 维度 | 领域的做法 | 我们的做法 | 判断 |
|---|---|---|---|
| **监督信号** | 答案端 KL 蒸馏（Still）/ 闭式注意力输出匹配（AM）/ 免训练（多数） | 网页续写 LM loss；注意力残差 MSE | **错，且是主因**。LM loss 与下游实测反相关；MSE 目标疑似退化（`m≡0` 在解空间内） |
| **读回位置** | 输出端残差（IndexMem/Tensor Cache/RetentiveKV）或**替换式** KV（Still/AM） | 追加式 KV 进 softmax（旧）/ 输出残差（新） | 旧版错在**追加**而非替换（额外抢注意力质量）；新版对，但门被训到关闭 |
| **容量随长度** | 1/c 线性（Still）/ √N（KV Means 建议）/ 按预算比例（AM） | **固定 16 槽/head** | 错。比任何成功报告都激进一个数量级 |
| **训练/评测长度** | 明确要求匹配（Still 说不匹配会掉到无上下文基线之下） | 训练上下文覆盖 10k–119k（实测分块数 num 2/7/8），chunk/window 与评测一致 | **没问题**，2026-08-11 核实。"8k 训练"只属于最早的 `ckpt_stage2b` |
| **训练任务形态** | 答案端蒸馏 / QA 数据（Still、Cartridges、IndexMem 用 LongAlpaca SFT） | 文档尾部 128 token 续写 | **错**。长度对了，任务没对：续写 ≠ 检索 |
| **评测压缩比** | 10×–100×（ratio 0.1–0.01） | 主要在 3×（ratio 0.3） | 错。那里基线近无损、无空间 |
| **不确定性建模** | 熵（RetentiveKV）/ 二阶矩（MomentKV）/ 无（其余） | 高斯 (μ,σ²) + KL 门 + 摊销变分 | **唯一独有项**，但自家数据里 `dist` 输给 `point` 10/11 |
| **率-失真框架** | 已发表（RDKV/RateQuant/Spherical KV/CapKV） | §11 当作自有叙事 | 需改写定位 |

---

## 11. 处方（按执行顺序）

1. ~~先测 `mean(tgt²)`~~ **已于 2026-08-11 测完（`scratch_probe_gap_target.py`），结论改变了后续顺序：**
   达成 MSE 是平凡解的 **0.85–0.93**（所以"loss 0.003"确实几乎没有信息量），
   但 `R_opt`（只重调 per-head 门能拿到的最大下降）= **11–15%，不是 0**，且在**有信号的层里门已接近最优**
   （layer 26：达成 0.700 vs 下界 0.687）。**⇒ 目标没坏、门没坏，瓶颈是容量。**
   完整数据见 `CLAUDE.md` 2026-08-11 节。因此下面第 2 条（换目标）**必须排在容量之后**，
   否则是在一个已经触到自身上限的参数化上优化。另两个附带结论：`--ratio_mode random` 有害
   （固定 0.3 训练的 ckpt 在 0.3 处 R_opt 15.5% vs 随机比例的 11.0%）；信号只在最后几层，
   残差读出只挂深层即可省掉 25 层的开销与噪声。
2. **换成 Still 式答案端 KL 蒸馏**（满缓存 teacher vs 压缩缓存 student，只在答案 token 上）。
3. ~~训练长度对齐评测长度~~ —— **已核实本来就是对齐的**（10k–119k，chunk/window 同评测）。
   要改的是**训练任务形态**：现在是文档尾部续写，应换成答案端监督（与第 2 条同一件事）。
4. **容量随上下文缩放**（1/c 或 √N），放弃固定 16 槽。
5. **评测区间移到 ratio 0.1–0.01**，并只在 headroom 表里有正空间的数据集上做主实验。
6. 若仍要保留分布式记忆这个卖点：把对照设成 **RetentiveKV 的熵门** 与 **MomentKV 的二阶矩**，
   在极端压缩区证明"贝叶斯后验方差 > 熵 / 频率派矩"。这是当前唯一还没被占的命题。

---

## 12. 相关本仓库文档

- `kv_cache_survey.md` — 2026-07-22，**只覆盖 training-free**，家族划分的来源
- `CLAUDE.md` §"Literature sweep 2026-08-11" — 本文件的浓缩版 + 对实验结论的直接后果
- `CLAUDE.md` §"Literature sweep 2026-08-09" — IndexMem/Tensor Cache/KV Means/VECTOR 的首轮扫描
- `CLAUDE.md` §"Competing work 2026-08-03" — MomentKV/Kalman 系/CapKV/Titans/Larimar 与两处引用订正
- `theory_distributional_memory.md` §9–§11 — 我们的理论与方法设计（§11 的率-失真叙事需按本文件第 6 节改写）
- `kv_direction_positioning.md` — Path A/B 决策与接受概率估计
