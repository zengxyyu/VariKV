# 机器人领域 Memory 研究综述

> 整理时间：2026年4月
> 覆盖：NeurIPS / ICML / ICLR / CoRL / RSS / ICRA 2024–2026

---

## 一、Memory 在机器人里的四种形式

| 记忆类型 | 机器人场景 | 具体形式 |
|---------|----------|---------|
| **情景记忆 Episodic** | 回忆具体过去经验 | 关键帧检索、向量数据库 |
| **工作记忆 Working** | 保持当前任务上下文 | Transformer context、SSM 隐状态 |
| **语义记忆 Semantic** | 世界知识 / 空间地图 | 3D 语义点云图、场景图 |
| **程序记忆 Procedural** | 运动技能怎么做 | 检索过去的 sub-trajectory |

**2024–2026 主流趋势：情景检索 + 工作记忆融合**
> 存选择性关键帧/观测索引，只检索相关帧，与当前感知融合。
> 镜像神经科学：海马体（情景存储）↔ 前额叶（当前控制工作记忆）

---

## 二、三大顶会相关论文

---

### ICLR 2024

#### R2I: Mastering Memory Tasks with World Models — **Oral（前 1.2%）**

- **作者**：Mohammad Reza Samsami et al.（Mila / McGill University）
- **核心问题**：DreamerV3 的 GRU 在长序列任务中丢失信息，记忆失效
- **Memory 机制**（工作记忆）：把世界模型内部的 GRU 替换成 **S4 状态空间模型（SSM）**，隐状态作为理论上无限长的工作记忆
- **结果**：Memory Maze **超人类水平**；BSuite、POPGym SOTA；训练速度 9×
- **意义**：证明 SSM 是机器人世界模型的最优 working memory 结构

---

### NeurIPS 2024

#### AMAGO-2: Breaking the Multi-Task Barrier in Meta-RL with Transformers — Poster

- **作者**：Jake Grigsby et al.（UT Austin）
- **核心问题**：多任务 meta-RL 中 Transformer 因任务间 loss 尺度差异训练不稳定
- **Memory 机制**（工作记忆）：**768 帧历史**作为 Transformer context window，整个 episode 历史即工作记忆；critic/actor loss 转为分类目标（HL-Gauss）解决多任务不稳定
- **结果**：Meta-World ML45、多游戏 Atari、多任务 POPGym SOTA
- **意义**：in-context meta-RL 可扩展到数十个任务，机器人靠 context 记住整个 episode

---

### ICLR 2025

#### STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning — Poster

- **作者**：Marius Memmel et al.（University of Washington WEIRD Lab）
- **核心问题**：few-shot 部署时，如何用已有演示快速适应新场景，不重训练
- **Memory 机制**（程序情景记忆）：检索**最相关的 sub-trajectory 片段**（非完整演示），用 视觉基础模型 embedding + DTW 动态时间规整匹配运动段，作为数据增强微调策略
- **结果**：few-shot 新场景下大幅超过 baseline，计算开销不随数据量正比增长
- **意义**：机器人"想起我以前是怎么做这个动作的"——程序情景记忆的实际应用

---

### ICLR 2026

#### MemoryVLA: Perceptual-Cognitive Memory in VLA Models — Poster

- **作者**：Hao Shi et al.
- **核心问题**：主流 VLA（OpenVLA、pi-0）每步独立，无法处理非马尔科夫长时任务（需记住之前做了什么）
- **Memory 机制**（双记忆，仿海马体-前额叶）：
  - 感知 token（低级）+ 认知 token（高级语义）= **工作记忆**
  - 跨时间步持久存储的 **Perceptual-Cognitive Memory Bank** = 情景记忆
  - 工作记忆 attention 查询 Memory Bank，检索历史 token，注入 diffusion action expert
- **结果**：SimplerEnv-Bridge **+14.6%**；长时任务 **+26 points**；真实机器人 12 任务 84%
- **意义**：当前 memory-augmented VLA SOTA；双记忆架构直接对应神经科学

#### MemER: Scaling Up Memory for Robot Control via Experience Retrieval — Poster

- **作者**：Jennifer J. Sun et al.（Physical Intelligence，pi-zero 团队）
- **核心问题**：长时操作任务（如"扫灰尘再把物品放回原位"）需要分钟级记忆，全历史代价高
- **Memory 机制**（层级情景记忆）：
  - 高层策略（Qwen2.5-VL-7B 微调）：从历史中检索**关键帧**，生成语言指令
  - 低层策略（π0.5 微调）：执行语言指令
- **任务**：Object Search、Counting Scoops、Dust & Replace（均需分钟级记忆）
- **结果**：所有三个长时真实机器人任务 SOTA；关键帧选择可解释
- **意义**：来自工业顶级实验室（pi.ai）的长时记忆方案，当前最强真实机器人长时任务结果

#### Ctrl-World: Controllable Generative World Model — Poster

- **作者**：Yanjiang Guo et al.（Stanford + 清华）
- **核心问题**：世界模型在长时想象中时序不一致，接触丰富任务尤甚
- **Memory 机制**（工作记忆）：生成未来帧时注入**稀疏历史帧**（对应 pose 信息）；模型 attend 相似历史状态稳定长时一致性
- **训练数据**：DROID 数据集（95k 轨迹，564 场景）
- **结果**：稳定生成 20 秒以上一致视频；imagined data 做 SFT → 策略成功率 **+44.7%**
- **意义**：世界模型里的 memory 能免费生成合成训练数据，效果显著

---

### CoRL 2024

#### DynaMem: Online Dynamic Spatio-Semantic Memory — Spotlight + Workshop Best Paper

- **作者**：Peiqi Liu et al.（NYU + Meta）
- **Memory 机制**（动态语义记忆）：实时维护并更新 3D 点云语义地图，用 CLIP/VLM 回答开放词汇物体定位查询；物体移动时主动追踪，消失时触发重搜索
- **结果**：250+ 开放词汇导航+操作任务；动态场景中大幅超过静态地图 baseline
- **意义**：真实家庭环境的动态语义记忆，解决"物体被移动后机器人找不到"问题

#### Embodied-RAG: Non-parametric Embodied Memory — Main

- **作者**：Quanting Xie et al.（CMU）
- **Memory 机制**（分层语义 memory forest）：语言描述按粒度（房间级 → 物体级 → 事件级）组织为树形结构；查询时遍历树检索最相关片段
- **结果**：公里级环境中 250+ 多跳空间问答任务成功
- **意义**：机器人版层级 RAG，情景+语义记忆结合

---

### ICRA 2025

#### ReMEmbR: Long-Horizon Spatio-Temporal Memory for Robot Navigation

- **作者**：Abrar Anwar et al.（NVIDIA + UT Austin）
- **Memory 机制**（情景记忆，带时间戳的向量数据库）：
  - 记忆构建阶段：VILA 视觉模型对机器人相机流生成带位置+时间戳的 caption，存入向量数据库
  - 查询阶段：LLM agent 迭代语义检索，回答空间/时间/描述性问题
- **贡献**：引入 **NaVQA benchmark**
- **结果**：超过所有 LLM/VLM baseline，支持"30 分钟前那个杯子在哪"类问题
- **意义**：机器人情景记忆召回的首个完整系统——把被动视觉流转化为可查询的"个人日记"

---

## 三、趋势总结

```
2024：工作记忆（SSM / context window）为主
2025：情景记忆检索（keyframe / sub-trajectory retrieval）兴起
2026：情景 + 工作 双记忆融合（MemoryVLA, MemER）成主流
```

---

## 四、与我们研究的关联

| 工作 | Memory 形式 | 缺失的东西 |
|-----|------------|----------|
| MemoryVLA | 确定性向量槽 | **无不确定性建模** |
| MemER | 关键帧检索 | **无概率分布，无 KL 控制** |
| DynaMem | 点云地图 | **无贝叶斯更新** |

**我们的方案**（概率槽 (μ, σ²) + 变分自由能控制更新）在机器人操作侧同样是空白。
可以考虑把机器人操作作为 future work 的第二个应用场景。

---

## 五、参考文献

| 论文 | arXiv | 会议 |
|-----|-------|------|
| R2I (Recall to Imagine) | — | ICLR 2024 Oral |
| AMAGO-2 | — | NeurIPS 2024 |
| STRAP | — | ICLR 2025 |
| MemoryVLA | 2508.19236 | ICLR 2026 |
| MemER | 2510.20328 | ICLR 2026 |
| Ctrl-World | 2510.10125 | ICLR 2026 |
| DynaMem | 2411.04999 | CoRL 2024 |
| Embodied-RAG | 2409.18313 | CoRL 2024 |
| ReMEmbR | 2409.13682 | ICRA 2025 |
