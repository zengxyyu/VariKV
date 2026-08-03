# LLM / VLM / VLA 基础知识手册

> 面向有深度学习基础、做过 NLP/CV 的研究者
> 从零建立对大模型体系的完整认知

---

## 目录

1. [LLM 是什么](#1-llm)
2. [Transformer 和注意力机制](#2-transformer)
3. [KV-Cache：为什么重要](#3-kv-cache)
4. [VLM 是什么](#4-vlm)
5. [生成型 VLM：扩散模型基座](#5-生成型-vlm)
6. [视频 VLM：核心挑战](#6-video-vlm)
7. [VLA 是什么](#7-vla)
8. [三者关系与任务全景](#8-全景)
9. [关键术语速查](#9-术语)

---

## 1. LLM

### 1.1 定义

**Large Language Model（大语言模型）**：
- 输入：文字 token 序列
- 输出：下一个 token 的概率分布（自回归生成）
- 核心结构：Transformer（几乎所有现代 LLM）

### 1.2 自回归生成

LLM 不是一次输出整句话，而是**逐 token 生成**：

```
输入：「今天天气」
→ 预测下一个 token：「很」（概率最高）
→ 输入：「今天天气很」
→ 预测下一个 token：「好」
→ 输入：「今天天气很好」
→ 预测下一个 token：「。」（句子结束）
```

每次生成一个 token，把它拼到输入末尾，再生成下一个。

### 1.3 Token 是什么

Token 不等于字或词，是**子词（subword）单元**：

```
英文：「unhappiness」→ [「un」, 「happ」, 「iness」]  3个token
中文：「人工智能」  → [「人工」, 「智能」]             2个token
代码：「def foo():」 → [「def」, 「 foo」, 「():」]    3个token

规律：
  常见词 → 1个token
  罕见词 → 多个token
  数字通常每位是1个token：「2026」→ 4个token
```

**实际影响**：模型的「上下文长度」是 token 数，不是字数。
中文一个字大约 1 个 token，英文一个词大约 1-2 个 token。

### 1.4 主流 LLM 一览

| 模型 | 机构 | 特点 |
|------|------|------|
| GPT-4 | OpenAI | 闭源，最强通用能力 |
| LLaMA-3 | Meta | 开源，学术界基础模型 |
| Qwen2.5 | 阿里巴巴 | 开源，中文强，128K上下文 |
| InternLM2.5 | 上海AI Lab | 开源，长文本能力强 |
| Gemini 2.0 | Google | 闭源，原生多模态 |
| Claude 3.5 | Anthropic | 闭源，代码和推理强 |

### 1.5 参数量与能力的关系

```
1B  以下：手机端部署，能力有限
7B  左右：单张 GPU 可运行，学术研究主流
13B 左右：两张 GPU，性能明显提升
70B 左右：需要多卡，接近 GPT-3.5 水平
405B+   ：数据中心级，接近 GPT-4 水平

学术研究一般用 7B-13B（一张 A100/H100 可微调）
```

### 1.6 训练方式

```
预训练（Pre-training）
  数据：海量文本（Common Crawl、书籍、代码等，万亿 token 级）
  目标：预测下一个 token（自监督，不需要人工标注）
  计算：极贵，GPT-4 训练耗资约 1 亿美元
  结果：模型学会语言规律和世界知识
    ↓
指令微调（Instruction Fine-tuning / SFT）
  数据：指令-回答对（几万到几十万条，需要人工构造）
  目标：让模型学会「回答问题」而不是「续写文本」
  计算：便宜，几小时到几天
  结果：变成可以对话的助手
    ↓
对齐训练（RLHF / DPO）
  数据：人类偏好数据（哪个回答更好）
  目标：让模型更有用、更安全、更符合人类偏好
  方法：RLHF（强化学习）或 DPO（直接偏好优化）
```

---

## 2. Transformer

### 2.1 整体结构

```
输入 token 序列
    ↓ Token Embedding（把 token id 变成向量）
    ↓ Positional Encoding（加入位置信息）
    ↓ × N 层 Transformer Block
       ├── Multi-Head Self-Attention
       ├── Add & LayerNorm
       ├── Feed-Forward Network（FFN/MLP）
       └── Add & LayerNorm
    ↓ LM Head（把向量映射回词表概率）
输出：每个位置的 token 概率
```

### 2.2 自注意力机制

每个 token 都会「看」其他所有 token，决定关注谁：

```
对于 token i：
  Q_i（Query）= 「我在找什么信息？」
  K_j（Key）  = 「token j 是什么类型的信息？」
  V_j（Value）= 「token j 的实际内容」

计算过程：
  score_{ij} = Q_i · K_j^T / √d     ← 相似度
  weight_{ij} = softmax(score_{ij})  ← 归一化权重
  output_i = Σ_j weight_{ij} · V_j  ← 加权求和

直觉：
  「我」这个 token 的 Q，会和「今天天气很好」里
  「今天」「天气」「好」的 K 计算相似度，
  相似度高的 K 对应的 V 会被更多地融入输出。
```

**计算复杂度**：O(n²·d)，其中 n 是序列长度。
长序列（n=100K）时计算量爆炸，这是长上下文的核心挑战。

### 2.3 多头注意力（Multi-Head Attention）

不是只做一次注意力，而是做 H 次（H 个头），每次关注不同的语义维度：

```
头1：关注「语法结构」
头2：关注「语义关系」
头3：关注「指代关系」
...
头H：关注「？」（模型自己学出来的）

最后把 H 个头的结果拼接 → 再线性变换 → 输出
```

### 2.4 FFN（Feed-Forward Network）

每个 Transformer 层里除了注意力还有 FFN：

```
FFN(x) = W_2 · activation(W_1 · x)

作用：存储和检索「事实知识」
研究发现：LLM 的事实记忆主要存在 FFN 的权重里
  （ROME 论文，NeurIPS 2022）
```

### 2.5 位置编码

Transformer 本身不知道 token 的顺序，需要额外注入位置信息：

| 方式 | 代表模型 | 特点 |
|------|---------|------|
| 绝对位置编码（Sinusoidal）| 原始 Transformer | 固定，难以外推 |
| 可学习位置编码 | BERT、GPT-2 | 灵活，但有长度上限 |
| RoPE | LLaMA、Qwen | 旋转编码，支持长度外推 |
| ALiBi | MPT | 线性偏置，泛化性好 |

---

## 3. KV-Cache

### 3.1 为什么需要 KV-Cache

自回归生成时，生成第 t 个 token 需要所有前 t-1 个 token 的 K 和 V：

```
不用 Cache：
  生成第1个token：计算 1 次注意力
  生成第2个token：重新计算 2 次注意力（含第1个）
  生成第n个token：重新计算 n 次注意力
  总计算量：O(n²)，极慢

用 Cache：
  生成第1个token：计算并存储 K_1, V_1
  生成第2个token：直接读 K_1,V_1，只算 K_2,V_2
  生成第n个token：直接读前面所有 KV，只算 K_n,V_n
  总计算量：O(n)，快 n 倍
```

### 3.2 KV-Cache 存什么

```python
# 每一层、每个 token 都有一对 K, V
# 以 LLaMA-7B 为例：
# - 32 层
# - 32 个注意力头
# - 每个头维度 = 128

K_cache.shape = [32层, 32头, 已生成token数, 128]
V_cache.shape = [32层, 32头, 已生成token数, 128]

# K_i：这个 token「是什么」的压缩表示（用于匹配/检索）
# V_i：这个 token「内容」的压缩表示（被检索后取出）
```

### 3.3 KV-Cache 的内存消耗

```
LLaMA-7B 每个 token 的 KV-Cache：
  32层 × 32头 × 128维 × 2(K+V) × 2字节(fp16) = 0.5 MB/token

4K 上下文  → 2 GB    （正常）
32K 上下文 → 16 GB   （一张 A100 装满了）
100K 上下文 → 50 GB  （装不下）

视频 VLM 额外问题：
  100帧 × 256 token/帧 = 25,600 个视觉 token
  → 额外 12.8 GB KV-Cache
  → 这就是视频 VLM 的核心矛盾
```

### 3.4 KV-Cache 管理的主要工作

| 论文 | 会议 | 方法 |
|------|------|------|
| StreamingLLM | ICLR 2024 | 保留「注意力汇聚 token」，丢弃中间 |
| SnapKV | NeurIPS 2024 | 根据注意力分布选重要 KV，压缩 10x |
| PyramidKV | arXiv 2024 | 不同层分配不同 KV 预算 |

---

## 4. VLM

### 4.1 VLM 的基本架构

```
图像/视频输入
      ↓
视觉编码器（ViT 或 CLIP）
      ↓  把像素变成向量序列
Projector（MLP 或 Q-Former）
      ↓  把视觉向量映射到 LLM 的向量空间
LLM（LLaMA / Qwen / InternLM）
      ↓
文字输出
```

这是「三段式范式」，由 LLaVA（NeurIPS 2023 Oral）确立。

### 4.2 视觉编码器

**ViT（Vision Transformer）**：

```
输入：图像（如 224×224 像素）
处理：
  1. 切成 14×14 = 196 个 patch（每个 patch 16×16 像素）
  2. 每个 patch 线性投影成向量
  3. 经过 Transformer 层处理
输出：256 个向量（196 patch + 1 CLS token），每个维度 768-1024

本质：把图像变成「视觉 token」，和文字 token 类似
```

**常用视觉编码器**：

| 编码器 | 来源 | 特点 |
|--------|------|------|
| CLIP ViT-L | OpenAI | 视觉-语言对齐预训练 |
| EVA-CLIP ViT-G | BAAI | 参数量更大（1.8B），能力更强 |
| SigLIP | Google | 改进的对比学习目标 |
| InternViT | 上海AI Lab | InternVL 系列自研 |

### 4.3 Projector 的两种类型

**MLP（简单，目前主流）**：

```
视觉 token [N, 1024]
    ↓ 两层线性层 + GELU
视觉 token [N, 4096]（LLM 维度）

优点：简单，效果好，训练稳定
缺点：不压缩，N 个 token 原样送进 LLM
```

**Q-Former（压缩，早期常用）**：

```
视觉 token [N, 1024]
    ↓ 用 32 个可学习 Query token 做 Cross-Attention
压缩 token [32, 768]

优点：大幅压缩视觉 token 数量
缺点：压缩过激，细节丢失；现在基本被 MLP 取代
```

### 4.4 VLM 的任务类型

**理解类（Visual Understanding）**：

```
视觉问答（VQA）：
  输入：图像 + 「图中的猫是什么颜色？」
  输出：「橙色」

图像描述（Captioning）：
  输入：图像 + 「描述这张图」
  输出：「一只橙色的猫坐在窗台上...」

文档理解（Document Understanding）：
  输入：发票/报告图片 + 「总金额是多少？」
  输出：「¥1,234.56」

视频理解：
  输入：视频 + 「这段视频发生了什么？」
  输出：文字描述
```

**生成类（Visual Generation）**：

```
文生图：
  输入：「一只橙色的猫坐在窗台上」
  输出：图像
  代表：DALL-E 3、Stable Diffusion、FLUX

文生视频：
  输入：「海浪拍打岩石的慢动作」
  输出：视频
  代表：Sora、CogVideoX、Wan

图生图：
  输入：参考图像 + 编辑指令
  输出：修改后的图像
  代表：ControlNet、InstructPix2Pix

统一模型（既理解又生成）：
  代表：Janus（DeepSeek）、Show-o、Gemini 2.0
```

> 注意：生成型 VLM 和理解型 VLM 架构完全不同，
> 详见第 5 节。

### 4.5 主流 VLM 一览

| 模型 | 机构 | 底座 LLM | 开源 | 特点 |
|------|------|---------|------|------|
| GPT-4V/4o | OpenAI | — | ✗ | 最强闭源 |
| Gemini 2.0 | Google | — | ✗ | 原生多模态 |
| Claude 3.5 | Anthropic | — | ✗ | 文档理解强 |
| LLaVA-1.6 | UW+MS | LLaMA | ✓ | 学术范式奠基 |
| InternVL3 | 上海AI Lab | InternLM | ✓ | 开源SOTA |
| Qwen2.5-VL | 阿里 | Qwen2.5 | ✓ | 中文强，OCR强 |
| LLaVA-Video | NTU+字节 | Qwen2 | ✓ | 视频理解 |

### 4.6 主流 Benchmark

**图像理解**：

| Benchmark | 测试内容 |
|-----------|---------|
| MMMU | 大学水平多学科题目 |
| MMStar | 综合多模态能力 |
| DocVQA | 文档问答 |
| ChartQA | 图表理解 |
| MathVista | 数学视觉推理 |

**视频理解**：

| Benchmark | 测试内容 | 说明 |
|-----------|---------|------|
| VideoMME | 综合视频理解 | 当前最主流 |
| LVBench | 长视频（>1小时）| 专注长视频 |
| MLVU | 多任务长视频 | 多种任务类型 |
| EgoSchema | 第一人称视频 | 日常活动理解 |

---

## 5. 生成型 VLM：扩散模型基座

### 5.1 理解型 vs 生成型的根本区别

```
理解型 VLM（你的研究方向）：
  图像/视频 → ViT → MLP → LLM → 文字输出
  核心主干：LLM（自回归 Transformer）
  推理方式：一次前向传播

生成型 VLM：
  文字 → 文字编码器 → 条件信号
  噪声 → VAE ← 去噪网络（U-Net 或 DiT）← 迭代去噪 → 图像/视频
  核心主干：扩散模型（Diffusion Model）
  推理方式：50-100 步迭代去噪
```

### 5.2 扩散模型的基本原理

```
训练阶段（加噪）：
  真实图像 x_0
    → 逐步加入高斯噪声
    → x_1, x_2, ..., x_T（纯噪声）
  模型学习：给定 x_t，预测加入的噪声 ε

推理阶段（去噪）：
  纯高斯噪声 x_T
    → 模型预测并去除噪声
    → x_{T-1}, x_{T-2}, ..., x_0（生成的图像）
  文字条件通过 Cross-Attention 注入每一步
```

### 5.3 文生图：核心基座

**完整架构：**

```
文字 prompt
  ↓ CLIP / T5 文字编码器
条件向量 c
  ↓ Cross-Attention 注入
去噪网络（U-Net 或 DiT）← 高斯噪声
  ↓ 50步迭代去噪
潜在特征 z（压缩的图像表示）
  ↓ VAE 解码器
生成图像
```

**VAE 的作用**：
```
不在像素空间直接去噪（太慢），而是先压缩：
  图像（512×512×3）→ VAE 编码 → 潜在特征（64×64×4）
  在 64×64 的潜在空间去噪，速度提升 64 倍
  → 这就是 Latent Diffusion Model（LDM）的核心思想
```

**去噪网络的演进：U-Net → DiT**

| 架构 | 代表模型 | 特点 |
|------|---------|------|
| U-Net | SD 1.5、SDXL | CNN + 跳跃连接，成熟稳定 |
| DiT | SD3、FLUX、Sora | 纯 Transformer，易于扩展 |

**主流开源文生图基座：**

| 模型 | 机构 | 去噪网络 | 文字编码器 | 开源 |
|------|------|---------|-----------|------|
| Stable Diffusion 1.5 | Stability AI | U-Net | CLIP | ✓ |
| SDXL | Stability AI | U-Net（大）| CLIP×2 | ✓ |
| SD3 / SD3.5 | Stability AI | DiT（MM-DiT）| CLIP + T5 | ✓ |
| FLUX.1 | Black Forest Labs | DiT | CLIP + T5 | ✓（部分）|
| DALL-E 3 | OpenAI | — | — | ✗ |
| Imagen 3 | Google | — | T5-XXL | ✗ |

### 5.4 图生图：在文生图基座上加条件控制

图生图不是独立的基座，而是在文生图基座上增加条件：

| 方法 | 原理 | 代表 | 会议 |
|------|------|------|------|
| img2img | 给参考图加噪再去噪 | SD 自带 | — |
| ControlNet | 额外控制网络（边缘/深度/姿态）| ControlNet | ICCV 2023 |
| IP-Adapter | 图像 prompt 适配器 | IP-Adapter | 2023 |
| InstructPix2Pix | 文字指令编辑图像 | InstructPix2Pix | CVPR 2023 |

```
ControlNet 示意：
  参考姿态图 → ControlNet（锁定权重的 U-Net 副本）
                    ↓ 控制信号
  文字 prompt → 主 U-Net → 生成符合姿态的图像
```

### 5.5 文生视频：扩展时间维度

**核心思路**：把图像的空间 2D 注意力扩展为时空 3D 注意力

```
文生图（2D）：
  空间注意力：每个 patch 对其他空间位置做注意力

文生视频（3D）：
  时空注意力：每个 patch 对其他空间位置 + 其他时间帧做注意力

主要方式：
  ① 在 2D DiT 中插入时间注意力层（AnimateDiff 思路）
  ② 直接设计 3D DiT（Sora、CogVideoX 思路）
```

**3D VAE**：
```
视频（T帧 × H × W × 3）
  ↓ 3D VAE 编码
潜在特征（T/4帧 × H/8 × W/8 × C）
  在压缩的时空潜在空间去噪
  ↓ 3D VAE 解码
生成视频
```

**主流开源文生视频基座：**

| 模型 | 机构 | 架构 | 开源 | 特点 |
|------|------|------|------|------|
| CogVideoX-5B | 清华/智谱 | 3D DiT | ✓ | 学术界常用 |
| Wan 2.1 | 阿里 | 3D DiT | ✓ | 目前开源最强 |
| HunyuanVideo | 腾讯 | 3D DiT | ✓ | 高质量长视频 |
| AnimateDiff | 学术 | SD + 时序模块 | ✓ | 轻量，基于 SD |
| Sora | OpenAI | DiT | ✗ | 最强闭源 |
| Veo 2 | Google | — | ✗ | 闭源 |

### 5.6 统一模型：理解 + 生成

最新趋势：一个模型同时做理解和生成。

| 模型 | 机构 | 方式 | 特点 |
|------|------|------|------|
| Janus-Pro | DeepSeek | 理解用 LLM，生成用独立 DiT 解码器 | 解耦，效果好 |
| Show-o | NUS | 统一 token 预测（AR + 扩散混合）| 完全统一 |
| Gemini 2.0 | Google | 原生多模态，输出含图像 | 闭源 |
| LlamaGen | 学术 | 把图像 token 化后用 LLM 生成 | 纯 AR，无扩散 |

```
主要挑战：
  理解型 VLM 需要：精确的语义理解，输出离散 token
  生成型 VLM 需要：高质量像素生成，输出连续信号
  两者目标存在张力，统一仍是开放问题
```

---

## 6. 视频 VLM

### 5.1 视频处理的核心矛盾

```
视频 = 图像序列（时间维度）

100 帧视频的 token 数：
  100 帧 × 256 token/帧 = 25,600 个视觉 token

LLM 的上下文限制：
  典型：4K-32K token
  加上文字问题：视觉 token 要控制在 4000 以内

矛盾：25,600 >> 4,000
```

### 5.2 三种解决思路

**① 暴力稀疏采样**

```
100帧 → 均匀采样 16 帧 → 送进 LLM
问题：丢失 84% 的视频内容
```

**② Token 压缩**

```
每帧 256 token → 压缩成 32 token → 16帧 × 32 = 512 token
代表：Video-XL（VST压缩）、DyCoke（动态剪枝）
问题：压缩有损，细节丢失
```

**③ 记忆机制（最有研究价值）**

```
视频流 → 逐段处理 → 压缩写入记忆 → 只把记忆送给 LLM
代表：MA-LMM（FIFO记忆）、∞-Video（时间衰减记忆）
本文方向：变分自由能驱动的概率记忆
```

### 5.3 视频 VLM 的架构演进

```
2023：LLaVA 三段式（图像）
        ↓
2023：Video-LLaMA（加 Q-Former 处理时序帧）
        ↓
2024：MA-LMM（加 Memory Bank，CVPR 2024）
        ↓
2025：∞-Video（无需训练的连续时间记忆，ICML 2025）
      ReWind（指令引导记忆，CVPR 2025）
      MemStream（稀疏 KV + MoE 检索，arXiv 2026）
        ↓
2026：变分自由能记忆（本文方向）
```

### 5.4 Dynamic Tiling（动态分块）

InternVL3 等模型处理高分辨率图像的方式：

```
普通方式：图像 resize 到 224×224，丢失细节
Dynamic Tiling：

原始帧（如 896×448）
    ↓ 切成 448×448 的 tile

┌─────────┬─────────┐
│  tile1  │  tile2  │
│ 448×448 │ 448×448 │
└─────────┴─────────┘

每个 tile → InternViT → 256 token
2 个 tile → 512 token

+ 1 个缩略图（看全局）→ 256 token

总计：768 token/帧（这个分辨率下）
最大：12 tile × 256 + 256 = 3328 token/帧
```

---

## 6. VLA

### 6.1 定义

**Vision-Language-Action Model（视觉-语言-动作模型）**：

```
输入：图像/视频（当前环境） + 文字指令
输出：动作（不是文字！）

动作的形式：
  连续动作：机械臂关节角度 [θ1, θ2, θ3, θ4, θ5, θ6, 抓力]
  离散动作：移动方向 [前/后/左/右/上/下]
  GUI 动作：[点击(320,240)] [输入「hello」] [滚动(0,300)]
```

### 6.2 VLA 的核心组件

```
视觉输入（摄像头/屏幕） → 视觉编码器
文字指令               → 文字编码器
                              ↓
                        多模态融合（LLM 主干）
                              ↓
                        动作预测头（Action Head）
                              ↓
                        动作输出（连续/离散）
```

**和 VLM 的唯一区别**：输出头不同。
- VLM：语言建模头（LM Head）→ 输出 token 概率
- VLA：动作头（Action Head）→ 输出动作向量

### 6.3 主要应用场景

**机器人控制**：

```
RT-2（Google DeepMind，Science 2023）：
  输入：机器人摄像头画面 + 「把可乐罐放进回收箱」
  输出：机械臂的 7 个关节的连续控制信号
  特点：用 VLM 直接输出机器人动作 token

π0（Physical Intelligence，2024）：
  输入：多视角摄像头 + 任务描述
  输出：25Hz 的动作控制流
  特点：Flow Matching 生成平滑动作轨迹
```

**GUI/电脑操控**：

```
CogAgent（清华，CVPR 2024）：
  输入：电脑桌面截图 + 「帮我搜索今天的天气」
  输出：点击坐标序列

Claude Computer Use（Anthropic，2024）：
  直接控制电脑完成任务
```

**自动驾驶**：

```
MLLM-Driver 等：
  输入：行车摄像头 + 导航指令
  输出：方向盘角度 + 油门 + 刹车
```

### 6.4 VLA 的主要挑战

```
① 动作精度：
   文字生成容错高，动作控制容错极低
   机械臂误差 > 1cm 可能导致任务失败

② 反应速度：
   LLM 推理慢（几秒），机器人需要毫秒级响应
   解决方案：轻量化、缓存、异步控制

③ 安全性：
   错误动作可能损坏设备或伤人
   需要安全约束层

④ 数据稀缺：
   机器人操作数据收集成本极高
   Open X-Embodiment 项目尝试跨机器人共享数据
```

---

## 7. 全景

### 7.1 三者关系

```
                    输入                   输出              典型应用
──────────────────────────────────────────────────────────────────────
LLM    │  文字                   │  文字        │  GPT-4、对话、代码
──────────────────────────────────────────────────────────────────────
VLM    │  图像/视频 + 文字       │  文字        │  图像问答、视频理解
理解型  │                         │              │  InternVL、Qwen-VL
生成型  │  文字（+图像）          │  图像/视频   │  DALL-E、Sora
统一型  │  图像/视频 + 文字       │  文字+图像   │  Gemini、Janus
──────────────────────────────────────────────────────────────────────
VLA    │  图像/视频 + 文字指令   │  动作        │  RT-2、机器人、GUI
──────────────────────────────────────────────────────────────────────
```

### 7.2 继承关系

```
LLM（文字处理核心）
 │
 ├── + 视觉编码器 + Projector
 │         ↓
 │       VLM（理解型）
 │         │
 │         ├── 换输出头（文字→动作）→ VLA
 │         │
 │         └── 加生成模型（扩散模型等）→ VLM（生成型）
 │
 └── + 强化学习 / 工具调用 → LLM Agent
```

### 7.3 Memory 在各领域的角色

```
LLM：
  参数记忆（知识存在权重里）
  KV-Cache（当前对话的工作记忆）
  RAG（检索增强，外部长期记忆）

VLM（视频理解）：
  参数记忆（通用视觉语言知识）
  视觉 KV-Cache（当前帧的工作记忆）
  Memory Bank（跨帧的历史记忆）← 本文研究重点

VLA：
  参数记忆（运动技能）
  情景记忆（任务历史轨迹）
  工作记忆（当前状态估计）
```

---

## 8. 术语

| 术语 | 全称 | 含义 |
|------|------|------|
| LLM | Large Language Model | 大语言模型 |
| VLM | Vision-Language Model | 视觉语言模型 |
| VLA | Vision-Language-Action Model | 视觉语言动作模型 |
| ViT | Vision Transformer | 视觉 Transformer |
| CLIP | Contrastive Language-Image Pre-training | 对比语言图像预训练 |
| SFT | Supervised Fine-Tuning | 监督微调 |
| RLHF | Reinforcement Learning from Human Feedback | 人类反馈强化学习 |
| DPO | Direct Preference Optimization | 直接偏好优化 |
| LoRA | Low-Rank Adaptation | 低秩适配（参数高效微调）|
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| KV-Cache | Key-Value Cache | 注意力键值缓存 |
| Token | — | 模型处理的最小文本/视觉单元 |
| Embedding | — | 向量表示，把离散符号映射到连续空间 |
| Projector | — | 连接视觉编码器和 LLM 的映射层 |
| Q-Former | Querying Transformer | 带可学习查询的压缩 Transformer |
| ELBO | Evidence Lower Bound | 证据下界（变分推断的优化目标）|
| VQA | Visual Question Answering | 视觉问答 |
| Benchmark | — | 标准评测数据集 |
| SOTA | State-of-the-Art | 当前最优性能 |
| Hallucination | — | 幻觉（模型生成不存在的内容）|
| Perplexity | — | 困惑度（语言模型的评估指标，越低越好）|

---

*文档版本：v1.0 | 2026-04-03*
