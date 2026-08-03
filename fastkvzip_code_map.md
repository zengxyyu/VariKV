# Fast KVzip / KVzip 代码地图（Path B 插入指南）

> 整理时间：2026-07-30 | 基于本地 clone 的源码通读（`external/KVzip`、`external/FastKVzip`）
> 用途：给方案 B（分布式记忆吸收被驱逐 KV）标出**确切的插入锚点**，上服务器后照此动手。
> ⚠️ **行号基于本地这份代码，服务器上以实际代码为准**（尤其张量形状、per-head 变长布局需加载真模型确认）。

---

## 0. 一句话总览
KVzip 与 FastKVzip **共享同一套剪枝流程**（prefill → 打分 → prune → 物理驱逐 → generate），唯一区别是"打分怎么来"：
- **KVzip**：打分 = 上下文重建注意力（`attention/score.py`）
- **FastKVzip**：打分 = 训练好的轻量 gate（`prefill/attention/gate.py`）

**Path B 的锚点在"物理驱逐"那一步**：被驱逐的 KV 目前是**直接丢弃**的，你把它改成"喂进分布式记忆"。

---

## 1. 端到端工作流（KVzip `demo.py`，最清晰的 API 样板）

`external/KVzip/demo.py:31-47`：
```python
model = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M")   # 1. 加载
kv = model.prefill(context, do_score=True)          # 2. 预填 KV + 打分
kv.prune(ratio=0.3)                                 # 3. ★剪枝: 留30%、扔70%
model.generate(query_ids, kv=kv, update_cache=False)# 4. 用压缩缓存推理
```
- `ratio` = 保留比例（0.3 = 留 30%，驱逐 70%）。
- 复现时先跑这个 demo（Qwen2.5-7B-1M）确认能通。

---

## 2. KVzip 打分：上下文重建注意力

文件 `external/KVzip/attention/score.py`，类 `KVScore`：
- `_get_score()`（`score.py:36-65`）：用"重复上下文"的 query 对着缓存 key 做注意力（`score.py:44-62` 拼 sink + KV chunk + repeat chunk），softmax 后：
  ```python
  score = attn_weights.amax(dim=(-3, -2))   # score.py:63  每个 KV 收到的最大注意力 = 重要性
  ```
- `_threshold()`（`score.py:88-102`）：按 ratio 取 top，返回布尔 mask `valids`（True=保留）。
- `_threshold_uniform()`（`score.py:104-120`）：每 head 均匀预算版本。

---

## 3. ★ 驱逐锚点（Path B 在这里插入）

文件 `external/KVzip/attention/kvcache.py`。两处：

**(a) `prune()`（`kvcache.py:123-138`）** —— 决定谁留谁走：
```python
self.valid, thres = self._threshold(self.score, ratio)   # 布尔 mask
rmv = (self.valid == False)   # kvcache.py:132  ← 被驱逐的 KV 掩码
```

**(b) `prepare_init()`（`kvcache.py:152-166`）** —— **物理执行驱逐**：
```python
valid = self._get_valid(layer_idx, klen)                 # True=留 False=扔
self.key_cache[layer_idx]   = self.key_cache[layer_idx].contiguous().view(-1,dim)[valid.view(-1)]
self.value_cache[layer_idx] = self.value_cache[layer_idx].contiguous().view(-1,dim)[valid.view(-1)]
#            ↑ 只保留 valid；其余 [~valid] 被直接丢弃、不存到任何地方
```

### Path B 要做的改动（概念）
在丢弃 `[~valid]` **之前**，把被驱逐的 `(k, v)` 喂进你的分布式记忆：
```python
# 伪代码，插在 prepare_init 的驱逐行附近
evicted_k = key_cache_flat[(~valid).view(-1)]      # 被驱逐的 keys
evicted_v = value_cache_flat[(~valid).view(-1)]    # 被驱逐的 values
self.mem.write(evicted_k, evicted_v)               # ← 你的 FreeEnergyMemory(KV版): 自由能门控写入分布式 slot
# 保留原有的 valid 保留逻辑不变
```
读回：generate 时，让注意力除了看保留的精确 KV，还看 `self.mem.read()` 读出的 token（对应 memory_module.py 的 read）。

### 注意（务必在服务器上确认的细节）
- 布局是**每 head 变长**（`kvcache.py:168-172` 用 `cu_seqlens_k` 给 FlashAttention）——被驱逐的 KV 不是规整矩形，写入你的记忆前要想清楚按 head / 按层怎么聚合。
- 有两个 cache 类：`RetainCache`（打分用，`prune` + `prepare_init` 在这）和 `EvictCache`（`kvcache.py:219+`，"实际驱逐"的精简版）。先看清 demo 用的是哪个（`model/wrapper.py` 里决定）。
- FastKVzip 的对应文件是 `external/FastKVzip/prefill/attention/kvcache.py`（676 行版，结构相同，含 gate 分支）——**你实际改的是这个**，因为方案 B 基座是 FastKVzip。

---

## 4. FastKVzip 的 gate（学习式打分，替代 KVzip 重建）

文件 `external/FastKVzip/prefill/attention/gate.py`：
- `Weight`（`gate.py:47-102`）：每层一个小 `nn.Module` = `q_proj` + `k_proj`（低秩）+ 可学习 `k_base`（sink，`gate.py:72`）。
- forward（`gate.py:77-94`）：sigmoid 形式的 sink-attention 分数，输出 `bsz × n_head × seq` 的重要性。
- `load_gate()`（`gate.py:21-44`）：打分模块**可插拔分发**（`""`/`expect`/`snap`/`head`/`fastkvzip`）——**这说明"换打分方式"是设计好的扩展点**。
- gate 权重从 HF `Jang-Hyun/Fast-KVzip` 自动下载（`gate.py:149-151`）；训练用 `train_gate.sh` + `optim.py`（BCE 蒸馏 KVzip 分数，<1 H100 小时）。

**启示**：FastKVzip 已示范"把打分换成学习模块"。方案 B 主线是**吸收被驱逐 KV**（§3）；若以后想更进一步，也可把打分本身换成自由能信号（更接近方案 A）——但先做 B。

---

## 5. 复现命令（阶段 0，来自 README）

环境（`FastKVzip/README.md`）：
```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn==2.7.3 --no-build-isolation
cd csrc && make && cd ../prefill && pip install -r requirements.txt
```
评测：
- prefill 密集：`cd prefill && python eval_chunk.py -g fastkvzip -m <MODEL_ID> -d all` → `python -m results.parse ...`
- 解码密集（数学）：`cd math && python run_math.py --method fastkvzip --kv_budget 4096 --model_path Qwen/Qwen3-8B --dataset_name aime24`
- 已放出 gate 的模型：Qwen2.5-{7,14}B-Instruct-1M、Qwen3-{4,8,14}B、Qwen3-8B-FP8、gemma-3-12b-it（自动下载，**无需自训**）。

---

## 6. 文件速查表

| 关注点 | KVzip | FastKVzip |
|--------|-------|-----------|
| 入口/API 样板 | `demo.py` | `prefill/`(README)、`math/run_math.py` |
| 模型封装 | `model/wrapper.py` | `prefill/model/wrapper.py` |
| 打分 | `attention/score.py`（重建） | `prefill/attention/gate.py`（学习 gate）+ `score.py` |
| **驱逐锚点** | **`attention/kvcache.py` prune/prepare_init** | **`prefill/attention/kvcache.py`（676行版）** |
| 注意力 kernel | `attention/attn.py` | `prefill/attention/attn.py` |
| gate 训练 | — | `train_gate.sh` + `optim.py` |
| license | `LICENSE`(MIT) | **无 LICENSE 文件（发论文前问作者）** |

---

## 7. 上服务器后的下一步（对接 HANDOFF.md）
1. 复现 FastKVzip（阶段 0）——先跑通、数字对得上。
2. 在 `prefill/attention/kvcache.py` 定位真实的驱逐行（对应本文件 §3），确认张量布局。
3. 做方差消融（阶段 1，生死门）——此时可先用简化的"吸收被驱逐 KV"原型。
4. 正式移植 `memory_module.py` → KV 版，补齐四个理论缺口（见 `theory_distributional_memory.md` §9.7）。

---

*生成：2026-07-30 | 基于 external/ 下本地源码通读；行号以服务器实际代码为准*
