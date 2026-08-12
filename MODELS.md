# MODELS.md —— 训练好的记忆模块清单

> **本文件的表格用 `scratch_model_registry.py --md` 自动生成，不要手改那部分。**
> 手工维护的只有「每代的用途」与「评测结果」两节。
> 重新生成：`.venv/bin/python scratch_model_registry.py --md`
>
> **没有任何 LLM 被训练过。** backbone 全程冻结，这些全是记忆模块（7B 那些是 0.33M 参数）。
> 全部位于 `varikv/ckpt*/`，不在仓库根目录。

## 一句话现状（2026-08-12）

**`varikv/ckpt_kl/s2b_dist_k16.pt` 是本项目第一个有真实下游正收益的 ckpt**：
`scbench_kv` @ratio 0.1 上 32.60 → **54.20**（+21.60，95% CI [+15.20,+27.60]，
HRR 60.7%）。同架构、同评测，只把训练目标从 `lm` 换成多位置 teacher KL，
从 −43 分变成 +21.60 分 —— **「当初是不是训练错了」这个问题的答案是「是」。**

三条必须同时说的限制：
1. **只在 `scbench_kv` 一个数据集、一个比例（0.1）上测过。** 它是 11 个数据集里
   唯一有大 headroom 的（35.6 分），先例是三个 `gap_*` ckpt 在 10 个数据集上全 null。
   七数据集扫描进行中（`scratch_klsweep.sh`）。
2. **`dist` 胜 `point` 有混淆项**：point 学到的门开到两倍（0.265 vs 0.131），
   而「门越开分越低」是本项目早已建立的规律，其生成长度只有 48.9（基线 120.5）
   ⇒ 退化输出。所以这不能算「方差携带信息」的证据。
3. **v1 用的是修复前的代码**（backbone 未冻结、无 seed、无固定验证窗口）。
   数学无误，但复现性待验：`v2b` 是干净的复现臂。

---

## 世代与用途

| 代 | 目录 | 用途 / 关键特征 |
|---|---|---|
| stage1 | `ckpt/` | 1.5B 合成 needle 任务，K∈{2,4,8,16,32,64} × tier{2,4,5}。**结论：K 越大越差** |
| stage2a | `ckpt_real/` | 1.5B 真实语料 fineweb-edu。**结论：丢弃胜过吸收，反转 stage1** |
| stage2b-v0 | `ckpt_stage2b/` | 首次接入 harness；训练 2048/256/8k ≠ 评测 16000/4096（配置错配） |
| stage2b-v1 | `ckpt_stage2b_matched/` | 训练配置对齐评测。**KV 注入 ⇒ 10 数据集上 dist 输给 point 10:1** |
| stage2b-v2 | `ckpt_stage2b_retain/` | 改建在 `RetainCache` 上（基线所用机制）。门未训练，停在初值 0.018 |
| stage2b-res | `ckpt_stage2b_res/` | 残差读出 + `--obj lm`。门单调开到 0.186 ⇒ **下游崩 −56…−68** |
| stage2b-gap | `ckpt_gap_*/` | 残差 + `--obj gap`。门训到 0.014（**低于初值**）⇒ 10 数据集全 null |
| **v1-KL** | **`ckpt_kl/`** | **残差 + 多位置 teacher KL ⇒ 首个下游正结果** |
| v2a-KL | `ckpt_kl_v2a/` | v1 配置 + 全部修复，但 `min_chunks=1` 误滤成 14/34 篇 ⇒ 不是干净复现 |
| v2s-KL | `ckpt_kl_v2s/` | 流式：10 篇长文档、每步 4 次驱逐、800 步。验证恢复率**为负** |
| v2b-KL | `ckpt_kl_v2b/` | **v1 的干净复现**：修复后代码 + `min_chunks 0`（全 34 篇）。训练中 |

### 三条贯穿所有世代的规律

- **门关着 ⇒ 与基线逐字相同；门开着 ⇒ 更差；让训练自己决定 ⇒ 它关门。**
  在 `lm` / `gap` 目标下成立（4 个数据集复现）。**teacher KL 打破了它** ——
  这是第一个训练把门主动开到 0.13 且下游变好的目标。
- **训练指标好 ≠ 下游好。** `lm` 的 loss 掉到 1–2、门单调打开，下游崩 30–45 分。
  所以任何训练侧读数都必须配下游评测才算数。
- **绝不横向比 `results.parse` 的相对行**：每个 run 用自己的满缓存分数做分母。
  一律报绝对分 + 逐样本配对 bootstrap。

---

## 评测结果（`scbench_kv` @ratio 0.1，100 样本，同批基线，逐样本配对）

满缓存 68.20，基线 32.60，headroom 35.60 分。★ = 95% CI 不含 0。

| ckpt / 方法 | 绝对分 | HRR | Δ vs 同批基线 | 备注 |
|---|---|---|---|---|
| **`ckpt_kl/dist`** | **54.20** | **60.7%** | **+21.60 [+15.20,+27.60] ★** | 迄今最好 |
| 质心 K=1024（免训练） | 43.60 | 30.9% | +11.00 [+6.60,+15.20] ★ | 无 ckpt，`attention/centroid.py` |
| 质心 K=16（免训练） | 42.20 | 27.0% | +9.60 [+5.20,+14.00] ★ | 同上 |
| 等预算对照 mb1024 | 35.60 | 8.4% | +3.00 [+0.60,+5.60] ★ | 同字节改成多留真实 KV |
| 基线 @0.1 | 32.60 | — | — | FastKVzip |
| `ckpt_kl/point` | 14.60 | −50.6% | −18.00 [−25.20,−11.00] ★ | 门开过头 ⇒ 生成退化 |
| `ckpt_stage2b_res/dist` | 4.60–11.00 | — | −25…−67 ★ | `lm` 目标 |
| `ckpt_gap_*`（三个） | ≈基线 | ≈0 | 未分离 | 10 数据集 × 5 比例 = 45 格里 44 格未分离 |

**关键自检（每次报结果都要做）**
- 各 arm 的 `full__` 必须一致：本批全是 68.20 ⇒ 满缓存参照干净（P0-A guard 生效）
- **不是靠长回答骗宽松子串匹配**：`ckpt_kl/dist` 生成 107.4 字符，**比基线 120.5 更短**，
  含 gold 从 32.6% 升到 54.2%
- 额外内存 = 1792 条 KV = 保留预算的 **0.09%**，不是预算作弊

### 进行中

| 实验 | 内容 | 判据 |
|---|---|---|
| `scratch_klsweep.sh` | `ckpt_kl/dist` × 7 个数据集 + 各自 ratio-0.1 基线 | +21.60 是方法还是单数据集现象 |
| v2a / v2s 评测 | 四个 ckpt × `scbench_kv` | 修复与流式是否改变结果 |
| `scratch_v2b_wait.sh` | v2b 训练（干净复现臂） | **+21.60 能否复现** |

---

## 可复原性

**2026-08-12 起 ckpt 里存了完整的 argparse namespace。** 之前的只存
`memory`/`mode`/`num_slots`/`model` 四个键，训练配置只能靠日志副作用反推。

| ckpt | 代 | 训练完成 | 模型 | mode | K | 参数 | σ(gate) mean/max | args 可复原 |
|---|---|---|---|---|---|---|---|---|
| `varikv/ckpt/k16_tier2.pt` | stage1 | 08-07 06:52 | ? | ? | ? | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k16_tier4.pt` | stage1 | 08-07 06:53 | ? | ? | ? | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k16_tier5.pt` | stage1 | 08-07 06:53 | ? | ? | ? | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k2_tier2.pt` | stage1 | 08-07 12:05 | ? | ? | 2 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k2_tier4.pt` | stage1 | 08-07 12:06 | ? | ? | 2 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k2_tier5.pt` | stage1 | 08-07 12:07 | ? | ? | 2 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k32_tier2.pt` | stage1 | 08-07 10:16 | ? | ? | 32 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k32_tier4.pt` | stage1 | 08-07 10:18 | ? | ? | 32 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k32_tier5.pt` | stage1 | 08-07 10:21 | ? | ? | 32 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k4_tier2.pt` | stage1 | 08-07 12:06 | ? | ? | 4 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k4_tier4.pt` | stage1 | 08-07 12:07 | ? | ? | 4 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k4_tier5.pt` | stage1 | 08-07 12:07 | ? | ? | 4 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k64_tier2.pt` | stage1 | 08-07 10:18 | ? | ? | 64 | 0.34M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k64_tier4.pt` | stage1 | 08-07 10:17 | ? | ? | 64 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k64_tier5.pt` | stage1 | 08-07 10:18 | ? | ? | 64 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k8_tier2.pt` | stage1 | 08-07 12:06 | ? | ? | 8 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k8_tier4.pt` | stage1 | 08-07 12:06 | ? | ? | 8 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt/k8_tier5.pt` | stage1 | 08-07 12:08 | ? | ? | 8 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_gap_fix03/s2b_dist_k16.pt` | stage2b-gap | 08-09 16:06 | Qwen2.5-7B-1M | dist | 16 | 0.33M | 0.032 / 0.390 | ❌ 只有 4 个键 |
| `varikv/ckpt_gap_rand/s2b_dist_k16.pt` | stage2b-gap | 08-09 16:06 | Qwen2.5-7B-1M | dist | 16 | 0.33M | 0.014 / 0.338 | ❌ 只有 4 个键 |
| `varikv/ckpt_gap_rand/s2b_point_k16.pt` | stage2b-gap | 08-09 16:07 | Qwen2.5-7B-1M | point | 16 | 0.33M | 0.024 / 0.509 | ❌ 只有 4 个键 |
| `varikv/ckpt_kl/s2b_dist_k16.pt` | v1-KL | 08-12 06:52 | Qwen2.5-7B-1M | dist | 16 | 0.33M | 0.131 / 0.832 | ✅ |
| `varikv/ckpt_kl/s2b_point_k16.pt` | v1-KL | 08-12 06:52 | Qwen2.5-7B-1M | point | 16 | 0.33M | 0.265 / 0.969 | ✅ |
| `varikv/ckpt_kl_v2a/s2b_dist_k16.pt` | v2a-KL | 08-12 09:05 | Qwen2.5-7B-1M | dist | 16 | 0.33M | 0.125 / 0.977 | ✅ |
| `varikv/ckpt_kl_v2a/s2b_point_k16.pt` | v2a-KL | 08-12 09:02 | Qwen2.5-7B-1M | point | 16 | 0.33M | 0.340 / 0.993 | ✅ |
| `varikv/ckpt_kl_v2s/s2b_dist_k16.pt` | v2s-KL | 08-12 10:03 | Qwen2.5-7B-1M | dist | 16 | 0.33M | 0.041 / 0.327 | ✅ |
| `varikv/ckpt_kl_v2s/s2b_point_k16.pt` | v2s-KL | 08-12 10:00 | Qwen2.5-7B-1M | point | 16 | 0.33M | 0.062 / 0.823 | ✅ |
| `varikv/ckpt_real/real_k16_tier2.pt` | stage2a | 08-07 14:17 | ? | ? | 16 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_real/real_k16_tier4.pt` | stage2a | 08-07 14:20 | ? | ? | 16 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_real/real_k16_tier5.pt` | stage2a | 08-07 14:18 | ? | ? | 16 | 0.40M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b/s2b_dist_k16.pt` | stage2b-v0 | 08-08 10:28 | Qwen2.5-7B-1M | dist | 16 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b/s2b_point_k16.pt` | stage2b-v0 | 08-08 10:26 | Qwen2.5-7B-1M | point | 16 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b_matched/s2b_dist_k16.pt` | stage2b-v1 | 08-08 12:47 | Qwen2.5-7B-1M | dist | 16 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b_matched/s2b_point_k16.pt` | stage2b-v1 | 08-08 12:47 | Qwen2.5-7B-1M | point | 16 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b_res/s2b_dist_k16.pt` | stage2b-res | 08-09 13:15 | Qwen2.5-7B-1M | dist | 16 | 0.33M | 0.186 / 0.912 | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b_res/s2b_point_k16.pt` | stage2b-res | 08-09 13:15 | Qwen2.5-7B-1M | point | 16 | 0.33M | 0.287 / 0.978 | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b_retain/s2b_dist_k16.pt` | stage2b-v2 | 08-09 10:49 | Qwen2.5-7B-1M | dist | 16 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |
| `varikv/ckpt_stage2b_retain/s2b_point_k16.pt` | stage2b-v2 | 08-09 10:55 | Qwen2.5-7B-1M | point | 16 | 0.33M | —（无残差门） | ❌ 只有 4 个键 |

### 有完整训练配置的 ckpt

- **`varikv/ckpt_kl/s2b_dist_k16.pt`**
  `obj=kl  mode=dist  num_slots=16  ratio=0.1  ratio_mode=fixed  max_ctx=32768  chunk=16000  window=4096  target_len=256  steps=1500  lr=0.0001  gate_lr=0.02  residual=True  ctx_pos=random  kl_weight=sensitive`
- **`varikv/ckpt_kl/s2b_point_k16.pt`**
  `obj=kl  mode=point  num_slots=16  ratio=0.1  ratio_mode=fixed  max_ctx=32768  chunk=16000  window=4096  target_len=256  steps=1500  lr=0.0001  gate_lr=0.02  residual=True  ctx_pos=random  kl_weight=sensitive`
- **`varikv/ckpt_kl_v2a/s2b_dist_k16.pt`**
  `obj=kl  mode=dist  num_slots=16  ratio=0.1  ratio_mode=fixed  max_ctx=32768  chunk=16000  window=4096  target_len=256  steps=1500  lr=0.0001  gate_lr=0.02  residual=True  ctx_pos=random  kl_weight=sensitive  min_chunks=1  detach_every=1  n_short=29  n_long=5  seed=42`
- **`varikv/ckpt_kl_v2a/s2b_point_k16.pt`**
  `obj=kl  mode=point  num_slots=16  ratio=0.1  ratio_mode=fixed  max_ctx=32768  chunk=16000  window=4096  target_len=256  steps=1500  lr=0.0001  gate_lr=0.02  residual=True  ctx_pos=random  kl_weight=sensitive  min_chunks=1  detach_every=1  n_short=29  n_long=5  seed=42`
- **`varikv/ckpt_kl_v2s/s2b_dist_k16.pt`**
  `obj=kl  mode=dist  num_slots=16  ratio=0.1  ratio_mode=fixed  max_ctx=64000  chunk=16000  window=4096  target_len=256  steps=800  lr=0.0001  gate_lr=0.02  residual=True  ctx_pos=random  kl_weight=sensitive  min_chunks=4  detach_every=1  n_short=0  n_long=10  seed=42`
- **`varikv/ckpt_kl_v2s/s2b_point_k16.pt`**
  `obj=kl  mode=point  num_slots=16  ratio=0.1  ratio_mode=fixed  max_ctx=64000  chunk=16000  window=4096  target_len=256  steps=800  lr=0.0001  gate_lr=0.02  residual=True  ctx_pos=random  kl_weight=sensitive  min_chunks=4  detach_every=1  n_short=0  n_long=10  seed=42`

### 代码版本对应

| ckpt | 训练用的提交 | 说明 |
|---|---|---|
| `ckpt_kl/*` | **`222cef7`** | teacher KL 首版；修复前（backbone 未冻结、无 seed、无验证窗口） |
| `ckpt_kl_v2a/*`、`ckpt_kl_v2s/*` | `0bd84fb` | 含全部修复，但 `--min_chunks` 默认还是 1（误滤文档） |
| `ckpt_kl_v2b/*` | `7e8bac1` 及之后 | `--min_chunks` 默认已改为 0 |

`ckpt_kl` 的出处可由 ckpt 自身证明：它存了 26 个 argparse 键，而**缺**
`seed`/`detach_every`/`min_chunks`/`val_windows`/`val_every`/`n_short`/`n_long` ——
这七个全是 `0bd84fb` 才加的。

恢复 v1 的精确代码：`git show 222cef7:scratch_stage2b_train.py`
