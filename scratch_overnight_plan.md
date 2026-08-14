# 过夜自动探索：决策树（2026-08-14 夜）

> 用户已休息，授权按实验结果自动推进。**本文件是给上下文被压缩后的自己看的**——
> 醒来先读这里，再读 `varikv_b_method.md` 第 9 节，不要重新推导已有结论。
> 每完成一格就在本文件打勾并提交。

## 当前在跑

| 任务 | 位置 | 预计 |
|---|---|---|
| `set_marginal` 三臂训练：sm_r3 × 3 种子 | GPU 0/1/2，`scratch_ctrl_logs/train_smr3_s{0,1,2}.log` | ~40 min |
| `set_marginal` 三臂训练：sm_cont × 2 种子 | GPU 6/7，`scratch_ctrl_logs/train_smc_s{0,1}.log` | ~40 min |
| **下游评测**：学习版 B 三臂 × 100 样本 | GPU 3/4/5，`scratch_ctrl_logs/bench_{stateful,memoryless,shuffled}.log` | 数小时 |

下游基线已有：`__r05b_chunk16k_w4096`（`scratch_r05_report.py:per_sample` 读，ratio 0.1）。

---

## 判据（**先定死，不许事后挑**）

**主判据** `Δ_history = stateful − shuffled`，全局 Δacc，跨种子 t 检验，df=n−1 双侧 95%。
**次判据** `stateful − memoryless`。
**下游判据**：对基线的配对 bootstrap，绝对分，★ = 95% CI 排除 0。

三条统计纪律（已付过学费，别再犯）：
1. 单次训练不是一次测量（v1 的 +21.60 三次重训跨度 39 分）。
2. 配对 bootstrap 量化的是评测集抽样噪声，**不是优化器方差**。
3. 「与 0 不可分」≠「为零」。要宣布为零必须做 **TOST 等价检验**，最小有意义效应
   预先定为 **δ = 0.02**（约等于 `cur−s0` 增益 0.06 的三分之一）。

---

## 决策树

### A. `set_marginal` 训练出结果后

- **A1 `Δ_history` 显著为正（任一语料）** ⇒ B 活着。依次做：
  1. 补齐种子到 5（另一语料也补齐）。
  2. 用 sm 的 stateful/shuffled checkpoint 跑**下游** `scbench_kv @0.1`，三臂。
  3. 只有下游也为正，才动"第二个 base policy"（见 C）。
- **A2 不显著** ⇒ 做 TOST（δ=0.02）。
  - 若**等价成立**（CI 落在 ±0.02 内）⇒ 这是**第三条独立否定**，且是对着正确靶子的。
    转 B（把负结果做扎实），不再改架构。
  - 若**既不显著也不等价**（功效不足）⇒ 加种子到 8，再判一次。

### B. 若判为负：把负结果做扎实（不再堆架构）

按顺序，每步都便宜：

1. **TOST 等价检验**，报 90% CI 与 δ=0.02 的关系。写进 `varikv_b_method.md` §9。
2. **借 ForesightKV 的做法补一个特征**：它用跨 chunk 的**衰减注意力累计**
   （decay 0.9 + 最近 8/16/32 窗口）作为特征，这是**已发表、可观测、且我们没用过**
   的一种"历史"。加进凸探针作为第 4 层 `(d) +注意力累计`。
   - 若它有效而"驱逐决策历史"无效 ⇒ 结论精确化为「**有用的是注意力质量的累计，
     不是决策的累计**」。这本身是个干净的、可发表的区分。
3. **记录清楚"什么是稳的"**：记忆无关的学习打分器 `+0.038~+0.067`（30/30 篇）；
   门控分在决策边界近乎随机（set_marginal 靶下 **0.532/0.546/0.533**）。
4. 更新 `CLAUDE.md` 的标准结论段与 `varikv_b_method.md` §9。

### C. 若判为正：立刻去 base-policy-agnostic

不要继续围着 FastKVzip 调。`load_gate()` 已支持 `""`/`expect`/`snap`/`head`/`fastkvzip`，
换 base 几乎免费：

1. 用 `-g snap`（SnapKV）或 `-g expect`（Expected Attention）重跑教师 + 训练 + 下游。
2. 若 `Δ_history > 0` 在两个 base 上都成立 ⇒ 论文从 "FastKVzip patch" 升级为
   **a general stateful control framework for KV eviction**。
3. 论文的研究问题定为：**存活缓存 `C_t` 是不是未来缓存控制的充分统计量？**
   而不是 "memory 能不能改进 FastKVzip"。

### D. 下游评测出结果后（与 A 独立）

- 用 `scratch_r05_report.py` 的 `per_sample` + `boot` 做配对 bootstrap，基线 `__r05b`。
- 三臂都报。**注意**：`stateful` 赢基线不足以支持 B——那可能只是学了个更好的打分器
  （ForesightKV/KVP/DBTrimKV 的地盘）。B 的证据必须是 `stateful > shuffled`。
- 若 `memoryless` 赢基线而三臂之间无差 ⇒ 记录为"学习打分器有效、历史无效"，
  这与训练侧结论一致，写进文档。

---

## 可以借鉴的做法（都已核实过原文）

| 来源 | 可借鉴的具体做法 | 用在哪 |
|---|---|---|
| **ForesightKV** 2602.03203 | ① 跨 chunk 的**衰减注意力累计**特征（decay 0.9 + 近窗 8/16/32）；② 监督蒸馏之后再用 **GRPO** 在 MDP 下微调，奖励是低熵 token 的 LM 损失尖峰 | ① 立刻可加（见 B2）；② 只在 A1 成立后考虑——它直接优化下游而非代理指标 |
| **KVP** 2602.10238 | 奖励是**跨预算的驱逐代价曲线 AUC**，即预算无关训练 | 我们只在单一 ratio 上训；若下游泛化差，这是现成的修法 |
| **DBTrimKV** 2605.09649 | 末端投影**跨层/头共享**以把分数放到同一尺度做全局排名 + 容量惩罚 `L_cap` | 我们已共享参数，但没有 `L_cap`；若 `retain_delta` 漂移明显可加 |
| **Attention Matching** 2602.16284 | key 选择用 **OMP**——"下一个选谁取决于已经选了谁"的**离线**版 | 这是 B 的离线对照。若 B 成立，OMP 就是天花板参照；若 B 不成立，说明在线摊销这一步丢掉了 OMP 的全部好处 |

---

## 硬性纪律

- **每次改代码后跑 `ast.parse`**，并报告替换是否匹配（曾有 `str.replace` 静默失败）。
- **后台任务用 `setsid`**：前台 bash 工具超时会 SIGTERM 整个进程组，`nohup` 只挡 SIGHUP。
- **`pkill`/`pgrep` 的模式不要出现在同一条 bash 命令里**，否则会匹配到工具自己的命令行
  并自杀（已发生三次）。
- **嵌套模型的自检查损失，不是准确率。**
- 每完成一格提交一次，commit message 说明**测了什么、结论是什么、以及不能由此推出什么**。
