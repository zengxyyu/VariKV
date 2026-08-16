# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A research workspace whose core contribution is `FreeEnergyMemory` — a probabilistic memory module storing distributional slots `(μ_k, σ²_k)` that uses variational free energy (ELBO) / KL divergence to gate memory updates.

**Active direction (decided 2026-07-24): KV-cache compression, targeting ICLR 2027.** The idea is being moved from its original home (long-video VLM memory inside InternVL3-8B, framed for NeurIPS 2026) to a new application: a distributional memory that *absorbs evicted KV-cache entries* instead of dropping them, built on top of Fast KVzip and trained with a frozen LLM. The video framing below and in `free_energy_memory_proposal.md` is the historical origin, not the current goal.

`memory_module.py` is still the **video-era** module (input = video segments); porting it to the KV setting (input = evicted KV) is pending work — see `HANDOFF.md` §4 for the four theory-driven changes required during that port. `memory_module_video_backup.py` is a frozen copy of the video version.

## Read First — where the truth lives, and the current status

**This file holds what a session needs *before* acting** — orientation, commands, harness facts,
and the bug/trap lists that must never be re-derived. The dated result narratives were split
into `JOURNAL.md` on 2026-08-13. What remains is still layered by date, and **later sections
supersede earlier ones — several early claims are explicitly retracted further down.** Do not
act on a claim from the middle of this file without checking whether a later section or one of
the dedicated docs below revises it. When they disagree, the dedicated doc wins.

| read this | for |
|---|---|
| `RESULTS_GRID.md` | **全网格：4 条臂 × 11 panel × 8 个 ratio 的配对 Δ**（2026-08-16）。学习残差 v2/v3 与训练无关质心 K=16/K=1024 放在同一张表、同一基线上。由 `scratch_all_report.py` 生成，**别手改**。三个结论：ρ=0.1 上质心 +3.66 **赢过**学习残差 +1.02；两条线的最佳工作点不同（质心 0.1、残差 0.2）；Retr.MultiHop 上四条臂同向失败 |
| `RESULTS_2026-08-12.md` | **the single entry point for every measured result.** Supersedes the result tables in this file |
| `JOURNAL.md` | **the chronological record, split out of this file on 2026-08-13.** Every dated result narrative — Stage 1, Stage 2b's first benchmark, the Figure-11 reproduce, the residual round, the 08-11 sweeps, the teacher-KL round, the forensics. This file keeps the standing conclusions and points there for how each was obtained |
| `MODELS.md` | the checkpoint registry (38 ckpts). Read before citing any checkpoint |
| `EVAL_PROTOCOL.md` | **which hyperparameter was tuned on which dataset, and the rules the final report must follow.** Read before choosing `--varikv_gate_scale` or any eval hyperparameter — the contamination trail is deliberate and must stay traceable |
| `NEXT_STEPS.md` | the frozen v5 roadmap + falsifiable hypotheses, with five methodological corrections that retroactively affect earlier probes (e.g. all damage must be projected through `W_O`) |
| `P0_FINDINGS.md` | the training-free diagnostics (missed mass, oracle ceiling, all-or-nothing correction, MGF accuracy) |
| `FINDINGS_DENOISING.md` | the interventional evidence for FastKVzip's own "denoising" attribution — the one place this project goes a step beyond the paper it builds on |
| `varikv_method_spec.md` | the code-level formulas of the method |
| `kv_inference_acceleration_2026.md` | competitor analysis, with per-claim provenance markers |
| `HANDOFF.md` | the *original* 2026-07-30 execution plan. **Historical** — its GO/NO-GO plan has been executed and its central bet largely falsified. Read it for the red lines, not for what to do next |

**Status as of 2026-08-13 — read this before planning any paper-shaped work.** The two things
this repo had been treating as its results are both compromised:

- **The training-free centroid is comprehensively scooped** (commit `3176194`, PDFs read
  directly, not via summary). `ResKV` reproduces the whole construction — `k̄`/`v̄`/`log n`
  summary + shared-softmax read-out (Eq 10–17), `log n_j` as the load-bearing term (Eq 14),
  modular add-on to an existing eviction method, reliability-gated and query-dependent
  injection (Eq 18–22) — and `Attention Matching` independently owns "mean-before-exp
  systematically underestimates mass" plus the `log w_j` bias correction and the
  `(2d+1)/2d` byte argument. Scale matches too: ResKV +1.02 LongBench / +3.38 RULER vs our
  +3.66 over 11 panels. **We approximately reproduced a published method with a weaker
  variant of it.** ResKV also clusters by k-means in key space where `centroid.py` uses
  position bins, so P0 §5.4's "position-local clustering does not reduce score dispersion"
  is a critique of our implementation choice, not an impossibility result — and the 13.9×
  mass-underestimate measurement and the `γ ≈ 0.75` shrinkage story built on it must be
  re-measured with k-means clusters before being treated as findings.
  **Re-measured 2026-08-16 — see `P0_FINDINGS.md` §5.4b, and note it retracts a wrong
  version written the same day.** Split by clustering scheme and measured through `W_O`
  (the only decision-relevant projection), the second-order denominator term cuts the
  error by **19–58%**, more the better the clustering, and clustering choice matters
  hugely: position 0.9366 → eucl-Lloyd 0.9045 → Cq-Lloyd 0.7961 → score-oracle 0.4045
  at K=16. So §5.4's refutation holds only for **position bins**. The reconciliation with
  §5.5's downstream refutation of second-order is that this metric is **fidelity to full
  cache**, and fidelity has since been measured not to be the objective — second-order
  does its job well, and its job is the wrong one. Caveat that must travel with the
  table: `--n` defaults to **8 documents**, and the saved arrays carry no sample id, so
  no document-level pairing is possible.
- **The learned module is not a memory.** 86% of its gain is mediated by *which KV are
  retained* (frozen-mask 2×2), its positive result is irreproducible (four trainings spread
  34.8 points), and `dist` is not separated from `point` once sampling is matched.

What survives is **analysis, not method**: the causal forensics of the learned module, and
the interventional finding that restoration fidelity and task utility diverge (Retr.MultiHop
is pulled 49.47 → 41.07 *toward* full cache as K grows, i.e. more faithful and worse).

## Two Codebases in This Workspace

1. **The research module** (`memory_module.py`, `stage1/`, and the `.md` docs) — the local, hand-written work. This is what most edits touch. Local code and comments are written in Chinese; match that when editing.
2. **`external/` — vendored upstream code, now committed into this repo** (changed 2026-08-11; previously excluded by `.gitignore`, and each clone kept its own `.git`. Both `.git` directories have since been **deleted**, so these are plain directories under this repo's history now — there is no upstream remote to diff or pull from):
   - `external/FastKVzip/` — from `github.com/Janghyun1230/FastKVzip`. **This is the intended fork base for Path B.** The real eviction code to modify is `prefill/attention/kvcache.py` (676 lines, gate branch).
   - `external/KVzip/` — from `github.com/snu-mllab/KVzip` (Fast KVzip's ancestor). Cleaner API sample; `demo.py` is the clearest end-to-end template. **Never modified by us.**
   - Neither is a dependency of `memory_module.py`; they are read/reproduced/modified per the KV plan. FastKVzip ships **no LICENSE file** and is still a preprint; it has nevertheless been pushed to `github.com/zengxyyu/VariKV` by the owner's decision.
   - The FastKVzip tree **carries substantial local edits** (loader fixes, MRCR support, and the whole VariKV integration) — see "Local modifications to the vendored FastKVzip clone" below, and `patches/` for the replayable diff against upstream.

**Path B insertion anchor** (verified in local source 2026-07-30): `external/FastKVzip/prefill/attention/kvcache.py`, class **`EvictCache`**, method **`_sample_cache` (lines 190-193)** — `mask = torch.cat(valid_list)` then `key_cache[layer_idx] = key_cache[layer_idx][mask]`. The `[~mask]` entries are dropped and stored nowhere; write them into the distributional memory *before* the mask is applied. Two caveats: (1) the layout is **per-head variable-length** (cache flattened to `[Σ_heads len_k_head, dim]`, boundaries in `cu_seqlens_k`), so evicted KV is not a rectangular tensor and the write must decide head/layer aggregation; (2) this path calls the AdaKV kernel, so `csrc` must be built (nvcc) first.

Note `RetainCache` — the **default**, and what the Stage-0 reproduce exercised — physically drops nothing; it keeps the full cache and only subsamples via a `valid` mask in `prepare()`. So the reproduce path is *not* where Path B goes. Stage 1 (the GO/NO-GO variance ablation) does not need this anchor at all; it uses simple recency/sliding-window eviction. The anchor matters for Stage 2. `load_gate()`'s dispatch (`""`/`expect`/`snap`/`head`/`fastkvzip`) is a designed extension point if the scoring is later swapped for a free-energy signal.

## Environment

**Use `/home/ubuntu/zxy/vlm-memory/.venv`** (`.venv/bin/python`, or `source .venv/bin/activate`) for everything. Built 2026-07-30, verified working: Python 3.12.3, torch 2.7.0+cu128, flash-attn 2.7.3, transformers 4.51.3, datasets 3.6.0. The machine has 8× H100 80GB and no `python` alias (use `python3`), no conda/uv, and **no system nvcc**.

Four non-obvious setup fixes that are already applied — re-apply if the venv is ever rebuilt (`scratch/install/install_chain.sh` automates most of it):

1. `apt install python3.12-venv` before `python3 -m venv .venv` (ensurepip is missing on the base image).
2. torch: `pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128`.
3. flash-attn 2.7.3 has **no prebuilt wheel for torch 2.7** and cannot source-build (no nvcc). Install the **torch2.6** wheel instead — ABI-compatible, verified on GPU — kept at `scratch/install/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`, via `pip install --no-deps`, then `pip install einops` separately.
4. `external/FastKVzip/prefill/requirements.txt`: bump `pandas==2.0.3` → `2.2.3` (2.0.3 has no py3.12 wheel). The fixed-up list is `scratch/install/reqs_fixed.txt`.

## Running the Code

```bash
# Run the memory module standalone test
.venv/bin/python memory_module.py
```

The test simulates InternVL3-8B parameters (d_model=4096, 16 slots, 32 tokens/slot) and processes 10 video segments. Expected output: ~540M parameters, free_energy values decreasing over segments, final memory shape `[512, 4096]`.

Call `model.reset()` between videos to zero-initialize `mem_mu` and restore `mem_logvar` to -2.0.

### The commands actually used day to day

All eval runs go through `external/FastKVzip/prefill/eval_chunk.py` — the VariKV/centroid
arms are **flags on the upstream script**, not separate entry points (`args.py:42-70`).

```bash
# --- one eval job: training-free centroid, one panel, custom ratios -------------
cd external/FastKVzip/prefill
CUDA_VISIBLE_DEVICES=0 VARIKV_RATIOS=0.3,0.2 ../../../.venv/bin/python -B eval_chunk.py \
    -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip --num 100 \
    --prefill_chunk 16000 --window_size 4096 --level pair \
    -d scbench_kv --tag _c23_16 --centroid_k 16      # --centroid_rope post(default)|inv

# --- same, but the learned module (mutually exclusive with --centroid_k) --------
    -d scbench_kv --tag _kl --varikv_ckpt ../../../varikv/ckpt_kl/s2b_dist_k16.pt \
    --varikv_residual --varikv_gate_scale 0.5            # see EVAL_PROTOCOL.md first
    # eval-time surgery: --varikv_gate_from, --varikv_ablate {logvar,precision,eta}

# --- train a memory module through the same path eval uses ---------------------
# The defaults (max_ctx 8192 / chunk 2048 / window 256) are NOT the eval config and
# produced the first round's confound. Always pass the matched config explicitly:
.venv/bin/python -u scratch_stage2b_train.py --obj kl --residual --mode dist \
    --num_slots 16 --ratio 0.1 --max_ctx 32768 --chunk 16000 --window 4096 \
    --target_len 256 --kl_weight sensitive --ctx_pos random --seed 42 \
    --steps 1500 --val_windows 4 --val_every 100 --out varikv/ckpt_kl_vNEW

# --- stage-1 synthetic ablation (standalone, no external/ dependency) ----------
.venv/bin/python stage1/data.py build        # writes stage1/{train,val}.jsonl
.venv/bin/python varikv/train.py --tier 5    # tiers 1 and 3 need no training
.venv/bin/python varikv/evaluate.py --tier 1 2 3 4 5

# --- reporting -----------------------------------------------------------------
.venv/bin/python scratch_model_registry.py --md      # regenerate MODELS.md tables
```

### 保真度 2×2：只有一半立住（2026-08-16）

`scratch_probe_fidelity.py` 直接测答案位置上的 `KL(p_full‖p)`（两臂共用同一串
teacher-forced token —— 各自调 `generate_answer` 会得到长度不同的答案，33 vs 35，
根本没法比），n=30：

| panel | 任务 Δ | KL base → ours | 判定 |
|---|---|---|---|
| Retr.MultiHop | −9.96★ | 0.2575 → 0.1779 | **t=−3.49 显著更接近 full** |
| Retr.KV | +4.40★ | 0.3576 → 0.3157 | t=−1.61 **不可分** |

**立住的**：MultiHop 上残差显著更忠实于满缓存、同时掉 9.96 分 ⇒
**"更忠实于满缓存 ≠ 任务上更好"是测出来的**，不是从分数曲线推的。

**没立住的**：Retr.KV 的 +4.40 **不能**归因于保真度——那里的提升与零不可分，
**机制仍未查明**。基于 n=4 预览把 Retr.KV 归到 "fidelity recovery" 的说法已撤回。

### VariKV-B 的结论（2026-08-14 夜，下游完成）：残差有效、记忆无效

**这是项目第一个未被撤回的正结果，同时也是对它自己中心命题的严格否定。完整方法说明见
`varikv_b_method.md`，那里的 §9.0 是权威版本。**

> 一个 0.6M 参数的**学习残差修正**加在 FastKVzip 的驱逐分数上，Retr.KV @10% 缓存
> 稳定 **+4.27 ± 0.23**（3 个训练种子，与 v1 的 39 分跨度是两个世界）。
> **在本架构、本监督、本工作点下，递归历史相对同等匹配的学习打分器没有可测的增量收益。**

限定词一个都不能省，下面三条是为什么：

1. **只有训练侧 ranking 指标**支持"实际等价"（TOST，δ=0.02 预注册）。**下游不支持**：
   `stateful − shuffled = +0.20 [−1.80, +2.20]`，这个区间连 ±1 分都排除不了，
   只能说"未检测到"，不能说"等于零"。
2. **配对 bootstrap 重采样的是评测样本，不是训练种子。** 下游 `stateful` 有 3 个种子而
   `shuffled` 只有 1 个——这个配对**目前不成立**，要补齐 3–5 个匹配种子后对
   逐种子的 `D_s = Y_s^{stateful} − Y_s^{shuffled}` 做配对推断，而不是把上千条样本
   当成上千次独立的历史实验。
3. **被否定的是"以过去保留/驱逐**动作**为条件"这一种历史**，`H_t = {(x_j, a_j)}`，
   不是所有时序信息。而且 `R_t/E_t` 基本由 `s⁰_t` 决定 ⇒ 记忆很大程度只是把自己的
   过去决策再编码一次；它知道"我以前删了谁"，**不知道"我以前删对了吗"**。
   要救记忆应换成 **outcome-conditioned**（写入未来实际使用/效用），而不是继续调 GRU。

三条**独立**路径一致否定历史假设：凸线性探针（30 篇留一）、非线性三臂训练（TOST 等价）、
下游配对 bootstrap。且在两种教师靶子（`U^full` / `U^setmarginal`）、两种查询类型
（续写 / 检索）、有无冗余结构（1 份 / 3 份重复事实）下都成立。

**不要再往这个记忆架构上加东西。** 当前最值钱的问题已经从"怎样把记忆做强"变成
**"这 +4.27 到底来自哪里"**——`memoryless` 的输入含 `z`、`margin`、`log(σ_h/σ_g)` 和逐
`(层,头)` 的 `M_init`，而 `level="pair"` 是全局阈值化，所以增益完全可能主要来自
**跨层/头的分数尺度重校准**（校准问题，非排序问题），而不是 KV 语义。
`attention/calib_scorer.py` 用 **224 个参数**的逐 `(层,头)` 仿射变换检验它。

两条也别再当成已定论：`U^setmarginal` 的下游 +4.20 与 `U^full` 的 +4.27 只差 0.07，
**单点对 ±0.23，分不开**；ratio 耦合也比"浅"要深——候选池是 `Near(τ_ρ)`，
`τ_ρ` 是 ratio 的分位数，所以**训练任务本身依赖 ρ**，即使 `U^full` 标签不依赖。

### VariKV-B 的核心命题在数据层面没有线性支持（2026-08-14）

**读这条之前不要再改 B 的架构。** `scratch_probe_histinfo.py` 用**凸**模型问了 B 的
中心问题——在当前特征之外，历史有没有增量预测价值——从而绕开了此前每一次零结果都
可以归咎的优化因素（α 太小、GRU 压方向、读出凸组合、训练轨迹方差）。

设置：教师 trace 10 篇，**留一交叉验证**（单次 8/2 划分不够——历史特征是逐
(文档,chunk,层) 共享的，独立观测数远小于行数，2 篇留出时文档级方差主导）。线性排序器
＋按 |ΔU| 加权的成对 logistic 损失，固定设计矩阵 + LBFGS 解到 |g|~1e-6。

| 模型 | 加权成对准确率（10 篇均值） |
|---|---|
| (a) `s0`（FastKVzip 门控） | 0.585 |
| (b) 当前特征（z / margin / ‖k‖ / ‖v‖ / s0） | **0.645，10/10 篇都赢 (a)** |
| (c) (b) + 历史特征 | 0.651（方向类）/ 0.642（冗余类） |

**历史的增量价值与 0 不可分，两族特征都是：**

- 方向类（与已保留/已驱逐 running mean 的余弦）：Δacc **+0.0059 ± 0.0169, t=1.11**
- 冗余类（与此前具体保留/驱逐条目的 max 及 top-5 相似度）：Δacc **−0.0034 ± 0.0105, t=−1.03**

冗余类是次模性框架直接指向的那一族（"这个候选是不是和已经留下的重复"——那是 **max**
不是 mean，一堆互相正交的已保留键其均值范数接近 0，均值对覆盖度是很差的代理）。
它如果有信号就该在这里出现。没有。

两个自检，两个都曾把我引向错误结论，记下来别重犯：

1. **嵌套单调性要查损失，不是准确率。** (c) 的特征空间包含 (b)，所以**被最小化的那个
   量**在训练集上必然 (c) ≤ (b)；准确率是另一个泛函，加特征后完全可以降。首版拿准确率
   做自检，误判成"未收敛"。
2. **"凸 ⇒ 优化不会失败"只有解到收敛才成立。** 首版用 3000 步 Adam，跨种子 sd 达
   0.034 且 (c) 训练损失高于 (b) —— 凸问题收敛后跨种子方差应为 0，那个 sd 本身就是
   没收敛的证据。改成固定设计矩阵 + LBFGS 后 |g| 降到 1e-6。

**这条与文献核实合起来才是完整判断**（三篇 arXiv 页面均已读过，2026-08-14）：

| 论文 | 与本项目重合 | 是否有决策历史状态 |
|---|---|---|
| **ForesightKV** 2602.03203 | Golden Eviction 用未来注意力造监督 + **pairwise ranking loss** 蒸馏 + **MDP** 表述。即本项目教师的设计 | **无**。有跨 chunk 的衰减注意力累计（decay 0.9），但那是注意力质量，不是决策 |
| **KVP** 2602.10238 | 未来效用训练轻量 policy，逐 token (k,v,pos) | 无，且刻意如此 |
| **DBTrimKV** 2605.09649 | 跨层/头共享投影的全局预算竞争 | **无，明确设计成 token 入缓存时算一次就冻住** |

所以未被占据的轴**只剩"以驱逐决策的历史为条件"**，而那正是本项目自己的数据说没有信号
的轴。`(b) > (a)` 的 +0.06 虽然稳（10/10 篇），但那是上面三篇的地盘。

**尚未排除的唯一替代解释：教师的查询分布。** U 是在 fineweb **续写**的 16 个 target
位置上算的；检索型查询下"我是不是已经留了一个相似的"可能更重要。要停 B 之前应先做这
一个测试——教师跑一遍只要 4 分钟，探针几分钟，成本可忽略。

### The silent-degeneracy trap: `ratio × clen ≤ window_size` makes eviction a no-op

**Check this before choosing any (`--ratio`, `--window`, context length) triple for
*training*.** Found 2026-08-14 by `scratch_probe_sigma.py`; it had already silently
invalidated the VariKV-B teacher's default config.

`model/wrapper.py:271-277` rescales the ratio to account for the always-retained local
window. When the requested budget is smaller than the window it takes the other branch:

```python
if chunk_ratio * clen < window_size:
    window_size = int(chunk_ratio * clen)
    chunk_ratio = 0.0                      # ← not "a bit smaller", exactly zero
```

and `score.py:_threshold` with `ratio = 0.0` computes `n = max(int(N*0)-1, 0) = 0`, so
`thres = score_sort[0]` is the **maximum** score and `valid = score > thres` is **all
False**. Consequences:

- The whole evictable range is dropped; **the retained set is exactly the local window**
  and does not depend on the gate scores at all.
- **Any method that works by perturbing the score is a no-op by construction** — the
  threshold is recomputed as the max of the *perturbed* score, so `score > τ` stays all
  False no matter how large Δs is. VariKV-B cannot move a single bit here.
- Nothing errors, and `prune_chunk` is still called once per chunk, so a "count the
  evictions" guard (`--min_prunes`) does **not** catch it.

Fingerprints in a log: the printed `Local window` differs from the `--window_size` you
passed, and the threshold sits above the 99th percentile of the scores.

The non-degeneracy condition is `clen > window_size / ratio` — **40,960 tokens** at the
standard ratio 0.1 / window 4096. Measured document lengths against it:

| corpus | n | tokens | at ratio 0.1 / window 4096 |
|---|---|---|---|
| `fineweb_10k` | 68 | ~8k–31k | **all degenerate** |
| `fineweb_10k_cat` | **10** (not 5) | ~91k–121k | usable |

So the FastKVzip gate-training corpus's *short* half cannot train anything that acts on
the eviction decision at this operating point, and `--max_ctx 32768` (the old teacher
default) truncates the long half below the threshold too — i.e. the default config was
degenerate for all 34 documents. `scratch_ctrl_teacher.py` now defaults to
`--max_ctx 131072 --n_short 0 --n_long 10` and hard-skips any document failing the
condition. Evaluation is unaffected: `scbench_kv` at 169k gives `0.1 × 169428 = 16943 >
4096`, effective chunk_ratio 0.078, and training at ~119k gives 0.068 — well matched.

**Never read `results.parse`'s relative rows across runs** — each run normalises by its own
full-cache score, and the memory perturbs that reference (see the empty-memory section).
Use the paired-bootstrap report scripts on **absolute** scores instead: `scratch_cen23_report.py`
/ `scratch_centroid_report.py` (centroid), `scratch_klsweep_report.py` (teacher-KL round),
`scratch_kvres_report.py` and `scratch_gapsweep_report.py` (residual rounds). If you adapt one
to a new dataset, copy `scratch_kvres_report.py`'s self-check of its per-sample parse against
`results.parse`'s absolute rows.

**`VARIKV_RATIOS` must be exported for the parse step too** — `eval.py` and `results/parse.py`
each carry their own `set_ratios()`, and a parse that does not see it silently prints 0.00.

### Fast KVzip reproduce / eval

Run from `external/FastKVzip/prefill/`. One dataset shard on one GPU, then parse:

```bash
cd external/FastKVzip/prefill
CUDA_VISIBLE_DEVICES=0 python -B eval_chunk.py -g fastkvzip -m Qwen/Qwen3-8B \
    -d scbench_kv --idx 0 --num 34            # --idx/--num shard the dataset
python -B -m results.parse -m qwen3-8b_fastkvzip_chunk16k_w4096 -d scbench_kv_short
```

- `get_data_list` auto-substitutes `_short`/`_mid` onto base dataset names — pass the base name to `eval_chunk.py` but the substituted name to `results.parse`. The substitution fires **only for Qwen3 / Gemma3** models; **Qwen2.5-7B-1M gets no substitution and therefore runs full-length contexts** (e.g. `scbench_prefix_suffix` at 112k tokens vs. the 20k `_short` version) — the same dataset name is far more expensive on the Figure-11 model than on Qwen3-8B.
- The result path a run writes to is `results/{dataset}/{i}_{model_shortname}{tag}/output-{level}.json`, where `tag` and `level` are derived from the gate and script (`args.py`). `scratch_repro_full.py` reimplements that derivation in `resolve_dataset`/`result_tag`/`model_shortname` — reuse those helpers rather than re-deriving the naming when writing new drivers.
- Parse prints one row per compression ratio, in order `[1.0, 0.75, 0.5, 0.4, 0.3, 0.2]`. The paper's "70% eviction near-lossless" headline is the **0.3** row.
- **True context lengths on Qwen2.5-7B-1M** (from the comments in `data/load.py:get_data_list`, spot-checked against a live run: `prefix_suffix` measured 112,577 vs. 112,635 annotated). Because this model gets no `_short`/`_mid` substitution, several datasets are **an order of magnitude longer than on Qwen3**:

  | dataset | tokens | | dataset | tokens |
  |---|---|---|---|---|
  | `gsm` | 86 | | `scbench_summary` | 117,806 |
  | `squad` | 203 | | `scbench_choice_eng` | 119,299 |
  | `scbench_many_shot` | 26,474 | | `scbench_qa_eng` | 122,101 |
  | `scbench_repoqa` | 72,499 | | `scbench_vt` | 124,551 |
  | `scbench_prefix_suffix` | 112,635 | | `scbench_mf` | **149,860** (Qwen3 uses `_mid`) |
  | | | | `scbench_kv` | **169,428** (Qwen3 uses `_short`) |

  Any cost estimate written against the Qwen3 `_short` numbers (~20k for `kv`/`mf`) understates Qwen2.5 by ~8×.
- **Several datasets hold far fewer than the `--num 100` default** (read off `loaded, #data:` in the eval logs, 2026-08-03): `scbench_choice_eng` **18**, `scbench_qa_eng` **20**, `scbench_many_shot` 54, `scbench_summary` 70, `scbench_repoqa` 88, `scbench_vt` 90. `gsm`/`squad`/`prefix_suffix`/`mf`/`kv` do give 100. This has two consequences: a job finishing with 18/100 results is **complete, not truncated** (the reason the scheduler judges completion by marker file, never by counting); and method-vs-method gaps on `choice_eng`/`qa_eng` rest on 18–20 examples, so a few-point difference there is noise, not signal.
- **On Qwen3, cost is driven by generation length, not context length.** Measured on Qwen3-8B (all 6 ratios, seconds/example): `gsm` 83s at a *77-token* context, `many_shot` 21s at 26k, `repoqa` 335s at 68k, `summary` 179s at 102k. Qwen3 is a reasoning model and emits long thinking traces, so short-context datasets are not cheap. **This does not carry over to Qwen2.5-7B-1M**, which is not a reasoning model — there both terms matter, and generation length is set per-dataset by `utils/func.py:set_gen_length` (48–512 tokens; `gsm`/`repoqa` get 512, most get 96). See `scratch/probe/timing_probe.sh` / `scratch/probe/fig11_probe.sh`.
- One `-d all` sweep for one (model, method) ≈ **33 GPU-hours** — measured on **Qwen3-8B**, where `kv`/`mf`/`prefix_suffix` run their short/mid variants. On Qwen2.5-7B-1M the same sweep is substantially more expensive (see the length table above) — **measured 2026-08-04: 5 methods × 11 datasets in 15.5h on 8 GPUs ≈ 124 GPU-h, i.e. ≈25 GPU-h per (model, method)**, plus ~10 GPU-h for MRCR. The full paper grid (6 models × 5 methods) ≈ 1000 GPU-hours ≈ 5 days on 8×H100, more with the 14B models.
- Two loaders needed a **pyarrow fallback** (already patched into `data/load.py`): `scbench_mf_mid` and `squad` both die on `Feature type 'List' not found` because their parquet was written by datasets 4.x. Upgrading datasets would break transformers 4.51.3, so the patch reads the parquet directly. All 11 `-d all` datasets load now.
- The default `--kv_type retain` needs no custom CUDA kernel. `--kv_type evict` calls the AdaKV kernel — **already built** (CUDA 12.8 toolkit + `csrc`, verify with `python -c "import torch, tiny_api_cuda"`; importing torch first is required or libc10.so is missing).

### What the paper's experiments actually are

The paper has **20 figures and 4 tables**. Verified against `scratch/refs/fastkvzip_paper.txt` on 2026-08-03 by reading every caption. Only a small subset is worth reproducing for the Path-B project — the map below exists so nobody re-derives it.

**Main results (§4)**

- **Figure 11 = prefill-intensive main result**, on **Qwen2.5-7B-1M** (the `run.sh` default). Ratios 0.2→1.0, 5 methods. The body text (§4.2) says "across **12 datasets**" — the caption's three categories (retrieval-intensive / contextual understanding / high redundancy) are groupings of those 12, not a smaller set. The 12 = 9 SCBench tasks + SQuAD + GSM + **MRCR**; `-d all` covers only the first 11.

  **Panel name ↔ dataset id** (recovered from the PDF text 2026-08-11 — the subplot titles survived extraction even though the data points did not). The paper uses SCBench display names, so grepping the paper for `scbench_kv` finds nothing; it is the panel titled **Retr.KV**. Figure 11 is 3 rows × 4 columns:

  | row (paper category) | panels |
  |---|---|
  | Retrieval | OpenAI MRCR ・ **Retr.KV** (`scbench_kv`) ・ Retr.Prefix-Suffix (`scbench_prefix_suffix`) ・ Code.RepoQA (`scbench_repoqa`) |
  | Contextual QA | SQuAD (`squad`) ・ GSM8K (`gsm`) ・ En.QA (`scbench_qa_eng`) ・ En.MultiChoice (`scbench_choice_eng`) |
  | Redundancy | En.Summary (`scbench_summary`) ・ **Retr.MultiHop** (`scbench_vt`) ・ Math.Find (`scbench_mf`) ・ ICL.ManyShot (`scbench_many_shot`) |

  Two things follow. The y-axes are **absolute** accuracy (Retr.KV's ticks are 0/20/40/60, Code.RepoQA's are Pass@1), so our numbers are comparable to the figure only when reported absolute — never as `results.parse`'s relative rows. And the x-axis is **0.2→1.0**, so any result at ratio 0.1/0.05 has no counterpart in the paper at all.
- **Figure 12 = cross-model generalization**, the same 12 datasets averaged, for Qwen3-8B, Qwen2.5-14B-1M, Gemma3-12B, Qwen3-8B-FP8.
- **Figure 13 = decoding-intensive main result** (§4.3) — AIME24 + MATH on Qwen3-8B/14B, baselines R-KV / SnapKV / early-stopping-of-thinking. A second main battleground, entirely separate from the prefill one. Nothing here has been run.
- **Figure 1(a)** = KVPress benchmark (RULER-4K, Qwen3-8B) vs. Dec-2025 SOTA — a different harness. **Figure 1(b)** + **Figure 10** + **Table 2** = efficiency (prefill/decode latency and memory; gate training cost).

**Gate-design ablations — deliberately NOT reproduced**: Figures 5, 7, 9, 15, 16, 17, 18 (training target scores, gate inputs, gate architecture, loss curves, training data size, sink-key/projection config, local window size). These argue for the authors' gate design. Path B uses the **released gate weights** and does not retrain gates, so they are orthogonal. The one exception: if `load_gate()` is ever swapped for a free-energy scoring signal, Figure 9 (architecture comparison) becomes the relevant reference.

**Analysis, not performance**: Figures 6, 14, 20, Table 3. **Method schematics**: Figures 2, 3, 4, 8, Table 1. **Related-work comparison**: Figure 19 + Table 4 (TrimKV, DMS) — cite, don't run.

**Critical caveat for any "did we reproduce it?" claim**: Figure 11 is a **line plot, and the paper publishes no corresponding numeric table** — not in the body, not in the appendix. `scratch/refs/fastkvzip_paper.txt` is PDF-extracted text, so only axis ticks survive, not data points. **Point-by-point numeric comparison against the paper is therefore impossible with what's on disk.** Only two quantitative claims in the prose can be checked: "maintains full-cache performance at a **30–40%** KV budget ratio" (§4.2; extraction mangles `30~40%` into `30940%`), and "Fast KVzip outperforms all baselines while **matching** KVzip" — note the paper positions KVzip as a tie, not a loss, its selling point being half the prefill cost. Extracting the plotted points would need the original vector PDF, which is not in `external/`.

### Reproduce driver

`scratch_repro_full.py` is the canonical reproduce driver — an 8-GPU job scheduler over the (model × method × dataset) grid, one worker per GPU, resumable. Prefer it over hand-rolled launch scripts; the older `scratch/repro_0730_qwen3/repro_subset.sh` / `scratch/repro_0730_qwen3/wait_and_parse.sh` are superseded.

```bash
.venv/bin/python scratch_repro_full.py --plan     # print job plan + progress, run nothing
.venv/bin/python scratch_repro_full.py --run --models Qwen/Qwen2.5-7B-Instruct-1M \
    --datasets squad scbench_many_shot --methods fastkvzip kvzip
```

Two design points worth knowing before editing it: completion is judged by a **marker file** written only when the log ends in `Finished.` — counting result files breaks because some datasets have fewer than `--num` examples (`scbench_many_shot` has 54, so a count-based check reruns forever). And `kvzip` runs `eval.py` (unchunked prefill) while the other four methods run `eval_chunk.py`; the `METHODS` table carries the per-method script/gate/level triple.

`scratch_fig11_driver.sh` chains the whole Figure-11 job end to end: wait for the main scheduler → run MRCR → parse all 12 datasets. It supersedes `scratch/repro_0731_qwen25/fig11_parse_all.sh` (11 datasets, no MRCR) and `scratch/repro_0731_qwen25/fig11_parse.sh` (3 datasets).

### MRCR — the 12th dataset (wired up 2026-08-03)

MRCR does not go through the normal path and needed real work to attach. What makes it different: its samples never touch `DataWrapper`, grading is a `SequenceMatcher` ratio against an answer that must start with a per-sample `random_string_to_prepend`, results land in `results/mrcr/` and are read by `results/parse_mrcr.py` (output is `{ratio: mean score}` JSON, **not** the row-per-ratio format `results.parse` prints), and it needs dedicated eval scripts. `scratch_repro_full.py:MRCR_SCRIPT` maps the normal script to the MRCR one; `run_job` swaps it when `dataset == "mrcr"`.

**The load-bearing decision — how the KVzip baseline runs.** Upstream ships only `eval_chunk_mrcr.py` (chunked-prefill-evict), and `run.sh` gives exactly one MRCR command, for fastkvzip. The tempting shortcut is `eval_chunk_mrcr.py -g ""`, but that is **wrong**: on the other 11 datasets the KVzip baseline runs `eval.py`, which does *not* chunk-evict — it prefills in full, scores once, then prunes per ratio. Using `-g ""` with the chunked script yields "KVzip scoring + chunked eviction", a configuration that appears nowhere in the paper, and would make the MRCR cell inconsistent with the other 11. So `eval_mrcr.py` (new, local) ports `eval.py`'s unchunked flow onto MRCR data. Three source facts make that port valid, all verified by reading the code:

- `model.prefill(do_score=True)` with `gates is None` dispatches to `self.scoring()` — KVzip's context-reconstruction scoring (`model/wrapper.py`).
- `RetainCache.prune` (`attention/kvcache.py:304`) recomputes the mask from the **original** `self.score` every call rather than pruning cumulatively, so the descending ratio sequence is safe on one cache. (`EvictCache.prune` at line 148 *does* delete physically and cannot be called repeatedly — `eval.py` avoids it by forcing `kv_type="retain"`, and `eval_mrcr.py` does the same.)
- `model.generate` defaults to `update_cache=False` and ends with `kv.slice(seen_token_prev)`, rolling back the KV added during generation — so one cache can be generated from six times.

**Two upstream bugs had to be patched** in `eval_chunk_mrcr.py`, both fatal to a batch run: it ignored `--num` and looped `range(args.idx, len(dataset))` — all 2400 samples × 6 ratios ≈ **300 GPU-hours** (`args.py` has defined `--num` all along and every other eval script honours it); and it never printed `Finished.`, which is exactly the string the scheduler uses to mark a job complete, so MRCR would have been judged failed and rerun forever.

Verified statically before launch: scheduler-derived result paths match what all 5 method/script combinations actually write, byte for byte; `parse_mrcr.py`'s glob `*_{model}` does not swallow other methods' directories when kvzip's tag is empty; the dataset loads on CPU with all required fields (`prompt`/`query`/`answer`/`random_string_to_prepend`/`n_tokens`, first sample 26,499 tokens, no auth needed). **Not yet verified: an actual GPU run** — all 8 GPUs were busy with the main sweep and squeezing in a smoke test risked OOM-ing 40 live jobs.

## Stage 2a — real corpus, before touching the harness (2026-08-07)

`scratch_stage2a.py` + `varikv/realdata.py` (+ `stage2_cache/`, cached tokenised shards). Long-context language modelling on fineweb-edu documents, keeping `varikv`'s own regular per-token layout and changing **only the data**. The point of the intermediate step: a full per-head harness integration mixes two independent risks (does the method work on real text / is the varlen port correct), and doing both at once makes a failure undiagnosable.

Task: prefill a document (triggering eviction + absorption), score nll on the **held-out last `target_len` tokens** — the standard evaluation for Infini-attention-class compressive memories.

**The trap this exposed, recorded in `realdata.py`'s docstring:** stage 1 truncates the context *tail* because its needle is at the front, whereas real-text LM must truncate the *head* — the text adjacent to the target is the strongest predictor, and truncating it destroys the task.

**Result (60 samples, K=16) — negative, and it inverts stage 1's headline:**

| tier | nll | |
|---|---|---|
| **t1 discard** (sliding window) | **2.3031** | best |
| t5 dist + **random** evict | 2.9078 | |
| t5 dist (free-energy evict) | 2.9283 | |
| t2 point | 3.0429 | |
| t4 fe+point **random** | 3.1306 | |
| t4 fe+point | 3.1615 | |
| t3 MomentKV | 3.3274 | |

Two things follow. **Discarding beats absorbing on real text** — the opposite sign of stage 1's "88% of the gain is don't-throw-it-away", so that finding does not transfer off the synthetic needle task. And **random eviction again beats free-energy eviction** (both at t4 and t5), independently replicating stage 1's finding that these tasks carry no eviction signal — this time on natural language. The random-eviction control was built into this driver from the start precisely because of the stage-1 lesson.

## Stage 2b — VariKV wired into Fast KVzip (2026-08-07)

`external/FastKVzip/prefill/attention/memcache.py` — **`MemoryEvictCache(EvictCache)`**, the first time this project's method touches the real benchmark harness. Subclasses rather than reimplements, so the per-head varlen layout (`[Σ_heads len_k_head, dim]` + `cu_len_k`) and the AdaKV kernel binding come for free. Overrides `_sample_cache` (the discard anchor), `_get_valid` (mask offset), `update`/`slice` (position tracking), `init_score` (guard), `prune*` (budget).

Per-head segment invariant: `[ mem_0 … mem_{M-1} | real_0 … real_{L_h-1} ]`. The memory prefix **must** sit at the front — `update_flatten_view` appends new KV to each head segment's *tail*, so a trailing prefix would be split apart on the next update. Front placement also makes the memory the "oldest" keys, which is what `flash_attn_varlen_func(causal=True)`'s bottom-right alignment requires for them to be always-attendable without leaking future tokens.

**Framing that fell out of the integration and is better than the original plan:** don't replace their eviction scoring, only take over *what happens to what they throw away*. **VariKV = FastKVzip + absorption.** Same model, same data, same eviction decisions; the only variable is discard vs absorb. That isolates the half of the method that shows signal, and sidesteps the half that no task so far can adjudicate.

### Bugs found in this integration — all silent-failure class

| # | Bug | Why it matters |
|---|---|---|
| 1 | **`snap` gate is incompatible** | It scores *after* `update()`, on the returned flattened cache — which now contains our memory prefix. Index/length misalignment, no error raised. Guarded with `assert_gate_compatible`. |
| 2 | **`prefill` is `@torch.inference_mode()`** | Inference tensors can *never* enter autograd, even after leaving the context — training was impossible. Split into `prefill`/`_prefill_impl` with a `varikv_train` flag; default path unchanged. |
| 3 | **`slice()` not overridden** | The harness generates once per question and rolls back each time; `pos_track` grew monotonically, so RoPE inverse rotation eventually used wrong positions — the exact failure `rope.py` exists to prevent. |
| 4 | **Memory KV are extra cache and must be accounted for** | `threshold()` sizes the budget as `ratio × len(score)` and `score` excludes memory, so at a given nominal ratio we use `r·ctx + M·H·L` KV against the baseline's `r·ctx`. Resolved by *measuring* rather than by surgery — see below. |
| 5 | `_swap_out` used in-place index assignment | Severs autograd. Rebuilt with `torch.cat`. |
| 6 | dtype mismatch (fp32 memory × bf16 model) | Hard error in the encoder matmul. |
| 7 | `expected_attn=None` | ELBO reconstruction was unweighted, inconsistent with `F_i`'s distortion definition. Now fed from the gate scores — `score[layer]` is `[1,H,ctx_len]` indexed by **absolute position**, which `pos_track` supplies. |

**Bug 4 took two failed fixes before landing on "don't fix it, measure it". Both failures are worth remembering.**

- *v1 — scale `ratio` down by `M/ctx_len`.* Looks obviously right, isn't: `adakv-layer`'s safeguard redistributes budget across layers and `threshold` selects by a score cutoff rather than an exact count, so ratio→retained is nonlinear. Measured error **+0.29% to −16.85%**.
- *v2 — demote the M lowest-scoring real KV once per layer at the first prune.* Exact on 1.5B (H=2) + `adakv-layer`, and it required also writing the demotion back into `self.valid[l][h]`, since `prune_chunk` rebuilds the ledger from it afterwards. Still abandoned: doing surgery inside upstream's `valid`↔`len_k` bookkeeping is fragile, and the payoff is small.
- *Landed — `measured_kv()`, no surgery.* The overhead is a **constant `M·H·L`**, exactly predictable and now measured. Report it and put actual cache size on the x-axis. At M=16 it is +0.14% (ratio 0.3) / +0.38% (ratio 0.1); at **M=256 it becomes 28,672 entries ≈ 6% at ratio 0.1**, where using nominal ratio as the x-axis would visibly inflate results.

**A measurement artifact cost more time than any of the real bugs — do not repeat it.** `level="pair"` thresholds **globally across all layers and heads**, and turning memory on changes hidden states → gate scores → how budget is distributed across layers. Measuring **layer 0 only** therefore showed "−28%" then "−38%" and looked like a catastrophic regression; the total across all 28 layers was conserved to +0.14%. With `adakv-layer` (uniform per-layer budget) layer 0 *is* representative, which is why 1.5B never showed it. **Always sum over all layers when checking cache size.**

**The integration acceptance checks → `JOURNAL.md`** (1.5B with the `expect` gate, then the
target 7B with the real gate on a 169k-token `scbench_kv` sample). What they establish and you
need not re-verify: absorption disabled ⇒ output byte-identical to native `EvictCache`; the
`M·H·L` budget overhead is exactly as predicted (+1,788 measured against 1,792 predicted);
`retain` vs `evict` produce byte-identical generations, so building on `EvictCache` is safe.
One upstream quirk to avoid: at `chunk_ratio=1.0` generation returns `''` for *every* config
including native — do not use ratio 1.0 as a sanity baseline.

### Training inside the harness — two more silent failures

`scratch_stage2b_train.py` trains the memory through **the same path evaluation uses** (real gate, per-head eviction, `MemoryEvictCache`), so there is no train/test layout mismatch. Getting it to work needed two fixes beyond the `inference_mode` one:

1. **OOM at 78.2 GB.** The LLM is frozen, but read-out KV participate in every later forward, so the graph spanned all 5 chunks of a 7B model. Fixed with truncated BPTT in a form specific to this design: `detach_readback=True` makes prefill insert **detached** memory KV (so the prefill forwards build no graph at all), and `refresh_with_grad()` afterwards re-inserts one grad-carrying read-out. Gradient then flows `loss → target forward → memory KV → decoder → memory state → encoder`. **30.7 GB, 1.6 s/step** at 8k context. Cost: loses the second-order effect of early read-outs on later hidden states — the standard truncation trade-off.
2. **`update_flatten_view` has no backward.** It is AdaKV's custom CUDA op; memory KV pass through it during the target forward and `grad_fn` is silently severed. Symptom is the dangerous kind: **loss falls normally while `|grad|max` is exactly 0.00e+00** — it looks like training and learns nothing. `MemoryEvictCache._flat_insert` is a pure-PyTorch equivalent, used only when `torch.is_grad_enabled()` and the cache already requires grad, so inference keeps the fast kernel. After the fix `|grad|max ≈ 7.9e-02`.

**Always assert a nonzero gradient norm before trusting a training run in this harness** — both failures above are invisible in the loss curve.

### Training data: use upstream's loader verbatim

FastKVzip's gate is trained on **FineWeb-Edu** (§3.3, appendix A.1), and `data/load.py:load_fineweb` is that exact loader — the paper's spec is recoverable from the code:

| paper A.1 | `load_fineweb` |
|---|---|
| "10K to 30K tokens" | `min_len, max_len = 10000, 30000` |
| "concatenating … yielding 100K" | the `-cat` variant accumulates to 100k |
| "a total of 1M tokens" | `if total > 10**6: break` |

Selection is **deterministic** (`np.arange` + length filter, no seed), so the same documents are reproducible byte for byte — verified by hashing two independent loads.

**The gate training code IS released, in the repo root — not under `prefill/`.** `train_gate.sh` → `prefill/feature.py` (dump hidden states + KVzip scores) → `optim.py` (fit the gates). `feature.py` hard-codes the exact composition:

```python
folders = [("fineweb_10k", 29), ("fineweb_10k_cat", 5)]   # first 29 docs + first 5 docs
```

Measured with the Qwen2.5-7B-1M tokenizer: **29 docs → 434.9K tokens** (10,444–26,492) plus **5 docs → 547.6K** (103,372–119,319) = **0.98M**, matching the paper's "1M training tokens" (§3.3) and "500K + 500K" (A.1). **Paper and code agree** — an earlier note here claimed they conflicted because `load_fineweb` caps each variant at `total > 10**6`; that cap only bounds what the *loader returns*, while training consumes only the first 29 / first 5. Use `load_fineweb(name)[:29]` and `[:5]`, not a token-budget rule (a "500K from each" heuristic picks 39 docs, not 34).

The paper states *why* FineWeb-Edu: it has **no overlap with the downstream evaluation datasets**. So "train on generic web text, evaluate on retrieval tasks" is a deliberate design choice against test-distribution leakage, **not** a distribution-shift defect — do not "fix" it by training on SCBench.

**`load_fineweb` was broken in this environment and had never been run here**: `samples[i]` with `i` a `numpy.int64` raises `TypeError` under datasets 3.6.0. Patched with `i = int(i)` (same class of version-skew as the pyarrow fallback above).

**What is *not* worth copying from FastKVzip's training setup:** its lr 0.2 / 5K steps / batch 1K / BCE loss are tuned for **per-layer binary gates distilling KVzip's reconstruction scores** — a different module with a different objective from our encoder/decoder memory trained on LM loss. Align the **data distribution and the eviction environment** (chunk 16000, window 4096, ratio, context lengths); do not transplant the optimiser hyperparameters.

**The first real-benchmark result (2026-08-08, `scbench_many_shot`, −15…−25 points at every
ratio) → `JOURNAL.md`.** It carries a train/eval config confound and is superseded, but two
measurement traps from it stand: a run that writes into an **existing** result directory gets
averaged together with the stale samples by `results.parse` (always use a distinct `--tag`),
and a leading underscore in `--tag` produces a **double** underscore in the directory name, so
a hand-built `-m` string silently parses 0 samples and raises `ZeroDivisionError`.

### Known limits — do not mistake these for working

- **`prune()` (unchunked) leaves the model generating empty strings — but so does native `EvictCache`.** Verified identical across `evict` / `memory` / `memory+absorb`. This is the upstream path `eval.py` avoids by forcing `kv_type="retain"`; `eval_chunk.py` uses `prune_chunk` and is unaffected. Not our bug, not fixed.
- **`batch>1` unsupported** (guarded). Position tracking, per-layer state slicing and absorb padding all assume B=1.
- ~~**The memory is untrained for any target model.**~~ Superseded: ten 7B checkpoints exist (inventory in `MODELS.md`). Trained memory does not fix the degradation — under KV injection it stays 30–45 points down, and under the residual read-out it only reaches parity by closing its own gate.
- **Capacity is the open design question.** Measured at ratio 0.3: 4,080 retained real KV per head vs M=16 memory ⇒ **0.39% of visible KV**, summarising ~13k evicted tokens at 800:1. A `<3%` share now prints an automatic warning. Raising `num_slots` is the obvious move, but stage1 measured *larger K is worse* (at a very different scale), so it needs re-sweeping — and now that the budget is charged honestly, a larger K costs real retained KV.

### Local modifications to the vendored FastKVzip clone

**The vendoring situation changed on 2026-08-11 — read this before editing anything under `external/`.** The clones are now **committed into this repo** (commit `9914cc1`, 237 files, 9.2 MB), by the repo owner's decision, and both clones' own `.git` directories were **deleted**. Consequences:

- **Upstream diff and pull are no longer possible.** The upstream commits at the moment of deletion were FastKVzip `e04afaa`, KVzip `5d84729`, recorded in `patches/fastkvzip_upstream_commit.txt`.
- **`patches/fastkvzip_local.patch`** (176 added / 12 removed across 9 files) is the mechanically-replayable record of our edits to *upstream-tracked* files. **It is incomplete by construction**: the three files we *added* (`attention/memcache.py`, `attention/memcache_retain.py`, `eval_mrcr.py`) were untracked in the clone's git, so `git diff` never saw them. They are ordinary files in this repo now, so they are safe — but do not treat the patch as a full inventory.
- **`patches/kvzip_local.patch` contains no code changes.** Its 178 lines are entirely the deletion of KVzip's own `.gitignore`; the KVzip clone has never been modified. (An earlier note claimed KVzip carried 178 lines of local modifications — that was a misreading of the patch.)
- Both clones' own `.gitignore` files were deleted too. Harmless: this repo's root `.gitignore` still excludes `results/`, `*.pt` and `__pycache__/` recursively, which is why the 223 MB of eval outputs under `external/FastKVzip/prefill/results/` stayed out of the commit.
- **The workflow is now "edit in place and commit".** There is no re-clone scenario to re-apply patches to. If you change an upstream file, refresh the patch (`cd external/FastKVzip && git diff` no longer works — regenerate by diffing against a fresh upstream clone in a temp dir, or just rely on this repo's history).
- FastKVzip still ships no LICENSE and is still a preprint (arXiv 2601.17668). It is now pushed to `github.com/zengxyyu/VariKV` on `main`; that was the owner's call and is not to be re-litigated.

Local edits, for orientation:

| File | Change |
|---|---|
| `attention/attn.py` | **the residual read-out hook** — stashes the pre-`prepare` post-RoPE query as `_varikv_q` when `residual_mode`, then `attn_output += past_key_value.memory_residual(_varikv_q, layer_idx)` at line 149. **Called unconditionally**, with no "nothing absorbed yet" guard — this is the empty-memory injection documented in the 2026-08-11 section. Was missing from this table until 2026-08-11 |
| `data/load.py` | pyarrow fallback for `scbench_mf_mid` and `squad` (datasets 4.x wrote their parquet; `Feature type 'List' not found` under datasets 3.6.0); `load_fineweb` takes `int(i)` (numpy.int64 indexing broke under datasets 3.6.0) |
| `eval_chunk_mrcr.py` | honour `--num`; print `Finished.` |
| `eval_mrcr.py` | **new file** — unchunked MRCR eval for the KVzip baseline |
| `attention/memcache.py` | **new file** — `MemoryEvictCache`, the VariKV integration (Stage 2b) |
| `attention/memcache_retain.py` | **new file** — `MemoryRetainCache`, the same method built on `RetainCache` instead. This is what the baselines actually run (`args.py` default), and it is *simpler*: the cache stays rectangular `[B,H,M+seq,dim]`, so "original position = sequence index" and none of the `pos_track` machinery is needed; `self.valid` is cumulative, so the newly-evicted set slices straight out of `evict_range` with no double-absorption risk. Now the default of `--varikv_kv_type` |
| `model/wrapper.py` | `kv_type="memory"` and `"memory_retain"` dispatch; `chunk_ratio` branch now excludes `"memory"` alongside `"evict"` (both use nested-list `valid`, not a tensor); `prefill` split into `prefill`/`_prefill_impl` so `varikv_train` can bypass `inference_mode` |
| `args.py` | five VariKV flags: `--varikv_ckpt` (giving it enables the memory), `--varikv_slots`, `--varikv_kv_type`, `--varikv_readout {normal,zero}` (the zero-read-out ablation), `--varikv_residual` |
| `eval_chunk.py` | builds the memory from the ckpt and derives the tag. **`n_groups` must be passed when the ckpt is residual**, or loading dies on `Unexpected key(s): residual_gate` |
| `eval.py`, `results/parse.py` | both gained a `set_ratios()` honouring `VARIKV_RATIOS`. **They must agree** — see the parse trap in the 2026-08-10 section |

### Reproduce status — summary

**Full narrative, per-dataset tables and caveats → `JOURNAL.md`.** Figure 11's full sweep
(Qwen2.5-7B-1M × 5 methods × 12 datasets) **completed 2026-08-04**, 15.5 h on 8×H100, 0 failures;
raw tables in `scratch_fig11_full_results.log`. It is **not** a point-by-point reproduction and
cannot be — the paper publishes no numeric table for Figure 11, and the local copy is
PDF-extracted text with the data points lost. What is verified is threshold behaviour
(fastkvzip 101.64 / 100.54 relative at ratio 0.4 / 0.3, i.e. the paper's "near-lossless at a
30–40% budget") and method ordering, including snapkv's collapse.

Two standing caveats: **`scbench_prefix_suffix` is very noisy at n=100** and non-monotone even
for the winning method — dropping it turns an apparent +5.75 fastkvzip-over-kvzip gap into a
tie, which is what the paper claims, so never build an argument on that one dataset. And
**stray `results/` directories with ~2 entries are timing probes, not reproductions.**

## Stage 1 — the GO/NO-GO variance ablation

`stage1/data.py` builds the synthetic dataset for the make-or-break experiment (distributional vs. point absorption of evicted KV). It is standalone — no dependency on `memory_module.py` or `external/`.

The design is load-bearing, not arbitrary: samples come in **two kinds that must both be present**. `retain` (fact appears early, only same-format distractors after it → tests low KL ⇒ *don't* update ⇒ resist washout) and `update` (the fact is genuinely rewritten mid-context → tests high KL ⇒ *do* update). A fixed-rate point memory cannot be optimal on both at once; testing only one kind lets a tuned fixed rate tie, i.e. a false negative. The `n_distract` axis (0/200/800/2000) is the predicted-effect-size knob: advantage should grow with distractor count and vanish near 0.

```bash
.venv/bin/python stage1/data.py          # print n_distract → token-length table only
.venv/bin/python stage1/data.py build    # write stage1/{train,val}.jsonl
```

`build` emits 3200 train / 400 val over the 4 distractor levels. Measured context lengths: `n_distract` 0 → 109 tokens, 200 → 3,526, 800 → 13,817, 2000 → 34,357. Neither jsonl is committed — regenerate as needed (seeds are fixed: train 0, val 1234).

**The results, the broken-evaluator diagnosis and the free-energy-eviction post-mortem →
`JOURNAL.md`.** Four conclusions from that round that still bind:

- **nll on the answer tokens is the metric, not exact match.** EM is identical across all five
  tiers and comes entirely from the `nd=0` bucket where nothing is evicted; at 377:1–4231:1
  compression of high-entropy random-string answers it has no resolution.
- **88% of the benefit is "don't throw it away"** (t1→t2, −1.44 nll), which Infini-attention
  established in 2024. The project's own two contributions split 2% / 9%.
- **Larger K is worse** — every trained tier degrades monotonically 16→64. The "capacity is too
  small to show an effect" escape hatch is closed on this task.
- **Stage 1 cannot adjudicate eviction at all**: random eviction beats every principled
  criterion including published Expected Attention, so nothing here licenses a claim that
  free-energy eviction is bad — only that this task cannot tell. Published counterpart:
  Error Certificates (2607.21475), which proves deterministic top-k eviction error is
  unidentifiable. Cite it rather than apologising for that row.

## `varikv/` — the method implementation (written 2026-08-03)

VariKV — variational free-energy eviction. Implements the §11 "Option 3 unified free-energy eviction" method. Standalone — deliberately **not** wired into Fast KVzip yet (see "deviations" below).

The four-tier killer ablation (§11.7) is **two orthogonal switches, not four codepaths** — `Config.ablation(tier)` sets them:

| tier | `evict_policy` | `absorb_mode` | degenerates to | trains? |
|---|---|---|---|---|
| 1 | `recency` | `discard` | **sliding window — NOT KVzip, see below** | no |
| 2 | `recency` | `point` | Infini-attention / Tensor Cache | yes |
| 3 | `recency` | `moment` | **MomentKV** (second moments) | **no — training-free** |
| 4 | `free_energy` | `point` | IndexMem-like | yes |
| 5 | `free_energy` | `dist` | **VariKV** | yes |

**Tier 1 is mislabelled everywhere and must not be reported as "KVzip" (flagged 2026-08-07).** `config.py:215` sets it to `("recency", "discard")` with the comment *"丢弃：驱逐策略无关"* — **that comment is wrong**. When nothing is absorbed, *which* KV you keep is not irrelevant, it is the only thing that determines performance. So tier 1 as implemented is a **recency / sliding-window baseline**, the weakest eviction rule there is. Real KVzip scores by reconstruction attention and is far stronger.

Consequence: every "Δ vs tier 1" number in the sweep measures the gap to a sliding window, **not** to KVzip. That is fine for stage 1 — `HANDOFF.md` deliberately starts with simple recency eviction to isolate "distributional vs point absorption" — but two things follow. (1) Do not print `discard(KVzip)` in report tables (`scratch_stage1_sweep_report.py:NAME`) or write "beats KVzip" in the paper on the strength of this tier. (2) Any real claim against KVzip/FastKVzip requires wiring in their actual scoring — that is the Stage-2 Path-B integration, not something tier 1 stands in for.

Tier 3 was added 2026-08-03 after the literature sweep: MomentKV already keeps count + key mean + value mean + value-key covariance *without training*, so "beat a point-mean memory" no longer proves anything. **The decisive comparison is 3 vs 5** — both hold second-order information, so the independent variable collapses to "Bayesian belief with KL gating and variance-aware read-out" vs "frequentist moment statistics". `varikv/moment.py` is an approximate reimplementation under this repo's architecture (its query-dependent first-order correction `C·q/√d` cannot be expressed as a static effective KV); the paper must run the official implementation, not this.

The 5th row of the §11.3 degeneracy table (drop the KL term → Expected Attention) needs no new tier: set `free_energy.lam = 0`. **This degeneracy is empirically verified**, not just asserted — at λ=0 the rank correlation of F with D is 1.000 and with KL is 0.056. Sweeping λ traces the rate-distortion working points (0.3 balances the two terms, hence the default; 3.0 makes KL dominate). That sweep is a ready-made sensitivity analysis for the paper.

```bash
.venv/bin/python varikv/train.py --tier 5            # then 2, 4 (1 and 3 need no training)
.venv/bin/python varikv/evaluate.py --tier 1 2 3 4 5
```

`scratch_stage1_driver.sh` chains the whole thing: wait for the Figure 11 sweep to free the GPUs → train tiers 2/4/5 in parallel on three cards → evaluate all five. Results land in `scratch_stage1_results.log`, per-tier logs in `scratch_stage1_logs/`.

**Samples shorter than the budget never trigger eviction**, so memory never participates and the loss has no `grad_fn` at all — `train.py` skips them (in stage1, `n_distract=0` is only 109 tokens, a quarter of the data). Evaluation keeps them, as a check that memory does not *harm* contexts that already fit.

Files: `config.py` (two switches + all hyperparameters), `memory.py` (slots, precision-weighted update, read-out, and `residual_gate` for the output-side path), `free_energy.py` (F_i, expected attention, amortised predictor), `cache.py` (chunked prefill → evict → absorb → read back), `rope.py` (inverse/forward rotation — see below), `realdata.py` (Stage 2a: fineweb-edu long-document loader + cache), `moment.py` (tier-3 MomentKV), `train.py`, `evaluate.py`.

~~`point` and `dist` share identical structure and parameter count and run the same
precision-weighted update — the only difference is whether the precision term carries
information.~~ **Wrong, corrected 2026-08-12 by reading `memory.py:266-322`.** Parameter
count is equal, but **four things change at once**:

| | `dist` | `point` |
|---|---|---|
| observation precision | `τ_obs = exp(−logvar_q)` | `1` |
| slot overwrite resistance | `τ_old = exp(−logvar)` | `1` |
| **write strength** | **`η_i = σ(α·z(KL_i) − β)`, content-dependent** | **`σ(scalar)`, a learned constant** |
| read-out | decoder sees `logvar`; `sample_on_read` active | mean only |

The third row is the dangerous one: it is **surprise-gated writing, not distributional
representation**. So `dist − point` is *not* a measurement of "does variance help" — it
bundles an independent mechanism. **Any claim that the distributional memory beats the
point memory requires a surgical ablation** (disable the logvar read / the precision
terms / the KL-gated `η` one at a time). The teacher-KL round's apparent `dist` 54.20 vs
`point` 14.60 was **not** such evidence: sampling was unmatched, both v2 reruns are
unseparated, and as of 2026-08-14 the 54.20 itself is known to be one draw of a
39-point-wide training-run distribution.

### Bugs that only surface at real scale — do not "simplify" these back

Each of these let the code run, the loss fall, and numbers come out, while **silently not running the method described in the paper**. All were invisible on a tiny random model and only appeared at `d_latent=64`, `K=16`, real KV.

1. **`logvar_init` must be ≥0, not −2.** An empty memory has absorbed nothing and must be *uncertain*; it becomes certain (and thus overwrite-resistant) only as precision accumulates. Starting at −2 both makes the empty memory refuse its first write and inflates initial KL by 1/σ_p² to ~10³.
2. **Precision needs a forgetting factor** (`precision_decay=0.95`). `τ_new = τ_old + Σηᵢτ_obs` grows without bound in a streaming setting → memory becomes so confident that μ stops updating → it rejects all new information, which fails every `update`-type sample by construction.
3. **The write gate must use a chunk-wise z-score of KL, not raw KL.** Raw KL spans ~4 orders of magnitude as memory evolves (0.05 → 1589 measured), so any fixed (α, β) saturates: gate ≡ 0.12 early, ≡ 1.00 later, std ≈ 0 at both ends. The gate never operates in its sensitive band and `dist` silently degrades to unconditional full writing. After the fix, low- vs high-surprise write rates are 0.047 vs 0.994. Note `eta_beta` is consequently **0.0**, not the 2.0 that suited raw KL.
4. **The write gate must be probability-normalised: `gate_ik = w_ik · η_i`.** Folding allocation and strength into one sigmoid let the row sum reach **K** (measured 0.66–16.0) — one observation writing at full strength into *every* slot, i.e. the same information counted K times. That breaks "independent observations have additive precision", the premise the whole Bayesian update rests on. Decoupled: `η_i` (scalar, how much this observation writes in total) × `w_ik` (softmax allocation, sums to 1) ⇒ row sum = η ≤ 1, with discrimination preserved (0.03 vs 0.99).
5. **Memory keys must be stored pre-RoPE.** See the dedicated section below — this one threatens the method's core claim, not just its accuracy.
6. **The write gate's precision must be capped by effective sample size** (`max_eff_obs=1.0`). A chunk evicts hundreds of *adjacent, highly correlated* KV entries; treating them as that many independent observations is textbook pseudo-replication. Measured: each slot accumulated ≈8 per absorb, steady state `τ = 8/(1−γ) = 160`, so `1/τ ≈ 0.006` and the variance was crushed against the clamp. Scaling first and second moments together caps precision growth without touching the mean.
7. **Slot variance must include content dispersion, not just `1/τ`.** Using only `1/τ` asserts "more observations ⇒ more certain", but a slot summarising hundreds of *heterogeneous* tokens should be **less** reliable, not more. Variance is now `1/τ + Var_content`, tracked by first/second-moment recursion. Verified: coherent content → logvar −3.05, dispersed content → −2.53.
8. **`logvar` from the network needs a soft bound, not a hard clamp.** `to_logvar` is trainable and the ELBO reconstruction term keeps pushing it down; a hard clamp has exactly zero gradient at the boundary, so it welds shut. Fix: `lo + (hi−lo)·sigmoid(raw)`, with the bias initialised via logit so the initial output is still `logvar_init`.

Bugs 6–8 all manifest as the same symptom — **variance collapse** — which would silently produce a false negative in the make-or-break experiment, since `dist`'s entire advantage over `point` lives in the variance. Measured progression of "fraction of slots pinned at the `logvar` lower bound" after 60 training steps: **99.3% → 98.3% → 81.4% → 0.0%**. Training also improved as a side effect (`lm_loss` after 60 steps: 4.42 → 3.82). If a future change makes variance collapse again, these three are where to look.

9. **Normalising F by scale alone is not enough — it must be by *spread*.** Ranking is driven by dispersion, not by means. After dimensional normalisation alone, `std(D_n)` ≈ 0.69 constant while `std(KL_n)` grows 2e-4 → 7e-2, so F's ranking was 99% determined by D (rank corr with KL only 0.09) even though KL's *mean* had long overtaken D's. That is exactly the Expected-Attention degeneracy. Fix: divide each term by its own **running** std (`f_normalize="running"`). Running stats are a dataset-level quantity, so unlike a per-chunk z-score they keep `F_i` a function of `KV_i` and the memory state, preserving λ's meaning as the rate-distortion Lagrange multiplier.

General lesson: eviction only ever uses the **ranking** of F, so any monotone transform is free — that is what licenses all the rescaling above. But *which* statistic you normalise by matters: per-chunk z-scoring is numerically safest yet forfeits F's absolute semantics, while running stats keep them at the cost of some lag (hence `v_scale_momentum=0.9` as a compromise; 0.95 lets KL dominate, 0.5 balances best but collapses back toward per-batch statistics).

### RoPE: memory keys must be stored pre-RoPE (fixed 2026-08-03)

This is the most consequential fix so far, and its consequence is **semantic, not just numerical**.

Cached keys are post-RoPE (verified in the vendored source: `prefill/attention/attn.py:54` applies `apply_rotary_pos_emb` before `update()` at line 81). A memory slot is a precision-weighted *average* of absorbed keys, and RoPE does not distribute over addition:

> `α·R_p k + (1−α)·R_p' k' ≠ R_φ(α·k + (1−α)·k')`  — MemRoPE, arXiv 2603.12513

So a slot built from post-RoPE keys **is not a valid key at any position**. Measured consequence: the same slot's inner product with a query swung between −17 and +13 as query position went 0 → 16384, with no pattern. And an *untrained* memory was already flipping the top-1 prediction on 5 of 7 answer tokens — it was winning attention through unconstrained logit magnitude, not content.

The deeper problem: mixing phases **inflates σ² for reasons unrelated to epistemic uncertainty**. Without this fix, the variance is not measuring what the paper claims it measures, and HANDOFF red line 1 ("variance must be functional") fails — a reviewer could fairly call it Bayesian paint over phase noise.

Fix (`varikv/rope.py`, the standard solution in the literature): inverse-rotate each evicted key by `R_p^{-1}` back to a position-free frame before absorbing, and re-rotate at read time to the slot's **position centroid** (the UPL optimum in EPL, arXiv 2409.14364; same rule derived independently in MemoSight). Because `R(δ)R(p) = R(p+δ)`, re-rotation is pure algebra — no extra forward pass. Precedents: Still (2606.07878 §2.2) "position-free frame" — **read the whole of that paper, not just §2.2; the 2026-08-11 sweep found it is the closest published relative of this entire project**, Landmark Attention (2305.16300), StreamingLLM (2309.17453). Verified: our rotation matches HF's `apply_rotary_pos_emb` exactly (0.00e+00), round-trip error 2.4e-7, float positions supported (centroids are not integers). Side effect: tier 2's `lm_loss` went from *above* tier 1 (5.574 vs 4.965) to *below* it (4.174 vs 4.381) — untrained memory stopped being a pure noise source.

Note FastKVzip already ships a different, also-valid answer in its `expect` gate (`prefill/attention/baseline.py:143-161`): marginalise the rotation matrix over future query positions, `R̄ = (1/T)Σ_j R_{t+j}`, applied to both mean and covariance. It is simultaneously a fix, a ready-made baseline, and our nearest methodological neighbour (it too is a Gaussian-over-queries method).

### The BPTT gradient path — counterintuitive, verified

**A prior worry that turned out to be wrong; do not "re-fix" it.** The concern was that `stage1`'s needle sits in the first 0.2–2% of the context, gets absorbed in chunk 0, and would therefore lose its gradient to `truncate_bptt`. Measured: not so — the mu right after the *first* absorption receives gradient under `truncate_bptt` ∈ {0, 2, 4} alike.

The reason is that `lm_loss` reaches memory through exactly one path: **the read-out effective KV is written into the cache and participates in every subsequent forward pass**. Turning `read_memory` off leaves `lm_loss` with no `grad_fn` at all. That path never goes through `self.mu`'s cross-chunk recursion, so `detach_state()` cannot sever it.

`truncate_bptt=0` is still the default — recurrence being trainable is part of "memory is a recurrent state, not a parameter" — but memory is bounded by `max_train_context=4096` instead. Measured: a 13.8k context without truncation OOMs at ~57 GB; 4k is safe. Since the fixed RoPE makes memory position-independent, train-short/infer-long is sound (standard for Infini-attention-style methods).

### Deviations from the design docs — know these before comparing to §11

- **per-token eviction, not per-head.** A token's KV is kept or demoted across all layers/heads together, so the cache stays a rectangular tensor. Per-head eviction is what forces FastKVzip's `[Σ_heads len_k_head, dim]` + `cu_seqlens_k` layout and a custom kernel; adopting it before the method is validated would entangle "is the method right" with "is the layout right".
- **`D_i` only accounts for the v-perturbation** (`ā²·‖v−v̂‖²`); k's perturbation affects output only at second order through the softmax and is dropped.
- **Query statistics come from layer 0's `q_proj` only**, used as a proxy for all layers, to avoid a full extra forward pass.
- **Not connected to Fast KVzip.** Migration is a layout-adaptation job; the memory module and F predictor carry over unchanged.

These are engineering simplifications that do not change whether the method is faithful to the theory. Still open on the theory side:

- ~~The mixture ELBO is missing the component-assignment KL.~~ **Already closed** in commit `7e82a7d` ("complete mixture ELBO") — `memory.py:202` adds `log K − H(w)` and `config.py:72 include_assignment_kl` defaults to **True**. This entry said otherwise until 2026-08-07; do not re-implement it. (The `Σ_k w_k KL(q‖p_k)` bound itself is correct — verified by Monte Carlo: true 6.694 ≤ bound 8.535.)

  **But the term is now a live suspect for the "larger K is worse" result.** Minimising `log K − H(w)` *maximises* the entropy of the responsibilities, i.e. pushes assignment to spread evenly across slots. With K=64 that means every observation writes a little into all 64 slots, so each slot gets a blurry, weakly-differentiated update — exactly the degradation measured (t2 +0.33, t4 +0.64, t5 +0.44 going K=16→64). The config comment already anticipates this ("留开关是因为它会改变行为，需要能做对照消融"). The ablation to run is `include_assignment_kl=False` at K=16 vs 64.
- **"Bayesian surprise" is used loosely.** The code computes `KL(q(z|e)‖p(z|M))` — how much an observation disagrees with memory. Itti & Baldi's Bayesian surprise is `KL(posterior‖prior)` — how much memory *changed*. Related but not the same; be precise when citing.
- **Memory capacity may be too small to show an effect.** Compression ratio 377:1 (3.5k context) to **4231:1** (34k), and the read-out contributes only `K·T` = 16 effective KV against a budget of 512 — **3% of visible KV**. Raising `K` costs almost no parameters (slot inits are only `K×d_z`), so this is a cheap knob if the tiers come out indistinguishable.
- **bf16 numerics.** Precision accumulation showed 5.98% relative error in bf16 (values are large, mantissa is 8 bits), and `logvar` was observed pinned at the −4 clamp where its gradient is zero — monitor the fraction of slots at the boundary.

Two faithfulness gaps *were* closed on 2026-08-03:

- **ELBO reconstruction now lives in attention-output space** (closed 2026-08-03). It was `‖ê−e‖²` in KV space while `F_i` used `ā²·‖v−v̂‖²` — so "free energy" named two quantities with *different distortion definitions*, and §11.1's "one scalar unifying both decisions" was merely nominal. `absorb()` now takes `expected_attn` and weights reconstruction by `N·ā`, matching `F_i`'s distortion and realising rate-distortion's "spend bits by importance". Verified: identical to unweighted when attention is uniform (correct — no importance signal available), diverging up to ~8% as it sharpens. Side benefit: the aux loss dropped from ~90 to ~8–29, since unattended KV no longer inflates reconstruction error.
- **The F predictor now sees a memory summary** (closed 2026-08-03). Its target contains `KL_i`, which depends on the current slots, so without memory state it could only learn an average-case KL. Feeding it exact information is self-defeating — that needs an `encode`, which is precisely what amortisation exists to avoid. The compromise is a pooled summary (slot `μ` mean ⊕ `logvar` mean, `2·d_latent` dims, near-zero cost). Verified the prediction now shifts as memory evolves.

## Literature sweep 2026-08-09 — summary

**Full sweep → `JOURNAL.md`** (IndexMem, Tensor Cache, Infini-attention, KV Means, VECTOR, with
per-paper code availability). It was run after Stage 2b measured a 30–40 point loss, and its
finding drove the redesign: **every method that works integrates the memory at the attention
*output* as a gated residual; we were the only one injecting it into the KV cache**, and the
zero-readout ablation showed injection alone accounted for the entire loss. Three facts from it
still bind:

- **IndexMem (ICML 2026) is this design done correctly** — learnable importance indexer +
  latent memory over evicted tokens + frozen backbone, but read as `o = o_attn + g(q)·m(q)`
  with an exact `g→0` fallback and no RoPE/position/softmax competition. *(Superseded as
  nearest neighbour by `Still`, and the centroid arm by ResKV — see the 08-11 sweep and
  "Read First".)*
- **Gains only exist under aggressive eviction** (negligible at 25%, substantial at 75–90%),
  which the headroom map below independently confirms.
- **KV Means is the only paper on our side of the fence** (memory injected as KV into softmax)
  and reports that a *fixed-size* state struggles at long context, needing a **√N growth
  schedule**. Our fixed 16 slots per head is the configuration it reports as failing.
- **Infini-attention never reproduced** — HuggingFace's attempt is titled "A failed
  experiment". Cite it as the paradigm ancestor, not as an established result.

## The residual read-out was built and measured (2026-08-10) — the regression is fixed, and the memory still earns nothing

**Result tables and the checkpoint inventory → `JOURNAL.md`.** The standing conclusion:
`memory.py:56 residual_gate` / `--varikv_residual` removed the catastrophic KV-injection
regression and gives an exact fallback, but every configuration where the memory actually
participates is worse than baseline, and training closes the gate — `ckpt_gap_rand` trained its
gate *below* its 0.018 init and scores a paired Δ of exactly 0.00 with CI [0,0], i.e. absent
rather than tied. Gate closed ⇒ byte-identical to baseline; gate open ⇒ worse; monotone in the
gate on two independent datasets.

### Two measurement traps hit while producing the table above

1. **Do not read `results.parse`'s relative-performance rows across runs.** Each run normalises by *its own* full-cache score, and those differ between runs (36.30–38.15 measured here for a path memory was assumed not to affect). **The "bf16 / GPU nondeterminism" this entry used to blame is the wrong explanation** — the real cause is the empty-memory injection found on 2026-08-11 (see that section): the full-cache reference is generated *by the memory run itself*, and the memory perturbs it. Dividing by a smaller denominator inflates the ratio: `gap_fix03` reads **103.03 vs baseline 96.08** at ratio 0.75, apparently +7 points; the absolute paired difference is **+1.48, not separated**. Report absolute scores with a paired test over samples — `scratch_gap_eval_report.py` does this.
2. **`results/parse.py` has its own `set_ratios()`, and it must see the same `VARIKV_RATIOS` as the eval did.** Parsing a custom-ratio run without exporting the variable makes parse look for the default `[1.0, 0.75, 0.5, 0.4, 0.3, 0.2]`, find nothing, and print **0.00** for every missing ratio — silently, no error. Both ends are local additions and both read the same env var.

Corollary to the standing "hand-built `-m` strings" warning: the double underscore in `fastkvzip__ret_…` comes from a leading underscore in the `--tag` value, not from the tag machinery. Runs tagged `gapf` produce a **single** underscore (`fastkvzip_gapf_…`). Read the `Results saved at` line from the run's log rather than reconstructing the path.

## 2026-08-11 — `scbench_kv` residual results, and the headroom map that reframes every negative result so far

### The finding that matters most: there was almost nothing to recover

Absolute accuracy of the **FastKVzip baseline** across 11 datasets (Qwen2.5-7B-1M, chunk 16000 / window 4096, `level=pair`, 100 samples where available), with headroom = full-cache − score at ratio 0.2, i.e. how much a perfect absorption mechanism could possibly win back inside the paper's x-axis:

| paper panel | dataset | full | @0.3 | @0.2 | **headroom @0.2** |
|---|---|---|---|---|---|
| **Retr.KV** | `scbench_kv` | 68.20 | 65.40 | 45.20 | **+23.00** |
| Retr.Prefix-Suffix | `scbench_prefix_suffix` | 50.00 | 51.60 | 39.20 | +10.80 (very noisy set) |
| GSM8K | `gsm` | 70.00 | 66.00 | 63.00 | +7.00 |
| En.MultiChoice | `scbench_choice_eng` | 79.17 | 76.39 | 72.22 | +6.95 (n=18) |
| ICL.ManyShot | `scbench_many_shot` | 37.78 | 35.93 | 32.96 | +4.82 |
| Code.RepoQA | `scbench_repoqa` | 58.64 | 59.09 | 57.73 | +0.91 |
| Math.Find | `scbench_mf` | 33.17 | 34.17 | 32.33 | +0.84 |
| SQuAD | `squad` | 93.21 | 93.32 | 92.65 | +0.56 |
| En.Summary | `scbench_summary` | 36.63 | 36.71 | 36.29 | +0.34 |
| En.QA | `scbench_qa_eng` | 39.43 | 44.51 | 44.06 | **−4.63** |
| **Retr.MultiHop** | `scbench_vt` | 41.07 | 42.67 | 46.09 | **−5.02** |

**Inside the paper's operating range, `scbench_kv` is the only dataset with real headroom.** Nine others are ≤ 11 points and two are *negative* — on `scbench_vt` the baseline **improves** monotonically as the cache shrinks (41.07 full → 46.09 at ratio 0.2). This reframes the whole project's run of negative results: for 10 of 11 datasets, "absorb the evicted KV to recover lost accuracy" had no target to hit. It is not (only) that the method fails; it is that the experiment was mostly run where success is arithmetically impossible. Any future dataset choice should start from this column.

**The result tables this section was built on → `JOURNAL.md`**: `dist` loses to `point` on
**10 of 11 datasets** under KV injection (sometimes by an order of magnitude), and the residual
read-out does **not** rescue retrieval-intensive data — on `scbench_kv` with the gate actually
open it is −56…−68 points, so the earlier "KV injection was the whole cause of the collapse"
must be narrowed to "…on `many_shot`". Also there: the **9-dataset × 3-checkpoint sweep**
(27 jobs, 56.7 GPU-h) whose aggregate effect is **zero to two decimal places** — 44 of 45 cells
unseparated for the closed-gate checkpoint. Per-dataset GPU-hour costs for planning any future
grid are in that section.

### Measured: the `gap` loss is ~90% of the trivial solution, but the **gate is near-optimal and capacity is the real bottleneck**

`scratch_probe_gap_target.py`, run 2026-08-11 on the target 7B with the real gate through the same path training uses. Three quantities per (sample, layer): `triv = mean(tgt²)` (the MSE of the trivial `m ≡ 0` solution), the achieved MSE, and

    R_opt = Σ_h [⟨m̂_h, tgt_h⟩² / ‖m̂_h‖²] / Σ_h ‖tgt_h‖²,   m̂ = m / σ(gate)

i.e. **the largest relative MSE reduction obtainable by re-tuning the per-head gates alone** (per head, because the gate is per head). 3 fineweb docs, contexts 10.9k–15.6k, chunk 16000.

| ckpt | ratio | `triv` | achieved | achieved/triv | **R_opt** | gate σ |
|---|---|---|---|---|---|---|
| `ckpt_gap_rand` dist | 0.3 | 0.004298 | 0.003862 | **0.898** | **11.03%** | 0.0143 |
| `ckpt_gap_rand` dist | 0.1 | 0.021098 | 0.019650 | 0.931 | 7.91% | 0.0143 |
| `ckpt_gap_fix03` dist | 0.3 | 0.004177 | 0.003539 | **0.847** | **15.49%** | 0.0317 |

**Half the earlier suspicion is confirmed: the printed loss is nearly the trivial value.** "loss 0.003" is a 10–15% improvement on emitting zero, so it carries essentially no information about convergence — same class of trap as the F-predictor collapse (loss 0.0419 vs 0.0421 for the constant 0). Never again report this loss without the `mean(tgt²)` reference beside it.

**The other half is refuted, and this is the important part: the gate is not mis-trained, it is close to optimal.** Per-layer at ratio 0.3 (`gap_rand`), R_opt is concentrated in the late layers — **layer 26: 31.3%, layer 24: 12.6%, layer 27: 10.7%**, layers 6/8/11 at 3–4.6%, and **all 22 remaining layers below 1%, most below 0.1%**. In the layers that do carry signal the trained gate already extracts nearly all of it: layer 26's achieved/triv is 0.700 against a floor of `1 − 0.313 = 0.687`; layer 24 is 0.888 vs 0.875; layer 27 is 0.913 vs 0.893. **So training closed the gate in 25 of 28 layers because that is the correct decision there** — the read-out genuinely has nothing to contribute in those layers. The hypothesis "the content is useful but the gate/optimiser failed to exploit it" is dead.

**Therefore the bottleneck is representational capacity, not the objective and not the gate.** 16 slots per head can repair only ~11–15% of the attention-output damage that eviction causes, and only in the last few layers. A 15% repair, passed through `o_proj`, is not going to move a benchmark — which is exactly what every evaluation has shown.

**Consequence for the prescription: capacity must be fixed before the objective.** Swapping in Still-style answer-token KL would be optimising a parametrisation that is already near its own ceiling under the current target.

Two more findings that are directly actionable:

- **`--ratio_mode random` hurt.** `gap_fix03` (trained at a fixed 0.3) beats `gap_rand` at ratio 0.3 on every metric — R_opt 15.49% vs 11.03%, achieved/triv 0.847 vs 0.898 — and its gate is correspondingly more open (σ 0.0317 vs 0.0143, 12% vs 5% of head-groups above 0.1). Random-ratio training dilutes the target. Prefer fixed-ratio training, or train one memory per operating point.
- **Aggressive compression is the right battleground, but not for the reason assumed.** At ratio 0.1 the damage is 4.9× larger (`triv` 0.0211 vs 0.0043) and the memory recovers **3.5× more in absolute terms** (0.00167 vs 0.00047) — but a *smaller fraction* (7.9% vs 11.0%), so the unrepaired damage is **5× larger** (0.0194 vs 0.0039). More headroom does not mean the current memory can reach it.
- Signal living only in the last few layers is a cheap design lead: attaching the residual read-out **only to the late layers** would drop 25 layers' worth of compute and 25 layers' worth of noise injection at no measured cost.

Probe limitation to keep in mind: contexts were 10.9k–15.6k, far below the 169k of `scbench_kv`. R_opt at real eval length is not measured.

### The memory injects even when it is empty — `full__` is not a clean reference

`attn.py:149` calls `past_key_value.memory_residual(...)` **unconditionally** whenever residual mode is on. There is no "nothing absorbed yet ⇒ skip" guard, so a memory whose slots are still at their init values is softmaxed and added into the attention output.

Consequence for measurement: `eval_chunk.py:73` prefills with **no chunking and no eviction** to produce the full-cache reference (`full__`, via `_prepare_query` → `model.generate`). Nothing is evicted there, so the memory is empty — but it still injects, so that reference is "full cache + empty-memory residual", not the baseline. Evidence that this is real and not nondeterminism:

| run | ckpt | ratio-1.0 score |
|---|---|---|
| `rb` / `kvlb` | none | **68.20 / 68.20** |
| `kvres_dist` / `kvl_res_dist` | `stage2b_res` dist | **66.80 / 66.80** |
| `kvres_point` / `kvl_res_point` | `stage2b_res` point | **68.60 / 68.60** |
| `kvlgf` / `kvlgr` dist / `kvlgr` point | the three `gap_*` | 67.20 / 67.80 / 70.40 |

Same ckpt on different GPUs in independent jobs ⇒ byte-identical; different ckpt ⇒ different. The ckpt determines the "full-cache" number, which it could not do if the memory were absent. This also **replaces the bf16/nondeterminism explanation** previously given for the 36.30–38.15 full-cache spread on `many_shot`.

Two further implications: the "gate → 0 gives an exact fallback" property is narrower than claimed — it holds for the *gate*, but there is no guard for an *empty memory*; and the same unconditional injection happens in every chunk before the first eviction, so it perturbs the real ratio rows too, not just `full__`. Fix is one line at the top of `memory_residual` (return 0 while `_absorbed_upto == 0`, or scale by absorbed mass) — deliberately **not** applied yet, because it would make new numbers incomparable with the two runs in flight.

### What the P0 diagnostics found — read `P0_FINDINGS.md`

Run the same night, all training-free. The headline is that **the null result above is a
representation failure, not a dead research question**:

- **Missed mass is large**: `M = D_E/(D_R+D_E)` mean **0.316**, P90 0.770 over 4.21M
  (layer, q-head, token) points. "FastKVzip only evicts negligible mass" is refuted.
- **Correlation says nothing, intervention says a lot.** Token-level Spearman between local
  damage and behavioral divergence is ~0.03 (negative at the answer token), but restoring the
  exact counterfactual `Δo` for **all** heads cuts `B = KL(p_full‖p_pruned)` by **55.7%**
  (`B[mean]` −79%). **Never again judge a local quantity's importance by correlation.**
- **The oracle ceiling is therefore −55.7% / −79%**; the remaining ~45% is inherited trajectory
  drift that per-layer local correction cannot reach.
- **Partial restoration is harmful.** Cross-head cancellation keeps only **0.253** of the summed
  per-head damage, and restoring a subset (top-80 of 784) drove `B` from 3.32 to 6.44 on one
  sample while being near-perfect on others. **Correction must be all-or-nothing within a layer**
  — which rules out budget schemes that repair only some heads, in ResKV and IndexMem too.
- **The Gaussian second-order MGF is accurate at the realistic operating point and improves with
  context length** (`r_MGF` median 0.973 at N=128k, W=8192; N=16k is the worst case), so fixed `K`
  is not a length-scalability problem. The P90≈1.9 tail is fixable by **one stored log-correction
  scalar per cluster** — `ε`'s within-cluster/across-cluster variance ratio is only **0.15**.

## 2026-08-12/13 — the teacher-KL round, and the two results that reframe the project

**Read `RESULTS_2026-08-12.md` first — it is the single entry point for every measured
result and supersedes this section where they disagree.** `MODELS.md` has the
checkpoint-by-checkpoint table. This section records only what a future Claude must not
re-derive.

**The two headline findings, both settled 2026-08-13:**

1. **The training-free centroid is the strongest measured arm — but it is a published
   method, not our contribution** (scooped by ResKV + Attention Matching, established
   2026-08-13; see "Read First" above before writing any of this up). `attention/centroid.py`,
   all 11 Figure-11 panels at ratio 0.1, full dataset size, paired bootstrap on absolute
   scores: **K=1024 gives 6 significantly positive panels, 1 negative, mean Δ +3.66, and
   its sign agrees with headroom on 10/11.** The one significant negative
   (Retr.MultiHop) is the one panel whose headroom is *negative* — compression there
   beats full cache — so losing there is what faithful restoration must do.
   **Capacity is NOT shown to be the bottleneck downstream** — an earlier version of this
   entry claimed "K=1024 beats K=16 on 10 of 11 panels", which is simply wrong. Measured
   paired: **7 higher, 1 tie, 3 lower** by score (8/11 closer to full cache by fidelity),
   and **only 2 of 11 panels have a separated paired CI** — Code.RepoQA +4.09 ★ and
   Retr.MultiHop −1.91 ★ (the latter being *more faithful but lower scoring*). So the P0
   probe's capacity claim is **not** confirmed downstream; 64× more capacity buys a
   separated gain on one panel. Corollary: **K=16 has the better efficiency story**
   (0.095% of the budget vs K=1024's 6.08%), so do not chase K=2048/4096.
2. **The learned 0.33M module is not a memory — it is a gate-score perturbation.** The
   frozen-mask 2×2 (`scratch_probe_maskmed.py`, 40 samples) decomposes its +51.00 into
   **selection +44.00 ★** (force the no-memory arm to use the memory arm's `valid` mask
   and it gains almost everything) and **representation +5.00, not separated**. So 86% of
   the gain is mediated by *which KV are retained*, not by reading absorbed content back.
   The v2b control seals it: the failed checkpoint shifts the retained set **more**
   (3.18% vs 2.07%) and gains nothing — what matters is *which* entries change, not how
   many. Consistent with the 4-arm decomposition (all gain from the content × prefill-
   injection interaction) and with forensic v2 (correction direction orthogonal to `Δo`).
   **Scope narrowed 2026-08-14:** this decomposes *that checkpoint's* +51.00, and the
   +51.00 is now known to be a lucky draw (see the retraction above). The decomposition is
   still the right reading of what the module does mechanically — it moves the mask, not
   the representation — but "86% of the gain is selection" describes one trajectory's gain,
   not a reproducible effect size. It is what motivated VariKV-B (control the score, never
   the attention output), and that motivation is unaffected.

Two standing traps this round re-confirmed. **Never trust a ★ on partial samples**: at
38/100 Math.Find read −3.95 separated; at 100/100 it is −2.33 not separated. And
**mask statistics must be summed globally before dividing** — `level="pair"` allocates
budget across all layers and heads, so per-(layer,head) ratio averaging is badly skewed
(the old probe reported "1.76% dropped / 9.83% added", impossible when `|B|=|F|`).

### What was built

- **`--obj kl`** in `scratch_stage2b_train.py`: `L = Σ_t w_t·KL(p_full ‖ p_pruned+memory)`,
  supervising ~255 positions after a randomly-placed context window instead of a
  document's last 128 tokens. Every step also forwards a **pruned-but-memoryless**
  reference so the log reports `gap = KL(p_F‖p_P)`, `resid = KL(p_F‖p_V)` and
  `recov = 1−resid/gap`. That third number exists because of the gap-objective trap
  (loss 0.003 was only 10–15% better than the trivial `m ≡ 0`); **never report a KL
  loss without the size of its target beside it.**
- **`attention/centroid.py`**: `CentroidRetainCache`, the training-free
  count-aware-centroid estimator with normalization-aware read-out
  (`r_j = a·k̄_j + log n_j`, `o = λ o_R + (1−λ) o_E`, `L_R` from flash's `softmax_lse`).
  Four acceptance checks pass, including "swap the summary for the exact evicted set
  ⇒ recovers full-cache attention to 3.7e-06".
- Eval-time surgery switches: `--varikv_gate_scale`, `--varikv_gate_from`,
  `--varikv_ablate {logvar,precision,eta}` (all enter the tag).

**The measurements → `JOURNAL.md`** (and `RESULTS_2026-08-12.md`, which is authoritative). The
four things they establish, since planning depends on them:

- ~~**The objective was a first-order cause of failure.**~~ **RETRACTED 2026-08-14 — the
  +21.60 was training-run variance.** Two retrains of **byte-identical v1 code** (worktree
  `/home/ubuntu/zxy/vlm-memory-repro21`, driver `scratch_repro21.sh`) score **−17.40★** and
  **−8.75★** on the same Retr.KV @0.1 eval where the original scored **+21.60★**. Three draws
  of one procedure span **39 points** with CIs excluding zero in disagreeing directions; v1
  was unseeded, so these are legitimate trajectories of the same recipe. `ckpt_kl_v2a`'s
  −13.20 was therefore never a `min_chunks=1` corpus artifact — it is an ordinary member of
  the spread. **Consequences: the project has no surviving positive learned result, and the
  "orthogonal correction yet +21.6 points" puzzle from `forensic2` dissolves rather than
  needing an explanation.** The four generalization failures listed here before (8 panels
  mean +1.41, Prefix-Suffix/RepoQA gaining nothing despite more headroom, Retr.MultiHop
  −18.36★) still stand as measurements; they are just no longer "failures of an otherwise
  working method". **Standing rule from this: one training run is not a measurement — report
  n≥3 seeds and the across-seed spread, or report nothing.** A paired bootstrap over
  evaluation samples was correct here and still misled, because it quantifies eval-set
  sampling noise while the dominant variance was in the optimiser.
- **"Distributional beats point" has zero support once sampling is matched** — the v1 gap of
  +39.60★ became unseparated in both v2 reruns, and v1's point arm had degenerate output
  (48.9 characters vs the baseline's 120.5).
- **Streaming training made things worse**, and the corpus is why: all 68 `fineweb_10k`
  documents are under 32,256 tokens, so at chunk 16000 more than one eviction per step can only
  come from `fineweb_10k_cat` — hence `--n_short` / `--n_long`.
- **The centroid's algebra, not its capacity, is what works**: `log n_j` alone is worth **67×**
  in recovered missing mass, while 64× more slots buys +1.40 points. And **naive post-RoPE
  averaging beats the theoretically-correct position-free frame** (+6.80 vs +1.20 unseparated)
  — the averaging acts as an implicit low-pass filter. **Do not "fix" this.**
- The **forensic probes** (`scratch_probe_forensic2.py`) are also there: the learned correction
  has roughly the right magnitude but is **orthogonal to the true local gap**, per head and
  after cross-head summation. Two retractions live in that section — v1 aimed at `o_E` instead
  of `Δo`, and the proposed `L = KL + λ·L_structure` fix would most likely drag 54.20 down
  toward the centroid's 43.60. Everything there is compared **in the read-out frame**, which
  matters because `memcache_retain` stores keys pre-RoPE and `centroid.py` stores them
  post-RoPE, and the two frames differ by 74–93%.

### Two standing warnings this round produced

- **Training-side metrics anti-correlate with downstream.** `ckpt_kl_v2a/dist` has the
  best fixed-validation recovery of six checkpoints (**+10.5%**) and the worst
  downstream score (**−13.20**); `ckpt_kl_v2s/dist` has the worst validation
  (**−145.7%**) and is nearly neutral downstream (−2.00). A validation curve is not
  evidence about a benchmark, not even directionally.
- **Report absolute scores only when every arm has all its samples.** The paired Δ is
  stable across subsets (+23.44 at n=93, +21.60 at n=100) while the absolute baseline
  moves 29.68 → 32.60, because the intersection drops whichever samples the slowest
  arm has not reached. Evaluation itself is deterministic (greedy decoding; verified
  byte-identical across GPUs), so a changing number always means a changing sample set.

### Scheduler lesson

Judging a GPU idle by "memory used < 2 GB" is wrong: an eval job's memory dips below
that during generation, and a second job gets dispatched onto the same card (observed).
Use `nvidia-smi --query-compute-apps` — process presence is binary — and require two
readings 20 s apart. Also, two schedulers that both claim "the first free GPU" will
race; give them disjoint candidate lists.

## Literature sweep 2026-08-11 — the objective is what is wrong, and two papers already did this correctly

Run after the `scbench_kv` results above. Conclusion: **the architecture is fine and the niche is nearly closed; what we got wrong is the training objective (LM continuation instead of answer-side distillation) and the capacity schedule (fixed K instead of scaling with context).** Every claim below was read out of the paper, not inferred from a title. Full competitor analysis, with per-paper open-source status, is in `kv_inference_acceleration_2026.md`.

### `Still` (2606.07878, 2026-06-05) — our design, done correctly

| | Still | ours |
|---|---|---|
| backbone | **frozen** | frozen ✓ |
| compactor | one small per-layer Perceiver; latent queries cross-attend the full per-layer KV; **compaction in a single forward pass**, no per-context optimisation | encoder/decoder + slots, amortised ✓ |
| read-back | compact KV **replace** the prefix cache entries inside the softmax | KV injection (old) / output residual (new) ✓ |
| **objective** | **forward KL between a full-context teacher and the compact-cache student, masked to answer-side tokens only**; 4-domain QA set, ~120k items, AdamW 4e-5 | ① fineweb-edu continuation LM loss ② MSE on the attention residual |
| result | **8×–200×**; beats KV-Distill by 8–22 points on RULER; compares against H2O, SnapKV, StreamingLLM, Attention Matching, KV-Distill | collapses at 3× |

The only substantive difference is the objective, and ours is measurably the wrong one: the LM-loss variant is **anti-correlated** with downstream accuracy (loss 1–2 and the gate opening monotonically while accuracy collapses), and the residual-MSE variant is a suspected degenerate optimum. Still distils **the full-cache model's own output distribution on answer tokens** — that is the target that works.

**Still rates train/eval horizon mismatch as fatal rather than minor.** Verbatim: *"the 8k-trained compactor **collapses below the no-context floor** when deployed at 128k without matching training."* Treat it as a first-order design constraint for any future run.

**But do not apply that criticism to the current checkpoints — checked 2026-08-11 and it does not hold.** An earlier version of this section claimed "we trained at 8k and evaluated at 26k–169k"; that is the config of the *earliest* ckpt (`varikv/ckpt_stage2b/`) only. For `ckpt_stage2b_res` and `ckpt_gap_*`, `scratch_stage2b_train.py` loads **byte-for-byte FastKVzip's own gate-training data** (`load_fineweb("fineweb_10k")[:29]` + `("fineweb_10k_cat")[:5]` — 34 docs, 0.98M tokens, lengths **10,444–119,319**, logged by the script itself), and the chunk-count distribution in the training logs (chunk 16000: `num 2` ×397, `num 7` ×132, `num 8` ×88) proves training contexts reached **112k–128k** — the long documents went in whole, not truncated. Same corpus, same chunk/window as the eval. **The residual mismatch is the task, not the length**: training regresses the last 128 tokens of a document (LM continuation), evaluation is retrieval with the question at the end. That is the thing to fix, and it is the same fix as the objective (item 1 below).

**Recording-keeping gap this exposed:** the checkpoints store only `memory` / `mode` / `num_slots` / `model` — **no training args**, so `max_ctx`, `lr`, `steps`, `obj` etc. cannot be recovered from a `.pt` and had to be inferred from log side-effects. Add the argparse namespace to the save dict.

Two more of its findings bear directly on our design choices. Its retained cache grows as **1/c — linear in context, explicitly not O(1)**; and it is honest that compaction is *not* lossless, full-context inference still wins at extreme ratios. Combined with KV Means' earlier report that a fixed-size state needs a **√N** growth schedule, our **fixed 16 slots per head regardless of context length** is more aggressive than any configuration anyone reports succeeding with (at 169k and ratio 0.3 that is ~800:1 per head).

### `Attention Matching` (2602.16284, MIT) — the same objective as our `gap`, solved in closed form and training-free

- Fits compact keys, per-token attention-mass biases and values **per KV head per layer** by matching attention outputs and attention mass over reference queries. Bias fit = NNLS, value fit = OLS, key selection = highest aggregated attention or OMP. **No training of anything.**
- The compact KV **replace original entries inside the softmax** (plus a scalar logit bias). So "synthesised KV entering the softmax" is a *working* design — our collapse was implementation, not a law of nature. Note the difference from ours: it *replaces* retained keys, it does not append extra slots on top of a full budget.
- 10% cache: QASPER F1 **0.428 vs 0.104** baseline (Qwen3-4B). 1% cache: LongBench v2 **61.7%**. Beats H2O+, SnapKV, PyramidKV, **KVzip** and KVMerger over 20×–100×.
- ~150 s per context (fast variant), ~100× cheaper than Cartridges. Models Qwen3-4B / Llama3.1-8B / Gemma3-12B; QuALITY, LongHealth, QASPER, LongBench v2, RULER.

**This is the same optimisation problem as our `--obj gap`** (fit the attention-output difference the eviction caused). They solve it exactly in ~2 s of linear algebra; we solve it with SGD through a bottlenecked encoder plus a gate and land on loss 0.003 with the gate shut. Strong evidence that our failure is in the parametrisation/optimisation, not in the choice of target.

**Its stated limitation is the one opening left for us**: restricting compact keys to *subsets of original keys* limits extreme compression, and *"directly learning compact keys could improve extreme compression"* — Cartridges (gradient-based) does beat it at 100× on LongHealth. So a **learned** compact representation still has room, but only at 10×–100×, not at the 3× we have been testing.

### The niche has narrowed further since the 2026-08-09 sweep

- **RetentiveKV (2605.04075)** — absorbs evicted KV into a matrix-valued state `S_t = H_t ⊙ S_{t-1} + A_t ⊙ (k_t^T v_t)`, reads back **as an output-side residual**, frozen LLM, and is explicitly **"uncertainty-aware"** — via cross-modal attention **entropy** with a sigmoid gate, *not* variance or KL. Multimodal (LLaVA-1.5-7B, Qwen3-VL), up to 5× cache reduction. So "uncertainty-aware absorption of evicted KV into a state" is now published; ours differs only in using a Gaussian and a KL.
- **MomentKV (2606.01563)** is closer than this file previously recorded: its moment statistics **both** steer which tokens to evict **and** provide a post-eviction correction of the attention output. That is our §11 "one quantity decides both decisions" claim, already in print, training-free.
- **Rate-distortion framing for KV is now published**: **RDKV (2605.08317)** casts eviction and quantisation as two endpoints of one bit-allocation problem; also **RateQuant (2605.06675)** and **Spherical KV (2605.18856)** ("rate-distortion retention"). §11's rate-distortion narrative can no longer be presented as new.

What is left of our differentiator: Gaussian `(μ,σ²)` slots with KL-gated writes and amortised variational (not closed-form Kalman, not entropy, not moments) inference. And our own 11-dataset measurement says `dist` loses to `point` 10 times out of 11.

### One published result vindicates a finding of ours

**Error Certificates for KV-Cache Eviction via Randomized Design (2607.21475)** proves (Thm 1) that under **deterministic top-k** eviction the eviction error is **unidentifiable** — evicted values can be changed arbitrarily without changing anything the server retains — and that a Poisson randomised design restores identifiability (Thm 2). Accuracy is not sacrificed: median relative error at a 25% budget is **0.0317 (Poisson + Hájek) vs 0.0447 (top-k)** vs 0.2386 (uniform), and on question-aware LongBench the deterministic-vs-Poisson gap is ≤0.7 points. Models Qwen2.5-1.5B/7B/32B, Llama-3.1-8B, Mistral-7B.

This is the published counterpart of our stage-1 result that **random eviction beat every principled criterion including Expected Attention**. That was not a broken experiment; randomised selection really is competitive, and score-based ranking has a formal identifiability problem. Cite this rather than apologising for that row.

Relatedly, **"When Does Value-Aware KV Eviction Help?" (2605.08234)** formalises exactly the instrument we built on 2026-08-11: compression is **non-monotone** (a compressed cache can underperform, match, or *exceed* FullKV), and value-aware reranking helps in **72.6% of "positive-margin" cells** (where FullKV beats the baseline) versus **32.4%** elsewhere. Our headroom table is the right diagnostic, and its conclusion — improvements only exist where the baseline actually lost something — is now citable rather than merely ours.

### Venue status — what is actually oral

| paper | venue | how verified |
|---|---|---|
| **ThinKV** (thought-adaptive, hybrid quantisation+eviction, **near-lossless at <5% cache**, DeepSeek-R1-Distill / GPT-OSS / AceReason) | **ICLR 2026 Oral** | `iclr.cc/virtual/2026/oral/10009981` |
| **KVzip** (our base's ancestor) | **NeurIPS 2025 Oral** | the venue page is labelled **"2025 Oral Poster"** |
| Cartridges | ICLR 2026 | in the ICLR 2026 proceedings; oral/poster not established |
| KV Cache Transform Coding, TurboQuant, ReST-KV | ICLR 2026 | conference papers, no oral marker found |
| IndexMem | ICML 2026 | `/poster/63943` |
| CAOTE, Norm-Guided ℓ2 eviction | ICLR 2026 **Workshop** | workshop posters, not main conference |

**Trap worth remembering: do not infer presentation format from a neurips.cc/iclr.cc URL path.** KVzip's page lives at `neurips.cc/virtual/2025/poster/118741` and is nevertheless an **Oral Poster** — the path is `/poster/` for orals too. Read the format label on the page. An earlier version of this sweep called KVzip a poster on exactly that bad inference.

Not established: a complete ICML 2026 oral list for this topic (the virtual site did not yield verifiable oral pages), and NeurIPS 2026 decisions, which should not exist yet as of 2026-08. **ThinKV is the only KV-compression oral at a 2026 top venue that is verified here** — and it plays on the decode-side reasoning battleground (FastKVzip's Figure 13), which this project has never touched.

### The field, in families — use this to place any new paper

| family | representatives |
|---|---|
| eviction / selection | H2O, SnapKV, PyramidKV, KVzip, **FastKVzip** (our base), AhaKV, OBCache, CAOTE, LKV, Epiphany-Aware |
| quantisation | KIVI (ICML'24), KVQuant, TurboQuant (ICLR'26), RateQuant |
| joint rate-distortion | **RDKV**, Spherical KV, HqeKV, **ThinKV** (ICLR'26 Oral) |
| merging / low-rank / dimensionality | KVMerger, KVSlimmer, CSKV, Palu/FDC, Lexico, KV Cache Transform Coding |
| **latent compaction / learned compact KV** | **Cartridges**, **Attention Matching**, **Still**, KV-Distill ← *the strongest results, and our real competition* |
| **compressive memory absorbing evicted KV** | Infini-attention, LESS, Tensor Cache, IndexMem, KV Means, MomentKV, RetentiveKV ← *the family we thought we were in* |
| architectural / sharing | MLA, GQA, CLA, YOCO |
| inference-time sparse attention | InfLLM-V2, FlexPrefill, SparDA, STS |
| systems / offloading | Mooncake (FAST'25 best paper), LMCache, ShadowKV, PQCache, InfiniPot-V |
| theory / diagnostics | **Error Certificates** (randomised design), **When Does Value-Aware Eviction Help** (non-monotone diagnostic), VASE (stochastic eviction) |

Survey: *Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization*, **ACL 2026 Findings**, arXiv **2607.08057** (system-aware taxonomy: temporal / spatial / structural). Tracking lists: `jjiantong/Awesome-KV-Cache-Optimization`, `October2001/Awesome-KV-Cache-Compression`.

### What to change, in priority order

0. **Capacity first — this reordering is a measured result, not a preference.** The 2026-08-11 probe shows the current 16-slots-per-head read-out is already near its own ceiling under the `gap` target (the gate is near-optimal; R_opt is only 11–15%). Scale capacity with context (1/c or √N) *before* touching the objective, otherwise item 1 optimises a parametrisation that cannot express the answer. Consider also attaching the read-out only to the late layers, where all the signal is.
1. **Replace the objective with answer-token KL distillation against the full-cache teacher** (Still's target). Neither of our two objectives is this, and both are measured or suspected to be broken. Do this *after* item 0.
2. ~~Match the training horizon to the evaluation horizon.~~ **Already matched — verified 2026-08-11, do not spend time here.** What is unmatched is the *task*: swap document-tail continuation for answer-side supervision (same change as item 1).
3. **Make capacity scale with context** (1/c linear, or √N) instead of a fixed 16 slots.
4. **Evaluate at 10×–100× compression** (ratio 0.1–0.01), the only regime where anyone reports gains and — per our own headroom map — the only regime where headroom exists.
5. Before any of the above, settle whether the `gap` objective is degenerate by measuring `mean(tgt²)` (the loss of the trivial `m ≡ 0` solution). Cheap, and it decides whether item 1 is a fix or a rewrite.

Sources: [Still](https://arxiv.org/html/2606.07878v1) · [Attention Matching](https://arxiv.org/abs/2602.16284) · [Cartridges](https://arxiv.org/abs/2506.06266) · [ThinKV](https://iclr.cc/virtual/2026/oral/10009981) · [KVzip](https://neurips.cc/virtual/2025/poster/118741) · [RetentiveKV](https://arxiv.org/html/2605.04075) · [MomentKV](https://arxiv.org/abs/2606.01563) · [RDKV](https://arxiv.org/abs/2605.08317) · [Error Certificates](https://arxiv.org/html/2607.21475v2) · [When Does Value-Aware KV Eviction Help](https://arxiv.org/html/2605.08234v1) · [survey](https://arxiv.org/abs/2607.08057)

## Competing work — literature sweep 2026-08-03

**Nobody has published "absorb evicted KV into a distributional (μ,σ²) memory gated by KL/free energy."** The gap is real. But four clusters have closed in since the direction was set, and two of them threaten specific claims. Sources below come from a literature sweep that distinguished verified-in-full-text from inferred; **arXiv IDs and paper contents were not independently re-checked by me** — read the first three yourself before they shape the paper.

**Changes the Stage-1 baseline.** **MomentKV (2606.01563)** already keeps "compact moment statistics over the evicted token set, including a count, key mean, value mean, and value-key covariance", with a closed-form first-order correction, and is **training-free**. So the honest bar is no longer "beat a point-mean memory" — beating that proves nothing. The `point` tier should be upgraded to a MomentKV-style second-moment baseline, or MomentKV added as a fifth tier.

**Threatens the §10 "unify existing methods as Σ=I special cases" narrative.** **Memory by Design (2605.31163)**, **Kalman Linear Attention (2602.10743, ICML 2026)** and **Gated KalmaNet (2511.21016, CVPR 2026)** all use mean+covariance memory where "the covariance tracks uncertainty over stored associations, steering writes toward uncertain directions", and already claim DeltaNet/GLA/Mamba-2 as covariance-reset special cases. The closed-form Kalman route is published. What is left to us: they use **exact Kalman — no KL, no ELBO, unimodal — and never touch KV eviction**. So the differentiator must be amortised-variational vs closed-form Kalman, which promotes theory §9.3/§11.4.4's "non-conjugacy must be real" (mixture prior + nonlinear decoder, both already implemented) from theoretical hygiene to **the core point of distinction**.

**Already taken, so don't claim them.** CapKV (2604.25975) owns "unify eviction under one information-theoretic objective with a linear-Gaussian attention surrogate". Titans (2501.00663) owns surprise-gated writing (but point-valued MLP memory, no variance). Surprise-Gated Robot Episodic Memory (2606.03787) already publishes diagonal-Gaussian KL gating (but the Gaussian is a detector only, storage is point-valued, domain is robotics).

**The cleanest gap statement for the intro:** **Larimar (2403.11901)** is the one work carrying the Kanerva Machine into LLMs, and it explicitly "treats the memory as **deterministic**", recasting the Bayesian update as least squares. Nobody put σ² back.

**Two citation corrections** (apply to `theory_distributional_memory.md` / `kv_direction_positioning.md` before submission):
- Dynamic Kanerva Machine is **1811.09556**. The ID 1901.02670 currently in the docs is a complex-analysis paper.
- Do **not** cite survey 2508.10824 for "Titans uses KL divergence thresholds" — that claim is false and is propagating through search engines.

## Scratch Files

Working artifacts — evidence for the results recorded above, not part of the research module. Archived phases live under `scratch/` (see `scratch/README.md`); a phase's files sit at the repo root while it runs and move under `scratch/` when it is done. Reorganised 2026-08-03.

**`scratch/` — archived, one directory per phase**

| Path | Contents |
|---|---|
| `scratch/install/` | venv/CUDA install logs and scripts, the flash-attn torch2.6 wheel, `reqs_fixed.txt` |
| `scratch/repro_0730_qwen3/` | 2026-07-30 first pass (Qwen3-8B, fastkvzip only, 3 datasets; `scbench_mf` failed on the parquet bug). Superseded |
| `scratch/repro_0731_qwen25/` | 2026-07-31 second pass (Qwen2.5-7B-1M × 5 methods × 3 datasets). `fig11_results.log` is the parsed table, `fig11_run.log` the scheduler trace. Its `fig11_parse*.sh` are superseded by `scratch_fig11_driver.sh` |
| `scratch/probe/` | 2-example timing probes. **Run one before committing GPU-days to a new (model, dataset) pair** — cost is not inferable from context length. Their leftovers in `results/` are ~2 dirs per dataset, not eval results |
| `scratch/refs/` | `fastkvzip_paper.txt` — the only local copy of the paper, PDF-extracted (figure data points lost) |

**Repo root — two finished phases, not yet archived.** Both runs completed on 2026-08-04, so nothing here is live and the "don't move these" warning no longer applies. **Pending chore: fold the Figure-11 files into `scratch/repro_0803_fig11_full/` and the Stage-1 files into `scratch/stage1_0804/`** — except `scratch_repro_full.py`, which stays at the root because it is the general reproduce driver, not a throwaway.

| File | Phase | Notes |
|---|---|---|
| `scratch_repro_full.py` | — | the 8-GPU scheduler / general reproduce driver. **Keep at root**, despite the prefix |
| `scratch_fig11_driver.sh` | fig11 | end-to-end driver (wait → MRCR → parse 12 datasets) |
| `scratch_fig11_full_run.log` / `scratch_fig11_full_results.log` | fig11 | scheduler trace / per-dataset parsed tables (~600 KB) |
| `scratch_repro_full_logs/` | fig11 | per-job logs plus `.done__*` completion markers |
| `scratch_stage1_driver.sh` | stage1 | wait for GPUs → train tiers 2/4/5 → evaluate all 5 |
| `scratch_stage1_results.log` | stage1 | driver summary (the all-zeros eval — see the Stage-1 status section) |
| `scratch_stage1_logs/` | stage1 | `train_tier{2,4,5}.log`, `eval_all.log`. Untracked as of this writing |
| `scratch_stage1_sweep.sh` / `_report.py` / `_logs/` / `_results*.log` | stage1 | the 3-capacity × 5-tier sweep and its paired-bootstrap report (the report filters non-finite values — see the NaN corollary) |
| `scratch_debug_{evict,pred,pred2,F,nan,nan2,eval…}.py` | stage1 | the diagnostics behind the "predictor collapsed to a constant" and NaN findings |
| `scratch_evict_variants.py` | stage1 | the 10-criterion eviction sweep whose random baseline showed stage1 cannot judge eviction |
| `scratch_stage2a.py` | stage2a | real-corpus driver (train/eval 5 tiers + random-eviction control) |
| `scratch_stage2a_results.log` | stage2a | the tier table |
| `scratch_stage2b_train.py` | stage2b | trains through the harness. Key flags: `--residual`, `--obj {lm,gap}`, `--ratio_mode {fixed,random}`, `--kv_type memory_retain` |
| `scratch_stage2b_{smoke,sweep,verify7b}.py` | stage2b | smoke test, 11-dataset sweep driver, 7B verification |
| `scratch_verify_residual.py` | stage2b | **the 4 acceptance checks for the residual path** — memory stays out of the cache but absorption still happens / gate→0 is byte-identical to baseline / gate open changes output / gradient reaches both decoder and gate |
| `scratch_diag_longctx.py` | stage2b | why `dist` collapses as context grows (squad 203 tok → 97.6 relative; vt 125k → 54.5; kv 169k → **0.29**) |
| `scratch_gap_eval.sh` / `scratch_gap_eval_report.py` | stage2b | the 2026-08-10 residual evaluation and its paired-bootstrap report. **Use the report, not `results.parse`'s relative rows** |
| `scratch_kvres_eval.sh` / `scratch_kvres_report.py` | stage2b | the `scbench_kv` residual evaluation (2026-08-10 night) and its paired-bootstrap report. The report **self-checks** its per-sample parse against `results.parse`'s absolute rows — copy that pattern when adapting it to another dataset (it is hard-coded to one `DATA`). Its header claim about `rbkv` is wrong (it justifies skipping the standard interval by calling the `gap_*` ckpts already-proven-identical to baseline, but `rbkv` loaded an *untrained* gate; treat that interval as missing data) |
| `scratch_gapstd_eval.sh` | stage2b | the three `gap_*` ckpts × `scbench_kv` × standard interval (2026-08-11) |
| `scratch_gapsweep.py` / `scratch_gapsweep_logs/` | stage2b | the three `gap_*` ckpts × the other 9 datasets, 27 jobs. Marker-resumable, longest-first scheduling, and workers can be told to wait for another run to finish before claiming a GPU (`--wait_gpus`) |
| `scratch_probe_gap_target.py` | stage2b | **the trivial-solution / capacity-ceiling probe.** Reports `mean(tgt²)` (the `m ≡ 0` MSE) beside the achieved MSE, plus `R_opt` = the best relative reduction obtainable by re-tuning the per-head gates. Monkey-patches `_attn_gap` / `memory_residual` rather than editing `memcache_retain.py`, so it is safe to run while eval jobs are in flight (editing the harness file would change what newly-launched jobs load). ~20 GB, a few minutes; run it before trusting any `gap`-objective loss curve |
| `scratch_stage2b_logs/` | stage2b | all of the above runs' logs; `sweep/` holds the 11-dataset sweep |

**The 2026-08-11→13 diagnostic round** — these are the probes behind everything in
`P0_FINDINGS.md` and `RESULTS_2026-08-12.md`. All are training-free and most monkey-patch the
harness rather than editing it, so they are safe to run while eval jobs are in flight.

| File | What it answers |
|---|---|
| `scratch_probe_damage.py` / `_report.py` / `scratch_probe_intervene.py` | P0: how much mass eviction actually misses, and the oracle ceiling from restoring the exact counterfactual `Δo`. **Intervention, not correlation** — the correlation is ~0.03 and means nothing |
| `scratch_probe_mgf.py` / `_stability.py`, `scratch_debug_mgf_terms.py` | whether the Gaussian second-order MGF is accurate at the realistic operating point |
| `scratch_estimator_ladder.py` | P1: at **equal bit budget**, more point prototypes vs fewer distributional summaries (E0→E5) |
| `scratch_probe_layerjoint.py` | whether the layer-level net damage compresses better than the per-head sum — 75% of per-head damage cancels within a layer, and every competing method corrects per-head |
| `scratch_probe_cov_rank.py` | the rank structure the summaries would have to capture |
| `scratch_verify_centroid.py` | **the 4 acceptance checks for the centroid read-out** — including "swap the summary for the exact evicted set ⇒ recovers full-cache attention to 3.7e-06". Run before trusting any centroid number |
| `scratch_probe_forensic.py` / `_forensic2.py` | does the learned memory reconstruct the evicted KV, or repair the local gap? **v1 aimed at the wrong target (`o_E` instead of `Δo`) — use v2** |
| `scratch_probe_memswap.py` → `scratch_probe_4arm.py` | memory or context-independent steering vector? memswap swaps only *after* prefill and therefore cannot answer it; the 4-arm 2×2 configures all four arms from the first chunk and is the one to trust |
| `scratch_probe_evictshift.py` → `scratch_probe_maskmed.py` | does the memory change *which* KV get evicted, and is that shift the causal mediator? The frozen-mask 2×2 says 86% of the gain is selection |
| `scratch_probe_massdir.py`, `scratch_probe_cluster.py` | is the centroid's residual error mass or direction, and does k-means-in-key-space clustering (ResKV Eq 12) beat position bins? **The newest probe — its `γ ≈ 0.75` shrinkage story is provisional pending re-measurement with k-means clusters** |
| `scratch_probe_supervision.py` | which training documents actually carry gradient under the eviction-sensitive weighting |
| `scratch_cen23.sh` / `_report.py`, `scratch_centroid_sweep.sh` / `_report.py` | the centroid sweeps (ratio 0.1, then the paper-range 0.3/0.2) and their paired-bootstrap reports |
| `scratch_klsweep.sh` / `_report.py` | the teacher-KL round. `PANEL` in the report is the dataset-id ↔ paper-panel-name map |
| `scratch_p234.sh`, `scratch_v2b_*.sh`, `scratch_v3_autoeval.sh` | the GPU-waiting chain drivers. **Judge a GPU free by `nvidia-smi --query-compute-apps`, not by memory used** — an eval job's memory dips below 2 GB during generation and a second job lands on the same card |
| `scratch_model_registry.py` | regenerates `MODELS.md`'s tables from the checkpoints on disk |

Keep new throwaway scripts on the `scratch_` prefix at the root, and archive them under `scratch/` once their phase is done. **This has fallen behind** — the root now holds ~200 scratch files from five finished phases (fig11, stage1, stage2a/b, the gap sweeps, the P0/forensic round) plus the `.npy` artifacts of the clustering probe. `scratch/README.md`'s "do not move these, runs in flight" warning refers to the Figure-11 sweep that finished 2026-08-04 and is stale.

## Architecture of `memory_module.py`

`FreeEnergyMemory` (subclass of `nn.Module`) implements a 5-step pipeline:

1. **Compress** (`compress`): Cross-attention collapses N visual tokens → 1 evidence vector `e_t`
2. **Prior** (`get_prior`): Cosine similarity between `e_t` and memory slot means → weighted mixture prior `p(z|M_{t-1})`
3. **Free Energy** (`compute_free_energy`): Recognition network infers posterior `q_φ(z|e_t)`, computes KL divergence + reconstruction loss = variational free energy `F`
4. **Update** (`update_memory`): Adaptive update rate `η_k = sigmoid(α·KL·w_k - β)` — high KL + high slot relevance → large update
5. **Read** (`read`): Each slot's mean vector → `tokens_per_slot` tokens, weighted by certainty (`-logvar`); returns `[K*T, d_model]` for LLM input

**Key design**: Memory slots store distributions `(μ_k, σ²_k)` not fixed vectors. `logvar` small = certain = resistant to overwriting.

**Memory is a recurrent STATE, not a parameter** (fixed 2026-07-22): The slots' *initial* values are learned parameters `mem_mu_init` / `mem_logvar_init` (optimizer-trained). The *running* slots `mem_mu` / `mem_logvar` are plain tensors, cloned from the init in `reset()` and updated by the `η_k` recurrence inside the forward pass with ordinary tensor ops — so gradients flow through the recurrence and the memory is genuinely trained end-to-end. This replaced an earlier bug where the slots were `nn.Parameter` updated via `.data =` assignment, which detached them from autograd and silently made the "end-to-end trained free-energy memory" claim false. Call `reset()` at the start of each video. Note: the write content (`mu_q`/`logvar_q`) now carries gradient; the gate magnitude `η` still stop-gradients `KL` (a deliberate control-signal choice — remove the `.detach()` in `update_memory` to backprop through the gate too). Because the running state is carried across segment `forward()` calls without detaching, a long video builds a full BPTT graph — add truncated BPTT if memory is tight.

**Known open design issue (not yet fixed)**: `get_prior` collapses the K-slot mixture into a *single* averaged Gaussian (`einsum` weighted-average of means and logvars) before the KL. This discards the multimodality that theoretically justifies the amortized/variational approach over closed-form Kalman (see `theory_distributional_memory.md`). Recommended fix: compute per-slot Gaussian KLs and combine weighted by `w_k` (a closed-form upper bound on KL-to-mixture) instead of averaging first. Discuss the objective before changing.

**Note**: All inline comments in `memory_module.py` are written in Chinese.

**Integration point** (see bottom of `memory_module.py`): Insert after MLP Projector output in `InternVL/internvl_chat/model/internvl_chat.py → forward()`. History frames write memory; current frame prepends `memory_tokens` to `vit_embeds`. The `is_history_frame` flag must be supplied by the caller — it is not in the current codebase.

**Training loss**: `total_loss = lm_loss + 0.01 * free_energy`

## Research Documents

The six most load-bearing ones are tabulated in "Read First" at the top; this section is the
complete index, including the historical ones.

- `JOURNAL.md` — **the chronological record**, split out of this file on 2026-08-13, verbatim and
  in original order. Read an entry for *how a conclusion was reached and what was ruled out*;
  it is history, and later entries retract earlier ones
- `RESULTS_2026-08-12.md` — **the single entry point for measured results** (Chinese). 44 eval
  jobs + 7 trainings; one uniform reporting convention (Retr.KV @0.1, same-batch baseline,
  per-sample paired bootstrap, ★ = 95% CI excludes 0, absolute scores only at full n). The
  dataset-id ↔ paper-panel-name map lives in `scratch_klsweep_report.py:PANEL`
- `EVAL_PROTOCOL.md` — **the anti-contamination record** (Chinese): which inference hyperparameter
  (`--varikv_gate_scale`) was tried on which panel and with what result, plus the rules the final
  report must obey. Deliberately demotes **no** dataset — an earlier revision demoted `scbench_kv`
  to a dev set and was retracted, because it is the only panel with real headroom. Read it before
  picking any eval-time hyperparameter
- `NEXT_STEPS.md` — the **frozen (v5)** roadmap and falsifiable hypotheses (Chinese). Carries five
  methodological corrections that invalidate parts of earlier probes — most importantly that all
  damage quantities must be projected through `W_O` before being compared across heads/layers,
  which means the "signal only in layers 24/26/27" conclusion needs recomputing. H1 is already
  marked judged-false by data
- `P0_FINDINGS.md` — the training-free diagnostic round: missed mass, the −55.7% oracle ceiling,
  "correction must be all-or-nothing within a layer", and the Gaussian MGF accuracy result
- `FINDINGS_DENOISING.md` — interventional evidence for FastKVzip's own "compression denoises
  attention" attribution: a monotone series on Retr.MultiHop where restoring more evicted content
  moves the score *down* toward full cache. The one place this project goes beyond its base paper
- `varikv_method_spec.md` — the method at code level (formulas as implemented, not as designed)
- `scratch_gpt_review_brief.md` — the brief prepared for external review; useful as a compact
  statement of the method and its open problems
- `MODELS.md` — **the checkpoint registry: every trained memory module with its
  generation, training config, learned gate, and downstream result.** Its tables are
  generated by `scratch_model_registry.py --md` (38 checkpoints across 11 dirs, so a
  hand-written list goes stale immediately); only the "generation / purpose" and
  "results" sections are hand-maintained. **Read this before citing any checkpoint** —
  it records which ones have recoverable training args (6 of 38) and which code commit
  trained each one
- `README.md` — the public-facing writeup of VariKV (English; the only doc written for outside readers). **Its tier table is stale**: it still lists the pre-2026-08-03 *four*-tier design, where tier 3 = free_energy+point and tier 4 = VariKV. The code has five tiers (tier 3 is now MomentKV, VariKV is tier 5), so the README's `--tier 4` / `--tier 1 2 3 4` commands no longer run what it says they run. Fix before showing it to anyone.
- `HANDOFF.md` — **START HERE.** Execution entry point for the active KV-cache direction (written in Chinese): what's decided (Path B, ICLR 2027, base = Fast KVzip), what NOT to re-investigate, the reproduce → variance-ablation (GO/NO-GO) → build plan, the four theory-driven code changes, and the honest ~10–20% accept-probability calibration. The method was further pinned on 2026-07-30 to **Option 3 "free-energy unified eviction"** (one scalar `F_i = D_i + λ·KL_i` decides both *which* KV is demoted into memory and *how much* is written) — but stage 1 (the GO/NO-GO variance ablation) deliberately uses simple recency/sliding-window eviction first, to isolate "distributional vs. point absorption" before adding the unified eviction.
- `free_energy_memory_proposal.md` — full theoretical writeup of the *video-era* framing (theory, math, related work, experiment plan). Historical origin, not the current goal.
- `theory_distributional_memory.md` — theory of why distributional (μ,σ²) memory can beat point memory (Bayesian filtering / rate-distortion / multimodality). §9 = rigorous KV-cache instantiation + the four load-bearing theory gaps; §10 = cognitive-science anchors + intro skeleton + the "unify existing methods as Σ=I special cases" narrative; **§11 = full method design for Option 3 (`F_i` definition, degenerate-special-cases table, amortized F-predictor, four-tier ablation, staged de-risking)**.
- `fastkvzip_code_map.md` — **source-code map of the cloned Fast KVzip / KVzip repos**: end-to-end prune pipeline and scoring (reconstruction attention vs. learned gate). Its Path-B anchor discussion predates the on-server verification — where it and the "Two Codebases" section above disagree, the section above (`EvictCache._sample_cache`) is the checked one.
- `kv_direction_positioning.md` — positioning for the KV pivot: Path A vs B (B decided), three-layer competitor/ancestor map, the make-or-break variance-ablation experiment, §7 accept-probability, §8 step-by-step execution checklist. Note IndexMem/Tensor Cache (closest precedents) have NO public code — comparison papers only.
- `kv_cache_survey.md` — survey of training-free KV cache compression (5 mechanism families + top-venue oral/poster status), plus a reproducibility verdict on Fast KVzip (easy–moderate, gates released). **Scoped to training-free only, so it cannot see this project's real competition** — for that use the file below
- `kv_inference_acceleration_2026.md` — **the reference doc for competitor analysis (written 2026-08-11, Chinese).** The full 2026 KV-inference-acceleration landscape across 7 families, ~40 papers, each with venue/status, key technique, **verified open-source status**, and the specific difference from ours; plus a per-dimension table of where our design diverges from what works, and a 6-step prescription. Every claim carries a provenance marker (✔ read that day / ○ verified in an earlier sweep / △ unverified background knowledge) — **respect those markers, especially before citing a repo URL or a venue tier**
- `memory_orals_2025_2026.md` — verified list of memory-related ORAL papers at NeurIPS/ICML/ICLR 2025–2026, with confidence tags; the finding that video-VLM memory gets zero top-venue orals
- `robotics_memory_survey.md` — survey of memory in robotics (NeurIPS/ICML/ICLR/CoRL/RSS/ICRA 2024–2026)
- `video_vlm_survey.md` — survey of video VLM methods
- `robotics_experiment_guide.md` — experiment setup guide
- `llm_basics.md` — LLM fundamentals reference
- `interview_prep.md` — interview preparation notes
- `latent reasoning/Reasoning_Methods_Comparison.md` — comparison of CoT/Coconut/Huginn/CODI/SCoRe/RiM along two axes: architecture-change vs. training-only, and token-external vs. latent-internal reasoning; includes thesis positioning notes (HiCI/PMI/SDMTR sit in the "architecture + latent" quadrant alongside Huginn and RiM)

## Key Hyperparameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `d_model` | 4096 | Must match InternVL3-8B hidden size |
| `num_slots` (K) | 16 | Number of memory slots |
| `tokens_per_slot` | 32 | Tokens per slot when reading → 512 total memory tokens |
| α, β in update | 2.0, 2.0 | `KL·w_k > 1` threshold for meaningful update |
| λ (aux loss) | 0.01 | Free energy loss weight |
