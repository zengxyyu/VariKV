# JOURNAL.md — the chronological record

Split out of `CLAUDE.md` on 2026-08-13, verbatim and in original order. `CLAUDE.md` kept the
things a session needs *before* acting — orientation, commands, harness facts, the
never-re-derive bug and trap lists — and points here for the narrative of how each result was
obtained.

**This file is history, and later entries retract earlier ones.** For results, the authority is
`RESULTS_2026-08-12.md`; for checkpoints, `MODELS.md`; for the current status, `CLAUDE.md`'s
"Read First". Read an entry here for *how a conclusion was reached and what was ruled out*, not
to learn what is currently true.

---

## From “Stage 2b — VariKV wired into Fast KVzip (2026-08-07)”

> Integration verification, the first negative benchmark result, and the Figure-11 reproduce. The reference half of this section — the silent-failure bug table, the upstream training-data loader, the known limits, and the vendored-clone modification table — stayed in `CLAUDE.md`.

### Verified — 1.5B (H=2, `expect` gate)

- **Absorption disabled ⇒ output byte-identical to native `EvictCache`.**
- Layout self-consistency (`len_k` / `cu_len_k` / `pos_track`) across {memory on, off} × {after 3 generations}.
- `absorb` padding equivalence: μ/logvar differ by 1e-17 and the residual is **independent of pad count** (float32 casts in the F/position paths account for the 1e-7 remainder).
- Gradient reaches 14/17 memory parameters with `varikv_train=True` (vs 6/17 under inference_mode); the missing ones are `point_gate_logit` etc., inactive in `dist` mode.
- Guards fire correctly: KVzip reconstruction scoring (`gates=None`) and `batch>1`.

### Verified — Qwen2.5-7B-Instruct-1M, real `fastkvzip` gate, real SCBench (2026-08-08)

The target configuration, not a proxy: 28 layers × **4** kv heads (G=112, vs 1.5B's 56), released gate weights, `level="pair"`.

- **Real `scbench_kv` sample, 169,035-token context, ratio 0.3** — pipeline runs end to end and emits correctly-formatted answers (native `6290fbbe-…`, memory `62c9b332-…`; both wrong vs gold, as expected with an untrained memory on a hard task).
- **Budget accounting confirmed on real data**: 5,682,691 → 5,684,479, i.e. **+1,788 against a predicted `M·H·L` = 1,792** (+0.03%).
- Synthetic 42k context: totals conserved at +0.14% (ratio 0.3) and +0.38% (ratio 0.1), the latter exactly the 1,792 overhead.
- Absorption-disabled output byte-identical to native at both ratios.

**Unexplained upstream quirk, not ours:** at `chunk_ratio=1.0` (no pruning at all) generation returns `''` for *every* config including native `EvictCache`. Evaluations run at ratio < 1, so it does not block anything — but do not use ratio 1.0 as a sanity baseline.

### First real-benchmark result (2026-08-08) — negative, and with a known confound

`scbench_many_shot`, 20 identical samples, canonical `eval_chunk.py` scoring (`results.parse`, same as Figure 11). Memory injected via new `--varikv_ckpt` / `--varikv_slots` args, everything else byte-identical to the baseline run.

| ratio | FastKVzip baseline | VariKV `dist` | control `point` |
|---|---|---|---|
| 0.75 | **100.00** | 86.96 | 86.96 |
| 0.5 | **95.65** | 76.09 | 80.43 |
| 0.4 | **100.00** | 76.09 | 86.96 |
| 0.3 | **97.83** | 78.26 | 80.43 |
| 0.2 | **91.30** | 78.26 | 80.43 |

Adding memory **costs 15–25 points at every ratio**, and `point` ≥ `dist` at most ratios. Worse than "no effect": 16 slots per head (0.4% of visible KV) are *actively disrupting* attention.

**Confound — this run's memory was trained at `chunk=2048 / window=256 / 8k` against an eval at `16000 / 4096 / 26k`**, so it is not yet a verdict on the method. Retraining with matched config + upstream data is in flight. Two measurement traps hit while producing this table, both worth remembering: the baseline first wrote into the **existing Figure-11 result directory** and `results.parse` averaged 54 stale samples alongside 5 new ones (fixed with a distinct `--tag`); and the result directory carries a **double underscore** (`fastkvzip__s2b20_…`), so a hand-built `-m` string silently parses 0 samples and raises `ZeroDivisionError`.

**`retain` vs `evict` is not a confound — measured, not assumed.** The baselines all ran `kv_type="retain"` (the `args.py` default; neither `run.sh` nor `scratch_repro_full.py` overrides it) while `MemoryEvictCache` derives from `EvictCache`. On real `scbench_kv` data with identical scoring, the two produce **byte-identical generations at every ratio**, so building on `EvictCache` is safe.

### Reproduce status

**Figure 11, 3 of 12 datasets: qualitatively consistent with the paper. Not a numeric reproduction — and cannot be, see the caveat above.** Run 2026-07-31: `Qwen2.5-7B-Instruct-1M` × **all 5 methods** × 3 datasets (`squad`, `scbench_many_shot`, `scbench_prefix_suffix`) × 6 ratios, 100 samples each (54 for many_shot) — 15 jobs, 2.9h on 8×H100, 0 failures. Raw numbers in `scratch/repro_0731_qwen25/fig11_results.log`.

What matches: at ratio **0.3** fastkvzip scores 100.1 / 95.1 / 103.2 relative, satisfying the paper's "near-lossless at a 30–40% budget"; and the method ordering follows Figure 11 — fastkvzip ≥ kvzip > duoattn > expected ≫ **snapkv, which collapses** (36.5 on squad, 7.6 on prefix_suffix). The 3 datasets happen to sample one from each of Figure 11's three categories (prefix_suffix → retrieval, squad → contextual understanding, many_shot → high redundancy), which makes the subset more informative than a random 3.

Two things to state whenever these numbers are cited: (1) **no point-by-point check against the paper was performed**, because the paper publishes no numeric table for Figure 11 — the agreement above is ordering and threshold behaviour only; (2) `scbench_prefix_suffix` is **very noisy** and non-monotonic even for the winning method (73.6 at ratio 0.75 but 115.6 at 0.4), so per-point gaps on that dataset are not meaningful at n=100 — do not build an argument on the large fastkvzip-vs-baseline margin there without more samples.

**Full Figure 11 sweep: COMPLETE (finished 2026-08-04 01:22).** All 12 datasets × 5 methods on Qwen2.5-7B-1M. Main sweep `{done: 40, failed: 0, skipped: 15}` in **15.5h** on 8×H100 (the 15 skips are the 3 datasets already done on 2026-07-31); MRCR then ran all 5 methods (107–145 min each) and the 12-dataset parse followed automatically. The ~15h + 3–4h pre-run estimate held. Longest jobs: `scbench_repoqa` ~400 min/method, `scbench_kv` ~300 min. Raw per-dataset tables in `scratch_fig11_full_results.log`, scheduler trace in `scratch_fig11_full_run.log`.

**The Figure-11 curve, averaged over the 11 `results.parse` datasets** (MRCR is scored separately, see below). `results.parse` prints one block *per dataset*, so this average had to be computed from the log:

| method | 1.0 | 0.75 | 0.5 | 0.4 | **0.3** | 0.2 |
|---|---|---|---|---|---|---|
| **fastkvzip** | 100.00 | 99.07 | 100.53 | 101.64 | **100.54** | 93.78 |
| kvzip | 100.00 | 100.34 | 98.63 | 97.12 | 94.79 | 84.78 |
| expected | 100.00 | 99.14 | 98.46 | 90.46 | 75.00 | 55.14 |
| duoattn | 100.00 | 97.75 | 92.06 | 86.58 | 73.19 | 40.27 |
| snapkv | 100.00 | 80.65 | 63.82 | 57.58 | 52.20 | 46.21 |

Both of the paper's checkable prose claims hold. "Maintains full-cache performance at a **30–40%** budget" — fastkvzip is 101.64 / 100.54 at 0.4 / 0.3, cleanly lossless. Method ordering matches Figure 11, including snapkv's collapse.

**The "matching KVzip" claim needs the noise caveat to come out right, and it does.** Over 11 datasets fastkvzip beats kvzip by +5.75 at ratio 0.3 — which would *contradict* the paper, since it positions the two as a tie (its selling point being half the prefill cost, not higher accuracy). But **+57.20 of that gap is `scbench_prefix_suffix` alone**; the other 10 datasets range −4.40…+4.98. Drop that one dataset and it is 100.27 vs 99.67 — **a tie, exactly as the paper claims**. This is the concrete payoff of the standing warning that `prefix_suffix` is very noisy at n=100: an argument built on the 11-dataset average would have been an artifact of one dataset. Excluding the two small-n sets (`qa_eng` n=20, `choice_eng` n=18) as well moves nothing (99.16 vs 98.61).

**MRCR** is scored by `parse_mrcr.py` in **absolute** score, not relative, so it does not enter the table above. At ratio 0.3: kvzip **45.87** > fastkvzip 38.44 > expected 20.72 ≈ duoattn 18.99 ≈ snapkv 18.24 (full-cache baseline ≈46.4). This is the one dataset where kvzip clearly beats fastkvzip. Note kvzip's own full-cache row reads 45.91 rather than 46.37 because it runs the separate unchunked `eval_mrcr.py` path.

Standing caveat, unchanged: none of this is a point-by-point check against the paper, which publishes no numeric table for Figure 11. What is verified is threshold behaviour and method ordering.

Earlier 2026-07-30 run on **Qwen3-8B** (wrong model for Figure 11, kept only as a Figure-12 data point): relative performance at ratio 0.3 was 89.9 / 93.6 / 101.7 on `scbench_kv_short` / `scbench_prefix_suffix_short` / `scbench_many_shot`, with unexplained retrieval collapse at ratio 0.2 (28.7 / 12.8) where Figure 12 shows ~0.6–0.8. That run also submitted `scbench_mf`, which **failed** on the datasets-4.x parquet error; the pyarrow fallback was written afterwards. So it covered 3 datasets, single-method, and is superseded by the Qwen2.5 runs.

**Do not mistake stray `results/` directories for reproductions.** `gsm`, `scbench_repoqa`, `scbench_summary`, `scbench_vt`, `scbench_choice_eng`, `scbench_qa_eng`, `scbench_mf_mid` each contain only ~2 result dirs — those are the 2-sample timing probes from `scratch/probe/timing_probe.sh` / `scratch/probe/fig11_probe.sh`, not eval runs.

Timing measured on Qwen2.5-7B-1M, seconds/example over all 6 ratios: `squad` 12.8 (203-token context), `scbench_many_shot` 10.2 (26k), `scbench_prefix_suffix` 93.8 (112k).

## From “Stage 1 — the GO/NO-GO variance ablation (2026-08-04 → 08-07)”

> The synthetic needle task: the broken evaluator, the capacity sweep, and the diagnosis of why free-energy eviction fails. The task design and the `stage1/data.py` commands stayed in `CLAUDE.md`.

### Stage-1 status: first end-to-end run done, **GO/NO-GO still unanswered — the evaluator is broken**

Ran 2026-08-04 01:21→02:03 via `scratch_stage1_driver.sh` (train tiers 2/4/5 in parallel on 3 GPUs, 1500 steps, `budget=256`, then evaluate all 5 tiers at `limit=120`). Logs in `scratch_stage1_logs/`, summary in `scratch_stage1_results.log`.

**Training worked and the tier ordering came out as the theory predicts** — final `lm_loss` **tier 5 (VariKV) 2.25 < tier 4 (point) 2.77 < tier 2 (recency+point) 3.98**. Free energy and the predictor loss are both well-behaved (tier 5: F ≈ 33, predictor 0.055). This is genuine signal that the distributional memory is learning something the point memory is not.

**Evaluation is a dead read: exact-match is `0.000` in every single cell — all 5 tiers, both kinds, all 4 distractor levels.** Do not read this as "the method failed"; it cannot be a method result, for two reasons:

- **Tier 1 also scores 0.000**, and tier 1 is `absorb_mode=discard` — no memory participates at all.
- The **`n_distract=0`** bucket also scores 0.000, and those samples are 109 tokens against `budget=256`, so **no eviction is even triggered**. That cell is the bare frozen `Qwen/Qwen2.5-1.5B-Instruct` answering a 109-token retrieval question, scored by `exact_match`, which is a *lenient substring* test (`gold in pred`). A working harness cannot score 0 there.

#### Root cause found and fixed (2026-08-07): a trailing space in the prompt

**The prompt ended with `"[ANSWER] "`, and the answer was tokenised separately.** Under Qwen's BPE that is a genuine off-manifold sequence:

```
"[ANSWER] " + "crimson-kite-33"   →  [..., ']', ' ', 'cr', 'imson', ...]   # 220, 5082, 45445
"[ANSWER] crimson-kite-33"        →  [..., ']', ' crimson', ...]           # 96019
```

A standalone space token (220) is almost never followed by a *space-less* `'cr'` — natural text merges them into the single token `' crimson'` (96019). Consequences, both confirmed by measurement:

- **Evaluation**: after emitting the dangling space the model's greedy continuation goes anywhere but the answer — the observed `'40\n[ENDLOG]…'` / `'86\n\n[END_OF_LOG]…'` garbage.
- **Training is corrupted too, not just eval.** `forward_loss` teacher-forces on the same `q_ids`+`a_ids` split, so tiers 2/4/5 were trained to produce a token sequence the model would essentially never emit. **The `lm_loss` ordering reported above was measured against that corrupted target** — it is suggestive but must be re-measured.

A second, independent defect: **`max_new_tokens=16` was too short.** The model prefaces its answer with "The current value of user_X is …", which alone eats ~10 tokens.

Measured on tier 1, 8 samples, pure HF forward with no memory involved — the two defects are separable and both real: `16 tok + space` **0/8** → `48 tok + space` 2/8 → `48 tok, no space` **7/8**.

**What was NOT wrong — do not re-investigate.** `generate()`'s hand-rolled `position_ids` / `mem.n_seen` / `mem.n_mem` bookkeeping was the prime suspect and is **correct**. Verified at tier 1 (`n_mem=0`, no eviction) against a single-shot HF forward over the same tokens: top-1 identical on every sample, `max|Δlogit| ≤ 0.45` (bf16 chunked-prefill noise).

The fix is three lines — `stage1/data.py:render` and `varikv/train.py:encode_sample` now end the prompt at `"[ANSWER]"` and move the space onto the answer's first token (`tok(" " + answer)`), and `evaluate.py:generate` defaults to `max_new_tokens=48`. **`stage1/*.jsonl` does not need regenerating** — it stores `context`/`question`/`answer` separately and never the assembled prompt.

**Post-fix tier-1 baseline** (`--tier 1 --limit 60 --budget 256`): overall **0.267**, and the shape is exactly right for the ablation — `all/0` = **0.941** (109-token contexts, below budget, no eviction ⇒ plain frozen LLM, near-ceiling) and `all/200` = `all/800` = `all/2000` = **0.000** (context evicted, tier 1 discards, needle gone). That is a clean floor: tiers 2–5 have to beat 0.000 at every non-trivial distractor level.

#### A second bug found by the same run: NaN from a 1-token final chunk

`torch.std` with the default `unbiased=True` returns **NaN for a single element**, and `NaN.clamp_min(1e-6)` is still NaN — the four z-score sites all had `.clamp_min(1e-6)`, which guards `std == 0` but **not** `std == NaN`. When `len(context) ≡ 1 (mod prefill_chunk)` the final chunk holds exactly one token and one absorb poisons all 57,344 memory elements. Measured: sample i=144 has 34,305 tokens = 67×512 + **1** and always breaks; the neighbouring 34,405-token sample (final chunk 101) is fine — which is why single-sample debugging missed it. Only 1 of 160 samples per config, but one NaN makes the whole column's mean NaN.

Fixed with `unbiased=False` at all four sites (`memory.py` write gate, `free_energy.py` `_zscore`, predictor target, running std): population std is 0 at n=1, the clamp then takes over and the z-score degenerates to 0 — which is also the semantically right answer, since "surprise relative to the rest of this chunk" is undefined for a single observation. **Training was unaffected** (`max_train_context=4096`, and 4096 % 512 = 0, so the final chunk is never degenerate); only evaluation needed re-running.

Corollary bug in the analysis script: NaN fed into the bootstrap produced **fake "separated" verdicts** with plausible-looking CIs. `scratch_stage1_sweep_report.py` now filters non-finite values and reports how many it dropped. A tight confidence interval is not evidence that the underlying numbers were real.

### Stage-1 result (2026-08-07): a real but small effect, and half the method does nothing

Clean sweep, zero NaN, zero dropped samples: 3 capacities × 5 tiers × 160 samples (40 per distractor level), `budget=256`, Qwen2.5-1.5B-Instruct, paired bootstrap over identical samples. Raw data `scratch_stage1_sweep_logs/`, driver `scratch_stage1_sweep.sh`, report `scratch_stage1_sweep_report.py`.

**Primary metric is `nll` on the answer tokens, not exact match.** EM is 0.231 for *every* tier and every K, and all of it comes from the `nd=0` bucket where the context is below budget and no eviction happens. Compression is 377:1–4231:1 and the answers are high-entropy random strings (`jade-shrike-85`), so character-exact recovery is impossible for any lossy memory. EM has no resolution here; nll separates the tiers cleanly.

| K | t1 sliding-window | t3 MomentKV | t2 point | t4 fe+point | **t5 VariKV** |
|---|---|---|---|---|---|
| 16 | 3.9524 | 3.8576 | 2.5099 | 2.4725 | **2.3212** |
| 32 | 3.9524 | 3.7793 | 2.5578 | 2.6591 | **2.4378** |
| 64 | 3.9524 | 3.7267 | 2.8388 | 3.1172 | **2.7617** |

**Decomposition at K=16 — this is the number that matters for the paper:**

| change | Δnll | share of total gain |
|---|---|---|
| discard → store a point mean (t1→t2) | **−1.44** | **88%** |
| recency → free-energy eviction (t2→t4) | −0.04 | 2% |
| point → distributional (t4→t5) | −0.15 | 9% |

So 88% of the benefit is "don't throw it away", which Infini-attention established in 2024. The project's own two contributions split 2% / 9%.

**What replicates: distributional absorption.** `t5 vs t4` is separated at every capacity and the margin *grows* with K: −0.151 [−0.231, −0.070] at K=16, −0.221 at K=32, −0.356 at K=64. That is the cleanest evidence the variance carries information — the two tiers have identical structure and parameter count, so the only variable is whether the precision term informs the read-out.

**What does not: free-energy eviction.** Marginal at K=16 (−0.04) and **actively harmful** at higher capacity — at K=64, `fe+point` (3.1172) is 0.28 *worse* than plain `recency+point` (2.8388). Half the §11 method (the "one scalar decides both decisions" claim) is currently not earning its place. Diagnosing this is the highest-information open question.

**Capacity hurts.** Every trained tier degrades monotonically with K (paired, all separated): t2 +0.33, t4 +0.64, t5 +0.44 going 16→64. The "memory capacity may be too small to show an effect" risk in the notes above is **refuted** — more slots is worse, so that escape hatch is closed. Note t5's advantage over t2 also shrinks (−0.19 → −0.08, not separated at K=64).

**Three things this run does NOT license claiming.** (1) Not a win over KVzip — tier 1 is a sliding window, see the warning above. (2) Not a win over MomentKV — tier 3 is this repo's approximate reimplementation, and the 1.54-nat gap to it is the *least* trustworthy number in the table despite looking the most impressive. (3) Not task success — EM is 0 on every sample that actually got compressed.

### Why free-energy eviction fails: the amortised predictor collapsed to a constant (diagnosed 2026-08-07)

**The free-energy eviction arm has never actually been tested.** What tiers 4/5 run at eval time is not free-energy eviction — it is a broken approximation of it. Chain of evidence:

1. **At eval, `score()` never computes exact F.** `free_energy.py:239` — when `not self.training`, it returns `self.predicted(...)` and nothing else. That is deliberate (amortisation is the whole efficiency story, HANDOFF red line 2), but it means eviction quality *is* predictor quality.
2. **The predictor's ranking is anti-correlated with the exact F it distils.** Spearman ρ(pred, exact) = **−0.28** at ctx 4096, −0.30 at 8k, −0.36 at 16k. Negative **at the training length**, so this is not a train/eval transfer failure — it never learned. (Probe restores `v_scale`/`d_std`/`kl_std` around each `exact()` call; those buffers set the D-vs-KL weighting, so a naive probe changes what it measures.)
3. **The predictor outputs a near-constant.** `std(F_pred)` = **0.047** against `std(target)` = 1.0. Its loss is 0.0419; the loss of *literally always emitting 0* is **0.0421**. Distillation bought a 0.5% improvement over the trivial baseline — the `predictor 0.043` in the training logs looks healthy and means nothing.
4. **Root cause: the distillation target is catastrophically heavy-tailed.** Within-chunk z-scored F_exact, n≈1e6: **96.4% of tokens have |z| < 0.1**, 99.4% < 0.5, but the 99.9th percentile is **27.0**. Kurtosis **702** (normal = 3). ~0.16% of tokens carry all the variance. z-scoring fixes scale, not shape.
5. **Huber then makes "predict the constant 0" near-optimal.** The comment at `free_energy.py:270` says Huber was chosen over MSE so outliers would not dominate the gradient. It worked too well: with 96% of the mass already at ~0 and the outliers' gradient capped, the loss-minimising constant is 0 and there is no pressure to learn any ordering.

Downstream, this explains the eviction behaviour measured directly (`scratch_debug_evict.py`, needle-retention probe via `mem.token_pos`): at K=16 free-energy eviction keeps *early* tokens (mean relative position 0.574 vs recency's 0.975) and retains part of the needle in 3/8 samples vs recency's 0/8 — so its small K=16 advantage is an **accident of stage1's needle being early**, not F working. At K=64 it drifts back toward the tail (0.843) and needle retention goes to 0, which is why it loses to plain recency there.

**Fix applied 2026-08-07 — rank-based distillation target.** Eviction consumes only the ordering of F, so the value target was replaced with within-chunk normalised ranks (uniform on [-1,1]): a monotone transform that discards zero ranking information while being bounded, uniformly spread and outlier-free. `free_energy.py:261` carries the full rationale.

**The fix works — and it makes eviction worse.** After 1500 steps, ρ(pred, exact) = **+0.78** at every context length (from −0.28; +0.15 at 400 steps), so the amortised predictor now genuinely tracks the exact free energy. But the resulting eviction is *worse*, not better:

| K=16, evicted samples only | old (broken predictor) | new (rank target, ρ=0.78) |
|---|---|---|
| tier 4 `fe+point` | 2.8114 | **3.0181** (+0.21 worse) |
| tier 5 `fe+dist` | 2.6096 | 2.5929 (−0.02, flat) |
| tier 2 `recency+point` (unaffected) | 2.8612 | 2.8612 |

So with the amortisation now demonstrably faithful, **F is a worse eviction criterion than plain recency**: fixed tier 4 is **+0.157 above** tier 2. The old tier-4 result was not "free-energy eviction working" — the broken predictor's accidental early-token bias happened to catch stage1's early needle (needle retention 3/8, mean kept position 0.574). Following F correctly retains the needle **0/8** and pulls kept positions to 0.663, still far from recency's 0.975 — i.e. correct F evicts recent context, which matters more for next-token prediction than F's distortion+surprise accounting credits.

#### …but stage1 cannot adjudicate eviction at all — a random baseline wins

Full criterion sweep with **exact** scoring (bypassing the predictor entirely, so this tests `F` itself, not the amortisation), tier 4 = point absorption so the only variable is the eviction rule. K=16, 60 evicted samples, `scratch_evict_variants.py`:

| criterion | nll | vs recency | vs random |
|---|---|---|---|
| **random** | **2.6984** | −0.034 | 0.000 |
| recency | 2.7324 | 0.000 | +0.034 |
| `D` only (λ=0 ⇒ Expected Attention) | 2.7882 | +0.056 | +0.090 |
| `F` λ=0.1 | 2.8234 | +0.091 | +0.125 |
| `−D` | 2.8236 | +0.091 | +0.125 |
| `F` λ=0.3 (current) | 2.8449 | +0.113 | +0.147 |
| `−KL` | 2.8933 | +0.161 | +0.195 |
| `F` λ=3 | 2.9462 | +0.214 | +0.248 |
| `−F` | 2.9640 | +0.232 | +0.266 |
| `KL` only | 2.9970 | +0.265 | +0.299 |

**Random eviction beats every principled criterion, including recency and including published Expected Attention (`D` alone).** That is the load-bearing observation: it means **stage1 carries no signal about eviction policy**, so nothing here licenses a claim that free-energy eviction is bad — only that this task cannot tell. This is consistent with the task's design intent: `HANDOFF.md` specifies stage 1 to isolate *absorption*, with deliberately simple eviction. Using it to judge eviction is a category error. (Earlier notes in this file framed the result as "F loses to recency"; that framing was wrong and is superseded by this row.)

Two internal trends *are* consistent, on a task with no eviction signal, so treat them as hypotheses rather than findings:
- **Performance degrades monotonically in λ**: λ=0 → 2.7882, 0.1 → 2.8234, 0.3 → 2.8449, 3 → 2.9462, and `KL` alone is the single worst of all ten. The surprise term is what drags F down.
- **The sign is not simply inverted.** `−F` (2.9640) is worse than `F` (2.8449), so hypothesis "the rate-distortion convention is backwards for eviction" is **refuted**. Note `−KL` does beat `KL`, hinting the KL *component's* sign may be backwards — but even at its better sign it loses to plain `D`.

To actually test eviction, the experiment must move to a setting where a sensible baseline is clearly better than random — i.e. real long-context benchmarks through the Fast KVzip harness (Stage 2), not the synthetic needle task.

**The absorption half carries the method.** With the fixed predictor, tier 5 still beats tier 2 by −0.268 — so VariKV's advantage survives despite its eviction rule being a liability rather than because of it.

Diagnostics live in `scratch_debug_evict.py` (what eviction keeps), `scratch_debug_pred.py` / `scratch_debug_pred2.py` (predictor ranking quality vs K and vs context length).

Verdict: **not a NO-GO, but far from a publishable GO.** The core mechanism shows a real, replicated, statistically separated effect, but it is an order of magnitude smaller than the trivial "memory vs no memory" effect it sits on top of, and it rests on two baselines that are both weaker than the papers they stand in for. Next steps in priority order: (a) replace tiers 1 and 3 with real KVzip scoring and official MomentKV, (b) diagnose why free-energy eviction is worthless-to-harmful, (c) move to a task whose answer is not a high-entropy random string.

## Literature sweep 2026-08-09 — the design is wrong in a specific, published way

Run after Stage 2b measured a 30–40 point loss on real benchmarks. **Every method that works integrates the memory at the attention *output* as a gated residual. We are the only one injecting it into the KV cache — and the zero-readout ablation shows that injection alone accounts for the entire loss.**

| method | memory read-out | gate | LLM | outcome |
|---|---|---|---|---|
| **IndexMem** (2605.25475, **ICML 2026**) | `o = o_attn + g(q)·m(q)`, **residual** | `g(q)∈[0,1]`, `g=0` ⇒ exact fallback | **frozen** | +25 pts RULER-16K at extreme compression |
| **Tensor Cache** (2605.22884) | `y = y_local + σ(g)·m_t`, **residual** | learned scalar, per-head λ/η | trained e2e | NLL 5.14 vs 6.00 full-KV @32k; beats Infini-attention |
| **Infini-attention** (2404.07143) | `sigmoid(β)·A_mem + (1−β)·A_local`, **residual** | learned β | trained e2e | **HF reproduction failed** |
| **KV Means** (2605.09877) | **extra KV into softmax** | none | trained from scratch | needs a *growable* state; fixed state struggles long-context |
| **VECTOR** (2605.23258) | keeps real keys, rebuilds only V via OLS | — | calibration only | +9.73 @ pc=0.90 |
| **ours** | **extra KV into softmax** | **none** | frozen | **−30 to −40 pts** |

**IndexMem is our method done correctly, and it is published.** Learnable indexer predicting KV importance + latent memory compressing evicted tokens + frozen backbone — the same three pieces. **Superseded 2026-08-11: `Still` (2606.07878) is a closer match still — same amortised-encoder-with-frozen-backbone design, and it works. See the 2026-08-11 literature sweep; the difference is the training objective, not the architecture.** Differences that matter:

- **Read-out is a gated residual, never a cache insert.** `m(q) = Linear(q)ᵀM / (Linear(q)^⊙2·b + ε)`, then `o = o_attn + g(q)·m(q)`. The gate gives an **exact fallback**: `g→0` recovers the baseline at zero cost. Our design has no such escape — the 16 injected KV always consume softmax mass.
- **Memory is a fast-weight matrix** `M ∈ [d_model/8, d_model]` plus a stabiliser `b`, updated by outer products `M ← λM + η Σ Linear(kᵢ)⊗vᵢᵀ` — not slots, no encoder/decoder, no RoPE round-trip.
- **Gains appear only under aggressive eviction**: negligible at 25%, "noticeable" at 50%, "substantial" at 75–90%. Matches VECTOR (gains only at pc∈{0.75,0.90}) and our own measurement that the FastKVzip baseline leaves **0.00 headroom at ratio ≥ 0.3**.

**Infini-attention never reproduced.** Google released no code or models; HuggingFace's attempt is titled *"A failed experiment"*. Their finding — *"long context performance gets worse as we increase the number of times we compress the memory"*, needle-in-1st-segment failing completely, the balance factor collapsing to 0.5 — is the same shape as our value-norm divergence over absorb rounds. Citing it as the paradigm ancestor is fine; treating its results as established is not.

**Consequences for this project.** (1) The KV-injection integration must be replaced by an output-side gated residual before any further experiments — it is the measured cause of the loss and every working method already does this. (2) The evaluation ratio must move to ≥75% eviction; at 0.3 there is provably nothing to recover. (3) The remaining differentiator against IndexMem/Tensor Cache is only "distributional `(μ,σ²)` + KL-gated writes vs. point/fast-weight" — and `dist` ≈ or < `point` in every measurement so far.

**(1) and (2) were carried out on 2026-08-09/10 — see the next section for what happened.** Short version: (1) worked, in the sense that the collapse disappeared and the gate gives an exact fallback; but with the fallback available, training closes the gate. (2) confirmed the ceiling is small — at ratio 0.1 the baseline only loses 5.93 absolute points, so that is the entire recoverable budget on `scbench_many_shot`.

Sources: [IndexMem ICML 2026](https://icml.cc/virtual/2026/poster/63943) · [IndexMem arXiv](https://arxiv.org/html/2605.25475) · [Tensor Cache](https://arxiv.org/html/2605.22884) · [KV Means](https://arxiv.org/html/2605.09877) · [VECTOR](https://arxiv.org/html/2605.23258v1) · [HF Infini-attention reproduction](https://huggingface.co/blog/infini-attention)

**Code availability** (searched 2026-08-09 across arXiv full text, OpenReview, GitHub):

| paper | code | note |
|---|---|---|
| **KV Means** (2605.09877) | **✓ [`featherless-ai/KVM-paper`](https://github.com/featherless-ai/KVM-paper)** | forked from `recursal/KVM-paper`; model + training + Triton kernels + lm_eval; checkpoints at HF `recursal/key-value-means` |
| IndexMem (2605.25475) | ✗ | no statement anywhere in the full text; OpenReview has no supplementary |
| Tensor Cache (2605.22884) | ✗ | no code statement in full text |
| Infini-attention (2404.07143) | ✗ | Google released nothing; third-party reimplementations only |

**KV Means is the only runnable one — and it is the only paper on our side of the fence** (memory injected as KV into softmax rather than fused at the output). Its finding is therefore directly load-bearing for us: a *fixed-size* state "struggles with extremely long contexts"; it needs a **growable** state (√N schedule) to be competitive. We use a fixed 16 slots injected into softmax — the configuration it reports as the failing one. Caveat on transfer: KVM trains 120M/350M models from scratch, we freeze a 7B.

**IndexMem's training setup, for reference** (ours in parentheses): SFT on **LongAlpaca** (fineweb-edu continuation), WSD schedule 100 warmup → 1e-3, 2000 stable + 2000 decay → 7.5e-6, **4100 steps** (1500 steps). Long-context *instruction* data plausibly forces the loss to depend on distant content; generic web continuation does not — which is the second-layer defect measured on our side (loss stays ~1.8 while eval collapses).

**Where our design actually diverges from IndexMem.** Of its three components — learnable importance indexer, latent memory over evicted tokens, frozen backbone — we match the indexer (we substitute FastKVzip's released gate, deliberately) and the frozen backbone exactly. Everything hinges on the memory:

- **IndexMem's memory never pretends to be a token.** It is a fast-weight matrix read as `m(q) = Linear(q)ᵀM / (Linear(q)^⊙2·b + ε)`, added as `o = o_attn + g(q)·m(q)`. No RoPE, no positions, no key-norm competition — because it never enters softmax.
- **Ours synthesises fake KV.** Every hard piece of engineering in `memcache*.py` — position tracking, RoPE inverse/forward rotation to a slot centroid, mask offsetting, per-head varlen padding — exists *only* because we decided the memory must masquerade as tokens. That decision is the one the ablation falsified.

**So the real difference is not "distributional (μ,σ²) vs point/fast-weight" — the claimed contribution — but "residual compensation vs KV injection", which we never treated as a design choice at all.** And on the claimed axis, `dist` is ≈ or worse than `point` in every measurement to date.

## From “The residual read-out was built and measured (2026-08-10)”

> Checkpoint inventory (now superseded by `MODELS.md`), the `scbench_many_shot` table, and the gate-value analysis. The two measurement traps from this round stayed in `CLAUDE.md`.


The sweep above ends with "the KV-injection integration must be replaced by an output-side gated residual." That was done (`memory.py:56 residual_gate`, `--varikv_residual` / `--residual`), trained two different ways, and evaluated. **Verdict: the catastrophic regression is gone, but every configuration in which the memory actually participates is significantly worse than the baseline; the best solution training can find is to switch the memory off.**

### Every checkpoint on disk — 32 of them, three stages

No LLM was ever trained. The backbone is frozen throughout; these are memory modules (0.33M params for the 7B ones).

**All of them live under `varikv/`** — the stage-2b dirs are `varikv/ckpt_stage2b*`, `varikv/ckpt_gap_*`, not repo-root paths (an earlier version of this table listed them at the root; verified 2026-08-11, no root-level `ckpt_*` exists).

| stage | dir | n | model | what |
|---|---|---|---|---|
| 1 | `varikv/ckpt/` | 18 | Qwen2.5-**1.5B** | K∈{2,4,8,16,32,64} × tiers {2,4,5}, synthetic needle task |
| 2a | `varikv/ckpt_real/` | 3 | Qwen2.5-1.5B | tiers {2,4,5}, K=16, real corpus (fineweb-edu) |
| 2b | `varikv/ckpt_stage2b/` | 2 | Qwen2.5-**7B**-1M | first harness integration; train cfg 2048/256/8k ≠ eval 16000/4096 |
| 2b | `varikv/ckpt_stage2b_matched/` | 2 | 7B | config matched to eval |
| 2b | `varikv/ckpt_stage2b_retain/` | 2 | 7B | rebuilt on `RetainCache` (what the baselines run) |
| 2b | `varikv/ckpt_stage2b_res/` | 2 | 7B | **residual read-out, `--obj lm`** — gate open (σ 0.186 / 0.287) |
| 2b | `varikv/ckpt_gap_fix03/` | 1 | 7B | residual, **`--obj gap`**, fixed ratio 0.3 — **dist only, no point control** |
| 2b | `varikv/ckpt_gap_rand/` | 2 | 7B | residual, `--obj gap`, random ratio per step |

### Results — `scbench_many_shot`, 54 contexts × 5 queries, paired bootstrap over samples

Absolute scores with the paired Δ against the baseline; ★ = 95% CI excludes zero. Baseline full-cache 37.78.

| config | gate σ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 |
|---|---|---|---|---|---|---|
| baseline (absolute) | — | 36.30 | 37.78 | 38.89 | 35.93 | 32.96 |
| `gap_fix03` dist | 0.032 | +1.48 | −1.11 | −0.00 | −0.74 | +1.11 |
| `gap_rand` dist | 0.014 | +0.37 | 0.00 | −0.00 | −0.37 | −0.37 |
| `gap_rand` point | 0.024 | +0.74 | +0.37 | −1.11 | −0.74 | −0.37 |
| **`stage2b_res` dist** | **0.186** | **−17.78★** | **−9.63★** | **−9.63★** | −5.19 | **−6.30★** |
| **`stage2b_res` point** | **0.287** | **−5.56★** | **−8.52★** | **−9.26★** | −4.81 | −1.85 |

At the aggressive ratios the literature sweep prescribed (0.1 / 0.05), where the recoverable headroom actually is — baseline drops 37.78 → 31.85, i.e. **5.93 absolute points is the entire ceiling**:

| config | 0.1 | 0.05 |
|---|---|---|
| baseline (absolute) | 31.85 | 32.59 |
| `gap_fix03` dist | +1.48 [−0.74,+3.70] | +0.74 |
| `gap_rand` dist | **+0.00 [0.00,0.00]** | +0.37 |
| `gap_rand` point | +0.37 | +0.74 |
| 〔old〕 KV injection dist | **−12.59★** | **−10.00★** |
| 〔old〕 read-out zeroed `rozero` | **−8.15★** | **−7.41★** |

### The gate value is the load-bearing number

`gap_rand` dist scores a paired Δ of *exactly* 0.00 with CI [0,0] — identical predictions on all 54 samples. That is not a tie, it is the memory being absent. The learned gates explain the whole table:

| ckpt | objective | σ(gate) mean | share of the 112 head-groups > 0.01 |
|---|---|---|---|
| `ckpt_gap_rand` dist | gap | **0.0143** | 12% |
| `ckpt_gap_rand` point | gap | 0.0240 | 26% |
| `ckpt_gap_fix03` dist | gap | 0.0317 | 27% |
| `ckpt_stage2b_res` dist | lm | **0.1862** | 75% |

The gate initialises at **0.018**, so `gap_rand` dist trained itself *below* its starting point. The clean dichotomy:

- **gate closed** ⇒ byte-identical to baseline. Verified on real benchmarks, not just in the unit check: with the gate left at its init value, `many_shot` and `scbench_kv` both reproduce the baseline at all six ratios exactly (`rb` / `rbkv` runs, which load `ckpt_stage2b_retain` — a ckpt with no trained gate).
- **gate open** ⇒ significantly worse than baseline (the two `stage2b_res` rows).
- **let training decide** ⇒ it closes the gate.

So the residual design's exact-fallback property works exactly as intended, and what training discovered is that falling back is optimal. `dist` is still ≈ or worse than `point` — 15 cells, point ahead in most, none separated.

**What this does and does not settle.** It settles that KV injection was the cause of the 30–45 point collapse (`rozero` isolates it: zero the memory *content* and keep only the injection, still −8 and separated). It does **not** settle that absorption is worthless — this is one dataset in the high-redundancy category, where a compressed summary of evicted tokens has the least to offer. `scbench_kv` and `scbench_mf` (retrieval-intensive) were run only in the old KV-injection configuration, where `scbench_kv` scored 0.29 / 0.00 / 0.00 relative; they have **not** been re-run with the residual read-out. That is the first thing to do next. **Done on 2026-08-10/11 for `scbench_kv` — see the 2026-08-11 section; the answer is that the residual read-out does not help there either, and the "one dataset in the high-redundancy category" escape hatch is closed.**

## From “2026-08-11 — `scbench_kv` residual results and the 9-dataset sweep”

> The result tables. The headroom map, the empty-memory defect, the capacity-ceiling probe and the P0 summary — the parts that are reference rather than narrative — stayed in `CLAUDE.md`.

### `dist` loses to `point` on 10 of 11 datasets

From the 9-dataset sweep (`ckpt_stage2b_matched`, KV-injection read-out, 27 jobs, all markers present) plus `scbench_kv` and `many_shot` at the same `_full` tag. Paired Δ against the baseline at ratio 0.2:

| dataset | dist Δ | point Δ | | dataset | dist Δ | point Δ |
|---|---|---|---|---|---|---|
| `gsm` | −3.00 | **+2.00** | | `choice_eng` | −19.91 | −6.02 |
| `squad` | −3.55 | −2.22 | | `vt` | −26.80 | −17.82 |
| `many_shot` | −3.70 | −4.44 | | `mf` | −32.33 | −8.00 |
| `summary` | −3.85 | −4.20 | | `prefix_suffix` | −39.00 | −33.00 |
| `qa_eng` | −18.00 | −6.57 | | `kv` | −43.00 | −3.00 |
| | | | | `repoqa` | −49.55 | −31.59 |

`point` wins on 10 of 11, sometimes by an order of magnitude (`mf` 0.00 vs 24.33, `kv` 2.20 vs 42.20). Earlier entries in this file say `dist` is "≈ or worse than" `point`; at 11-dataset scale that is too gentle — **the claimed contribution is contradicted across the board**, and the only near-neutral cells (`gsm`, `squad`, `summary`, `many_shot`) are the ones with no headroom anyway.

**Unexplained and worth diagnosing: for `dist`, ratio 0.75 is frequently the *worst* of all ratios** — `summary` 4.65 (its other ratios are 32–34), `vt` 5.16, `mf` 0.67, `kv` 7.00, `repoqa` 13.18. Milder compression producing worse output is not an information-loss trade-off; it is the shape of a structural bug. It points the same direction as the empty-memory defect below: the memory does damage precisely when it has little to say.

### The residual read-out does not rescue retrieval-intensive data

`scbench_kv` (169,035-token contexts, 100 samples), absolute accuracy, paired bootstrap vs the `rb` baseline. Report script: `scratch_kvres_report.py` (self-checks its per-sample parse against `results.parse`'s absolute rows).

**The `gap_*` column is complete as of 11:22 UTC 2026-08-11** — `scratch_gapstd_eval.sh`, three ckpts × 100 samples × the standard interval, 7h10m on 3 GPUs, all `Finished.`.

| ratio | baseline | `stage2b_res` dist (σ 0.186) | same, point (σ 0.287) | `gapf` dist (σ 0.032) | `gapr` dist (σ 0.014) | `gapr` point (σ 0.024) |
|---|---|---|---|---|---|---|
| 0.75 | 68.80 | 11.00 (−57.80★) | 11.20 (−57.60★) | 67.40 (−1.40) | 68.60 (−0.20) | 69.00 (+0.20) |
| 0.5 | 71.60 | 4.60 (−67.00★) | 3.60 (−68.00★) | 67.80 (**−3.80★**) | 71.40 (−0.20) | 72.20 (+0.60) |
| 0.4 | 66.40 | 5.00 (−61.40★) | 3.00 (−63.40★) | 63.20 (**−3.20★**) | 67.00 (+0.60) | 66.20 (−0.20) |
| 0.3 | 65.40 | 9.60 (−55.80★) | 2.60 (−62.80★) | 64.00 (−1.40) | 65.60 (+0.20) | 66.80 (+1.40) |
| 0.2 | 45.20 | 8.60 (−36.60★) | 2.60 (−42.60★) | 43.80 (−1.40) | 45.60 (+0.40) | 46.00 (+0.80) |
| 0.1 | 32.60 | 7.60 (−25.00★) | 0.80 (−31.80★) | 30.60 (−2.00) | 32.00 (−0.60) | 32.40 (−0.20) |
| 0.05 | 2.00 | 0.80 | 0.40 | 2.20 | 2.20 | 2.20 (baseline is at the floor) |

★ = 95% CI excludes zero.

**This is the decisive cell, and it is a null result.** `scbench_kv` is the *only* dataset with real headroom inside the paper's ratio range (23 absolute points at ratio 0.2). The three current ckpts land exactly on the baseline there: across `gapr`'s ten cells the largest deviation is +1.40 and **not one is separated**. There is no longer a "wrong dataset / no headroom" escape.

Two regularities now replicated on four datasets:

- **Gate closed ⇒ baseline, gate open ⇒ worse, training ⇒ closes the gate.** Holds on `many_shot`, `scbench_kv`, `prefix_suffix` and the low-ratio interval.
- **The more open the gate, the worse the score.** `gapf` (σ 0.032) is the only config here with separated cells and both are negative; on `prefix_suffix` it loses 3.3–6.6 points where `gapr` (σ 0.014) loses 1.4–2.8. Monotone in the gate, on two independent datasets.

**It also confirms the probe, and the two measurements are independent.** The 2026-08-11 probe put `R_opt` at 11–15% with the signal confined to 3 of 28 layers; a repair of that size, pushed through `o_proj`, is not measurable downstream — which is exactly this table. And note the direction *reverses* between the two: `gapf` looks **better** on the probe (R_opt 15.5% vs 11.0%) and is **worse** downstream. Fitting the attention gap more closely does not make the model more accurate — direct evidence that the `gap` objective is misaligned with the downstream metric, separate from the fact that its loss sits near the trivial solution.

So the residual read-out **softened** the collapse on `many_shot` (−5…−18) but does nothing of the kind on retrieval data: with the gate actually open it is −56…−68. The earlier conclusion "KV injection was the whole cause of the 30–45 point collapse" must be narrowed to "…on `many_shot`".

### The `--obj gap` objective was a live suspect for being degenerate — partially confirmed, see the measurement above

Training-log comparison of the two objectives (both matched config, chunk 16000 / window 4096):

| objective | final loss | `|g|`max | gate σ trajectory |
|---|---|---|---|
| `lm` (`ckpt_stage2b_res`) | 1–2, noisy | 3.4e-02 | 0.095 → **0.186**, monotonically opening |
| `gap` (`ckpt_gap_*`) | **0.003** | **1e-04** | init 0.018 → **0.014**, i.e. *below* its starting point |

The gap objective is `MSE(g·m, o_full − o_pruned)` (`memcache_retain.py:295`), and `m → 0` is inside its solution space, so a loss of 0.003 is consistent with the memory having learned nothing and the gate correctly switching itself off. **This is the same trap as the F-predictor collapse** (loss 0.0419 vs 0.0421 for emitting the constant 0). The one number that settles it is `mean(tgt²)` — the MSE of the trivial `m ≡ 0` solution. Not yet measured; a single-sample probe would do it. Do not read "loss 0.003" as convergence until that comparison exists.

Note the `lm` objective has its own diagnosed defect: its loss falls to 1–2 and the gate opens monotonically while downstream accuracy collapses — fineweb-edu continuation loss is decoupled from, and here anti-correlated with, retrieval accuracy.

### One claim in `scratch_kvres_eval.sh`'s header is wrong

It justifies skipping the standard interval for the `gap_*` ckpts with "gate-closed configs were already proven byte-identical to baseline by `rbkv`". `rbkv` loaded `ckpt_stage2b_retain`, whose gate sits at its **init** 0.018 and was never trained; the `gap_*` gates are trained (0.014 / 0.024 / 0.032, max 0.26–0.40, 4–12% of the 112 head-groups above 0.1). They are not the same configuration, and the measurement agrees — `gapf` reads 30.60 at ratio 0.1 where the baseline reads 32.60. Treat the standard interval for those ckpts as missing data, not as redundant.

### The 9-dataset sweep (finished 2026-08-12) — the aggregate effect is exactly zero

`scratch_gapsweep.py`, **27 of 27 jobs complete**, 56.7 GPU-h. The three `gap_*` ckpts ×
9 datasets × 5 ratios, baselines reused from the `_full` tag (same configuration).
Report: `scratch_gapsweep_report.py` (paired bootstrap on absolute scores, token-level;
the report also handles partial runs by truncating both arms to the common sample count).
Raw table: `scratch_gapsweep_results.log`.

**Tally over 9 datasets × 5 ratios = 45 cells per config:**

| config | gate σ | separated + | separated − | not separated | mean Δ |
|---|---|---|---|---|---|
| `gapf` dist | 0.032 | 4 | 5 | 36 | **−0.05** |
| `gapr` dist | 0.014 | 0 | 1 | **44** | **+0.08** |
| `gapr` point | 0.024 | 1 | 2 | 42 | **−0.00** |

**The aggregate effect is zero to two decimal places.** The only structure is dataset-specific
and self-cancelling: `gapf` is significantly **positive** on `scbench_vt` at 4 of 5 ratios
(+1.69 / +3.51 / +2.31 / +2.31) and significantly **negative** on `scbench_prefix_suffix` at all
5 (−2.60 … −5.00). `gapr dist` — the ckpt whose gate trained itself *below* its initial value —
is not separated in 44 of 45 cells.

Together with `scbench_kv` (the only dataset with real headroom, also null), **the three current
checkpoints are indistinguishable from the baseline across 10 datasets**. This is now a complete
negative result on the 16-Gaussian-slot + residual design, not a partial one.

Note `scbench_vt` is the dataset whose baseline *improves* under compression (41.07 full → 46.09
at ratio 0.2, i.e. negative headroom), so `gapf`'s gain there is more plausibly mild denoising
than information recovery — do not cite it as evidence the method works.

Measured per-dataset cost for one config over 5 ratios, useful for planning any future grid:
`repoqa` 5.83 h, `prefix_suffix` 3.23, `mf` 2.97, `vt` 2.20, `summary` 1.88, `gsm` 1.19,
`qa_eng` 0.60, `squad` 0.55, `choice_eng` 0.44 — **18.9 GPU-h per config for those 9**, plus
~5.8 h for `scbench_kv`.

MRCR cannot join this table: it runs `eval_chunk_mrcr.py`, and the VariKV injection was never
wired into that path. So the ceiling for these sweeps is 11 of Figure 11's 12 panels.

### In flight as of 2026-08-11 04:30 UTC

Both runs use the three `gap_*` ckpts with `--varikv_residual`, tags `gfsd` / `grsd` / `grsp` (distinct per ckpt because `gap_fix03/dist` and `gap_rand/dist` are both `dist` mode and result dirs carry only the mode).

- ~~`scratch_gapstd_eval.sh` — the three ckpts × `scbench_kv` × standard interval.~~ **Finished 11:22 UTC, 7h10m, all three `rc=0` — results in the table above.**
- `scratch_gapsweep.py` — the three ckpts × the other 9 datasets, 27 jobs, marker-resumable, longest-first. Baselines are **not** re-run (the `_full` tag from `scratch_stage2b_sweep.py` is the same configuration). 56.7 GPU-h total; workers on GPUs 0–2 wait for the `scbench_kv` run to print `ALL DONE` before taking work. ETA ~13:30–14:00 UTC.

## From “2026-08-12/13 — the teacher-KL round”

> The result tables and the forensic probes. The section's two headline findings, what was built, the standing warnings and the scheduler lesson stayed in `CLAUDE.md`.

### The one positive result, and its four limits

`ckpt_kl/dist` on **Retr.KV** @ratio 0.1: 32.60 → **54.20** (+21.60, CI
[+15.20,+27.60], HRR 60.7%). Same architecture and same eval as the `lm`-objective
checkpoint that scored −43; only the objective changed. **So "was the training wrong"
is answered: yes, it was a first-order cause.**

Then it fails four ways:

1. **It does not generalize.** Eight panels, each with its own same-batch ratio-0.1
   baseline: **1 significantly positive, 1 significantly negative, 6 unseparated,
   mean Δ +1.41.**
2. **"No headroom" is not the excuse.** Retr.Prefix-Suffix (+41.40) and Code.RepoQA
   (+46.35) have *more* headroom at ratio 0.1 than Retr.KV (+35.60) — their baselines
   collapse to 8.60 and 12.71 — and the memory recovers nothing (−0.60, −0.71).
   So the selectivity is about **what kind of content must be recovered**.
3. **It actively harms a panel where compression helps.** **Retr.MultiHop**'s
   ratio-0.1 baseline (49.47) beats full cache (41.07); the memory drags it to 31.11,
   **−18.36★, ten points below full cache.** The design has no mechanism for deciding
   whether to speak — the gate is one constant per (layer, kv-head) and never sees the
   query. Same disease as the centroid run's 36-better/20-worse split.
4. **It has not been reproduced.** `ckpt_kl_v2a` (fixed code, byte-identical sampling)
   scores **−13.20★**. That run is confounded by a `min_chunks=1` default that silently
   cut the corpus from 34 documents to 14, which is why `ckpt_kl_v2b` exists.

### "Distributional beats point" now has zero support

| | `dist − point` | training data |
|---|---|---|
| v1 | **+39.60 [+33.60,+45.80] ★** | sampling **not** matched (no seed, two processes) |
| v2a | +2.80 [−1.40,+6.80] **unseparated** | byte-identical |
| v2s | −2.60 [−5.80,+0.60] **unseparated** | byte-identical |

The v1 gap was sampling noise plus gate amplitude: point's gate learns σ=0.265 against
dist's 0.131, and point's generations are 48.9 characters against the baseline's 120.5
— degenerate output. Combined with the four-way difference between the two modes
(see `varikv/memory.py`'s header), **the claimed contribution remains unsupported.**

### Streaming training made things worse, not better

`max_ctx=32768 / chunk=16000` gives exactly 2 chunks and **one** eviction per step, so
v1/v2a never exercised streaming at all (measured: 1.03 prune_chunk calls per prefill).
`ckpt_kl_v2s` fixes that (10 long documents, 5 chunks, 4 evictions) and is **worse**:
validation recovery −145.7% and downstream ≈ baseline. Corpus constraint worth knowing:
**all 68 `fineweb_10k` documents are under 32,256 tokens**, so at chunk 16000 more than
one eviction can only come from `fineweb_10k_cat`, which holds 10 documents of 103k–122k
(not the 5 that `feature.py` takes). Hence `--n_short` / `--n_long`.
`--detach_every 4` OOMs: `kl_to_mixture` builds a `[B,G,N,K]` tensor that is 7 GB alone
at N=16000.

### The training-free centroid arm is the one clean positive

On Retr.KV @0.1 against an **equal-byte** control (spend the same bytes retaining more
exact KV, ratio 0.1061 → 35.60): K=16 gives **+6.60★**, K=1024 **+8.00★**. Against the
plain baseline that is +9.60 / +11.00. Two facts fall out:

- **A count-aware centroid costs 257 scalars against an exact KV entry's 256**, so
  "K centroids vs K exact KV" is a fair fight at every K, and retained is 16,903 per
  (layer, kv-head) at ratio 0.1 — so even K=1024 is only +6.08% of the budget.
- **Capacity is not the bottleneck**: 64× more capacity buys +1.40 points. What mattered
  was the algebra — `log n_j` alone is worth **67×** in recovered missing mass (true
  median 0.715, centroid estimate 0.239, drop `log n_j` and it collapses to 0.0037).
- **Naive post-RoPE averaging beats the theoretically-correct position-free frame**
  (+6.80 vs +1.20, the latter unseparated), even though the fastest `inv_freq` component
  is 1.0 rad/token so any cluster wider than ~6 tokens fully decorrelates it. The
  averaging acts as an implicit low-pass filter that keeps only the phase-coherent
  components; inverse-rotating re-imposes a single phase on components that are already
  decorrelated. Do not "fix" this.

### The learned memory neither reconstructs the evicted KV nor repairs the local attention gap — and the second half of that took two probes to get right

`scratch_probe_forensic.py`, training-free, on the target 7B with the real gate.
**Everything is compared in the read-out frame** — the learned slots are `apply_rope`'d
to their position centroid exactly as `memory_residual` does at inference. This matters:
`memcache_retain` stores keys **pre-RoPE** (inverse-rotated on write) while
`centroid.py`'s default `post` mode stores them **post-RoPE**, and the two frames differ
by 74–93%, so a raw `cos(k_i, k̂_j)` between them would measure rotation rather than
content.

| metric | | **Retr.KV** (memory scores **+21.60**) | **Retr.Prefix-Suffix** (memory −0.60) | random baseline |
|---|---|---|---|---|
| **A. addressability** `mean_i max_j cos(k_i, k̂_j)` | learned | **0.0810** | **0.0745** | **0.1545** |
| | centroid | 0.7689 | 0.7829 | |
| **C. value direction** `cos(m(q), o_E(q))` | learned | **−0.0124** | **0.0208** | ~0 |
| | centroid | 0.7864 | 0.7910 | |

(140 and 133 (layer, head, group) triples; the random baseline is `max` over 16 random
unit vectors in R^128, which is 0.1545 — so **the learned memory is *below* chance**.)

**The learned memory does not reconstruct the evicted KV, anywhere — including on the
panel where it gains 21.60 points.** Its value output is orthogonal to the true evicted
attention output (−0.0124 on Retr.KV, i.e. very slightly *anti*-aligned). The
training-free centroid does reconstruct (0.77–0.79 on both) and gains only +11.00.

So the working checkpoint is not doing what the project's narrative says. But **"it does
not reconstruct KV geometry" does not by itself establish "it carries no functional
information"** — a learned slot has no obligation to look like an original key; it could
be a learned *address* that still routes correctly. Establishing the stronger claim needs
a functional target, and my first attempt used the wrong one.

#### The probe's own target was wrong the first time — `o_E` instead of `Δo`

The residual read-out is `o = o_R + g·m(q)` while full attention is
`o_full = λ·o_R + (1−λ)·o_E`, so

    o_full − o_R = (1−λ)(o_E − o_R) ≡ Δo

**The memory must approximate `Δo`, not `o_E`.** The two can point in completely different
directions: with `o_R=[10,0]` and `o_E=[8,2]`, `o_E` points roughly right while
`Δo ∝ [−2,2]` does not, so `cos(m,o_E)≈0` is perfectly compatible with
`cos(g·m,Δo)≈1`. Metric C above would call a *perfect* residual correction noise.
This identity is used elsewhere in this repo (`scratch_probe_damage.py`, and the
"exact local counterfactual identity" section of this file) — the first forensic simply
aimed at the wrong quantity. `scratch_probe_forensic2.py` is the corrected version:
target `Δo`, everything projected through `W_O`, and reported both per-head and
summed over heads.

| corrected metric (median) | **Retr.KV** (+21.60) | **Prefix-Suffix** (−0.60) |
|---|---|---|
| **D1 direction** `cos(W_O δ̂, W_O Δo)`, learned | **−0.0056** | **+0.0033** |
| **D1 direction**, centroid | **0.8465** | **0.9059** |
| **D2 magnitude** `‖W_O δ̂‖/‖W_O Δo‖`, learned | **1.1667** | 0.8266 |
| **D2 magnitude**, centroid | 0.0713 | 0.1227 |
| layer-level `cos` after summing heads, learned | 0.0108 | 0.0369 |
| layer-level `cos`, centroid | 0.6803 | 0.7127 |

**The learned correction is orthogonal to the true local gap, with roughly the right
magnitude** (ratio 1.17 / 0.83) — per head and after cross-head summation alike. The
centroid is the mirror image: right direction (0.85–0.91) at 7–12% of the needed
magnitude, which is exactly why it recovers only 11.00 / 3.60 points.

So: **the +21.60 on Retr.KV is not obtained by repairing the local attention gap.** The
memory injects a vector of about the right size pointing somewhere unrelated to what
eviction removed, and the benchmark score goes up. The mechanism is unexplained.

#### Two claims of mine that this retracts

- ~~"the shortcut hypothesis is established"~~ — not established. What is measured is
  that the correction is neither a KV reconstruction nor a local-gap repair. Whether it
  encodes transferable evicted content in some other basis is open.
- ~~"add `L = KL + λ·L_structure`, the centroid is a ready-made teacher"~~ —
  **do not do this.** `L_structure` pulls `δ̂` toward `Δo`, i.e. toward the centroid's
  direction, and the centroid scores **43.60** on Retr.KV against the learned memory's
  **54.20**. Forcing the alignment would most likely drag 54.20 down toward 43. This was
  the pre-registered second branch of `forensic2` and it is the branch that fired.
- The `d_z` reading is also retracted: both panels sit at or below chance on
  addressability while one gains 21.60 downstream, so addressability has no causal
  relation to the score, and a per-token autoencoder would in any case only test
  `256→64→256` rather than the hard part, `N tokens → 16 slots`.

#### The three live hypotheses

| | status |
|---|---|
| H1 functionally-correct residual representation | **refuted** by `forensic2` (D1 ≈ 0) |
| H2 representation shortcut / non-local compensation | **open, now the leading candidate** |
| H3 v1 was simply a lucky training trajectory | **CONFIRMED 2026-08-14 — see below** |

## 2026-08-14 — the +21.60 does not reproduce: it was training-run variance

Two independent retrains of **byte-identical v1 code** (git worktree at
`/home/ubuntu/zxy/vlm-memory-repro21`, checked out to v1's commit so no current-tree
change can leak in; driver `scratch_repro21.sh`), evaluated on Retr.KV @ratio 0.1 against
the same `__r05b` baseline with the same paired bootstrap:

| run | n | absolute | paired Δ vs baseline |
|---|---|---|---|
| original `ckpt_kl/dist` | 100 | 54.20 | **+21.60 [+15.20,+27.60] ★** |
| replicate r1 | 100 | 15.20 | **−17.40 [−23.20,−12.00] ★** |
| replicate r2 | 96 | 22.29 | **−8.75 [−14.38,−3.33] ★** |

**Three draws of the same code span 39 points, and all three CIs exclude zero in
disagreeing directions.** v1's training was unseeded (no `manual_seed`, and the fineweb
sampler draws per-process), so the replicates are legitimately different trajectories of
the same procedure — which is exactly what makes this decisive: the procedure's
*downstream score* is not a property of the method, it is a property of the draw.

What this settles, and what it costs:

- **The one positive result in the project is retracted.** Every statement of the form
  "answer-token KL distillation recovers 21.60 points on Retr.KV" must be read as one
  sample from a distribution that also contains −17.40. `ckpt_kl_v2a`'s −13.20 was never
  a "failure to reproduce due to the `min_chunks=1` corpus bug" — it is an ordinary
  member of the same spread, and the corpus bug is a red herring for this question.
- **H2 is dissolved, not answered.** `forensic2` measured a correction that is orthogonal
  to `Δo` yet coincides with +21.6 points, and I called that "the mechanism is
  unexplained". There is no longer a coincidence to explain: a random-direction
  correction of roughly the right magnitude produces a *random* score change, and we
  happened to see the top of the range first. The forensic measurements themselves
  (D1 ≈ 0 for learned, 0.85–0.91 for centroid) stand — they were run on fixed
  checkpoints and do not depend on this.
- **`dist` vs `point` is unaffected** — it was already unsupported (v2a +2.80, v2s −2.60,
  both unseparated), and this only explains why v1's +39.60 looked so large.
- **The training-free centroid is now the project's only surviving positive result.** It
  has no training trajectory to be lucky in: it is deterministic given the data, and its
  11-panel mean of +3.66 at ratio 0.1 was measured with per-panel paired bootstraps.

Methodological rule this buys, to be applied to every future learned arm: **a single
training run is not a measurement.** Report n≥3 seeds with the across-seed spread, or
report nothing. The threshold for "significant" here was never the paired-bootstrap CI
over *samples* — that CI was correct and still misled, because it quantifies sampling
noise in the evaluation set while the dominant variance was in the optimiser.

---

## 2026-08-16 下午 —— `U^NLL` oracle 失败了，而它失败的方式是本轮最有价值的结果

### 起因

`FINDINGS_DENOISING.md` 与当天上午的保真度 2×2 已经测出：Retr.MultiHop 上残差
**显著更忠实于满缓存**（KL 0.2575 → 0.1779，t=−3.49）却掉 9.96 分。而项目现有的两个
教师靶子 `U^full` / `U^setmarginal` 的目标函数都是 `F(S) = −‖W_O(o_full − o_S)‖²`，
最优解都在满缓存 —— 所以它们在 MultiHop 上教的方向被证明是错的。

自然的修法是换成未来预测损失 `L(S) = −Σ_j log p(y_j|S)`，它的最优解**不必是满缓存**。
但换教师要重跑教师 + 训练 + 下游。所以先用 brute-force 验证靶子是否真的错位：
`scratch_probe_nll_oracle.py`，每篇取全局阈值附近 32 个 `(层, kv头, token)` 候选，
逐个翻转 `kv.valid` 那一位重算答案 NLL。

**预注册判读**：Retr.KV 强正、MultiHop 弱或反号 ⇒ 靶子错位是 −9.96 的原因；两个 panel
都强正 ⇒ 靶子不是病根。

### 结果落在第三种：两个 panel 都是零

20 篇 × 32 候选（scbench_kv 共 100 条、scbench_vt 共 90 条，各取前 20）：

| panel | 合并 ρ(U^NLL, U^attn) | p | 逐样本中位 | 样本为正比例 |
|---|---|---|---|---|
| Retr.KV | +0.0321 | 0.42 | +0.0089 | 50.0% |
| Retr.MultiHop | −0.0128 | 0.75 | +0.0468 | 55.0% |

一个结构性差异比相关系数更能说明问题：`U^attn` **90.9% 为正**（构造使然——它的最优解
是满缓存，删任何东西几乎恒为负收益），而 `U^NLL` 只有 **49.5% 为正**（阈值附近的候选
里一半删掉反而更好）。

### 但这个零结论**作废**，因为标签本身没有信度

`scratch_probe_nll_stab.py`，三个对照，6 篇 × 24 候选：

```
A 确定性   同掩码两次 NLL |Δ| = 0.000e+00（6/6）      ⇒ 没有数值底噪
B 块级     去掉 256 条：top −0.0026±0.0073  bot −0.0002±0.0023  rand +0.0054±0.0260
C 可复现   ρ(U^NLL(S), U^NLL(S')) 合并 −0.2193，逐样本中位 −0.089
           6 篇: −0.17 −0.16 −0.19 −0.02 +0.16 +0.03
```

C 是决定性的，且不看 ρ 也能读出来：

```
std(U_S − U_S')     = 6.44e-3
√(var_S + var_S')   = 6.22e-3     ← 两个**完全独立**的量应有的差
```

**两次测量之差比"两个独立随机量之差"还大 ⇒ 共享成分 ≤ 0。** 同一条 KV，把背景换掉
1%，测出来的"效用"和它自己毫不相干。

于是 oracle 的零相关**不含任何关于靶子对错的信息**：观测相关最多约为真相关 ×
√(标签信度)，信度 ≈ 0 时观测必然是 0（经典衰减）。**这次实验作为仪器失败了，而不是
作为结论为负。** 早先写下的"两个 panel 都是零 ⇒ 注意力靶子与真实预测效用无关"当天
即撤回。

**对照 B 的设计错误也记下来**：`G=256` 只占保留集的 **0.0139%**（`level="pair"` 下
保留集是 28 层 × 4 头 × ~165k × 10% ≈ **1.85M 格**）。三臂测不出差别是必然的。挑块
大小要按保留集**总量**的比例，不要按"每头多少条"的直觉。n=1 冒烟时那个抢眼的
`top −0.0237` 是假象，n=6 就没了。

### 当天下午撤回：对照 C 自己的量纲也是错的

GPT 复查时指出，`--perturb 0.01` 取的是**全体可驱逐格子**的 1%，而不是保留集的 1%；
而且它是**直接 toggle**，不是等量互换。核算（有效 chunk_ratio 0.078，不是名义 0.1）：

| | |
|---|---|
| 全体可驱逐格子 | 28×4×165k ≈ **18.5M** |
| 保留集 `\|S\|` | ≈ **1.44M** |
| 翻掉 | 185k = 保留集的 **12.8%** |
| 被翻的 ~90% 原为驱逐态 ⇒ 预算 | 1.44M → 1.60M，**+10.8%** |

所以首版比较的根本不是"两个邻近的等预算存活集合"，而是"10% 预算的缓存"对
"约 10.8% 预算、且随机塞进 17 万条低分 token 的缓存"。**这与当天上午在对照 B 上
抓到的是同一个量纲错误**（`G=256` 只占保留集 0.0139%）—— 抓到了一次，紧接着在
另一个对照上原样又犯一次。

`ρ = −0.22` 因此只支持"单条效用对缓存构型敏感"，**不支持**"等预算局部扰动下没有
稳定成分"。以下两条当天写下、当天撤回：

> ~~逐 token 边际标签这条路对任何方法都堵着，包括 ForesightKV 与 KVP~~
> ~~+4.27 不可能是"token 重要性"，只能是集合级/预算级的量~~

除了量纲，这两条在推理上也各有独立的漏洞：

1. **实现值抖 ≠ 期望不可学。** `U_i(S)` 不稳定只说明不存在 set-independent 的
   token 内在边际；`E_S[U_i(S)|x_i]` 仍可能是稳定靶子（Shapley 与 leave-one-out 的
   经典区别）。噪声抬高的是方差不是偏差，代价是所需数据量 ∝ 1/信度 —— 那是个可测的量，
   不是定性的"堵死"。
2. **ForesightKV / KVP 与本探针不是同一个数学对象。** 它们的监督是**满缓存上的未来
   注意力**，一次算完、不依赖存活集合；本探针测的是 set-marginal。这条外推是错的。
3. **最直接的一条**：被训练的打分器根本不以 `U^NLL` 为标签，它用的是 `U^attn`——那个量
   **给定 S 是闭式确定的**。所以 `U^NLL` 的信度对"+4.27 是什么"没有直接约束力。

**站得住的只剩一条**：`U^NLL` oracle 的零相关不能读成"注意力靶子与预测效用无关"，
因为观测相关 ≈ 真相关 × √信度，而信度没有独立测过。**那次实验作为仪器失败。**

### 替代实验（`scratch_probe_nll_stab.py` 已重写）

1. **严格等预算互换**：从 `S∖cand` 抽 `n_swap` 踢出、同时从 `S̄∖cand` 抽 `n_swap`
   放进，`|S'| = |S|` 用断言保证。
2. **ε 相对 `|S|`**，扫 0.1% / 0.5% / 2% / 10%，输出的是**信度曲线**而非单点。
   最小 ε 就 ρ≈0 才是首版想证的那件事；ρ 随 ε 单调衰减则是"局部稳定、全局
   set-dependent"，教师可用但需按 1/信度 放大数据量。
3. **同一批扰动下同时测 `U^attn` 的信度**（闭式，几乎免费）。这是最有决策价值的一格：
   `U^attn` 高而 `U^NLL` 低 ⇒ 教师标签自洽但它代理的东西不自洽（真·靶子问题）；
   两个都低 ⇒ 教师标签自身就抖。
4. 两档扰动分布：`random`（全局 set sensitivity 上界）与 `boundary`（只在阈值邻域
   互换 —— 真实 reranker 只改动决策边界附近的成员，这才是 v2 残差的工作点）。

**+4.27 的机制改由 `calib_scorer.py` 四臂裁决，在它们跑完前不写结论。**
配套的 `scratch_probe_armdiag.py` 补两个机制量，让"affine≈full ⇒ 是校准"从暗示变成
证据：头内 Kendall τ / 逐对翻转比例（`bias` 构造上恒为 0，但 `affine` 的
`ds'/ds⁰ = 1 + α a_h sech²(·)` 在 `a_h < −1/α` 时会局部反号，必须实测），以及逐
(层,头) 的预算再分配 `|ΔB|/B`。

### 立住的那条，比原本想问的问题更重要

> ~~**在驱逐阈值附近，"这一条 KV 的效用"不是一个良定义的量。**~~
> **↑ 当天下午撤回，见上一节。** 对照 C 的扰动是保留集的 12.8% 且改了 10.8% 预算，
> 只能支持"对缓存构型敏感"，不能支持"等预算局部扰动下无稳定成分"。

当时据此写下的三条后果（**均已撤回**，理由见上一节）：逐 token 教师全被堵死、
ForesightKV/KVP 同受影响、+4.27 不可能是 token 重要性。与 **Error Certificates
(2607.21475)** 的类比也随之降级为"待验证的联想"——那篇的不可辨识是"服务端没保留
足以估计误差的信息"，形式本就不同，当时就标注了是类比不是证明。

### 由此推出的下一问

效应真实存在（Retr.KV 满缓存 68.20 → ratio 0.1 约 45），只是不驻留在单条上。
**那它驻留在什么尺度？** `scratch_probe_nll_grain.py` 把去掉的比例从 1e-5 扫到 0.2，
看 `top` 与 `bot` 从哪个粒度起相对同粒度 `random` 的散布分得开。那个最小粒度就是
新教师该用的粒度。若要到 1e-2–1e-1 才分得开，逐 token 标签注定被噪声淹没。

---

## 2026-08-17 —— 拆解四臂重做在 v2 档上，+4.27 的机制有了答案

### 为什么重做

既有的 `dec_*` 四臂训在**扩容后的 30 篇**上（23/7 划分、40×23 = **920 步**），而 v2
训在 **10 篇**上（8/2、40×8 = **320 步**）。要解释 v2 的 +4.27，消融臂必须与它同数据、
同划分、同步数、同教师靶子，否则"架构差异"与"3 倍数据 + 2.9 倍梯度更新"混在一起。

中途还查出一个更根本的错：**曾经建议"四臂的匹配参照用 v3"，那是错的。** v3
（`ctrl_smc_s*`）的 trace 目录是 `scratch_ctrl_traces_sm_cont`、教师靶子是
`--utility set_marginal`，与四臂用的 `U^full` 根本不是同一批数据，只是篇数碰巧同为
23/7。所以旧四臂**从来没有匹配的分母**。

### 数据档怎么锁死的

新建 `scratch_ctrl_traces_v2_10/`（10 个符号链接，不复制 3 GB），并用 sha256 逐个核对
= v2 训练时读到的那 10 篇。证据链：十个文件的 **ctime** 都是 12:39:18–12:41:56，
早于 14:00 的 v2 ckpt 且未被 14:10 的扩容触碰；教师日志两次运行 `>>` 共享，每个
`docNNN` 只出现一次 ⇒ 扩容只写了 doc010–029。摘要现已写进 `varikv_v2.py:V2_TRACES`，
从此是硬失败而非静默换数据。

训练参数与 v2 逐项相同（从 v2 ckpt 里存的 `args` 读出来比对），**只多一个 `--arch`**；
每个训练日志都验证过是 `训练 8 篇 / 验证 2 篇`（验证集 doc003 + doc007）。

### 结果（3 种子，分母是 v2 自己的三个种子）

Retr.KV @0.1，基线 32.60，n=100：

| 臂 | 参数 | s0 | s1 | s2 | 跨种子 |
|---|---|---|---|---|---|
| `bias` | 225 | +0.20 | +1.00 | −0.20 | +0.33 ± 0.50 |
| `affine` | 225 | +0.00 | −0.20 | +0.20 | +0.00 ± 0.16 |
| **`scalar`** | **4,482** | +4.20★ | +5.20★ | +4.80★ | **+4.73 ± 0.41** |
| `kv` | 53,378 | +4.40★ | +4.60★ | +4.40★ | +4.47 ± 0.09 |
| **v2** | 637,828 | +4.40★ | +4.40★ | +4.00★ | **+4.27 ± 0.19** |

@0.2：`bias` +2.07±0.19、`affine` +0.13±0.62、`scalar` **+20.80±0.65**、
`kv` +20.27±3.62、v2 +18.80（仅 1 种子——v2 的三个种子只评过 0.1）。

种子级配对（3 对 3）@0.1：

```
scalar − bias     +4.40 ± 0.43   逐种子 +4.00 +4.20 +5.00
scalar − affine   +4.73 ± 0.50   逐种子 +4.20 +5.40 +4.60
kv     − scalar   −0.27 ± 0.34   逐种子 +0.20 −0.60 −0.40
scalar − v2       +0.47 ± 0.47   逐种子 −0.20 +0.80 +0.80
kv     − v2       +0.20 ± 0.16   逐种子 +0.00 +0.20 +0.40
```

### 读法

1. **4,482 参数追平 637,828（142×）**，两个 ratio 都不可分且一致略高。
2. **KV 内容不必要**：`scalar` 看不到 K/V 却与 `kv` 臂不可分，且更稳
   （0.2 上 ±0.65 vs ±3.62，`kv` 的 s1 掉到 +15.20）。
3. **不是简单校准**：`bias`/`affine` 几乎精确为零 ⇒ 位置+尺度表达不了这个增益。

`scalar` 比 `affine` 多的只有 `margin = (s⁰−τ_global)/σ_g` 与 `log(σ_h/σ_g)` ——
它知道**全局那一刀切在哪**，能只在决策边界附近做局部形变。下一个消融
（`scalar` 去掉 `margin`）能直接判定这条。

`armdiag` 在 30 篇档上测过 `bias`/`affine`/`scalar` 的头内 Kendall τ = **1.000000**、
逐对翻转恰为 0（只有 `kv` 会重排序，τ 最低 0.153）；v2 档版本还没重测。

### 顺带三个工程教训

- **`kill -0` 对僵尸子进程返回成功**，所以"按 PID 回收 GPU 锁"永不释放 —— 8 张卡
  全上锁而卡上无进程，下一轮 `gpu_claim` 永久阻塞。改成整轮 `wait` 后统一释放，
  再改成 `flock` 工作池（无栅栏，卡一空就补任务）。
- **Bash 工具调用超时会 SIGTERM 整个进程组**，`nohup ... &` 的子进程一起死。要
  `setsid` 脱离，且最好由一个独立的 kickoff 脚本发起。
- **`ratio × clen < window` 时 `chunk_ratio` 被置 0**，保留集变成"最后 `ratio×clen`
  个 token"——预算照给，但**选择从按分数变成按最近**，改分数的方法恒为 no-op。
  门槛 `clen > 4096/ratio`：0.2 只有 gsm/squad 过不去，0.05 卡掉 repoqa，
  **0.02 全部过不去**。这与"精度崩掉"是两回事，FastKVzip 图 11 在那些点分数好看是正常的。

---

## 2026-08-17 下午 —— 机制探针三连，以及干净版 v2 的一面红旗

### 单调性：从抽样升级到网格级证书

外部复查指出 `armdiag` 的 τ=1.000000 是**经验观察**不是数学保证。对，而且比批评的更弱：
它逐 (层,头) 抽 **2 万对**，每头约 15 万 token（全对 1.1e10），只支持"翻转率 < 5e-5"。

标量族可以证得更硬：头内固定 chunk 时 `mg = A·z+B`、`rs`、`e` 都是常数，网络退化成
一元函数，于是 `ds'/ds = 1 + α·sech²(φ)·φ'(z)`（σ_h 约掉）。`φ'(z)` 用 autograd 算，
**必须是全导数** `∂φ/∂z + A·∂φ/∂m`——有限差分验过（差 2.6e-4；漏 margin 通道会差 1.7e-2）。

v2 档、896 组真实状态 × 4001 点网格：`scalar` 三个种子 min ds'/ds = **+0.190 / +0.038
/ +0.091**，0/896 非单调；`bias` 恒为 1（`a ≡ 0`）；`affine` 解析下界 0.908。

**s1 余量只有 0.038** ⇒ 单调是优化器碰巧落在那一侧，不是架构性质。最终定型应写成
构造性单调的参数化。

### 措辞修正：affine 不是静态方法

之前把对立轴写成"static vs runtime-conditioned"是错的。`affine` 通过
`z=(s−μ_h)/σ_h` 和输出 `×σ_h` 已经自适应运行时**局部**统计量；缺的是 `τ`、`σ_g`。
正确的轴是**局部标准化 vs 实际全局竞争状态**。

### state-aliasing 假说：测了，不成立（至少在逐 token 层面）

假说：同一个 `z` 在不同竞争状态下被混叠，`(A,B)` 解除混叠。用条件方差下降度量，
**加置换对照**（(A,B) 在同头各 chunk 间打乱，保留分箱与样本量）——不加对照的话，
多加条件变量必然机械降方差，裸比值没有意义。

63 万个近阈值候选：`Var(U|z,l,h) = 1.70e-4` → `Var(U|z,l,h,A,B) = 1.66e-4`，
裸 R = 0.9749，置换对照 R_shuf = **1.0025 ± 0.0002**，超出对照的下降 **+2.8%**。

方向为正且与对照干净分离，**但只有 2.8%**。所以：若因子消融仍显示 `szm ≫ sz`，
机制必须在**集合/预算层面**解释（哪个头能过线几个），不能讲成"解除逐 token 混叠"。
这条把假说空间收窄了，是有用的负结果。

### 干净版 v2 的红旗

训练侧与原版**完全重叠**（末 epoch：loss 0.6444/0.6447/0.6426 vs 0.6448/0.6447/0.6422，
全局Δacc +0.0130/+0.0157/+0.0133 vs +0.0160/+0.0140/+0.0140）。

但下游 Retr.KV：干净版 **+1.40（不显著，n=1）** vs 原版 **+4.27 ± 0.19★**；
@0.2 是 **+13.80★** vs **+18.80★**。

现在只有 s0 跑完，不能下结论；而训练侧与下游历史上是**反相关**的
（`ckpt_kl_v2a` 验证第一、下游最差），所以训练侧吻合救不了这条。唯一已知的差异是
**新训练的随机初始化与原版不同**（原版构造那 8 个死模块时消耗了随机数）。
三个种子齐了必须查清——若都在 +1~2，后面所有基于干净版的结论都悬空。

### 阴性对照通过

`ρ=0.02` 在 Retr.KV / PrefSuf / Math.Find 上恰好 **+0.00**，正是构造性预测值
（`ratio×clen < window` ⇒ 只留最近的 token，门控分数不参与）。
SQuAD/GSM8K 非零是因为只有 203/86 token，压根不进驱逐路径。

---

## 2026-08-17 中午 —— 一个 CLI 默认值把整批评测跑成了另一个方法

### 起因

干净版 v2 训练侧与原版**完全重叠**（末 epoch loss 0.6444/0.6447/0.6426 vs
0.6448/0.6447/0.6422），下游却只有 **+1.40**（Retr.KV @0.1，n=1）而原版 +4.27★。
我把嫌疑排在随机初始化上。**排错了。**

外部复查指出 `args.py` 的 `--ctrlm_mode` 默认 `"stateful"`，而 `eval_chunk.py` 用
`args.ctrlm_mode or _ck.get("mode")` —— 非空字符串恒为真，ckpt 存的 mode 永不生效。
日志一看即实：`v2c_* → mode=stateful`，`ctrl_b_a1_* → mode=memoryless`。

原版逃过是因为它的启动脚本显式传了 `--ctrlm_mode memoryless`；四臂逃过是因为
`eval_chunk.py:142` 对 `arch != "memory"` 强制改回 memoryless。**只有我写的
`scratch_master_queue.sh` 什么都没传。**

### 代价实测（同 ckpt 只换 mode，Retr.KV，n=40）

| ratio | memoryless | stateful | 差值 | 逐样本相同 |
|---|---|---|---|---|
| 0.1 | **+5.50★** [+2.50,+8.50] | +0.00 [−3.50,+3.50] | **+5.50★** [+2.00,+9.50] | 27/40 |
| 0.2 | **+25.50★** [+18.50,+32.50] | +18.00★ [+10.50,+25.50] | **+7.50★** [+2.50,+13.00] | 21/40 |

**干净版 v2 修好后 +5.50★，不低于原版的 +4.27 ± 0.19★。** 那个红旗完全是这个 bug。

为什么 stateful 下会崩：`to_compat_ckpt` 把 8 个 writer 模块**填零**（注释写了
"memoryless 下它们不参与"——假设被陈述但没被强制）。stateful 下 writer 真的执行，
全零 GRU 给出 `r=z=σ(0)=0.5`、`n=tanh(0)=0` ⇒ `M ← 0.5·M`，11 个 chunk 衰减到
`2⁻¹¹ ≈ 5e-4`；`dir_decay=0` 使 `D ← 0.5·D + 0.5·mean(x)` 吸进真实数据。

### 我在这条上判断错了两次

1. 只凭机制推理就写"这完全解释了 +1.40 vs +4.27，根本不是初始化方差"——当时**没有
   任何测量**。
2. 紧接着看到 **n=6** 的部分结果（两 mode 前 6 个样本逐位相同）就说"不支持我的预测"。
   n=40 上差异清清楚楚（+5.50★）。

第二次尤其该记：项目早有「Never trust a ★ on partial samples」的规矩，但我只在**正
结论**上守它，看到部分样本上的"无差异"时却当成了负证据。**部分样本不足以支持任何
方向的结论。**

### 顺带修好的

- `--ctrlm_mode` 默认改 `None` + `is None` 解析 + 覆盖时告警；`eval_chunk_mrcr.py` 同步。
- 教师侧加 `--regime {v2,v3,custom}`，从**源头**声明语料，带篇数断言（上游漂移当场停机）；
  每个 trace 存 `regime` + 原文 sha256，自描述。
- `scratch_ctrl_teacher.py --help` 一直是崩的（`--n_keep` 的 help 里有字面量 `0.895%`，
  argparse 会做 `%` 格式化）。既存 bug，转义修掉。
- `scratch_pool.sh` 的 `eval "CUDA_VISIBLE_DEVICES=$G $j"` —— 作业串是
  `cd X && env ... python`，而 `VAR=x cmd1 && cmd2` 里赋值**只对 cmd1 生效**，
  python 拿不到 CVD ⇒ torch 自己挑卡，实测 GPU1/GPU2 各叠 3 个进程、显存 34 GB。
  改成 `( export CUDA_VISIBLE_DEVICES=$G; eval "$j" )`，并验证语义（旧 `CVD=` 空、
  新 `CVD=7`）。**这不是调度器的锁失效** —— `gpu_claim` 的两道检查一直是对的。

### verify-backward：补上 verify-train 的盲区

`verify-train` 只比了前向 loss 与随机数流。两边完全可能算出同一个 loss 而梯度不同
（中间量被 detach、加法顺序不同）。**训练轨迹由梯度决定。** 共权重、同种子、同一篇
trace 下各跑一次 `backward()`：

    loss  原版 1.8265886307   本文件 1.8265886307   |Δ| = 0.00e+00
    共享参数 12 个全部有梯度   max|grad_orig − grad_v2| = 5.821e-11（q_read.weight）
    原版 8 个死模块的 17 个张量，收到非零梯度的：0 个

5.8e-11 是 float32 累加序的量级。而"memoryless 下那 8 个模块不参与"至此是**测出来的**，
不再是读代码的结论——那正是砍掉它们的前提。

---

## 2026-08-17 下午 —— 掩码重叠：一个规范自由度差点变成"发现"

### 起因

`scalar`（只看分数统计量）与 `kv`（只看 K/V）下游不可分（−0.27 ± 0.34）。外部复查
正确指出：**总分相同不等于同一个函数**（极端反例：一臂改对前 50 题、另一臂后 50 题）。
逐样本先测了一遍：Pearson +0.723、逐样本 Δ 完全相同 79/100，但"被改好"的样本
Jaccard 只有 0.500 —— 中间态，需要往下一层看掩码。

### 差点被当成发现的那个数

首版用 trace 存的固定 `τ` 判定 `s⁰+Δs > τ`，得到 **翻上 63 : 翻下 91,944**。
外部复查指出真实 Top-B 下 `|S'|=|S|=B`、两者必须相等，所以这个定义不是等预算交换。
**对，而且根因比"预算漂移"更根本：**

> `Δs` 的**全局常数偏移是规范自由度** —— 真实流水线在 `s'` 上重算阈值，给所有条目
> 加同一个常数对决策的影响**恰好为零**。用固定 τ 判定就是把一个数学上不可辨识的量
> 当成了真实效应。

实测：`Δs` 均值 −0.0665（= −0.76×标准差），而 τ 之上的候选离 τ 中位只有 0.00885，
`|Δs|` 中位 0.02299（2.6 倍）⇒ 上方的几乎全被压下去。**那个不对称不是伪影，是我的
度量把无效量当真了。**

### 改成等预算 Top-B 之后

| | 固定 τ | 等预算 Top-B |
|---|---|---|
| 翻上 : 翻下 | 63 : 91,944 | 34,469 : 34,469 |
| `Jaccard(F_scalar, F_kv)` | 0.9838 | **0.8645** |
| `J(S_scalar, S_kv)` | — | **0.8989** |
| 同向翻转占交集 | 1.000 | **1.000** |
| `Pearson(Δs)`（含/去全局均值） | +0.9782 | +0.9782 / +0.9782 |
| 随机基线 | 0.079 | 0.060（实测是其 14.4 倍） |

结论方向不变、强度下调。**措辞收紧为**："两臂在当前 trace 与决策边界附近产生了近乎
相同的有效选择修正策略"，**不是**"同一个机制"——0.865 意味着约 13% 的边界决策各走
各路，正对应 `armdiag` 里 `kv` 是唯一会头内重排序的臂（τ 最小 0.153）。

### 同时撤回一句

~~"逐头 MLP 也能从原始 KV 里把头级统计量再推出来"~~ —— **错**。`kv` 臂只看单 token 的
`(K_i,V_i)`，而 `μ_h, σ_h, σ_g, τ` 是十几万 token 的集合级量，单 token 无法恢复。
它多半学的是 `(K,V) → ŝ⁰` 的代理（头内 `z = A_h·s⁰ + B_h` 与 `s⁰` 排序一致，
不必知道 `μ_h, σ_h` 的数值）。这让 `kv` 臂**更有意思**，不是更平凡。

### 顺带更正一个统计量用错

前一条记录里"112 个码本常向量两两 cos 0.0707 ⇒ 几乎正交 ⇒ 重度依赖头身份"有两处问题：
(1) 0.0707 是**有符号**均值，而外部复查拿它比 `E|cos| = √(2/(π·128)) ≈ 0.0705`，
    那是**绝对值**均值的零假设，对不上；
(2) 更要紧的是**缺初始化对照**，而 `dir_type` 全局共享，初始化时就制造正相关。

补了对照（同构造 5 个随机种子）：初始化有符号均 **+0.1120 ± 0.0146**，训练后 +0.0707
——**训练后反而更低**。真正动了的是 `|cos|` 0.120 → **0.356** 与 P5 −0.03 → **−0.53**。
所以正确说法是训练把方向**分化**了（不是"正交"，`|cos|=0.356` 相当相关），而且
**这只证明 v2 学了逐头方向，不证明头身份必要**——`szmr0` 才是因果检验。

预注册保留：**预测 `szmr0 < scalar`**。这是预测不是证据。

---

## 2026-08-17 15:04 —— 干净版 v2 在关键 panel 上满 n=100 确认，红旗解除

`v2cbench_scbench_kv_s0` 跑完（9 个 ratio × 100 条 × 169k，约 8 小时）。Retr.KV 是
11 个 panel 里**唯一有真实 headroom** 的（满缓存 68.20 → ratio 0.2 只有 45.20），
也是 `+4.27` 的所在，所以这一格不等凑够 6 格就单独算了。

| ratio | 干净版 `v2c_s0` | 原版 `v2`（`__g8v2`） |
|---|---|---|
| 0.2 | **+19.00★** [+14.00,+24.00] | +18.80★ [+14.00,+23.60] |
| 0.1 | **+4.20★** [+1.40,+7.00] | +4.40★ [+1.60,+7.40] |

两者都是单种子、同一批 `__g8base` 基线、同一 bootstrap 口径 —— **几乎完全重合**。

至此「干净版 v2 复现不出原版」这条红旗**彻底解除**，它从头到尾就是 `--ctrlm_mode`
默认值把评测跑成 stateful 造成的（那次 n=40 的 A/B 已经量化过代价：memoryless
+5.50★ vs stateful +0.00）。现在 n=100 给 +4.20/+19.00，与 A/B 的 +5.50/+25.50
在噪声内一致（n=40 vs n=100）。

**六级等价性验收 + 这一格下游数字**合起来，`varikv_v2.py` 可以正式当作 v2 的实现来用：
423,298 参数（原版 637,828 的 66.4%），推理路径逐位等价，训练梯度差 5.8e-11。

---

## 2026-08-17 15:17 —— 保序重标定 ≡ 逐头配额分配：定理、验证、以及一条谁都没预料到的判决

外部建议提出「Quota Replay」为最高优先级因果实验：取 `scalar` 每头最终拿到的名额，
丢掉它的逐 token 修正，头内退回 FastKVzip 原排序，看能否复现 +4.73。

**这个实验对 `scalar` 是恒等式，不是实验** —— 因为它的前提我们已经证明了。

`calib_scorer.py:164-169`：`scalar` 族的 `raw` 只吃 `z, mg, rs, e_h`，全是 `s⁰` 与
头-chunk 常量的函数 ⇒ 头内是 `s⁰` 的一元函数。`scratch_probe_monotone.py` 早已给出
896 组 × 4001 点的网格单调性证书（0/896 非单调）。单调 + 全局 Top-B ⇒ 选出的集合
必然等于 `∪_h Top_{b_h}(s_h⁰)`。外部建议构造的反例（"在头内大幅重排"）对 `scalar`
**数学上不可能**。

还是跑了 `scratch_probe_quota.py`（零 GPU，3 篇 × 22 chunk，s0），因为它把网格证书
换成**真实候选分数上的实测**，并带一个天然阴性对照（`kv` 看不到 `s⁰`，可以重排）：

| 臂 | 配额重放逐位相同 | 头内逆序对 | **\|Δb\| 均值** | R²_static | 下游（3 种子） |
|---|---|---|---|---|---|
| `bias` | 22/22 | 0.0000% | 4.87 | 0.7190 | +0.33 ± 0.50 |
| `affine` | 22/22 | 0.0000% | **20.74** | 0.9028 | **+0.00 ± 0.16** |
| `scalar` | 22/22 | 0.0000% | **20.50** | 0.7575 | **+4.73 ± 0.41** |
| `kv` | **0/22** | **33.67%** | 19.93 | 0.7272 | +4.47 ± 0.09 |

**没预料到的那条判决在 `|Δb|` 那一列**：`affine` 搬动配额 20.74、`scalar` 搬动 20.50,
几乎等量，而下游 +0.00 vs +4.73。**搬多少不重要，搬给谁才重要** —— 这杀死了"增益来自
更激进的重分配"这个懒解释。两者唯一的差别是 `affine` 只适配局部 `(μ_h, σ_h)`，
`scalar` 还看到全局竞争态 `(τ, σ_g)`。

**`R²_static` 有非平凡零假设。** `bias` 是真正静态的平移，R² 也只有 0.7190 —— 因为
固定 `δ_h` 造成的配额变动取决于边界密度 `d_h = N_h f_h(τ)`，而它随 chunk 变。这是
Jacobian `db_h = d_h(dδ_h − Σ_j d_j dδ_j / Σ_j d_j)` 的实测显形。要按 0.719 当基准读：
等幅度下 `scalar` 0.7575 vs `affine` 0.9028。

**一处量纲更正**：`kvcache.py:160-176` 的 `prune_chunk` 只对当前 chunk 的
`evict_range` 定阈值，`cat` 到 `valid` 上，旧决策从不回溯 ⇒ 预算**逐 chunk 独立分配**,
自由度是 `b_{c,h}` = 11 chunk × 112 头 ≈ **1,232**，不是 112。

**同时收窄了 P1-2 的三条措辞**（先前写过头）：`+24.4σ` 的 σ 是置换零分布标准差不是
抽样标准误；`R²=0.2053` 还有 80% 方差没解释，不能说 `(A,B)` 是充分统计量；它是
**预测性**证据，"+4.73 因果上来自预算重分配"由上面的定理独立解决，不是由 P1-2。
`ΔB` 也不是不能用，是不能裸用 —— 应先用 `(B0, 头身份, f_h(τ))` 残差化。

**新颖性风险已记入 `ICLR_PLAN.md`**：Ada-KV / LKV / LU-KV / GraceKV / DBTrimKV 五篇
（据外部转述，**均未读原文**）已占据"学习/自适应头级预算"。读完之前不得声称分配是新的。

---

## 2026-08-17 15:45 —— 一个代码级更正，把「上下文依赖」从充分解释降回必要条件

外部复核指出：**`bias` 不是"跨 chunk 固定的平移"**。查 `calib_scorer.py:157,172` 与
`scratch_ctrl_teacher.py:450`（`sig_h = f0.std(-1)` 逐 chunk 现算）：

    Δs_{c,h} = α · σ_{c,h} · tanh(b_h) = σ_{c,h} · η_h

它是**固定的归一化偏好**，raw-score 平移量本身随 `σ_{c,h}` 变。**这个更正是对的，
而且它正好抽掉了「静态重分配买不到任何东西」的支撑** —— bias 本来就带 chunk 依赖，
它的 +0.33 还混着优化器有没有找到最优 `η` 的问题。

同时更正定理 2：先前写"给定任意 `b`，取 `δ_h = T − s_{h,(b_h)}` ⇒ 逐头平移用尽
整个空间"。**在有界约束 `|Δs| ≤ α σ_h` 下这是假的。** 正确版本是**有界决策等价**：
任何有界保序重标定产生的选择，都存在同界的逐头常数平移复现它（证明见 ICLR_PLAN
§四之五；关键是区间 `(T−s_{(b)}, T−s_{(b+1)})` 的两端分别被 `≤ a_h`、`> −a_h` 夹住）。

### 结构性判定取代训练性推断（`scratch_probe_static.py`，零 GPU）

正确的问法与训练无关：**存在任何固定 `{η_h}`（含最优的），使 `scalar` 的配额轨迹
可复现吗？** 约束对 `{η_h}` 与 `{T_c}` 都是线性的，加 `η_h ∈ [−α,α]` 就是 LP。

| 臂 | 归一化 `δ=σ_{c,h}η_h` 违反(σ)/条数 | 固定 raw `δ=δ_h` 违反(raw)/条数 | 下游 |
|---|---|---|---|
| `bias` | **0.0000 / 0 可行** ✓对照 | 0.4637 / 527 不可行 ✓对照 | +0.33 |
| `affine` | 10.2746 / **528** | 0.8084 / 446 | **+0.00** |
| `scalar` | 3.9547 / 107 | 6.3983 / 634 | **+4.73** |
| `kv` | 33.3347 / 755 | 9.9296 / 965 | +4.47 |

两个对照都按预期工作：`bias` 在归一化类下必然可行（`η=α·tanh(b)` 就是解），在 raw
类下必然不可行（它不是 raw 平移）。两块单位不同，不可跨块比。

**结论 5**：不存在任何固定策略能复现 `scalar` 的配额轨迹 —— 结构性的，无优化器混淆。

**结论 6，反过来打自己**：`affine` 偏离静态**更远**（528 条 vs 107 条），却拿 +0.00。
⇒ **上下文依赖是必要的，但不充分。** 先前写的"全部 +4.73 来自配额的上下文依赖性"
把必要条件当成了充分解释，**撤回**。决定性的是**方向**，不是幅度、也不是偏离静态的程度。

### 同时撤回/收窄的其它三条

- ~~`kv` 有重排能力却换不来任何东西~~ —— +4.47 可能是"配额 +4.5 + 重排 0"，也可能是
  "配额 +2 + 重排 +2.5"。要拆开必须对 `kv` 做真正的 quota replay。
- **独立自由度是 1,221 不是 1,232** —— `J·1 = 0` ⇒ `rank(J) = H−1 = 111` 每 chunk。
- **算力口径**：减少的是学习式校准网络的调用次数（O(N/C)），**不是端到端延迟**。
  基线打分、阈值、Top-B、KV 剪枝、注意力全不变。不得写 "16,000× faster"。

### 新增两条方法论警告

- **存在 ≠ 可学。** 定理 2 只说存在等价 `δ*_{c,h}`，没说它是 `(q, rs, e)` 的低复杂度
  函数 —— 真实配额还取决于经验 CDF（偏度、多峰、阈值处密度）。direct allocator
  第一版失败**不能**反推配额解释是错的。
- **网格不是证明。** 4001 点上 `ds'/ds > 0` 不等于区间上恒正，s1 余量只有 0.0384。
  需要 Lipschitz 上界补成严格证书，或最终采用构造性单调参数化。

---

## 2026-08-17 16:00 —— 第一个因子臂跑满：预注册预言 `sz ≈ 0` **被推翻**

`d10bench_sz_s0` 完成（n=100）。预注册的四结局判读表里，这一格判 A 还是 B。

| 臂 | 输入 | 函数类 | Retr.KV @0.1 |
|---|---|---|---|
| `affine` | `z` | 单个 `tanh(a·z+b)`，逐头 2 参数 | **+0.00 ± 0.16**（3 种子） |
| **`sz`** | `z` + 头嵌入 | **任意 MLP**（4,226 参数） | **+3.80★（n=100，仅 s0）** |
| `scalar` | `z` + **`mg`** + **`rs`** + 头嵌入 | 任意 MLP（4,482） | +4.73 ± 0.41（3 种子） |

**`sz` 完全看不到全局竞争态（无 `mg`、无 `rs`），却拿回 scalar 增益的约 80%。**
这落在预注册表的**结局 B**：「全局竞争态叙事推倒」。

**必须带的限定**：**只有 1 个种子跑满。** 项目规矩是 n≥3 才下定论，且
`sz_s1` 现在 30/100 读 +5.33★~ —— 部分样本上的 ★ 一律不信（Math.Find 在 38/100
读 −3.95★、跑满变 −2.33 不显著）。所以这是**第一读数，不是结论**。

### 如果三个种子都站住，什么变、什么不变

**不变**：保序重标定 ≡ 逐头配额分配的**定理完全不受影响**。`sz` 的特征是
`z` + 头嵌入，头内固定 chunk 时仍是 `s⁰` 的一元函数 ⇒ 仍然保序 ⇒ 仍然是**纯配额器**。
`|Δs| ≤ ασ_h`、有界决策等价、边界密度 Jacobian，全部照旧。

**变**：「知道全局那一刀切在哪是关键信息」这个**子命题要撤**。`sz` vs `affine`
在**信息完全相同**（都只有 z 与头身份）的前提下差 +3.80，隔开它们的是
**z 上的函数类丰富度**，不是信息量。

**精炼后的叙事（尚待验证）**：上下文依赖并没有消失，只是它的**进入方式**变了 ——
不是靠把 `(τ, σ_g)` 喂给网络，而是靠一个**固定的丰富形变**与该 chunk **实测 z 分布**
的交互，诱导出随 chunk 变化的有效平移。这与线性可行性判定是自洽的：那里证明的是
**没有固定平移**能复现轨迹，而固定形变本来就不是固定平移。

### 顺带修掉报表两处

- **未跑满的格子曾进跨种子聚合。** `szr` 因此打出 `+12.26 ± 7.74` —— 把 n=1 的
  `+20.00★` 和 n=84 的 `+4.52★` 平均了。现在 <90% 基线样本的格子标 `~` 且不进聚合。
- **`ARMS` 是手抄的第二份列表**，新因子臂整片不显示。改成从 `CalibScorer.MODES`
  派生 + 断言全覆盖，参数量从 ckpt 现读。这与 `scratch_ctrl_train.py` 的 `--arch`
  是同一类 bug，**同一个教训犯了第二次**：任何"选项列表"都要从源头派生。

### 16:20 补 `szr_s0` = +4.40★（n=100）—— 分界线在函数类，不在信息量

种子 0 上所有跑满的格子（同一批 `__g8base` 基线，n=100）：

| 臂 | 参数 | 输入 | s0 |
|---|---|---|---|
| `bias` | 225 | 逐头常数 | +0.20 |
| `affine` | 225 | `tanh(a·z+b)`，逐头 2 参数 | **+0.00** |
| `sz` | 4,226 | z + 头嵌入 | **+3.80★** |
| `szr` | 4,354 | z + `rs`（有 σ_g，无 τ） | **+4.40★** |
| `scalar` | 4,482 | z + `mg` + `rs` + 头嵌入 | +4.20★ |
| `kv` | 53,378 | K/V，看不到 s⁰ | +4.40★ |
| `v2` | 637,828 | 全部 + 记忆架构 | +4.40★ |

**从 4,226 到 637,828 参数，凡是能在头内表达任意单调形变的，全部落在 +3.8～4.4。**
`scalar` 自己三个种子的散布是 ±0.41，所以这一列里的差基本都在单种子噪声内。
掉到零的只有两个**受限函数类**（`bias` 逐头常数、`affine` 单个 `tanh(az+b)`，
且实测 `a ∈ [−0.092, +0.091]` 近似退化成纯归一化平移）。

⇒ **当前读数指向：分界线不是"知道多少"，是"能表达多丰富的单调形变"。**
`rs` 边际约 +0.6、`mg` 再加约 0（4.40 → 4.20，反向且在噪声内），
KV 内容 0，记忆架构 0，参数量 150× 也是 0。

预注册表的**结局 C**（`szr≈scalar` 而 `szm≪scalar` ⇒ 关键是跨头尺度校准）
**看起来不成立**：`szm_s0` 在 72/100 上读 +5.56★~，方向相反。当前最像**结局 B**。

**仍然是 1 个种子。** `sz_s1` 47/100 读 +4.26★~、`szr_s1` 19/100 读 +3.16~，
按规矩都不作数。三个种子齐了才下定论。

### 17:00 干净版 v2 补到 2 种子：ratio 0.2 很稳，**ratio 0.1 要放松措辞**

| ratio | v2c s0 | v2c s1 | 干净版跨种子 | 原版 v2 |
|---|---|---|---|---|
| 0.2 | +19.00★ | +20.20★ | **+19.60 ± 0.60**（2 种子） | +18.80★（1 种子） |
| 0.1 | +4.20★ | **+2.40 不显著** [−0.20,+5.00] | +3.30 ± 0.90（2 种子） | +4.27 ± 0.19（3 种子） |

**今早那句"两者逐项重合"是基于 s0 单种子写的，现在要放松**：ratio 0.2 上依然极稳，
但 **ratio 0.1 上第二个种子掉到 +2.40 且 CI 含零**。两条 CI 重叠很多
（[−0.20,+5.00] vs [+1.60,+7.40]），所以**不能说干净版更差**，只能说
**0.1 上的一致性比一个种子看起来的要松**。

信噪比可以解释一部分：ratio 0.1 的效应 ~+4 而配对 bootstrap 半宽 ~2.8，
ratio 0.2 的效应 ~+19 而半宽 ~5 —— 0.1 上种子级噪声占比大得多。这也是原版 v2
在 0.1 上要三个种子才给出 ±0.19 的原因。

**等 `v2c_s2`（在队列里）齐了才能下三种子结论。** 在那之前不要引用
"干净版 = 原版" 这句话的强形式。

### 17:10 `szm_s0` 落地 ⇒ 预注册判决：**结局 B**（种子 0，全部 n=100）

| 臂 | 参数 | 输入 | s0 |
|---|---|---|---|
| `bias` | 225 | 逐头常数 | +0.20 |
| `affine` | 225 | `tanh(a·z+b)` | **+0.00** |
| `sz` | 4,226 | z + 头嵌入，**无全局态** | **+3.80★** |
| `szr` | 4,354 | z + `rs` | **+4.40★** |
| `szm` | 4,354 | z + `mg`（τ、σ_g 都有） | **+4.80★** |
| `scalar` | 4,482 | 三个通道齐全 | +4.20★ |
| `kv` | 53,378 | K/V | +4.40★ |
| `v2` | 637,828 | 全部 + 记忆架构 | +4.40★ |

**整条信息阶梯是平的**，宽度 1.0，而 `scalar` 自己三种子散布就是 ±0.41（跨度 1.0）。
四个因子臂是**四次独立训练、输入各不相同**，全部落进同一条带 —— 若输入重要，
本应看到排序。没有。

**结局 A/C/D 全部排除**（A 需 `sz≈0`、C 需 `szm≪scalar`、D 需只有 `scalar` 有效）。

**撤回子命题**：「知道全局那一刀切在哪是关键信息」。此前 CLAUDE.md 写过
"`scalar` 比 `affine` 多的输入只有 margin 与 rs ⇒ 它知道全局阈值位置" —— 输入差是
事实，但把增益归给这个**信息差**是错的，真正隔开它们的是**函数类**：`sz` 与
`affine` 信息完全相同（都只有 z 与头身份），差 +3.80。

**不受影响**：等价定理。`sz`/`szr`/`szm` 的特征都是 `s⁰` 与头-chunk 常量的函数 ⇒
头内一元 ⇒ 保序 ⇒ 纯配额器。

**命题雏形**：增益 = 逐头配额重分配；唯一必要条件是能表达足够丰富的单调形变来选出
那个配额。全局竞争态 0、KV 内容 0、记忆架构 0、150× 参数量 0。`sz` 以 **4,226**
参数追平 `v2` 的 **637,828**（**151×**）。

**限定：全部单种子。** `sz_s1` 75/100、`szr_s1` 48/100、`szm_s1` 31/100 一律不作数。

### 17:40 `szmr0_s0` = +4.40★ 与 `sz` 第二种子 —— 极简性结论，以及结局 B 复现

| 臂 | 参数 | 逐头学习参数 | s0 | s1 | 跨种子 |
|---|---|---|---|---|---|
| `sz` | 4,226 | 头嵌入 | +3.80★ | +4.20★ | **+4.00 ± 0.20** |
| `szmr0` | **2,434** | **无** | **+4.40★** | — | +4.40（1 种子） |
| `scalar` | 4,482 | 头嵌入 | +4.20★ | +5.20★ | +4.73 ± 0.41 |
| `v2` | 637,828 | 头嵌入 + 记忆 | +4.40★ | +4.40★ | +4.27 ± 0.19 |

**两条：**

1. **结局 B 复现。** `sz` 两个种子 **+4.00 ± 0.20** —— 散布比 `scalar` 自己的 ±0.41
   还紧，且与 `v2` 的 +4.27 ± 0.19 基本重叠（**151× 参数差**）。"整条信息阶梯平坦"
   不是单次训练的运气。
   （写这条时 `sz_s1` 还在 97/100 读 +4.12，跑满定格 +4.20 —— 漂移 0.08。
   这顺带验证了报表那个"≥90% 才进聚合"的阈值定得合理。）
2. **`szmr0` 是目前最强的极简性结论。** 2,434 参数、**一个在全部 112 个头之间共享的
   3 输入 MLP、没有任何逐头学习参数**，与 637,828 参数的 `v2` 打成 +4.40 对 +4.40。
   **262×**。

**它并非"不区分头"** —— 头级适配全部是**结构性**的，不是学出来的：
`z = (s⁰−μ_h)/σ_h` 逐头标准化、输出 `Δs = α·σ_h·tanh(·)` 逐头缩放。再用恒等式

    mg = e^{rs}·z + q_h ,      q_h = (μ_h − τ)/σ_g

`(z, mg, rs)` 与 `(z, q_h, rs)` 等价 ⇒ 这个共享 MLP 实际看到的是**一个逐 token 变量
`z` + 两个头-chunk 运行时统计量 `(q_h, rs_h)`**。所以它是**条件于运行时统计量**的
一族形变，而不是条件于**学出来的头身份**。

**这正好是外部建议里 direct allocator 的输入集**（它提议
`δ = α σ tanh g(q, rs, e, ρ)`），而 `szmr0` 是它去掉 `e` 的版本。⇒ **该输入集充分性
已经有实证**。剩下的一步是把逐 token 求值改成逐头求值 —— 由等价定理，决策内容不损失，
但需要直接预测配额而非形变，而配额还依赖分数分布（"存在 ≠ 可学"那条警告仍然适用）。

`szmr0` 仍然保序：`z, mg, rs` 全是 `s⁰` 与头-chunk 常量的函数 ⇒ 头内一元 ⇒ 纯配额器。

**限定**：`szmr0` 仅 1 种子；`szr_s1` 70/100、`szm_s1` 54/100、`sz_s2` 23/100 不作数。

### 18:20 `szr_s1` = +2.60★ ⇒ 两种子 +3.50 ± 0.90；网格补齐 v2c 全 11 panel

跑满 ≥2 种子的全貌（Retr.KV @0.1，同一批 `__g8base` 基线）：

| 臂 | 参数 | 种子数 | 结果 |
|---|---|---|---|
| `affine` | 225 | 3 | **+0.00 ± 0.16** |
| `bias` | 225 | 3 | **+0.33 ± 0.50** |
| `sz` | 4,226 | 2 | **+4.00 ± 0.20** |
| `szr` | 4,354 | 2 | **+3.50 ± 0.90** |
| `scalar` | 4,482 | 3 | +4.73 ± 0.41 |
| `kv` | 53,378 | 3 | +4.47 ± 0.09 |
| `v2` | 637,828 | 3 | +4.27 ± 0.19 |

`szr` 的散布 ±0.90 明显比别的臂宽（+4.40 vs +2.60），但仍稳稳落在富函数类那一带，
远离 225 参数臂的零。**结论不变：分界线是函数类，不是信息量。**

### 一个小工具上连踩三层静默失败，值得记

`scratch_all_report.py --md` 一直**只打印不写文件**，靠人工把 stdout 贴进
`RESULTS_GRID.md`。今天在无人值守的循环里连着暴露三次：

1. 只打印不写 ⇒ 重生成后 `git diff` 是空的，**看起来像"没有变化"**；
2. `--md 2>&1 | tail -12` ⇒ 后台任务**只存下最后 14 行**，无任何提示；
3. `--md --write ... 2>&1 | tail -3` ⇒ 脚本抛 `FileNotFoundError`，
   但**管道的退出码是 `tail` 的 0**，任务报告 `completed`。

2 和 3 同根：**管道的退出码取自最后一段**。已修：脚本加 `--write`（自己写回，
相对路径按**脚本所在目录**解析 —— 它 import 时会 `chdir` 到 `prefill/`），
调用改成 `nohup ... > /tmp/grid.log 2>&1 &`，**不接管道**。

### 18:50 `szm_s1` = +4.00★ ⇒ 两种子 +4.40 ± 0.40。多种子全貌到齐

| 臂 | 参数 | 种子 | 结果 |
|---|---|---|---|
| `affine` | 225 | 3 | **+0.00 ± 0.16** |
| `bias` | 225 | 3 | **+0.33 ± 0.50** |
| `sz` | 4,226 | 2 | +4.00 ± 0.20 |
| `szr` | 4,354 | 2 | +3.50 ± 0.90 |
| `szm` | 4,354 | 2 | +4.40 ± 0.40 |
| `scalar` | 4,482 | 3 | +4.73 ± 0.41 |
| `kv` | 53,378 | 3 | +4.47 ± 0.09 |
| `v2` | 637,828 | 3 | +4.27 ± 0.19 |

**13 次独立训练、6 种不同输入集、参数量跨 151×，全部落在 +3.50～+4.73。**
掉到零的只有两个 225 参数的受限函数类。这已经不是"一个种子的巧合"能解释的了 ——
结局 B 稳。

未跑满但方向一致：`sz_s2` 71/100 +3.66★~、`szmr0_s1` 74/100 +4.32★~、
`szm_s2` 45/100 +4.44★~、`szr_s2` 50/100 +2.80~。

### 19:20 `szmr0` 复现 +4.20 ± 0.20；但"信息阶梯完全平坦"要**部分收回**

`szmr0_s1` = +4.00★ ⇒ 两种子 **+4.20 ± 0.20**。**2,434 参数、零逐头学习权重**，
对 `v2` 的 637,828 参数 +4.27 ± 0.19 —— **262× 差，均值与散布都对得上**。
这是今天最强的单条结果，极简性命题站住了。

**但同一张表里 `sz_s2` = +2.47（97/100，无 ★）**，把 `sz` 从 +4.00 ± 0.20 拉到
**+3.49 ± 0.74**。补了种子之后，一个我在种子 0 上看不见的分组浮出来了：

| 是否有 `mg`（携带全局阈值 τ） | 臂 | 结果 |
|---|---|---|
| **无** | `sz`（z） | **+3.60 ± 0.59**（3 种子，已定格） |
| **无** | `szr`（z+rs，有 σ_g 无 τ） | **+3.50 ± 0.90**（2 种子） |
| **有** | `szmr0`（z+mg+rs，无头嵌入） | **+4.20 ± 0.20**（2 种子） |
| **有** | `szm`（z+mg） | **+4.40 ± 0.40**（2 种子） |
| **有** | `scalar`（z+mg+rs+头嵌入） | **+4.73 ± 0.41**（3 种子） |

两个无 `mg` 的臂落在 **+3.49 / +3.50**，三个有 `mg` 的落在 **+4.20 / +4.40 / +4.73**。
**有 τ 的均值高约 +0.9，且种子散布约减半。**

**要收回的措辞**：我在种子 0 上写的"整条信息阶梯**是平的**"**过头了**。
那时 `sz_s0` = +3.80 与 `scalar_s0` = +4.20 只差 0.40，看起来在噪声内；补种子后
`sz` 的真实散布是 ±0.74（第三个种子掉到 +2.47），而 `mg` 组的散布只有 0.20–0.41。

**但也不能反过来说"τ 是关键信息"**：+0.9 的差、合并 SE 约 0.52 ⇒ **约 1.7σ**，
**不显著**。而且 `szr`（有 σ_g 无 τ）与 `sz`（都没有）完全并列，说明起作用的若有，
是 **τ 而非 σ_g**。三个种子补齐前不下结论。

**没有动摇的**：**主分界线仍然是函数类**。225 参数的受限族拿 0，任意富 MLP 拿
+3.5～+4.7 —— 这是**约 4 分的断崖**，比上面这个 +0.9 的二阶效应大四倍多，且
毫无争议。等价定理也不受影响（所有这些臂头内都是 `s⁰` 的一元函数 ⇒ 纯配额器）。

修正后的命题：*增益 = 逐头配额重分配；一阶必要条件是表达力足以选出配额（+4 分断崖），
可能还有一个约 +0.9 的二阶项来自知道全局阈值位置（1.7σ，未定）。*

### 19:38 `sz` 定格 +3.60 ± 0.59，以及一个直接影响当前判断的漂移量

`sz_s2` 跑满 = **+2.80★**，而它在 97/100 时读 **+2.47** —— **漂移 0.33**。
对比之前 `sz_s1` 在 97/100 上只漂 0.08，说明这个漂移**没有稳定上界**。

**这直接约束现在能说什么**：正在权衡的 `mg` 效应只有 **+0.9**，而剩下三个未跑满的
格子单个就可能移动 0.3 以上 —— `szr_s2` 80/100 读 +4.50★~、`szm_s2` 75/100 读
+5.07★~、**`szmr0_s2` 61/100 读 +2.95★~**。最后那个若坐实，`szmr0` 会从
+4.20 ± 0.20 掉到 ±0.6 量级，`mg` 那个分组就散了。

顺带把 `sz` vs `scalar` 的**逐种子配对**算出来（同种子号视为配对）：
逐种子差 −0.40 / −1.00 / −2.00，均值 **−1.13 ± 0.66**，SE 0.38，
**t = 2.97，df = 2 ⇒ p ≈ 0.10**。方向一致（3/3 为负）但 n=3 下不显著。
**三个第三种子落地前不下结论。**

---

## 2026-08-17 20:15 —— 因子臂 12/12 收尾：**只有函数类这一条是显著的**

全部三种子、全 n=100、同一批 `__g8base` 基线。

| 臂 | 参数 | 头嵌入 | `mg`(τ) | `rs`(σ_g) | 跨种子 |
|---|---|---|---|---|---|
| `affine` | 225 | — | — | — | **+0.00 ± 0.16** |
| `bias` | 225 | — | — | — | **+0.33 ± 0.50** |
| `sz` | 4,226 | ✓ | — | — | **+3.60 ± 0.59** |
| `szr` | 4,354 | ✓ | — | ✓ | **+3.67 ± 0.77** |
| `szmr0` | **2,434** | **—** | ✓ | ✓ | **+3.67 ± 0.77** |
| `szm` | 4,354 | ✓ | ✓ | — | **+4.60 ± 0.43** |
| `scalar` | 4,482 | ✓ | ✓ | ✓ | **+4.73 ± 0.41** |
| `kv` | 53,378 | ✓ | 看 K/V，看不到 s⁰ | | +4.47 ± 0.09 |
| `v2` | 637,828 | 全部 + 记忆架构 | | | +4.27 ± 0.19 |

**逐种子配对**（同种子号配对；t 用样本 sd，df=2）：

| 配对 | 逐种子 | 均值 | t | p | 判定 |
|---|---|---|---|---|---|
| **`sz` − `affine`** 函数类 | +3.80/+4.40/+2.60 | **+3.60** | **6.8** | **0.02** | **✓ 唯一显著** |
| `szm` − `sz` 加 `mg` | +1.00/−0.20/+2.20 | +1.00 | 1.44 | 0.29 | ✗ |
| `szr` − `sz` 加 `rs` | +0.60/−1.60/+1.20 | +0.07 | ~0 | — | ✗ |
| `scalar` − `szm` 再加 `rs` | −0.60/+1.20/−0.20 | +0.13 | ~0 | — | ✗ |
| `szmr0` − `scalar` 去头嵌入 | +0.20/−1.20/−2.20 | −1.07 | −1.53 | 0.27 | ✗ |

### 结论

**唯一站得住的因子是函数类。** 从 `affine`（逐头 `tanh(a·z+b)`，2 参数）换成
`sz`（同样只有 z 与头身份，但任意 MLP）拿 **+3.60，3/3 种子为正，p≈0.02**。
**没有任何一个输入通道单独站得住**：`mg` +1.00（2/3）、`rs` +0.07、头嵌入 +1.07（2/3），
全部 p > 0.2。

### 两条要修正的（都是我自己一两个种子上读出来又散掉的）

1. **~~`szmr0` +4.20 ± 0.20，262× 小而"均值与散布都对得上"~~** —— 第三种子 +2.60
   把它拉到 **+3.67 ± 0.77**。"与 `v2` 不可分"仍成立（区间重叠），但**"散布也对得上"
   那半句是错的**（0.77 vs 0.19）。
2. **~~有 `mg` 的三臂 vs 无 `mg` 的两臂差 +0.9~~** —— **不成立**。`szmr0` 有 `mg`
   却落在 +3.67，与无 `mg` 的 `szr` 逐位并列。那个分组是 2 种子上的巧合。

**今天两次在 1–2 个种子上读出模式、到 3 个种子散掉。** 两次都标了"待定"，是对的 ——
但也说明**在这个噪声水平上，n=2 不足以看出任何 ~1 分的效应**。以后 1 分量级的命题
直接按 n≥5 规划，不要先写再收。

### 没有动摇的

- **约 4 分的断崖**：225 参数受限族 0 分 vs 任意富 MLP +3.6～+4.7。
- **等价定理**：所有这些臂头内都是 `s⁰` 的一元函数 ⇒ 保序 ⇒ **纯配额器**。
- **极简性**：`sz` 4,226 参数（+3.60 ± 0.59）与 `szmr0` **2,434 参数、零逐头学习
  权重**（+3.67 ± 0.77）都与 `v2` 的 637,828 参数（+4.27 ± 0.19）不可分。
  **151× / 262×**，只是误差棒比两种子时看起来的宽。

**命题定稿**：*增益 = 逐头配额重分配；唯一测得出的必要条件是"能在头内表达任意单调
形变"。喂给它什么信息（全局阈值、头身份、KV 内容）、多少参数（跨 262×）、什么架构
（记忆 vs 纯 MLP），在 n=3 下都测不出差别。*

### 21:55 真实配额导出跑起来：理论的三条算术被实测确认，并拿到真实量级

`VARIKV_QUOTA_DUMP` 钩子（commit 9bc6d5a）第一条样本即产出 11 条记录（11 chunk）。
结构核验全过：

| 检查 | 结果 |
|---|---|
| 头数 | **112** = 28 层 × 4 kv 头 ✓ |
| 每 chunk 总预算 arm vs base | **逐 chunk 完全相同** ✓ |
| **`Δb` 每 chunk 之和** | **恒为 0**（11/11） ✓ |
| 自由度 | 11 × 112 = **1,232** 格；`Σ_h Δb = 0` ⇒ 独立自由度 11 × 111 = **1,221** ✓ |
| 有效 ratio | **0.0776**，不是 0.1 —— `wrapper.py:271-277` 的窗口重标定，与 CLAUDE.md 记的 0.078 吻合 |

前三条把 §四之五 里靠推导得到的量纲（1,232 个数 / 1,221 个独立自由度、`J·1 = 0`）
变成了**实测**。

**真实量级（trace 拿不到的）**：`scalar` 平均搬动 **75.89 个 KV 槽位/(chunk,头)**，
最大 **1,664**，527/1232 个头非零。每头平均配额约 `139147/112 ≈ 1242` ⇒
**约 6% 的配额在被重分配**，个别头几乎被整体改写。

⚠️ **先前 trace 上估的 `|Δb| = 20.50` 系统性偏小** —— 那里每 (chunk,层,头) 只存
768 个候选，而真实每头有约 1,242 个保留位。**凡是引用 trace 版 `|Δb|` 的地方都要
标明它是子总体量**；`affine` 20.74 vs `scalar` 20.50 那个"等量搬动"的比较仍然有效
（两臂同口径），但绝对值不能当真实值用。

---

## 2026-08-17 22:30 —— 真实配额上的留出分解：`scalar` 的重分配 **97% 是一个固定的逐头模式**

配额导出跑完（220 条 = 20 样本 × 11 chunk）。用**留出验证**（前 10 样本拟合、
后 10 样本评估，避开先前 `R²_static` 的样本内膨胀）：

| 模型 | 留出 R² |
|---|---|
| **逐头一个常数 `Δb_h`** | **+0.9697** |
| 逐 (头, chunk 位置) 一个常数 | **+0.9976** |

**不是少数大头灌水**：限制到 |Δb|>100 的 23 个头，R² 仍是 **+0.9703**。
集中度：前 5 头占总搬动量 47.2%、前 20 头占 88.4%，**只有 56/112 个头会动**。
残差 RMS **36.2 槽位**，约为每头配额（≈1,242）的 **3%**。

### 这要求修正今天下午那条结论的**措辞与重点**

我写过（LP 那一轮）：

> ~~不存在任何固定策略能复现 `scalar` 的配额轨迹 —— 这是结构性的 ⇒ 上下文依赖是必要的~~

**字面上仍然成立**（107/4928 条约束违反），但**重点完全带偏了**。两个测量并不矛盾，
因为它们说的是**两个不同的不变量**：

| | 固定的量 | 结果 |
|---|---|---|
| LP | 固定**归一化平移** `η_h` | 无法**精确**复现（2.2% 约束违反） |
| 留出分解 | 固定**槽位增量** `Δb_h` | 解释 **97%** 方差 |

两者由边界密度联系：`db_h = d_h·(dδ_h − …)`，`d_h = N_h f_h(τ)` 随上下文变。
**要得到恒定的 `Δb`，就必须用变动的 `η`；固定的 `η` 反而给出变动的 `Δb`。**

⇒ **不变的对象是槽位增量，不是平移量。** 这条同时解释了为什么 `bias` 只有 +0.33：
它固定的是 `η`，于是 `Δb` 随分数几何漂移；而 `scalar` 实际产出的是近乎恒定的 `Δb`。

### 直接的方法蕴含（下一个该做的实验）

若 97% 的方差由 **112 个整数**捕获，那么方法可能坍缩成：**没有网络、没有输入、
没有上下文——就是一张逐头配额增量表。** 最便宜的检验是拿这 20 个样本测出的逐头
`Δb_h` 均值直接回放（零训练），看能否复现 +4.73。需要在 harness 里加"按给定配额
选取"的注入路径。

**未验证的限定**：单数据集（Retr.KV）、单臂（`scalar` s0）、20 样本。这个固定模式
很可能是**数据集相关**的 —— 若换 panel 就变，"112 个整数"就不是方法而是过拟合。
跨 panel 稳定性必须先测。

### 23:20 跨 panel 证伪：**组内近乎静态，跨 panel 不迁移** ⇒「112 个整数」不是方法

三个 panel 各导一次配额（`scalar` s0，--num 10），与 Retr.KV 同法分析：

| panel | 样本 | chunk/样本 | \|Δb\| 均值 | 动的头 | **组内**留出 R²（逐头常数） |
|---|---|---|---|---|---|
| Retr.KV | 20 | 11 | 76.23 | 51 | **0.9697** |
| Retr.MultiHop | 10 | 8 | **600.93** | **111** | **0.9557** |
| Math.Find | 10 | 9–10 | 140.16 | 49 | **0.8672** |
| ICL.ManyShot | 10 | 2 | **0.00** | 0 | —（退化，见下） |

跨 panel 的逐头 `Δb_h` 向量相关：

| Pearson | Retr.KV | MultiHop | Math.Find |
|---|---|---|---|
| **Retr.KV** | 1.000 | **−0.204** | +0.271 |
| **MultiHop** | −0.204 | 1.000 | +0.179 |
| **Math.Find** | +0.271 | +0.179 | 1.000 |

（Spearman：0.211 / 0.577 / 0.264，同样低。）

**结论：组内 0.87–0.97 近乎静态，跨 panel 却不迁移，Retr.KV 与 MultiHop 甚至负相关。**
⇒ **撤回昨晚那条方法蕴含**：不存在一张能通吃的 112 整数表。commit a3bf0a3 里
预先标注的过拟合风险**兑现了**，标注得对。

### 但这比「112 个整数」更有意思，不是坏消息

三条合起来给出一个更完整的图像：

1. 等价定理 ⇒ 打分器**根本不做逐 token 的活**，它的决策内容就是配额；
2. **组内**它做的是一个近乎静态的重分配（97% / 96% / 87%）；
3. **跨 workload 正确的重分配不一样**，甚至反号。

⇒ **打分器的价值在于"从上下文推断出这个 workload 该用哪套配额"** —— 那是一个真函数，
不是查表。这也给了 `bias` 失败一个新的候选解释：它固定的那套配额是在 fineweb 上训的，
**跨 workload 通用的固定配额本来就不存在**。

**留下的一个干净的组内命题（值得测）**：在 Retr.KV 上直接拟合那 112 个 `Δb_h` 并回放。
若能复现 +4.7，则"组内静态表足够、网络的全部价值在于推断 workload"就成立。

### `ICL.ManyShot` 的 `Δb ≡ 0` 是退化陷阱，不是发现

26,474 token < `window/ratio = 4096/0.1 = 40,960` ⇒ `wrapper.py:271-277` 把
`chunk_ratio` 置零，保留集恰为局部窗口，**任何分数扰动按构造都是 no-op**
（CLAUDE.md 早记过）。所以 `Δb ≡ 0`、0 个头会动是**预期行为**。
这反过来印证 `RESULTS_GRID.md` 上 ManyShot 那些 `°` 退化标记是对的。

### 一条顺带的观察

`Retr.MultiHop` 的搬动量是 Retr.KV 的 **7.9 倍**（600.93 vs 76.23），且 **111/112
个头都在动**——而它恰恰是**压缩胜过满缓存、我们的方法伤害最大**的那个 panel
（v2 在那里 −4.60★）。打分器在那里非常用力地做着一件错事。

---

## 2026-08-18 01:10 —— **组内，整个学出来的方法坍缩成 112 个整数**

配额注入跑满 n=100。丢掉网络、丢掉逐 token 修正、头内退回 FastKVzip 原序，
只按 `b_base + Δb_h`（112 个数，**无输入、无上下文**）取 top-b：

| 臂 | 参数 | 全部 100 | 前 20（样本内） | 后 80（留出） |
|---|---|---|---|---|
| **静态配额表** | **112 个数** | **+4.20★** | +0.00 | **+5.25★** |
| `scalar` s0 | 4,482 | +4.20★ | +2.00 | +4.75★ |
| `v2` s0 | 637,828 | +4.40★ | +1.00 | +5.25★ |

**静态表 − `scalar` 逐样本配对 = +0.00 [−2.00, +2.00]，n=100。完全不可分。**

**不是过拟合**：静态表在它自己的拟合样本（前 20）上反而最低 +0.00，在 80 条留出上
+5.25★ 与 `v2` 持平。前 20 条对三条臂都低（+0.00/+2.00/+1.00），是样本难度效应。

预算断言全程未触发 ⇒ 每个 chunk 的总保留量与基线严格相等，比的是同一个压缩率。

### 与前面几条合起来的完整图像

1. **保序重标定 ≡ 逐头配额分配**（定理，已在 trace 上逐位验证，`kv` 阴性对照 0/22）
   ⇒ 打分器的决策内容**就是**配额，不做逐 token 的活；
2. **组内**正确的配额是一张**固定**的 112 数表 —— 4,482 参数的网络相对它买到
   **恰好零**（+0.00 [−2.00,+2.00]），637,828 参数也一样；
3. **跨 workload** 正确的表不同，甚至反号（Retr.KV vs Retr.MultiHop 相关 **−0.204**）。

> ⇒ **网络的全部功能是"推断这个 workload 该用哪张表"** —— 而它做这件事的水平，
> 与直接在该 workload 上拟合一张表**不可分**。

这对"学习式打分器"的叙事是负面结果，对分析是正面结果：它精确刻画了这 +4.7 是什么。
也顺带解释了本项目一路的负结果 —— 记忆架构、KV 内容、全局竞争态、150× 参数量，
优化的全是不重要的维度。

### 限定

- **一个种子的表**（`scalar` s0 导出）。应当用 s1/s2 的表各重跑一次。
- **一个 panel**。跨 panel 那张表不迁移，所以这**不是**一个可部署的方法 ——
  拿到某 workload 的表本身就需要先跑打分器。
- `Δb` 表在同 panel 前 20 个样本上测得；留出 80 条已排除过拟合。

### 01:40 配额表是 **panel 的属性，不是训练的产物**；以及 MultiHop 上的等预算被破坏

**① 三个种子的表几乎相同。** `scalar` s0/s1/s2 各导一次配额（Retr.KV，--num 20）：

| | 和 | \|Δb\| 均值 | 动的头 |
|---|---|---|---|
| s0 | −0.00 | 76.23 | 51/112 |
| s1 | −0.00 | 77.66 | 51/112 |
| s2 | +0.00 | 75.97 | 51/112 |

两两相关 **Pearson 0.998–0.999、Spearman 0.997–1.000**。三次独立训练收敛到同一张
112 数表。与跨 panel 的 −0.204~+0.271 对照：

> **表由 panel 决定，与训练种子无关。** 训练做的事就是可靠地把它找出来。

这也意味着 s1/s2 的注入评测**几乎学不到新东西**（表相关 0.999），不值得各烧 1.5 GPU-h。
改为排入**跨 panel 移植**：把 Retr.KV 的表注到 MultiHop 上，与在跑的
`qinj_vt`（MultiHop 自己的表）构成 2×2 —— 两者等预算，唯一差别是表的来源，
**直接因果检验「表是 workload 专属的」**。

**② 一个必须修正的说法：等预算只在 Retr.KV 上验过。** 逐 chunk 的 `Σ_h Δb`：

| panel | 非零 chunk | 最大偏差 |
|---|---|---|
| Retr.KV | 1/220 | −1（阈值处平局） |
| Math.Find | 1/99 | −1 |
| **MultiHop** | **37/80** | **−2,481**（约占该 chunk 保留量的 2.0%） |

**在 MultiHop 上 `scalar` 实际比基线少留约 2% 的缓存。** 先前写的"每 chunk 总预算
arm=base ✓"是只在 Retr.KV 上验的，**不能外推**。

两条后果：(a) MultiHop 上 `v2` 的 **−4.60★ 有一个真实混淆** —— 伤害里有多少只是用了
更少的缓存？(b) 在跑的 `qinj_vt` 因为有迭代配平，**是严格等预算的**，所以它测的
不是"复现 scalar 在 MultiHop 上的行为"，而是"在等预算下应用那张表"。
描述时必须写准，别把两者混为一谈。

---

## 2026-08-18 02:40 —— MultiHop 上的 2×2：等预算混淆解决；「用哪张表」在这里分不出来

`qinj_vt`（MultiHop 自己的表）与 `qxfer`（Retr.KV 的表移植）都跑满 n=90，
两者**严格等预算**（迭代配平 + 断言），唯一差别是表的来源。

| 臂 | Δ vs `__g8base`（基线绝对分 49.47） |
|---|---|
| ① MultiHop 自己的表 | **−8.80★** [−11.16, −6.31] |
| ② Retr.KV 的表移植 | **−7.07★** [−9.16, −4.98] |
| ③ `v2` 全网络（**非**等预算） | −9.96★ [−12.22, −7.60] |

| 配对 | 值 |
|---|---|
| ① − ② 对的表 vs 错的表 | **−1.73 [−3.96, +0.53]** 不显著 |
| ① − ③ 静态表 vs 全网络 | **+1.16 [−1.29, +3.51]** 不显著 |

### 三条结论

1. **等预算混淆解决，而且很小。** 先前担心 MultiHop 上的伤害有多少只是 `scalar`
   少留了约 2% 缓存（37/80 个 chunk，最大 −2,481）。**严格等预算的注入仍拿 −8.80★**，
   与非等预算的 `v2` 的 −9.96★ 配对差 +1.16 **不显著** ⇒ 缓存缺口至多解释约 1.2 分，
   **主体是重分配本身**。
2. **「网络 ≡ 一张配额表」在第二个 panel 上也成立**，而且是在**负方向**上：
   ① − ③ = +1.16 不可分。等价性与方法是帮还是伤无关。
3. **但这个 2×2 分不清「表无所谓」和「两张表都是坏的」。** 两张表都伤 7–9 分，
   而该 panel headroom 为负（满缓存 41.07 < 压缩后 46.09）—— 基线本就在最优之外的
   一侧，**任何方向的重分配都伤**。所以 ①≈② **不能**读成"表不是 workload 专属的"。

### 因此补了反向移植（在跑）

真正能因果证明 workload 专属性的，是把 **MultiHop 的表注到 Retr.KV**（那里方法有效）：

    Retr.KV + Retr.KV 的表   = +4.20★   （已测）
    Retr.KV + MultiHop 的表  = ?        ← `qrev_vttable_on_kv.log`，tag `_qrev`

若掉到 0 附近或转负，「每个 workload 有自己的最优静态分配」就有了因果证据；
若仍是 +4，则说明 Retr.KV 上的增益对表的具体内容不敏感，那前面"表由 panel 决定"
（三种子 0.999 / 跨 panel −0.204）就只是相关性，方法叙事要再收一次。

注入用的 vt 表已**居中**（原和 −418.8 → −0.00），使它是纯重分配，
避免迭代配平吸收掉一个整体平移而混入无关效应。

---

## 2026-08-18 04:20 —— 四处过度声明的修正，以及一个更深的解释框架

外部复核指出四处措辞过头。**逐条核实，四条都成立**，全部改正。

### ① 「不可分」≠「追平」—— 必须做 TOST（δ=±1 分，配对，df=2）

| 配对 | 差 | 下界 p | 上界 p | 判定 |
|---|---|---|---|---|
| `szmr0` − `v2` | −0.60 | 0.219 ✗ | 0.031 ✓ | **不能判等价** |
| `scalar` − `v2` | +0.47 | 0.024 ✓ | 0.125 ✗ | **不能判等价** |
| `sz` − `v2` | −0.67 | 0.185 ✗ | 0.015 ✓ | **不能判等价** |
| `kv` − `scalar` | −0.27 | 0.046 ✓ | 0.017 ✓ | **等价** |

⇒ **"2,434 参数追平 637,828"、"151× 小而不可分"这类说法要撤成"未检测到差异"。**
只有 `kv ≈ scalar` 真正通过了等价检验。静态配额表 vs `scalar` 的 +0.00 **[−2.00,+2.00]**
同理——区间宽到 ±2 分，不足以在 δ=1 下判等价。

### ② n=3 的设计**根本达不到**强显著

3 vs 3 的完全置换只有 `C(6,3)=20` 种 ⇒ **单尾最小 p = 0.050**。所以：

- `sz − affine` 的 `p=0.02`（配对 t）在 5 个对比下做 Bonferroni（α=0.01）**过不了**；
- 更根本的是，**任何 n=3 的臂间比较都不可能给出 p<0.05 的分布无关证据**。

⇒ 措辞改为 **"最大且唯一符号一致的对比"**，不是 "唯一显著"。它仍然很强
（+3.60，3/3 为正，两臂的三个种子取值区间完全不重叠：affine ⊂ [−0.20,+0.20]、
sz ⊂ [+2.80,+4.20]），但**是描述性的强，不是检验性的强**。

### ③ 「112 个整数、无输入、无上下文」**描述错了整条流水线**

实际是 `FastKVzip(当前上下文) → b⁰_{c,h}`，再 `+ Δb_h`，且头内选择用当前上下文的
`s⁰`。**只有那个修正项是固定的。** 正确叫法是
**"FastKVzip + workload 专属的静态配额偏移"**。

### ④ 「网络的全部功能是推断 workload」是过度拟人化

`sz` 连 `τ`、`σ_g` 都看不到却拿 +3.60，它不可能在"识别 workload"。更深的解释是
**分布几何**：把 `s⁰ = μ_h + σ_h z` 代入，

    s'_{c,h,i} = μ_{c,h} + σ_{c,h}·W_h(z_{c,h,i}),   W_h(z) = z + α·tanh g_θ(z, e_h)

于是配额

    b_{c,h} = N_{c,h}·[ 1 − F_{c,h}( W_h^{-1}( (τ'_c − μ_{c,h}) / σ_{c,h} ) ) ]

**同一条 warp 作用在不同的 `F_{c,h}` 上，自然给出不同配额，不需要任何 workload 分类器。**
这一条同时解释了三个先前看似矛盾的结果：组内 `F` 相似 ⇒ 配额 97% 恒定；跨 panel `F` 不同
⇒ 表不迁移且反号；`sz` 不知道全局阈值也行 ⇒ 全局 Top-B 本身就把各头 warp 后的分布联立了。

### 实测这条 warp（`scratch_sz_warp.npy`，零 GPU）

`W_h(z)` 在 z∈[−4,8] 上 **处处单调**（min W' = +0.633，0/112）。形状：

| | W(0)−0 | W(6)−6 | 跨头相关 | 第1成分 |
|---|---|---|---|---|
| 训练后 s0/s1/s2 | −0.396/−0.449/−0.427 | **−0.970/−0.975/−0.994** | +0.933~+0.957 | 76.5~81.4% |
| **随机初始化 ×3** | +0.007/+0.086/+0.079 | −0.304/+0.380/+0.113 | **+1.000** | **100.0%** |

**随机初始化对照给出分层结论**（这是 P1-3 教的做法）：

- **单调性是架构性的**（随机初始化也 0/112）—— 不是学出来的，但加强了等价定理的适用性；
- **低维也是架构性的，而且随机更退化**（100% vs 76.5%）⇒ **训练是把各头曲线拉开，
  不是压低维**。不得声称"学到低维流形"；
- **形状是学出来的**：三个种子都收敛到"强压顶部尾巴（z=6 处 −0.97，几乎饱和到 −α）、
  中等压中位（−0.42）、底部基本不动"，随机初始化毫无此结构。

⇒ 机制是**对分数分布上尾的收缩**：顶部尾巴越重的头，在全局竞争中被折价越多。

### 一条比外部预期更强的：配额表的**子样本**稳定性

外部担心 0.998–0.999 只测了网络种子、没测样本子集。实测 200 次随机 10/10 split-half，
逐头 `Δb` 表的相关**最小 1.0000**。子集稳定性是满分。

### 已排入的三个对照（外部要求的，都成立）

1. **幅度匹配移植** `_qnm`：MultiHop 表 ×λ=0.1289 使 `|Δb|₁` 与 Retr.KV 表相等（8538）。
   原样移植的 7.9× 幅度差**确实**分不清"方向错"与"搬太猛"。
2. **头置换对照** `_qperm`：Retr.KV 自己的表，只打乱"搬给谁"（与原表相关 −0.027，
   `|Δb|₁` 完全相同）。这是最锋利的"方向重要吗"检验。
3. **全零表恒等检验** `_qzero`：基线用 `score > thres`（严格），注入用 argsort 取 exact
   top-b，**平局时语义可能不同**（Retr.KV 上已见 1/220 个 chunk 差 1 条）。用零表逐位比。

### 一个调度 bug：topup 的幂等检查有漏洞

只查"卡上有没有进程"不够 —— 同卡已有一个 topup worker 正处在**两个作业之间的轮询
间隙**时卡上确实是空的，于是第二个也起来，两个一起抢队列。实测 GPU0 叠了 2 个作业
（53 GB），而另外 4 张卡全空。已改为 `/tmp/varikv_topup/<gpu>` **每卡一把 mkdir 锁**
（记 PID，持有者死了回收，trap 退出时释放）。被叠的作业已杀、半截结果目录已删、重排。

---

## 2026-08-18 05:30 —— 三处再收紧，`_qzero` 给出一个我没预料到的重要对照

外部复核指出三处仍然过头。**逐条核实，三条都成立。**

### ① ~~「单调性是架构性的」~~ —— 撤回，改为**参数化/初始化先验**

`W'(z) = 1 + α·sech²(g)·∂g/∂z`，只要 `∂g/∂z < −1/(α·sech²(g))` 就能为负。
**当前架构数学上允许非单调。** 3 个随机初始化都单调，只说明 residual 形式
`z + α·tanh(·)` 与初始化尺度**强烈偏向**单调，不是结构保证。
论文若要把 rank-preserving 写成定理，必须换成**构造性单调**参数化
（如 `W(z) = c + ∫ softplus(r(t))dt` 或单调样条）。

### ② ~~「低维是架构性的」~~ —— 同样撤回

随机初始化 PC₁=100% 是因为**所有头共用一个 MLP 且头嵌入初始很小**，各头曲线本就
几乎重合。训练后能到 76.5% **恰恰证明函数空间没有被限制到 rank-1**。
正确写法：随机初始化因共享参数化呈现近乎 rank-1 的共享 warp；训练把各头曲线分开。

### ③ 「上尾收缩 ⇒ 重尾的头被折价更多」—— 前半有强证据，后半是**待验假说**

配额同时依赖 `F_{c,h}, μ, σ, τ'_c, W_h`，而 `τ'_c` 还是所有头联立的结果。
不能从 `W(6)−6 < 0` 直接推出重尾头被罚更多。**已降级为假说**，验证方法：
测每头尾重指标（如 `q.99−q.90`、`CVaR₀.₉₅`）与 `Δb_h`、与 `D_h = W_h(6)−6` 的相关。

### ④ 我自己的统计口径不自洽（外部指出，成立）

我说「3v3 置换 C(6,3)=20 ⇒ 最小 p=0.05」，但同时把分析描述成**按种子号配对**。
配对的分布无关检验是**符号翻转**，只有 `2³=8` 种 ⇒ **最小单尾 p = 0.125**。
所以 `sz−affine` 的 `p=0.02`（配对 t）**完全依赖 df=2 的正态性假设**，
分布无关证据最强只能到 0.125。措辞必须相应减弱。

### ⑤ split-half：Pearson=1 不等于逐项相等（外部指出，成立）——补齐了逐项统计

| | 值 |
|---|---|
| Pearson | 最小 1.0000 |
| **RMSE** | 均值 **1.43 槽位**（对比 `|Δb|` 均值 76.2、跨头 sd 205.0） |
| 最大单头差 | 均值 8.9，最大 23.4 槽位 |
| 符号一致率 | 均值 **98.4%**，最小 96.4% |

⇒ 补上逐项统计后结论**不变且更强**：表在样本子集间确实极稳，不只是排序稳。

### ⑥ 「幅度匹配要匹配**实际**搬动」（外部指出，成立，而且 `_qperm` 真被污染了）

离线复刻 clamp/round/迭代配平（`n = hi−lo` 逐头相同、`b_base` 已有 ⇒ 可精确重算）：

| 表 | **实际**每 chunk 搬动 | 被 clamp 的格 |
|---|---|---|
| Retr.KV 自己的 | **4269.0** | 415 |
| ~~朴素头置换~~ | **2769.7**（只有 65%） | **2418** |
| 幅度匹配 MH `_qnm` | 4191.8（差 1.8%，**有效**） | 769 |
| 原样 MH `_qrev` | 30385.2（**7.1×**，确认混淆） | 1637 |

朴素置换把大 `Δb` 发给预算容纳不下的头，被 clamp 掉三分之一 ⇒ 若它表现差，
分不清"方向错"还是"搬得少"。**已杀掉重设计为分层置换**（按 `b_base` 分 8 层、
层内打乱）：实际搬动 **4269.0，与自己的表逐位相同**，`|Δb|₁` 相同，与原表相关
−0.044 ⇒ **唯一差别就是"搬给谁"**。

### ⑦ `_qzero` 的结果：tie 污染**不存在**，但它变成了一个更有价值的对照

外部把零表恒等检验列为最高优先级，是对的。结果分两层：

- **配额逐格完全相同**：`b_arm` vs `b_base` 在 22 个 chunk × 112 头上 **0/2464 不同**
  ⇒ 注入路径与基线阈值选出**同一批 KV**，**tie 语义不构成污染**（这条排除了）。
- **但生成文本 50/50 全不同**（n=10）。既然保留集相同，差异只能来自**额外跑打分器
  引入的数值不确定性**（bf16 + 169k 上下文，多余 GPU 计算改变 kernel 选择 ⇒ 微小浮点差
  ⇒ 贪心解码某处翻转）。注意这不与 CLAUDE.md 的"评测确定性"矛盾：那条验的是
  **同一代码路径**跨 GPU 逐位一致，这里是**不同代码路径**。

⇒ **`_qzero` 因此是所有 ctrl 臂比较的噪声基底**：它与基线选同一批 KV，唯一差别就是
这条数值噪声。已补到 n=100。若它读 +0.00 且 CI 窄，`+4.20` 安全；若它本身就有 ±2 的
波动，**所有对 `__g8base` 的 ★ 都要重新审视**。这个对照我先前没想到。

### 调度事故：两套锁命名空间 + 一次险些误杀

- topup 用 `/tmp/varikv_topup`、池子用 `/tmp/varikv_gpulock`，**互不阻塞** ——
  正是 CLAUDE.md 记过的"多个调度器各扫各的会叠卡"的重演。
- 更糟的是**加锁修复前启动的 worker 一直没死**，累积到约 28 个，其中无锁的仍在轮询，
  于是把新作业抓到已有作业的卡上（GPU0 叠了 2 个）。
- 清理时我按"锁文件里的 PID ≠ 自己"判定无锁 worker，**误把持锁 worker 的子 shell
  也算了进去**（它们 PPID 是持锁者）。所幸杀掉的都是空转轮询器，5 个评测无一中断 ——
  但判据应当用**进程树（PPID）**，不是锁文件比对。
- `pkill -f 'tag _qperm'` **匹配到了自己的命令行**（命令串里就含该文本），
  把执行它的 shell 杀了，退出码 144 = 128+16。中括号技巧也没用，因为同一条命令的
  `J=` 变量里还有一份字面量。**教训：`pkill -f` 要么在独立的命令里执行，
  要么改用 pgrep + 显式 PID。**

### 06:10 注入路径的逐位审计：**通过**（0 / 18,473,168 位）

外部复核要求「配额相同 ≠ token 集合相同，必须逐位比掩码」。**要求成立，但它举的
反例在这里数学上不可能**：两边都是"按同一个 `s⁰` 取 top-b"，而
`|{s>τ}| = b₀ ⇒ 那 b₀ 个严格大于 τ 的就是最大的 b₀ 个`；平局只出现在 `= τ` 处、
排在其后，不会进 top-b₀。

**两级验证都过：**

1. **离线（真实分数，含真实平局结构）**：2464 个 (chunk,层,头) 上
   `{s>τ}` 与 `top-|{s>τ}|` **0 个不同集**。注意平局在真实数据里很常见 ——
   样例头 256 个分数只有 **22 个不同值**（bf16 量化），恰好 `= τ` 的有 22 个。
2. **进程内逐位（`VARIKV_INJECT_SELFCHECK`）**：Δb=0 注入的 `valid` 与同一进程、
   同一 `score0` 下的基线掩码 `vb` 做 XOR，11 个 chunk 累计
   **0 / 18,473,168 位不同**。

⇒ **注入路径与基线选出同一批 KV，tie 语义不构成污染。** 这条审计彻底关闭。

**于是「文本 50/50 不同」只剩数值不确定性一个解释**——不是断言，是**排除法**：
掩码逐位相同 ⇒ 送进注意力的 KV 相同 ⇒ 差异只能来自额外打分器前向改变了
kernel 选择/规约顺序（bf16 + 169k）。其下游幅度由 `_qzero` n=100 直接测量。

### 06:40 一条新的修正：「MultiHop 的最优 warp 是恒等」推不出来 —— 改用 γ sweep 直接测

外部复核指出我上一轮口头说的「MultiHop headroom 为负 ⇒ 那里正确的 warp 就是恒等」
**不成立**。`headroom < 0` 只说明**压缩胜过满缓存**，不说明 FastKVzip 那个特定分配是
最优的。MultiHop 的最优 warp 完全可能是**另一个方向**的 warp：

    W*_MultiHop ≠ W*_Retr.KV     ✓（跨 panel 表相关 −0.204，已测）
    W*_MultiHop = Identity        ✗ 未经检验的额外假设

**（另两条批评是它没注意到 `a2a9bf2` 已经撤回过：「单调性/低维是架构性的」已改成
参数化+初始化先验，「上尾收缩是机制」已降级为假说。）**

#### γ sweep：一次实验回答三个问题

按 γ 缩放配额表 `γ·Δb_h`，构成一条单参数曲线：

| γ | 含义 | 状态 |
|---|---|---|
| **< 0** | **反向 warp**（原来压顶尾 ⇒ 现在抬顶尾） | 新排 |
| 0 | 基线（= `_qzero`） | 已有 |
| 0.5 | 半强度 | 新排 |
| 1 | 学到的表 | 已有（Retr.KV +4.20★、MultiHop −8.80★） |

它同时给出：**(a)** 反向 warp 的因果对照（外部一直要的）；**(b)** MultiHop 上恒等是不是
最优（若 γ=0 是曲线极大值则是，**若 γ<0 更好则 MultiHop 需要相反方向的校准** ——
那会是个相当有意思的发现）；**(c)** 「什么时候不该校准」的 do-no-harm 门控的经验依据。

已在 MultiHop 上启动 γ ∈ {−1, −0.5, +0.5}（各 n=90，直接绑卡不走队列，避免 worker
累积导致的叠卡）。Retr.KV 的 γ=−1 反向对照表也已备好待排。

**为什么用配额表而不是 warp 本身做缩放**：配额是**实际效应**，在它上面线性插值是
"该搬多少"这个问题的干净参数化；而缩放 warp 还要经过 clamp/配平的非线性，两者不等价。

---

## 2026-08-18 07:30 —— 非单调性全网格：11/11 都有压缩增益，但**分界变量不是"压缩增益"**

**先纠正可验证性**：Figure 11 无数值表、本地 PDF 数据点已丢失（CLAUDE.md 早记过），
所以任何"论文图上哪几个 panel 明显"的排序**不可从手上材料核实**。改用自己的网格。

### 基线在 8 个 ratio 上的完整曲线（`__g8base`，Qwen2.5-7B-1M）

| panel | full | 压缩增益 `max_{ρ<1}P(ρ)−P(1)` | **最优 ρ** | ρ=0.1 |
|---|---|---|---|---|
| **Retr.MultiHop** | 41.07 | **+8.40** | **0.1** | **49.5（就是峰值）** |
| Retr.Prefix-Suffix | 50.00 | +7.80 | 0.4 | 8.6 |
| En.QA | 39.43 | +5.08 | 0.3 | 19.7 |
| Retr.KV | 68.20 | +3.40 | 0.5 | 32.6 |
| En.MultiChoice | 79.17 | +2.78 | 0.75 | 59.3 |
| Code.RepoQA | 58.64 | +2.27 | 0.5 | 13.0 |
| GSM8K | 70.00 | +2.00 | 0.75 | 35.0 |
| ICL.ManyShot | 37.78 | +1.11 | 0.4 | 31.9 |
| Math.Find | 33.17 | +1.00 | 0.5 | 29.7 |
| SQuAD | 93.21 | +0.98 | 0.4 | 61.2 |
| En.Summary | 36.63 | +0.19 | 0.75 | 30.6 |

**11/11 个 panel 都存在某个 ρ<1 使压缩超过满缓存。** 论文归因于 attention denoising。

### 「压缩越受益、我们越伤害」**不成立** —— 是单点杠杆

| 与 v2@0.1 的相关 | 含 MultiHop | **去掉 MultiHop** |
|---|---|---|
| 压缩增益 | Pearson **−0.567**，Spearman **+0.024** | Pearson **+0.443**，Spearman +0.410 |
| ρ=0.1 处距自身峰值的 slack | +0.517 / +0.505 | +0.266 / +0.318 |

Pearson 与 Spearman 在「压缩增益」上分道扬镳（−0.567 vs +0.024），且**去掉 MultiHop
直接翻号** ⇒ 那个负相关**完全由一个点撑起**。外部预测的 `Corr(G_ours,G_comp)<0`
**不被数据支持**。

**同样要修正一个常见误读**：「压缩受益的地方我们都伤」只对 MultiHop 成立 ——
**En.QA 压缩增益 +5.08（第三大），我们在那里 +2.50，是帮的。**

### 站得住的是**分界**，不是相关强度

| panel | ρ=0.1 处 slack | v2 Δ@0.1 |
|---|---|---|
| **Retr.MultiHop** | **0.0** | **−9.96 伤** |
| Math.Find | 4.5 | −0.17 平 |
| En.Summary / ICL.ManyShot | 6.2 / 7.0 | +1.00 / +0.00 平 |
| En.MultiChoice / En.QA / SQuAD / GSM8K / Retr.KV / Code.RepoQA | 22.7–48.0 | +0.00 ~ +6.94 多为帮 |

> **唯一被严重伤害的 panel，正是唯一 slack = 0 的 panel。** 基线已在自身最优点上，
> 任何方向的改动都只能变差。

（相关系数不可靠：n=10、slack 的 95% CI [−0.167, +0.865] 含 0。报分界，不报相关。）

### 机制：保真度目标与任务目标错位（已有直接测量支撑）

MultiHop 上我们**更忠实于满缓存**（`KL(p_full‖p)` 0.2575 → 0.1779，t=−3.49 显著）
**却低约 10 分**。教师优化 `min D(p_S, p_full)`，而任务要 `max Acc(S)`；驱逐同时造成
**信息丢失**与**干扰项去除**，教师只看前者。当去噪收益 > 信息损失时，"修正"把有益的
去噪一起修掉。**在跑的 γ sweep 正是这条的直接检验** —— γ<0 若更好，说明 MultiHop
需要**相反方向**的校准，而不只是"别动"。

### 一个便宜且改变第二个问题成本的发现

`attention/gate.py:load_gate` 现成支持 **`""`(KVzip) / `expect` / `snap` / `head` /
`fastkvzip`** —— **换 base scorer 只是一个 `-g` 标志**，没有整合成本。

⇒ 泛化工作应**拆两半**：**诊断半**（别的 scorer 是否也有跨头校准病理：测 `φ_h`
异质性、`A_h` 跨头跨度、上尾异质性）**不需要训练、不依赖我们的方法定型，现在就能做**，
约 1 GPU-h/scorer；**方法半**（在别的 scorer 上训练+评测）强依赖方法定型，应等
`_qperm` 与 γ sweep 出结果。

**另一处要纠正外部说法**：它说「降成 quota table 就难跨 scorer，而校准形式可以」——
**由等价定理二者是同一对象**，配额不迁移则校准也不迁移。可能迁移的是**函数形式
`W(z)` 作用到别的 scorer 的 z 上**，这要单独验证，不能靠"更抽象"来假定。

---

## 2026-08-18 08:10 —— 两处修正：一处是我的逻辑错误，一处是 winner's curse

### ① 撤回：「slack=0 ⇒ 任何方向的改动都只能变差」

我上一条写的这句**数学上不成立**。`slack = 0` 只说明在 **FastKVzip 自己那族按 ρ
参数化的策略**里，ρ=0.1 是**测过的**最好点：

    P_base(0.1) ≥ P_base(0.2), P_base(0.3), …

它**完全不排除**存在另一个等预算集合 `S'`，`|S'| = |S_base|` 而 `Acc(S') > Acc(S_base)`。

    ratio 轴上的最优  ≠  策略空间的最优

**而 γ<0 那一臂正是在测这个**：若反向搬更好，就直接构造出了反例。
正确表述只能是：**基线在 ρ=0.1 已处于自身压缩强度的最优，所以盲目"恢复信息"的
修正在那里特别危险** —— 而不是"任何改动都会坏"。

### ② winner's curse：「11/11 都有压缩增益」**是错的，真正站得住的只有 2 个**

朴素做法取 **8 个 ratio 的最大值**减 full cache，评测噪声下最大值系统性偏高。
改用**分半选择去偏**（在样本的 A 半上选 ρ*，在 B 半上评估，500 次随机分半）：

| panel | 朴素 max−full | **去偏** | 95% CI | 判定 |
|---|---|---|---|---|
| **Retr.MultiHop** | +8.40 | **+8.39** | [+5.87, +11.12] | **真** |
| **Retr.Prefix-Suffix** | +7.80 | **+7.75** | [+4.40, +11.20] | **真** |
| En.QA | +5.08 | +2.54 | [−1.95, +6.48] | 噪声 |
| Retr.KV | +3.40 | +2.76 | [−1.60, +6.80] | 噪声 |
| Code.RepoQA | +2.27 | +1.80 | [−1.36, +4.78] | 噪声 |
| GSM8K | +2.00 | +1.39 | [−9.05, +8.00] | 噪声 |
| ICL.ManyShot | +1.11 | **+0.01** | [−2.96, +2.96] | 噪声 |
| Math.Find | +1.00 | +0.27 | [−1.67, +2.00] | 噪声 |
| SQuAD | +0.98 | +0.66 | [−0.17, +1.59] | 噪声 |
| En.Summary | +0.19 | **−0.18** | [−1.33, +0.51] | 噪声 |

两个幸存者去偏后几乎不缩水（7.80→7.75、8.40→8.39），因为峰值远高于噪声；
小增益的那些**去偏后归零甚至翻号**（ManyShot +1.11→+0.01、Summary +0.19→−0.18）。
（En.MultiChoice n=18，样本不足，未纳入。）

### ③ 而这两个"真·去噪"的 panel，在我们方法下表现**完全相反**

| panel | 压缩最优 ρ | 我们操作点 | slack | 我们的 Δ |
|---|---|---|---|---|
| **Retr.MultiHop** | **0.1** | 0.1 | **0** | **−9.96★** |
| **Retr.PrefSuf** | 0.4 | 0.3 | 有 | **+8.40★**（v2c 两种子 +8.20±0.00★，全网格最大增益） |

> **这彻底否掉了「压缩受益 ⇒ 我们伤害」。** 两个唯一经 CI 确认有真实去噪收益的
> panel，一个是我们最大的败绩，另一个是我们最大的胜绩。区分它们的**不是"压缩帮不帮
> 这个任务"，而是"在我们操作的 ratio 上基线是否已在自身峰值"。**

这比相关系数强得多（那个 n=10、CI 含 0），因为它是在**同一类任务（真去噪）内部**
做的对比，把"任务是否受益于压缩"这个混淆因子控制住了。

### ④ 跨 scorer 诊断的设计要改（外部指出，成立）

- **`A_h = σ_h/σ_g` 不能跨 scorer 直接比**：它对全局仿射不变，但不同 scorer 的分数可
  相差任意**非线性**单调变换。只能当 diagnostic feature，不能当"校准质量"的真值。
- **要比的是标准化后的 shape**：先 `z = (s−μ_h)/σ_h`，再比 `q.99−q.90`、峰度、
  `CVaR.95` 的**跨头方差**。这恰好也是理论要的——`affine` 拿 0 而非线性拿 +3.6，
  说明起作用的是 **shape 异质性**，不是位置/尺度异质性。
- **"头内排序质量"必须先定义参照**。不能再用 `U^NLL`（已知逐 token 不稳）。
  统一用 `U^attn`（给定 S 闭式确定）算 Kendall τ，否则跨 scorer 无统一口径。

---

## 2026-08-18 08:40 —— **一个读表错误，和它揭开的一个必须正视的问题**

### 我的错误

上一条我用 `awk -F'|'` 取 `$9/$10/$11` 并标成 `ρ=0.3/0.2/0.1`，**实际是
`ρ=0.2/0.1/0.05`，整体错开一格**。于是我把 `Retr.PrefSuf` 在 ρ=0.2 的 `+8.40★`
当成了 ρ=0.3 的值，并据此写下"两个真去噪 panel 表现完全相反、PrefSuf 是我们最大胜绩"。

**正确的 `Retr.PrefSuf`（v2）：**

| ρ | 0.75 | 0.5 | **0.4** | 0.3 | 0.2 | 0.1 |
|---|---|---|---|---|---|---|
| 基线绝对分 | 36.8 | 48.2 | **57.8（峰值）** | 51.6 | 39.2 | 8.6 |
| v2 | +10.80★ | +0.80 | **−17.60★** | +4.60★ | +8.40★ | +2.40★ |
| v2c | +11.70★ | +2.60 | **−18.90★** | +4.20★ | +8.20★ | +2.00 |

**ρ=0.4 恰是它的压缩最优点，我们在那里砸掉 17.6–18.9 分。** 结论完全反转。

### 反转后的图像：**在每个 panel 自己的最优 ρ\* 上，我们 8/11 为负**

| panel | ρ\* | 基线@ρ\* | 我们@ρ\* |
|---|---|---|---|
| Retr.PrefSuf | 0.4 | 57.8 | **−17.60** |
| Retr.MultiHop | 0.1 | 49.5 | **−9.96** |
| ICL.ManyShot | 0.4 | 38.9 | −4.81 |
| Retr.KV | 0.5 | 71.6 | −4.60 |
| En.QA | 0.3 | 44.5 | −4.33 |
| Code.RepoQA / SQuAD / En.Summary | | | −1.59 / −0.54 / −0.09 |
| GSM8K / Math.Find / En.MultiChoice | | | +3.00 / +0.17 / +0.00 |

**panel 内部跨 ratio 的检验（n=77 格，控制住任务本身）**，比先前跨 panel 的 n=10 强得多：

| | Δ 均值 | 为负 |
|---|---|---|
| slack < 3（基线接近自身峰值） | **−1.41** | 23/33 |
| slack > 15（基线已崩） | **+3.44** | 3/19 |

Spearman +0.435（n=77）。**`slack` 框架存活并被大幅加强，但方向与我上一条说的相反。**

### 由此暴露出的、必须写进论文的检验：**调好 ρ 的裸基线**

| panel | 裸基线最好 | (ρ) | 我们@0.1 | 我们各 ρ 最好 | 差距 |
|---|---|---|---|---|---|
| Retr.KV | **71.6** | 0.5 | **37.0** | 68.0 | **−3.6** |
| Retr.PrefSuf | 57.8 | 0.4 | 11.0 | 56.2 | −1.6 |
| Retr.MultiHop | 49.5 | 0.1 | 39.5 | 41.3 | **−8.1** |
| ICL.ManyShot | 38.9 | 0.4 | 31.9 | 35.9 | −3.0 |
| Code.RepoQA | 60.9 | 0.5 | 13.9 | 61.6 | +0.7 |
| GSM8K | 72.0 | 0.75 | 35.0 | 75.0 | +3.0 |
| Math.Find / En.Summary / En.MultiChoice / SQuAD / En.QA | | | | | +0.8 / +0.2 / +0.0 / −0.3 / −0.6 |

> **我们的最好 > 裸基线最好，只有 4/11，且赢面 +0.2～+3.0、输面 −8.1～−1.6。**
> 在 Retr.KV 上我们据以立论的 `+4.20★` 对应绝对分 **37.0**，而调好 ρ 的裸基线是
> **71.6 —— 落后 34.6 分**。

**这不否定同 ratio 下的受控比较**（那仍是有效的机制证据），但它决定了论文能说什么：

- ✗ 「我们改进了 KV 压缩」—— **不成立**
- ✓ 「在**固定的激进预算下**，我们改进了压缩」—— 成立
- 且必须**显式给出 ratio-tuned oracle 这一列**，否则审稿人一定会问

**为什么这个说法仍然有意义**：真实部署里 ρ 由显存预算决定，而且 prefill 时**不知道
任务是什么**，无法逐任务调 ρ。所以"给定预算下更好"才是部署相关的命题。
但这必须明写，不能靠"同 ratio 比较"含糊过去。

**`slack` 框架恰好解释了为什么**：我们的方法在修复**过度压缩**造成的损伤，
所以只在 ratio 设得过激时有用；一旦基线本就在自己的最优点，加回保真度就是伤害。
这也是 fidelity ≠ utility 最强的形式。

### 教训

`awk -F'|'` 取列极易错位（表头前导空列 + 中文列名宽度不一）。**任何按列号取表格值的
分析，必须先把表头字段编号打出来逐一核对**，或者直接按表头名匹配，不要数列。

---

## 2026-08-18 09:10 —— **我把比较设成了错的问题；正确的结构是「断崖恢复」**

### 我错在哪

上一条我拿「调好 ρ 的裸基线最好值」（Retr.KV ρ=0.5 → 71.6）去比「我们在 ρ=0.2 的
64.0」，据此写「我们的最好只在 4/11 上胜出」。**那是拿 2 倍压缩去比 5 倍压缩 ——
根本不是压缩方法的比较。** 压缩方法的有效比较只有两种：**等预算**（同 ρ）与
**等质量**（达到同分数各需多大 ρ）。这条要撤回其"否定方法"的读法。

### Retr.KV 的全貌（full = 68.20）

| ρ | 1.0 | 0.75 | 0.5 | 0.4 | 0.3 | **0.2** | 0.1 |
|---|---|---|---|---|---|---|---|
| 基线 | 68.2 | 68.8 | **71.6** | 66.4 | 65.4 | **45.2 ← 断崖** | 32.6 |
| **我们** | — | 68.0 | 67.0 | 67.0 | 67.4 | **64.0** | 37.0 |

**我们那一行从 ρ=0.75 一路到 ρ=0.2 基本是平的（67–68）。基线在 0.3→0.2 掉 20.2 分，
我们没掉。** 断崖那一格我们补回 **18.80 = 跌幅的 93%**。

**等质量口径**：我们 ρ=0.2 的 64.0，裸基线要 ρ≈0.293 才能达到 ⇒ **缓存缩减 1.47×**。
（其它 panel：En.MultiChoice 1.54×、PrefSuf 1.34×、Retr.KV@0.1 1.35×、RepoQA 1.25×。）

### 核心检验：**我们补的正是基线刚丢掉的**（n=66 个 panel×ratio 格）

对每一格算「基线相对上一档跌了多少」`drop(ρ) = P(ρ_prev) − P(ρ)`，与我们的 `Δ(ρ)`：

| 基线刚跌 | 我们 Δ 均值 | 为正 |
|---|---|---|
| **几乎没跌 < 2 分** | **−1.41** | 13/44 |
| 小跌 2–10 | **+2.06** | 10/12 |
| **大跌 > 10 分** | **+5.83** | **9/10** |

**Pearson +0.462，Spearman +0.610。**

> **一句话：我们的方法补回基线刚丢掉的东西。基线没丢东西的地方，我们只是加噪声。**

这比先前的 `slack` 说法准确得多，也解释了所有先前看似矛盾的观察。

### 恢复比例还有二阶结构：**跌太狠了也补不回来**

| panel | 断崖 | 跌幅 | 我们补回 |
|---|---|---|---|
| **Retr.KV** | 0.3→0.2 | 20.2 | **93%** |
| En.MultiChoice | 0.2→0.1 | 13.0 | **54%** |
| En.Summary | 0.2→0.1 | 5.6 | 18% |
| SQuAD / En.QA / PrefSuf | 0.2→0.1 | 24–31 | 8–10% |
| Code.RepoQA | 0.2→0.1 | **44.8** | **2%** |

跌幅 13–20 分时能补回 54–93%；跌幅 >24 分时只补回 2–10%。**配额重分配只能修
"分配错了"，修不了"信息根本没留下"** —— 它只搬约 6% 的预算。

### MultiHop 因此被统一进来，不再是特例

它在可用区间（ρ≥0.1）内**根本没有断崖**：41.9 → 40.4 → 40.4 → 42.7 → 46.1 → 49.5，
一路上升到 ρ=0.1。最大单步跌幅只有 1.5 分。**没有可补的东西，所以我们只能伤。**
这与"基线几乎没跌的格子里 Δ 均值 −1.41"完全一致 —— MultiHop 不是例外，是那一类的极端。

### 论文能说什么（修正后）

- ✓ **等预算**：在断崖处 +18.80★（Retr.KV ρ=0.2）
- ✓ **等质量**：**1.47× 缓存缩减**（Retr.KV），这是压缩论文的标准口径
- ✓ 机制：**恢复过度压缩造成的配额错配**，且只在"错配"而非"信息缺失"主导时有效
- ⚠ **必须同时报**：基线未跌的区间我们**伤**（Retr.KV ρ=0.5 −4.6）⇒ 需要 do-no-harm
  门控。这不是致命缺陷，是方法的适用范围，但必须显式给出，不能只报断崖那一格。
- ⚠ ρ=0.5 处基线高于 full cache（71.6 vs 68.2）是**断崖之前**的现象，与我们的增益
  不在同一工作点，两者不冲突。

---

## 2026-08-18 10:00 —— **随机置换我们自己的配额表，比我们的表好 7 倍**（已验证，未解释）

### 结果（Retr.KV @ρ=0.1，**共同样本集 n=87**）

| 臂 | 搬动/chunk | 绝对分 | Δ | 95% CI |
|---|---|---|---|---|
| 基线 | 0 | 27.59 | — | |
| **零表（噪声基底）** | 0 | 27.59 | **+0.00** | **[+0.00, +0.00]** |
| **我们学到的表** | 4269 | 31.72 | **+4.14★** | [+1.84, +6.90] |
| `scalar` 网络 | — | 31.95 | +4.37★ | [+1.84, +7.13] |
| **MH 的表（幅度匹配）** | 4192 | 51.49 | **+23.91★** | [+18.85, +28.74] |
| **我们表的分层置换** | 4269 | **58.85** | **+31.26★** | [+25.52, +36.78] |

### 排除的三种解释

1. **不是配置错误**：三张表确实不同（`kv` vs `perm` 相关 −0.044）、`|Δb|₁` 完全相同、
   进程环境确认各用各的表。
2. **不是抽样差异**：上表已在**共同样本集**上算（各臂原本 100/89/100/100/87/100）。
3. **不是注入 bug**：`_qzero` 噪声基底恰好 **+0.00 [+0.00,+0.00]**；另跑一个
   inject+dump 的诊断，置换表的**实际搬动 4269.0 与离线预测逐位一致**、实际 Δb 与
   请求表**相关 +1.0000**、零配额头 71→56 与预测一致。

### 机制：查了五个量，**全都区分不出**

| | 基线 | 我们的表 | 分层置换 | MH幅度匹配 |
|---|---|---|---|---|
| 配额熵 | 0.5727 | 0.5862 | 0.5876 | 0.5954 |
| Gini | 0.8980 | 0.8919 | 0.8913 | 0.8874 |
| 零配额头 | **72** | 56 | 56 | 15 |
| 给饥饿头的净预算 | 0 | 75 | 251 | 3631 |
| 与层号的相关 | — | 0.205 | 0.214 | −0.375 |
| 最富 10 头净 Δb | — | −1933 | −936 | −3056 |
| **下游 Δ** | — | **+4.14** | **+31.26** | **+23.91** |

熵/Gini 几乎相同；零配额头数我们与置换**完全一样**；给饥饿头的预算 MH 是置换的 14 倍
却分数更低。**没有一个量与下游分数单调。暂时无法解释。**

### 一个顺带的结构性发现

**普通高斯随机表达不到同样的实际搬动量**：`realized` 随 λ **饱和在 2164**（目标 4269）。
原因是 **72/112 个头 `b_base=0`**，随机表把约一半负质量撒在这些空头上，全被 clamp 掉。
我们的表与分层置换都把负质量放在**有预算可给**的头上，所以能达到 4269。
⇒ **分层置换本来就是唯一可行的等幅度随机对照**，这一点是对的。

### 这意味着什么（保守表述）

- **等价定理仍然成立**：我们的表 +4.14 ≈ `scalar` 网络 +4.37。
- 但 **`+4.20` 不能读成"网络学到了好的分配"**。它只说明"做了某种大幅重分配"，
  而网络找到的那个**远不是好的**。
- **`bias` 只搬 4.87 槽位拿 +0.33** ⇒ 幅度本身是必要的，不是"随便动动就行"。
  所以现在的图像是：**幅度必要，方向……随机的反而更好，原因不明。**

### 最重要的限定：**这一切都在 ρ=0.1，那是个退化工作点**

ρ=0.1 上基线 **72/112 个头拿零配额**、绝对分已崩到 27.59。而**我们的头条
`+18.80★` 是在 ρ=0.2 的断崖处**。置换对照必须在那里重做才算数。

先前我直接把 ρ=0.1 的表注到 ρ=0.2 —— **那是错的**，两个 ratio 的 `b_base` 完全不同
（约 2 倍、饥饿头少得多），注过去只是一张错配的表。已杀掉重来：正在 ρ=0.2 上
重新导出配额（`_qdump02`），拿到之后再做 own vs 置换的对照。

---

## 2026-08-18 10:40 —— 三条外部批评的核实：一条被数据否掉、一条是我的疏漏、一条是真 bug

### ① 「训练 z 只覆盖阈值附近 ⇒ `W(6)` 是外推」—— **被数据否掉**

实测训练 trace 的 z 分布（近阈值候选，n=831,488）：

| 分位 | 0.001 | 0.01 | **0.5** | 0.99 | 0.999 | max |
|---|---|---|---|---|---|---|
| z | −5.77 | −1.59 | **+3.40** | +16.21 | +36.96 | **+95.60** |

**>z=2 占 74.8%，>z=4 占 35.9%，>z=6 占 13.4%。** z=6 有充分的训练支持，
`W(6)−6 = −0.97` 是**内插不是外推**。

顺带：近阈值候选的 z 中位数是 **+3.40**（高斯下 90 分位只有 +1.28）——**分数分布本身
就有很重的右尾**，正是 warp 在压的那个，两者自洽。

### ② 但它让我发现自己刻画 warp 的量程不全（我的疏漏）

先前用 z∈[−4, 8]，而真实 z 到 **+95.6**。（单调性证书当初用的是 [−14.65, 87.36]，
那个没问题；是**形状刻画**不全。）全量程重做：

| | z=−10 | z=0 | z=3 | z=6 | z=15 | z=40 | z=90 | min W′ | 非单调 |
|---|---|---|---|---|---|---|---|---|---|
| 训练后 s0/s1/s2 | +0.35~+0.94 | −0.42 | −0.85 | −0.98 | **−0.999** | **−0.999** | **−0.999** | +0.56~+0.67 | **0/112** |
| 随机初始化 ×2 | ±0.7 | ±0.03 | ±0.1 | ±0.13 | ±0.16 | ±0.3 | ±0.5 | +0.91~+1.00 | 0 |

**单调性在全量程 [−15, 95] 上仍然成立。** 但暴露一个机制细节：
**z>6 之后 warp 完全饱和在 −α，对分布顶部退化成纯逐头常数平移，不再区分高分 token。**
形状只在 z∈[−10, 6] 里起作用，而 13.4% 的近阈值候选在 z>6。

### ③ `°` 退化标记算错了 —— **真 bug，已修**

`wrapper.py:281-291` 的真实逻辑是**两步**：

```python
if clen < prefill_chunk_size:            # 16000
    window_size = int(window_ratio*clen) # window_ratio 默认 0.02  ← 短上下文重标定
if chunk_ratio*clen < window_size:
    chunk_ratio = 0.0                    # ⇒ 退化
```

报表写死了 `4096/r`，**漏掉第一步**。逐 panel 重算：

| panel | clen | **实际窗口** | 新规则退化的 ratio | 旧规则 |
|---|---|---|---|---|
| **gsm** | 86 | **1** | **[]** | 全部 8 个 ← **错** |
| **squad** | 203 | **4** | **[]** | 全部 8 个 ← **错** |
| 其余 9 个 | ≥26k | 4096 | 与旧规则逐格一致 | 一致 |

只影响两个短上下文 panel，但影响很大：它们**每一格都被误标为退化**。这也解释了
网格里 SQuAD 出现 `+4.00±0.14★°` 这种自相矛盾的格 —— `°` 声称"应为 0"却有显著正值，
看起来像实现 bug，**实际是标记本身算错**。已改为与 runtime 逐行一致的 `_degenerate()`。

**通用教训（第二次）**：报表脚本**不要重新模拟 runtime 逻辑**。这与"选项列表不要手抄
第二份"是同一类错误。最干净的做法是让 runtime 把 `effective_window` 与
`effective_chunk_ratio` 落盘（配额导出里已经有 `ratio` 字段，就是重标定后的值），
报表直接读。

### ④ 已接受但尚未做的三条（都是设计而非实现问题）

- **`affine → sz` 不是纯单因素**：同时改了函数类、参数共享结构、容量。要钉死"非线性
  本身"需要补一条**逐头非线性**臂（每个 (层,头) 一条 4/8 结点单调样条，不共享、不加
  额外特征）。若 ≈ `sz`，才能说缺的是非线性容量而非参数共享。
- **`szmr0 vs scalar` 不是容量匹配的 head-identity 消融**（4482→2434）。应改用
  **Zero-ID**（保留 embedding 槽与参数量，把 `e_h` 置零）或 **Shuffle-ID**
  （固定一个随机置换 `e_h → e_{π(h)}`）。
- **`scalar vs kv` 是架构比较不是特征消融**。只能说"达到这个量级不需要显式读 K/V"，
  不能说"K/V 零增量价值"——那需要 matched 的 `scalar` vs `scalar+KV`。
