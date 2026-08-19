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

### 11:00 `°` 标记：三层修复（上一条只改了脚本没重生成表格）

**用户指出表格里 GSM8K 仍带 `°`** —— 确实，我上一轮改了 `scratch_all_report.py`
却**没有重新生成 `RESULTS_GRID.md`**。而且重生成第一次也失败了：用了
`nohup ... &` 而没有 `setsid`，**被 Bash 工具的进程组清理杀掉**（CLAUDE.md 早记过这条，
我自己给循环写的规则里却漏了 setsid）。

#### 修复一：runtime 落盘实测值，报表不再重推

`model/wrapper.py` 加一行（纯增量、编译通过）：

    [effective] clen=… window=… chunk_ratio=… degenerate=…

`chunk_ratio == 0` 就是结构性退化的**定义**。这与「选项列表不要手抄第二份」是同一类
错误，**第二次犯**：报表脚本不得重新模拟 runtime 逻辑。

#### 修复二：公式回退用**实测**验证过

拿配额导出里的 effective ratio 对公式：

| panel | 名义 ρ | 实测 effective | 公式预测 | |
|---|---|---|---|---|
| Retr.KV | 0.1 | 0.07764 | 0.07770 | ✓ |
| Retr.KV | 0.2 | 0.18013 | 0.18018 | ✓ |
| Retr.MultiHop | 0.1 | 0.06939 | 0.06940 | ✓ |
| **ICL.ManyShot** | 0.1 | **0.0** | 0.0 | ✓ 确认退化 |
| Math.Find | 0.1 | 0.07224 | 0.07471 | ✗ |

`Math.Find` 不符是因为 **`TOK` 存的是标注长度不是实测**：反解得真实 clen≈136,890，
标注 149,860，差 **8.7%**。

#### 修复三：边界格不再硬判（我上一条说错了）

我说过"即使 clen 有 ±25% 误差也没有格子会翻转"——**错**。裕度 `ρ·clen/W` 落在
[0.8, 1.25] 的有 3 格，其中 **`squad`@ρ=0.02 裕度 1.015，几乎正好卡在边界**，
clen 差 2% 就翻。现在这类格返回 `None` 并标 **`?`**（不可判），不硬判。

#### 结果

| panel | 旧标记 | 新标记 |
|---|---|---|
| **gsm**（86 token，真实窗口 **1**） | 全 8 格 `°` | **全部正常** |
| **squad**（203，窗口 **4**） | 全 8 格 `°` | 仅 ρ=0.02 标 `?` |
| repoqa | 0.05 `°`、0.02 `°` | 0.05 → `?`、0.02 `°` |
| kv | 0.02 `°` | 0.02 → `?` |
| 其余 6 个长上下文 | 0.02 `°` | 不变 |

全表 `°` 从 40+ 降到 **18**，新增 4 个 `?`。

**直接后果**：SQuAD @ρ=0.1 的 `+4.17±0.27★` 现在是**干净的显著正值**。先前它带着 `°`
（"结构上应为 0"）看起来像实现 bug，**实际是标记算错**。GSM8K 整行同理。

### 11:40 一个新发现：`Δb` 与基线配额的相关随 ratio 单调走强

给 ρ=0.3/0.5 建置换对照时发现的（不是刻意找的）：

| ρ | 零配额头 | \|Δb\| 均值 | **corr(Δb, b₀)** | 分层置换的 \|相关\| |
|---|---|---|---|---|
| 0.1 | **72/112** | 76 | **−0.271** | 0.015 ✓ |
| 0.2 | 46 | 906 | −0.399 | 0.015 ✓ |
| 0.3 | 27 | 2277 | −0.524 | 0.235 |
| **0.5** | **2** | 5913 | **−0.699** | **0.631 ✗ 打不乱** |

> **压缩越温和，学到的"分配"就越接近一条简单规律：从配额多的头拿、给配额少的头。**
> 到 ρ=0.5 时这条规律占了约一半方差（`r² ≈ 0.49`）。

这直接解释了为什么 ρ=0.5 的**分层**置换失效——按 `b₀` 分层再层内打乱，**保留的恰好
就是这条主导规律**（相关 0.631）。所以在那里它不是"打乱搬给谁"的对照。

**改用全局置换 + λ 校准**：不分层地打乱，再用二分把**实际搬动量**校准回去。
得到 ρ=0.5 的对照：搬动量差 **0.0%**、与原表相关 **0.065**、与 `b₀` 相关 **0.112**
（原表 0.699）。**这个对照专门打掉"从富头拿给穷头"这条规律**，因此它与原表的比较
回答的是：那条简单规律是不是全部？

**方法学要点**：随机对照要打乱的是**你声称重要的那个结构**。按 `b₀` 分层在 ρ=0.1
（相关只有 −0.271）是合适的，在 ρ=0.5（−0.699）就不合适了 —— 同一种"随机对照"在
不同工作点上打乱的东西不一样，必须逐点检查它到底破坏了什么。

---

## 2026-08-18 12:10 —— **该不该校准**升为 P0；三个候选门控输入被否掉

外部复核指出的这条我核对后**完全同意，而且它比 ρ=0.1 的置换反常更上层**：

    ρ=0.5：基线 71.6（**高于满缓存 68.2**），我们 **−4.60** —— 破坏了一个本来就好的状态
    ρ=0.2：基线 45.2，我们 **+18.80** —— 但 64.0 仍**低于**基线自己在 ρ=0.5 的 71.6

> 所以当前方法是**修复器，不是优化器**：它救的是过度压缩造成的灾难，
> 没有找到比 baseline 最优点更好的 selection。**缺的那一层是"诊断器"。**

### 用现有配额导出直接测：什么无标签统计量能预测「该不该校准」

| panel | ρ | 零配额头 | 配额熵 | Gini | slack | **我们 Δ** |
|---|---|---|---|---|---|---|
| Retr.KV | 0.5 | 2.6% | 0.9001 | 0.479 | 0.0 | **−4.60** |
| Retr.KV | 0.3 | 24.5% | 0.8073 | 0.677 | 6.2 | +2.00 |
| **Retr.KV** | **0.2** | 41.3% | 0.7371 | 0.777 | 26.4 | **+18.80** |
| Retr.KV | 0.1 | 65.1% | 0.5727 | 0.898 | 39.0 | +4.40 |
| MultiHop | 0.1 | 12.5% | 0.6543 | 0.850 | 0.0 | **−9.96** |
| Math.Find | 0.1 | 68.2% | 0.5994 | 0.885 | 4.5 | −0.17 |
| ManyShot | 0.1 | 100% | 0.0000 | 0.000 | 7.0 | +0.00（退化） |

| 候选门控输入 | Pearson | Spearman |
|---|---|---|
| 零配额头比例 | +0.264 | +0.393 |
| 配额熵 | +0.054 | **−0.143** |
| Gini | +0.130 | +0.143 |
| **距自身峰值 slack** | **+0.691** | **+0.919** |

**两条结论：**

1. **外部提议的三个无标签输入（零配额头、配额熵、Gini）都不预测。** 关键反例在
   Retr.KV 内部：零配额头随 ρ **单调上升** 2.6→24.5→41.3→65.1%，而 Δ
   **非单调**（−4.60 → +2.00 → **+18.80** → +4.40，峰在 ρ=0.2）。
   **任何零配额头的单调函数都不可能预测它。**
2. **唯一有预测力的 `slack` 需要任务标签** —— 它是"基线离自己峰值多远"，
   推理时观测不到。**最好的预测量恰恰是不能当门控用的那个。**

### Δ 在 slack 上非单调，与"恢复比例"二阶结构一致

slack 0 → 无可修（且伤）；slack 26 → 修回 +18.80；slack 39 → 只 +4.40。
与先前测的一致：**跌 13–20 分能补回 54–93%，跌 >24 分只补 2–10%** ——
配额重分配只搬约 6% 预算，**能修"分配错了"，修不了"信息根本没留下"**。
所以门控要预测的其实是两个量的乘积：**可修复份额 × 损伤量**，而不是单一"损伤有多大"。

### 下一步：一个真正无标签、且直指"丢了多少"的候选

现有三个候选都在描述**配额分布的形状**，而 slack 描述的是**丢了多少信息**。
所以应该找后者的无标签代理。最自然的是**被驱逐的分数质量**：

    m_evict = Σ_{i∉S} s⁰_i / Σ_i s⁰_i     （或用 softmax 权重）

推理时完全可得、不需要标签。现有配额导出只存了 `b_arm/b_base/thres`，没有分数本身，
所以要给 dump 加一个字段（阈值上下的 `s⁰` 质量）再重测。这是下一个零训练实验。

**另外记一条方法学**：`slack` 虽不能当门控，但它定义了门控要**逼近的 oracle**。
最终目标可以写成让 gated 版本贴近上包络：

    P_gated(ρ) ≈ max( P_base(ρ), P_scalar(ρ) )

### 12:50 跨 scorer 泛化的前置：**SnapKV 与 Expected Attention 在 Retr.KV 上彻底归零**

排这两个基线是为跨 scorer 诊断做前置。结果是"做不了"：

| base scorer | ρ=0.2 | ρ=0.1 | 生成内容（样本 0） |
|---|---|---|---|
| **FastKVzip** | **28.67** | **8.67** | `"cb59052b-9128-…"` ✓ 正确检索 |
| SnapKV | **0.00** | 0.00 | `The value associated with the key "6ab6ea3e-…` 复述问题 |
| Expected Attention | **0.00** | 0.00 | `The value associated with the specified key "6ab6…` 同上 |

**不是配置 bug** —— 生成文本显示它们在复述问题而非检索答案。CLAUDE.md 早记过
"snapkv 的崩溃"，这里量化了：在 169k 上下文 + 10–20% 缓存下是**彻底归零**。

**后果**：跨 scorer 泛化的原计划在这个 panel 上堵死 —— 分数是 0 的 scorer 上
既没有可改进的东西，也测不出校准病理。

**我的疏漏**：排这两个基线时**没有先确认它们在该 panel 该 ratio 下可用**。
以后引入任何新 base scorer，第一步应当是**先扫它的工作区间**，再决定在哪里做对照。

**修法**：先找工作区间。(a) 更温和的 ratio（0.75 / 0.5）—— 它们可能只在 10–20%
缓存下崩；(b) 换短上下文 panel（`squad` 203 token、`gsm` 86、`many_shot` 26k）。
两者都便宜，等空卡即可。

### 顺带：Retr.KV 的反向 warp 对照（`_kvg-1`）跑满 n=100

| Retr.KV @0.1 | Δ |
|---|---|
| γ=+1 我们的表 | **+4.20★** |
| **γ=−1 反向** | **+0.00** [−2.80, +2.80] 中性 |
| 分层置换 | **+28.40★** |

**学到的那条轴正反加起来只跨 4.2 分，随机置换跨 28.4 分。** 对比 MultiHop 上
γ=−1 是 −33.20（那里方向极其重要）—— 同一个学到的方向，在两个 panel 上的
"方向重要性"差了一个量级。

### 13:20 SnapKV/Expected Attention 的 0.00 是**我的配置错误**，而错误本身印证了论文命题

用户追问那两个基线的 ratio、样本数与代码正确性，查下来是我排实验时用错了 `--level`。

#### 错在哪

我自己的 `scratch_repro_full.py:METHODS` 表就写着各方法要用不同的 level：

| 方法 | gate | **官方 level** | 我用的 |
|---|---|---|---|
| fastkvzip | `fastkvzip` | `pair` | `pair` ✓ |
| **Expected Attention** | `expect` | **`adakv-layer`** | `pair` ✗ |
| **SnapKV** | `snap` | **`pair-head`** | `pair` ✗ |

`score.py:threshold()` 按字符串分派：`"head" in level` → `_threshold_head`（**逐头均匀预算**）；
`"layer" in level` → `_threshold_layer`（逐层均匀，adakv 带 0.2 safeguard）；否则
→ `_threshold`（**全局阈值**）。

**所以我把 SnapKV 的分数送进了全局阈值。** SnapKV 的分数是逐头注意力导出的、
**跨头本就没有可比性**，全局比较下预算涌向少数头、其余饿死 ⇒ 归零。

> **讽刺的是：这恰恰是本项目在讲的跨头校准失效，只是以最极端的形式出现。**
> 但它不是 SnapKV 的合法基线，必须用 `pair-head` 重跑。

顺带更正规模：**不是 100 条，是 `--num 30`**，ratio 只有 0.2 与 0.1
（日志里的 `num 11` 是 chunk 数不是样本数）。重跑加上 ρ=0.75/0.5 以定位工作区间。

#### 换门控要不要训练：**不要**

SnapKV 与 Expected Attention 都是 training-free 打分器，`load_gate` 直接实例化。

#### 教师训练与门控的关系：**绑定的**

`scratch_ctrl_teacher.py:345` `--gate default="fastkvzip"`、388 行
`ModelKVzip(a.model,"retain",a.gate)`、441 行 `s0 = torch.stack(self.score,0)[..., lo:hi]`。
**trace 里的 `s0` 就是 FastKVzip 门控的分数**，而 `z=(s⁰−μ_h)/σ_h`、`mg`、`rs`
全由它导出。所以：

| | |
|---|---|
| **架构** | **门控无关** —— 只吃 `s⁰` 的统计量，不碰 K/V、不碰模型内部 |
| **权重** | **门控绑定** —— warp 是针对 FastKVzip 的分数分布校准的 |

⇒ **不是"训练一次通用"**。换 scorer 要 `--gate snap` 重新生成 trace 再训练
（脚本已暴露该参数，是一个标志的事）。这正是"泛化要拆成**诊断半**（不用训练）与
**方法半**（要训练）"的技术原因。

**教训**：引入任何新 base scorer，**第一步是查它官方要用的 level/配置**，
`METHODS` 表里早就写着 —— 我等于第三次犯了「已有的权威信息不去读，自己另起一套」。

### 13:40 泛化的坐标轴错了：决定方法适用性的是**阈值方案**，不是 scorer

用户追问「既然教师与门控绑定，你是要拿训练好的模型直接去评估别的门控吗」。
**不是** —— 排的重跑不带 `--ctrlm_ckpt`，是**纯基线**。但这个追问逼出了一个更重要的
结构性结论。

#### 逐个读阈值实现

| level | 实现 | 跨头预算能否变动 |
|---|---|---|
| `pair` | `_threshold`：全部 (层,头,token) 展平后一个全局阈值 | **能**，111 个自由度 |
| `adakv-layer` | `_threshold_layer`：**每层单独**对该层 [H,n] 展平做阈值 | **层内 4 头能竞争、跨层不能** ⇒ 28×3 = 84 |
| `pair-head` | `_threshold_head`：`k = int(n_seq*ratio)` 后逐头 `topk` | **不能，每头配额构造性固定** |

#### 于是一个可证的结论

我们的方法是**保序的逐头形变**。在 `pair-head` 下：

- 头内 top-k 是谁 —— 保序 ⇒ 改不了；
- k 是多少 —— `int(n_seq*ratio)` 固定 ⇒ 改不了。

    ⇒ **在 `pair-head` 下我们的方法是可证的 no-op（0 自由度）。**

这与 `°` 结构性退化是同一类"没有决策空间"，但成因不同：`°` 是**预算太小**，
这里是**预算被均匀分死**。

#### 泛化问题的坐标轴要换

我先前（以及外部复核）都把泛化想成「换 base scorer」。**真正决定适用性的是阈值方案：**

    方法生效  ⟺  阈值方案允许跨头预算竞争

- **FastKVzip** 用全局 `pair` ⇒ 跨头误校准才会咬人 ⇒ **这是我们方法的天然归宿，
  不是任意挑的一个 base**；
- **SnapKV** 用 `pair-head` ⇒ 预算均匀，**问题根本不存在**，也就无从修；
- **Expected Attention** 用 `adakv-layer` ⇒ 层内可竞争，**部分适用**（84/111 自由度），
  这是唯一值得做方法半的第二个 base；
- **Ada-KV / LKV 本身就是分配器** ⇒ 我们这一层要么冗余、要么需要另一种复合方式。

#### 对论文的影响

这**收窄但也澄清**了命题。不能写「一个通用的 KV 驱逐校准层」，应当写：

> 在**采用全局跨头阈值**的驱逐方案中，头内序基本可靠而跨头分数不可比；
> 一个保序校准即可修正，其决策内容等价于逐头配额重分配。

并且要明确：**采用均匀逐头预算的方案（如 SnapKV 官方配置）不在适用范围内 ——
不是我们做不好，是那里不存在这个问题。** 这比含糊地宣称通用性诚实得多，
也解释了为什么 Ada-KV 那条线与我们是"同一个问题的两种解法"而非竞品。

#### 还没做、但现在知道该怎么做的

- **诊断半**（不用训练）：在 `adakv-layer` 下测 Expected Attention 的**层内**跨头
  异质性，看是否也存在同样的病理。
- **方法半**（要训练）：只有 Expected Attention 值得做 —— `--gate expect` 重生成
  trace、重训，评测用 `adakv-layer`。SnapKV 不必做，做了也是 0。

### 14:10 `m_evict`：第一个可能可用的无标签门控输入，但**当前证据几乎是构造性的**

给配额导出加了「被驱逐的分数质量」字段后，在 Retr.KV 四个 ratio 上测：

| 名义 ρ | effective | **m_evict（原始）** | m_evict（softplus） | slack | 我们 Δ |
|---|---|---|---|---|---|
| 0.5 | 0.4876 | 0.0095 | 0.4803 | 0.0 | −4.60 |
| 0.3 | 0.2827 | 0.0409 | 0.6727 | 6.2 | +2.00 |
| 0.2 | 0.1802 | 0.1021 | 0.7737 | 26.4 | **+18.80** |
| 0.1 | 0.0777 | 0.3331 | 0.8831 | 39.0 | +4.40 |

| | 与 Δ 的 Spearman | |
|---|---|---|
| m_evict | **+0.679** | 与 slack 的 +0.698 基本相同 |
| **m_evict 与 slack 的相关** | **+0.908** | ← 看起来是好代理 |

#### 但这个 +0.908 **在 panel 内部几乎是构造性的**，不能当证据

在**同一个 panel 内部**，`m_evict` 与 `slack` **都随 ρ 单调**：压得越狠，丢的分数质量
越多，同时离自身峰值越远。所以它们高度相关**几乎是必然的**，不需要任何机制。

**真正的检验是跨 panel。** 关键反例已经摆在那里：

    Retr.KV  @ρ=0.1：effective 0.0776，slack **39.0**，我们 **+4.40**
    MultiHop @ρ=0.1：effective 0.0694，slack **0.0**，我们 **−9.96**

两者的 effective ratio 只差 10%，所以 `m_evict` **很可能也差不多** —— 而 slack 差了
**39 分**。**若 m_evict 在两者上接近，它就无法区分「该校准」与「不该校准」，
作为门控输入即告失败。**

⇒ **下一个必做实验**：在 MultiHop（以及 Math.Find）上用同一字段导一次配额，
比 `m_evict` 在跨 panel 上是否仍跟随 slack。**在那之前不得声称 m_evict 可用。**

这条也是一个通用的方法学提醒：**两个都随同一个自变量单调的量，其相关性不构成
预测力证据**。要检验预测力，必须在那个自变量被固定住的维度上比（这里就是固定 ρ、跨 panel）。

---

## 2026-08-18 14:40 —— **ρ=0.2 判决：置换反常是退化工作点特有，断崖处学到的分配值 +59.8 分**

| Retr.KV | own（学到的表） | perm（等幅度随机置换） | **own − perm** |
|---|---|---|---|
| **ρ=0.1**（72/112 头零配额） | +4.20★ | **+28.40★** | **−24.20** 置换赢 |
| **ρ=0.2**（断崖处） | **+25.80★** | **−34.00★** | **+59.80★** 学到的赢 |

**完全反转。** 两个 ratio 用的是同一套对照设计（分层置换、实际搬动量匹配、与原表相关
0.015）。⇒ **ρ=0.1 上"随机胜过学习"锁定为极端压缩下 global Top-B 的脆弱性**
（那里 72/112 个头被饿死，任何大幅重分配都比基线好），**不是方法普遍学错**。

这条把 f7d5577 那个"未解释的坏消息"降级为**工作点特有的现象**，并给出了它的边界：
ρ=0.1 是退化区，ρ=0.2 起学到的方向就有压倒性价值。

### 更强的一条：静态表在断崖处达到裸基线自己的峰值，缓存小 2.5×

`own` 的**绝对分 71.00**，而裸基线的峰值是 **71.6（在 ρ=0.5）**。

    等质量口径：ρ=0.2 达到 ρ=0.5 的质量  ⇒  **缓存缩减 2.5×**

（先前用 v2 的 +18.80 算出 1.47×，那是低估 —— 用 `scalar` 的表是 2.5×。）

### 一个意外：**静态表显著优于网络本身**

| | Δ@ρ=0.2 |
|---|---|
| 静态表 own（112 个数） | **+25.80★** |
| `scalar` 网络（4,482 参数） | +21.60★ |
| **静态表 − 网络** | **+4.20 [+1.20, +7.00]★** |

**在同一批 100 条样本上配对显著。** 静态表是网络逐 chunk 配额的**均值**，
所以这说明：**网络的逐 chunk 波动是有害噪声，抹平它反而更好。**

与 ρ=0.1 的对照有意思：那里两者一致（+4.20 vs +4.37，不可分），因为表本就 97% 静态；
到 ρ=0.2 网络的波动变大（`|Δb|` 均值从 76 涨到 906），波动就开始伤人。

⇒ **这直接支持"最终方法应该是配额表而非逐 token 网络"**，而且是有增益的简化，
不只是等价简化。但注意：表是在同 panel 20 条样本上测的，跨 panel 不迁移
（相关 −0.204），所以它仍不是可部署方法 —— 拿表本身要先跑一次打分器。

**限定**：单种子（`scalar` s0）的表。应当用 s1/s2 的表复现。

## 2026-08-18 — 断崖处的跨 panel 全表：正面 panel 不止一个；`m_evict` 作为门控输入失败

三条独立结果，全部零 GPU 从已有 eval 产物算出。

### ① ρ=0.2（断崖）上，干净版 v2 有**两个**显著为正的 panel，且都是 `Retr.*`

此前 `ICLR_PLAN.md` 标注的最大弱点是「现象基本只在 Retr.KV 上」。**那个判断是在 ρ=0.1
（退化工作点）下做的。** 在真正的工作点 ρ=0.2 上重新抽全 11 panel × 3 种子（每格
逐样本配对 bootstrap，★ = 95% CI 排除 0）：

| panel | 基线 | full | headroom | s0 | s1 | s2 | 均±散布 | **恢复率** | n |
|---|---|---|---|---|---|---|---|---|---|
| **Retr.KV** | 45.20 | 68.20 | **+23.00** | +19.00★ | +20.20★ | +22.60★ | **+20.60±1.50** | **90%** | 100 |
| **Retr.PrefSuf** | 39.20 | 50.00 | **+10.80** | +8.20★ | +8.20★ | +7.80★ | **+8.07±0.19** | **75%** | 100 |
| GSM8K | 63.00 | 70.00 | +7.00 | +0.00 | −2.00 | −1.00 | −1.00±0.82 | −14% | 100 |
| En.MultiChoice | 72.22 | 79.17 | +6.94 | +0.00 | +2.78 | +2.78 | +1.85±1.31 | 27% | 18 |
| ICL.ManyShot | 32.96 | 37.78 | +4.81 | +0.00 | +0.37 | −0.37 | +0.00±0.30 | 0% | 54 |
| Code.RepoQA | 57.73 | 58.64 | +0.91 | +0.68 | −0.23 | −0.45 | +0.00±0.49 | 0% | 88 |
| Math.Find | 32.33 | 33.17 | +0.83 | −0.83 | −0.00 | +0.33 | −0.17±0.49 | −20% | 100 |
| SQuAD | 92.65 | 93.21 | +0.56 | −0.38 | −0.44 | +0.57 | −0.08±0.46 | 100 |
| En.Summary | 36.29 | 36.63 | +0.34 | −0.40 | −0.04 | −0.28 | −0.24±0.15 | 70 |
| En.QA | 44.06 | 39.43 | −4.63 | +0.01 | +0.49 | −1.04 | −0.18±0.64 | 20 |
| **Retr.MultiHop** | 46.09 | 41.07 | **−5.02** | −5.87★ | −6.49★ | −8.98★ | **−7.11±1.34** | 142% | 90 |

**`Spearman(headroom, Δ) = +0.718`（11 panel）。**

**核心判读，措辞要严：**

1. **3/3 种子全 ★ 的 panel 恰好是 SCBench 的三个 `Retr.*` 任务**（KV / PrefSuf /
   MultiHop），而且**符号与 headroom 的符号三次全对**——两个正 headroom 上恢复
   75–90%，唯一负 headroom（压缩本来就赢过满缓存）上按预期变差。这是**方法在做
   它该做的事**的最强证据，不是碰运气。
2. **但恢复能力是任务型的，不是 headroom 型的。** 有 4.8–7.0 分 headroom 的
   GSM8K / En.MultiChoice / ICL.ManyShot **一分也没拿回来**。⇒ 驱逐分数的逐头配额
   重标定能修**检索型**损伤（哪几条 token 活下来决定能不能找到针），修不了
   **推理/ICL 型**损伤（那更像是弥散退化，不是丢了一根针）。**这是方法的真实边界，
   要写进 limitation，不要藏。**
3. **「去掉 Retr.KV 均值就塌」这个说法要重述。** 算术上仍然成立（11-panel 均值
   +1.977 → 去掉 Retr.KV 后 +0.114），但**11-panel 均值是错的统计量**：剩下 10 个
   里有 8 个 |headroom| ≤ 1，那里 Δ=0 是**正确结果而不是失败**。正确的说法是
   「在压缩真的有代价的 panel 上，检索类恢复 75–90%，推理/ICL 类恢复 ~0」。

**必须随表走的 caveat**：`CLAUDE.md` 记录 `scbench_prefix_suffix` 在 n=100 下很吵且
非单调。这里被部分化解——**配对差在固定 ρ 上抵掉了基线曲线的抖动**，且 3 个独立
种子给出 +8.20/+8.20/+7.80、散布仅 ±0.19。**没被化解的是 headroom 本身**：+10.80
来自单点基线与单点 full，继承了那条噪声曲线，所以「恢复 75%」这个**比值**比
「+8.07★」这个**配对差**弱一档。引用时优先引配对差。

### ② 置换优势 vs ρ：反转只发生在退化点

| ρ | 基线 | full | headroom | **own** | 等幅度分层置换 | own − perm |
|---|---|---|---|---|---|---|
| 0.5 | 71.6 | 68.2 | −3.4 | −4.80 | **−70.00** | +65.20★ |
| 0.3 | 65.4 | 68.2 | +2.8 | −0.60 | −13.40 | +12.80★ |
| **0.2** | 45.2 | 68.2 | +23.0 | **+25.80★** | −34.00 | **+59.80★** |
| 0.1 | 32.6 | 68.2 | +35.6 | +4.20 | **+28.40** | −24.20★ |

四个 ρ 中只有 ρ=0.1 反转。置换在 ρ=0.5 把 71.6 打到 **1.6** ⇒ 逐头配额分配在正常
工作点**极其脆弱**，随机重排即毁灭；它在 ρ=0.1 反而 +28.40，**只能说明那里基线
自己的配额分配已经差到随机都更好**（该处 65% 的头零配额），不能说明我们学错了。
⇒ **「随机胜过学习」锁定为退化工作点特有**，此前的悬案关闭。

### ③ 静态配额表跨种子几乎相同（+0.999）

用 s0/s1/s2 三个 `scalar` ckpt 各自在 Retr.KV @ρ=0.2 上导出逐 (chunk,头) 配额、
按 chunk 平均得 112 维表：

    corr(s0,s1) = +0.9994    corr(s0,s2) = +0.9988    corr(s1,s2) = +0.9986

⇒ 学到的配额重分配**几乎与随机初始化无关**，它在恢复模型自身的一个逐头结构性质，
不是优化器留下的痕迹。（下游复现正在跑：`_p02s1` / `_p02s2` / `_p02m3`。）

### ④ `m_evict` 作为「该不该校准」的门控输入：**失败**

上一轮预注册的判读条件是「若 `m_evict` 在 Retr.KV 与 MultiHop 上接近，它就区分不开」。
用三个 panel 在 ρ≈0.07–0.08 上的同口径导出：

| panel @ρ≈0.07–0.08 | effective ρ | **m_evict** | m_evict(softplus) | slack | 我们 Δ |
|---|---|---|---|---|---|
| **Retr.KV** | 0.0776 | **0.3331** | 0.8831 | **39.0** | **+4.40** |
| **Math.Find** | 0.0741 | **0.3331** | 0.8941 | 4.5 | −0.17 |
| MultiHop | 0.0694 | 0.2729 | **0.9219** | **0.0** | **−9.96** |

**Retr.KV 与 Math.Find 的 `m_evict` 完全相同（0.3331），而两者的 slack 是 39.0 vs 4.5、
Δ 是 +4.40 vs −0.17。** softplus 版方向还是反的——我们伤得最狠的 MultiHop 反而最高。
⇒ **`m_evict` 判死。** 至此四个无标签统计量全部失败：零配额头占比、配额熵、Gini、
`m_evict`。**「无害门控」不是随手能做出来的**，这条要写进 limitation 而不是当作待办。

顺带一个独立读数：零配额头占比随 ρ 单调（65.0% @0.078 → 41.2% @0.18 → 24.3% @0.283
→ 2.5% @0.488），但 **MultiHop@0.069 只有 12.5% 而 Math.Find@0.074 有 68%** ——
两个几乎同 ρ 的 panel 差 5 倍，所以它也确实不是 ρ 的函数、更不是效果的预测量。

### ⑤ 逐 ratio 展开后：命题从「两个 panel」升级为「**分数被推向满缓存**」，但有一个硬边界

上面 ①只看了 ρ=0.2 一行。把两个正 panel 的**全 ratio 曲线**摊开（干净版 v2，3 种子）：

**Retr.PrefSuf**（full = 50.00）

| ρ | 基线 | headroom | s0 | s1 | s2 | 均±散布 | 符号 |
|---|---|---|---|---|---|---|---|
| 0.75 | 36.80 | **+13.20** | +11.80★ | +11.60★ | +9.60★ | **+11.00±0.99** | ✓ 恢复 83% |
| 0.5 | 48.20 | +1.80 | +2.60 | +2.60 | −15.60★ | −3.47±**8.58** | 种子不稳 |
| 0.4 | **57.80** | **−7.80** | −19.20★ | −18.60★ | −27.80★ | **−21.87±4.20** | ✓ 该降 |
| 0.3 | 51.60 | −1.60 | +4.00★ | +4.40★ | +5.20★ | +4.53±0.50 | ✗（|headroom| 小） |
| **0.2** | 39.20 | **+10.80** | +8.20★ | +8.20★ | +7.80★ | **+8.07±0.19** | ✓ 恢复 75% |
| 0.1 | 8.60 | +41.40 | +2.00 | +2.00★ | +3.20★ | +2.40±0.57 | ✓ 符号对，量极小 |
| 0.05 | 1.20 | +48.80 | +0.20 | +0.20 | +0.20 | +0.20±0.00 | ✓ 符号对，量极小 |

**Retr.KV**（full = 68.20）

| ρ | 基线 | headroom | 均±散布 | 符号 |
|---|---|---|---|---|
| 0.75 | 68.80 | −0.60 | −1.33±0.82 | ✓ |
| 0.5 | 71.60 | −3.40 | **−6.07±0.68**（3/3★） | ✓ 该降 |
| 0.4 | 66.40 | +1.80 | −2.53±1.31 | ✗（|headroom| 小） |
| 0.3 | 65.40 | +2.80 | +2.53±1.79 | ✓ |
| **0.2** | 45.20 | **+23.00** | **+20.60±1.50**（3/3★） | ✓ 恢复 90% |
| 0.1 | 32.60 | +35.60 | +3.20±0.75 | ✓ 符号对，只恢复 9% |
| 0.05 | 2.00 | +66.20 | +0.73±0.09 | ✓ 符号对，只恢复 1% |

**两条结论，都比 ① 强：**

**(A) 「Δ 的符号 = headroom 的符号」在 2 panel × 7 ratio = 14 格里对了 12 格**，两个例外
（KV@0.4、PrefSuf@0.3）的 |headroom| 都 < 2.8 分、落在 headroom 自身的估计噪声里。
⇒ 可以把命题从「在某些数据集上涨分」升级为**「方法把分数推向满缓存」**——这是
*faithful restoration* 的直接行为证据，而且它**同时解释了所有掉分**：压缩本来就赢过
满缓存的格子（KV@0.5、PrefSuf@0.4、MultiHop 全程），我们必然、也应该把它拉下来。
这条不需要任何新实验，是已有数据的重读。

**(B) 但存在一个硬性操作窗口：基线必须还活着。** 恢复率对基线绝对分的依赖极陡——

    基线 ≳ 35：  恢复 75–90%（KV@0.2 90%、PrefSuf@0.75 83%、PrefSuf@0.2 75%）
    基线 < 10 ：  恢复 0.4–6%（PrefSuf@0.1 base 8.60 → 6%；@0.05 base 1.20 → 0.4%）
    KV@0.1 base 32.60 → 9%，正好卡在拐点上

**这不是"没调好"，是机制决定的。** 我们做的是**逐头预算再分配**，是把配额从一个头
挪到另一个头；ρ=0.1 时 65% 的头配额为零、ρ=0.05 时几乎全零，**没有东西可挪**。
方法不能凭空造出从未被保留的信息。⇒ **论文必须显式声明工作区间是"断崖处"而不是
"越极端越好"**，这与本项目此前"ratio 0.1–0.01 才是唯一有 headroom 的战场"的判断
（`CLAUDE.md` 2026-08-11 处方第 4 条）**方向相反**——那条处方按 headroom 选战场，
没有考虑机制的可达性。**据此撤回该处方第 4 条。**

**(C) 一个必须报的不稳定性**：PrefSuf@0.5 三个种子给出 +2.60/+2.60/**−15.60★**，
散布 ±8.58；@0.4 散布 ±4.20。同一 panel 在 ρ=0.2 只有 ±0.19。⇒ 种子稳定性本身是
ratio 的函数，**不能用某一个 ratio 的稳定性给整条曲线背书**。

**方法论教训（值得单列）**：PrefSuf@0.2 的 **+8.40★** 早就躺在 `RESULTS_GRID.md` 里
（原版 v2，单种子），与这里 3 种子的 +8.07±0.19 一致。「现象只在 Retr.KV 上」这个
被列为**最高风险**的判断，**是读 11-panel 均值而不是读逐 panel 列读出来的**。
⇒ **聚合统计量会藏掉第二个正结果。任何"只在 X 上有效"的结论，必须在逐格表上验证，
不能从均值反推。**

### ⑥ 配额表的不变性层级：`(panel, ρ)` 的函数，此外什么都不依赖

新导出 PrefSuf 与 MultiHop 在 ρ=0.2 的配额（`--num 20`，各 160 chunk），与已有的
Retr.KV 表放在一起，量四种"换掉一个变量表会变多少"：

| 换掉什么 | 表间相关 |
|---|---|
| **随机种子**（同 panel、同 ρ） | **+0.9986 ～ +0.9994** |
| **chunk**（同 panel、同 ρ；对半重估） | **+0.9999**；`R²_static` **0.85–0.94** |
| **panel**（同 ρ=0.2） | KV↔PrefSuf **+0.6353**、KV↔MultiHop **+0.4862**、PrefSuf↔MultiHop **+0.2591** |
| **ρ**（同 panel） | KV@0.1↔KV@0.2 **+0.1825**、VT@0.1↔VT@0.2 **+0.5350** |

**衰减对照已做，且不改变结论**：三张表的对半可靠性都是 0.999+，Spearman-Brown 全表
可靠性 0.9997–0.9999，所以跨 panel 的 +0.635 **去衰减后还是 +0.635**。那个差距是真的。

**⇒ 表 = f(panel, ρ)，与种子无关、与 chunk 基本无关。** 合起来给出网络的完整职能描述：

> **网络是一个 workload 分类器，输出一张 112 维的逐头配额表。**
> 它逐 chunk 的波动只占 6–15%（`R²_static` 0.85–0.94），而在 Retr.KV 上那部分被实测
> 为**有害噪声**（静态表 +25.80★ 显著优于网络 +21.60★，配对 +4.20 [+1.20,+7.00]★）；
> 它跨 panel / 跨 ρ 的变化才是本质（相关只有 0.18–0.64）。

**与 08-18 01:10 那条「跨 panel 那张表不迁移（KV vs MultiHop **−0.204**）」不矛盾**——
那是 **ρ=0.1**，本轮是 **ρ=0.2**。两个数都对。但并置之后要改口径：**不能说"表不迁移"，
要说"表是 (panel, ρ) 的函数"**，因为同一个 panel 换 ρ 也几乎换一张表
（KV@0.1↔KV@0.2 只有 +0.18）。

**预注册（结果未出，先写死判据）**：GPU1 正在用 **Retr.KV 的表按预算比例缩放 0.8499×**
打到 PrefSuf 上（`_ps02xfer`），GPU0 用 **PrefSuf 自己的表**（`_ps02tab`）。既然两表相关
+0.635，预测**迁移表拿到的分显著低于自表、但显著高于零**。若迁移表≈自表 ⇒ 那 0.635 之外
的部分不影响决策，表其实可跨 panel 复用（对方法是**利好**）；若迁移表≈0 或为负 ⇒ 必须
逐 workload 拿表，而拿表要先跑打分器，**方法不可部署**，只能作为对 +20.60 的刻画。

**一个附带的量纲读数**（同 ρ=0.2）：网络搬动的预算份额 `|Δb|/Σb` 为
Retr.KV **0.299%**、PrefSuf **0.139%**、MultiHop **0.877%** —— **我们伤得最狠的 panel
恰是网络动手最多的那个**，且 MultiHop 只有 **1.0%** 零配额头（KV 41.3%、PrefSuf 48.8%），
即"可搬的头最多"。这两件事可能是同一件事的两面（能搬所以搬得多），**所以不要急着
把 `|Δb|/Σb` 当成第五个门控候选** —— 它与零配额头占比强烈混淆，而后者已判失败。
要用它必须先在**固定零配额头占比**下比较，n=3 也远不够。

### ⑦ 外部分析评估（2026-08-18）：三条修正接受，一条方法论建议**方向搞反**

对着 `attention/score.py` 逐行核实，不是照着论述判。

**接受的三条：**

**(a) 术语必须从「cross-head」改成「cross-(layer,head)」。** `_threshold` 对
`score.reshape(-1)` 全局排序，池的是 **28 层 × 4 KV 头 = 112 组**，第 3 层 head0 与
第 20 层 head2 在同一个预算池里。写「跨头」会让审稿人问「到底来自跨头还是跨层」，
而当前 `pair` 把两者混在一起。DOF 计数已逐行核实：

| level | 代码 | 竞争域 | 配额自由度 |
|---|---|---|---|
| `pair-head` | `k = int(n_seq*ratio)` 逐头 topk | 112 个单元素域 | **0（构造性）** |
| `adakv-layer` | 逐层 `score.reshape(-1)` 求阈值 | 28 个 4 元素域 | **28×3 = 84** |
| `pair` | 全局 `score.reshape(-1)` | 1 个 112 元素域 | **111** |

**(b) 保序性目前只是 checkpoint 的经验性质，不是架构保证。** 这条本项目已记录
（`scalar` s1 的 `min ds'/ds` 只有 0.038），但与 no-op 定理**合起来**才看出后果：
no-op 定理的前提是「严格保序」，而我们的臂只是**恰好落在**那一类里。若换 `snap`
trace 重训，新 scalar 完全可能学成非单调、于是在 `pair-head` 下**不再是 no-op** ——
但那样「recalibrate, don't rerank」的故事也就没了。⇒ **最终方法必须写成构造性单调
的参数化**，这从"好习惯"升级为"定理成立的前提"。

**(c) Expected Attention 若走 `adakv-layer`，特征与竞争域不匹配。** 当前
`mg = (s−τ)/σ_g`、`rs = log(σ_h/σ_g)` 都是相对**全局** τ 与 σ_g 定义的，而
`adakv-layer` 的竞争边界是**逐层的 τ_l**。喂全局统计量给层内竞争的选择器是真实的
错配，必须改成 `mg_l = (s−τ_l)/σ_l`、`rs_l = log(σ_{l,h}/σ_l)`。**这条是代码级的、
之前没意识到，接受。**

**方向搞反的一条：建议把方法"升级"成 direct budget allocator（不预测 Δs，直接预测
`b_{l,h}`）以摆脱 threshold scheme 依赖，从而覆盖 SnapKV。**

机制上它说得对——由本项目自己的等价定理，保序 warp 的决策内容**就是**配额，直接
参数化配额严格更一般，且与 threshold 方案解耦。**但作为论文定位，这一步是往
Ada-KV / LKV 的正面战场走，不是拉开距离**，理由有三，前两条是硬的：

1. **Ada-KV 的机制就是 `adakv-layer` 本身**（层内把各头分数合池取 top-k）。而
   FastKVzip 的 `pair` 是它的**全局版**。⇒ 我们的基线**已经包含了自适应分配**，
   +20.60 不是"给均匀预算加上自适应分配"（那是 Ada-KV 的贡献），而是
   **"修正 raw-score 全局竞争所产生的分配"**。这个区别是我们唯一真正独有的命题，
   改叫 direct allocator 会把它主动丢掉。
2. **它没有解决可部署性，而是继承了它。** 我们今天刚测到表是 `f(panel, ρ)`：
   跨 panel 相关只有 +0.26~+0.64，**跨 ρ 更极端 —— PrefSuf@0.75 与 PrefSuf@0.2
   相关 +0.0055，同一个 panel 换预算就是完全无关的另一张表**。direct allocator
   同样要在推理时把这张表**推断**出来，问题原封不动。
3. 它提出的 `intervention gate G_φ` 被当作设计选项，但我们已有四个无标签统计量
   全部失败。**不过这一条要给它留余地**：我那四个是**无监督汇总统计量**，而 `G_φ`
   可以用教师（对满缓存的 KL）**有监督**训练——那不是同一件事，我的阴性结果
   **不构成对它的反驳**。它的真问题是能否跨未见 workload 泛化，目前双向都没证据。

**不接受但要改口的两条**：说「SnapKV 那里问题不存在」——我原话是「**当前保序形式**
在 `pair-head` 下可证 no-op，所以 SnapKV 的方法半不必做」，这是对的；但措辞确实
暗示永久放弃。正确说法：**SnapKV 只是对当前形式免疫，不代表自适应分配对它无价值**
（Ada-KV 正是干这个的）。同理 **Ada-KV / LKV 是核心 baseline，不是无关工作** ——
`CLAUDE.md` 早把"读这几篇原文"列为**阻塞新颖性声明**的待办，现在方法叙事已经收敛到
逐头预算分配，这条从"待办"升级为**投稿前必须完成**。（LKV = arXiv 2605.06676 为
外部转述，**我未读原文**，按本项目的 provenance 规矩不得据此下任何判断。）

### ⑧ 由 (a) 直接派生的新实验：把 +20.60 拆成「层内」与「跨层」

111 维配额空间有一个**精确正交分解**（112 = 28 层 × 4 KV 头）：

    across_{l,h} = L_l / 4,   L_l = Σ_{h∈l} Δb_{l,h}     层净变化均摊 → 27 维（Σ L_l = 0）
    within       = Δb − across                            层内 Σ=0     → 28×3 = 84 维
    27 + 84 = 111 ✓，且两者欧氏正交（across 层内为常数，within 层内和为零）

四张表的能量分配（已验正交性 < 1e-6 相对误差）：

| 表 | 跨层 across | 层内 within |
|---|---|---|
| Retr.KV@0.2 | 33.7% | 66.3% |
| PrefSuf@0.2 | 34.1% | 65.9% |
| MultiHop@0.2 | 38.8% | 61.2% |
| Retr.KV@0.1 | 26.2% | 73.8% |

**各向同性随机向量的期望是 84/111 = 75.7% 层内**，实测 61–74% ⇒ **跨层分量系统性地
比随机偏高**（33.7% vs 期望 24.3%），有真实跨层结构，但不占主导。

**能量不等于下游效果**，所以两个分量表已入队单独评（`_p02within` / `_p02across`，
n=100）。这个实验有一个很硬的推论：

> **若 within 分量单独就能拿回大部分 +20.60，则方法在只有 84 DOF 的 `adakv-layer`
> 下同样可用** —— 也就是不依赖 FastKVzip 的全局阈值，Expected Attention 那条路
> 直接打通；若必须 across 才行，则我此前"适用范围限于全局阈值方案"的窄口径是对的。

这是当前最便宜、判据最清楚的一个实验，且**它正是外部分析那条术语修正逼出来的** ——
接受批评的实际收益。

### ⑨ 竞争域定理的逐位验证：阴性对照过，且 `adakv-layer` **远不是 no-op**

`scratch_probe_domain.py`（新，零 GPU）。在真实 teacher trace 分数 + 真实训练好的
`scalar` 臂上，把同一批分数分别喂给三个阈值方案，逐位比较掩码。

**先看定理前提**：组内逆序对 **0 / 628,320 = 0.000000%** ⇒ 该 checkpoint 在实测分数
范围内严格保序，定理前提成立（注意这仍是**经验**性质，见 §⑦(b)）。

| level | 配额 DOF | 掩码翻转率 @ρ=0.2 | 配额变了的组 | **搬动量 Σ\|Δb\|/2 占总预算** | @ρ=0.1 |
|---|---|---|---|---|---|
| `pair-head` | 0 | **0.0000%** | **0 / 2,464** | **0.000%** | **0.000%** |
| `adakv-layer` | 84 | 5.5566% | 1,132 / 2,464 | **14.306%** | 12.783% |
| `pair` | 111 | 15.0646% | 1,323 / 2,464 | **37.671%** | 40.032% |

**阴性对照通过**：`pair-head` 下 0 位翻转、0 个组配额改变 —— no-op 定理不再只是纸上
证明加配额重放，而是**在 harness 真实的 `_threshold_head` 上逐位验证**过了。

**真正的新信息在第三列**：`adakv-layer` 实现了 `pair` 搬动量的 **38%**（14.31/37.67，
ρ=0.1 上 32%）。⇒ **跨层那 27 维虽然只占 24% 的维数、33.7% 的表能量，却贡献了 62%
的实际搬动量。** 机制上合理：层总量是大数，层间挪一次动几千个槽位；层内 4 头之间
挪，受该层预算封顶。

**必须同时声明的两条限制：**

1. **搬动量不是收益。** MultiHop 是搬得最多（0.877% 预算）同时伤得最狠（−7.11）的
   panel。这张表只界定「机制上够得着多少」，**不能推断增益在哪** —— 那由队列里的
   `_p02within` / `_p02across` 两个 n=100 评测裁决。
2. trace 每 (chunk,层,头) 只存 768 候选，这里是**子总体**上的 Top-B，绝对翻转率
   不可外推到推理时；三行同口径，**相对**比较有效。

**对 Expected Attention 路线的直接意义**：该方案走 `adakv-layer`，机制上够得着约
38% 的搬动量 —— **不是死路**。此前"适用范围限于全局阈值方案"的窄口径应放宽为
「`pair` 最优、`adakv-layer` 部分可用、`pair-head` 可证无效」。

### ⑩ 外部审计抓到一个正在污染实验的 bug；修完后跨 panel 迁移给出反预期结果

**这一条同时是「批评被证实」和「预注册预测被推翻」，两件都要记。**

#### (a) 离散化把「层内-only」偷偷变成了跨层干预 —— 批评正确，实验作废并重做

外部分析指出：`Δb^within` 与 `Δb^across` 的正交分解在**实数**上成立，但注入路径要
经过 `clamp(0,n) → round → 配平`，**runtime 实现的方向未必还是理论分量**。离线复现
`learned_ctrlcache.py` 的注入逻辑（220 个真实 chunk，Retr.KV @ρ=0.2）：

| 表 | clamp 到 0 的格/chunk | 请求−实现 L1 | 实现搬动量 | **层总量漂移** | 漂移/搬动 |
|---|---|---|---|---|---|
| full | 0.2 | 1,250 | 50,389 | 90,852 | 1.80 |
| **within** | **36.0** | **13,584** | 41,278 | **26,288** | **0.64** |
| across | 20.0 | 11,317 | 34,552 | 68,815 | 1.99 |

`within-only` 的**核心不变量是每层总量保持基线**，实测**逐层漂移均值 938.9 槽，只有
5.83% 的层无漂移**；`cos(实现within, 实现across) = +0.4509`（理论 0）；
`‖W+A−F‖/‖F‖ = 0.394`。**已跑的 `_p02within` 作业当场停掉并作废。**

根因比批评说的更彻底：**`within-only` 根本不能表示成固定的 112 维加性表**——它是
依赖运行时 `b0` 的**约束投影**。零配额头上 `clamp(0,n)` 截断（within 表每 chunk 截
36 格），缺口由配平循环在**全局**范围补，不受层内约束。

**修法**：把约束写进投影，加 `VARIKV_QUOTA_MODE = full | within | across`，三者共用
**同一张 full 表**：

    within : 每层总量锁死 = 基线层总量，层内按表分配后**在层内配平**
    across : 每层总量 = 基线 + L_l，层内按**基线比例**分配

修后复验（全 220 chunk）：**within 层总量漂移 0.00、无漂移层占比 100.00%**；
across 的层总量与理论层净变化 **corr = +1.0000**。

**一个措辞更正**：修后 `cos(within, across) = +0.28`，仍不是 0。这是**预期的** ——
`across` 用「层内按基线比例」而非「均摊」，两者不再构造性正交。**改用「各自锁死一个
约束来隔离通道」的表述，不再声称加性分解**；离散化后加性本就不成立。

**通用教训**：任何在连续空间做的分解，落到有 clamp / round / 配平的离散实现上都要
**逐不变量复验**，而且复验要在**真实 `b0`** 上做——理论分量的正交性不蕴含实现的正交性。

#### (b) 预注册判读出结果，方向与我的预测**相反**

Retr.PrefSuf @ρ=0.2（基线 39.20，full 50.00，headroom +10.80，n=100）：

| 臂 | Δ | 绝对分 |
|---|---|---|
| 自表（PrefSuf 自己导出） | +7.20 [+4.60,+10.00]★ | 46.40 |
| **迁移表（Retr.KV 表 ×0.8499 预算比）** | **+11.20 [+8.00,+14.60]★** | **50.40** |
| **配对差 自表 − 迁移表** | **−4.00 [−6.60,−1.40]★** | |

**外来的表显著优于本地表**，且迁移表拿到 headroom 的 **104%**（50.40 vs full 50.00）。
我在 §⑦ 里预注册的是「迁移显著低于自表、但显著高于零」，**前半句被推翻**。

按预注册判据，这一格判**可部署**：不需要在目标 workload 上先跑打分器。

**对照（同批）**：MultiHop @ρ=0.2 自表 **−12.53 [−15.11,−10.00]★**，比网络的
−7.11±1.34 **更差**。合起来是一个自洽的单一故事：

> **静态表 = 网络方向的「放大去噪版」。方向对时更好（Retr.KV +25.80 vs 网络 +21.60；
> PrefSuf 迁移 +11.20 vs 网络 +8.07），方向错时更坏（MultiHop −12.53 vs −7.11）。**

⇒ **「无害门控」因此更紧迫，不是更不紧迫** —— 静态表放大了收益也放大了损害。

**一个必须先排除的混淆，已入队**：迁移表的 |Δb| 均值 770，PrefSuf 自表只有 357，
**迁移表猛 2.2 倍**。所以 +11.20 > +7.20 可能只是「幅度不够」而非「方向更好」。
幅度匹配对照 `_ps02ownx`（PrefSuf 自表 ×2.157，|Δb| 精确匹配到 770.3）正在跑；
另加反向迁移 `_p02xferps` / `_p02xferpsx`（PrefSuf 表 → Retr.KV，按预算比 / 按幅度
匹配）检验对称性。**在幅度对照出来之前，不得把 +11.20 读成「方向可迁移」。**

### ⑪ 网络在 panel 内部**不看内容**：99.8% 的行为 = (头) + (头 × chunk 位置) 查找表

外部分析提出「static > dynamic 不代表上下文无用，可能是**自适应时间尺度错了**——
该按 context 适应而不是按 chunk」。这条零 GPU 可判，判完的结果比假说本身更有信息。

`scratch_quota_r02.jsonl` 的 `seq` 是**文档内 chunk 序号**（1..11），220 行 = 20 篇 ×
11 chunk，据此可做嵌套方差分解。

**第一步：文档级成分为零。** 模型 `Δb_{d,c,h} = α_h + β_{d,h} + ε_{d,c,h}`，
`σ²_β` 按 `Var(文档均值) − σ²_ε/C` 去偏（不去偏会把文档均值自身的抽样噪声算成信号）：

| panel | Var(文档均值) | 噪声预期 σ²_ε/C | 比值 |
|---|---|---|---|
| Retr.KV | 1.67e+03 | 3.06e+04 | **0.054** |
| PrefSuf | 9.50e+01 | 1.13e+04 | **0.008** |
| MultiHop | 6.26e+04 | 2.12e+05 | **0.295** |
| PrefSuf@0.75 | 1.20e+04 | 1.26e+06 | **0.010** |

四个全部 ≤1 ⇒ `σ²_β = 0`，**不是裁剪假象**。⇒ **一张 per-context 表捕获不到任何
静态表没有的东西，「按 context 适应」这条假说在 panel 内部不成立。**

**第二步：但比值远小于 1 说明还有结构没交代清楚。** 若 ε 真是 iid，文档均值方差应
**恰好等于** `σ²_ε/C`；实测只有它的 0.8%–29.5%，说明所谓「chunk 级变异」在文档内
几乎完全抵消 ⇒ 它是**跨文档共享的 chunk 位置效应**。再分解
`Δb = α_h + γ_{s,h} + 残差`（`γ` = 该 chunk 位置上跨文档平均）：

| panel | α_h 静态 | γ 位置×头 | **真残差** | 位置占非静态部分 |
|---|---|---|---|---|
| Retr.KV | 94.0% | 5.8% | **0.2%** | **96.6%** |
| PrefSuf | 85.0% | 14.9% | **0.1%** | **99.2%** |
| MultiHop | 92.3% | 4.2% | 3.5% | 54.4% |
| PrefSuf@0.75 | 83.3% | 16.5% | **0.2%** | **98.9%** |

**⇒ 三个 panel 上，(头) + (头 × chunk 位置) 两张表解释 99.8% 的总方差；真正依赖
文档内容的残差只有 0.1–0.2%。网络在 panel 内部根本不看内容，它是一张
`(头, chunk 序号)` 查找表。** MultiHop 是唯一例外（残差 3.5%，位置占比 54.4%）——
我们唯一伤害的 panel 也是唯一有真实内容依赖的，n=1，记录待查。

**这重新定义了「workload 分类器」这个说法。** 网络的输入只有分数统计量，而**同一
task format 在同一 chunk 位置上的分数分布几乎与文档无关**。所以它条件化的
「workload」粒度是**任务格式 × chunk 位置 × ρ**，不是文档。这与已测的三条完全一致：
换种子 +0.999、换 panel 0.26–0.64、换 ρ 低至 +0.0055。

**派生的可证伪预测，已入队 `_p02pos`（n=100）**：位置索引表 `[11,112]` 能复现网络
99.8% 的行为，而**网络比扁平表差**（+21.60 vs +25.80★，配对 +4.20★）
⇒ **预测位置索引表 ≤ 扁平表**。若成立，则「有害的那部分」就精确定位为**chunk 位置
效应**本身（早/晚 chunk 的差别化处理），而不是内容自适应——内容自适应根本不存在。
若位置表反而更好，说明扁平表的优势另有来源，当前解释要推翻。

注入代码已支持 2D 表：`[C,112]` 时按 chunk 序号取行，`lo` 回退（换序列）时计数归零。
逐行强制 Σ=0 保预算守恒。实测逐行 |Δb| 均值 `[638, 933, 980, 1009, 1015, 1010,
1013, 1015, 979, 910, 526]` —— **首尾两个 chunk 明显更保守**，中间趋于平台。

### ⑪-勘误（同日）：§⑪ 的标题结论**说过头了**，方差份额是个误导性口径

复查 §⑪ 时发现两处实质错误，都出在**用方差份额下绝对性结论**上。

**错误 1：「网络在 panel 内部根本不看内容」太强，MultiHop 上直接是错的。**
方差份额被巨大的**头主效应**稀释了。换成**绝对槽位**看同一批数据：

| panel | 表 \|Δb\| 均值 | γ 位置 sd | **E 内容残差 sd** | **E / \|Δb\|** | E / 逐头基线配额 |
|---|---|---|---|---|---|
| Retr.KV | 906 | 546 | **103** | **11.3%** | 3.8% |
| PrefSuf | 357 | 281 | **25** | **6.9%** | 1.1% |
| **MultiHop** | 2554 | 918 | **840** | **32.9%** | **32.3%** |
| PrefSuf@0.75 | 5427 | 2952 | **309** | **5.7%** | 3.1% |

MultiHop 的「3.5% 方差」对应 **840 槽的 sd，是它自己 \|Δb\| 的 33%**——**那不是残差，
那是主要成分之一**。⇒ **撤回「网络不看内容」这个无条件说法。** 站得住的版本是：

> 内容依赖在 Retr.KV / PrefSuf 上**远小于**头主效应与 chunk 位置效应
> （占 \|Δb\| 的 7–11%），在 **MultiHop 上不小（33%）**。

而 MultiHop 恰是我们唯一造成显著伤害的 panel。这两件事同时出现值得追，但 n=1，
**现在不得当成机制**。

**错误 2：即使在 Retr.KV 上，措辞也要靠保序性才成立。**
`Δb` 是每 (层,头) 对约 16k 个候选位置的**聚合计数**，聚合本身就抹掉 token 级内容
依赖，所以「配额与内容无关」**不蕴含**「逐 token 的 Δs 与内容无关」。正确的接法是
走保序性：该臂严格保序 ⇒ **选择完全由配额向量决定** ⇒ 配额的内容无关性等价于
**网络相对基线的增量贡献**的内容无关性。最终保留集里全部的内容依赖来自**基线打分器
的排序**。这反而是更干净的命题：

> **序数选择（内容相关，来自基线打分器） vs 基数分配（大体内容无关，来自表）**

**但它依赖一个尚未在评测分布上验过的前提。** 保序性只在 fineweb trace 上测过
（逆序对 0/628,320、配额重放 22/22）；scbench 的 z 范围可能落在
`scratch_probe_monotone.py` 的网格证书 z∈[−14.65, 87.36] 之外。已加
`VARIKV_RANK_SELFCHECK`：评测时把本臂的逐头配额按 `s⁰` 原序重放，与本臂实际掩码
逐位 XOR，0 才算成立。作业 `_rankchk` 已入队。**在它出来前，§⑪ 的因果措辞要挂条件。**

**同时确认无误的几项（复查过，不必再查）：**

- **展平顺序一致**：dump 写 `valid.sum(-1).flatten()`（`[L,H]` 行主序），注入读
  `b0.reshape(nL,H)`，离线分析用 `reshape(28,4)` —— 三处同为 (层, 头) 行主序。
- **`ratio=1.0` 不进 `prune_chunk`**（`wrapper.py:191`），所以满缓存参考不会推进
  位置计数、不会与 2D 表串位；dump 实测每样本恰好 11 次调用（220 行 / 20 篇）。
- **文档结构干净**：四个 panel 各 20 篇，chunk 数分别为 11 / 8 / 8 / 8，无残缺。
- **G 的估计噪声修正可忽略**（0.006%–0.18%），位置效应份额不是估计噪声撑起来的。
- **幅度对照 `_ps02ownx` 有效**：离线复现注入逻辑后，PrefSuf 自表 ×2.157 与迁移表的
  **实现搬动量**分别为 38,864 与 38,984，比值 **0.997**；clamp 分别只有 1.8 / 4.4
  格/chunk。⇒ 它确实是幅度匹配对照，不是又一个被离散化扭曲的量。
  （这一步是上一轮教训的应用：**表的 |Δb| 相等不等于实现搬动量相等**，必须复验。）
- **注入不改变基线**：零表下游 Δ = +0.00 [+0.00,+0.00]（n=100）已验过，所以
  `__g8base` 作为静态表实验的分母是合法的。
- **自表有轻微 train-on-test 且方向对我们有利**：PrefSuf 自表用样本 0–19 建、在
  0–99 上评，20% 重叠**偏袒自表**，而它仍然输给迁移表 4.00 分★。

**对 `_p02pos` 预测的相应弱化**：位置表丢弃的内容残差在 Retr.KV 上 sd 是 |Δb| 的
11.3%，不是可忽略量，所以「位置表 ≈ 网络 +21.60」这个点估计不成立。保留的仍是
**方向性**预测：**位置表 ≤ 扁平表**；若位置表反而更好，当前解释推翻。

### ⑫ 静态表优于网络**不是普遍现象**，只在 Retr.KV@0.2 上；外加一个自伤的运维教训

**(a) PrefSuf @ρ=0.75（第二个工作点，基线 36.80 / full 50.00 / headroom +13.20，n=100）**

| 臂 | Δ | 绝对 | 恢复率 |
|---|---|---|---|
| **静态表** `_ps75tab` | **+10.80 [+6.80,+15.00]★** | 47.60 | 82% |
| 网络 v2c s0 / s1 / s2 | +11.80★ / +11.60★ / +9.60★ | 48.60 / 48.40 / 46.40 | 89 / 88 / 73% |

**静态表与网络不可分**（静态 +10.80 落在网络三种子 +9.60…+11.80 的中间）。
⇒ **撤回任何「静态表普遍优于网络」的读法**：Retr.KV@0.2 上的 +25.80 vs +21.60
（配对 +4.20★）是**该工作点特有**的，不是方法的一般性质。合理解释是那里的 chunk
位置效应恰好有害；别处未必。

**(b) 运维教训：绝不要编辑有实例正在运行的 shell 脚本。**
`bash` 按**字节偏移**增量读取脚本文件。我在 `/tmp/qrun.sh` 有实例在跑时往里插了两行
（QMODE、XENV），偏移错位，运行中的实例在长命令返回后 seek 回旧偏移、读到半行并执行：

    /tmp/qrun.sh: line 13: ctrlm_ckpt: command not found

`_ps75tab` 因此以 rc=127 收场、日志被重定向截断。**幸而 100 个结果目录已落盘**，
数据没丢（上表就是从它们读出来的）。**改法：以后改启动器一律另存新路径**
（`/tmp/qrun_v2.sh`），不动运行中的文件。同类风险还包括编辑正在被 `source` 的文件。

### ⑬ 静态表跨三种子复现：+25.73 ± 0.09，且对同源网络的优势 +4.93 ± 0.57（3/3 ★）

Retr.KV @ρ=0.2，基线 45.20 / full 68.20 / headroom +23.00，全部 n=100，同一批基线。

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| 静态表 s0 | +25.80 | [+20.60,+31.00]★ | 71.00 | **112%** |
| 静态表 s1 | +25.60 | [+20.40,+30.80]★ | 70.80 | 111% |
| 静态表 s2 | +25.80 | [+20.60,+31.00]★ | 71.00 | 112% |
| 三种子**平均表** | +25.00 | [+19.80,+30.00]★ | 70.20 | 109% |
| 网络 v2c s0（架构参照） | +19.00 | [+14.00,+24.00]★ | 64.20 | 83% |

**跨三种子 +25.73 ± 0.09** —— 与三张表相关 +0.999 完全一致。

**平均表没有额外收益**（配对差 −0.60…−0.80，三次全不显著）。三张表已经几乎相同，
平均不带来信息，反而略微钝化。⇒ **不必做种子集成。**

**同源配对（静态表 sX vs 它所导出的 `scalar` 网络 sX）**，这是唯一正确的对照：

| 种子 | 网络 `scalar` | 静态表 | 配对差 静态 − 网络 |
|---|---|---|---|
| s0 | +21.60 | +25.80 | **+4.20 [+1.20,+7.00]★** |
| s1 | +20.00 | +25.60 | **+5.60 [+2.60,+8.60]★** |
| s2 | +20.80 | +25.80 | **+5.00 [+2.40,+7.60]★** |

**均值 +4.93 ± 0.57，3/3 种子 ★。** 丢掉 4,482 参数的网络、只留它的 112 个平均配额，
在这个工作点上**稳定多拿约 5 分**。

**但必须与 §⑫ 并读**：PrefSuf @ρ=0.75 上静态表与网络**不可分**。所以正确的命题是

> 静态表的优势是**工作点特有**的（Retr.KV@0.2 上 +4.93 ± 0.57，3/3 ★；
> PrefSuf@0.75 上不可分），**不是方法的一般性质**。

与 §⑪ 的机制解释一致：网络逐 chunk 的那部分几乎全是 **chunk 位置效应**，而位置效应
是否有害取决于工作点。`_p02pos`（位置索引表）正在跑，它直接检验这条。

**顺带**：静态表在 Retr.KV@0.2 的绝对分 71.00 **超过满缓存 68.20**（恢复率 112%）。
这是「压缩本身去噪」与「配额重分配」两个效应叠加，不是测量错误 —— 同 panel 的基线
在 ρ=0.5 上本来就有 71.60 > 68.20。

### ⑭ 保序前提在**评测分布**上验证通过：18.5M 位、0 位不同

§⑪-勘误 里挂的条件已解除。`VARIKV_RANK_SELFCHECK` 在真实 `scbench_kv` 评测中，
把本臂的逐 (层,头) 配额取出、按 `s⁰` **原序**重放，与本臂**实际**选出的掩码逐位 XOR：

    66 个 chunk，累计 0 / 18,478,208 位不同（每个 chunk 单独也都是 0）

此前保序性只有两条**训练分布**上的证据（fineweb trace 的逆序对 0/628,320、配额重放
22/22）与一个**网格**证书（z ∈ [−14.65, 87.36]，`scalar` s0 的 min ds'/ds = +0.19）。
风险在于 scbench 的 z 可能超出网格范围。**现在在评测分布上直接测过了。**

**因此下面这条从「依赖未验前提」升级为「已验证」：**

> 该臂严格保序 ⇒ **它的选择完全由 112 维配额向量决定**。于是
> **序数选择**（哪些 token 在头内排前面 —— 内容相关，完全来自基线打分器）与
> **基数分配**（每个头留几个 —— 学习部分的**全部**决策内容）
> 是**可分离的**，而且这个分离在评测时逐位成立，不是近似。

三条直接后果：

1. **静态表实验是合法的**：它们做的正是「同一配额 + `s⁰` 原序」，与网络的选择机制
   逐位同构，差别只在配额数值。所以 +25.73 ± 0.09 与网络 +20.80 ± 0.66 之间那
   +4.93 ± 0.57 是**纯粹的配额差异**造成的，不掺任何排序差异。
2. **§⑪ 的措辞可以定稿**：「配额与文档内容的关系」就是「学习部分与内容的关系」，
   不再需要为「聚合计数会抹掉 token 级依赖」留口子 —— 因为聚合计数**就是**全部
   决策内容。（但 §⑪-勘误 关于 MultiHop 内容残差占 |Δb| 33% 的更正**仍然有效**，
   那是配额本身的内容依赖，与本条无关。）
3. **等价定理的适用性不再需要 caveat**：`ICLR_PLAN.md §四之五` 与
   `scratch_probe_quota.py` 的定理此前都带着「trace 子总体、768 候选」的限制。
   本条在**完整候选集**（每 chunk 1.33M–1.79M 位）上直接验过，限制解除。

### ⑮ 投影器二次修正：他提的 Test 3 抓到真问题，但 `across` 的严重性被高估了

外部分析对 §⑩ 的修复提了两条。**逐条量过，一条对一条夸大。**

**(a) `across` 仍有 112 维全局配平 —— 方向对，量级被夸大。**
他称之为「论文级问题」。实测（60 个真实 chunk，Retr.KV @ρ=0.2）：

    |B_l 实际 − B_l 目标|：最大 **2 槽**，每 chunk 总偏离 **4.00 槽**，受影响 3.00/28 层
    占总预算 **0.0013%**（理论上界 28×0.5 = 14 槽，实测在界内）

⇒ 这是**定义不干净**，不是结果污染。他关于 `corr = 1.0000 不能证明严格性` 的方法论
批评**完全正确**（相关系数对整体平移/缩放不敏感），正确的度量是
`max_l |B_l^actual − B_l^target|`，而不是相关。已改度量。

**(b) 他提的 Test 3 抓到一个真问题：`within` 下 `Δ_lh = c_l` 不是 no-op。**
逐层常数在层总量锁死时**应当**被完全消掉。实测旧写法 **20/20 chunk 都不是 no-op**。
根因：**零配额头上的 `clamp(0,n)` 打破对称性**（ρ=0.2 有 41.3% 的头 b0=0），层常数
分量借 clamp 泄漏进层内再分配。

**但泄漏量同样要量，不能只看定性**：现行 `within`（喂 full 表）vs 正确 `within`
（喂逐层去均值表）——

    搬动量 38,622 vs 39,090，**差异搬动量 468 = 现行的 1.21%**，逐格相同 99.1%

⇒ **正在跑的 `_p02win2` 实质上就是正确的层内干预，不必中止**；已另排 `_p02win3`
用修正版重跑做 provenance。

**已实施的构造性定义**（两个消融从**表**上就分开，不再共用 full 表让投影去"自动
消掉"多余分量）：

    Δ^W_{l,h} = Δ_{l,h} − mean_h Δ_{l,·}        within：层总量锁死基线，层内配平
    Δ^A_l     = Σ_h Δ_{l,h}                      across：两级整数投影
                                                  ① 28 维层总量先整数化并严格配平到 B
                                                  ② 层内按基线比例分配，逐层严格配平
                                                  **不再做 112 维全局配平**

**新增 `scratch_test_project.py`（零 GPU，7 条不变量单测，在真实 `b0` 上跑）**，全过：

| 单测 | 结果 |
|---|---|
| T1 Δ=0 恒等（三种 mode） | 0/10 违反 |
| T2 Σb=B 且 0≤b≤n（随机 ±3000） | 0 违反 |
| **T3 `within(Δ_lh=c_l)` = no-op** | **0/10**（修正前 20/20 失败） |
| T4 `across(Σ_h Δ_lh=0)` = no-op | 0/10 |
| T5 within 逐层总量 = 基线 | 0/280 层 |
| **T6 across 逐层总量 = 目标** | **0/280 层，最大偏离 0**（修正前 2） |
| T7 极端饱和（b0 全 0 / 全 n，Δ = ±99999） | 0 违反 |

**这个文件是硬性前置**：本项目已被离散化坑掉两批实验，`Π(x+y) ≠ Π(x)+Π(y)` 是
系统性陷阱，任何新投影模式上 GPU 前必须先过这七条。

### ⑯ 外部分析中**接受但尚未执行**的几条，以及一条必须驳回的

**接受（已记，未做）：**

1. **满缓存模仿教师在「压缩胜过满缓存」的格子上是错误目标。** 我们的教师是 `U^full`
   （满缓存下的注意力效用），而 Retr.KV@0.5 基线 71.60 > full 68.20、MultiHop 全程
   如此。**在那些格子上 `Δb* = b^full − b^0` 不是最优目标，是有害目标。** 这与我们
   自己测到的「sign(Δ)=sign(headroom) 14 格对 12 格」是同一件事的两面：方法忠实地
   朝满缓存推，而满缓存不总是对的。⇒ **任何最终教师必须以下游效用而非满缓存保真度
   为准**，这条推翻了「fidelity recovery」作为终极目标的定位。
2. **README 与当前研究主线已是两篇论文。** README 仍以
   `F_i = D_i + λ·KL_i`、Gaussian memory、absorption 为主方法，而当前全部强证据来自
   score calibration / quota / competition topology，**不依赖 KL、不依赖 σ²、不依赖
   absorb-readback**。投稿前必须处理，否则无法回答「KL 在哪」。
3. **`_ps75tab` 以 rc=127 收场（结果完好但退出码非零）应重跑**，纯 provenance 考虑。
   已排 `_ps75b`。
4. **最终 controller 应构造性保序**（直接参数化配额 + 基线序数排序），而不是训练完
   再验「碰巧单调」（`scalar` s1 的余量只有 0.038）。
5. **三分划分**：目前 PrefSuf 自表用样本 0–19 建、0–99 评。虽然这个重叠**偏袒自表**
   而自表仍输，但最终所有表 / policy / gate 必须有 construction / validation / test
   三分。

**驳回一条**：他建议「先冻结基础设施、停止排新实验」。单测写完只花了几分钟且已并入
流水线，不构成阻塞；八卡当前跑的六个作业都是**已预注册判据**的关键读数，停掉的代价
远大于收益。**正确做法是单测与实验并行，而不是串行。**

**仍未读、不得据转述下判断**：Ada-KV（2407.11550）、LKV（2605.06676）、
DBTrimKV / Make Each Token Count（2605.09649）。他这次给 `2605.09649` 同时用了
DBTrimKV 与 Make Each Token Count 两个名字，与本仓库先前记录的 DBTrimKV 一致，
**先前标注的「同一 ID 两个标题」冲突就此解除**（很可能是同一篇的标题与代号）。

### ⑰ 预注册判读出结果：跨 panel 迁移的优势**完全是幅度**，但换来一个更值钱的发现

Retr.PrefSuf @ρ=0.2（基线 39.20 / full 50.00 / headroom +10.80，n=100，同批基线）：

| 臂 | 表 \|Δb\| | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|---|
| 自表（原始） | 357 | +7.20 | [+4.60,+10.00]★ | 46.40 | 67% |
| **自表 ×2.157（幅度匹配）** | **770** | **+12.00** | **[+8.60,+15.60]★** | **51.20** | **111%** |
| 迁移表 Retr.KV×0.8499 | 770 | +11.20 | [+8.00,+14.60]★ | 50.40 | 104% |

**两个配对判读：**

    迁移 − 幅度匹配自表 = −0.80 [−3.20,+1.60]   不可分
    幅度匹配自表 − 原始自表 = +4.80 [+2.40,+7.20]★

**按 §⑦ 的预注册判据：「方向本身可迁移」不成立。** 迁移表相对自表的 +4.00★ 优势
**完全由幅度解释** —— 迁移表的实现搬动量是自表的 2.2 倍（离线复验过：38,984 vs
18,349）。把自表放大到同样搬动量后，两者不可分，而且自表还略高一点。

**措辞必须严**：不可分 ≠ 等价。CI 半宽 ±2.4 分，只能排除 3 分以上的差异，不能声称
紧等价。能说的是：**在匹配搬动量后，外来表与本地表在这个 panel 上没有可测差别。**

**换来的发现比原命题更值钱：网络系统性地「下手太轻」。** 一个**一维标量增益**
把 PrefSuf 自表放大 2.157 倍，值 **+4.80 [+2.40,+7.20]★**。这比任何架构改动都便宜。

两个 panel 的最优点位置耐人寻味：**Retr.KV 静态表恢复 112%、PrefSuf ×2.157 恢复
111%** —— 都落在「略微过冲满缓存」的位置。这提示可能存在一个与 panel 无关的
**恢复率靶点（≈110%）**，而网络对 PrefSuf 的输出恰好欠了 2.2 倍。**但这是 n=2 的
观察，不是结论**；幅度扫描已入队（PrefSuf ×3 / ×4.5，Retr.KV ×0.5 / ×1.5 / ×2.5）
来定位峰、并检验 Retr.KV 是否已在峰上。

**这条同时改变了「静态表 vs 网络」的解释。** 此前把 Retr.KV@0.2 上静态表赢网络
+4.93★ 归因于「去掉了有害的 chunk 位置波动」。现在必须并列第二种解释：**静态表的
逐 chunk 平均并不改变幅度**（表就是 Δb 的均值），所以那 +4.93 不能用幅度解释 ——
两个 panel 的机制可能不同。`_p02x15`/`_p02x25`/`_p02x05` 会给出 Retr.KV 上的
幅度—效果曲线，若 ×1.0 就在峰上，则位置波动解释成立；若峰在别处，则「静态表更好」
里也混着幅度成分。

### ⑱ 投影器收敛为唯一实现；外部分析的两条工程批评都接受

**(a) 单测原本是镜像实现，不是生产代码。** `scratch_test_project.py` 自己重写了
`rebal`/`project` 并声明「与 `learned_ctrlcache.py` 一一对应」。当时两边确实一致
（我核过），但这种结构会让**「生产改了、测试没改」或「两边带同一个错」时测试仍然
全绿**。本项目已被离散化坑掉两批实验，不能留这个口子。

**已重构**：新增 `external/FastKVzip/prefill/attention/quota_project.py`，
`project_quota` / `rebalance` 是**唯一实现**；`learned_ctrlcache.py` 与
`scratch_test_project.py`（含 T6 算期望值的那一步）都 import 它。写文件用
「写 .tmp 再 `os.replace`」原子替换，避免正在启动的作业读到半个文件。

**(b) 非 full 模式后面仍挂着一个通用的 112 维全局修补循环。** 按当前数学，
within（逐层锁基线）与 across（两级各自严格配平）都必然 `diff = 0`，所以那段
**现实中永不执行**。但留着它意味着：**将来投影一旦出 bug，修补循环会悄悄把预算
"修好"，同时破坏因果不变量，实验看起来照跑** —— 这正是本项目栽过两次的失败模式。
**已改为断言**：`project_quota` 内部对非 full 模式直接
`assert Σb == Btot`，不做兜底修补；full 模式仍走配平（它本来就需要）。

重构后 7 条不变量单测在 3 chunk 上全过（Δ=0 恒等 ×3 模式、预算守恒与上下界、
`within(Δ_lh=c_l)` no-op、`across(Σ_hΔ=0)` no-op、within 逐层总量=基线、
across 逐层总量=目标且最大偏离 0、极端饱和），20 chunk 完整版后台复跑。

**代价**：生产实现是 torch 版，比原 numpy 镜像慢，20 chunk 全套超过 2 分钟。
这是正确的取舍 —— 测得慢好过测了个假的。

### ⑲ 外部分析中**最有价值的一条建议**：跨方法配额移植（可立即做）

他建议把另一个 learned eviction 方法的最终逐 (层,头) 配额取出来、**丢掉它的 token
排序**、用 FastKVzip 的排序重放，看性能保留多少。若大部分保留，则论文从
「我们改进了 FastKVzip」升级为**「对一类 learned global eviction 方法的结构性解释」**。

**他想用 DBTrimKV/LKV，但那两篇没有本地代码。而本仓库现成就有三个别的打分器**：
`expect`（Expected Attention，`adakv-layer`）、`snap`（SnapKV，`pair-head`）、
KVzip 的重构式打分。**所以这个实验今天就能做，不需要任何新依赖。**

设计（2×2 起步，可扩成 3×3）：

    排序来自 A × 配额来自 B  →  评测
    A=B 即原方法；A≠B 即移植

已有的两个机制正好提供工具：`VARIKV_QUOTA_DUMP` 导出任一 gate 的逐 (层,头) 配额
（dump 里的 `b_base` 就是该 gate 自己的选择），`VARIKV_QUOTA_INJECT` 则用
**`score0` 的排序**重放给定配额 —— 两者合起来就是移植。

**必须先解决的一个量纲问题**：注入表是固定 112 维，而配额逐 chunk 变。已测得
(头)+(头×chunk 位置) 解释 99.8% 方差，所以用 **2D `[C,112]` 位置索引表**即可，
该支持已经在注入代码里。

**判据（先写死）**：若「FastKVzip 排序 + Expected 配额」≈「Expected 原方法」，
则两个方法的差异**主要在配额而非排序**；若远低于，则排序差异是主因。**任一方向
都是结论**，这是本项目少见的"怎么样都有信息"的实验。

**同时必须声明它测不到什么**：`pair-head`（SnapKV）的配额构造性均匀，所以
「FastKVzip 排序 + SnapKV 配额」= 「FastKVzip 排序 + 均匀配额」，那检验的是
**均匀配额相对全局竞争配额差多少**，不是 SnapKV 的特性。

### ⑳ 他其余判断的核对结果

**同意且已记**：`within`/`across` 当前定义正确；rank replay 的 0/18.5M 是全项目
最强的一条；`n=2` 的「110% 恢复靶点」不得写进结论（我原文已标 n=2）；旧的
`_p02win2`/`_p02acr2` 只作历史、正式表用 `_p02win3`/`_p02acr3`；最终 controller
应构造性保序而非训完碰巧单调；三分划分；满缓存模仿教师在「压缩胜过满缓存」处是
错误目标。

**同意但要收紧措辞**：他写「static universally beats dynamic 已被否掉」—— 对，
但正面命题只能说「Retr.KV@0.2 上网络的逐 chunk 配额波动净贡献为负；其他工作点未必」。

**仍不采信**：Ada-KV / LKV / DBTrimKV 的具体主张全部来自转述，**三篇原文我都没读**。
他这次对 LKV 的描述（"learned budgeting 是主要性能来源"）若属实，会直接压缩我们
「基数分配比序数排序重要」这条的新颖性 —— **正因为它这么关键，更不能靠转述定案**。

### ㉑ 版本对照表（哪个作业跑的是哪版投影代码）

投影器在一天内改了三次，必须记清楚每个作业对应哪版，否则以后无法判断哪些结果可比。

| 提交 | 时间 | 投影版本 | 用它的作业（派发时间） |
|---|---|---|---|
| （首版 `VARIKV_QUOTA_MODE`） | ~09:40 | within 喂 full 表 + 层内配平；across 层内 round 后**全局**补 | `_p02win2`（09:52）、`_p02acr2`（10:05） |
| `5b238c6` | **10:22** | **构造性**：within 显式逐层去均值；across 两级整数投影 | **`_p02win3`（10:34）** |
| `208ad65` | **11:10** | 抽成唯一实现 `attention/quota_project.py`，非 full 模式禁止兜底配平（行为等价，7 条单测全过） | `_p02acr3` 及之后的幅度扫描 |

**结论**：`_p02win3` 已经是修正版；`_p02acr3` 会用重构版。旧的 `_p02win2`/`_p02acr2`
与修正版差 1.21% 搬动量、99.1% 的格逐位相同，**作历史保留，正式表不用**。

### ㉒ 跨方法配额移植：前置检查与实验设计（已挂冒烟）

按 §⑲ 的判据把实验落地。本仓库的 `scratch_repro_full.py:METHODS` 给出 gate↔level 对应：

    fastkvzip → gate fastkvzip, level pair          （111 DOF，全局竞争）
    expected  → gate expect,    level adakv-layer   （84 DOF，层内 4 头竞争）
    snapkv    → gate snap,      level pair-head     （0 DOF，配额构造性均匀）
    duoattn   → gate head,      level pair

**完整设计（6 个作业）：**

    (a) expect 基线 @adakv-layer, n=100, ρ=0.2
    (b) snap   基线 @pair-head,   n=100, ρ=0.2
    (c) expect 配额导出（--num 20）
    (d) snap   配额导出（--num 20）
    (e) fastkvzip 排序 + expect 配额, n=100     ← 移植
    (f) fastkvzip 排序 + snap 配额,   n=100     ← 移植

**dump 取的是该 gate 自己的配额，不受 ctrlm 干扰**：dump 代码里
`_v0 = ... self.threshold(score0, ratio, level)[0]`，`score0` 是**未被修正**的 gate 分数，
所以 `b_base` 就是该方法自己的选择。已核代码。

**(f) 的语义要说清**：`pair-head` 的配额构造性均匀，所以「fastkvzip 排序 + snap 配额」
= 「fastkvzip 排序 + **均匀**配额」。它测的**不是 SnapKV 的特性**，而是
**全局竞争产生的非均匀配额相对均匀配额值多少**（固定排序）。这本身是个好对照 ——
它给出「配额自由度从 0 升到 111 值多少分」的直接读数。

**先跑冒烟**（`_smkexp` / `_smksnap`，各 2 条）：`expect` / `snap` 两个 gate 从未与
带 dump/inject 的 `LearnedControlRetainCache` 一起跑过，先确认不炸、且 dump 出的
`b_base` 形状与语义正确（尤其 `snap` 应当逐头相等）。**冒烟不过就不要排后面 6 个。**

启动器另存为 `/tmp/qrun2.sh`（加 gate/level 两参）——**没有改运行中的 `/tmp/qrun.sh`**，
遵守「绝不编辑有实例正在运行的脚本」这条（今天已因此打挂五个作业）。

### ㉓ 重构的四级验证，外加一处需要收紧的旧措辞

用户要求在推送前确认重构无误。**重构此前只有单测覆盖，没被真实作业跑过**，所以补了四级：

| 级别 | 方法 | 结果 |
|---|---|---|
| 静态 | 查已删变量（`_rebal` / `nL` / `tgt`）是否还有残留引用 | **0 处残留** |
| 单测 | `scratch_test_project.py` 20 chunk 真实 b0 | **7 条全过** |
| **端到端（CPU）** | 用真实 `KVScore.threshold` 造 `vb`，再**逐行**执行注入块的算术链（`argsort` → `scatter_` → 掩码），检查①掩码计数 == 配额 ②预算守恒 ③**Δ=0 时逐位等于基线** | **3 个 level × 3 个 mode × (Δ=0, 随机Δ) = 18 组全过** |
| **真卡** | `_p02acr3`（11:16 派发，用的正是重构版）在 GPU5 跑 across 模式 | **连过 9 个 chunk 无报错**；`project_quota` 内部**每 chunk 断言预算守恒**，9 次未触发 |

第三级是关键：它复现的是**生产块的完整算术链**（不只是 `project_quota`），唯一去掉的
是 LLM 本身。第四级则确认真卡路径可跑。

**顺带发现一处旧措辞要收紧：`adakv-layer = 84 DOF` 是理想化计数。**
`_threshold_layer` 用的是**分位阈值 + 严格大于**（`thres = score_sort[n]`,
`valid = score > thres`），不是精确 top-k，所以层总量只是**近似**固定。实测
`expect@adakv-layer` 的层总量跨层相对 std **0.0180**、极差/均值 **0.0711**。
⇒ **实现的可达空间略大于 84**（多出的那点来自阈值边界与平局）。此前把 84 说成精确
自由度不准确；**逐位探针测到的「`adakv-layer` 实现 `pair` 搬动量的 38%」是实测量，
不受这条影响**。

### ㉔ 冒烟通过，且独立证实了竞争域定理；移植实验的量纲前提也查清了

`expect@adakv-layer` 与 `snap@pair-head` 首次与带 dump 的 cache 同跑，**双双 rc=0**，
各导出 22 chunk。

| gate@level | 层内 4 头逐位相等 | 层内相对离散 | 零配额头 | Σb（ρ=0.2, scbench_kv） |
|---|---|---|---|---|
| `snap@pair-head` | **True** | **0.0000** | 0.0% | 302,695 |
| `expect@adakv-layer` | False | 1.0415 | 0.0% | 295,724 |
| `fastkvzip@pair`（已有） | False | 1.1230 | 41.3% | 302,711 |

**`snap` 的配额逐位均匀 = 0 DOF，这是竞争域定理在一个完全不同的打分器上的独立证实**
（此前只在我们自己的臂上做过阴性对照）。

**两条移植实验的前提，现在查清了：**

1. **预算不等**：`expect` 比 `fastkvzip` 少留 **2.31%**。⇒ 移植表必须取
   `Δ = mean(b_expect) − mean(b_fastkvzip)` **再减均值强制 Σ=0**，移植的是
   **等预算下的配额形状**；否则比的不是同一个压缩率，分数无效。
2. **零配额头差异巨大**：`fastkvzip@pair` 有 **41.3%** 的头配额为零，而 `expect`/`snap`
   都是 **0.0%**。这不是小差别 —— 全局竞争会把整个头饿死，层内竞争与均匀配额不会。
   ⇒ 移植 `expect` 的配额到 `fastkvzip` 上，等于**同时**去掉了饿死头，两个效应会
   混在一起。**报结果时必须把「饿死头占比」作为协变量一起报**，不能只报分数差。

另一个意外读数：**`fastkvzip@pair` 的层内离散（1.1230）只比 `expect@adakv-layer`
（1.0415）高一点点** —— 全局竞争相对层内竞争，并没有显著加大**层内**的不均匀度，
它的额外自由度主要花在**跨层**上。这与 §⑨ 测到的「跨层贡献 62% 的搬动量」一致。

### ㉕ 撤回一条推理，并补上最危险的那个对照：防饿死

外部分析三条修正，**两条我接受，其中一条直接推翻我自己的推理**。

**(a) 撤回「跨层贡献 62% 的搬动量」（§⑨）。** 代码核实：`score.py:104`

    safeguard = 0.2 if "adakv" in level else 0

`adakv-layer` 在层内竞争**之前**先给每个头保护 `int(n_kept·0.2)` 个 top-K 条目。
所以 `pair`（37.67%）→ `adakv-layer`（14.31%）的搬动量下降**同时**包含两件事：
① 去掉跨层自由度；② 加了一个逐头 20% 保护。**两者不可分**，我把差值全部归给
跨层是错的。**能说的只有**：`adakv-layer` 这个具体实现产生了 `pair` 约 38% 的搬动量。
真正回答「层内 vs 跨层」的是 `_p02win3` / `_p02acr3` 这对**因果消融**，它们在跑。

**(b) 同一条 safeguard 顺带解释了另一个观察，并引出真正的威胁。**
`expect@adakv-layer` 的零配额头是 **0.0%**，此前我以为是 Expected Attention 分数的
性质 —— **不是，那就是 `safeguard=0.2` 这个逐头地板**。而 `fastkvzip@pair` 在
ρ=0.2 有 **41.3%** 的头零配额。于是一个非常危险的简单替代解释浮出来：

> **我们的 +25.80 会不会只是「别把头饿死」？** 若成立，整套配额校准理论就塌成一个
> **已被 Ada-KV 的 safeguard 覆盖**的启发式。

**离线预检（零 GPU，220 chunk）——质量上强烈证伪，但不足以定案：**

| 度量 | 实测 | 随机基准 |
|---|---|---|
| 学到的**正向配额**流向饿死头的占比 | **2.3%** | 41.3% |
| `corr(表 Δb_h, 该头被饿死的频率)` | **+0.0121** | 0 |
| 干预后零配额头 | 41.3% → **23.3%** | — |
| 纯地板 `b_min=1` 需搬动 | **46 槽 = 学到搬动量的 0.1%**，零配额头直接归 0 | — |

学到的干预**主动避开**饿死头（2.3% ≪ 41.3%），相关约为零。**但质量不等于效果** ——
0.1% 的重分配原则上也可能有大效果，所以必须真跑。

**已实现 `VARIKV_QUOTA_MODE=floor`**（进唯一实现 `quota_project.py`）：完全**不用**
学到的方向，只强制 `b_g ≥ b_min`，缺口按 `(b⁰−b_min)⁺` 比例从富余头等量扣回，总预算
不变。新增单测 **T8**（`b ≥ min(b_min, n)` 且预算守恒），8 条全过后才入队。
已排 `b_min ∈ {8, 32, 128, 512}`，n=100。

**预注册判据**：若某个 `b_min` 拿到接近 +25.80，则我们的方法就是防饿死启发式，
**整个「配额方向重要」的叙事作废**；若地板只拿到几分而学到的表拿 +25.80，则
**「方向本身重要」得到一个强对照**。这是本项目少见的「怎么样都有结论」的实验，
而且它比跨方法移植更该先做 —— 它直接回答「Retr.KV@0.2 为什么涨这么多」。

**(c) 第三条修正也接受：平均配额表不足以判定 ranking vs allocation。**
他的论证干净：对同一 chunk 位置，不同文档的 Expected 配额不同，平均后再喂给
文档 1，结果差**分不清**是「FastKVzip 排序不好」还是「没给它文档 1 真正的配额」。
⇒ 必须补**逐样本逐 chunk 的精确移植**：同一样本先跑 Expected 存下精确
`b^Exp_{sample,chunk,l,h}`，再用 FastKVzip 的排序按该配额重放。那样
**配额完全相同、唯一变量是排序**。这需要注入支持「绝对配额」而非「增量表」，
以及跨样本的索引 —— 待实现。**平均表版本仍有价值，但它回答的是另一个问题
（这种配额策略能否作为可部署的通用策略），不能混为一谈。**

**他一处措辞我采纳但要更准**：`adakv-layer` 的 84 是**设计**自由度；实测层总量跨层
相对 std 0.0180 来自**分位阈值 + 严格大于**下的平局。我此前写「可达空间略大于 84」
不准确 —— 正确说法是「84 是设计维度，实现上保留数有离散抖动，该抖动是否可被校准器
系统性利用**未测**」。

### ㉖ 反向迁移：**迁移是强不对称的**，「方向不重要」被推翻

`_p02xferps` / `_p02xferpsx` 落地（n=100，同批基线）。Retr.KV @ρ=0.2，基线 45.20 /
full 68.20 / headroom +23.00：

| 臂 | 表 \|Δb\| | Δ | 95% CI | 恢复率 |
|---|---|---|---|---|
| **自表（Retr.KV 原生）** | 906 | **+25.80** | [+20.60,+31.00]★ | **112%** |
| 迁移 PrefSuf→KV，按预算比 ×1.177 | 420 | +2.40 | [−0.60,+5.60] | 10% |
| **迁移 PrefSuf→KV，幅度匹配 ×2.538** | **906** | **+2.00** | [−1.80,+5.60] | **9%** |

    幅度匹配迁移 − 自表 = −23.80 [−28.80,−18.60]★
    按预算比迁移 − 自表 = −23.40 [−28.80,−18.20]★
    两种迁移之间          = −0.40 [−3.60,+2.80]  不可分（⇒ 这里幅度也不是瓶颈）

**幅度已精确匹配到 906，仍然只拿到 9%。⇒ 方向绝对重要。**

**这修正了 §⑰ 的措辞。** 那里由正向一格（KV表→PrefSuf 与幅度匹配自表不可分）得出
「方向可迁移不成立、优势完全是幅度」。**加上反向后，正确的命题是：**

> **迁移是强不对称的。** KV 表 → PrefSuf 与 PrefSuf 自表不可分（−0.80，n=100）；
> PrefSuf 表 → KV 只有自表的 9%（−23.80★，幅度已匹配）。
> ⇒ **「配额方向无关紧要」被反向这一格直接证伪**；§⑰ 那句只在正向那一格成立，
> 不能推广。

**由此立刻暴露正向那一格缺一个对照。** Retr.KV 上做过等幅度分层置换（ρ=0.2 上
perm = −34.00★，与 own 差 +59.80★），**PrefSuf 上从没做过**。所以「KV 表在 PrefSuf
上和自表一样好」有两种解释：

    (i) KV 表确实携带了对 PrefSuf 有效的方向；
    (ii) **PrefSuf 的「好分配」盆地很宽**，任何等幅度的合理扰动都能拿 ~+11。

**已入队两个置换对照**（幅度与 PrefSuf 自表 ×2.157 **完全相同**，Σ=0 保持）：

    _ps02pg  全局置换   与原表 corr = −0.0267   （打乱「哪个头拿多少」）
    _ps02pl  层内置换   与原表 corr = +0.5611   （保持每层总量，只乱层内 4 头）

**预注册判据**：若置换也拿到 ~+11 ⇒ 解释 (ii)，PrefSuf 盆地宽，「KV 表可迁移」这句
话没有内容；若置换塌到 ~0 或为负 ⇒ 解释 (i)，方向真的迁移过去了。层内置换介于两者
之间，还能顺带告诉我们 PrefSuf 上的收益是靠跨层结构还是层内结构。

**当前对「方向 vs 幅度」的完整图景（三处证据，不要只引其中一处）：**

| 证据 | 说明什么 |
|---|---|
| Retr.KV：等幅度分层置换 −34.00★ vs own +25.80★ | 方向重要（同 panel 内） |
| Retr.KV：PrefSuf 表幅度匹配只拿 9%（−23.80★） | 方向重要（跨 panel，**新**） |
| PrefSuf：自表 ×2.157 比 ×1 多 +4.80★ | 幅度也重要（同 panel 内） |
| PrefSuf：KV 迁移表 ≈ 幅度匹配自表（−0.80） | **待置换对照裁决**，目前无法解释 |

⇒ **方向与幅度都重要，且方向的证据更硬（两处 ★，其中一处跨 panel）。**

### ㉗ 四个 n=100 落地：静态表全 ratio 曲线，以及一条意外的强结论——**跨层-only 是有害的**

**① 静态表的完整 ratio 曲线（Retr.KV，full = 68.20，全部 n=100）**

| ρ | 基线 | headroom | 静态表 Δ | 95% CI | 恢复率 |
|---|---|---|---|---|---|
| 0.5 | 71.60 | **−3.40** | **−4.80** | [−8.60,−1.20]★ | 141% |
| 0.3 | 65.40 | +2.80 | −0.60 | [−5.20,+3.60] | −21% |
| **0.2** | 45.20 | **+23.00** | **+25.80** | [+20.60,+31.00]★ | **112%** |
| 0.1 | 32.60 | +35.60 | +4.20 | [+1.99,+6.60]★ | 12% |

`sign(Δ) = sign(headroom)` 在 4 格里对 **3 格**，唯一例外（ρ=0.3）的 |headroom| 只有
2.8 分、且 Δ 不显著。ρ=0.5 上 headroom 为负（压缩本就赢满缓存），静态表**按预期掉分**
且显著 —— 这是「方法把分数推向满缓存」在**静态表**上的独立复现（此前只在网络上验过）。

**② ρ=0.2 的机制拆解（全部 n=100，同批基线 45.20，headroom +23.00）**

| 臂 | Δ | 95% CI | 恢复率 | 对扁平表配对 |
|---|---|---|---|---|
| 扁平静态表（参照） | +25.80 | [+20.60,+31.00]★ | 112% | — |
| 位置索引表 [11,112] | +23.40 | [+18.80,+28.20]★ | 102% | −2.40 [−5.20,+0.20] 不可分 |
| **跨层-only** | **−8.40** | **[−14.80,−2.20]★** | **−37%** | **−34.20 [−39.20,−29.20]★** |
| 网络 `scalar` s0 | +21.60 | [+17.20,+26.20]★ | 94% | −4.20 [−7.00,−1.20]★ |

**(a) 最强的一条：跨层-only 主动有害（−8.40★）。**
只改各层总预算、层内保持基线比例 ⇒ **掉 8.40 分**，而完整表涨 25.80。
⇒ **「把预算挪到对的层」不是增益来源，反而是负贡献。** 由于完整表**包含**这个分量，
要么层内-only 单独强于 +25.80，要么两者有强交互。`_p02win3` 在跑，会裁决。

**这同时把 §⑨/§㉕ 的教训钉死**：**搬动量不是收益** —— 跨层是搬动量最大的通道
（此前测到它在总搬动量里占大头），却是唯一单独为负的通道。**任何用搬动量代理收益的
推理都不成立**，我此前那条已撤回的「62% 来自跨层」如果当时没撤，现在会被这个结果
正面打脸。

**口径**：`_p02acr2` 用的是**首版**投影。当时的离线审计显示它的 across 语义是干净的
（层总量与理论层净变化 corr = +1.0000，全局补偿只动 ≤2 槽/层、占预算 0.0013%），
所以这个 −8.40★ 可信；重构版 `_p02acr3` 在跑，会复核。

**(b) 位置索引表：实验无分辨力，不得当作预测被证实。**
预注册预测是「位置表 ≤ 扁平表」。实测点估计 −2.40，方向一致，**但**

    位置表 − 扁平表  = −2.40 [−5.20,+0.20]   不可分
    位置表 − 网络    = +1.80 [−0.40,+4.00]   不可分

**两边都不可分** ⇒ 三个臂（网络 +21.60、位置 +23.40、扁平 +25.80）挤在 4.2 分区间内，
而单臂 CI 半宽就有 4–5 分。**这个实验分辨不出任何东西，结论是「未定」而不是
「预测成立」。** 要判定需要把 CI 缩到 ±1 分量级 —— 按当前噪声约需 n≈1000，
或者改用同一批样本的配对设计并增加种子。**先记为未决，不要写进结论。**

**(c) 网络 vs 扁平表 −4.20★** 与 §⑬ 的 +4.93±0.57（s0 那格正是 +4.20）逐项一致，
再次确认「丢掉网络只留 112 个数」在这个工作点上稳定多拿约 4–5 分。

### ㉘ 跨方法**精确**配额移植：实现完成、对齐已验，链式作业已挂

外部复核指出「平均配额表分不清『接收方排序不好』与『没给它这个文档真正的配额』」，
这条对，且平均表版**回答的是另一个问题**（可部署性）。本条实现精确版。

**新增 `VARIKV_QUOTA_ABS`（逐样本逐 chunk 的绝对配额注入）** 与
`scratch_build_absquota.py`（dump jsonl → npz）。

设计要点，以及每一条为什么必须这么做：

1. **绝对配额而非增量表**。捐赠方与接收方用**逐位相同**的配额向量 ⇒ 唯一变量是排序。
   预算自然相等，**无需减均值**（那是平均表版为了对齐压缩率才要做的）。
2. **对齐是这条路径最危险的失败模式**：cache 每样本重建，样本号只能靠模块级计数。
   错位不会报错，只会让实验拿**别的样本**的配额跑完、看起来一切正常。
   ⇒ npz 里同存 `lo`/`hi`，注入端**每个 chunk 都断言匹配**，宁可崩不要静默错位。
   离线复放 22 个 chunk：对齐失败 0；**阴性对照**（人为把 `lo[0,0]` 改 1）断言能抓到。
3. **捐赠方的配额必须是干净的 —— 这里差点埋一个坑。** dump 需要
   `LearnedControlRetainCache`（带 dump 能力），但那要求加载 ckpt，而学习臂一旦生效，
   它改变的保留集会**污染后续 chunk 的 `score0`**，而 `score0` 正是 dump 要采的量。
   本想用 `--ctrlm_alpha 0`，查代码发现它走 logit 路径
   （`_p = min(max(0/alpha_max, 1e-6), …)`）⇒ **alpha ≈ 1e-6·alpha_max，不是精确 0**，
   `active` 仍为真、Δs 仍非零。⇒ 新增 **`VARIKV_CTRL_OFF=1`** 硬关学习臂，
   `score` 保持 `score0` 不变，该次运行与原生基线**逐位相同**但保留 dump 能力。

**链式作业已挂**（自动抢空卡，两阶段）：

    阶段1  expect@adakv-layer, VARIKV_CTRL_OFF=1, n=100, 导出配额
           → 同一次运行**既是干净的 Expected Attention 基线，也是配额来源**
    建 npz  scratch_build_absquota.py
    阶段2  fastkvzip@pair, VARIKV_QUOTA_ABS=<npz>, n=100
           → **FastKVzip 排序 + Expected 逐位相同的配额**

**预注册判据**：阶段2 ≈ 阶段1 ⇒ 两方法的性能差异主要在**配额**而非排序；阶段2 远低于
阶段1 ⇒ Expected 自己的**排序**也重要。**任一方向都是结论。**

**必须随结果一起报的协变量**：`fastkvzip@pair` 在 ρ=0.2 有 **41.3%** 的头零配额，
而 `expect@adakv-layer` 是 **0.0%**（那是 `safeguard=0.2` 造成的，见 §㉕）。所以
阶段2 若大涨，**不能直接归给「Expected 的分配形状更好」** —— 它同时解除了饿死。
排队中的 `_flr*` 地板对照正是为了把这两者分开。

### ㉙ 幅度扫描第一段：峰是**宽平台**不是尖峰；rc=127 的 provenance 债已清

**① PrefSuf @ρ=0.2 幅度扫描**（基线 39.20 / full 50.00 / headroom +10.80，全部 n=100）

| γ | 表 \|Δb\| | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|---|
| 1.000 | 357 | +7.20 | [+4.60,+10.00]★ | 46.40 | 67% |
| **2.157** | 770 | **+12.00** | [+8.60,+15.60]★ | **51.20** | **111%** |
| 3.000 | 1071 | +10.20 | [+6.60,+13.80]★ | 49.40 | 94% |

    γ=2.157 − γ=1      = +4.80 [+2.40,+7.20]★     上升段**是实的**
    γ=3.0   − γ=2.157  = −1.80 [−3.80,+0.20]      **不可分**

⇒ **最优是 γ ∈ [2,3] 的一个宽平台，不是尖峰。** 只能说「从 γ=1 放大到 ~2 显著更好」，
**不能说峰在 2.157**——那两点分不开。γ=4.5 在跑，会告诉我们平台右端在哪。

**对方法是好消息**：γ 取 2 还是 3 差别不显著 ⇒ **对幅度不敏感**，不需要精调。
若最终方法要学一个 γ，这意味着容错区间很宽。

**措辞纪律**：此前记的「两个 panel 最优点都落在恢复率 ≈111–112%」现在有第三个点
（γ=3.0 → 94%），且它与 111% 那点不可分 ⇒ **那条 n=2 的观察仍不成立为结论**，
平台内任何一点都符合数据。

**② `_ps75b` 干净重跑：与 `_ps75tab` 逐样本完全相同（100/100），双双 +10.80★。**

这一条同时确认三件事：**(a)** 评测是确定性的（同配置两次运行逐样本同分）；
**(b)** 那个 rc=127 作业（我编辑运行中的 `/tmp/qrun.sh` 导致 bash 按旧字节偏移读到
半行）**数据完好**，此前的判断正确；**(c)** provenance 债已清，现在有 rc=0 的产物。

⇒ 其余四个同因 rc=127 的作业（`_p02s1`/`_p02s2`/`_p02m3`/`_p03tab`）**不必重跑**：
同一失败机制、同样 100/100 结果目录完整，且本条已给出「崩在收尾不影响结果」的
直接证据。若审稿需要干净退出码再补。

### ㉚ 幅度曲线补完：**倒 U 成立，升降两侧都显著**，内点最优已确立

PrefSuf @ρ=0.2（基线 39.20 / full 50.00 / headroom +10.80，全部 n=100，同批基线）：

| γ | 表 \|Δb\| | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|---|
| 1.000 | 357 | +7.20 | [+4.60,+10.00]★ | 46.40 | 67% |
| **2.157** | 770 | **+12.00** | [+8.60,+15.60]★ | **51.20** | **111%** |
| 3.000 | 1071 | +10.20 | [+6.60,+13.80]★ | 49.40 | 94% |
| 4.500 | 1607 | +6.40 | [+3.00,+9.80]★ | 45.60 | 59% |

**相邻配对：**

    γ=2.157 − γ=1.0   = +4.80 [+2.40,+7.20]★     上升段**显著**
    γ=3.0   − γ=2.157 = −1.80 [−3.80,+0.20]      不可分（平台）
    γ=4.5   − γ=3.0   = −3.80 [−6.40,−1.20]★     下降段**显著**
    γ=4.5   − γ=1.0   = −0.80 [−3.60,+2.00]      不可分

**⇒ 内点最优的存在性已确立**（升、降两侧各有一个 ★），这比上一条「宽平台、峰未定」
强一档。峰的**位置**仍未分辨（2.157 与 3.0 不可分），能说的是**最优落在 γ ∈ [2,3]**。

三条可写进论文的读数：

1. **网络学到的幅度是次优的，而且差得不小** —— γ=1（网络自己的输出）只拿 67% 的
   headroom，调到 γ≈2–3 拿 94–111%。**一个一维标量**值 +4.80★。
2. **容错带很宽**：γ 在 [2,3] 内不可分 ⇒ 若最终方法要学 γ，不需要精调。
3. **过冲会伤，但伤得温和**：γ=4.5（4.5 倍！）仍是 +6.40★，与 γ=1 不可分。
   ⇒ **不是刀锋**，这对可部署性是好消息；但也说明 γ 不能不管——从 3 到 4.5 掉 3.80★。

**与「静态表 vs 网络」的关系**：Retr.KV 上静态表赢同源网络 +4.93±0.57，此前归因于
去掉有害的 chunk 位置波动。**逐 chunk 平均不改变幅度**，所以那条不能用幅度解释；
但 Retr.KV 自己的幅度曲线（`_p02x05/x15/x25` 在跑，进度 26–41）会告诉我们
×1.0 是否已在峰上 —— 若不在，则「静态表更好」里也混着幅度成分。

**措辞纪律再收紧一次**：此前那条「两个 panel 最优点都落在恢复率 ≈111–112%」现在有
四个点（67/111/94/59%），**平台内两点不可分** ⇒ 该观察**仍不成立为结论**，只能说
「PrefSuf 的最优恢复率在 94–111% 之间，即略微过冲满缓存」。

### ㉛ 层内/跨层裁决：**两个分量单独都不行，增益全在交互里**（并因此下调 adakv-layer 前景）

Retr.KV @ρ=0.2（基线 45.20 / full 68.20 / headroom +23.00，全部 n=100，同批基线）。
111 维配额空间 = 84 维层内 + 27 维跨层。

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| **完整表（111 维）** | **+25.80** | [+20.60,+31.00]★ | 71.00 | **112%** |
| 层内-only（84 维，修正版） | **+2.20** | [−2.00,+6.40] | 47.40 | 10% |
| 跨层-only（27 维） | **−8.40** | [−14.80,−2.20]★ | 36.80 | −37% |

    层内 − 完整 = −23.60 [−29.20,−18.00]★
    跨层 − 完整 = −34.20 [−39.20,−29.20]★
    层内 − 跨层 = +10.60 [+4.20,+17.00]★
    层内(修正版) − 层内(旧投影) = +0.24 [−1.18,+1.65] 不可分（n=85）

**核心结论：`+2.20 + (−8.40) = −6.20` vs 完整表 `+25.80` —— 差 32.00 分。
增益完全来自两个通道的交互，任何一个边际单独锁死都会摧毁它。**

**机制上说得通**：正确的分配可能是「第 L 层要更多总量，**并且**层内要给到具体某个头」。
只改层总量（跨层-only）会把多出来的预算**按基线比例**摊给该层各头 —— 送错了头；
只改层内（层内-only）能挪到对的头，但该层总量不够、无处可挪。**两件事必须同时做。**

**三条后果，其中一条要下调我此前的判断：**

1. **不能把增益拆成「先修层、再修头」两步来讲。** 边际分解在这里是失效的；
   111 维必须**联合**优化。注意这**不**否定分层参数化本身 —— 一个先定层总量、
   再在层内**自由**分配的分层 allocator 仍有全部 111 自由度；被否定的是
   「任一边际单独承载效应」。
2. **⚠ 下调 Expected Attention 路线的前景。** `adakv-layer` 暴露的恰好就是这 84 维
   层内子空间（外加 safeguard）。层内-only 只拿 **+2.20（不显著）** ⇒ **我们的方法
   在 `adakv-layer` 下大概率只有很小的收益。** 这**修正**了 §⑨ 由「`adakv-layer`
   实现了 `pair` 38% 的搬动量」得出的乐观判断 —— **又一次「搬动量不是收益」**：
   38% 的搬动量只换来 10% 的恢复率，而且不显著。
   口径改为：**`pair` 是本方法唯一被验证有效的竞争拓扑；`adakv-layer` 上预期收益很小
   （+2.20 [−2.00,+6.40]）；`pair-head` 可证为零。**
3. **旧投影版可用性得到确认**：修正版与旧版层内-only 配对 +0.24 不可分，
   证实离线测到的 1.21% 搬动量差异确实无关紧要。⇒ `_p02acr2`（旧版跨层）的
   −8.40★ 同样可信，`_p02acr3` 的复核不是必需的（仍在跑，跑完顺带确认）。

**诚实的边界**：层内-only 的 +2.20 与 0 不可分，但 CI 上界 +6.40（28% 恢复率）
**没有**排除一个中等正效应。所以准确说法是「n=100 下层内-only 未测到效应，
且可排除大于 +6.4 的效应」，不是「层内完全无用」。

### ㉜ 为什么交互这么强：Retr.KV@0.2 的「好分配」盆地**极窄**（cos 0.70 买不到东西）

§㉛ 测到两个边际单独都失效、增益全在交互里。这一条量化"差多远才算差"——零 GPU，
用 `project_quota` 在真实 `b0` 上算各干预的**实现** Δb，再与完整表的实现 Δb 求余弦。

| 臂 | 搬动量 | **cos(实现Δb, full)** | 实测下游 Δ | 恢复率 |
|---|---|---|---|---|
| full（完整表） | 50,539 | +1.0000 | +25.80★ | 112% |
| 层内-only | 38,778 | **+0.6992** | +2.20 | 10% |
| 跨层-only | 45,948 | **+0.6865** | −8.40★ | −37% |
| **PrefSuf 表迁移（幅度匹配）** | 48,424 | **+0.6602** | +2.00 | **9%** |
| 等幅度分层置换 | 30,083 | +0.1461 | −34.00★ | −148% |

**⇒ cos ≈ 0.66–0.70 的方向，恢复率一律塌到 9–10%（或为负）；只有 cos = 1.0 拿到 112%。**

**为什么这条可信（关键在第三行）**：层内/跨层的 cos ≈ 0.69 **有构造性成分** ——
它们是完整表在互补子空间上的投影，能量占比 66.3% / 33.7% 对应
`√0.663 = 0.814`、`√0.337 = 0.581`，离散化后实测 0.699 / 0.687。**但 PrefSuf 迁移表
不是投影**，它是另一个 panel 独立学出来的表，**碰巧也落在 cos 0.660，也碰巧只拿 9%**。
三个机制完全不同的扰动给出同一个「cos 0.66–0.70 → ~10%」，这才让"窄盆地"读法立得住。

**这把四条此前分散的证据统一成一句话**：

> **Retr.KV @ρ=0.2 上，任务效用不是分配方向的平滑函数，而是尖峰的。**
> 方向要几乎完全对；差到 cos 0.7 就等于没做。

**边界（必须同报）**：这是**单 panel、单 ratio** 的性质。PrefSuf 上盆地可能宽得多 ——
证据是 **KV 表（与 PrefSuf 自表 cos 仅 +0.635）在 PrefSuf 上与自表不可分**，这在窄盆地
模型下**不该发生**。⇒ **两个 panel 的盆地宽度不同**，这本身是个待验命题。

**由此给排队中的 PrefSuf 置换对照一个更锐的预注册预测**：

    若 PrefSuf 盆地与 Retr.KV 一样窄 ⇒ `_ps02pg`（cos ≈ −0.03）应塌到 ~0 或负，
                                      `_ps02pl`（cos ≈ +0.56）应只剩个位数
    若 PrefSuf 盆地确实更宽       ⇒ 两个置换都能拿到可观的分数

**任一结果都直接判定"KV 表能迁到 PrefSuf"是有内容还是空话**，因为窄盆地下 cos 0.635
的外来表本不该管用。

### ㉝ 跨层-only 的重构版复核通过；投影版本问题就此关闭

`_p02acr3`（重构版投影，n=100）落地，与首版 `_p02acr2` 对照：

| 臂 | 投影版本 | Δ | 95% CI | 恢复率 |
|---|---|---|---|---|
| 完整表（111 维） | 重构 | +25.80 | [+20.60,+31.00]★ | 112% |
| 跨层-only | 首版 | −8.40 | [−14.80,−2.20]★ | −37% |
| **跨层-only** | **重构** | **−7.80** | **[−14.00,−1.80]★** | −34% |
| 层内-only | 修正 | +2.20 | [−2.00,+6.40] | 10% |

    跨层-only[重构] − [首版] = +0.60 [−1.60,+2.80]  不可分（逐样本相同 78/100）

**投影版本问题就此关闭**：两个消融的首版与修正/重构版都不可分
（层内 +0.24 [−1.18,+1.65]，跨层 +0.60 [−1.60,+2.80]）。⇒ 三次投影修改都是
**定义清洁度**的改进，**没有一次改变过任何结论**。这也印证了当时的离线量化：
within 泄漏 1.21% 搬动量、across 层总量偏离 ≤2 槽（占预算 0.0013%），确实无关紧要。

**注意 78/100 逐样本相同**：两版投影在 22 个样本上确实产生了不同的保留集，
但聚合后不可分。⇒ **「逐样本不同」与「结论不同」是两回事**，判定要看配对区间，
不能看逐样本一致率。

**最终的层内/跨层分解表（全部重构版口径，n=100）**：

    完整表 111 维   +25.80★   112%
    层内-only 84 维  +2.20     10%   （不显著）
    跨层-only 27 维  −7.80★   −34%
    两者相加        −5.60  vs 完整 +25.80  ⇒ 差 31.40，增益全在交互

### ㉞ 移植实验的判据错了，且是查参照值时发现的；改成 2×2 并给出正确判据

**起因是一个通用读取器**。此前所有读数片段都硬编码 `*_fastkvzip{tag}` 与
`output-pair.json`，而移植要比的两个臂**目录前缀与输出文件名都不同**：

    捐赠方  <i>_<model>_expect__expbase_.../output-adakv-layer.json
    接收方  <i>_<model>_fastkvzip__xpFKVqExp_.../output-pair.json

硬编码版会**静默返回空再打印 0.00** —— 本项目已被这一类静默零坑过
（`VARIKV_RATIOS` 未导出给 parse 时同样打印 0.00）。新增
`scratch_read_scores.py`：按 `*_<model>_*<tag>_*/output-*.json` 通配，
**匹配到 0 个目录时抛错而不是返回空**（阴性对照已验）。

**用它一读就发现真正的问题**：`_expbase` 在 ρ=0.2 上 41 个样本**均值 0.00**。
查原始 Figure-11 复现（无 ctrlm，n=100）确认这不是 bug：

| ρ | 0.2 | 0.3 | 0.4 | 0.5 | 0.75 |
|---|---|---|---|---|---|
| **Expected Attention** | **1.40** | 9.80 | 41.00 | 56.00 | 62.40 |
| FastKVzip | 45.20 | 65.40 | 66.40 | 71.60 | 68.80 |

**⇒ 我预注册的判据「阶段2 ≈ 阶段1 ⇒ 差异主要在配额」是错的。** 捐赠方在这个
工作点上**本身就没有性能可移植**（1.40 分），"接近它"毫无信息量。

**正确的判据（重新预注册）**：两个方法在 ρ=0.2 的差距是 **45.20 − 1.40 = 43.80 分**。
移植格的分数**落在这 43.80 分区间里的位置**就是分解：

    cell2 = FKV排序 + Exp配额（`_xpFKVqExp`，在跑）
        ≈ 45.20 ⇒ FKV 的排序对配额**鲁棒**，Expected 的失败在**排序**
        ≈ 1.40  ⇒ **配额**主导，Expected 的失败在配额

    cell3 = Exp排序 + FKV配额（`_xpExpqFKV`，新挂链）
        ≈ 45.20 ⇒ Expected 的排序其实够好，只是配额毁了它 ⇒ **配额主导**
        ≈ 1.40  ⇒ Expected 的排序确实差 ⇒ **排序主导**

**两格互为镜像，一致才可信**；若 cell2 与 cell3 给出相反结论，说明存在
**排序 × 配额交互**（这在本项目已有先例：层内/跨层就是极强交互）。

**已挂第二条链** `/tmp/chain_xplant2.sh`：阶段1 `_fkvbase`（fastkvzip@pair +
`VARIKV_CTRL_OFF=1` + 配额导出，n=100，既是干净 FKV 基线也是配额源）→ 建
`scratch_absq_fkv.npz` → 阶段2 `_xpExpqFKV`（expect@adakv-layer +
`VARIKV_QUOTA_ABS`，即 **Expected 排序 + FastKVzip 配额**）。

**仍必须同报的协变量**：`fastkvzip@pair` 41.3% 饿死头 vs `expect@adakv-layer` 0.0%
（后者源于 `safeguard=0.2`）。cell3 若大涨，其中一部分只是"给 Expected 加了饿死头"，
而 cell2 若大跌，一部分只是"解除了 FKV 的饿死"。**排队中的地板对照正是分开这两者的。**

**一条应当先想到却没想到的教训**：**选捐赠方之前必须先看它在目标工作点的分数。**
我直接按「有现成 gate」挑了 Expected Attention，没查它在 ρ=0.2 上只有 1.40。
若要在**双方都有性能**的地方做移植，正确的工作点是 **ρ=0.4**（Exp 41.00 vs
FKV 66.40，差 25.40）或 **ρ=0.5**（56.00 vs 71.60）。**待 ρ=0.2 的 2×2 出结果后，
若两格互相矛盾或都贴边，就改到 ρ=0.4 重做。**

### ㉟ Retr.KV 的幅度曲线：峰就在 ×1.0，**「网络系统性下手太轻」是错的**

Retr.KV @ρ=0.2（基线 45.20 / full 68.20 / headroom +23.00，全部 n=100）：

| γ | 表 \|Δb\| | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|---|
| 0.5 | 453 | +24.80 | [+20.00,+29.80]★ | 70.00 | 108% |
| **1.0** | 906 | **+25.80** | [+20.60,+31.00]★ | **71.00** | **112%** |
| 1.5 | 1360 | +20.20 | [+14.60,+25.80]★ | 65.40 | 88% |
| 2.5 | 2266 | +7.80 | [+1.60,+14.00]★ | 53.00 | 34% |

    γ=1.0 − γ=0.5 = +1.00 [−2.20,+4.20]    不可分
    γ=1.5 − γ=1.0 = −5.60 [−9.60,−1.80]★
    γ=2.5 − γ=1.5 = −12.40 [−16.20,−8.60]★

**① 预注册问题有答案了：峰就在 ×1.0（平台 [0.5,1.0]），放大显著有害。
⇒ 「静态表赢同源网络 +4.93±0.57」**不**掺幅度成分**，那条替代解释被排除。
（本来也不该掺 —— 逐 chunk 平均不改变幅度 —— 现在有了直接证据。）

**② 撤回「网络系统性地下手太轻」。** 两个 panel 的最优 γ 相反：

| panel | γ=1 的恢复率 | 最优 γ | 调幅值多少 |
|---|---|---|---|
| **Retr.KV** | **112%** | **≈0.5–1.0** | **0（γ=0.5 与 γ=1 不可分）** |
| **PrefSuf** | 67% | **≈2–3** | **+4.80★** |

⇒ **网络的幅度标定是 panel 相关的：Retr.KV 上标定得对，PrefSuf 上偏保守 2.5 倍。**
上一条写的"系统性下手太轻"只在 PrefSuf 上成立，**已撤回**。这也意味着「学一个 γ」
要解决的是**与配额表本身同样的推断问题**（γ = f(workload)），不是一个免费的标量。

**③ 一个把两条结果统一起来的图像：效用地形在方向上窄、在幅度上宽。**

    沿**正确方向**缩放：γ=0.5 仍拿 108%（与 γ=1 不可分）—— 幅度容错
    **偏离方向**：cos 0.66–0.70 一律只剩 9–10% —— 方向不容错

两者对照鲜明：**把干预砍掉一半几乎不损失，把方向转到 cos 0.7 就等于没做。**
（幅度容错只在**下调**方向成立；上调过 1.5 就显著变差。）

**④ 顺带：`_p02win2` 跑满 100，与修正版配对 −0.20 [−1.80,+1.20] 不可分**
（此前 n=85 时是 +0.24）。层内-only 两版分别 +2.40 / +2.20，都不显著。
投影版本问题第三次确认关闭。

### ㊱ **一个平凡的防饿死地板追平了整张学习表** —— 本轮核心叙事必须重估

`_flr8` 落地（n=100）。`floor` 模式**完全不用学到的方向**，只强制 `b_g ≥ 8`，
缺口按比例从富余头扣回、总预算不变：

| 臂 | Δ | 95% CI | 绝对 | 恢复率 | 搬动量 |
|---|---|---|---|---|---|
| 完整表（学到的方向） | +25.80 | [+20.60,+31.00]★ | 71.00 | 112% | 50,539 |
| **地板 `b_min=8`** | **+27.00** | **[+21.60,+32.60]★** | **72.20** | **117%** | **409（0.81%）** |
| 配对 地板 − 完整表 | **+1.20** | [−1.80,+4.20] **不可分** | | | |

**这正是 §㉕ 预注册的失败条件**：「若某个 `b_min` 接近 +25.80 ⇒ 我们只是防饿死启发式，
叙事作废」。它不只是接近，点估计还略高，且**只搬动 0.81% 的预算**，同时把零配额头
从 41.3% 直接清到 **0.0%**。

**⇒ 学到的配额方向**不是必需的**。** 这条必须写在最前面，不能藏在细节里。

### ㊲ 连带推翻 §㉜ 的「方向窄盆地」—— 那是与饿死缓解**混淆**的伪结论

§㉜ 由「cos 0.66–0.70 的三个扰动都只拿 9–10%」推出「效用在方向上是尖峰的」。
把地板加进同一张表，结论立刻崩：

| 臂 | cos(实现Δb, 完整表) | 饿死缓解 | Δ |
|---|---|---|---|
| 完整表 γ=1 | +1.0000 | 27.0% | +25.80 |
| γ=0.5 | +0.9971 | 18.1% | +24.80 |
| γ=2.5 | +0.9474 | **41.3%** | **+7.80** |
| 层内-only | +0.6992 | 3.9% | +2.20 |
| 跨层-only | +0.6865 | 11.0% | −7.80 |
| PrefSuf迁移 | +0.6602 | −5.2% | +2.00 |
| 等幅度置换 | +0.1461 | −1.0% | −34.00 |
| **地板 `b_min=8`** | **+0.3069** | **41.3%** | **+27.00** |

    Spearman(cos, Δ)        = +0.476
    Spearman(饿死缓解, Δ)   = **+0.778**
    Spearman(cos, 饿死缓解) = +0.407   ← 两个预测量本身就相关，所以此前无法区分

**cos 0.31 拿 +27.00，而 cos 0.66–0.70 拿 +2 到 −8** ⇒ **余弦不是决定量**。
§㉜ 的那四个"失败方向"恰好也是"饿死缓解低"的四个（3.9% / 11.0% / −5.2% / −1.0%），
**两个解释在那批数据里完全共线**，我当时只检验了其中一个。**§㉜ 撤回。**

**但饿死缓解也不是充分解释**：γ=2.5 把饿死完全清零（41.3%）却只拿 **+7.80**，
γ=0.5 只缓解 18.1% 却拿 **+24.80**。⇒ **过度搬动会独立地伤害**。

**当前最站得住的工作假说**（尚需检验，别当结论）：

> 结果 ≈ f(**消除了多少饿死**, **额外搬动了多少**)　—— 把头喂活，然后**尽量少动别的**。

这与 Ada-KV 的 `safeguard=0.2`（逐头保护 top 20%）是同一个直觉，**所以新颖性风险陡增**。

**已排三个作业检验它的普适性**：`_psflr8` / `_psflr32`（PrefSuf@0.2 —— 那里 γ=1 只恢复
67%，若地板也能追平自表 ×2.157 的 +12.00，则该 panel 也塌）、`_vtflr8`（MultiHop@0.2 ——
学习表在那里 −7.11，**地板若也伤，说明伤害同样与饿死无关；地板若不伤，则学习表的伤害
是它自己的方向造成的**，这反过来是"方向有内容"的证据）。

**方法论教训**：§㉜ 是"用一个预测量解释一组数据，却没检验共线的第二个预测量"。
本项目已有先例（`m_evict` 与 slack 在 panel 内共线）。**规则升级：提出任何"X 解释了
这批结果"之前，必须先列出与 X 共线的候选量并逐个排除。**

### ㊳ 地板结果**复现了**：两个不同的平凡地板都追平学习表

`_flr32` 落地（n=100）。Retr.KV @ρ=0.2：

| 臂 | Δ | 95% CI | 绝对 | 恢复率 | 搬动量 | 干预后零配额头 |
|---|---|---|---|---|---|---|
| 完整学习表 | +25.80 | [+20.60,+31.00]★ | 71.00 | 112% | 50,539 | 14.3% |
| 地板 `b_min=8` | **+27.00** | [+21.60,+32.60]★ | 72.20 | 117% | **406（0.80%）** | **0.0%** |
| 地板 `b_min=32` | **+25.80** | [+20.40,+31.20]★ | 71.00 | 112% | **1,742（3.4%）** | **0.0%** |

    地板8  − 完整表 = +1.20 [−1.80,+4.20]  不可分
    地板32 − 完整表 = +0.00 [−2.80,+3.00]  不可分
    地板32 − 地板8  = −1.20 [−4.20,+1.80]  不可分

**⇒ 不是单点巧合。两个 b_min 相差 4 倍、搬动量相差 4.3 倍的平凡地板，都与学习表
不可分**（b_min=32 的点估计恰好也是 +25.80）。**「学到的配额方向在 Retr.KV@0.2 上
不带来任何超出『别把头饿死』的收益」现在是复现过的负面结果。**

**顺带支持工作假说的一条**：8 → 32 多搬了 1,336 槽（+329%），而两者的零配额头都已是
0.0%，分数变化 −1.20（不可分）。⇒ **饿死一旦消除，额外搬动就不再带来收益**
（也还没造成可测的伤害；`_flr128`（7,466 槽）与 `_flr512` 会看到伤害是否出现）。

**当前对 Retr.KV@0.2 最简洁的描述**：

> 基线 FastKVzip 在 ρ=0.2 让 **41.3% 的 (层,头) 拿到零配额**。把它们各喂 8–32 个槽位
> （占预算 0.8–3.4%），就能拿到 **+25.8 ~ +27.0 分**，即 headroom 的 112–117%。
> 4,482 参数的学习打分器、112 维配额表、以及本轮所有方向性分析，**都没有超过这个
> 三行启发式**。

### ㊴ PrefSuf 置换对照：盆地**不宽**，迁移有内容 —— 与地板结果并不矛盾

`_ps02pg` 落地（n=100）。PrefSuf @ρ=0.2（基线 39.20 / full 50.00 / headroom +10.80）：

| 臂 | 与自表 cos | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|---|
| 自表 ×2.157（参照） | +1.000 | +12.00 | [+8.60,+15.60]★ | 51.20 | 111% |
| **全局置换（幅度完全相同）** | **−0.027** | **−18.60** | [−22.60,−14.60]★ | 20.60 | **−172%** |
| 迁移 KV→PrefSuf | +0.635 | +11.20 | [+8.00,+14.60]★ | 50.40 | 104% |

    置换 − 自表 = −30.60 [−34.80,−26.40]★
    迁移 − 置换 = +29.80 [+25.80,+34.00]★

**按预注册判据**：置换塌到 −18.60（远低于 ~+11）⇒ **PrefSuf 盆地也窄**，
⇒ **「KV 表可迁到 PrefSuf」是有内容的** —— 一张 cos 只有 0.635 的外来表恢复 104%，
而同幅度的随机置换掉 18.60 分。

### ㊵ 把地板与置换两条结果放在一起：**方向不是任意的，但也不唯一**

表面上矛盾：地板（cos 0.31）在 Retr.KV 上拿 +27.00，而置换（cos −0.03）在 PrefSuf 上
拿 −18.60；Retr.KV 的置换（cos 0.15）也拿 −34.00。**正确的调和是区分"随机低余弦"
与"特定低余弦"**：

1. **破坏结构的随机扰动在两个 panel 上都是灾难**
   （Retr.KV 置换 −34.00★、PrefSuf 置换 −18.60★，两者幅度都与自表严格相同）。
   ⇒ **配额方向不是任意的。**
2. **但好方向不止一个。** 防饿死地板是一个**特定的**低余弦方向（cos 0.31），
   在 Retr.KV 上与学习表不可分（+27.00 vs +25.80）。
   ⇒ **「学到的方向是唯一好方向」为假。**
3. 所以 §㉜「窄盆地」的撤回仍然成立（**余弦不是决定量**），但不能反推成
   「方向无所谓」—— 置换的两个 ★ 直接否定后者。

**当前最准确的表述**：

> 配额空间里存在**多个**互相远离的好方向（学习表、防饿死地板），
> 而**绝大多数**方向（随机置换、边际投影、外来表迁到 Retr.KV）是坏的。
> 余弦相似度**无法**把好坏分开 —— 这正是四个无标签门控全败的同一个困难在另一处显形。

### ㊶ 因此现在**唯一**决定性的待测项：地板在 PrefSuf 上是否也追平学习表

    若 `_psflr8`/`_psflr32b` ≈ +12.00（自表 ×2.157）
        ⇒ 「学到的方向不带来超出防饿死的收益」**在两个 panel 上都成立**，
          本轮的方法命题基本作废，只剩「防饿死地板」这个已被 Ada-KV safeguard 覆盖的启发式。
    若地板在 PrefSuf 上明显低于 +12.00
        ⇒ PrefSuf 正是**学到的方向真正挣到钱**的地方，
          而 Retr.KV@0.2 的 +25.80 恰好可以被一个更简单的干预替代。
          那样命题要改写成「方向在某些 workload 上必需」，并且必须解释为什么 Retr.KV 不是。

`_psflr8` 已 63/100，`_psflr32b`（修好不可行地板后重排）在队列。

### ㊷ 一个断言按设计工作了，以及它暴露的边界

`_psflr32` 以 **rc=1** 失败：`地板 32 × 112 = 3584 > 总预算 3514`。那是 PrefSuf 某个
文档的**末尾短 chunk** —— 「每头至少 32」在那里数学上无解。**断言崩掉好过静默做错事**，
这是它存在的意义。

**修法不是放宽断言**，而是走到约束边界：`b_min ← min(b_min, Btot // (L·H))`，
即地板不可行时**饱和到均匀分配**（该约束的连续极限）。新增单测 **T9**（极小总预算
3514 / 112 / 5 下，`b_min ∈ {32,128,4096}` 都必须不崩且满足全部不变量），**9 条全过**。

`scbench_kv` 没碰到这个边界（Σb_base ≈ 302,711，末 chunk 也够大）；`_psflr8`
（8×112=896 < 3514）同样安全，所以只有 `b_min=32` 在 PrefSuf 上触发。

### ㊸ 地板 b_min 曲线**极其平坦**；PrefSuf 层内置换同样是灾难但有梯度

**① 地板 b_min 曲线（Retr.KV @ρ=0.2，全部 n=100）**

| 臂 | Δ | 95% CI | 恢复率 | 搬动量 |
|---|---|---|---|---|
| 完整学习表 | +25.80 | [+20.60,+31.00]★ | 112% | 50,539 |
| 地板 `b_min=8` | +27.00 | [+21.60,+32.60]★ | 117% | 406 |
| 地板 `b_min=32` | +25.80 | [+20.40,+31.20]★ | 112% | 1,742 |
| 地板 `b_min=128` | +23.40 | [+17.60,+29.00]★ | 102% | 7,430 |

    地板32  − 地板8   = −1.20 [−4.20,+1.80]  不可分
    地板128 − 地板32  = −2.40 [−5.80,+0.80]  不可分
    地板128 − 完整表   = −2.40 [−6.01,+1.20]  不可分

**`b_min` 跨 16 倍、搬动量跨 18 倍，三个点两两不可分，也都与完整表不可分。**
点估计单调下降（117% → 112% → 102%），方向与「过度搬动伤害」一致，**但没有一对
达到显著**。⇒ 只能说「在 8–128 这个范围内，地板对 `b_min` 极不敏感」，
**不能说已经看到伤害**。`_flr512`（搬动量约 3 万）会看它在哪里真的塌。

**这条进一步加重了负面结论**：不是"某个精调的地板恰好追平"，而是**整段 b_min
都追平**。学习表在这个工作点上没有任何可测的优势。

**② PrefSuf @ρ=0.2 的两个置换（幅度均与自表 ×2.157 严格相同）**

| 臂 | 与自表 cos | Δ | 95% CI | 恢复率 |
|---|---|---|---|---|
| 自表 ×2.157 | +1.000 | +12.00 | [+8.60,+15.60]★ | 111% |
| **层内置换（保每层总量）** | **+0.561** | **−9.40** | [−13.00,−6.00]★ | −87% |
| 全局置换 | −0.027 | −18.60 | [−22.60,−14.60]★ | −172% |

    层内置换 − 自表      = −21.40 [−25.40,−17.40]★
    层内置换 − 全局置换  = **+9.20 [+5.80,+12.60]★**

**两条读数：**

1. **cos +0.561 仍然是灾难（−9.40★）。** 与地板在 Retr.KV 上 cos **+0.31** 却拿
   **+27.00** 并列看 ⇒ **余弦与结果之间没有单调关系**，再次确认 §㉜ 的撤回是对的，
   而且这次是在**另一个 panel** 上独立确认。
2. **但存在清晰的梯度**：全局置换 −18.60 → 层内置换 −9.40 → 自表 +12.00。
   **保住每层总量值 +9.20★**，占「置换到自表」总落差 30.60 的 30%。
   ⇒ **层总量携带真实信息**。这与 Retr.KV 上「层内-only +2.20 / 跨层-only −7.80」
   并不矛盾：那里测的是「只保留一个通道」，这里测的是「只破坏一个通道」，
   两者都指向**层总量与层内分配必须同时对**。

**已排** `_flr01a`（ρ=0.1）与 `_flr03a`（ρ=0.3）的地板，补全「地板 vs 学习表」的
完整 ratio 曲线对照 —— 学习表在 ρ=0.1 只有 +4.20★、ρ=0.3 是 −0.60（不显著），
**若地板在这两处也追平，则学习表在整条曲线上都是冗余的。**

### ㊹ **决定性判读：地板在 PrefSuf 上远远不够 —— 学到的方向在那里是必需的**

`_psflr8` 落地（n=100）。PrefSuf @ρ=0.2，基线 39.20 / full 50.00 / headroom +10.80。
**协变量**：基线零配额头 **47.9%**，地板 `b_min=8` 后 **0.0%**，搬动 **480 槽**。

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| 自表 γ=1 | +7.20 | [+4.60,+10.00]★ | 46.40 | 67% |
| **自表 γ=2.157（最优）** | **+12.00** | [+8.60,+15.60]★ | **51.20** | **111%** |
| 迁移 KV→PrefSuf | +11.20 | [+8.00,+14.60]★ | 50.40 | 104% |
| **地板 `b_min=8`** | **+3.00** | **[+1.00,+5.00]★** | 42.20 | **28%** |

    地板 − 最优自表   = −9.00 [−12.20,−6.00]★
    地板 − 自表 γ=1   = −4.20 [−6.80,−1.80]★
    地板 − 迁移表      = −8.20 [−11.20,−5.40]★

**⇒ §㉕ 预注册的"叙事作废"条件没有触发。** 地板只拿 28%，而学到的方向拿 111%，
**差 9.00 分且高度显著**；连**未调幅**的自表（+7.20）都显著赢地板 **+4.20★**。

**因此本轮的核心命题要收窄，而不是作废：**

> **学到的配额方向在 PrefSuf 上是必需的**（比防饿死地板高 +9.00★），
> **在 Retr.KV 上不是**（与地板不可分）。

**一个把"饿死缓解解释一切"这个假说打死的独立角度**：两个 panel 的基线饿死率相近
（Retr.KV **41.3%**、PrefSuf **47.9%**，后者还更高），两处的地板都把它清到 0.0%，
但**一处追平学习表、另一处只拿 28%**。⇒ **饿死率高低不决定地板够不够用。**
这与 §㊱ 从共线性角度得到的"饿死缓解不充分"结论一致，但是**从另一个方向**独立成立。

**Retr.KV@0.2 的地位因此改变**：它不是"方法最有效的 panel"，而是
**"恰好存在更简单替代方案的 panel"**。+25.80 仍然是真的、可复现的（三种子 ±0.09），
但**它不能作为"学到的方向重要"的证据** —— 那个证据现在只在 PrefSuf 上。

**已排 `_psflr128`**（PrefSuf 地板 b_min=128）。`_psflr32b` 已 80/100。
**若整个地板族在 PrefSuf 上都停在个位数**，则"学到的方向在 PrefSuf 上必需"这条
就有了一族对照支撑，而不是单点。

### ㊺ 三个地板对照同时落地：伤害来自方向，且三个 panel 的图像互不相同

**① Retr.KV 地板曲线的右端断了（`_flr512`）**

| b_min | Δ | 95% CI | 恢复率 | 搬动量 |
|---|---|---|---|---|
| 8 | +27.00 | [+21.60,+32.60]★ | 117% | 406 |
| 32 | +25.80 | [+20.40,+31.20]★ | 112% | 1,742 |
| 128 | +23.40 | [+17.60,+29.00]★ | 102% | 7,430 |
| **512** | **+17.00** | [+10.40,+23.20]★ | **74%** | ~30,000 |

    512 − 128   = −6.40 [−10.40,−2.60]★
    512 − 完整表 = −8.80 [−13.60,−4.20]★

⇒ **「过度搬动独立地伤害」现在在地板族上也确立了**（此前只在 γ 扫描上确立）。
地板不是"随便设多大都行"，它也有内点最优（8–32 之间的平台）。

**② PrefSuf 地板族**整族**失败（`_psflr32b`）**

| 臂 | Δ | 95% CI | 恢复率 |
|---|---|---|---|
| 最优自表 γ=2.157 | +12.00 | [+8.60,+15.60]★ | 111% |
| 自表 γ=1（未调幅） | +7.20 | [+4.60,+10.00]★ | 67% |
| 地板 `b_min=8` | +3.00 | [+1.00,+5.00]★ | 28% |
| **地板 `b_min=32`** | **+4.00** | [+1.00,+7.00]★ | **37%** |

    地板32 − 最优自表 = −8.00 [−11.20,−4.80]★
    地板32 − 地板8   = +1.00 [−1.60,+3.60]  不可分

⇒ **不是单点失败，是整族失败。**「学到的方向在 PrefSuf 上必需」现在有**一族对照**
支撑，不再是单点。（`_psflr128` 在跑，会补第三点。）

**③ MultiHop：地板几乎什么也没做，而学习表造成严重伤害（`_vtflr8`）**

MultiHop @ρ=0.2，基线 46.09 / full 41.07 / **headroom −5.02（压缩本就赢满缓存）**。
**协变量：基线零配额头只有 0.3%。**

| 臂 | Δ | 95% CI | 绝对 |
|---|---|---|---|
| 静态表（学到的方向） | **−12.53** | [−15.11,−10.00]★ | 33.56 |
| **地板 `b_min=8`** | **−0.09** | **[−1.24,+1.11] 不显著** | 46.00 |

⇒ **MultiHop 上的伤害来自学到的方向本身，不是"任何预算扰动都会伤"。**
地板在那里几乎是 no-op（没有饿死可救），分数纹丝不动；而学习表照样大幅搬动预算，
掉 12.53 分。

### ㊻ 三个 panel 的图像互不相同，而**饿死率无法给它们排序**

| panel | 基线零配额头 | headroom | 地板 | 学到的方向 | 判定 |
|---|---|---|---|---|---|
| **Retr.KV** | 41.3% | +23.00 | **+27.00★（117%）** | +25.80★（112%） | **地板够用，方向非必需** |
| **Retr.PrefSuf** | **47.9%** | +10.80 | +3.00~+4.00★（28–37%） | **+12.00★（111%）** | **方向必需（+8~9★）** |
| **Retr.MultiHop** | **0.3%** | **−5.02** | −0.09（不显著） | **−12.53★** | **方向有害** |

**三条读数：**

1. **饿死率不排序**：47.9% > 41.3% ≫ 0.3%，但结论是"必需 / 非必需 / 有害"，
   与饿死率高低**没有单调关系**。⇒ 再次否掉「饿死缓解解释一切」。
2. **headroom 排序得上**：+23.00 / +10.80 / −5.02 对应"地板够用 / 方向必需 / 方向有害"。
   但 headroom **需要满缓存参考**，推理时拿不到 ⇒ **仍然不能当门控**。
3. **地板是个"安全但弱"的干预**：三个 panel 上分别是 +27.00★ / +3.00★ / −0.09（不显著）
   —— **从不造成显著伤害**。而学到的方向是"强但危险"：+25.80★ / +12.00★ / **−12.53★**。
   ⇒ **这正是「无害门控」要解决的问题的精确形状**：不是"要不要介入"，而是
   **"用弱的还是用强的"**。这比此前的二值门控设定更可实现，值得作为方法方向记下来。

### ㊼ 地板的完整 ratio 曲线：**在 ρ=0.1 上大幅胜过学习表，并推翻「工作区间窄」**

Retr.KV，full = 68.20，全部 n=100，同批基线：

| ρ | 基线 | headroom | 学习表 | **地板 `b_min=8`** | 配对 地板 − 表 |
|---|---|---|---|---|---|
| 0.3 | 65.40 | +2.80 | −0.60 | **+6.00★** | **+6.60 [+2.20,+11.00]★** |
| 0.2 | 45.20 | +23.00 | +25.80★ | +27.00★ | +1.20 [−1.80,+4.20] 不可分 |
| **0.1** | 32.60 | **+35.60** | **+4.20★** | **+33.60★（94%）** | **+29.40 [+24.20,+34.60]★** |

**协变量（ρ=0.1）**：基线零配额头 **64.7% → 0.0%**，地板只搬 **618 槽 = 总预算的 0.47%**。
**绝对分 66.20，满缓存 68.20 —— 在 10% 缓存下距满缓存只差 2 分。**

**⇒ 撤回「工作区间窄」这条长期结论。** 此前写的是：

> ~~恢复率对基线绝对分极度敏感：基线 ≳35 时恢复 75–90%，基线 <10 时只剩 0.4–6%。
> 机制上必然：我们做的是逐头预算再分配，ρ=0.1 时 65% 的头零配额、**没有东西可挪**。~~

**「没有东西可挪」正好说反了。** 那 65% 的饿死头**不是障碍，是机会** —— 只要给它们
各 8 个槽（占预算 0.47%），就能拿回 94% 的 headroom。**塌到 12% 的是学习表，不是问题本身。**

这也**改变了「工作点在断崖」的判断**：地板在 ρ=0.1（94%）比在 ρ=0.2（117% of 23.00
= 绝对 +27.00）拿回**更多绝对分数**（+33.60 vs +27.00）。⇒ **更激进的压缩反而是地板
更好的战场**，与学习表恰好相反。

**对本轮核心命题的影响（要收得更紧）**：此前说"学到的方向在 Retr.KV 上非必需（与地板
不可分）"，只在 ρ=0.2 上成立。**在 ρ=0.1 与 ρ=0.3 上，学到的方向不只是非必需，而是
显著劣于平凡地板**（−29.40★ / −6.60★）。

### ㊽ 移植 2×2 第一格：**Expected 的失败在配额，不在排序**

Retr.KV @ρ=0.2，n=100。用 `VARIKV_QUOTA_ABS` 把 Expected 的**逐样本逐 chunk 精确配额**
喂给 FastKVzip 的排序：

| 臂 | 绝对 | Δ vs FKV | 95% CI |
|---|---|---|---|
| FastKVzip 自身 | 45.20 | — | — |
| Expected Attention 自身（参照） | 1.40 | −43.80 | — |
| **FKV 排序 + Exp 配额** | **2.60** | **−42.60** | [−48.60,−36.40]★ |

**移植格落在 [1.40, 45.20] 这 43.80 分区间的 3% 位置** —— 几乎完全塌到 Expected 自身。

**⇒ FastKVzip 优秀的排序在 Expected 的配额下完全救不回来。两方法 43.80 分的差距
几乎全部来自配额，不是排序。** 这是"基数分配主导"的直接因果证据。

**但必须等镜像格**（`_xpExpqFKV`：Exp 排序 + FKV 配额，在跑）。若配额主导成立，
那一格应当**很高**（接近 45.20）。**两格不一致就说明存在排序×配额交互，单格不可信。**

**协变量（必须同报）**：Expected 的配额零饿死（源于 `safeguard=0.2`），FastKVzip 41.3%。
所以这次移植**同时**给 FKV 换上了一个零饿死的分配 —— 而它**塌了**。
⇒ **这是对「饿死缓解解释一切」的又一次独立否证**：把 FKV 换成零饿死分配不但没帮忙，
反而毁掉了它。**关键不在有没有饿死，而在饿死之外的分配形状对不对。**

### ㊾ PrefSuf 地板族第三点：`b_min=128` → −0.20（不显著）

| 臂 | Δ | 95% CI | 恢复率 |
|---|---|---|---|
| 最优自表 γ=2.157 | +12.00 | [+8.60,+15.60]★ | 111% |
| 地板 `b_min=8` | +3.00 | [+1.00,+5.00]★ | 28% |
| 地板 `b_min=32` | +4.00 | [+1.00,+7.00]★ | 37% |
| **地板 `b_min=128`** | **−0.20** | [−4.00,+3.60] 不显著 | −2% |

地板族在 PrefSuf 上非单调（8 → 32 → 128 给 28% → 37% → −2%），峰在 32 附近，
**整族都远低于学习表的 111%**。⇒ **「方向在 PrefSuf 上必需」有完整的三点对照支撑。**

### ㊿ **移植 2×2 完整且镜像一致：两个已发表方法之间的差距 100% 是配额、0% 是排序**

Retr.KV @ρ=0.2，n=100，逐样本**精确**配额移植（`VARIKV_QUOTA_ABS`，npz 存 `lo`/`hi`
逐 chunk 断言对齐）。两方法相差 **43.80 分**（FKV 45.20 vs Expected 1.40）。

| 排序来自 | 配额来自 | 绝对 | Δ vs FKV | 95% CI | **落点** |
|---|---|---|---|---|---|
| FKV | FKV | 45.20 | — | — | 100% |
| **FKV** | **Exp** | **2.60** | **−42.60** | [−48.60,−36.40]★ | **3%** |
| **Exp** | **FKV** | **44.20** | **−1.00** | **[−4.40,+2.60] 不可分** | **98%** |
| Exp | Exp | 1.40 | −43.80 | [−49.80,−37.80]★ | 0% |

**镜像检查通过**（这是本实验的可信度前提）：换配额几乎走完整个区间
（100% → 3%），换排序几乎不动（100% → 98%，**且与 FastKVzip 不可分**）。
两格方向相反、幅度互补 ⇒ **没有排序×配额交互，分解是干净的**。

**⇒ 43.80 分的差距约 100% 来自配额、约 0% 来自排序。**

**两条比"我们自己的臂"强得多的推论：**

1. **Expected Attention 的 token 排序其实和 FastKVzip 一样好。** 给它 FastKVzip 的配额，
   它拿到 **44.20**，与 FastKVzip 自身的 45.20 **不可分**。它在 ρ=0.2 上崩到 1.40
   **完全是分配问题**。⇒ 这条不是关于我们的方法，而是**关于两个已发表方法的结构性
   事实**，比"我们学到的东西其实是个 allocator"强一个量级。
2. **对「饿死缓解」的最强一次否证。** FKV 的配额有 **41.3% 的头零配额**，Expected 的
   配额 **0%**（`safeguard=0.2`）。而**好的那个配额恰恰是 41.3% 饿死的那个**：
   把 41.3%-饿死的 FKV 配额给 Expected ⇒ 44.20（优秀）；把零饿死的 Exp 配额给 FKV
   ⇒ 2.60（灾难）。**"别饿死头"作为解释被彻底推翻** —— 决定性的是**分配形状**，
   而饿死与否甚至方向都反了。

**一个必要的限定**：这是**单 panel（Retr.KV）单 ratio（0.2）**的结果，且 ρ=0.2 上
Expected 处于崩溃区（1.40）。在 Expected 有真实性能的 ρ=0.4（41.00）/0.5（56.00）上
重做，才能说这是一般性质而非崩溃区特例。**这是当前最值得排的下一个实验。**

### 51. kv 臂首次在 PrefSuf@0.2

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| 最优自表 γ=2.157（参照） | +12.00 | [+8.60,+15.60]★ | 51.20 | 111% |
| v2c 网络 s0（参照，637,828 参数） | +8.20 | [+5.00,+11.60]★ | 47.40 | 76% |
| **kv 臂 s0**（53,378 参数） | **+8.60** | [+5.59,+11.80]★ | 47.80 | **80%** |
| **kv 臂 s1** | **+8.40** | [+5.40,+11.40]★ | 47.60 | 78% |

**kv 两种子 +8.50 ± 0.10**，与 637,828 参数的 v2c（+8.20）**不可分**，
但都明显低于**调过幅度**的静态表（+12.00）。⇒ 在 PrefSuf 上，**幅度调节（γ≈2.157）
带来的 +3.5~4 分，比"用 12 倍参数的记忆架构"更值钱**。
`scalar` 在 PrefSuf 上的对照由过夜扫描提供（`_sc11_s0/s1`）。

### 52. `scalar` 首次离开 Retr.KV：三个 panel 基本惰性，**但 ρ=0.1 的塌陷跨 panel 复现**

过夜扫描 seed 0 的前三个（小 panel 先完成）。同批 `_g8base` 基线配对，n 为该 panel
的真实样本数（`choice_eng` 18、`qa_eng` 20、`many_shot` 54 —— 都是数据集本身的大小，
不是被截断）。

| panel | 满缓存 | ρ=0.75 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|---|---|
| En.MultiChoice | 79.17 | +0.00 | +0.00 | **+6.94★** | +1.39 | −1.39 | +4.17 | −2.78 |
| En.QA | 39.43 | −3.58 | +0.90 | −0.00 | −3.53 | −1.99 | +1.35 | +1.92 |
| ICL.ManyShot | 37.78 | +0.74 | **−2.59★** | **−4.07★** | −0.37 | ⌀ | ⌀ | ⌀ |

⌀ = 见下方退化说明。**21 格里只有 3 格 ★，其中两格是负的。**

**① 必须先扣掉的两格：静默退化陷阱又出现了。** ManyShot 在 ρ=0.1/0.05 上
Δ 恰好 `+0.00 [+0.00,+0.00]` —— 这正是 CLAUDE.md 记录的
`ratio × clen ≤ window_size` 指纹。日志逐条确认：

    clen=18614 window=1861 chunk_ratio=0.000000 degenerate=True     ← ρ=0.1
    clen=18614 window=930  chunk_ratio=0.000000 degenerate=True     ← ρ=0.05

`chunk_ratio` 被归零、window 缩成 `ratio×clen`，保留集**恒等于局部窗口**，
任何分数扰动都是构造性 no-op。**这两格不是"方法没用"，是"实验不存在"**，
必须标 ⌀ 而不是记 0。（ρ=0.2 上 `0.2×18614=3723 < 4096` 也退化，同样标 ⌀。）
`choice_eng`/`qa_eng` 全部 7 个 ratio 都 `degenerate=False`，可用。

**② 真正的发现：学习臂在 ρ=0.1 的塌陷不是 Retr.KV 独有的。**
这两个 panel 在 ρ=0.1 有**很大 headroom**，而 `scalar` 几乎一分没恢复：

| panel | ρ=0.1 基线 | headroom | `scalar` Δ | 恢复率 |
|---|---|---|---|---|
| En.MultiChoice | 59.26 | **+19.91** | +4.17 不可分 | 21% |
| En.QA | 19.67 | **+19.76** | +1.35 不可分 | 7% |
| （参照）Retr.KV | 32.60 | +35.60 | +4.20★ | 12% |

三个 panel、三个 20~36 分的 headroom，学习臂一律只拿 1~4 分（7~21%）。
⇒ **「学习表在 ρ=0.1 塌掉」是跨 panel 的性质，不是 Retr.KV 的特例。**
而在 Retr.KV 上，**同一个 ρ=0.1、同一个 headroom，反饿死地板拿 +33.60★（94%）**。

**⇒ 由此产生当前最高价值的下一个实验**：把地板 `b_min` 搬到
`choice_eng`/`qa_eng` 的 ρ=0.1 上。若地板在这两个 panel 也恢复大部分 headroom，
则"学习方向在低 ρ 无用、结构性下限才是关键"就从单 panel 上升为跨 panel 命题；
若不恢复，则 Retr.KV@0.1 的 +33.60 是 panel 特例，必须据此收紧论断。
**注意这是一个能双向证伪的判据，不是确认性实验。**

**③ headroom 符号律继续成立。** 三格 ★ 中两格 headroom 符号明确，两格都同号
（MultiChoice@0.4 headroom +5.56 / Δ +6.94；ManyShot@0.4 headroom −1.11 / Δ −4.07）；
第三格 ManyShot@0.5 headroom 恰为 0.00，无法判符号。En.QA 在 5 个 ratio 上
headroom 为负（压缩胜过满缓存，与 MultiHop 同族），`scalar` 在那里也一律不动或微负。

**④ 一个报告口径提醒**：`choice_eng` n=18 ⇒ 一条样本 = 5.56 分。
那个 +6.94★ 只相当于 1.25 条样本改对，**不要单独引用**。

### 53. **读了 Ada-KV（NeurIPS 2025）—— 逐头最小预算保证是先验技术，"反饿死地板"不是新机制**

阻塞项之一，原文读了（`arxiv.org/abs/2407.11550` + v4 全文）。**它直接落在我们整条
结论所在的轴上**，必须据此改写新颖性主张。

**Ada-KV 拥有的：**

- 标题即 *Optimizing KV Cache Eviction by **Adaptive Budget Allocation***，
  自称 **"the first head-wise adaptive budget allocation strategy"**。
- **分配规则（Algorithm 1）**：把各头的注意力权重拼起来 → 取**全局 top-B** →
  数每个头被选中几个 → 那就是该头的预算。**这与 `level=pair` 是同一个算子。**
- **safeguard（Algorithm 2 line 8）**：`B_i = α·B_i + (1−α)·(B/h)`，保证每个头
  不低于均匀分配的一部分。本仓库 `attention/score.py:133-137` 的实现是等价目的的
  硬下限版本：

      n_safe = int(int(k_len*ratio) * 0.2)                    # 仅 "adakv" 分支
      scores.scatter_(-1, topk(scores, n_safe).indices, +inf) # 每头前 n_safe 强制保留

- **Theorem 3.1**：给出驱逐前后注意力输出的 **L1 loss 上界**，并以此推导分配。
- 增强 SnapKV / Pyramid，Llama-3.1-8B，Ruler 13 个 + LongBench 16 个数据集。

**⇒ 必须改口的一条**：我们的"反饿死地板 `b_min`"**与 Ada-KV 的 safeguard 是同一类
机制**（保证每头一个最小预算），且该实现**就在本仓库里**，只是挂在 `adakv-layer` 上。
**不能把地板当作新机制。**

**但这没有抹掉那个测量，它换了个更准确的框架**：我们所有实验跑的是 `level=pair`
（`_threshold`，**无 safeguard**）。所以那个 +33.60 的正确读法是

> **FastKVzip 的默认 `level=pair` 丢掉了其祖先已发表的 safeguard，
> 而在 ρ=0.1 的 Retr.KV 上，这一omission 让它损失了 94% 的可恢复 headroom。**

这是关于**基线配置**的经验发现，不是新机制。**必须做的对照**：直接跑
`-g fastkvzip --level adakv-layer`（即带 safeguard 的先验技术配置）在 Retr.KV
ρ=0.1/0.2 上的分数。若它已经拿到 +33 量级，则地板一分新颖性也没有；若拿不到，
差在哪里就是真正需要解释的东西。**已排队**（`/tmp/lvl_sched.sh`）。
注意该对照**同时**改了两件事（加 safeguard、且把分配限制成逐层均匀），
是"已发表配置"的对照，不是纯 safeguard 消融。

**Ada-KV 明确**没有**做的（这是我们剩下的地盘）**：

1. **不区分排序与配额。** 全文始终把 Ada-KV 的分配与 Top-k 排序绑在一起，
   **没有任何跨方法移植实验**（拿 A 的排序配 B 的配额）。我们的移植 2×2 是新的。
2. 没有"保序重标定 ≡ 逐头配额分配"这类等价陈述，也没有把学习到的分数形变
   归约成一张整数配额表。
3. **不讨论饿死**：safeguard 隐式处理，但论文不分析、不可视化零配额头。
4. 它的理论靶子是**注意力输出保真度的上界**，而我们已测到**保真度与任务效用会背离**
   （Retr.MultiHop 上更忠实反而更差）。这构成对其优化目标的实证质疑。

**一条重要的边界**：Ada-KV 的分配是**层内跨头**（Algorithm 1 在层内拼接），
对应本仓库的 `adakv-layer`（84 维层内子空间）；而 `pair` 还跨层（111 维）。
我们测到**增益全在层内×跨层的交互**（层内 +2.20 不显著、跨层 −7.80★、完整 +25.80★）
⇒ **Ada-KV 覆盖的那 84 维单独拿不到这个效应。**（限定：该消融做的是把*我们学到的
增量*投影到层内子空间，不是直接消融 Ada-KV 自身的分配，不能直接等同。）

### 54. 核验：移植 2×2 **没有**被 `level` 混淆

发现两格跑在不同 level 上（`_g8base`/`_xpFKVqExp` 是 `output-pair.json`，
`_expbase`/`_xpExpqFKV` 是 `output-adakv-layer.json`），这本可以毁掉上一条结论
——"排序"的差异会与"分配方案"的差异混在一起。逐行读注入代码
（`learned_ctrlcache.py:206-212`）：

    sc  = score0[:, 0]                            # 本臂自己的基线分数 = 它的排序
    idx = argsort(sc, descending=True)            # 逐 (层,头) 按自己的分数排
    nv.scatter_(1, idx, arange(n) < q[:, None])   # 每头恰取前 q_lh 名
    valid = nv.reshape(L, H, n)                   # **整个覆盖**第 130 行算出的 valid

`valid` 被**完全重写**，第 130 行 `self.threshold(score, ratio, level)` 的结果
（连同 `adakv-layer` 的 safeguard）**被丢弃**。⇒ `level` 在这两格里是**残留标签**，
不进入决策。**2×2 干净，上一条结论不变。**

顺带这让饿死否证更锋利：Exp 的配额之所以零饿死，**正是因为 `adakv-layer` 上了
Ada-KV 的 safeguard**；把这个"教科书式健康"的配额换给 FKV 的排序，分数塌到 2.60。
而 FKV 那个 41.3% 饿死的配额换给 Expected，拿 44.20。
**与地板结果不矛盾**：地板只在 FKV 自己的分配上加下限、动 0.47~0.8% 预算，
移植是整体换掉分配。**小幅下限有益，整体换形状有害** —— 两者都指向"形状"而非"饿死率"。

### 55. **反饿死地板不能跨 panel —— 而且这是对饿死假说最干净的一次否证**

18:37 读到三个地板作业。设计时就写明这是**能双向证伪**的判据，结果落在约束我们的
那一侧。同批 `_g8base` 基线配对，ρ=0.1。

| panel | headroom | 学习表 `scalar` | 地板 b_min=8 | 地板 b_min=32 |
|---|---|---|---|---|
| **Retr.KV**（参照，n=100） | +35.60 | +4.20★（12%） | **+33.60★（94%）** | — |
| En.MultiChoice（n=18） | +19.91 | +4.17（21%）不可分 | +1.39（7%）不可分 | +2.78（14%）不可分 |
| En.QA（n=20） | +19.76 | +1.35（7%）不可分 | +2.07（10%）不可分 | 在跑 |

**关键在协变量：实验不是空转，饿死确实被清零了。** 从配额 dump 直接算：

| panel | 基线饿死率 | 地板后 | b<8 的头 | 搬动预算 | 预算守恒 |
|---|---|---|---|---|---|
| Retr.KV | 64.7% | 0.0% | — | 0.47% | ✓ |
| En.MultiChoice | **55.68%** | **0.00%** | 62.68% → 0.00% | 1.14% | 94863 = 94863 ✓ |
| En.QA | **56.53%** | **0.00%** | 63.24% → 0.00% | 1.17% | 93210 = 93210 ✓ |

**⇒ 三个 panel 的饿死率几乎相同（55.7% / 56.5% / 64.7%），被同一个算子全部清零，
任务上的恢复却差一个数量级（7~14% vs 94%）。消除饿死不是充分条件。**
这是**第四次**独立否证"饿死缓解解释一切"，也是最干净的一次 —— 前三次是间接的，
这一次是**直接操纵 + 协变量匹配**。

**两条必须同时写下的收紧：**

1. **`Retr.KV@0.1 的 +33.60 是 panel 特例**，不能外推。原来的表述"地板在 ρ=0.1
   恢复 94% headroom"必须限定到 Retr.KV。
2. **但不能说"地板无效"** —— n=18 时一条样本 5.56 分，CI 宽达 ±9.7。
   能排除的是**Retr.KV 量级**的效应（94% 对 MultiChoice 是 +18.7，远在
   [−8.33,+9.72] 之外；对 QA 是 +18.6，远在 [−2.58,+6.70] 之外），
   **排除不了中等效应**（30% 恢复仍在 CI 内）。

**Retr.KV 到底特殊在哪，目前无法归因。** 与 panel 身份共线的量至少有四个：
上下文长度（169k vs 101k）、样本数（100 vs 18/20）、基线绝对分（32.60 vs
59.26/19.67）、任务类型（精确键值检索 vs 语义 QA）。**三个 panel 分不开四个候选。**
已排 `scbench_prefix_suffix` @0.1（112k、n=100，`_psflr01a/b`）作为下一个观测点 ——
它长上下文、满样本，能同时松开"长度"与"样本数"两条。

### 56. SQuAD 的 "+32 headroom" 是个量纲陷阱，不并入跨 panel 论断

`_sc11_squad_s0` 完成。ρ≥0.2 全部惰性（headroom 全在 ±1 内，|Δ| ≤ 0.65，无一显著），
但 ρ=0.1 上 headroom 跳到 **+32.01**、`scalar` 拿 **+3.89★**，ρ=0.05 headroom
**+77.05**、+1.36 不可分。

看上去像"第四个 ρ=0.1 塌陷证据"，**但它不是同一个物理量**。日志实证：

    clen=146 window=2 chunk_ratio=0.087500 degenerate=False
    clen=178 window=3 chunk_ratio=0.086000 degenerate=False

SQuAD 的上下文只有 **146~178 个 token**，ρ=0.1 时 window 被缩到 **2**，
保留约 **15 个 token**。所以那 +32 的"headroom"衡量的是**把一小段文字摧毁得多优雅**，
与 169k token 的 Retr.KV 不是同一件事，也不在论文 x 轴（0.2–1.0）内。

**⇒ 不把 SQuAD@0.1/0.05 并入"学习臂在低 ρ 只恢复 7~21%"那条跨 panel 论断。**
（`degenerate=False` 是对的 —— 它没走 `chunk_ratio=0` 那个分支，是真的在驱逐；
陷阱在量纲，不在退化。**两个不同的坑，别混。**）
**也因此不在 SQuAD 上做地板实验**：每头配额只有个位数，`b_min=8` 会被
`project_quota` 的可行性钳制饱和成均匀分配，测不出东西。

### 57. 又三个 panel：**撤回「学习臂在低 ρ 只恢复 7~21%」**，范围其实是 −3% 到 +92%

`_sc11_scbench_vt_s0` / `_sc11_scbench_mf_s0` / `_sc11_gsm_s0` 完成。

**先扣掉 GSM8K —— 它是 SQuAD 那个量纲陷阱的第二例。** 日志实证
`clen=100 window=2`（100~102 token），ρ=0.1 时只剩约 10 个 token。
它那 +35.00 / +65.00 的"headroom"与 169k 的 Retr.KV 不是同一件事。
**排除在跨 panel 论断外**（`degenerate=False` 是对的，确实在驱逐；坑在量纲）。
`scalar` 在 GSM8K 上七个 ratio 全部不显著（|Δ| ≤ 4.00，CI 宽达 ±9）。

**Retr.MultiHop（clen=124,529，真长上下文，n=90）—— 同一 panel 内符号随 ratio 翻转：**

| ρ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | **0.05** |
|---|---|---|---|---|---|---|---|
| headroom | −0.80 | +0.71 | +0.71 | −1.60 | −5.02 | −8.40 | **+8.71** |
| `scalar` Δ | −3.56★ | −2.80★ | −2.44★ | −6.09★ | −9.11★ | −6.44★ | **+8.00★** |

**六个 ratio 显著为负，最后一个显著为正、且恢复 92% 的 headroom。**
这是本项目第一次看到**同一 panel 内、同一个臂，符号随工作点翻转**。
（补充口径：MultiHop 上压缩本来就胜过满缓存，ρ=0.1 基线 49.47 > 满缓存 41.07；
到 ρ=0.05 基线才塌到 32.36。若以"压缩能达到的最好值 49.47"为参照，
ρ=0.05 损失 17.11，`scalar` 挽回 8.00 = 47%。**两种口径都要报，因为
"headroom 相对满缓存"在这个 panel 上会低估损失。**）

**Math.Find（clen=120,671，n=100）：全惰性。** 七个 ratio 无一显著，|Δ| ≤ 1.33；
唯一有真 headroom 的 ρ=0.05（+15.83）拿 −0.50。

**⇒ 撤回清单第 13 条**：~~学习臂在 ρ=0.1 面对 20~36 分 headroom 一律只恢复 7~21%~~。
在**真长上下文**的 panel 上，恢复率实测跨度是 **−3% 到 +92%**：

| panel | clen | ρ | headroom | 恢复率 |
|---|---|---|---|---|
| Retr.MultiHop | 124k | 0.05 | +8.71 | **+92%★** |
| En.MultiChoice | 101k | 0.1 | +19.91 | +21% ns |
| Retr.KV | 169k | 0.1 | +35.60 | +12%★ |
| En.QA | 101k | 0.1 | +19.76 | +7% ns |
| Math.Find | 120k | 0.05 | +15.83 | −3% ns |

**没有可见的预测量**：headroom 大小不预测（+35.60 拿 12%，+8.71 拿 92%），
上下文长度不预测，panel 类别也不预测。三个 panel 时看到的"7~21% 一致性"
是**小样本下的巧合**，n=5 就散开了。

**⇒ 同时给出一条方法论**：这已是本轮第二次「三点看起来一致、第四第五点推翻」
（前一次是 `szmr0` 的散布）。**跨 panel 的一致性主张至少要 5 个 panel。**

### 58. **headroom 符号律出现第一批真正的反例**

此前 14 个格子（2 panel × 7 ratio）全部同号，写成了"符号律"。MultiHop 的完整
ratio 曲线给出**两个显著的反例**：

| ρ | headroom | `scalar` Δ | |
|---|---|---|---|
| 0.5 | **+0.71** | **−2.80★** | 反号且显著 |
| 0.4 | **+0.71** | **−2.44★** | 反号且显著 |

两处 headroom 都只有 +0.71（约等于噪声），所以这不是强反例，但**它们的 CI 排除了
零**，不能当作噪声打发。**正确表述**：符号律在 |headroom| 明显非零时成立，
在 |headroom| ≲ 1 的区域**没有预测力，甚至会反号**。这与"该 panel 上方向本身有害"
是一致的 —— 没有可挪的余量时，任何干预都只会掉分。

### 59. En.QA 地板族补齐：10% / −6%，整族无效

| 臂 | Δ | 95% CI | 恢复率 |
|---|---|---|---|
| 学习表 `scalar` | +1.35 | [−3.80,+6.43] | 7% |
| 地板 b_min=8 | +2.07 | [−2.58,+6.70] | 10% |
| 地板 b_min=32 | −1.10 | [−5.54,+3.58] | **−6%** |

两点地板族 10% / −6%，与学习表一样都不可分。**加上 MultiChoice 的 7%/14%，
四个地板点全部远低于 Retr.KV 的 94%** —— 第 55 条的结论不因补齐而改变。

### 60. **PrefSuf 地板：排除了「长度」与「样本数」两个共线量，Retr.KV 的 94% 仍无解释**

`_psflr01a/b` 完成。这一格是特意为**解共线**排的：PrefSuf 在**上下文长度（112k）
与样本数（100）上都对齐 Retr.KV**，只有任务与基线分不同。

Retr.PrefixSuffix @ρ=0.1（满缓存 50.00、基线 **8.60**、headroom **+41.40**、n=100
⇒ 1.00 分/条）：

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| 地板 b_min=8 | **+7.80** | [+5.40,+10.40]★ | 16.40 | **19%** |
| 地板 b_min=32 | **+12.40** | [+9.20,+15.80]★ | 21.00 | **30%** |

饿死率 **71.78% → 0.00%**，搬动 1.336%，预算守恒 99965 = 99965。

**四个 panel 的地板全表（ρ=0.1）：**

| panel | clen | n | headroom | 基线绝对 | 饿死→0 | b8 | b32 |
|---|---|---|---|---|---|---|---|
| Retr.KV | 169k | 100 | +35.60 | 32.60 | 64.7% | **+33.60★ (94%)** | — |
| **PrefSuf** | **112k** | **100** | +41.40 | 8.60 | 71.8% | **+7.80★ (19%)** | **+12.40★ (30%)** |
| MultiChoice | 101k | 18 | +19.91 | 59.26 | 55.7% | +1.39 (7%) ns | +2.78 (14%) ns |
| En.QA | 101k | 20 | +19.76 | 19.67 | 56.5% | +2.07 (10%) ns | −1.10 (−6%) ns |

**两条结论，一正一负：**

1. **地板不是 Retr.KV 独有的** —— 现在有**两个 panel 拿到显著为正**的地板增益
   （94% 与 19~30%）。上一条写的"地板不能跨 panel"要修正为**"跨 panel 但幅度分级"**。
   而且 PrefSuf 上 **b32 > b8**（+12.40 vs +7.80，CI 几乎不重叠），有单调趋势 ⇒
   已排 `_psflr01c`（b_min=128）追这条趋势。（注意 ρ=0.2 上该 panel 的地板族是
   8/32/128 → 28%/37%/**−2%**，128 会塌；ρ=0.1 未必同形。）

2. **但 Retr.KV 的 94% 仍然无法解释，而且四个候选共线量现在**全部**被排除：**

   | 候选 | 是否能解释 Retr.KV 的 94% | 证据 |
   |---|---|---|
   | 上下文长度 | **否** | PrefSuf 112k 对齐，只拿 19~30% |
   | 样本数 | **否** | PrefSuf n=100 对齐，只拿 19~30% |
   | 基线绝对分 | **否，非单调** | 32.60→94%、8.60→19~30%、59.26→7~14%、**19.67→10%** |
   | 任务族（检索 vs 语义） | **否** | PrefSuf 也是检索型，仍只有 19~30% |
   | 饿死率 | **否**（早已排除） | 四个 panel 55.7~71.8% 全部清零，结果差 5 倍 |

   ⇒ **「Retr.KV 为什么能被 0.47% 的预算搬动换回 94% headroom」目前没有任何
   已测量的量能解释。** 这是当前最大的一个未解释项，且它恰好是本项目全部
   正结果的来源 panel —— **在解释它之前，不应把 Retr.KV 的数字当作方法的代表性能。**

### 61. En.Summary：惰性，一个微小但显著的格

`scalar` 跨 panel 第 8 个（clen 约 102k，n=70）。七个 ratio 里六个 |Δ| ≤ 0.48
且全不显著；唯一 ★ 是 ρ=0.1 的 **+1.05 [+0.33,+1.77]**，占其 +5.99 headroom 的 **18%**。
ρ=0.05 headroom +9.38 却拿 −0.13。

加进恢复率表后，真长上下文 panel 的跨度仍是 **−3% ~ +92%**，
第 13 条撤回（低 ρ 恢复率无跨 panel 一致性）不变，且样本更足。

### 62. **先验技术对照：Ada-KV 的已发表配置在这里塌到 1.80，地板的价值来自「下限 × 全局分配」这个组合**

`_adakv01` 完成。Retr.KV @ρ=0.1，n=100，全部与 `level=pair` 基线逐样本配对。

| 配置 | 绝对 | Δ vs 基线 | 95% CI | 恢复率 |
|---|---|---|---|---|
| FastKVzip `level=pair`（基线，**无** safeguard） | 32.60 | — | — | 0% |
| **Ada-KV 配置 `level=adakv-layer`（有 safeguard）** | **1.80** | **−30.80** | [−37.20,−24.80]★ | **−87%** |
| `pair` + 反饿死地板 b_min=8 | **66.20** | **+33.60** | [+28.80,+38.20]★ | **94%** |

（满缓存 68.20，headroom +35.60。）

**⇒ 先验技术的已发表配置不但没拿到这个效应，反而是全场最差。** 所以
"FastKVzip 丢掉了祖先的 safeguard、补上就好了"这个说法**不成立** —— 补 safeguard
的那个已发表配置在这个工作点是灾难。

**但必须同时说清混淆**（排它的时候就写明了）：`adakv-layer` **同时改两件事** ——
(a) 加 safeguard，(b) **把分配限制成逐层均匀**（84 维层内子空间，而 `pair` 是 111 维
全局）。我们早已测到跨层分配极其重要（层内 +2.20ns / 跨层 −7.80★ / 完整 +25.80★），
所以这 −30.80 完全可能全部来自 (b)。**这一格证明不了 safeguard 有害。**

**缺的那格已排**：`--level layer` —— `score.py:104` 只在 level 含 `adakv` 时给
`safeguard=0.2`，所以 `layer` 就是"逐层均匀 + **无** safeguard"，正好补成 2×2：

|  | 无 safeguard | 有 safeguard |
|---|---|---|
| **全局分配**（pair，111 维） | 32.60（基线） | **66.20**（我们的地板） |
| **逐层均匀**（84 维） | **`_lyr01` 在跑** | 1.80（adakv-layer） |

（口径提醒：我们的 b_min=8 与 Ada-KV 的 `n_safe = 0.2·ρ·k_len` 不是同一个算子，
后者大得多，所以左右两列的"safeguard"强度不同；但**同一行内**的
`layer` vs `adakv-layer` 是干净的 safeguard 消融。）

**目前能站住的表述**：地板的价值来自**「最小预算下限 × 全局跨层分配」这个组合**，
而这个组合**两个方法都没有** —— FastKVzip 有全局分配、无下限；Ada-KV 有下限、
但分配是层内的。**各自的零件都是先验技术，组合不是。**（撤回 11 保持：
下限本身不是新机制。）

### 63. PrefSuf 的完整 ratio 曲线：**这个 panel 的基线非单调到无法支撑逐 ratio 论断**

`_sc11_scbench_prefix_suffix_s0` 完成。满缓存 50.00：

| ρ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|---|
| **基线** | 36.80 | 48.20 | **57.80** | 51.60 | 39.20 | 8.60 | 1.20 |
| headroom | +13.20 | +1.80 | −7.80 | −1.60 | +10.80 | +41.40 | +48.80 |
| `scalar` Δ | +14.20★ | **−17.60★** | −4.00★ | +4.60★ | +8.40★ | +3.00★ | +0.40 |

**基线本身在 1.20 到 57.80 之间非单调震荡，ρ=0.4 的 57.80 还高过满缓存 50.00。**
CLAUDE.md 早就警告过这个 panel"n=100 下极噪、连获胜方法都非单调"，这里是最极端的
一次证实。**后果**：该 panel 的"恢复率"列在多数 ratio 上没有意义（算出 −978%、−287%
这种数），**除 ρ=0.2 外不要引用它的逐 ratio 数字** —— ρ=0.2 有三种子且基线稳定。

**一条真正有信息的**：ρ=0.1 上 `scalar` 只有 **+3.00★（7%）**，而**地板拿
+7.80★/+12.40★（19%/30%）**。这与 ρ=0.2 的排序**正好相反**（那里方向必需，
学习表比地板高 +8~9★）。⇒ **"地板 vs 学习方向"孰优，在同一个 panel 内也随 ratio 翻转。**
这与第 57 条（MultiHop 上符号随 ratio 翻转）是同一类现象：
**工作点是与 panel 同等重要的一个自变量，不能在报告里被平均掉。**

### 64. PrefSuf 地板族出现内部最优（b_min=32），且三点全部胜过学习表

`_psflr01c` 补齐。PrefSuf @ρ=0.1，满缓存 50.00、基线 8.60、headroom +41.40、n=100：

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| 地板 b_min=8 | +7.80 | [+5.40,+10.40]★ | 16.40 | 18.8% |
| **地板 b_min=32** | **+12.40** | [+9.20,+15.80]★ | 21.00 | **30.0%** |
| 地板 b_min=128 | +5.00 | [+2.20,+7.60]★ | 13.60 | 12.1% |
| 学习表 `scalar` s0 | +3.00 | [+1.20,+5.00]★ | 11.60 | 7.2% |

**两条**：

1. **地板有内部最优，峰在 b_min=32**，且与该 panel 在 **ρ=0.2** 上的族
   （8/32/128 → 28% / **37%** / −2%）**峰位一致**。Retr.KV@0.2 上 8/32/128 三点
   互相不可分、512 才变差。⇒ **b_min≈32 是目前唯一一个跨 panel、跨 ratio 都不差的
   取值**，这算一条（弱的）可操作结论。注意 b128 在 ρ=0.2 塌到 −2% 而在 ρ=0.1 仍
   +12.1%★ —— **族的形状本身也随 ratio 变**。
2. **在这个 panel 的 ρ=0.1 上，地板三点全部显著胜过学习表**（18.8/30.0/12.1 vs 7.2）。
   连同 ρ=0.2 上学习表反超地板 +8~9★，**同一 panel 内孰优随 ratio 翻转**再添一例。

### 65. `scalar` 两种子（小 panel）

seed1 的 `choice_eng`/`qa_eng` 完成，与 seed0 配成两种子（同一批 `_g8base` 基线）：

**En.MultiChoice**（n=18 ⇒ 5.56 分/条）

| ρ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|---|
| headroom | −2.78 | +1.39 | +5.56 | +2.78 | +6.94 | +19.91 | +25.93 |
| s0 / s1 | 0/0 | 0/+1.39 | **+6.94★/+5.56★** | +1.39/+1.39 | −1.39/+1.39 | +4.17/**+8.33★** | −2.78/−2.78 |

**En.QA**（n=20 ⇒ 5.00 分/条）：**两种子、七个 ratio、十四格无一显著**，
|Δ| ≤ 3.65；ρ=0.1 均值 +1.05（5%）、ρ=0.05 均值 +1.92（8%）。

⇒ MultiChoice 上 **ρ=0.4 两种子都 ★**（均值 +6.25，占 headroom 112%），
ρ=0.1 均值 +6.25（31%，1/2 种子 ★）；QA 上**什么都没有**。
两种子散布不大（ρ=0.1 上 4.17 vs 8.33 = 0.75 条样本），所以这不是种子噪声，
是**两个 panel 的真实差异**。

### 66. 第三次栽在「猜选项名」上 —— `--level layer` 不存在

`_lyr01`/`_lyr02` 以 **rc=2** 秒退，日志只有 argparse 的 usage。
`args.py:22` 的合法取值是 `["pair","pair-head","pair-layer","adakv-layer",""]`
—— **没有 `layer`**。正确名字是 **`pair-layer`**，已重发为 `_plyr01`/`_plyr02`。
（`score.py:104` 的判据是 `"layer" in level` 走 `_threshold_layer`、
`"adakv" in level` 才给 `safeguard=0.2`，所以 `pair-layer` 确实是"逐层均匀 + 无
safeguard"那一格。）

**这是本项目第三次同类失败**（前两次：`scratch_ctrl_train.py` 的 `--arch` 手抄选项
漏了因子臂、12 个训练全被 argparse 拒且调度器看不出异常；以及本轮）。
**规矩**：**发作业前先 grep 源码里的 `choices=`，不要凭印象写选项名**；
argparse 失败是 rc=2 且日志只有 usage，**调度器会把它记成正常完成**。

### 67. **`adakv-layer` 在两个 ratio 都塌，而且它独立复证了移植 2×2 —— 同时改写了那条结论的措辞**

`_adakv02` 完成。Retr.KV，n=100，对同 ratio 的 `level=pair` 基线配对：

| ρ | pair 基线 | headroom | `adakv-layer` 绝对 | Δ | 95% CI |
|---|---|---|---|---|---|
| 0.1 | 32.60 | +35.60 | **1.80** | −30.80 | [−37.20,−24.80]★ |
| 0.2 | 45.20 | +23.00 | **2.40** | −42.80 | [−48.80,−36.80]★ |

**独立复证**：ρ=0.2 上，`adakv-layer` 的 **2.40** 与移植格
"FKV 排序 + Exp 精确配额" 的 **2.60** 几乎相同 —— 两条完全不同的实现路径
（原生 level 机制 vs 逐样本精确配额注入）给出同一个数。**移植 2×2 的分解是对的。**

**但它改写了措辞，而且是往不利方向改。** 关键在 `args.py:125-132`：level 是
**按门控自动派生**的 ——

    "expect" in gate  →  adakv-layer
    "snap"   in gate  →  pair-head
    否则              →  pair

所以 `_expbase` 是 Expected Attention 的**规范配置**（没跑错），
而两个方法之间那 43.80 分的差距，**主要归因于驱逐结构（level）这个粗粒度配置选择**，
不是什么隐藏的精细配额形状。这**不是**新发现的机制，是两篇论文各自公开声明的配置差异。

**仍然值得说的两点**：① **量级** —— 一个结构选择值 43.80 分；
② 固定 level 后，**两个门控只差 1.00 分**（2.40 vs 1.40）——
而门控恰恰是两篇论文各自的核心贡献。

**但 ② 目前还站不住**，因为两边都在崩溃区（1~2 分），没有分辨力。
判定它需要 gate × level 的 2×2：

| Retr.KV @ρ=0.2 | `level=pair` | `level=adakv-layer` |
|---|---|---|
| fastkvzip 门控 | **45.20** | **2.40** |
| expect 门控 | **`_exppair02` 在跑** | **1.40** |

已排 `_exppair02`(ρ=0.2) 与 `_exppair01`(ρ=0.1)（新脚本 `/tmp/qrun_gl.sh` 支持
同时指定 gate 与 level）。**若 expect@pair ≈ 45**，则"门控几乎不重要、全是结构"
成立且有分辨力；**若显著更低**，则门控重要、只是需要结构允许它发挥。

**一条必须同时改的**：`RESULTS_ALLOC_2026-08-18.md` 里 5 基线对比表中
Expected Attention 的分数**与它的 level 绑定**，引用时必须写明
"Expected Attention（其规范配置 `adakv-layer`）"，不能简写成方法名 ——
否则读者会以为那是打分器的差距。

### 68. **读了 DBTrimKV（2605.09649，真实标题 *Make Each Token Count*）—— 它拿走两件我们当作自己的东西，但没拿走分析**

**证据等级说明**：经 WebFetch 读了 arXiv 摘要页与 HTML v1 全文（由模型抽取），
**我没有亲自读 PDF**。下面的公式与数字按抽取结果记录，**写进论文前必须对 PDF 复核**。
（CLAUDE.md 早先关于这篇的判断是**基于外部转述**，本条是第一次直接读原文。）

**它拥有的（两条都直击我们的主张）：**

1. **学习到的逐 (层,头) 投影 + 跨层头共享读出，使分数可全局比较、全局竞争预算。**

       g_{ℓ,h}(x) = σ( w_g^T · Proj_{ℓ,h}(x) + b_g )

   `Proj_{ℓ,h}` 是逐 (层,头) 的两层 MLP（隐藏维 512），`(w_g, b_g)` **跨层头绑定**。
   原文：*"we maintain one global KV budget and retain the entries with the highest
   predicted utility across all layers, heads, and modalities."*
   ⇒ **这在结构上就是我们的 `scalar` 臂**（逐 (层,头) 学习形变 + 全局 top-B）。
   我们那条"因子消融里唯一站得住的是逐头形变的函数类丰富度"，
   **实际上是在刻画他们的架构**。

2. **"压缩可以胜过满缓存"，而且有定理。** 摘要原文：*"full-cache attention is not
   always optimal: irrelevant tokens can dilute attention away from useful evidence,
   so selective, learnable eviction can improve generation rather than merely
   approximate the full cache."* Proposition 3.1（近似平局的干扰项迫使注意力稀释）
   + Corollary 3.2（优先保留有用 token 降低稀释）。实测 MMDU 上达到 vanilla 的
   **114.46%**、MathVision 上 51.97 vs 48.68。
   ⇒ **`FINDINGS_DENOISING.md` 的核心观察、以及我们 headroom 表里那些负 headroom
   panel（MultiHop 41.07→46.09、En.QA），现在都是已发表结论，还配了理论。**

**它明确没有做的（全文核实，这是我们剩下的全部）：**

| 我们的 | DBTrimKV |
|---|---|
| **保序重标定 ≡ 逐头配额分配**（定理 + 0/18,478,208 位验证） | **无**。抽取原文：*does not separate ranking from allocation as equivalent procedures… No ablation isolates budget equivalence* |
| **排序 vs 配额的跨方法移植 2×2** | 无 |
| 把 637,828 参数的网络归约成 **112 个整数** | 无 |
| 配额空间分解 111 = 84 层内 + 27 跨层 | 无 |
| **饿死率测量与地板族** | **无**：不报逐头配额分布、不讨论零配额头 |

**一个反过来对我们有利的点**：抽取说他们的全局策略是"global ranking, **not**
per-head quota assignment"。但**由我们的定理，逐 (层,头) 单调形变 + 全局 top-B
恰恰就是逐头配额分配** —— 他们在做配额分配而不自知。
⇒ **我们的分析正是在解释他们的方法在做什么。**

**其他核实到的**：`r_{t,i} = β_i^{t−i}`，且 *"we estimate it only once when the token
enters the cache"* ⇒ **CLAUDE.md 第 ⑥ 条（不要把新颖性建在"静态 vs 动态"上）
基于转述写的，现由原文确认无误。**
**基准无重叠**：他们用 LLaVA-1.5-7B / Qwen3-VL-8B/4B / Qwen3-4B，
数据是 VQA/video/math（VQAText、MME、GQA、MMStar、MathVision、VideoMME、MMDU、
AIME24、GSM8K、MATH-500），**与 SCBench + Qwen2.5-7B-1M 完全不撞**。

**⇒ 定位必须改：从"方法"改为"对一类已发表方法的分析"。**
这个类现在至少有两个成员 —— Ada-KV（结构性的跨头预算分配）与 DBTrimKV
（学习到的跨层头校准 + 全局竞争）。我们能提供的是**它们都没有的东西**：
把"学习打分器"归约为"逐头配额分配"，给出等价定理与位级验证，
用跨方法移植把排序与配额分开测，并指出**112 个整数就够**。
这是一篇分析/立场论文的骨架，不是方法论文的。

**另一条现在更值钱的**：我们测到 **4,482 参数（乃至 112 个整数）就能追平
637,828 参数**，而 DBTrimKV 用的是逐 (层,头) 的 512 维两层 MLP。
**"这类方法的有效自由度远小于其参数量"** 是一条能直接施加于已发表方法的论断。

### 69. **复现检查通过（100/100 逐样本相同），并顺带追认了四臂结果不受 mode 默认值 bug 影响**

过夜扫描的 `_sc11_s0` 完成。Retr.KV @ρ=0.2，与历史 `_d10scalar_s0` 逐样本比对：

| | Δ vs 同批基线 | 95% CI | 绝对 |
|---|---|---|---|
| 历史 `_d10scalar_s0` | **+21.60** | [+17.20,+26.20] | 66.80 |
| 新跑 `_sc11_s0` | **+21.60** | [+17.20,+26.20] | 66.80 |
| **逐样本完全相同** | **100/100** | | |

**两条：**

1. 评测在 169k token 的 panel 上端到端确定性，再次确认。
2. **顺带关掉一个悬着的疑点。** 历史那批的结果目录后缀是 `ctrlmstat8_scalar`
   （因为 `--ctrlm_mode` 的默认值曾是 `"stateful"`），而新跑显式传了
   `--ctrlm_mode memoryless`、目录后缀是 `ctrlmmemo8_scalar`。
   **两者逐样本相同** ⇒ `eval_chunk.py:142` 对 `arch≠memory` 强制改回 memoryless
   **确实生效**，那个后缀只是命名假象。**⇒ 整张四臂分解表（bias/affine/scalar/kv）
   不受 mode 默认值 bug 影响，可以放心引用。** 此前这只是读代码得出的结论，
   现在是测出来的。

### 70. **`scalar` 在 Retr.KV 上是一个单点尖峰，而地板随 headroom 单调**

新跑覆盖了此前从未测过的 ratio。满缓存 68.20：

| ρ | 基线 | headroom | 学习 `scalar` | 恢复率 | 地板 b_min=8 |
|---|---|---|---|---|---|
| 0.75 | 68.80 | −0.60 | −1.20 ns | — | — |
| 0.5 | **71.60** | **−3.40** | −2.00 ns | — | — |
| 0.4 | 66.40 | +1.80 | +0.40 ns | 22% | — |
| 0.3 | 65.40 | +2.80 | −3.80 ns | −136% | **+6.00★** |
| **0.2** | 45.20 | +23.00 | **+21.60★** | **94%** | **+27.00★** |
| 0.1 | 32.60 | +35.60 | +4.20★（历史三种子） | 12% | **+33.60★ (94%)** |
| 0.05 | **2.00** | **+66.20** | **+0.80 ns** | **1%** | 已排 `_flr005` |

**三条结论：**

1. **学习臂只在一个 ratio 上工作。** ρ=0.2 拿 94%，其余六个 ratio 全部不显著或为负
   —— 包括 headroom 更大的 ρ=0.1（12%）与 ρ=0.05（**+66.20 的 headroom 只恢复 1%**）。
2. **地板不是这样：它随 headroom 单调上升**（0.3 → +6.00★、0.2 → +27.00★、
   0.1 → +33.60★）。⇒ **两者是性质不同的东西**，不只是强弱之别。
   已排 `_flr005`/`_flr005b`（Retr.KV @0.05，headroom +66.20）检验这条单调性
   会不会在极端点断掉。
3. **而且它峰值不在自己的训练点上。** 教师 trace 的 `--ratio` 默认 **0.1**，
   且所有 `scratch_ctrl_teacher*.sh` 都不传该参数 ⇒ 训练用的候选池是
   `Near(τ_{ρ=0.1})`。**它在训练 ratio 上只拿 12%，在没训过的 ρ=0.2 上拿 94%。**
   所以"它被调到了自己的工作点"这个自然解释**不成立**，
   ρ=0.2 为什么特殊仍然无解 —— 这与第 60 条"Retr.KV 的 94% 五候选全排除仍无解释"
   是同一个未解释项的两个侧面。

**一条记录缺口**：`scratch_ctrl_traces_v2_10/doc*.pt` 的顶层键只有
`['H','L','chunks','doc']`，**不存生成时的 args**，所以上面的训练 ratio 是从
"脚本不传该参数 ⇒ 用默认值"推出来的，不是从数据里读出来的。
**与 CLAUDE.md 已记录的"ckpt 不存训练 args"是同一类缺口，教师 trace 应一并补上。**

**顺带一条与 DBTrimKV 呼应的**：ρ=0.5 时 Retr.KV 基线 **71.60 > 满缓存 68.20**
⇒ **在温和压缩下，压缩胜过满缓存的现象在 Retr.KV 上也存在**，
不只在 MultiHop / En.QA。这正是 DBTrimKV 注意力稀释定理描述的效应。

### 71. **MultiHop 的符号翻转两种子全复现（14/14 格显著），且给出一个对照极紧的跨 panel 比较**

`_sc11_scbench_vt_s1` / `_sc11_scbench_mf_s1` 完成。

**Retr.MultiHop（clen 124k，n=90，满缓存 41.07）—— 两种子、七个 ratio、14 格全部 ★：**

| ρ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | **0.05** |
|---|---|---|---|---|---|---|---|
| headroom | −0.80 | +0.71 | +0.71 | −1.60 | −5.02 | −8.40 | **+8.71** |
| s0 | −3.56 | −2.80 | −2.44 | −6.09 | −9.11 | −6.44 | **+8.00** |
| s1 | −2.44 | −2.09 | −2.04 | −4.98 | −5.33 | −5.20 | **+8.40** |
| 均值 | −3.00 | −2.44 | −2.24 | −5.53 | −7.22 | −5.82 | **+8.20** |

**⇒ 符号随工作点翻转的现象在两个种子上完全复现，且每一格都显著。**
这不再是单种子的观察。种子散布多数在 1 分内，最大是 ρ=0.2（−9.11 vs −5.33，差 3.78）。

**这也改进了先前"headroom 符号律有真反例"的写法。** 看整条曲线，
Δ 与 headroom 明显同向，唯二"反号"的两格 headroom 都只有 **+0.71**（约等于零）。
更准确的描述是：**在这个 panel 上，学习臂 = 一个约 −2.5 分的常数代价
＋ 一个跟随 headroom 的分量**。零 headroom 处它净亏约 2.5 分，
headroom +8.71 处净赚 +8.20。**"反例"其实是那个常数代价，不是符号律失效。**
（限定：7 个点拟合两参数，别把系数当结论；定性部分才是站得住的。）

**Math.Find（clen 120k，n=100，满缓存 33.17）—— 两种子 14 格无一显著**，
|Δ| ≤ 1.50；**包括 headroom +15.83 的 ρ=0.05，两种子 −0.50 / −1.00。**

**由此得到一个共线量控制得最紧的跨 panel 比较：**

| panel | clen | n | ρ | headroom | 恢复率 |
|---|---|---|---|---|---|
| **Retr.MultiHop** | 124k | 90 | **0.05** | +8.71 | **+94%（两种子 ★★）** |
| **Math.Find** | 120k | 100 | **0.05** | +15.83 | **−5%（两种子不显著）** |

**上下文长度、ratio、样本量、headroom 符号全部对齐（Math.Find 的 headroom 还更大），
恢复率却差 100 个百分点。** ⇒ 第 57 条"低 ρ 恢复率无跨 panel 一致性"的最强证据，
且把共线候选进一步压缩到 **panel/任务本身**。

### 72. **`scalar` 臂给定 (panel, ratio) 后几乎是确定性的：42 格两种子中位差 0.45 分**

seed1 的 `prefix_suffix` 与 `summary` 完成，两种子覆盖到 6 个 panel × 7 个 ratio。
逐格算 `|Δ_{s0} − Δ_{s1}|`：

| 统计量 | 值 |
|---|---|
| 覆盖 | **42 格**（vt / mf / prefix_suffix / summary / choice_eng / qa_eng） |
| 中位 | **0.45 分** |
| 均值 | 0.80 分 |
| 最大 | 4.17（`choice_eng` ρ=0.1，n=18 ⇒ 仅 **0.75 条样本**） |
| < 1.0 分 | **29 / 42** |
| < 2.0 分 | 38 / 42 |

**对照：v1 记忆架构三次训练的下游跨度是 39 分**（+21.60 / −17.40 / −8.75，
CLAUDE.md 已记的那条撤回）。

⇒ **两件事**：① 项目那条"一次训练不是一次测量、必须报 n≥3 种子"的规矩，
**在因子臂上不再是瓶颈** —— `scalar` 的行为由 (panel, ratio) 决定，训练种子贡献不到 1 分；
② 这**支持**「网络的全部功能是推断该 workload 用哪张配额表」那条刻画：
若它学的是一个近乎确定的映射，种子散布本就该小。**注意这不追认 v1 的任何结论** ——
v1 是另一个架构、另一个目标，它的 39 分跨度依然成立。

### 73. PrefSuf 两种子全曲线：相邻 ratio 之间摆动 31 分，且可复现

满缓存 50.00，n=100：

| ρ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 | 0.1 | 0.05 |
|---|---|---|---|---|---|---|---|
| 基线 | 36.80 | 48.20 | **57.80** | 51.60 | 39.20 | 8.60 | 1.20 |
| headroom | +13.20 | +1.80 | −7.80 | −1.60 | +10.80 | +41.40 | +48.80 |
| s0 | +14.20 | −17.60 | −4.00 | +4.60 | +8.40 | +3.00 | +0.40 |
| s1 | +13.60 | −16.60 | −5.20 | +3.80 | +8.40 | +2.60 | +0.20 |
| 均值 | **+13.90★★** | **−17.10★★** | −4.60★★ | +4.20★★ | **+8.40★★** | +2.80★★ | +0.30 -- |

**ρ=0.75 的 +13.90 与 ρ=0.5 的 −17.10 只隔一个工作点，相差 31 分，且两种子都显著。**
这是"工作点是与 panel 同等重要的自变量"最极端的一例 —— 而且**它不是噪声**：
该 panel 基线虽非单调，**配对差本身两种子高度一致**（差 ≤ 1.2 分）。

**顺带一条复现**：ρ=0.2 两种子都是 **+8.40**，与历史三种子 **+8.07 ± 0.19** 一致。

**但 MultiHop 那个"常数代价 + headroom 跟随"的模型不能外推到这里**：
PrefSuf 在 headroom 仅 +1.80 的 ρ=0.5 上亏 **17 分**，而 MultiHop 在零 headroom 处
只亏约 2.5 分。**⇒ 那个两参数描述是 panel 局部的，不要当成通则。**

### 74. En.Summary 两种子：惰性，唯一 ★ 仍是 ρ=0.1

| ρ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 | **0.1** | 0.05 |
|---|---|---|---|---|---|---|---|
| headroom | −0.19 | +0.04 | −0.05 | −0.08 | +0.34 | **+5.99** | +9.38 |
| 均值 Δ | +0.19 | +0.35 | +0.05 | −0.48 | −0.12 | **+1.03★★** | −0.17 |

两种子一致到 0.3 分以内。唯一显著格是 ρ=0.1 的 **+1.03（headroom 的 17%）**；
headroom 更大的 ρ=0.05（+9.38）拿 −0.17。**又一个"headroom 大不等于能恢复"的例子。**

### 75. **safeguard × 分配范围 2×2 合上：崩溃全部来自「逐层均匀」，Ada-KV 的 safeguard 本身中性到有害**

`_plyr01`/`_plyr02` 完成（`--level pair-layer` = 逐层均匀且**无** safeguard，
因 `score.py:104` 只在 level 含 `adakv` 时给 `safeguard=0.2`）。
Retr.KV，n=100，全部对同 ratio 的 `pair` 基线逐样本配对，满缓存 68.20。

**ρ=0.1（pair 基线 32.60，headroom +35.60）**

| | 无 safeguard | 有 safeguard |
|---|---|---|
| **全局 pair（111 维）** | 32.60（基线） | **66.20 = +33.60★**（地板 b_min=8） |
| **逐层均匀（84 维）** | **1.60 = −31.00★** | 1.80 = −30.80★ |

**ρ=0.2（pair 基线 45.20，headroom +23.00）**

| | 无 safeguard | 有 safeguard |
|---|---|---|
| **全局 pair** | 45.20（基线） | **72.20 = +27.00★** |
| **逐层均匀** | **8.40 = −36.80★** | 2.40 = −42.80★ |

**行内 safeguard 的净效应（唯一干净的 safeguard 消融）：**

| ρ | adakv-layer − pair-layer | 判定 |
|---|---|---|
| 0.1 | **+0.20** [+0.00,+0.60] | **不可分** |
| 0.2 | **−6.00** [−8.60,−3.60] | **★ 显著有害** |

**三条结论，都改写了先前的记载：**

1. **`adakv-layer` 的崩溃与 safeguard 无关，全部来自「把分配限制成逐层均匀」。**
   ρ=0.1 上 `pair-layer` 单独就是 1.60，与带 safeguard 的 1.80 不可分。
   ⇒ 上一轮标注的"该格改两件事、无法归因"现已归因完毕：**是 (b) 不是 (a)。**
2. **Ada-KV 的 safeguard 在本 harness 上中性到有害**（ρ=0.2 上 −6.00★）。
   这是对先验技术的一条有对照支持的负面结果，不是我们方法的优势宣称。
3. **地板的收益需要两个条件同时成立**：**全局跨层分配** ＋ **小幅下限**。
   我们的 b_min=8 挂在全局分配上拿 +33.60/+27.00；Ada-KV 的
   `n_safe = 0.2·ρ·k_len`（比 8 大若干数量级；本轮未实测其绝对值）
   挂在逐层均匀上拿 +0.20/−6.00。**"零件都是先验技术、组合不是"这条表述
   现在有了 2×2 支撑，而不再只是推断。**

**一个能把强度维度也接上的旁证**：Retr.KV@0.2 的地板族 8/32/128 互相不可分，
而 **512 → +17.00★，比 128 低 6.40★**。Ada-KV 的 `n_safe` 在这个工作点估计落在
128 与 512 之间，**正好是我们测到地板开始变差的区间** —— 与 ρ=0.2 上那个 −6.00★
方向一致。**但这是量级推断不是实测**（`k_len` 未从日志读出），写论文前要实测 `n_safe`。

**顺带一条独立于 safeguard 的强结论**：**仅仅把分配从全局改成逐层均匀，
就损失 31~37 分**（1.60 / 8.40 vs 基线 32.60 / 45.20）。
这与此前"增益全在层内×跨层交互"（层内 +2.20 ns、跨层 −7.80★、完整 +25.80★）
指向同一件事 —— **跨层分配自由度是承重的**。两者方法不同（那次是把*学到的增量*
投影到子空间，这次是约束*基线分配方案*本身），结论一致。

### 76. **撤回「地板随 headroom 单调」：它在 ρ=0.05 上同样崩掉（第 14 条撤回）**

`_flr005` 完成。Retr.KV，n=100，满缓存 68.20：

| ρ | 基线 | headroom | 地板 b_min=8 | 95% CI | 恢复率 | 地板绝对 | 学习臂 |
|---|---|---|---|---|---|---|---|
| 0.3 | 65.40 | +2.80 | **+6.00** | [+3.20,+9.00]★ | 214% | 71.40 | −3.80 ns |
| 0.2 | 45.20 | +23.00 | **+27.00** | [+21.60,+32.60]★ | 117% | 72.20 | +21.60★ |
| 0.1 | 32.60 | +35.60 | **+33.60** | [+28.80,+38.20]★ | 94% | 66.20 | +4.20★ |
| **0.05** | **2.00** | +66.20 | **+0.40** | **[+0.00,+1.00] 不可分** | **1%** | **2.40** | +0.80 ns |

**⇒ 撤回清单第 14 条**：两轮前写的"**地板随 headroom 单调上升**、与学习臂的单点尖峰
性质不同"**说过头了**。地板确实覆盖更宽的工作带（0.3/0.2/0.1 三个 ratio 都 ★，
而学习臂实质只有 ρ=0.2 一个点），**但两者在 ρ=0.05 上一起归零**
（地板 +0.40、学习臂 +0.80，都不显著）。**正确表述是"更宽的工作带"，不是"单调"。**

**一个物理读法，而且它给「没有东西可挪」这个说法划出了正确的边界**：
ρ=0.05 上基线绝对分只有 **2.00 / 68.20** —— 缓存已被摧毁到答案不存在于任何
保留集里。地板与学习表都是**固定预算下的重分配**，当总预算小到没有任何分配能保住答案时，
重分配自然无效。**注意这不复活先前那条撤回**：ρ=0.1 上"没有东西可挪"是**错的**
（地板在那里拿 94%），而 **ρ=0.05 上它是对的**。
⇒ **重分配在本 panel 上的有效下界落在 0.05 与 0.1 之间。**

**一条顺带的**：ρ=0.3 上地板绝对分 **71.40 > 满缓存 68.20**（恢复率 214%）
—— 又一个压缩胜过满缓存的实例，与 DBTrimKV 的注意力稀释定理同向。

**缺失的协变量**：`_flr005`/`_flr005b` 发作业时 dump 位传了 `-`，所以**这两格没有
饿死率读数**。若要论证"ρ=0.05 上饿死已被清零但仍无用"（那会是比现在更强的陈述），
需要重跑带 dump 的版本。**目前只能说结果为零，不能说机制。**

**已排后续**：`_vtflr005`/`_vtflr005b`（MultiHop @0.05，n=90，带 dump）——
那是**学习臂唯一在长上下文 panel 上真正生效的点**（+8.20★★ / 94%），
地板在那里成不成，直接决定"地板 ≥ 学习方向"能否再扩一个 panel。

### 77. **撤回上一轮的重构：不是「结构 vs 门控」，六格全部由「用了哪个配额向量」解释**

`_exppair02` 完成，gate × level 2×2 合上（Retr.KV @ρ=0.2，n=100，对 `pair` 基线配对）：

| | `level=pair` | `level=adakv-layer` |
|---|---|---|
| **fastkvzip 门控** | **45.20** | 2.40（−42.80★） |
| **expect 门控** | **1.60（−43.60★）** | 1.40（−43.80★） |

**我的预测错了。** 上一轮写的是"若 expect@pair ≈ 45 则门控不重要、全是结构"，
实测 expect@pair = **1.60**。按这张表单独看，**门控与 level 的主效应各值约 43 分**，
而两者都改只到 1.40 —— 是"改任何一个都塌"的结构，**不是可加分解**。

**但把它与移植 2×2 放在一起，六格由一个变量统一解释 —— 配额向量本身：**

| 排序来自 | 配额来自 | 绝对 |
|---|---|---|
| FKV | **FKV 的 pair 配额** | **45.20** |
| **Exp** | **FKV 的 pair 配额**（精确移植） | **44.20**（−1.00 不可分） |
| FKV | Exp 的配额（精确移植） | 2.60 |
| FKV | FKV 分数 + **逐层均匀**池化得到的配额 | 2.40 |
| Exp | Exp 的 pair 配额 | 1.60 |
| Exp | Exp 的 adakv-layer 配额 | 1.40 |

**规律**：**只要用 FastKVzip 的 pair 配额，两种排序都拿 44~45；
只要换成任何别的配额，两种排序都塌到 1.4~2.6。**
⇒ **决定性变量是配额向量，排序几乎无关。** 这不但没推翻移植 2×2，
反而把它从 4 格扩到 6 格，**并且解释了上一轮那个看似矛盾的 `_adakv02`**：
逐层均匀池化即使喂给它 FastKVzip 的好分数，产出的**配额**也是坏的（2.40）。

**因此撤回上一轮的措辞（撤回 15）**：~~那 43.80 分差距主要归因于驱逐结构这个粗粒度
配置选择~~。**正确表述**：配额向量 = f(分数, 池化方案)；
**只有「FastKVzip 分数 + 全局池化」这一个组合产出好配额**，
换分数或换池化都不行，而**给定配额后排序基本不影响结果**。

**一个必须同时说的限定**：`expect` 门控在 ρ=0.2 上本来就崩（其规范配置 1.40），
所以"Exp 的配额坏"与"Exp 这个方法在此工作点整体失效"在本数据下**分不开**。
要分开需在 Expected 有真实性能的 ρ=0.4（41.00）/0.5（56.00）上重做整张表 ——
**这仍是最值得排的下一个实验**（已挂了很多轮，一直没排上卡）。

### 78. `n_safe` 已从已有 dump 反推出来（此前标为未实测）

不用新作业。PrefSuf 的配额 dump 给出：每 chunk 总预算 99,965、头组数 112、
有效 ratio 0.0660 ⇒ **每 (层,头) 可驱逐条目 `k_len ≈ 13,533`**（与 chunk 16000 相符）。
代入 `n_safe = int(int(k_len·ρ)·0.2)`：

| ρ | Ada-KV 的 `n_safe` | 我们的地板族 |
|---|---|---|
| 0.1 | **≈ 270** | 8 / 32 / 128 互不可分 |
| 0.2 | **≈ 541** | 8/32/128 互不可分，**512 → +17.00★（比 128 低 6.40★）** |

⇒ **ρ=0.2 上 Ada-KV 的 safeguard 强制每头保留约 541 条，正落在我们实测地板
开始变差的 512 之上**；而我们测到的 safeguard 净效应恰是 **−6.00★**，
与地板族 512-vs-128 的 **−6.40★** 数量级与方向都吻合。
**⇒ 上一轮标注的"量级推断、未实测"现已量化，且与独立测量自洽。**
**限定**：`k_len` 由 PrefSuf 的 dump 推得后套用到 Retr.KV（两者同为 chunk 16000，
但不是同一批数据），是近似而非逐样本实测。

### 79. Retr.KV@0.05 地板族补齐：b_min=32 同样为零

| 臂 | Δ | 95% CI | 绝对 |
|---|---|---|---|
| b_min=8 | +0.40 | [+0.00,+1.00] 不可分 | 2.40 |
| b_min=32 | +0.40 | [−0.40,+1.20] 不可分 | 2.40 |

（基线 2.00，满缓存 68.20。）**两点都为零，撤回 14 得到加强**：
ρ=0.05 的失效不是 b_min 取值问题。已排 `_flr005c`（带 dump）补饿死率协变量。

### 80. **MultiHop@0.05：学习方向明确胜过地板（92~96% vs 59%）—— 加上全部九格的对照记分牌**

`_vtflr005`/`_vtflr005b` 完成。MultiHop @ρ=0.05，满缓存 41.07、基线 32.36、
headroom **+8.71**、n=90（**1.11 分/条**）：

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| 学习臂 s0 | +8.00 | [+5.87,+10.13]★ | 40.36 | **92%** |
| 学习臂 s1 | +8.40 | [+6.31,+10.53]★ | 40.76 | **96%** |
| 地板 b_min=8 | +1.29 | [−0.44,+3.02] 不可分 | 33.64 | 15% |
| **地板 b_min=32** | **+5.11** | [+2.98,+7.25]★ | 37.47 | **59%** |

协变量：饿死 **39.12% → 0.00%**，搬动 **2.98%** 预算。
（注意该 panel 基线饿死率明显低于其他 panel 的 55~72%。）

**⇒ 这是第二个「学习方向必需」的工作点**（第一个是 PrefSuf@0.2）。
地板在这里也起作用（b32 拿 59%★），但**明显不及学习臂的 92~96%**。
且**地板族在此单调上升**（8→32：15%→59%），与 Retr.KV 上 b_min=8 已足够不同。

**地板 vs 学习方向的完整记分牌（九个 (panel, ratio) 格）：**

| panel, ρ | 学习臂 | 最佳地板 | 胜者 |
|---|---|---|---|
| Retr.KV @0.3 | −3.80 ns | **+6.00★ (214%)** | 地板 |
| Retr.KV @0.2 | +21.60★ (94%) | **+27.00★ (117%)** | 地板 |
| Retr.KV @0.1 | +4.20★ (12%) | **+33.60★ (94%)** | **地板，差距悬殊** |
| Retr.KV @0.05 | +0.80 ns | +0.40 ns | 平（都死） |
| **PrefSuf @0.2** | **+8.40★ (78%)** | 37% | **学习** |
| PrefSuf @0.1 | +3.00★ (7%) | **+12.40★ (30%)** | 地板 |
| MultiChoice @0.1 | +4.17 ns | +2.78 ns | 平（都不显著） |
| En.QA @0.1 | +1.35 ns | +2.07 ns | 平（都不显著） |
| **MultiHop @0.05** | **+8.20★★ (94%)** | +5.11★ (59%) | **学习** |

**⇒ 地板胜 4、学习胜 2、平 3 —— 两者都不占优。**
先前几轮里"地板 ≥ 学习方向"与"方向在 X 上必需"这两种说法都只在各自的子集上成立；
**正确的总结是：哪一个更好由 (panel, ratio) 决定，且目前没有可预测它的量。**
这与「Retr.KV 的 94% 五候选全排除仍无解释」是同一个未解释项。

### 81. **seed1 的复现检查也通过：逐样本 100/100，+20.00 精确重现**

| | Δ vs 同批基线 | 95% CI |
|---|---|---|
| 历史 `_d10scalar_s1` | **+20.00** | [+15.40,+24.60] |
| 新跑 `_sc11_s1` | **+20.00** | [+15.40,+24.60] |
| **逐样本完全相同** | **100/100** | |

⇒ **用户设定的复现判据两个种子全部通过**（s0 +21.60、s1 +20.00，各自 100/100）。
连同第 69 条，评测确定性与「四臂表不受 `--ctrlm_mode` 默认值 bug 影响」都已实测坐实。

### 82. gate × level 的 ρ=0.1 行：与 ρ=0.2 同型

| 配置 | 绝对 | Δ vs pair 基线 |
|---|---|---|
| fastkvzip / pair | 32.60 | — |
| fastkvzip / adakv-layer | **1.80** | −30.80★ |
| **expect / pair** | **1.80** | −30.80★ |

**两个"只改一件事"的格子给出逐位相同的 1.80** —— 与 ρ=0.2 上 2.40 / 1.60 的图像一致：
**换分数或换池化，任何一个都足以把配额毁掉，且毁掉的程度相同。**
支持第 77 条的统一解释（决定性变量是配额向量）。

**已排 ρ=0.4/0.5 的四个作业**（`_exppair04/05` = expect/pair，`_adakv04/05` =
fastkvzip/adakv-layer）—— 在 **Expected 有真实性能的工作点**（41.00 / 56.00）
重做这张表，以拆开「Exp 配额坏」与「Exp 在崩溃区整体失效」。这是挂了很多轮的实验。

### 83. Code.RepoQA：惰性；ρ=0.05 是静默退化第三例；ρ=0.1 是迄今最大的「有 headroom 却拿不到」

`_sc11_scbench_repoqa_s0` 完成。满缓存 58.64，n=88（**1.14 分/条**）：

| ρ | 0.75 | 0.5 | 0.4 | 0.3 | 0.2 | **0.1** | 0.05 |
|---|---|---|---|---|---|---|---|
| 基线 | 58.18 | 60.91 | 59.55 | 59.09 | 57.73 | **12.95** | 2.50 |
| headroom | +0.45 | −2.27 | −0.91 | −0.45 | +0.91 | **+45.68** | +56.14 |
| Δ | **+2.73★** | −1.36 | +0.23 | −0.00 | −0.23 | +1.59 ns | ⌀ |

**三条：**

1. **ρ=0.05 标 ⌀ —— 静默退化第三例**（前两例 ManyShot、以及这次）。
   日志逐条确认：`clen=65607 window=3280 chunk_ratio=0.000000 degenerate=True`，
   **88/88 条样本**。`0.05 × 65607 = 3280 < window 4096` ⇒ 构造性 no-op。
   Δ 恰为 `+0.00 [0,0]` 的指纹又一次奏效，**「先 grep degenerate 再下结论」这条规矩值回票价**。
2. **ρ=0.1 是目前非退化长上下文格里 headroom 最大的一个（+45.68），学习臂只拿 3%（不显著）。**
   与 Retr.KV@0.1（headroom +35.60、12%★）构成又一组对照：**都是长上下文、满样本、
   headroom 巨大，恢复率 3% vs 12%**，再次说明 headroom 不预测恢复率。
   该 panel 在 0.2→0.1 之间有个悬崖（57.73 → 12.95）。
3. **ρ=0.75 的 +2.73★ 是 headroom 符号律的第三个真反例**（headroom 仅 +0.45）。
   与 MultiHop 的两个（headroom +0.71）同型：**|headroom| ≲ 1 时符号律无预测力**，
   这条限定现在有三个独立实例。

**已排**：`_vtflr005c`/`_vtflr005d` = MultiHop@0.05 地板 **b_min=64/128**。
理由：该点地板族**单调上升**（8→32 给 15%→59%），而 MultiHop@0.05 是仅有的两个
「学习方向胜出」格之一（学习 92~96%）。**若更大的 b_min 追平 94%，
则「MultiHop 上方向必需」这条就被证伪** —— 这是对我们自己结论的直接攻击性检验。

### 84. **六格饿死率全表：饿死缓解假说彻底钉死，且「搬动量」与收益反相关**

`_flr005c`（Retr.KV@0.05 地板 b_min=8，带 dump）完成，补上先前自己造成的协变量缺口。
现在六个 (panel, ratio) 格都有配额 dump，**全部被同一个算子把饿死清零**：

| panel @ρ | 基线饿死 | 地板后 | 搬动预算 | 地板效果 |
|---|---|---|---|---|
| MultiHop @0.05 | **39.12%** | 0.00% | 2.98% | +1.29 ns (15%) |
| MultiChoice @0.1 | 55.68% | 0.00% | 1.14% | +1.39 ns (7%) |
| En.QA @0.1 | 56.53% | 0.00% | 1.17% | +2.07 ns (10%) |
| **Retr.KV @0.1** | 64.70% | 0.00% | **0.47%** | **+33.60★ (94%)** |
| PrefSuf @0.1 | 71.78% | 0.00% | 1.34% | +7.80★ (19%) |
| **Retr.KV @0.05** | **81.58%** | 0.00% | **3.35%** | **+0.40 ns (1%)** |

（均为 b_min=8；预算逐格守恒，如 Retr.KV@0.05 的 44,378 = 44,378。
Retr.KV@0.05 基线的**每头中位配额是 0** —— 一半以上的头颗粒无收。）

**两条，都比先前的三格版本强：**

1. **饿死率与收益无单调关系，若有也是反的。** 饿死最严重的一格（81.58%）效果最差
   （1%），而效果最好的一格（94%）饿死只有 64.70%。跨度：饿死 39~82%、收益 1~94%。
   **六格全部被清零到恰好 0.00%，结果却差两个数量级** ⇒
   **「消除饿死」既不充分、也不与收益单调相关。** 这是第五次否证，也是样本最全的一次。
2. **「搬动了多少预算」同样不是收益代理，而且是反的**：收益最高的
   Retr.KV@0.1 只搬 **0.47%**，收益最低的 Retr.KV@0.05 搬了 **3.35%（7 倍）**。
   ⇒ 与先前「余弦不是收益代理」并列，**第三个被排除的直觉代理量**
   （余弦、搬动量、饿死率）。

**⇒ 剩下的唯一可查方向是配额的「形状」而非其统计量。** 已排 `_flr01d`
（Retr.KV@0.1 地板带 dump）——那是地板拿 94% 的那一格，目前**唯一没有 dump 的高收益格**。
拿到后可对比各 panel 的配额分布形状（集中度、哪些层/头被抬起），
看 Retr.KV@0.1 是否在形状上与众不同。**这是对「Retr.KV 的 94% 无解释」的下一步攻击。**

### 85. **攻击性检验成功：加大 b_min 后地板追平并超过学习臂，「MultiHop 上方向必需」被证伪**

`_vtflr005c`/`_vtflr005d` 完成。MultiHop @ρ=0.05，满缓存 41.07、基线 32.36、
headroom **+8.71**、n=90：

| 臂 | Δ | 95% CI | 绝对 | 恢复率 |
|---|---|---|---|---|
| 学习臂 s0 | +8.00 | [+5.87,+10.13]★ | 40.36 | 92% |
| 学习臂 s1 | +8.40 | [+6.31,+10.53]★ | 40.76 | 96% |
| 地板 b_min=8 | +1.29 | [−0.44,+3.02] ns | 33.64 | 15% |
| 地板 b_min=32 | +5.11 | [+2.98,+7.25]★ | 37.47 | 59% |
| **地板 b_min=64** | **+8.18** | [+5.78,+10.62]★ | 40.53 | **94%** |
| **地板 b_min=128** | **+9.02** | [+6.80,+11.38]★ | 41.38 | **104%** |

**⇒ 撤回 16：「MultiHop@0.05 上学习方向必需」被自己排的检验证伪。**
地板族在此单调上升，b64 追平、b128 **超过**学习臂。**九格记分牌里两个「学习胜出」
格现在只剩 PrefSuf@0.2** —— 已排 `_psflr02d`/`_psflr02e`（b_min=64/256）对它做同样的检验。
（该 panel 的族此前是 8/32/128 → 28%/37%/**−2%**，峰在 32 且 128 已塌，
所以那里未必重演；**但在测之前不能再声称「方向必需」**。）

**一条必须同时写下的限定，否则会读成「地板更强」**：
**地板的 `b_min` 是逐格用后见之明挑的** —— 同一个 panel 换 ratio、同一个 ratio 换
panel，最优 `b_min` 都不同（Retr.KV@0.2 上 8/32/128 不可分而 512 变差；
PrefSuf@0.2 峰在 32、128 塌到 −2%；**MultiHop@0.05 峰在 ≥128**）。
先前写的「b_min≈32 是唯一跨 panel 跨 ratio 都不差的取值」在这里也**不成立**
（32 只拿 59%，128 拿 104%）。
⇒ **正确表述**：**给定逐格 oracle 选定的 b_min，地板可以匹配或超过学习臂；
但两者都需要「知道这个 workload」，只是所需的先验形式不同**
（一个是标量 b_min，一个是整张配额表）。**这反而加强了主命题**：
真正的游戏是挑对逐头配额，而**不存在与 workload 无关的答案**。

### 86. **ρ=0.4 上 `adakv-layer` 反而最好、还超过满缓存 —— 池化方案的优劣随 ratio 翻转**

`_exppair04`/`_adakv04` 完成。Retr.KV @ρ=0.4（满缓存 68.20，pair 基线 66.40，
headroom 仅 **+1.80**，n=100）：

| 配置 | 绝对 | Δ vs pair 基线 | 95% CI |
|---|---|---|---|
| fastkvzip / pair | 66.40 | — | — |
| **fastkvzip / adakv-layer** | **72.20** | **+5.80** | [+2.60,+9.00]★ |
| **expect / pair** | **1.80** | **−64.60** | [−70.20,−58.60]★ |

**两条，第一条推翻我先前的表述：**

1. **`adakv-layer` 不是「总是崩」—— 在 ρ=0.4 上它是最好的，绝对分 72.20
   还高过满缓存 68.20。** 而在 ρ=0.2 / 0.1 上它是 2.40 / 1.80（−42.80★ / −30.80★）。
   ⇒ **同一个结构选择的符号随工作点翻转**，与先前在**学习臂**上看到的现象同型，
   现在在**结构**上也出现。**先前「逐层均匀约束导致崩溃」这句必须限定到 ρ ≤ 0.2**
   （撤回 17）。这又是一次「在子集上下全称判断」—— 我只测了 0.1/0.2 就概括了。
2. **Expected 的分数在全局池化下崩得更彻底**：ρ=0.4 上 expect/pair = **1.80**，
   而它的规范配置（adakv-layer）在该 ratio 有 41.00。
   ⇒ **门控与池化是耦合的**：Expected 的分数只在逐层均匀下可用，
   FastKVzip 的分数在 ρ≤0.2 需要全局池化、在 ρ=0.4 反而更喜欢逐层均匀。
   **「只有 FKV 分数 + 全局池化产出好配额」这句要限定到 ρ ≤ 0.2。**

**一个直接有用的副产品**：`-g fastkvzip --level adakv-layer` 在 ρ=0.4 上比默认配置
**白拿 +5.80★**，且绝对分超过满缓存。这是**对基线配置的可操作改进建议**，
不涉及任何我们自己的方法。**待 ρ=0.5 的 `_adakv05` 确认是否延续。**

### 87. **更正：`_adakv04` 的 +6.45★ 是读了未跑完的样本，终值是 +5.80★**

03:49 那次读 `_adakv04` 时它**还在 GPU3 上运行**，我按部分样本报了
「绝对 72.26 / Δ +6.45 [+3.01,+9.89]★」。作业完成后（n=100，日志有 `Finished.`）
终值是 **绝对 72.20 / Δ +5.80 [+2.60,+9.00]★**。已在 `JOURNAL.md` 与
`RESULTS_ABLATION.md` 中全部替换。

**结论方向不变**（仍显著为正、仍高过满缓存 68.20），但**这是又一次违反项目自己的
规矩**：`Never trust a ★ on partial samples`。此前只在正结论上守过这条，
这次栽在一个**我很想要的**正结论上。**新规矩：读任何结果前先确认该 tag 的
`n == 基线 n` 且日志有 `Finished.`**，别只看 `qrun_done.log`（它记的是进程退出，
而我这次是在进程退出**之前**读的）。

### 88. gate × level 的完整 ratio 曲线：`adakv-layer` 也是**单点尖峰**

Retr.KV，n=100，对同 ratio 的 `pair` 基线配对（满缓存 68.20）：

| ρ | pair 基线 | headroom | **fastkvzip / adakv-layer** | **expect / pair** |
|---|---|---|---|---|
| 0.5 | 71.60 | −3.40 | *在跑（n=98，未完成，不引用）* | **38.40（−33.20★）** |
| 0.4 | 66.40 | +1.80 | **72.20（+5.80★）** | 1.80（−64.60★） |
| 0.2 | 45.20 | +23.00 | 2.40（−42.80★） | 1.60（−43.60★） |
| 0.1 | 32.60 | +35.60 | 1.80（−30.80★） | 1.80（−30.80★） |

**两条：**

1. **`adakv-layer` 只在 ρ=0.4 上赢**（+5.80★ 且绝对分高过满缓存），
   在 0.2/0.1 上是 −42.80★ / −30.80★ 的灾难。⇒ **它与学习臂同型：单点尖峰。**
   两者的尖峰还不在同一个 ratio（学习臂在 0.2、`adakv-layer` 在 0.4）。
   **「逐层均匀约束导致崩溃」必须限定到 ρ ≤ 0.2**（撤回 17 已记）。
2. **`expect/pair` 在 ρ=0.5 上是 38.40，在 ρ=0.4 上塌到 1.80** —— 相邻工作点之间
   一个 36.6 分的悬崖。⇒ 把 Expected 的分数放到全局池化下，**在 ρ≥0.5 尚可用、
   ρ≤0.4 完全不可用**。这进一步支持「门控与池化耦合」，并且说明
   **主命题里「只有 FKV 分数 + 全局池化产出好配额」这句的适用区间是 ρ ≤ 0.4**。

**已排跨 panel 检验**：`_psadakv04`/`_vtadakv04`/`_mfadakv04`（PrefSuf / MultiHop /
Math.Find 在 ρ=0.4 的 `adakv-layer`）—— 看那个 +5.80★ 是 Retr.KV 独有还是通用。
若通用，则「把 FastKVzip 在 ρ≈0.4 换成 adakv-layer」是一条**不涉及我们任何方法的、
可直接用的基线改进**；若只在 Retr.KV 上，则它与那个未解释的 94% 可能同源。

### 89. **更正：ManyShot@0.2 只有 33% 的样本退化，不该整格作废**

把退化判定从「公式套标注长度」换成「读日志里 runtime 实测的 `degenerate=` 标记」后，
发现同一名义 ratio 下**各样本长度不一、退化与否可以混合**：

| panel | ρ=0.2 | ρ=0.1 | ρ=0.05 |
|---|---|---|---|
| **ICL.ManyShot** | **33% 退化** | 100% | 100% |
| Code.RepoQA | 0% | 0% | **100%** |
| Retr.KV | 0% | 0% | 0% |

⇒ **先前写的「ManyShot ρ≤0.22 三格全是构造性 no-op、标 ⌀」对 ρ=0.2 过头了**
（撤回 18）。那一格三分之二的样本在真实驱逐，所以它有非零 Δ（两种子 +0.56±0.19）
—— 这个非零**不是噪声也不是矛盾**，是那 67% 非退化样本的真实效应。
ρ=0.1 / 0.05 仍是 100% 退化，标 ⌀ 正确。

**表格的标记体系相应改成三档**：`°` 全格退化 / `~` **部分**退化（占比见脚注）/
`?` 标注长度撑不住判定。**`~` 格既不能整格作废、也不能当正常格读。**

**方法学**：二值的「退化 / 不退化」在**样本长度不齐**的 panel 上本来就不成立。
先前之所以没暴露，是因为判定用的是**每 panel 一个标注长度**的公式；
换成逐样本实测后这件事立刻显形。**凡是"按 panel 给一个数"的量，
在样本异质的 panel 上都要先问一句"这个量逐样本是否一致"。**

### 90. gate × level 曲线补完（`_adakv05` 实为已完成，上轮 n=98 是最后两条未落盘）

上一轮判 `_adakv05` 未完成、拒绝引用，是对的做法但判据用早了 —— 它已有
`Finished.` 与 100 个结果目录，只是我读的那一刻最后两条尚未写盘。完整读数：

| ρ | pair 基线 | headroom | **fkv / adakv-layer** | **expect / pair** |
|---|---|---|---|---|
| 0.5 | 71.60 | −3.40 | 68.60（−3.00 不可分） | 38.40（−33.20★） |
| 0.4 | 66.40 | +1.80 | **72.20（+5.80★）** | 1.80（−64.60★） |
| 0.2 | 45.20 | +23.00 | 2.40（−42.80★） | 1.60（−43.60★） |
| 0.1 | 32.60 | +35.60 | 1.80（−30.80★） | 1.80（−30.80★） |

（Retr.KV，全部 n=100 且日志均有 `Finished.`。）

**`adakv-layer` 的单点尖峰由完整数据坐实**：ρ=0.5 与 pair 不可分、**ρ=0.4 显著为正
且绝对分 72.20 > 满缓存 68.20**、ρ≤0.2 塌到 2 分以下。**已排 `_adakv03`（ρ=0.3）**
定位 0.4→0.2 之间那道 48 分的落差是渐变还是悬崖。

**顺带一条操作性说明**：`n < 基线 n` 既可能是"没跑完"，也可能是"刚好卡在写盘中间"。
**判完成要看 `Finished.`，样本数只用来防"读到一半"**；两个都查才不会像上一轮那样
既误报未完成、又像更早那次误读部分样本。

### 91. **攻击最后一格失败：PrefSuf@0.2 上学习方向确实胜过整个地板族**

`_psflr02d`(b64)/`_psflr02e`(b256) 完成，补全该 panel 的地板族（**全部 n=100 且
`Finished.`**）。满缓存 50.00、基线 39.20、headroom **+10.80**：

| 臂 | Δ | 95% CI | 恢复率 |
|---|---|---|---|
| **学习臂 s0 / s1** | **+8.40 / +8.40** | 均 ★ | **78%** |
| 最优自表 γ=2.157 | +12.00 | [+8.60,+15.60]★ | 111% |
| 地板 b_min=8 | +3.00 | [+1.00,+5.00]★ | 28% |
| **地板 b_min=32（族内最优）** | **+4.00** | [+1.00,+7.00]★ | **37%** |
| 地板 b_min=64 | +3.00 | [−0.20,+6.20] ns | 28% |
| 地板 b_min=128 | −0.20 | [−4.00,+3.60] ns | −2% |
| 地板 b_min=256 | −5.80 | [−9.60,−2.20]★ | −54% |

**⇒ 攻击失败，撤回 16 不扩大。** MultiHop@0.05 那格被 b64/b128 反超（94%/104%），
**但同样的攻击在 PrefSuf@0.2 上不成立**：地板族有内部峰（b32，37%），
之后单调劣化，**没有任何一点接近学习方向的 78%**。
⇒ **九格记分牌里「学习方向胜出」还剩这一格，且它经受住了五点攻击。**
"学习到的方向在任何工作点都不必需"这句**不成立**。

**顺带一个数据完整性问题（第三次同类）**：`_psflr32` 只有 **37 个结果目录、无
`Finished.`**（早先被杀），而我最初正是从这 37 条读出 b32 的值。所幸另有完整的
`_psflr32b`（n=100），两者给出的恢复率**碰巧都是 37%**，所以先前记录未出错 ——
**但那是运气**。已全面审计 26 个地板 tag，**只有 `_psflr32` 不完整**，其余全部 n 足且
`Finished.`。

### 92. **配额「形状」也解释不了 Retr.KV@0.1 —— 第六个被淘汰的候选**

`_flr01d`（Retr.KV@0.1 地板带 dump）完成，**逐位复现 `_flr01a` 的 +33.60★**。
拿到它之后，六个格的配额形状可以横向比了（全部为基线配额 `b_base` 的统计量）：

| panel @ρ | 饿死 | 搬动 | **Gini** | **非零头中位配额** | **top10% 头占预算** | 地板收益 |
|---|---|---|---|---|---|---|
| **Retr.KV @0.1** | 64.0% | 0.94% | **0.897** | **432** | **86.5%** | **94%★** |
| PrefSuf @0.1 | 71.8% | 1.34% | 0.910 | 948 | 91.1% | 19%★ |
| MultiChoice @0.1 | 55.7% | 1.14% | 0.891 | 425 | 83.8% | 7% ns |
| En.QA @0.1 | 56.5% | 1.17% | 0.894 | 430 | 84.6% | 10% ns |
| MultiHop @0.05 | 39.1% | 2.98% | 0.951 | 19 | 95.0% | 15% ns |
| Retr.KV @0.05 | 81.6% | 3.35% | 0.960 | 75 | 99.7% | 1% ns |

**关键对照**：Retr.KV@0.1 与 MultiChoice@0.1 / En.QA@0.1 在**三个形状统计量上几乎
一模一样**（Gini 0.897 vs 0.891/0.894、非零头中位 432 vs 425/430、
top10% 占比 86.5% vs 83.8%/84.6%），**而收益是 94% vs 7%/10%——差 13 倍**。

⇒ **配额分布的形状不解释它。这是第六个被淘汰的候选**（前五：上下文长度、样本数、
基线绝对分、任务族、饿死率；加上更早的余弦与搬动量，实际已淘汰八个量）。

**但必须写清这条否证的边界**：我算的三个统计量**都是置换不变的** ——
它们只描述"配额分布长什么样"，**不描述"是哪些 (层,头) 拿到了配额"**。
所以正确结论是：**置换不变的形状统计量不解释它，逐头身份仍可能解释**。
⇒ **下一个探针应比较「被抬起的是哪些 (层,头)」**，而不是再算一个分布统计量。

**一处数字更正**：先前六格表里 Retr.KV@0.1 记的是「饿死 64.7%、搬动 **0.47%**」，
但那一格当时**没有 dump**（`_flr01a` 的 dump 位传了 `-`），那两个数**没有来源**。
现在实测是 **饿死 64.03%、搬动 0.943%**。结论方向不变（仍是六格中搬动最少的，
且与收益反相关：0.94% 拿 94%、3.35% 拿 1%），**但倍数从 7× 改为 3.6×**。

### 93. `adakv-layer@ρ=0.4` **不跨 panel**，那 +5.80★ 是 Retr.KV 独有

| panel | 满缓存 | pair 基线 | adakv-layer | Δ |
|---|---|---|---|---|
| **Retr.KV** | 68.20 | 66.40 | **72.20** | **+5.80★（且超满缓存）** |
| Retr.MultiHop | 41.07 | 40.36 | 39.07 | −1.29 ns |
| Math.Find | 33.17 | 34.00 | 31.33 | **−2.67★（有害）** |

⇒ **"把 FastKVzip 在 ρ≈0.4 换成 adakv-layer 是一条免费的基线改进"这个设想被否**：
它只在 Retr.KV 上成立，在 Math.Find 上**显著有害**。
**而这又是一次 Retr.KV 成为异类** —— 与那个未解释的 94% 同源的可能性上升。
（`_psadakv04` 仍在跑，补第四个 panel。）
