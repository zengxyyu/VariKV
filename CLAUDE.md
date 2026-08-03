# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A research workspace whose core contribution is `FreeEnergyMemory` — a probabilistic memory module storing distributional slots `(μ_k, σ²_k)` that uses variational free energy (ELBO) / KL divergence to gate memory updates.

**Active direction (decided 2026-07-24): KV-cache compression, targeting ICLR 2027.** The idea is being moved from its original home (long-video VLM memory inside InternVL3-8B, framed for NeurIPS 2026) to a new application: a distributional memory that *absorbs evicted KV-cache entries* instead of dropping them, built on top of Fast KVzip and trained with a frozen LLM. **A future Claude should read `HANDOFF.md` first** — it is the execution entry point (what's decided, what not to re-investigate, the reproduce → variance-ablation → build plan). The video framing below and in `free_energy_memory_proposal.md` is the historical origin, not the current goal.

`memory_module.py` is still the **video-era** module (input = video segments); porting it to the KV setting (input = evicted KV) is pending work — see `HANDOFF.md` §4 for the four theory-driven changes required during that port. `memory_module_video_backup.py` is a frozen copy of the video version.

## Two Codebases in This Workspace

1. **The research module** (`memory_module.py`, `stage1/`, and the `.md` docs) — the local, hand-written work. This is what most edits touch. Local code and comments are written in Chinese; match that when editing.
2. **`external/` — vendored upstream clones**, the code base for the active KV direction (each keeps its own `.git`; do not treat this repo as their parent):
   - `external/FastKVzip/` — clone of `github.com/Janghyun1230/FastKVzip`. **This is the intended fork base for Path B.** The real eviction code to modify is `prefill/attention/kvcache.py` (676 lines, gate branch).
   - `external/KVzip/` — clone of `github.com/snu-mllab/KVzip` (Fast KVzip's ancestor). Cleaner API sample; `demo.py` is the clearest end-to-end template.
   - Neither is a dependency of `memory_module.py`; they are read/reproduced/modified per the KV plan. Note: FastKVzip ships **no LICENSE file** — clear code license with the author before publishing anything derived from it.
   - The FastKVzip clone **already carries local edits** (loader fix, MRCR support) — see "Local modifications to the vendored FastKVzip clone" below before re-cloning or pulling.

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
- One `-d all` sweep for one (model, method) ≈ **33 GPU-hours** — measured on **Qwen3-8B**, where `kv`/`mf`/`prefix_suffix` run their short/mid variants. On Qwen2.5-7B-1M the same sweep is substantially more expensive (see the length table above); the in-flight run is the first real measurement. The full paper grid (6 models × 5 methods) ≈ 1000 GPU-hours ≈ 5 days on 8×H100, more with the 14B models.
- Two loaders needed a **pyarrow fallback** (already patched into `data/load.py`): `scbench_mf_mid` and `squad` both die on `Feature type 'List' not found` because their parquet was written by datasets 4.x. Upgrading datasets would break transformers 4.51.3, so the patch reads the parquet directly. All 11 `-d all` datasets load now.
- The default `--kv_type retain` needs no custom CUDA kernel. `--kv_type evict` calls the AdaKV kernel — **already built** (CUDA 12.8 toolkit + `csrc`, verify with `python -c "import torch, tiny_api_cuda"`; importing torch first is required or libc10.so is missing).

### What the paper's experiments actually are

The paper has **20 figures and 4 tables**. Verified against `scratch/refs/fastkvzip_paper.txt` on 2026-08-03 by reading every caption. Only a small subset is worth reproducing for the Path-B project — the map below exists so nobody re-derives it.

**Main results (§4)**

- **Figure 11 = prefill-intensive main result**, on **Qwen2.5-7B-1M** (the `run.sh` default). Ratios 0.2→1.0, 5 methods. The body text (§4.2) says "across **12 datasets**" — the caption's three categories (retrieval-intensive / contextual understanding / high redundancy) are groupings of those 12, not a smaller set. The 12 = 9 SCBench tasks + SQuAD + GSM + **MRCR**; `-d all` covers only the first 11.
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

### Local modifications to the vendored FastKVzip clone

`external/FastKVzip/` keeps its own `.git` and has no LICENSE file. Local edits made so far — re-apply if it is ever re-cloned:

| File | Change |
|---|---|
| `data/load.py` | pyarrow fallback for `scbench_mf_mid` and `squad` (datasets 4.x wrote their parquet; `Feature type 'List' not found` under datasets 3.6.0) |
| `eval_chunk_mrcr.py` | honour `--num`; print `Finished.` |
| `eval_mrcr.py` | **new file** — unchunked MRCR eval for the KVzip baseline |

### Reproduce status

**Figure 11, 3 of 12 datasets: qualitatively consistent with the paper. Not a numeric reproduction — and cannot be, see the caveat above.** Run 2026-07-31: `Qwen2.5-7B-Instruct-1M` × **all 5 methods** × 3 datasets (`squad`, `scbench_many_shot`, `scbench_prefix_suffix`) × 6 ratios, 100 samples each (54 for many_shot) — 15 jobs, 2.9h on 8×H100, 0 failures. Raw numbers in `scratch/repro_0731_qwen25/fig11_results.log`.

What matches: at ratio **0.3** fastkvzip scores 100.1 / 95.1 / 103.2 relative, satisfying the paper's "near-lossless at a 30–40% budget"; and the method ordering follows Figure 11 — fastkvzip ≥ kvzip > duoattn > expected ≫ **snapkv, which collapses** (36.5 on squad, 7.6 on prefix_suffix). The 3 datasets happen to sample one from each of Figure 11's three categories (prefix_suffix → retrieval, squad → contextual understanding, many_shot → high redundancy), which makes the subset more informative than a random 3.

Two things to state whenever these numbers are cited: (1) **no point-by-point check against the paper was performed**, because the paper publishes no numeric table for Figure 11 — the agreement above is ordering and threshold behaviour only; (2) `scbench_prefix_suffix` is **very noisy** and non-monotonic even for the winning method (73.6 at ratio 0.75 but 115.6 at 0.4), so per-point gaps on that dataset are not meaningful at n=100 — do not build an argument on the large fastkvzip-vs-baseline margin there without more samples.

**Full Figure 11 sweep: launched 2026-08-03, in flight.** The remaining 8 datasets × 5 methods (40 jobs) plus MRCR × 5 methods, on Qwen2.5-7B-1M. Driver `scratch_fig11_driver.sh` runs the MRCR stage and the 12-dataset parse automatically when the main scheduler exits; results accumulate in `scratch_fig11_full_results.log`, scheduler trace in `scratch_fig11_full_run.log`. Estimated ~15h for the main sweep + 3–4h for MRCR, extrapolated from the first 12 minutes of progress — treat as an estimate, not a measurement. Early health check passed on all 8 datasets including the two risky ones: `scbench_kv` prefills 169,035 tokens at 9.7 GB KV cache without OOM, and `scbench_mf` loads at full 147k length.

Earlier 2026-07-30 run on **Qwen3-8B** (wrong model for Figure 11, kept only as a Figure-12 data point): relative performance at ratio 0.3 was 89.9 / 93.6 / 101.7 on `scbench_kv_short` / `scbench_prefix_suffix_short` / `scbench_many_shot`, with unexplained retrieval collapse at ratio 0.2 (28.7 / 12.8) where Figure 12 shows ~0.6–0.8. That run also submitted `scbench_mf`, which **failed** on the datasets-4.x parquet error; the pyarrow fallback was written afterwards. So it covered 3 datasets, single-method, and is superseded by the Qwen2.5 runs.

**Do not mistake stray `results/` directories for reproductions.** `gsm`, `scbench_repoqa`, `scbench_summary`, `scbench_vt`, `scbench_choice_eng`, `scbench_qa_eng`, `scbench_mf_mid` each contain only ~2 result dirs — those are the 2-sample timing probes from `scratch/probe/timing_probe.sh` / `scratch/probe/fig11_probe.sh`, not eval runs.

Timing measured on Qwen2.5-7B-1M, seconds/example over all 6 ratios: `squad` 12.8 (203-token context), `scbench_many_shot` 10.2 (26k), `scbench_prefix_suffix` 93.8 (112k).

## Stage 1 — the GO/NO-GO variance ablation

`stage1/data.py` builds the synthetic dataset for the make-or-break experiment (distributional vs. point absorption of evicted KV). It is standalone — no dependency on `memory_module.py` or `external/`.

The design is load-bearing, not arbitrary: samples come in **two kinds that must both be present**. `retain` (fact appears early, only same-format distractors after it → tests low KL ⇒ *don't* update ⇒ resist washout) and `update` (the fact is genuinely rewritten mid-context → tests high KL ⇒ *do* update). A fixed-rate point memory cannot be optimal on both at once; testing only one kind lets a tuned fixed rate tie, i.e. a false negative. The `n_distract` axis (0/200/800/2000) is the predicted-effect-size knob: advantage should grow with distractor count and vanish near 0.

```bash
.venv/bin/python stage1/data.py          # print n_distract → token-length table only
.venv/bin/python stage1/data.py build    # write stage1/{train,val}.jsonl
```

`build` emits 3200 train / 400 val over the 4 distractor levels. Measured context lengths: `n_distract` 0 → 109 tokens, 200 → 3,526, 800 → 13,817, 2000 → 34,357. Neither jsonl is committed — regenerate as needed (seeds are fixed: train 0, val 1234).

## `varikv/` — the method implementation (written 2026-08-03)

VariKV — variational free-energy eviction. Implements the §11 "Option 3 unified free-energy eviction" method. Standalone — deliberately **not** wired into Fast KVzip yet (see "deviations" below).

The four-tier killer ablation (§11.7) is **two orthogonal switches, not four codepaths** — `Config.ablation(tier)` sets them:

| tier | `evict_policy` | `absorb_mode` | degenerates to |
|---|---|---|---|
| 1 | — | `discard` | KVzip / FastKVzip |
| 2 | `recency` | `point` | Infini-attention / Tensor Cache |
| 3 | `free_energy` | `point` | IndexMem-like |
| 4 | `free_energy` | `dist` | **this method** |

The 5th row of the §11.3 degeneracy table (drop the KL term → Expected Attention) needs no new tier: set `free_energy.lam = 0`. **This degeneracy is empirically verified**, not just asserted — at λ=0 the rank correlation of F with D is 1.000 and with KL is 0.056. Sweeping λ traces the rate-distortion working points (0.3 balances the two terms, hence the default; 3.0 makes KL dominate). That sweep is a ready-made sensitivity analysis for the paper.

```bash
.venv/bin/python varikv/train.py --tier 4            # then 2, 3
.venv/bin/python varikv/evaluate.py --tier 1 2 3 4
```

Files: `config.py` (two switches + all hyperparameters), `memory.py` (slots, precision-weighted update, read-out), `free_energy.py` (F_i, expected attention, amortised predictor), `cache.py` (chunked prefill → evict → absorb → read back), `rope.py` (inverse/forward rotation — see below), `train.py`, `evaluate.py`.

`point` and `dist` share **identical structure and parameter count** (verified equal) and run the same precision-weighted update — the only difference is whether the precision term carries information. That is what keeps the ablation clean: the sole independent variable is "does variance help".

### Bugs that only surface at real scale — do not "simplify" these back

Each of these let the code run, the loss fall, and numbers come out, while **silently not running the method described in the paper**. All were invisible on a tiny random model and only appeared at `d_latent=64`, `K=16`, real KV.

1. **`logvar_init` must be ≥0, not −2.** An empty memory has absorbed nothing and must be *uncertain*; it becomes certain (and thus overwrite-resistant) only as precision accumulates. Starting at −2 both makes the empty memory refuse its first write and inflates initial KL by 1/σ_p² to ~10³.
2. **Precision needs a forgetting factor** (`precision_decay=0.95`). `τ_new = τ_old + Σηᵢτ_obs` grows without bound in a streaming setting → memory becomes so confident that μ stops updating → it rejects all new information, which fails every `update`-type sample by construction.
3. **The write gate must use a chunk-wise z-score of KL, not raw KL.** Raw KL spans ~4 orders of magnitude as memory evolves (0.05 → 1589 measured), so any fixed (α, β) saturates: gate ≡ 0.12 early, ≡ 1.00 later, std ≈ 0 at both ends. The gate never operates in its sensitive band and `dist` silently degrades to unconditional full writing. After the fix, low- vs high-surprise write rates are 0.047 vs 0.994. Note `eta_beta` is consequently **0.0**, not the 2.0 that suited raw KL.
4. **The write gate must be probability-normalised: `gate_ik = w_ik · η_i`.** Folding allocation and strength into one sigmoid let the row sum reach **K** (measured 0.66–16.0) — one observation writing at full strength into *every* slot, i.e. the same information counted K times. That breaks "independent observations have additive precision", the premise the whole Bayesian update rests on. Decoupled: `η_i` (scalar, how much this observation writes in total) × `w_ik` (softmax allocation, sums to 1) ⇒ row sum = η ≤ 1, with discrimination preserved (0.03 vs 0.99).
5. **Memory keys must be stored pre-RoPE.** See the dedicated section below — this one threatens the method's core claim, not just its accuracy.
6. **Normalising F by scale alone is not enough — it must be by *spread*.** Ranking is driven by dispersion, not by means. After dimensional normalisation alone, `std(D_n)` ≈ 0.69 constant while `std(KL_n)` grows 2e-4 → 7e-2, so F's ranking was 99% determined by D (rank corr with KL only 0.09) even though KL's *mean* had long overtaken D's. That is exactly the Expected-Attention degeneracy. Fix: divide each term by its own **running** std (`f_normalize="running"`). Running stats are a dataset-level quantity, so unlike a per-chunk z-score they keep `F_i` a function of `KV_i` and the memory state, preserving λ's meaning as the rate-distortion Lagrange multiplier.

General lesson: eviction only ever uses the **ranking** of F, so any monotone transform is free — that is what licenses all the rescaling above. But *which* statistic you normalise by matters: per-chunk z-scoring is numerically safest yet forfeits F's absolute semantics, while running stats keep them at the cost of some lag (hence `v_scale_momentum=0.9` as a compromise; 0.95 lets KL dominate, 0.5 balances best but collapses back toward per-batch statistics).

### RoPE: memory keys must be stored pre-RoPE (fixed 2026-08-03)

This is the most consequential fix so far, and its consequence is **semantic, not just numerical**.

Cached keys are post-RoPE (verified in the vendored source: `prefill/attention/attn.py:54` applies `apply_rotary_pos_emb` before `update()` at line 81). A memory slot is a precision-weighted *average* of absorbed keys, and RoPE does not distribute over addition:

> `α·R_p k + (1−α)·R_p' k' ≠ R_φ(α·k + (1−α)·k')`  — MemRoPE, arXiv 2603.12513

So a slot built from post-RoPE keys **is not a valid key at any position**. Measured consequence: the same slot's inner product with a query swung between −17 and +13 as query position went 0 → 16384, with no pattern. And an *untrained* memory was already flipping the top-1 prediction on 5 of 7 answer tokens — it was winning attention through unconstrained logit magnitude, not content.

The deeper problem: mixing phases **inflates σ² for reasons unrelated to epistemic uncertainty**. Without this fix, the variance is not measuring what the paper claims it measures, and HANDOFF red line 1 ("variance must be functional") fails — a reviewer could fairly call it Bayesian paint over phase noise.

Fix (`varikv/rope.py`, the standard solution in the literature): inverse-rotate each evicted key by `R_p^{-1}` back to a position-free frame before absorbing, and re-rotate at read time to the slot's **position centroid** (the UPL optimum in EPL, arXiv 2409.14364; same rule derived independently in MemoSight). Because `R(δ)R(p) = R(p+δ)`, re-rotation is pure algebra — no extra forward pass. Precedents: Still (2606.07878 §2.2) "position-free frame", Landmark Attention (2305.16300), StreamingLLM (2309.17453). Verified: our rotation matches HF's `apply_rotary_pos_emb` exactly (0.00e+00), round-trip error 2.4e-7, float positions supported (centroids are not integers). Side effect: tier 2's `lm_loss` went from *above* tier 1 (5.574 vs 4.965) to *below* it (4.174 vs 4.381) — untrained memory stopped being a pure noise source.

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

- **The mixture ELBO is missing the component-assignment KL.** With data-dependent responsibilities `w_k`, the full ELBO also carries `KL(w‖π) = log K − H(w)`; the code has only the weighted sum of conditional KLs. (The `Σ_k w_k KL(q‖p_k)` bound itself is correct — verified by Monte Carlo: true 6.694 ≤ bound 8.535.)
- **"Bayesian surprise" is used loosely.** The code computes `KL(q(z|e)‖p(z|M))` — how much an observation disagrees with memory. Itti & Baldi's Bayesian surprise is `KL(posterior‖prior)` — how much memory *changed*. Related but not the same; be precise when citing.
- **Memory capacity may be too small to show an effect.** Compression ratio 377:1 (3.5k context) to **4231:1** (34k), and the read-out contributes only `K·T` = 16 effective KV against a budget of 512 — **3% of visible KV**. Raising `K` costs almost no parameters (slot inits are only `K×d_z`), so this is a cheap knob if the tiers come out indistinguishable.
- **bf16 numerics.** Precision accumulation showed 5.98% relative error in bf16 (values are large, mantissa is 8 bits), and `logvar` was observed pinned at the −4 clamp where its gradient is zero — monitor the fraction of slots at the boundary.

Two faithfulness gaps *were* closed on 2026-08-03:

- **ELBO reconstruction now lives in attention-output space** (closed 2026-08-03). It was `‖ê−e‖²` in KV space while `F_i` used `ā²·‖v−v̂‖²` — so "free energy" named two quantities with *different distortion definitions*, and §11.1's "one scalar unifying both decisions" was merely nominal. `absorb()` now takes `expected_attn` and weights reconstruction by `N·ā`, matching `F_i`'s distortion and realising rate-distortion's "spend bits by importance". Verified: identical to unweighted when attention is uniform (correct — no importance signal available), diverging up to ~8% as it sharpens. Side benefit: the aux loss dropped from ~90 to ~8–29, since unattended KV no longer inflates reconstruction error.
- **The F predictor now sees a memory summary** (closed 2026-08-03). Its target contains `KL_i`, which depends on the current slots, so without memory state it could only learn an average-case KL. Feeding it exact information is self-defeating — that needs an `encode`, which is precisely what amortisation exists to avoid. The compromise is a pooled summary (slot `μ` mean ⊕ `logvar` mean, `2·d_latent` dims, near-zero cost). Verified the prediction now shifts as memory evolves.

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

Working artifacts — evidence for the results recorded above, not part of the research module. Finished phases live under `scratch/` (see `scratch/README.md`); the in-flight run's files stay at the repo root until it completes. Reorganised 2026-08-03.

**`scratch/` — archived, one directory per phase**

| Path | Contents |
|---|---|
| `scratch/install/` | venv/CUDA install logs and scripts, the flash-attn torch2.6 wheel, `reqs_fixed.txt` |
| `scratch/repro_0730_qwen3/` | 2026-07-30 first pass (Qwen3-8B, fastkvzip only, 3 datasets; `scbench_mf` failed on the parquet bug). Superseded |
| `scratch/repro_0731_qwen25/` | 2026-07-31 second pass (Qwen2.5-7B-1M × 5 methods × 3 datasets). `fig11_results.log` is the parsed table, `fig11_run.log` the scheduler trace. Its `fig11_parse*.sh` are superseded by `scratch_fig11_driver.sh` |
| `scratch/probe/` | 2-example timing probes. **Run one before committing GPU-days to a new (model, dataset) pair** — cost is not inferable from context length. Their leftovers in `results/` are ~2 dirs per dataset, not eval results |
| `scratch/refs/` | `fastkvzip_paper.txt` — the only local copy of the paper, PDF-extracted (figure data points lost) |

**Repo root — the in-flight Figure 11 sweep.** Do not move these while it runs: the scheduler is invoked by relative path, bash reads the driver by byte offset, and two of them are open stdout handles.

- `scratch_repro_full.py` — the scheduler (also the reproduce driver generally; not throwaway despite the prefix)
- `scratch_fig11_driver.sh` — end-to-end driver (wait → MRCR → parse 12 datasets)
- `scratch_fig11_full_run.log` / `scratch_fig11_full_results.log` — scheduler and driver stdout
- `scratch_repro_full_logs/` — per-job logs plus `.done__*` completion markers

When the sweep finishes, fold these into `scratch/repro_0803_fig11_full/`. Keep new throwaway scripts on the `scratch_` prefix at the root, and archive them under `scratch/` once their phase is done.

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

- `HANDOFF.md` — **START HERE.** Execution entry point for the active KV-cache direction (written in Chinese): what's decided (Path B, ICLR 2027, base = Fast KVzip), what NOT to re-investigate, the reproduce → variance-ablation (GO/NO-GO) → build plan, the four theory-driven code changes, and the honest ~10–20% accept-probability calibration. The method was further pinned on 2026-07-30 to **Option 3 "free-energy unified eviction"** (one scalar `F_i = D_i + λ·KL_i` decides both *which* KV is demoted into memory and *how much* is written) — but stage 1 (the GO/NO-GO variance ablation) deliberately uses simple recency/sliding-window eviction first, to isolate "distributional vs. point absorption" before adding the unified eviction.
- `free_energy_memory_proposal.md` — full theoretical writeup of the *video-era* framing (theory, math, related work, experiment plan). Historical origin, not the current goal.
- `theory_distributional_memory.md` — theory of why distributional (μ,σ²) memory can beat point memory (Bayesian filtering / rate-distortion / multimodality). §9 = rigorous KV-cache instantiation + the four load-bearing theory gaps; §10 = cognitive-science anchors + intro skeleton + the "unify existing methods as Σ=I special cases" narrative; **§11 = full method design for Option 3 (`F_i` definition, degenerate-special-cases table, amortized F-predictor, four-tier ablation, staged de-risking)**.
- `fastkvzip_code_map.md` — **source-code map of the cloned Fast KVzip / KVzip repos**: end-to-end prune pipeline and scoring (reconstruction attention vs. learned gate). Its Path-B anchor discussion predates the on-server verification — where it and the "Two Codebases" section above disagree, the section above (`EvictCache._sample_cache`) is the checked one.
- `kv_direction_positioning.md` — positioning for the KV pivot: Path A vs B (B decided), three-layer competitor/ancestor map, the make-or-break variance-ablation experiment, §7 accept-probability, §8 step-by-step execution checklist. Note IndexMem/Tensor Cache (closest precedents) have NO public code — comparison papers only.
- `kv_cache_survey.md` — survey of training-free KV cache compression (5 mechanism families + top-venue oral/poster status), plus a reproducibility verdict on Fast KVzip (easy–moderate, gates released)
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
