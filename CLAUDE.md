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

### Known limits — do not mistake these for working

- **`prune()` (unchunked) leaves the model generating empty strings — but so does native `EvictCache`.** Verified identical across `evict` / `memory` / `memory+absorb`. This is the upstream path `eval.py` avoids by forcing `kv_type="retain"`; `eval_chunk.py` uses `prune_chunk` and is unaffected. Not our bug, not fixed.
- **`batch>1` unsupported** (guarded). Position tracking, per-layer state slicing and absorb padding all assume B=1.
- ~~**The memory is untrained for any target model.**~~ Superseded: ten 7B checkpoints exist (inventory in the 2026-08-10 section). Trained memory does not fix the degradation — under KV injection it stays 30–45 points down, and under the residual read-out it only reaches parity by closing its own gate.
- **Capacity is the open design question.** Measured at ratio 0.3: 4,080 retained real KV per head vs M=16 memory ⇒ **0.39% of visible KV**, summarising ~13k evicted tokens at 800:1. A `<3%` share now prints an automatic warning. Raising `num_slots` is the obvious move, but stage1 measured *larger K is worse* (at a very different scale), so it needs re-sweeping — and now that the budget is charged honestly, a larger K costs real retained KV.

### Local modifications to the vendored FastKVzip clone

`external/FastKVzip/` keeps its own `.git` and has no LICENSE file. Local edits made so far — re-apply if it is ever re-cloned:

| File | Change |
|---|---|
| `data/load.py` | pyarrow fallback for `scbench_mf_mid` and `squad` (datasets 4.x wrote their parquet; `Feature type 'List' not found` under datasets 3.6.0) |
| `eval_chunk_mrcr.py` | honour `--num`; print `Finished.` |
| `eval_mrcr.py` | **new file** — unchunked MRCR eval for the KVzip baseline |
| `attention/memcache.py` | **new file** — `MemoryEvictCache`, the VariKV integration (Stage 2b) |
| `attention/memcache_retain.py` | **new file** — `MemoryRetainCache`, the same method built on `RetainCache` instead. This is what the baselines actually run (`args.py` default), and it is *simpler*: the cache stays rectangular `[B,H,M+seq,dim]`, so "original position = sequence index" and none of the `pos_track` machinery is needed; `self.valid` is cumulative, so the newly-evicted set slices straight out of `evict_range` with no double-absorption risk. Now the default of `--varikv_kv_type` |
| `model/wrapper.py` | `kv_type="memory"` and `"memory_retain"` dispatch; `chunk_ratio` branch now excludes `"memory"` alongside `"evict"` (both use nested-list `valid`, not a tensor); `prefill` split into `prefill`/`_prefill_impl` so `varikv_train` can bypass `inference_mode` |
| `args.py` | five VariKV flags: `--varikv_ckpt` (giving it enables the memory), `--varikv_slots`, `--varikv_kv_type`, `--varikv_readout {normal,zero}` (the zero-read-out ablation), `--varikv_residual` |
| `eval_chunk.py` | builds the memory from the ckpt and derives the tag. **`n_groups` must be passed when the ckpt is residual**, or loading dies on `Unexpected key(s): residual_gate` |
| `eval.py`, `results/parse.py` | both gained a `set_ratios()` honouring `VARIKV_RATIOS`. **They must agree** — see the parse trap in the 2026-08-10 section |

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

## Stage 1 — the GO/NO-GO variance ablation

`stage1/data.py` builds the synthetic dataset for the make-or-break experiment (distributional vs. point absorption of evicted KV). It is standalone — no dependency on `memory_module.py` or `external/`.

The design is load-bearing, not arbitrary: samples come in **two kinds that must both be present**. `retain` (fact appears early, only same-format distractors after it → tests low KL ⇒ *don't* update ⇒ resist washout) and `update` (the fact is genuinely rewritten mid-context → tests high KL ⇒ *do* update). A fixed-rate point memory cannot be optimal on both at once; testing only one kind lets a tuned fixed rate tie, i.e. a false negative. The `n_distract` axis (0/200/800/2000) is the predicted-effect-size knob: advantage should grow with distractor count and vanish near 0.

```bash
.venv/bin/python stage1/data.py          # print n_distract → token-length table only
.venv/bin/python stage1/data.py build    # write stage1/{train,val}.jsonl
```

`build` emits 3200 train / 400 val over the 4 distractor levels. Measured context lengths: `n_distract` 0 → 109 tokens, 200 → 3,526, 800 → 13,817, 2000 → 34,357. Neither jsonl is committed — regenerate as needed (seeds are fixed: train 0, val 1234).

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

`point` and `dist` share **identical structure and parameter count** (verified equal) and run the same precision-weighted update — the only difference is whether the precision term carries information. That is what keeps the ablation clean: the sole independent variable is "does variance help".

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

## The residual read-out was built and measured (2026-08-10) — the regression is fixed, and the memory still earns nothing

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

| ratio | baseline | `ckpt_stage2b_res` dist (gate σ 0.186) | same, point (σ 0.287) | `gap_*` (σ 0.014–0.032) |
|---|---|---|---|---|
| 0.75 | 68.80 | 11.00 (−57.80★) | 11.20 (−57.60★) | not run |
| 0.5 | 71.60 | 4.60 (−67.00★) | 3.60 (−68.00★) | not run |
| 0.4 | 66.40 | 5.00 (−61.40★) | 3.00 (−63.40★) | not run |
| 0.3 | 65.40 | 9.60 (−55.80★) | 2.60 (−62.80★) | not run |
| 0.2 | 45.20 | 8.60 (−36.60★) | 2.60 (−42.60★) | not run |
| 0.1 | 32.60 | 7.60 (−25.00★) | 0.80 (−31.80★) | 30.60 / 32.00 / 32.40, none separated |
| 0.05 | 2.00 | 0.80 | 0.40 | 2.20 (baseline itself is at the floor) |

★ = 95% CI excludes zero. So the residual read-out **softened** the collapse on `many_shot` (−5…−18) but does nothing of the kind on retrieval data: with the gate actually open it is −56…−68. The earlier conclusion "KV injection was the whole cause of the 30–45 point collapse" must be narrowed to "…on `many_shot`". The dichotomy stands unchanged: **gate closed ⇒ baseline, gate open ⇒ much worse, training ⇒ closes the gate.**

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

### The `--obj gap` objective was a live suspect for being degenerate — partially confirmed, see the measurement above

Training-log comparison of the two objectives (both matched config, chunk 16000 / window 4096):

| objective | final loss | `|g|`max | gate σ trajectory |
|---|---|---|---|
| `lm` (`ckpt_stage2b_res`) | 1–2, noisy | 3.4e-02 | 0.095 → **0.186**, monotonically opening |
| `gap` (`ckpt_gap_*`) | **0.003** | **1e-04** | init 0.018 → **0.014**, i.e. *below* its starting point |

The gap objective is `MSE(g·m, o_full − o_pruned)` (`memcache_retain.py:295`), and `m → 0` is inside its solution space, so a loss of 0.003 is consistent with the memory having learned nothing and the gate correctly switching itself off. **This is the same trap as the F-predictor collapse** (loss 0.0419 vs 0.0421 for emitting the constant 0). The one number that settles it is `mean(tgt²)` — the MSE of the trivial `m ≡ 0` solution. Not yet measured; a single-sample probe would do it. Do not read "loss 0.003" as convergence until that comparison exists.

Note the `lm` objective has its own diagnosed defect: its loss falls to 1–2 and the gate opens monotonically while downstream accuracy collapses — fineweb-edu continuation loss is decoupled from, and here anti-correlated with, retrieval accuracy.

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

### One claim in `scratch_kvres_eval.sh`'s header is wrong

It justifies skipping the standard interval for the `gap_*` ckpts with "gate-closed configs were already proven byte-identical to baseline by `rbkv`". `rbkv` loaded `ckpt_stage2b_retain`, whose gate sits at its **init** 0.018 and was never trained; the `gap_*` gates are trained (0.014 / 0.024 / 0.032, max 0.26–0.40, 4–12% of the 112 head-groups above 0.1). They are not the same configuration, and the measurement agrees — `gapf` reads 30.60 at ratio 0.1 where the baseline reads 32.60. Treat the standard interval for those ckpts as missing data, not as redundant.

### In flight as of 2026-08-11 04:30 UTC

Both runs use the three `gap_*` ckpts with `--varikv_residual`, tags `gfsd` / `grsd` / `grsp` (distinct per ckpt because `gap_fix03/dist` and `gap_rand/dist` are both `dist` mode and result dirs carry only the mode).

- `scratch_gapstd_eval.sh` — the three ckpts × `scbench_kv` × standard interval (0.75→0.2). 3 GPUs, ~208 s/sample, ETA ~10:00 UTC.
- `scratch_gapsweep.py` — the three ckpts × the other 9 datasets, 27 jobs, marker-resumable, longest-first. Baselines are **not** re-run (the `_full` tag from `scratch_stage2b_sweep.py` is the same configuration). 56.7 GPU-h total; workers on GPUs 0–2 wait for the `scbench_kv` run to print `ALL DONE` before taking work. ETA ~13:30–14:00 UTC.

Measured per-dataset cost for one config over 5 ratios, useful for planning any future grid: `repoqa` 5.83 h, `prefix_suffix` 3.23, `mf` 2.97, `vt` 2.20, `summary` 1.88, `gsm` 1.19, `qa_eng` 0.60, `squad` 0.55, `choice_eng` 0.44 — **18.9 GPU-h per config for those 9**, plus ~5.8 h for `scbench_kv`.

MRCR cannot join this table: it runs `eval_chunk_mrcr.py`, and the VariKV injection was never wired into that path. So the ceiling for these sweeps is 11 of Figure 11's 12 panels.

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
| `scratch_kvres_eval.sh` / `scratch_kvres_report.py` | stage2b | the `scbench_kv` residual evaluation (2026-08-10 night) and its paired-bootstrap report. The report **self-checks** its per-sample parse against `results.parse`'s absolute rows — copy that pattern when adapting it to another dataset (it is hard-coded to one `DATA`). Its header claim about `rbkv` is wrong; see the 2026-08-11 section |
| `scratch_gapstd_eval.sh` | stage2b | the three `gap_*` ckpts × `scbench_kv` × standard interval (2026-08-11) |
| `scratch_gapsweep.py` / `scratch_gapsweep_logs/` | stage2b | the three `gap_*` ckpts × the other 9 datasets, 27 jobs. Marker-resumable, longest-first scheduling, and workers can be told to wait for another run to finish before claiming a GPU (`--wait_gpus`) |
| `scratch_probe_gap_target.py` | stage2b | **the trivial-solution / capacity-ceiling probe.** Reports `mean(tgt²)` (the `m ≡ 0` MSE) beside the achieved MSE, plus `R_opt` = the best relative reduction obtainable by re-tuning the per-head gates. Monkey-patches `_attn_gap` / `memory_residual` rather than editing `memcache_retain.py`, so it is safe to run while eval jobs are in flight (editing the harness file would change what newly-launched jobs load). ~20 GB, a few minutes; run it before trusting any `gap`-objective loss curve |
| `scratch_stage2b_logs/` | stage2b | all of the above runs' logs; `sweep/` holds the 11-dataset sweep |

Keep new throwaway scripts on the `scratch_` prefix at the root, and archive them under `scratch/` once their phase is done.

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
