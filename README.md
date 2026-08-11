
# VariKV

**Variational Free-Energy Eviction for KV Cache Compression**

A single free-energy scalar decides *both* which KV entries get demoted into memory *and* how much of them gets written — instead of discarding evicted keys and values, they are absorbed into a distributional memory whose slots hold Gaussian beliefs `(μ_k, σ²_k)`.

> Status: research in progress. The method is implemented and unit-verified; the decisive experiment has not been run yet. Numbers here are sanity checks, not results.

## The idea

For each KV entry, define the cost of compressing it into memory:

```
F_i  =  D_i  +  λ · KL_i
        ↑         ↑
   distortion   surprise
```

- `D_i` — expected attention-output distortion if `KV_i` were replaced by its memory reconstruction, taken **in expectation over the future query distribution** (not over realised attention — that would reduce the method to H2O/SnapKV).
- `KL_i` — information gain of `KV_i` relative to the current memory mixture prior.

Then:
- **Eviction** — rank by `F_i`, keep the top-B exact, demote the rest. High `F` = expensive to compress = keep precise.
- **Writing** — the same `KL_i` drives the write rate; a Bayesian precision-weighted update means low-variance (confident) slots automatically resist overwriting.
- **Read-back** — slots decode into effective `(k̂, v̂)` that future queries attend to like ordinary KV.

Existing methods fall out as degenerate cases: drop the KL term → Expected Attention; use realised attention → H2O/SnapKV; make the memory point-valued → IndexMem-like; evict by recency only → Infini-attention/Tensor Cache; don't absorb at all → KVzip.

## Layout

```
varikv/
  config.py        two orthogonal switches (evict_policy × absorb_mode) + hyperparameters
  memory.py        slots, precision-weighted update, variance-aware read-out
  free_energy.py   F_i, expected attention, amortised F predictor
  cache.py         chunked prefill → evict → absorb → read back
  rope.py          inverse/forward rotation (see "RoPE" below)
  train.py         frozen LLM, trains only the memory + predictor
  evaluate.py      four-tier ablation
stage1/
  data.py          synthetic retain/update retrieval task with controllable distractors
```

## Ablation design

The four tiers are **two switches, not four codepaths**:

| tier | `evict_policy` | `absorb_mode` | degenerates to |
|---|---|---|---|
| 1 | — | `discard` | KVzip / FastKVzip |
| 2 | `recency` | `point` | Infini-attention / Tensor Cache |
| 3 | `free_energy` | `point` | IndexMem-like |
| 4 | `free_energy` | `dist` | **VariKV** |

`point` and `dist` share identical structure and parameter count and run the same precision-weighted update — the only difference is whether the precision term carries information. That keeps the comparison clean: the sole independent variable is whether variance helps.

Setting `free_energy.lam = 0` recovers pure-distortion scoring (Expected Attention); this degeneracy is verified empirically, and sweeping λ traces rate-distortion working points.

## One implementation detail worth knowing

Cached keys are stored **post-RoPE**, and RoPE does not distribute over addition:

```
α·R_p k + (1−α)·R_p' k'  ≠  R_φ(α·k + (1−α)·k')
```

A memory slot is a weighted average of absorbed keys, so building it from post-RoPE keys yields something that is not a valid key at *any* position — measured inner products with a query swung between −17 and +13 as query position varied, with no pattern. Worse, mixing phases inflates σ² for reasons unrelated to epistemic uncertainty, which would invalidate the entire premise that variance encodes confidence.

`varikv/rope.py` therefore inverse-rotates each evicted key into a position-free frame before absorbing, and re-rotates at read time to the slot's position centroid. Since `R(δ)R(p) = R(p+δ)` this is pure algebra, no extra forward pass.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.51.3 datasets==3.6.0 einops
```

Baseline reproduction uses [FastKVzip](https://github.com/Janghyun1230/FastKVzip) and [KVzip](https://github.com/snu-mllab/KVzip), which are **not vendored here** — clone them into `external/` yourself:

```bash
mkdir -p external && cd external
git clone https://github.com/Janghyun1230/FastKVzip
git clone https://github.com/snu-mllab/KVzip
```

Note FastKVzip ships no LICENSE file; clear usage with its authors before publishing anything derived from it.

## Running

```bash
python stage1/data.py build                        # generate the synthetic task
python varikv/train.py --tier 4                    # then 2, 3
python varikv/evaluate.py --tier 1 2 3 4
```
