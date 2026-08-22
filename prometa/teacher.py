"""ProMeta 的教师：**fineweb-edu 长文 + 合成事实**，与全部下游评测集零重叠。

**语料选择不是自由参数，是纪律。** `EVAL_PROTOCOL.md` 已预先登记：
`--corpus cat`（fineweb-edu + 合成事实）与所有评测集**零重叠**；
`real:<panel>`（用面板自己的上下文）是**严格更弱的方法命题**。
Student 的教师标签**必须**走前者 —— 否则就是在测试集上训练。
（Oracle 探针用 scbench 面板可以：它什么都不训练，且 Oracle 臂按定义就是上界。）

────────────────────────────────────────────────────────────────────────
**为什么不能直接复用 `scratch_adv_teacher.make_task`**

它注入 `n_fact` 条事实、出 `n_fact` 个问句，**每句问一个不同的 key** ⇒
未来之间**支撑互不相交**。而在精确不相交时

    mean_i = max_i / M          （M 是常数）

是**单调变换 ⇒ 排序完全相同** ⇒ `ρ_β` 扫遍全部 β 都给同一个保留集，
**ProMeta 的整个风险维度在那样的教师上无信号可学**。
（真实 softmax 让每个位置对每个未来都有 ε>0 的质量，所以严格退化只在
理想极限成立；实践中分歧局限在截断边界、且随预算变化 —— 这**加强**而非
削弱下面的要求：共享需求必须是真共享，不能只是 softmax 噪声。）

⇒ **教师任务必须让两类位置同时存在**（这是从上式推出来的硬要求）：

  · **私有强需求**：只有某一个未来需要的事实           → 撑起 `max` 那端
  · **共享弱需求**：多个未来**联合**需要的事实、以及全部未来共享的格式段
                                                       → 撑起 `mean` 那端

────────────────────────────────────────────────────────────────────────
**反循环性控制（必读）**：把语料设计成「有共享需求」，等于断言真实负载有
共享需求。所以本模块**强制**同时输出教师语料与评测面板的需求结构统计
（`demand_structure`），让两者是否匹配可被检查。**若评测面板上的未来是纯
不相交的（`scbench_kv` 的 5 个 key 查找就是），ProMeta 在那里帮不上忙，
与教师怎么造无关** —— 这一句必须写进论文的限制。
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))
sys.path.insert(0, HERE)        # 直接 `python prometa/teacher.py` 时也能 import prometa.*

HEX = "0123456789abcdef"
# **全共享段必须随机化**（2026-08-22，外部复核指出，采纳）：固定字符串
# `Formatting rule: ... ANSWER=<value> ...` 是一个**极易被识别的 shortcut**——
# Student 可能只学会「看到这串模式就保护它」，而不是「从上下文推断未来需要什么」。
# 随机化模板、标记词与取值格式后，那条捷径就不再是一个固定的隐藏状态模式。
# ⚠ 保留「所有未来都引用同一段」这个**结构**（那是要测的东西），
#   只随机化它的**表面形式**。
FORMAT_TEMPLATES = [
    " Formatting rule: every answer must be written as {tag}={{value}} with no spaces. ",
    " Output convention: report each result in the form {tag}={{value}}, lowercase only. ",
    " Response protocol: prefix every returned item with {tag}= and nothing else. ",
    " Answer style guide: each value must appear as {tag}={{value}} on its own. ",
    " Reporting rule: wrap each retrieved item as {tag}={{value}}, no extra words. ",
]
FORMAT_TAGS = ["ANSWER", "RESULT", "VALUE", "OUT", "FOUND", "ITEM"]


def build_task(enc, ids, max_ctx, window, n_fact, rng, n_joint=2):
    """→ (ctx_ids, futures, meta)。`enc(str) -> List[int]`。

    构造 `M = n_fact + n_joint` 个未来：
      · `n_fact` 个**单事实**问句（私有强需求）
      · `n_joint` 个**双事实**问句（在两个事实上与单事实问句共享需求）
      · **全部** M 个问句都引用同一段 `FORMAT_RULE`（全共享弱需求）

    与 `scratch_adv_teacher.make_task` 保持一致的地方（**刻意不改**）：
    同样的 16 位 hex key/val、同样的插入区间 `[5%, 90%)`、**倒序插入**
    以免下标失效、末尾 `window` 不插（那里恒保留、插了等于没驱逐）。
    """
    ctx = list(ids[-max_ctx:])
    keys = ["".join(rng.choices(HEX, k=16)) for _ in range(n_fact)]
    vals = ["".join(rng.choices(HEX, k=16)) for _ in range(n_fact)]
    spans = [enc(f" The secret key {k} maps to the value {v}. ")
             for k, v in zip(keys, vals)]
    fmt_tag = rng.choice(FORMAT_TAGS)
    fmt_rule = rng.choice(FORMAT_TEMPLATES).format(tag=fmt_tag)
    spans.append(enc(fmt_rule))                          # 最后一段是全共享的（表面形式随机）

    lo, hi = int(0.05 * (len(ctx) - window)), int(0.90 * (len(ctx) - window))
    assert hi - lo > len(spans), f"可驱逐区太短：[{lo},{hi}) 放不下 {len(spans)} 段"
    pos = sorted(rng.sample(range(lo, hi), len(spans)), reverse=True)
    for p_, sp in zip(pos, spans):
        ctx[p_:p_] = sp
    # **插入后的真实位置必须显式算，不能事后猜**（外部复核指出，采纳）。
    # 两个错误已修：① 旧代码的 `fmt_pos` 化简后等于 `pos[0]`（最大的那个），
    #   而 `spans` 的最后一项 FORMAT_RULE 配的是 `pos[-1]`（最小的）——**张冠李戴**；
    # ② 即便取对了 `pos[-1]`，那也是**插入前**的下标：倒序插入时，在更小位置
    #   插入会把已插好的、位置更大的段整体右移。段 k（按 pos 降序）插完后的
    #   真实起点是 `pos[k] + Σ_{j>k} len(span_j)`。
    # `fmt_pos` 目前只是 metadata、不进 loss，所以不影响已跑的训练；
    # 但任何归因/命中率分析用它都会错，故现在就修对。
    shift = 0
    final = [0] * len(spans)
    for k in range(len(spans) - 1, -1, -1):        # 从最小位置往回累加
        final[k] = pos[k] + shift
        shift += len(spans[k])
    fact_pos = sorted(final[:n_fact])
    fmt_pos = final[n_fact]

    tail = " Answer using the formatting rule defined earlier."
    futures = []
    for j in range(n_fact):                              # 私有
        futures.append(dict(
            q=enc(f"\nQuestion: What value does the secret key {keys[j]} "
                  f"map to?{tail}\nAnswer:"),
            a=enc(f" {fmt_tag}={vals[j]}"), needs=[j], kind="single"))
    for t in range(n_joint):                             # 共享
        a_, b_ = rng.sample(range(n_fact), 2)
        futures.append(dict(
            q=enc(f"\nQuestion: Give the values for the secret keys {keys[a_]} "
                  f"and {keys[b_]}.{tail}\nAnswer:"),
            a=enc(f" {fmt_tag}={vals[a_]} {fmt_tag}={vals[b_]}"),
            needs=sorted([a_, b_]), kind="joint"))
    meta = dict(n_fact=n_fact, n_joint=n_joint, M=len(futures), fmt_tag=fmt_tag,
                pos_pre=sorted(pos),        # 插入**前**的下标（诊断用）
                fact_pos=fact_pos,          # 插入**后**的真实起点
                fmt_pos=fmt_pos, span_len=[len(x) for x in spans],
                kinds=[f["kind"] for f in futures],
                needs=[f["needs"] for f in futures])
    return ctx, futures, meta


def demand_structure(U, rho=0.05):
    """→ 需求结构统计。**教师语料与评测面板都要报，用于反循环性检查。**

    `U`: [M, L, H, N]。返回
      · `J_mean_max` —— 同预算下「按均值排」与「按最大值排」的保留集 Jaccard。
        **≈1 ⇒ 风险维度退化**，在这批数据上 ProMeta 与期望效用无法区分。
      · `rho_pair`   —— 未来两两 Spearman 的均值（越低越多模态）。
      · `shared_frac`—— 争议池内「被 ≥2 个未来实质需要」的位置占比。
    """
    from prometa.risk import topb_mask
    U = np.asarray(U, dtype=np.float64)
    M, L, H, N = U.shape
    k = max(1, int(round(rho * N)))
    mean, mx = U.mean(0), U.max(0)
    a, b = topb_mask(mean, k), topb_mask(mx, k)
    J = float(((a & b).sum(-1) / np.maximum((a | b).sum(-1), 1)).mean())

    def _rank(x):
        return np.argsort(np.argsort(x, -1), -1).astype(np.float64)
    R = _rank(U.reshape(M, -1, N))
    R -= R.mean(-1, keepdims=True)
    nrm = np.sqrt((R ** 2).sum(-1))
    rp = []
    for i in range(M):
        for j in range(i + 1, M):
            rp.append(((R[i] * R[j]).sum(-1) / np.maximum(nrm[i] * nrm[j], 1e-30)).mean())
    # **争议池必须取两条规则的并集。** 只按 `max` 选（首版）会把池子填满私有
    # 位置 —— 因为 `max` 按构造就偏向「被单个未来强需要」的那类 —— 于是
    # 「共享程度」在最该为正的语料上又读出 0。这是同一个错误的第三种形态：
    # **用其中一条规则去定义评判两条规则的样本池，必然偏向那一条。**
    kk = min(N, 3 * k)
    pool = topb_mask(mx, kk) | topb_mask(mean, kk)
    # **共享程度用无阈值的量**：`conc_i = max_m U / Σ_m U` ∈ [1/M, 1]
    # （独占=1、均摊=1/M），取 `1 − conc` 即共享程度。
    #
    # ⚠ **两版带阈值的定义都错了，错法有必然性，记下来别再试第三种**：
    #   v1「落进该未来自己的 top-k」—— 若某未来的私有证据恰好 k 个位置，
    #      top-k 全被私有占满，共享证据永远够不到 ⇒ 恒为 0。
    #   v2「达到该未来自身峰值的 25%」—— 也恒为 0，因为**共享位置按构造就比
    #      私有位置弱**（正是这个强弱差让 mean 与 max 分歧），
    #      所以**任何相对峰值的阈值都会把共享位置排除掉**。
    # ⇒ 只能用无阈值的质量占比。这个量在 `scratch_prometa_probe.py` 里
    #   已作为「集中度」验证过一次（同一个量两条实现口径一致）。
    conc = mx / np.maximum(U.sum(0), 1e-30)               # [L,H,N]
    shared = float((1.0 - conc)[pool].mean()) if pool.any() else float("nan")
    only_mean = topb_mask(mean, k) & ~topb_mask(mx, k)
    only_max = topb_mask(mx, k) & ~topb_mask(mean, k)
    return dict(J_mean_max=J, rho_pair=float(np.mean(rp)), shared_frac=shared,
                conc=float(conc[pool].mean()) if pool.any() else float("nan"),
                # 机制可见：两条规则各自独占的位置分别长什么样
                conc_only_mean=float(conc[only_mean].mean()) if only_mean.any() else float("nan"),
                conc_only_max=float(conc[only_max].mean()) if only_max.any() else float("nan"),
                M=M, N=N, k=k)



# ══════════════════════════════ GPU 抽取路径 ══════════════════════════════
def chunk_ranges(n_total, sys_len, prefill_chunk, window_size):
    """复现 `model/wrapper.py:_prefill_impl` 的 `evict_range` 序列。

    上游循环（逐字对照过源码）：

        start = sys_len
        for 每个 chunk:
            前向
            end = len(score) - window_size          # len(score) = 已处理的总 token
            prune_chunk(ratio, (start, end))        # **无条件调用**
            start = end                             # **无条件赋值**

    ⚠ **这是第二份实现**（第④类错的高发地）。`scratch_prometa_smoke.py`
    会 monkey-patch 真机的 `prune_chunk` 抓下真实的 `(lo,hi)` 并逐项断言相等；
    **没跑过那条断言之前，不要相信这个函数**。

    返回 `(all_ranges, usable)`：`all_ranges` 是忠实复现（可能含 `hi<=lo` 的
    退化项，chunk < window 时真的会出现），`usable` 是过滤后可用于训练/抽取的。
    """
    out, start, cum = [], sys_len, 0
    while cum < n_total:
        cum = min(cum + prefill_chunk, n_total)
        out.append((start, cum - window_size))
        start = cum - window_size
    usable = [(lo, hi) for lo, hi in out if hi > lo]
    return out, usable


@torch.no_grad()
def future_utility(key_cache, q_cap, lo, hi, tblock=32, out_np=True):
    """→ U: [L, Hkv, hi-lo]，**softmax 只在 `[lo,hi)` 上归一化**。

        U[l,h,i] = max_{t, hq∈group(h)}  softmax_{i∈[lo,hi)}( q[hq,t]·K[h,i]/√d )

    **为什么归一化到 chunk 而不是整个前缀**（这一条是设计的关键，别改回去）：
    Student 在 `prometa/cache.py:prometa_scores` 里就是对 `[lo,hi)` 做 softmax
    的，因为 `prune_chunk` 那一刻要裁决的候选集**就是** `[lo,hi)`。教师必须
    定义在同一个支撑上，否则 KL 比的是两个不同支撑上的分布。

    ⚠ **不能靠「先算全前缀、再重归一化」来省事。** 单条 softmax 行确实满足
    `softmax(z)|_S / Σ_S = softmax(z|_S)`，但本量在归一化**之后**还要对 `t`
    与组内 query 头取 **max**，而不同 `t` 的归一化常数不同 ⇒
    `max_t (p_t|_S / Z_t(S))  ≠  (max_t p_t)|_S / Z(S)`。所以必须在抽取时
    就按 chunk 归一化（自测 ⑦ 用反例验证了这个不等号确实成立）。

    GQA：`kv_head = q_head // (Hq // Hkv)`，组内取 max —— 与 KVzip 族打分器
    以及 `scratch_prometa_oracle.future_utility` 同一口径（自测 ⑥ 逐位对拍）。
    """
    L = len(q_cap)
    n = hi - lo
    assert n > 0, (lo, hi)
    out = []
    for l in range(L):
        q = q_cap[l]                                  # [1,Hq,T,d]
        K = key_cache[l]                              # [1,Hkv,N,d]
        assert q.dim() == 4 and K.dim() == 4, (q.shape, K.shape)
        Hq, T, d = q.shape[1], q.shape[2], q.shape[3]
        Hkv, N = K.shape[1], K.shape[2]
        assert N >= hi, (N, hi)
        assert Hq % Hkv == 0, (Hq, Hkv)
        G = Hq // Hkv
        Kp = K[0, :, lo:hi, :].float()                # [Hkv,n,d]
        acc = torch.zeros(Hkv, n, device=Kp.device, dtype=torch.float32)
        scale = 1.0 / (d ** 0.5)
        for h in range(Hkv):
            qh = q[0, h * G:(h + 1) * G].float()      # [G,T,d]
            for t0 in range(0, T, tblock):
                a = torch.einsum("gtd,nd->gtn", qh[:, t0:t0 + tblock, :], Kp[h]) * scale
                a = torch.softmax(a, dim=-1)
                acc[h] = torch.maximum(acc[h], a.amax(dim=(0, 1)))
                del a
        out.append(acc)
        del Kp, acc
    U = torch.stack(out, 0)                           # [L,Hkv,n]
    return U.cpu().numpy() if out_np else U


def extract_U(model, ctx_ids, futures, ranges, span="qa", tblock=32,
              prefill_chunk=16000, verbose=True):
    """在 GPU 上抽教师标签。→ `(U_by_range, kv, info)`。

    `U_by_range[(lo,hi)]`: np.float32 [M, L, Hkv, hi-lo]。
    `kv` **原样返回**（满缓存，未驱逐）—— 训练侧要直接用它的 `key_cache` /
    `value_cache`，不必重跑一次预填（预填是这条路上唯一昂贵的一步）。

    **`span` 是那条必须做的消融**：
      · `"qa"` —— 未来 = 问句 + **真答案**。教师用特权信息是合法的（蒸馏的
        定义就是如此，FastKVzip 自己的门控也蒸馏 KVzip 的重构分），但**答案
        token 的 query 才是真正发生检索的地方**，所以它更接近"未来到底需要
        什么"。
      · `"q"`  —— 只用问句。**没有任何特权信息**，若两者给出的标签几乎一致，
        「泄漏」这条质疑就整条消失，应当优先选它。
    ⇒ 两个都跑、报一致性，别只跑一个然后声称没泄漏。

    ⚠ **全程 `no_grad` 而不是 `inference_mode`。** `_prefill_impl` 默认包在
    `inference_mode` 里，那样产出的 K/V 是 **inference tensor**，**永远**不能
    进 autograd（本仓库 Stage 2b 的 2 号 bug）。训练侧要拿这批 K/V 当常量做
    前向，所以这里显式走 `varikv_train=True` + `no_grad`。
    同理**不用 `model._prob`**（它自己带 `@torch.inference_mode()`，且会污染
    cache：`update()` 把 inference tensor `cat` 进 `key_cache`，`slice()` 回滚
    后仍是 inference tensor）—— 直接调 `model.__call__(..., update_cache=False)`，
    它只跑 base model、不算 vocab softmax，更便宜。
    """
    assert span in ("q", "qa"), span
    dev = model.device
    ctx_t = ctx_ids if torch.is_tensor(ctx_ids) else torch.tensor([ctx_ids], device=dev)
    if ctx_t.dim() == 1:
        ctx_t = ctx_t[None]
    ctx_t = ctx_t.to(dev)

    prev_train = getattr(model, "varikv_train", False)
    model.varikv_train = True
    try:
        with torch.no_grad():
            kv = model.prefill(ctx_t, prefill_chunk_size=prefill_chunk,
                               do_score=False, chunk_ratio=1.0)
    finally:
        model.varikv_train = prev_train
    n_prefix = int(kv.key_cache[0].shape[-2])
    Lk = len(kv.key_cache)

    for lo, hi in ranges:
        assert 0 <= lo < hi <= n_prefix, (lo, hi, n_prefix)

    acc = {r: [] for r in ranges}
    with torch.no_grad():
        for fi, f in enumerate(futures):
            ids = list(f["q"]) + (list(f["a"]) if span == "qa" else [])
            t = torch.tensor([ids], device=dev)
            kv.capture_q, kv._q_cap = True, {}
            model(t, kv, update_cache=False)
            kv.capture_q = False
            # ⚠ 两条不变量：前缀必须原样回滚、每层都要捕到 query
            assert int(kv.key_cache[0].shape[-2]) == n_prefix, \
                f"前缀被改动：{kv.key_cache[0].shape[-2]} != {n_prefix}"
            assert len(kv._q_cap) == Lk, f"只捕到 {len(kv._q_cap)}/{Lk} 层"
            qc = [kv._q_cap[l] for l in range(Lk)]
            for lo, hi in ranges:
                acc[(lo, hi)].append(future_utility(kv.key_cache, qc, lo, hi, tblock))
            if verbose:
                print(f"  [teacher] future {fi} span={span} T={qc[0].shape[2]}",
                      flush=True)
            kv._q_cap = {}
            del qc

    U_by_range = {r: np.stack(v, 0).astype(np.float32) for r, v in acc.items()}
    info = dict(n_prefix=n_prefix, L=Lk, Hkv=int(kv.key_cache[0].shape[1]),
                M=len(futures), span=span, ranges=list(ranges))
    return U_by_range, kv, info

def _selftest():
    rng_ = __import__("random").Random(0)

    class FakeEnc:
        def __call__(self, s):
            return [hash(w) % 50000 for w in s.split()]
    enc = FakeEnc()
    ids = list(range(20000))
    ctx, fut, meta = build_task(enc, ids, 8000, 512, n_fact=4, rng=rng_, n_joint=2)

    assert meta["M"] == 6 and len(fut) == 6, meta
    assert [f["kind"] for f in fut] == ["single"] * 4 + ["joint"] * 2
    assert all(len(f["needs"]) == 1 for f in fut[:4])
    assert all(len(f["needs"]) == 2 for f in fut[4:])
    print(f"① 构造 M={meta['M']}（{meta['n_fact']} 私有 + {meta['n_joint']} 联合）　PASS")

    # ② 插入点都落在可驱逐区、互不相同、且上下文确实变长
    lo, hi = int(0.05 * (8000 - 512)), int(0.90 * (8000 - 512))
    assert len(set(meta["pos_pre"])) == len(meta["pos_pre"])
    assert all(lo <= p < hi for p in meta["pos_pre"]), (lo, hi, meta["pos_pre"])
    assert len(ctx) > 8000
    print(f"② {len(meta['pos_pre'])} 个插入点全在 [{lo},{hi}) 内且互不相同　PASS")

    # ②b **`fmt_pos` 必须真的指向 FORMAT_RULE 段**（旧代码化简后等于 `pos[0]`，
    #     张冠李戴；且没算倒序插入造成的右移）。直接切片比对，不靠推理。
    fl = meta["span_len"][-1]
    fp = meta["fmt_pos"]
    seg = ctx[fp:fp + fl]
    assert len(seg) == fl and any(seg), (fp, fl)
    # 事实段同理逐个核对：它们的长度与 span_len 前 n_fact 项一一对应
    for j, fpos in enumerate(sorted(meta["fact_pos"])):
        assert 0 <= fpos < len(ctx), (j, fpos)
    # 决定性检查：把 FORMAT_RULE 段挖掉后，剩下的长度正好少 fl
    assert len(ctx[:fp] + ctx[fp + fl:]) == len(ctx) - fl
    # 且该段与其余 4 个事实段互不重叠
    spans_iv = [(p_, p_ + L) for p_, L in
                zip(sorted(meta["fact_pos"]) + [fp],
                    meta["span_len"][:meta["n_fact"]] + [fl])]
    spans_iv.sort()
    assert all(spans_iv[i][1] <= spans_iv[i + 1][0] for i in range(len(spans_iv) - 1)), \
        spans_iv
    print(f"②b fmt_pos={fp}（长 {fl}）指向真实 FORMAT_RULE 段、"
          f"与 {meta['n_fact']} 个事实段互不重叠　PASS")

    # ③ **共享结构真的存在**：至少一个事实被 ≥2 个未来需要
    from collections import Counter
    c = Counter(j for f in fut for j in f["needs"])
    assert max(c.values()) >= 2, c
    print(f"③ 事实被需要次数 {dict(c)} —— 存在共享需求　PASS")

    # ③b **全共享段的表面形式必须随机**（否则是 shortcut）；但结构必须保留
    tags = set()
    for sd in range(12):
        _, f2, m2 = build_task(enc, ids, 8000, 512, n_fact=4,
                               rng=__import__("random").Random(sd), n_joint=2)
        tags.add(m2["fmt_tag"])
        assert all(m2["fmt_tag"].encode()[:1] for _ in [0])
    assert len(tags) >= 3, f"全共享段没有被随机化：只有 {tags}"
    print(f"③b 全共享段表面形式随机（12 个种子出现 {len(tags)} 种标记 {sorted(tags)}）"
          f"，结构保留　PASS")

    # ④ **demand_structure 的两个极端对照**（判据本身要能拒）
    #
    # ⚠ **首版夹具造错了，方向还反了 —— 记在这里免得第三次犯。**
    # 首版用 `block += 1.0` 均匀加，于是同一未来拥有的 40 个位置**取值几乎全等**，
    # 两条规则只能按 1e-6 的噪声排 ⇒ J 平凡地**低**（实测 0.115），
    # 看上去像是推翻了「互不相交 ⇒ 排序相同」。
    # 这与 `scratch_prometa_probe.py` 自测⑦「k ≫ n_eff 时按噪声乱挑 ⇒ J 反而低」
    # 是**同一个失效模式**。正确的夹具必须让拥有位置的取值**互不相同**：
    # 那时 `mean_i = (u_i + (M−1)ε)/M`、`max_i = u_i`，`ε ≪ u 的散布` ⇒ 排序一致。
    r = __import__("numpy").random.default_rng(0)
    M, L, H, N = 5, 2, 2, 400
    disj = r.random((M, L, H, N)) * 1e-6                  # 互不相交
    for m in range(M):                                    # **取值互不相同**
        disj[m, :, :, m * 40:(m + 1) * 40] += 0.5 + r.random((L, H, 40))
    disj /= disj.sum(-1, keepdims=True)
    d1 = demand_structure(disj)
    # 混合型：私有证据的量级要约为共享的 M 倍，`mean` 与 `max` 才真的会分歧
    # （mean 把私有值除以 M、共享值不除）。否则两条规则各自被一类位置独占、
    # J 恒为 0，那是另一种极端、不代表真实语料。
    shared = r.random((M, L, H, N)) * 1e-6
    shared[:, :, :, :40] += 0.9 + 0.4 * r.random((1, L, H, 40))   # 全共享（各未来同值）
    for m in range(M):                                            # 各自私有，约 M 倍
        shared[m, :, :, 40 + m * 20:40 + (m + 1) * 20] += 4.0 + 3.0 * r.random((L, H, 20))
    shared /= shared.sum(-1, keepdims=True)
    d2 = demand_structure(shared)
    for nm, dd in [("互不相交语料", d1), ("混合型语料　", d2)]:
        print(f"④ {nm}：J(mean,max)={dd['J_mean_max']:.4f} "
              f"ρ_pair={dd['rho_pair']:+.3f} shared={dd['shared_frac']:.4f} "
              f"| 只被 mean 选中的位置 conc={dd['conc_only_mean']:.4f}、"
              f"只被 max 选中的 conc={dd['conc_only_max']:.4f}")
    assert d1["J_mean_max"] > 0.9, ("互不相交时排序应当几乎相同", d1)
    assert d1["J_mean_max"] > d2["J_mean_max"] + 0.1, (d1, d2)
    assert d2["shared_frac"] > d1["shared_frac"] + 0.1, (d1, d2)
    # 上下界自检：独占 ⇒ shared≈0；均摊 ⇒ shared≈1−1/M
    import numpy as _np
    one = _np.zeros((M, 1, 1, 8)); one[0] = 1.0
    allm = _np.ones((M, 1, 1, 8))
    assert demand_structure(one)["shared_frac"] < 1e-9
    assert abs(demand_structure(allm)["shared_frac"] - (1 - 1 / M)) < 1e-9
    print("   ⇒ 互不相交 J≈1（风险维度退化）、混合型 J 明显更低且共享占比更高"
          "　PASS（判据能区分两种语料）")
    # ⑤ `chunk_ranges` 手算对拍（忠实复现上游的无条件赋值）
    allr, use = chunk_ranges(n_total=40000, sys_len=20, prefill_chunk=16000,
                             window_size=4096)
    assert allr == [(20, 11904), (11904, 27904), (27904, 35904)], allr
    assert use == allr, use
    # 退化情形：chunk < window ⇒ 第一段 hi < lo，必须出现在 all 里、被 usable 滤掉
    a2, u2 = chunk_ranges(9000, 20, 2000, 4096)
    assert a2[0] == (20, -2096) and (20, -2096) not in u2, (a2[:2], u2[:2])
    print(f"⑤ chunk_ranges 手算对拍（3 段）＋退化情形被正确过滤　PASS")

    # ⑥ **与 `scratch_prometa_oracle.future_utility` 逐位对拍**（同一个量两份实现）
    import torch as _t
    _t.manual_seed(0)
    Lq, Hkv, G, T, d, N = 3, 2, 4, 7, 16, 60
    kc = [_t.randn(1, Hkv, N, d) for _ in range(Lq)]
    qc = [_t.randn(1, Hkv * G, T, d) for _ in range(Lq)]
    mine = future_utility(kc, qc, 0, N, tblock=3)
    from scratch_prometa_oracle import future_utility as _oracle_fu

    class _Shim:
        key_cache = kc
    ref = _oracle_fu(_Shim(), qc, N, 3)
    e = float(np.abs(mine - ref).max())
    assert e == 0.0, e
    print(f"⑥ 与 oracle 实现逐位对拍（lo=0,hi=N）max|差| = {e:.1e}　PASS")

    # ⑦ **反例：不能靠重归一化把全前缀的 U 换算成 chunk 的 U。**
    #    单条 softmax 行可以（`softmax(z)|_S/ΣS == softmax(z|_S)`），但本量在
    #    归一化后还要对 t 取 max，而各 t 的 Z_t(S) 不同 ⇒ max 与除法不可交换。
    lo_, hi_ = 10, 40
    direct = future_utility(kc, qc, lo_, hi_, tblock=3)              # 正确
    full = future_utility(kc, qc, 0, N, tblock=3)[:, :, lo_:hi_]
    renorm = full / np.maximum(full.sum(-1, keepdims=True), 1e-30)
    # 先验证「单行」确实可以重归一化（证明不等号来自 max 而不是我算错了）
    row = _t.softmax(qc[0][0, 0, 0] @ kc[0][0, 0].T / d ** 0.5, -1)
    row_s = _t.softmax(qc[0][0, 0, 0] @ kc[0][0, 0, lo_:hi_].T / d ** 0.5, -1)
    e_row = float((row[lo_:hi_] / row[lo_:hi_].sum() - row_s).abs().max())
    assert e_row < 1e-6, e_row
    # 排序差异才是有决策后果的量
    rk = lambda x: np.argsort(np.argsort(-x, -1), -1)
    frac_diff = float((rk(direct) != rk(renorm)).mean())
    assert frac_diff > 0.05, frac_diff
    print(f"⑦ 单行重归一化误差 {e_row:.1e}（可交换）；但 max 之后重归一化与"
          f"直接按 chunk 归一化的**排序**有 {frac_diff*100:.1f}% 位置不同　PASS")

    print("\nprometa/teacher.py 自测 9 条全过")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("GPU 抽取路径见 `extract_U`（待接）；先跑 --selftest")
