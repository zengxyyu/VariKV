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

HEX = "0123456789abcdef"
FORMAT_RULE = (" Formatting rule: every answer must be written as "
               "ANSWER=<value> with no spaces. ")


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
    spans.append(enc(FORMAT_RULE))                       # 最后一段是全共享的

    lo, hi = int(0.05 * (len(ctx) - window)), int(0.90 * (len(ctx) - window))
    assert hi - lo > len(spans), f"可驱逐区太短：[{lo},{hi}) 放不下 {len(spans)} 段"
    pos = sorted(rng.sample(range(lo, hi), len(spans)), reverse=True)
    for p_, sp in zip(pos, spans):
        ctx[p_:p_] = sp

    tail = " Answer using the formatting rule defined earlier."
    futures = []
    for j in range(n_fact):                              # 私有
        futures.append(dict(
            q=enc(f"\nQuestion: What value does the secret key {keys[j]} "
                  f"map to?{tail}\nAnswer:"),
            a=enc(f" ANSWER={vals[j]}"), needs=[j], kind="single"))
    for t in range(n_joint):                             # 共享
        a_, b_ = rng.sample(range(n_fact), 2)
        futures.append(dict(
            q=enc(f"\nQuestion: Give the values for the secret keys {keys[a_]} "
                  f"and {keys[b_]}.{tail}\nAnswer:"),
            a=enc(f" ANSWER={vals[a_]} ANSWER={vals[b_]}"),
            needs=sorted([a_, b_]), kind="joint"))
    meta = dict(n_fact=n_fact, n_joint=n_joint, M=len(futures),
                pos=sorted(pos), fmt_pos=pos[[i for i, p in
                                              enumerate(sorted(pos, reverse=True))
                                              ][0]] if pos else None,
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
    assert len(set(meta["pos"])) == len(meta["pos"])
    assert all(lo <= p < hi for p in meta["pos"]), (lo, hi, meta["pos"])
    assert len(ctx) > 8000
    print(f"② {len(meta['pos'])} 个插入点全在 [{lo},{hi}) 内且互不相同　PASS")

    # ③ **共享结构真的存在**：至少一个事实被 ≥2 个未来需要
    from collections import Counter
    c = Counter(j for f in fut for j in f["needs"])
    assert max(c.values()) >= 2, c
    print(f"③ 事实被需要次数 {dict(c)} —— 存在共享需求　PASS")

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
    print("\nprometa/teacher.py 自测 4 条全过")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print("GPU 抽取路径见 `extract_U`（待接）；先跑 --selftest")
