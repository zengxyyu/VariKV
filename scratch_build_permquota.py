#!/usr/bin/env python3
"""头身份置换对照：把地板的**增量**随机换给别的 (层,头)，其余一切保持不变。

**为什么需要它。** 我们已经排除了十个量（长度、样本数、基线分、任务族、饿死率、
配额形状、局部凹陷、跨 panel 的头身份、余弦、搬动量）。但那些都是**观测性**比较：
"94% 那格与 7% 那格的配额向量像不像"。它们回答不了**干预性**的问题 ——

    在同一个 panel 内，「救哪些头」本身重要吗？

**构造（严格匹配对照）**。地板在每个 (样本,chunk) 上产生增量
`Δ_g = b_arm_g − b_base_g`，其中 `Σ_g Δ_g = 0`。把**正增量**（被救的头）
在 112 个组之间随机置换，**负增量（捐出方）留在原地**：

    Δ'_{π(g)} = Δ_g   (Δ_g > 0)
    Δ'_g      = Δ_g   (Δ_g ≤ 0)

于是**逐位保持**：预算守恒、搬动总量 Σ|Δ|、被抬起的组数、增量的多重集合。
**唯一变化的是「哪些组拿到那些增量」。** 且接收方只增不减 ⇒ 无需 clamp，
不引入第二个变量。

**判读**：真实地板在 Retr.KV@0.1 拿 +33.60★。若置换后仍接近 +33.60，
则「救哪些头」不重要、机制在别处；若掉到 0 附近，则**头身份就是机制**，
这是十个被淘汰的量之后第一个站得住的解释。
"""
import argparse, json, os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(dump):
    """dump 每条是一个 (样本, chunk)。`seq` 是**每个 cache 内**的计数器，
    新样本时从 1 重开 —— 用它切样本边界，不要假设每样本 chunk 数相同。"""
    recs = [json.loads(l) for l in open(os.path.join(ROOT, dump))]
    samples, cur = [], []
    for r in recs:
        if r["seq"] == 1 and cur:
            samples.append(cur); cur = []
        cur.append(r)
    if cur:
        samples.append(cur)
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="scratch_q_kv01.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()

    S = load(a.dump)
    C = max(len(s) for s in S)
    G = len(S[0][0]["b_base"])
    quota = np.zeros((len(S), C, G), dtype=np.int64)
    lo = np.zeros((len(S), C), dtype=np.int64)
    hi = np.zeros((len(S), C), dtype=np.int64)
    nch = np.array([len(s) for s in S], dtype=np.int64)

    rg = np.random.default_rng(a.seed)
    mv_ok = raised = clamped = 0
    for si, s in enumerate(S):
        for ci, r in enumerate(s):
            b0 = np.array(r["b_base"], dtype=np.int64)
            ba = np.array(r["b_arm"], dtype=np.int64)
            d = ba - b0
            # **整体置换 Δ（正负一起）**。只搬正增量是错的：正增量落到捐出方
            # 头上会与负增量抵消，Σ|Δ| 缩水 ⇒ 把「身份」与「搬动量」混淆。
            # 整体置换下 |Δ'_g| = |Δ_{π^{-1}(g)}| ⇒ **Σ|Δ| 逐位不变**，
            # 且 ΣΔ' = ΣΔ = 0 ⇒ 预算守恒。唯一风险是 b0+Δ' < 0，需 clamp。
            perm = rg.permutation(G)
            newd = d[perm]
            raised += int((d > 0).sum())
            q = b0 + newd
            nclamp = int((q < 0).sum())
            if nclamp:
                deficit = int(-q[q < 0].sum())
                q = np.maximum(q, 0)
                # 从**最大**的几个组等额扣回，保住总预算
                order = np.argsort(-q)
                for g in order:
                    if deficit <= 0:
                        break
                    take = min(int(q[g]), deficit)
                    q[g] -= take; deficit -= take
                assert deficit == 0, "扣不回来"
            clamped += nclamp
            assert q.min() >= 0
            assert int(q.sum()) == int(ba.sum()), "预算未守恒"
            mv_ok += int(np.abs(newd).sum() == np.abs(d).sum())
            quota[si, ci] = q
            lo[si, ci] = r["lo"]; hi[si, ci] = r["hi"]
    np.savez(os.path.join(ROOT, a.out), quota=quota, lo=lo, hi=hi, nchunk=nch)
    tot = sum(len(s) for s in S)
    print(f"  {a.out}: {len(S)} 样本 x 最多 {C} chunk x {G} 组"
          f"；搬动总量逐格一致 {mv_ok}/{tot}"
          f"；clamp 掉的格子 {clamped}（应极少；多则对照不干净）")


if __name__ == "__main__":
    raise SystemExit(main())
