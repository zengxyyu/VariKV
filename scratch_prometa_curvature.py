#!/usr/bin/env python3
"""**决定性的上游问题**：下游准确率对「保住的未来需求质量」是线性的还是饱和的？
（零 GPU，只读已有的评测结果目录。）

────────────────────────────────────────────────────────────────────────────
为什么这一条排在所有 ProMeta 决策层设计之前
────────────────────────────────────────────────────────────────────────────
`scratch_prometa_objmatrix.py` 测到：在

    L_mean = (1/M) Σ_m Σ_{i∉S} U_{m,i}

上，**`mean` 规则是构造性精确最优**（线性目标 + 基数约束 ⇒ top-B 就是精确解），
其余 224/224 格全输。而 SCBench 的下游指标**就是 M 个问题的平均分**。

⇒ **只要「单题准确率 ≈ 该题保住质量的线性函数」，β=0 / 均值就可证最优，
任何未来聚合规则（熵风险、soft-min、比例公平、max-min）都没有空间。**
反过来，只有当单题准确率对保住质量**饱和**（凹）时，「别让任何一个未来饿死」
才可能赢 —— 那正是比例公平 `Σ_m log(ε+F_m)` 与地板故事的共同前提。

**这个前提可以从已有结果直接检验**：SCBench 每条样本自带 M 个问题，
`output-pair.json` 里 `qa` / `qa-1` / … **逐题落盘**。判分复用上游
`results.metric`（不手写第二份判分器）。

────────────────────────────────────────────────────────────────────────────
三条预注册读法
────────────────────────────────────────────────────────────────────────────
① **过散 vs 欠散**：把「每条样本答对几题」与独立零模型 `Binomial(M, p̄(ρ))`
   比方差。
   · 方差 **≫** 二项 ⇒ 样本整体全对或全错 ⇒ 这是**逐样本**效应，
     未来之间不争资源 ⇒ **公平型目标没有立足点**。
   · 方差 **≈** 二项 ⇒ 题目近似独立 ⇒ **均值就是对的聚合**。
   · 方差 **≪** 二项 ⇒ 题目此消彼长 ⇒ **样本内未来在争同一份预算**，
     这才是分配型方法的证据。
② **单题曲线的曲率**：`acc_m(ρ)` 的二阶差分符号。凹 ⇒ 饱和 ⇒ 有空间。
③ **失败是不是集中在固定的题**：若某一题恒定最难，那是题目难度不是分配。
   报「逐题准确率的跨题散布」与「失败题身份在 ρ 之间的一致性」。

    .venv/bin/python scratch_prometa_curvature.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))


def load_cells(ds, tag, model="qwen2.5-7b-instruct-1m"):
    """→ {ratio: {sample_idx: [每题得分]}}，判分复用上游 `results.metric`。"""
    from results.metric import f1_score, include_score
    # **判分必须按 panel 派发**（`results/metric.py:187-215` 的分支）：
    # `scbench_qa_eng` 是 `max(f1_score, include_score)`，**连续值**；
    # 只用 `include_score` 会给出一个不是该 panel 指标的数（第④类错）。
    def _score(pred, ref):
        if "qa_eng" in ds:
            return float(max(f1_score(pred, ref), include_score(pred, ref)))
        return float(include_score(pred, ref))
    root = os.path.join(HERE, "external/FastKVzip/prefill/results", ds)
    pat = os.path.join(root, f"*_{model}_*{tag}_*", "output-pair.json")
    files = sorted(glob.glob(pat))
    assert files, f"没有匹配：{pat}"
    out = {}
    for f in files:
        idx = int(os.path.basename(os.path.dirname(f)).split("_")[0])
        d = json.load(open(f))
        keys = [k for k in d if k == "qa" or k.startswith("qa-")]
        keys.sort(key=lambda k: 0 if k == "qa" else int(k.split("-")[1]))
        for k in keys:
            for entry in d[k]:
                r = float(entry[0][0])
                rec = entry[1]
                s = _score(rec["pruned"], rec["answer"])
                out.setdefault(r, {}).setdefault(idx, []).append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="scbench_kv")
    ap.add_argument("-t", "--tag", default="_g8base")
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    cells = load_cells(a.data, a.tag)
    rs = np.random.default_rng(0)

    print(f"# {a.data} tag={a.tag}　逐题曲率与分配特征（零 GPU）\n")
    print(f"{'ρ':>6} {'n':>5} {'M':>3} {'均值 acc':>9} {'每样本答对数 方差':>17}"
          f" {'二项零模型 方差':>16} {'方差比':>8}  判读")
    rows = []
    for r in sorted(cells, reverse=True):
        per = cells[r]
        # **按众数 M 分组，不是 max**：面板内未来数可能不齐（`scbench_qa_eng`
        # 实测 3 个样本 M=5、1 个 M=7）。取 max 会把绝大多数样本筛掉、
        # 然后在空数组上崩 —— 首版就是这么崩的。**也不许截断 M**（那会改变
        # 被测的未来集合）。
        from collections import Counter
        M = Counter(len(v) for v in per.values()).most_common(1)[0][0]
        keep = {i: v for i, v in per.items() if len(v) == M}
        if len(keep) < 8:
            print(f"{r:>6} {len(keep):>5} {M:>3}   样本不足 8 条，跳过（功效不够）")
            continue
        A = np.array([keep[i] for i in sorted(keep)], float)     # [n, M]
        # **二值性硬闸**（外部复核指出，采纳）：`Binomial(M,p)` 只有在得分
        # 严格 0/1 时才是正确的零模型。连续得分（如 qa_eng 的 f1）下
        # `M·p(1−p)` 不是 Var(K)，整张表会读错。
        uq = np.unique(A)
        if not set(uq.tolist()) <= {0.0, 1.0}:
            print(f"{r:>6} {A.shape[0]:>5} {M:>3}   **得分非二值**"
                  f"（unique 前 5 个 {uq[:5]}）⇒ Binomial 零模型不适用，跳过")
            continue
        n = A.shape[0]
        p = A.mean()
        k = A.sum(1)
        var_obs = k.var(ddof=1)
        var_bin = M * p * (1 - p)
        # 方差比的 bootstrap CI（重采样样本）
        bs = []
        for _ in range(a.boot):
            j = rs.integers(0, n, n)
            kk = A[j].sum(1)
            pp = A[j].mean()
            vb = M * pp * (1 - pp)
            bs.append(kk.var(ddof=1) / max(vb, 1e-12))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ratio = var_obs / max(var_bin, 1e-12)
        verdict = ("**过散（逐样本效应）**" if lo > 1 else
                   "**欠散（样本内争资源）**" if hi < 1 else "与独立不可分")
        print(f"{r:>6} {n:>5} {M:>3} {p:>9.4f} {var_obs:>17.4f}"
              f" {var_bin:>16.4f} {ratio:>8.3f}  {verdict} [{lo:.2f},{hi:.2f}]")
        rows.append((r, p, A))

    if len(rows) < 3:
        print("\n可用 ρ 少于 3 个，曲率与失败题分析跳过（**不是通过，是没查**）")
        return
    # ② 曲率
    rows.sort()
    rr = np.array([x[0] for x in rows]); pp = np.array([x[1] for x in rows])
    print(f"\n## ② 均值 acc 对 ρ 的曲率（二阶差分，<0 = 凹 = 饱和）")
    for i in range(1, len(rr) - 1):
        h1, h2 = rr[i] - rr[i - 1], rr[i + 1] - rr[i]
        d2 = 2 * ((pp[i + 1] - pp[i]) / h2 - (pp[i] - pp[i - 1]) / h1) / (h1 + h2)
        print(f"   ρ={rr[i]:<5} acc={pp[i]:.4f}  二阶差分={d2:+.3f}"
              f"  {'凹（饱和）' if d2 < 0 else '凸'}")

    # ③ 失败是否集中在固定的题
    print(f"\n## ③ 失败题身份是否固定（若固定 ⇒ 是题目难度不是分配）")
    for r, p, A in rows:
        pm = A.mean(0)
        print(f"   ρ={r:<5} 逐题 acc={np.array2string(pm, precision=3)}"
              f"  跨题散布 sd={pm.std():.4f}")
    base = rows[-1][2].mean(0)                 # 最大 ρ 作参照
    for r, p, A in rows[:-1]:
        c = np.corrcoef(A.mean(0), base)[0, 1] if A.mean(0).std() > 0 else float("nan")
        print(f"   ρ={r:<5} 逐题 acc 与 ρ={rows[-1][0]} 的相关={c:+.3f}"
              f"  {'（难度固定）' if c > 0.7 else ''}")

    print("\n## 判词")
    print("· 方差比 ≫1 ⇒ 逐样本效应，**公平型目标（比例公平 / max-min）没有立足点**；")
    print("· 方差比 ≈1 ⇒ 题目独立，**均值聚合可证最优，β 与分配层都没有空间**；")
    print("· 方差比 ≪1 ⇒ 样本内未来争同一份预算，**分配层值得做**。")


if __name__ == "__main__":
    main()
