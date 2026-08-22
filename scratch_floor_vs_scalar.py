#!/usr/bin/env python3
"""**地板（`_fl8`, b_min=8）vs 学习到的保序打分器（`scalar`）—— 逐 panel × 逐 ratio。**

两条臂动的是同一件事（固定预算下改变保留集），但机制完全不同：

  · `scalar`：4,482 参数的**逐头单调形变** `s' = s⁰ + α·σ_h·tanh(φ)`，
    由 §四之五 的等价定理，其决策内容 ≡ **逐头配额重分配**（学出来的方向）；
  · `_fl8`  ：**零参数**的反饿死地板，把每个零配额头抬到 `b_min=8`，
    预算由盈余头补齐（`quota_project`），**不看任何学到的方向**。

⇒ 「学到的方向」到底值不值 4,482 个参数，就是这两条臂的差。

**方法**：三者共用同一批基线 `__g8base`，**逐样本配对** bootstrap。
`scalar` 有 3 个训练种子 ⇒ 先在**逐样本**上对种子取均值（那是该臂期望分数的
无偏估计，且压掉训练种子噪声），再与地板配对；同时报跨种子散布作稳健性。

⚠ **完整性硬闸与 `scratch_utab_report.py` 同口径**：日志要有 `Finished.`
且样本数 == 该 panel 满量，否则该格记 `—`（partial 上的 ★ 一律不可信）。
⚠ **跨 panel 读之前先看「制度」列**：GSM8K / SQuAD 在 `b_min=8` 下需求/预算
13–109%，与其余格不是同一强度的干预（见 §十一之十三 §②）。

    .venv/bin/python scratch_floor_vs_scalar.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_grid_spec import NAME, NUM, PANELS, RATIOS, tag as mktag  # noqa: E402
from scratch_read_scores import read_scores                            # noqa: E402

# **scalar 的 tag 从 `scratch_all_report.py` 的 ARMS 抄来的规则**（那里是唯一真源）：
# 过夜扫描 `__sc11_s{S}` 覆盖 11 panel × 7 ratio × 3 种子；Retr.KV@0.1 只在更早的
# `__d10scalar_s{S}` 里。两个都试，取先命中的。
# ⚠ **必须传短 tag**：`read_scores` 会自己补 `_chunk16k_w4096` 后缀，
# 传全名反而匹配不到（首版就是这么全空的）。另：过夜扫描 `__sc11_s*`
# **不含 Retr.KV@0.1**（返回 n=0，不是报错），必须回退到 `__d10scalar_s*`
# —— 所以「n==0 也算没有」这条判定不能省。
SCALAR_TAGS = ["__sc11_s{S}", "__d10scalar_s{S}"]
SEEDS = (0, 1, 2)
# 高剂量制度的格（`scratch_floor_report.py` 生成，>5% 需求/预算）
HIDOSE = {("gsm", r) for r, _ in RATIOS} | {("squad", 0.1)}


def finished(tag):
    lg = os.path.join("scratch_ctrl_logs", tag.lstrip("_").split("_chunk")[0] + ".log")
    return os.path.exists(lg) and "Finished." in open(lg, errors="ignore").read()


def get(ds, tag, r, need):
    try:
        v = read_scores(ds, tag, r)
    except Exception:
        return None
    return v if (v and len(v) == need) else None


def boot(d, seed=0, B=20000):
    rs = np.random.default_rng(seed)
    s = d[rs.integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    return d.mean(), lo, hi, ("★" if lo > 0 or hi < 0 else "ns")


def main():
    print("# 地板 `_fl8`(b_min=8) vs `scalar`(4,482 参数) —— 逐 panel × 逐 ratio")
    print("# Δ 均相对同一批 `__g8base`，×100，逐样本配对 bootstrap，★ = 95% CI 排除 0\n")
    hdr = (f"{'panel':<16}{'ρ':>6}{'n':>5}{'基线':>8}"
           f"{'Δ 地板':>12}{'Δ scalar':>14}{'配对 地板−scalar':>22}{'制度':>8}")
    print(hdr); print("-" * len(hdr))
    tally = dict(floor_win=0, scalar_win=0, tie=0, cells=0)
    tally_fine = dict(floor_win=0, scalar_win=0, tie=0, cells=0)
    # 「谁更好」不能只看配对：还要分别看**各自伤害了几格**（安全性）
    arm_stat = {a: dict(pos=0, neg=0, ssum=0.0, n=0) for a in ("floor", "scalar")}
    by_panel = {}
    for ds, code, need, nm, fam in PANELS:
        for r, _rc in RATIOS:
            ft = mktag(ds, r, "_fl8")
            if not finished(ft):
                continue
            b = get(ds, "__g8base", r, need)
            f = get(ds, ft, r, need)
            if b is None or f is None:
                continue
            # scalar：逐种子取到的样本上求逐样本均值
            per = []
            for S in SEEDS:
                for t in SCALAR_TAGS:
                    v = get(ds, t.format(S=S), r, need)
                    if v is not None:
                        per.append(v); break
            if not per:
                k = sorted(b)
                df = np.array([f[i] - b[i] for i in k]) * 100
                m, lo, hi, st = boot(df)
                print(f"{nm:<16}{r:>6}{len(k):>5}{100*np.mean(list(b.values())):>8.2f}"
                      f"{m:>+10.2f}{st:<2}{'—':>14}{'—':>22}"
                      f"{'高剂量' if (ds,r) in HIDOSE else '微调':>8}")
                continue
            k = sorted(set(b) & set(f) & set.intersection(*[set(x) for x in per]))
            df = np.array([f[i] - b[i] for i in k]) * 100
            sc_per_sample = np.array([np.mean([x[i] for x in per]) for i in k])
            ds_ = (sc_per_sample - np.array([b[i] for i in k])) * 100
            dd = df - ds_
            mf, _, _, sf = boot(df)
            ms, _, _, ss = boot(ds_, seed=1)
            md, lo, hi, sd = boot(dd, seed=2)
            spread = np.std([100 * np.mean([x[i] - b[i] for i in k]) for x in per])
            reg = "高剂量" if (ds, r) in HIDOSE else "微调"
            print(f"{nm:<16}{r:>6}{len(k):>5}{100*np.mean([b[i] for i in k]):>8.2f}"
                  f"{mf:>+10.2f}{sf:<2}{ms:>+10.2f}{ss:<2}±{spread:<3.1f}"
                  f"{md:>+11.2f}{sd:<2}[{lo:+.1f},{hi:+.1f}]{reg:>8}")
            tally["cells"] += 1
            key = "floor_win" if (sd == "★" and md > 0) else \
                  "scalar_win" if (sd == "★" and md < 0) else "tie"
            tally[key] += 1
            if reg == "微调":
                tally_fine["cells"] += 1
                tally_fine[key] += 1
                for a, mm, stt in (("floor", mf, sf), ("scalar", ms, ss)):
                    arm_stat[a]["n"] += 1
                    arm_stat[a]["ssum"] += mm
                    if stt == "★":
                        arm_stat[a]["pos" if mm > 0 else "neg"] += 1
                q = by_panel.setdefault(nm, dict(f=0, s=0, t=0))
                q["f" if key == "floor_win" else
                  "s" if key == "scalar_win" else "t"] += 1
    print("\n## 合计")
    for nmm, t in [("全部格", tally), ("**只算微调制度**", tally_fine)]:
        print(f"{nmm}：{t['cells']} 格 —— 地板显著更好 **{t['floor_win']}**、"
              f"scalar 显著更好 **{t['scalar_win']}**、不可分 {t['tie']}")
    print("\n## 只算微调制度：两条臂各自的收益与伤害")
    for a in ("floor", "scalar"):
        t = arm_stat[a]
        print(f"  {a:<7} {t['n']} 格：显著为正 **{t['pos']}**、"
              f"**显著为负 {t['neg']}**、Δ 合计 {t['ssum']:+.2f}、"
              f"均值 {t['ssum']/max(t['n'],1):+.2f}")
    print("\n## 逐 panel（只算微调制度）：地板赢 / scalar 赢 / 不可分")
    for nm, q in by_panel.items():
        print(f"  {nm:<16} {q['f']} / {q['s']} / {q['t']}")
    print("\n⚠ 「不可分 ≠ 等价」。⚠ 高剂量格（GSM8K / SQuAD@0.1）与其余格"
          "不是同一强度的干预，不参与横向结论。")
    print("⚠ `scalar` 用逐样本跨 3 种子均值（压训练种子噪声）；括号后的 ± 是"
          "跨种子 Δ 的散布，读单格前先看它。")


if __name__ == "__main__":
    main()
