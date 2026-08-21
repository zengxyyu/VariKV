"""把 `sel-orc` 每一格的**选择与理由**摊开 —— 回答「0.00 到底是怎么来的」。

规则（抄自 scratch_all_report.py:441-449）：起点 0.0，只有**显著★且为正**的候选
才有资格参与取最大 ⇒ 两个候选都不显著为正时，结果恒为 0.00（= 选 g=0 不动）。
"""
import glob
import os
import sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")
import scratch_all_report as A                                   # noqa: E402

RAT = [r for r in A.RAT if r != 1.0]
SC_TPL = ("__sc11_s{S}_chunk16k_w4096_ctrlmmemo8_scalar",
          "__d10scalar_s{S}_chunk16k_w4096_ctrlmstat8_scalar")
RES = "/home/ubuntu/zxy/vlm-memory/external/FastKVzip/prefill/results"


def arm_delta(d, r, tpls_or_sfx, seeds=(0, 1, 2)):
    """→ (Δ, 显著, n, 种子数)；没有数据返回 None。"""
    B = A.per_sample(d, A.BASE_SFX.get(r, "__g8base_chunk16k_w4096"), r)
    if not B:
        return None
    per = []
    if isinstance(tpls_or_sfx, tuple):
        for S in seeds:
            for t in tpls_or_sfx:
                try:
                    x = A.per_sample(d, t.format(S=S), r)
                except Exception:
                    x = None
                if x:
                    per.append(x)
                    break
    else:
        try:
            x = A.per_sample(d, tpls_or_sfx, r)
        except Exception:
            x = None
        if x:
            per.append(x)
    if not per:
        return None
    ks = set(per[0])
    for p in per[1:]:
        ks &= set(p)
    ks &= set(B)
    if not ks:
        return None
    ks = sorted(ks)
    dd = np.array([np.mean([p[i] for p in per]) - B[i] for i in ks])
    m, lo, hi = A.boot(dd)
    return m * 100, (lo * hi > 0), len(ks), len(per)


hdr = (f"{'panel':15s}{'ρ':>6s} | {'scalar Δ':>11s} | {'gm1 Δ':>11s} | "
       f"{'sel-orc':>8s} | 理由")
print(hdr)
print("-" * len(hdr))
cnt = {"选scalar": 0, "选gm1": 0, "选不动(0.00)": 0}
nogm_cells = 0
for d, name in A.PANEL.items():
    for r in RAT:
        gsfx = A.gm1_sfx(d, r)
        has_gm_dir = bool(gsfx) and bool(
            glob.glob(os.path.join(RES, d, f"*{gsfx}*")))
        sc = arm_delta(d, r, SC_TPL)
        gm = arm_delta(d, r, gsfx) if has_gm_dir else None
        if sc is None and gm is None:
            continue
        cands = []
        if sc and sc[1] and sc[0] > 0:
            cands.append(("scalar", sc[0]))
        if gm and gm[1] and gm[0] > 0:
            cands.append(("gm1", gm[0]))
        if cands:
            pick, val = max(cands, key=lambda x: x[1])
            why = f"{pick} 显著为正且更大"
            cnt["选scalar" if pick == "scalar" else "选gm1"] += 1
        else:
            pick, val = "不动", 0.0
            bits = []
            bits.append("scalar 不显著为正" if sc else "scalar 无数据")
            bits.append("gm1 不显著为正" if gm else ("gm1 **未跑**" if not has_gm_dir
                                                    else "gm1 无数据"))
            why = " + ".join(bits)
            cnt["选不动(0.00)"] += 1
        if not has_gm_dir:
            nogm_cells += 1
        f = lambda t: ("—" if t is None else                      # noqa: E731
                       f"{t[0]:+.2f}{'*' if t[1] else ' '}")
        print(f"{name:15s}{r:>6} | {f(sc):>11s} | {f(gm):>11s} | "
              f"{val:>+8.2f} | {why}")
print()
print("统计：" + " ・ ".join(f"{k} {v} 格" for k, v in cnt.items()))
print(f"其中 **gm1 从未跑过** 的格：{nogm_cells} —— 这些格的 sel-orc 只是"
      f"「scalar vs 不动」的二选一，**不是三选一**。")
