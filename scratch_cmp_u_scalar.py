"""u 表 vs scalar：**同格逐样本配对**（不是两个 Δ 相减）。

为什么必须逐样本配对：两条臂对同一批基线各自算 Δ，再把两个 Δ 相减，得到的区间
是两个独立 bootstrap 的差，会高估方差；而且若两臂样本集不完全相同（某臂少跑了
几条），相减就是在比不同的东西。这里取三方交集后做配对 bootstrap。
"""
import sys
import numpy as np
sys.path.insert(0, "/home/ubuntu/zxy/vlm-memory")
from scratch_read_scores import read_scores            # noqa: E402
from scratch_all_report import BASE_SFX, per_sample    # noqa: E402

CELLS = [
    ("scbench_kv", "Retr.KV", [0.1, 0.2, 0.3, 0.5], "_uq01kv{t}"),
    ("scbench_prefix_suffix", "Retr.PrefSuf", [0.1, 0.2, 0.3, 0.4, 0.5, 0.75], "_uq01psr{t}"),
    ("scbench_vt", "Retr.MultiHop", [0.1, 0.2, 0.3, 0.4, 0.5, 0.75], "_uq01vt{t}"),
]
# u 表各格的实际 tag（命名不统一，逐格写死比猜规则可靠）
UTAG = {
    ("scbench_kv", 0.1): "_uq01r01", ("scbench_kv", 0.2): "_uq01r02",
    ("scbench_kv", 0.3): "_uq01r03", ("scbench_kv", 0.5): "_uq01r05",
    ("scbench_prefix_suffix", 0.1): "_uq01psr01",
    ("scbench_prefix_suffix", 0.2): "_uq01psr02",
    ("scbench_prefix_suffix", 0.3): "_uq01psr03",
    ("scbench_prefix_suffix", 0.4): "_uq01psr04",
    ("scbench_prefix_suffix", 0.5): "_uq01psr05",
    ("scbench_prefix_suffix", 0.75): "_uq01psr075",
    ("scbench_vt", 0.1): "_uq01vt01", ("scbench_vt", 0.2): "_uq01vt02",
    ("scbench_vt", 0.3): "_uq01vt03", ("scbench_vt", 0.4): "_uq01vt04",
    ("scbench_vt", 0.5): "_uq01vt05", ("scbench_vt", 0.75): "_uq01vt075",
}
SC_TPL = ("__sc11_s{S}_chunk16k_w4096_ctrlmmemo8_scalar",
          "__d10scalar_s{S}_chunk16k_w4096_ctrlmstat8_scalar")


def boot(dif, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(dif)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return d.mean() * 100, np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100


def scalar_ps(ds, r):
    """逐样本取种子平均。只用**每个种子都跑满**的样本，避免种子间样本集不齐。"""
    per = []
    for S in (0, 1, 2):
        for tpl in SC_TPL:
            try:
                s = per_sample(ds, tpl.format(S=S), r)
            except Exception:
                s = None
            if s:
                per.append(s)
                break
    if not per:
        return None, 0
    keys = set(per[0])
    for p in per[1:]:
        keys &= set(p)
    return {k: float(np.mean([p[k] for p in per])) for k in keys}, len(per)


print(f"{'panel':14s} {'ρ':>5s} {'n':>4s} {'u表Δ':>9s} {'scalarΔ':>9s} {'u−scalar':>10s}  95%CI")
tot = {"u": 0, "s": 0, "tie": 0}
for ds, pname, ratios, _ in CELLS:
    for r in ratios:
        ut = UTAG.get((ds, r))
        try:
            U = read_scores(ds, ut, r)
        except Exception:
            U = None
        B = per_sample(ds, BASE_SFX.get(r, "__g8base_chunk16k_w4096"), r)
        SC, nseed = scalar_ps(ds, r)
        if not U or not B or not SC:
            print(f"{pname:14s} {r:>5} {'—':>4s}  缺 " +
                  ",".join([x for x, v in (("u", U), ("base", B), ("scalar", SC)) if not v]))
            continue
        k = sorted(set(U) & set(B) & set(SC))
        du = np.array([U[i] - B[i] for i in k])
        dsq = np.array([SC[i] - B[i] for i in k])
        mu, _, _ = boot(du)
        ms, _, _ = boot(dsq)
        md, lo, hi = boot(du - dsq)
        star = "*" if lo * hi > 0 else " "
        if lo * hi > 0:
            tot["u" if md > 0 else "s"] += 1
        else:
            tot["tie"] += 1
        print(f"{pname:14s} {r:>5} {len(k):>4d} {mu:>+9.2f} {ms:>+9.2f} "
              f"{md:>+9.2f}{star} [{lo:+.2f},{hi:+.2f}] (种子{nseed})")
print(f"\nu 表显著更好 {tot['u']} 格 ・ scalar 显著更好 {tot['s']} 格 ・ 不可分 {tot['tie']} 格")
