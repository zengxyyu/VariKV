"""质心容量扫描的报告：绝对分数 + 逐样本配对 bootstrap + HRR + 等预算差。

**判据是 Δ(质心 − matched-budget)，不是 Δ(质心 − 基线)。** 只跟基线比会被问死：
"这些额外字节为什么不直接用来多留 KV"。一个质心 2d+1=257 scalars、一条 exact KV
2d=256 scalars ⇒ K 个质心 ≈ K 条 exact KV，两者是公平对决。实测保留量
16903/(层,head) @ratio 0.1 ⇒ K=1024 只多花 6.08% ⇒ 对照档是 ratio 0.1061。

不用 `results.parse` 的相对行：每个 run 用自己的满缓存分数做分母，跨 run 不可比
（2026-08-10/11 两次踩过）。这里读逐样本 json、调 harness 自己的 `evaluate_answer`
算绝对分，并把 ratio-1.0 那档当**一致性自检**：质心修正在满缓存前向上不生效
（非 flatten 分支拿不到 lse），所以各档的 `full__` 必须彼此相同。

用法： .venv/bin/python scratch_centroid_report.py
"""
import contextlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "external/FastKVzip/prefill"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "external/FastKVzip/prefill"))
from results.parse import parse_answer, evaluate_answer   # noqa: E402

DATA, TASK, MODEL = "scbench_kv", "qa", "qwen2.5-7b-instruct-1m"
D = 128
RETAINED_PER_HEAD = 16903          # 实测 @ratio 0.1

# (名字, 结果目录后缀, 该档的 ratio, K)
ARMS = [
    ("matched-budget（多留真实 KV）", "_mb1024_chunk16k_w4096", 0.1061, 0),
    ("质心 K=16", "_cen16_chunk16k_w4096_cen16", 0.1, 16),
    ("质心 K=109", "_cen109_chunk16k_w4096_cen109", 0.1, 109),
    ("质心 K=256", "_cen256_chunk16k_w4096_cen256", 0.1, 256),
    ("质心 K=1024", "_cen1024_chunk16k_w4096_cen1024", 0.1, 1024),
    ("质心 K=109 (inv-RoPE)", "_cen109inv_chunk16k_w4096_cen109_inv", 0.1, 109),
]

_MUTE = contextlib.redirect_stdout(io.StringIO())
with _MUTE:
    ANSW, SUBT = parse_answer(DATA)


def per_sample(suffix, ratio):
    """→ ({sample_idx: 该 ratio 的均分}, {sample_idx: full__ 的均分})，照抄 parse.py。"""
    pr, fu = {}, {}
    i = 0
    while True:
        f = f"./results/{DATA}/{i}_{MODEL}_fastkvzip_{suffix}/output-{'pair'}.json"
        if not os.path.exists(f):
            break
        d = json.load(open(f))
        p, q, answers = [], [], []
        for fmt in [k for k in d if k.startswith(TASK)]:
            for info, text in d[fmt]:
                if abs(float(info[0]) - ratio) < 1e-9:
                    p.append(text["pruned"])
                    q.append(text["full__"])
            answers.append(text["answer"])
        gold = ANSW[i] if ANSW else answers
        sub = SUBT[i] if SUBT else None
        if p:
            with _MUTE:
                pr[i] = float(np.mean(evaluate_answer(p, gold, DATA, TASK,
                                                      subtask=sub)))
                fu[i] = float(np.mean(evaluate_answer(q, gold, DATA, TASK,
                                                      subtask=sub)))
        i += 1
    return pr, fu


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    dif = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(dif), (n, len(dif)))
    s = dif[idx].mean(1)
    return dif.mean(), float(np.quantile(s, .025)), float(np.quantile(s, .975))


def main():
    got = {}
    for name, suf, ratio, K in ARMS:
        pr, fu = per_sample(suf, ratio)
        got[name] = (pr, fu, K)
        print(f"{name:34s} n={len(pr):3d}" + ("" if pr else "   ← 还没有结果"))
    have = {k: v for k, v in got.items() if v[0]}
    if len(have) < 2:
        print("\n结果不足，稍后再跑。")
        return
    common = sorted(set.intersection(*[set(v[0]) for v in have.values()]))
    print(f"\n共同样本 = {len(common)}")

    # ---- 自检：各档的 full__ 必须一致（质心修正不进满缓存前向） ----
    fulls = {k: np.mean([v[1][i] for i in common]) * 100 for k, v in have.items()}
    spread = max(fulls.values()) - min(fulls.values())
    print(f"[自检] 各档 full__ = " + "  ".join(f"{v:.2f}" for v in fulls.values())
          + f"   极差 {spread:.2f} "
          + ("✓ 一致，满缓存参照干净" if spread < 0.51 else
             "⚠ 不一致！参照被污染，Δ 不可信"))
    FULL = float(np.mean(list(fulls.values())))
    BASE = fulls and None

    mb_key = "matched-budget（多留真实 KV）"
    arr = {k: np.array([have[k][0][i] for i in common]) * 100 for k in have}
    base_mb = arr.get(mb_key)

    print("\n" + "=" * 110)
    print(f"{'配置':34s}{'绝对分':>9}{'额外字节':>9}{'HRR':>8}"
          f"{'  Δ vs matched-budget (95% CI)':>34}")
    print("-" * 110)
    for name, suf, ratio, K in ARMS:
        if name not in arr:
            continue
        s = arr[name]
        extra = (K * (2 * D + 1) / (RETAINED_PER_HEAD * 2 * D) * 100) if K else 6.08
        hrr = (s.mean() - (base_mb.mean() if base_mb is not None else s.mean())) \
            / max(FULL - s.mean(), 1e-9) * 100
        if base_mb is None or name == mb_key:
            cell = "（本档即对照）"
            hrr_s = "     —"
        else:
            m, lo, hi = boot(s, base_mb)
            star = "★" if (lo > 0 or hi < 0) else " "
            cell = f"{m:+7.2f} [{lo:+6.2f},{hi:+6.2f}]{star}"
            hrr_s = f"{hrr:>5.1f}%"
        print(f"{name:34s}{s.mean():>9.2f}{extra:>8.2f}%{hrr_s:>8}{cell:>34}")
    print("-" * 110)
    print(f"参照：满缓存 {FULL:.2f}（本批自测）")
    print("=" * 110)
    print("判读（预注册，跑之前定死）：")
    print("  质心显著 > matched-budget  ⇒ GO：概括被驱逐集合比多留精确 token 更值")
    print("  质心 > 基线但 ≤ matched    ⇒ 无 rate–quality 优势，作为压缩方法 NO-GO")
    print("  K=16 ≈ K=1024 且都涨       ⇒ 病因是接线（代数），容量结论作废")
    print("  全部 ≈ 对照               ⇒ 局部修复换不成精度 ⇒ 才轮到答案端 KL 蒸馏")
    print("  inv ≪ post                ⇒ 容量曲线混着 RoPE 相位效应，需重新归因")


if __name__ == "__main__":
    main()
