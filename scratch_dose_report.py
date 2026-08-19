#!/usr/bin/env python3
"""剂量–反应（`floorpath` λ 扫描）的读数与**机械判读**。

**为什么要有这个脚本而不是肉眼看**：判读表是在看到数字之前写进 `RESULTS_ABLATION.md`
的（预注册）。如果最后由我肉眼判断「这算不算单调上升」，预注册就形同虚设——
本项目已经在「峰位」「argmax」上栽过两次，都是事后挑标准。所以把三条判据写成代码。

λ 的含义：目标配额 `t = (1−λ)·b⁰ + λ·q_floor`，**直接注入**（不经过分数参数化，
所以不需要可达）。λ=0 ≡ 基线，λ=1 ≡ 地板。

预注册的三条判读（与文档逐字一致）：

  A 单调上升   ：点估计逐点不降，**且** Δ(1) − Δ(最小 λ) 配对分离
                 ⇒ 前提「地板收益来自抬饿死头」成立
  B 小 λ 早饱和：Δ(最小 λ) ≥ 0.7 × Δ(1)，**且** Δ(1) − Δ(最小 λ) **不**分离
                 ⇒ 收益另有来源；「整个 Q_box 只能抬 2.25%」与这个缺口无关
  C 非单调     ：存在内部 λ 显著**高于** Δ(1)
                 ⇒ 抬升过量有害，两条都不成立，需要新机制假说

三条互斥但**不穷尽**：可能三条都不满足（例如缓慢上升但两端不分离），
那时脚本报「不判定」，**不允许硬套最接近的一条**。

用法：.venv/bin/python scratch_dose_report.py            # 全部已完成的格
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_read_scores import read_scores, paired

LOGD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_ctrl_logs")
FULL_N = {"scbench_kv": 100, "scbench_prefix_suffix": 100, "scbench_vt": 90}

# (ds, ratio, {λ: tag})。λ=1 用同格的**地板 b8** 作为终点（floorpath λ=1 与它等价，
# 已在单测第 7 组逐位验过），不另跑一遍。
# `REQUIRED` = **预注册的 λ 集合**。判定必须在这一整套上做，缺一个就只报数据、
# 不下判词。否则会出现：先用子集判一次「A 单调」，λ=0.1 到了变成「不判定」，
# 于是我挑先前那个——这正是预注册要防的标准漂移，而本项目今天已经在
# 「峰位」与「argmax」上栽过两次。
CELLS = [
    ("scbench_kv", 0.1, {0.1: "_kvfl010", 0.25: "_kvfl025", 0.5: "_kvfl05",
                         0.75: "_kvfl075", 1.0: "_flr01a"},
     {0.1, 0.25, 0.5, 0.75, 1.0}),
    ("scbench_kv", 0.2, {0.25: "_kv2fl025", 0.5: "_kv2fl05", 1.0: "_flr8"},
     {0.25, 0.5, 1.0}),
]


def done(tag):
    p = os.path.join(LOGD, tag.lstrip("_") + ".log")
    return os.path.exists(p) and b"Finished." in open(p, "rb").read()


def main():
    for ds, r, lam_tag, required in CELLS:
        full = FULL_N[ds]
        base = read_scores(ds, "_g8base", r)
        assert len(base) == full, f"基线未跑满 {len(base)}/{full}"
        print(f"\n=== {ds} @ρ={r}   基线 {np.mean(list(base.values()))*100:.2f}"
              f"   n={full} ===")
        pts, miss = [], []
        for lam in sorted(lam_tag):
            tag = lam_tag[lam]
            if not done(tag):
                miss.append((lam, tag, "未完成")); continue
            a = read_scores(ds, tag, r)
            if len(a) != full:
                miss.append((lam, tag, f"n={len(a)}/{full}")); continue
            m, lo, hi, _ = paired(a, base)
            pts.append((lam, m, lo, hi, a))
            print(f"  λ={lam:<5} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]"
                  f"{'*' if lo > 0 or hi < 0 else ''}   ({tag})")
        for lam, tag, why in miss:
            print(f"  λ={lam:<5} {'—':>7}   {tag} {why}")
        have = {p[0] for p in pts}
        lack = sorted(required - have)
        if lack:
            print(f"  ⇒ **不判定（只报数据）**：预注册的 λ 集合还缺 {lack}。"
                  f"\n     **不允许先在子集上判一次** —— 那样 λ 补齐后若判词改变，"
                  f"我就会挑先前那个。")
            continue
        if len(pts) < 3 or pts[-1][0] != 1.0:
            print("  ⇒ **不判定**：点不足或缺 λ=1 终点")
            continue

        # 逐对相邻检验（供人读，不进判据）
        for i in range(1, len(pts)):
            m, lo, hi, _ = paired(pts[i][4], pts[i - 1][4])
            print(f"    λ={pts[i][0]} − λ={pts[i-1][0]}: {m:+6.2f} "
                  f"[{lo:+6.2f},{hi:+6.2f}]{'*' if lo > 0 or hi < 0 else ' ns'}")

        lo_lam, lo_d = pts[0][0], pts[0][1]
        hi_d = pts[-1][1]
        m_end, l_end, h_end, _ = paired(pts[-1][4], pts[0][4])
        end_sep = (l_end > 0 or h_end < 0)
        nondec = all(pts[i][1] >= pts[i - 1][1] - 1e-9 for i in range(1, len(pts)))
        # C：内部点显著高于 λ=1
        cbeat = []
        for lam, m, lo, hi, a in pts[:-1]:
            mm, ll, hh, _ = paired(a, pts[-1][4])
            if ll > 0:
                cbeat.append((lam, mm, ll, hh))

        print(f"    终点差 λ=1 − λ={lo_lam}: {m_end:+.2f} [{l_end:+.2f},{h_end:+.2f}]"
              f"{'*' if end_sep else ' ns'}")
        if cbeat:
            print(f"  ⇒ **C 非单调**：内部 λ 显著高于 λ=1 —— " +
                  ", ".join(f"λ={l}: +{m:.2f}[{a:+.2f},{b:+.2f}]" for l, m, a, b in cbeat))
            print("     两条主线都不成立，需要新的机制假说。")
        elif nondec and end_sep:
            print(f"  ⇒ **A 单调上升**：前提「地板收益来自抬饿死头」成立。"
                  f"配合「整个 Q_box 最多抬 2.25%」⇒ 可达性是该缺口的主因。")
        elif lo_d >= 0.7 * hi_d and not end_sep:
            print(f"  ⇒ **B 早饱和**：λ={lo_lam} 已拿到 {lo_d/hi_d*100:.0f}% 且与 λ=1 不可分"
                  f" ⇒ 收益另有来源，可达性叙事不适用于该缺口。")
        else:
            print(f"  ⇒ **不判定**：逐点不降={nondec}、终点分离={end_sep}、"
                  f"λ={lo_lam} 占比 {lo_d/hi_d*100:.0f}%（B 需 ≥70% 且不分离）。"
                  f"**不得硬套最接近的一条。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
