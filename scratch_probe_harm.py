"""把每格的 Δ 拆成「救回来的」与「弄坏的」，并检验 headroom 回归的量纲问题。

GPT 建议的 0→1 / 1→0 分解不能直接用 —— 逐样本分数是**分级的**
（scbench_kv 每样本 5 个子问题 ⇒ 取值 {0,.2,.4,.6,.8,1}）。
推广形式：

    up   = Σ_j max(a_j − b_j, 0) / n     （救回来的量）
    down = Σ_j min(a_j − b_j, 0) / n     （弄坏的量，负数）
    Δ    = up + down                      （恒等式，不是近似）

这样能区分两种完全不同的失败：
  * `up` 小、`|down|` 小 ⇒ 方法几乎没动，Δ≈0；
  * `up` 大、`|down|` 更大 ⇒ **有真实收益但被更大的破坏盖过** —— 这才是
    「缺 no-op 门控」的特征；
  * `up`≈0、`|down|` 大 ⇒ 方向本身就错，加门控也救不了。

第二件事：原 headroom 回归把 11 个 panel 的**原始分**混在一起回归，
而各 panel 的分数尺度差很多（GSM 摆动 14 分、Summary 摆动 2 分）。
这里同时报**按满缓存分归一化**的版本作为对照。
"""
import numpy as np
from scipy import stats as st
from scratch_read_scores import read_scores, paired

RAT = [0.75, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
PAN = ["scbench_kv", "scbench_prefix_suffix", "scbench_vt", "scbench_mf",
       "scbench_summary", "scbench_many_shot", "gsm", "scbench_choice_eng",
       "scbench_qa_eng", "squad", "scbench_repoqa"]


def split(a, b):
    """→ (Δ, up, down)，单位与 paired 一致（×100）。恒等式 Δ = up + down。"""
    k = sorted(set(a) & set(b))
    d = np.array([a[j] - b[j] for j in k]) * 100
    return d.mean(), d[d > 0].sum() / len(d), d[d < 0].sum() / len(d)


if __name__ == "__main__":
    rows = []
    for d in PAN:
        try:
            fu = np.mean(list(read_scores(d, "_g8base", 0.1, field="full__").values())) * 100
        except Exception:
            continue
        for r in RAT:
            try:
                b = read_scores(d, "_g8base", r)
                a = read_scores(d, f"_gc100_{d}", r)
                if len(a) != len(b):
                    continue
            except Exception:
                continue
            base = np.mean(list(b.values())) * 100
            dl, up, dn = split(a, b)
            rows.append((d, r, fu - base, dl, up, dn, fu, base))

    H = np.array([x[2] for x in rows]); D = np.array([x[3] for x in rows])
    UP = np.array([x[4] for x in rows]); DN = np.array([x[5] for x in rows])
    FU = np.array([x[6] for x in rows])

    print(f"n = {len(rows)} 格\n")
    print("=== ① 恒等式自检（Δ 必须等于 up+down）")
    print(f"  max|Δ − (up+down)| = {np.abs(D - (UP + DN)).max():.2e}")

    print("\n=== ② 按 headroom 分档，看 up 与 down 各自怎么变")
    print(f"{'headroom':<14}{'n':>4}{'Δ':>9}{'up 救回':>10}{'down 弄坏':>11}{'up/|down|':>11}")
    for lo, hi in [(-99, 0), (0, 5), (5, 15), (15, 99)]:
        m = (H >= lo) & (H < hi)
        if not m.sum():
            continue
        print(f"[{lo:>3},{hi:>3})     {m.sum():>4}{D[m].mean():>+9.2f}"
              f"{UP[m].mean():>+10.2f}{DN[m].mean():>+11.2f}"
              f"{UP[m].mean()/max(abs(DN[m].mean()),1e-9):>11.2f}")

    print("\n=== ③ 量纲检验：原始分 vs 按满缓存分归一化")
    for nm, y, x in [("原始 Δ ~ headroom", D, H),
                     ("相对 Δ/full ~ headroom/full", D / FU, H / FU)]:
        sl, ic, r, p, se = st.linregress(x, y)
        sp = st.spearmanr(x, y)
        print(f"  {nm:<30} 斜率 {sl:+.4f}  截距 {ic:+.4f}  R²={r**2:.3f}  "
              f"Spearman {sp[0]:+.3f} (p={sp[1]:.1e})")

    print("\n=== ④ 最伤的 5 格：是「没收益」还是「收益被更大破坏盖过」？")
    print(f"{'格':<26}{'headroom':>10}{'Δ':>9}{'up':>9}{'down':>10}")
    for i in np.argsort(D)[:5]:
        d, r = rows[i][0], rows[i][1]
        print(f"{d[:18]+'@'+str(r):<26}{H[i]:>+10.2f}{D[i]:>+9.2f}{UP[i]:>+9.2f}{DN[i]:>+10.2f}")
    print(f"\n=== ⑤ 最好的 3 格")
    for i in np.argsort(-D)[:3]:
        d, r = rows[i][0], rows[i][1]
        print(f"{d[:18]+'@'+str(r):<26}{H[i]:>+10.2f}{D[i]:>+9.2f}{UP[i]:>+9.2f}{DN[i]:>+10.2f}")
