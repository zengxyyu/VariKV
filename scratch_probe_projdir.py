#!/usr/bin/env python3
"""投影把地板的位移实现了多少？—— 零 GPU，在 teacher trace 上离线算。

**为什么需要它**：运行时日志只报 `|q_proj − q_floor|_1`。实测它（1252）与
`|q_floor − b0|_1`（1234）几乎相等——但**等距不等于落回原点**：`q_proj` 可能是
另一个等距点。这两种情形对「地板那 +33.60 还剩多少」的预测完全相反，
所以必须把位移分解开看。

度量（比 L1 更有信息）：记 `d_floor = q_floor − b0`、`d_proj = q_proj − b0`，

    实现比例  ρ_impl = <d_proj, d_floor> / ||d_floor||²     沿地板方向实现了多少
    幅度比    ||d_proj|| / ||d_floor||
    方向余弦  cos(d_proj, d_floor)

`ρ_impl ≈ 0` ⇒ 投影基本回到基线，地板干预被界抹掉；
`ρ_impl ≈ 1` ⇒ 位移基本保住；
`ρ_impl` 小但余弦高 ⇒ 方向对、幅度被压缩（这是「有界」最典型的表现）。

**口径限制（与 `scratch_probe_reach.py` 相同，必须一起引用）**：trace 每
(chunk,层,头) 只存 768 个近阈值候选，所以这里的 `b0` 与配额是**候选池内**的量，
不是推理时的真实绝对配额。候选池按 `|s⁰−τ|` 采样、本身有偏。结论应读作
「在近阈值结构上」成立，绝对数值不可外推到 eval。
"""
import glob, os, sys
import numpy as np, torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                        # noqa: E402
from attention.quota_project import reachable_project, slack_of        # noqa: E402


def main():
    bmin = float(os.environ.get("BMIN", "8"))
    sd = torch.load(f"{ROOT}/varikv/d10_scalar_s0.pt/memoryless.pt", map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"], n_slots=sd.get("slots", 8),
                    d_m=sd.get("dim", 128), mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"]); m.eval()
    alpha = float(m.alpha)

    rows, fails, nskip, extra = [], [], 0, []
    for f in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt")):
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"]); S0 = []
            for pl in ch["layers"]:
                nn_ = pl["n_near"]
                S0.append(pl["s0"][:, :nn_].float())                  # [H, n]
            sc = torch.cat(S0, dim=0)                                 # [L*H, n]
            sig = torch.cat([pl["sig_h"].float().reshape(-1) for pl in ch["layers"]])
            G, npt = sc.shape
            s0f = sc.reshape(-1)
            B = int((s0f > t).sum())
            if B < G or B >= len(s0f) - G:
                nskip += 1; continue
            # 基线配额：全局 top-B（与生产的 threshold 语义一致）
            vb = torch.zeros(len(s0f), dtype=torch.bool)
            vb[torch.topk(s0f, B).indices] = True
            b0 = vb.reshape(G, npt).sum(-1).float()
            # 地板目标：与 quota_project 的 floor 分支同构
            eff = min(bmin, B // G)
            tg = torch.maximum(b0, torch.full_like(b0, eff))
            ex = float(tg.sum() - B)
            if ex > 0:
                room = (b0 - eff).clamp(min=0); tr = float(room.sum())
                if tr < ex:
                    nskip += 1; continue
                tg = tg - room * (ex / tr)
            tg = tg.round().long().clamp(0, npt)
            dz = B - int(tg.sum())
            if dz != 0:
                idx = torch.argsort(-b0)
                for k in range(abs(dz)):
                    tg[int(idx[k % G])] += int(np.sign(dz))
            tg = tg.clamp(0, npt)
            if int(tg.sum()) != B:
                nskip += 1; continue
            try:
                q, l1 = reachable_project(tg, sc, B, alpha=alpha, sigma=sig)
            except RuntimeError as e:
                # **不静默跳过**：丢样本会让统计有偏，而且正是我刚在 floorproj 上
                # 批评过的模式。计数并在末尾报出来。
                fails.append(str(e)[:80]); continue
            df = (tg - b0.long()).float().numpy()
            dp = (q - b0.long()).float().numpy()
            nf = float(np.linalg.norm(df))
            if nf < 1e-9:
                nskip += 1; continue
            # 位移落在**哪些头**上？假说：`a_h = α·σ_h` 对饿死头极小，
            # 于是投影只能把预算调整**改道**到 σ_h 大的（非饿死）头上。
            # **不要在这里重算 σ** —— 上面第一处就是这么错的：候选池是近阈值截断，
            # 在它上面算 std 会系统性低估界。用 trace 里存的 `sig_h`（整块口径）。
            a_np = (alpha * sig.clamp_min(1e-6)).numpy()
            starved = (b0.numpy() == 0)
            raise_f = np.maximum(df, 0)                       # 地板想抬的量
            raise_p = np.maximum(dp, 0)                       # 投影实际抬的量
            hi_sig = a_np >= np.median(a_np)
            extra.append((
                float(np.abs(dp)[hi_sig].sum() / max(np.abs(dp).sum(), 1e-9)),   # 位移集中在高σ头的比例
                float(np.abs(df)[hi_sig].sum() / max(np.abs(df).sum(), 1e-9)),   # 地板的对照
                float(raise_p[starved].sum() / max(raise_f[starved].sum(), 1e-9)) if starved.any() else np.nan,
                float(np.corrcoef(np.abs(dp), a_np)[0, 1]) if np.std(np.abs(dp)) > 0 else np.nan,
                float(starved.mean()),
                # 位移落在**饿死头 vs 非饿死头**上的份额（这才是直接的切分）
                float(np.abs(dp)[starved].sum() / max(np.abs(dp).sum(), 1e-9)),
                float(np.abs(df)[starved].sum() / max(np.abs(df).sum(), 1e-9)),
            ))
            rows.append((
                float(np.abs(df).sum()),                       # |d_floor|_1
                float(np.abs(dp).sum()),                       # |d_proj|_1
                l1,                                            # |q_proj − q_floor|_1
                float(dp @ df) / (nf ** 2),                    # ρ_impl
                float(np.linalg.norm(dp)) / nf,                # 幅度比
                float(dp @ df) / (nf * max(np.linalg.norm(dp), 1e-12)),   # 余弦
                float(slack_of(tg, torch.sort(sc, dim=-1).values, alpha * sig.clamp_min(1e-6))),
            ))
    A = np.array(rows)
    print(f"teacher trace，b_min={bmin:.0f}，α={alpha:.6f}，{len(A)} 个 chunk"
          f"（构造性跳过 {nskip} 个，投影抛错 {len(fails)} 个）")
    if fails:
        print("  ⚠ 抛错样本（不应出现，出现即实现有 bug）：", fails[:3])
    print(f"  地板本身不可达的比例（slack ≤ 0）：**{(A[:, 6] <= 0).mean()*100:.1f}%**")
    print()
    nm = ["|d_floor|_1", "|d_proj|_1", "|q_proj−q_floor|_1",
          "**ρ_impl 沿地板方向实现比例**", "幅度比 ||d_proj||/||d_floor||", "方向余弦"]
    for i, s in enumerate(nm):
        print(f"  {s:<34} 中位 {np.median(A[:, i]):+8.4f}   "
              f"p10 {np.percentile(A[:, i],10):+.4f}  p90 {np.percentile(A[:, i],90):+.4f}")
    r = np.median(A[:, 3]); c = np.median(A[:, 5])
    print()
    if r < 0.15:
        print(f"  ⇒ **实现比例仅 {r*100:.1f}%**：有界修正基本抹掉了地板的位移。")
    elif r > 0.7:
        print(f"  ⇒ 实现比例 {r*100:.1f}%：位移基本保住，可达性不是主要约束。")
    else:
        print(f"  ⇒ 实现比例 {r*100:.1f}%：部分保住。")
    print(f"  ⇒ 方向余弦中位 {c:.3f}：" +
          ("方向对、幅度被压缩（有界的典型表现）" if c > 0.7 else "连方向都没保住"))
    E = np.array(extra)
    print("\n【位移落在哪些头上】（假说：界把预算调整改道到 σ_h 大的非饿死头）")
    lbl = ["|d| 落在高 σ 头的比例（投影）", "同上（地板，对照）",
           "**饿死头上「地板想抬的量」被实现的比例**", "corr(|d_proj,h|, σ_h)", "饿死头占比",
           "**|d_proj| 落在饿死头上的份额**", "**|d_floor| 落在饿死头上的份额（对照）**"]
    for i, s_ in enumerate(lbl):
        v = E[:, i][~np.isnan(E[:, i])]
        print(f"  {s_:<38} 中位 {np.median(v):+.4f}   p10 {np.percentile(v,10):+.4f}  p90 {np.percentile(v,90):+.4f}")

    # ---- 第二问：**整个可达集**最多能把饿死头抬多少？（不依赖任何投影） --------
    # 上面 1.84% 是「L1 最近那一点」的性质。要变成关于**集合**的陈述，直接解
    #     max Σ_{h∈S} q_h   s.t.  q_min(τ) ≤ q ≤ q_max(τ),  Σq = B
    # 给定 τ，最优是 S 取上界、S^c 取下界，可行当且仅当 Σ_S q_max + Σ_{S^c} q_min ≤ B；
    # 否则被预算卡住。于是
    #     R(τ) = min( Σ_S q_max(τ),  B − Σ_{S^c} q_min(τ) ) − Σ_S b0
    # 再对可行 τ 取最大。这是**精确**的，与目标函数无关。
    print("\n【可达集最多能把饿死头抬多少】（闭式，与投影/目标函数无关）")
    best_raise = []
    for f in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt")):
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"])
            sc = torch.cat([pl["s0"][:, :pl["n_near"]].float() for pl in ch["layers"]], 0)
            G, npt = sc.shape; s0f = sc.reshape(-1); B = int((s0f > t).sum())
            if B < G or B >= len(s0f) - G:
                continue
            vb = torch.zeros(len(s0f), dtype=torch.bool)
            vb[torch.topk(s0f, B).indices] = True
            b0 = vb.reshape(G, npt).sum(-1)
            S = (b0 == 0)
            if not bool(S.any()):
                continue
            sig2 = torch.cat([pl["sig_h"].float().reshape(-1) for pl in ch["layers"]])
            X = sc.float(); a_ = alpha * sig2.clamp_min(1e-6)     # **整块口径**，不是候选池
            ss = torch.sort(X, dim=-1).values
            hp = (X + a_[:, None]).reshape(-1); lp = (X - a_[:, None]).reshape(-1)
            N = hp.numel()
            t_hi = float(torch.topk(hp, min(B, N), largest=True).values[-1])
            t_lo = float(torch.topk(lp, min(B + 1, N), largest=True).values[-1])
            if not (t_lo < t_hi):
                continue
            # R(τ) 是两个反向单调函数的 min ⇒ **单峰**，网格只影响到最大值的逼近精度。
            # 用 NTAU 环境变量做稳健性检查；若 256/1024/4096 给同一个数就说明够密。
            taus = torch.linspace(t_lo, t_hi, int(os.environ.get('NTAU', '1024')),
                                  dtype=X.dtype)
            qmax = npt - torch.searchsorted(ss, (taus[:, None] - a_[None, :]).T.contiguous(),
                                            right=True).T
            qmin = npt - torch.searchsorted(ss, (taus[:, None] + a_[None, :]).T.contiguous(),
                                            right=True).T
            ok = (qmin.sum(1) <= B) & (qmax.sum(1) >= B)
            if not bool(ok.any()):
                continue
            capS = qmax[:, S].sum(1)
            budget = B - qmin[:, ~S].sum(1)
            R = torch.minimum(capS, budget) - int(b0[S].sum())
            R = R[ok]
            # 地板想抬的总量（同一 chunk 上）
            eff = min(bmin, B // G)
            tgf = torch.maximum(b0.float(), torch.full((G,), eff))
            want = float((tgf - b0.float()).clamp(min=0)[S].sum())
            best_raise.append((float(R.max()), want, int(S.sum()), G))
    BR = np.array(best_raise)
    frac = BR[:, 0] / np.maximum(BR[:, 1], 1e-9)
    print(f"  {len(BR)} 个 chunk。地板想抬的总量中位 {np.median(BR[:,1]):.0f}"
          f"（饿死头中位 {np.median(BR[:,2]):.0f}/{int(BR[0,3])} 个）")
    print(f"  **可达集内最大可抬升量 / 地板想抬的量** 中位 **{np.median(frac)*100:.2f}%**"
          f"   p10 {np.percentile(frac,10)*100:.2f}%  p90 {np.percentile(frac,90)*100:.2f}%")
    # **判词不能写死**：第一版把「整个 Q_box 里没有任何一点能把饿死头抬起来」
    # 硬编码在这里，而那次算错了 σ_h（见下），重算后是 45% —— 一句写死的话
    # 把一个被推翻的结论继续打印了出来。判词必须由数字生成。
    mf = float(np.median(frac))
    if mf < 0.10:
        print(f"  ⇒ **集合级**：整个 `Q_box` 最多只能实现地板意图的 {mf*100:.1f}%，"
              f"抬饿死头这条机制基本不可用。")
    elif mf > 0.30:
        print(f"  ⇒ **集合级**：整个 `Q_box` 最多能实现地板意图的 {mf*100:.1f}% ——"
              f"**相当可观**。「可达集抬不动饿死头」**不成立**；"
              f"不可达的是地板那个**精确配额**，不是「抬升」这件事本身。")
    else:
        print(f"  ⇒ **集合级**：最多实现 {mf*100:.1f}%，部分可用。")

    print("\n  ⚠ 口径一：trace 每 (chunk,层,头) 只有 256~768 个**近阈值候选**，"
          "配额是候选池内的量，\n     绝对数值不可外推到 eval。")
    print("  ⚠ 口径二（**曾经算错**）：界必须用 trace 里**存的** `sig_h`（建 trace 时"
          "用整块算），\n     **不能**在近阈值候选池上重算 std —— 后者系统性低估界"
          "（stored/pool 比值 p90 达 26.6×），\n     第一版因此把最大可抬升算成 2.25%，"
          "实为约 45%。运行时 `maxlift` 的日志与 45% 同量级，是这次纠错的独立佐证。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
