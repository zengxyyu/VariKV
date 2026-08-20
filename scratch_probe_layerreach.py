#!/usr/bin/env python3
"""**逐层**的可达广度：`Q_box` 能把哪些层的饿死头抬起来？零 GPU。

**为什么这一条现在最关键**：撤回 47 把机制从「覆盖多少头」改成「抬哪些层」——
实测抬起最前约 2 层的饿死头给 **+25.00★**，同样数量放别处给 **+0.80 ns**。
于是架构问题变成一个非常具体的问题：

    `|Δs| ≤ α·σ_h` 到底能不能抬起**前两层**的饿死头？

- **能** ⇒ 网络原则上够得着那些头，瓶颈不在可达性而在「没学到该抬它们」（`R_learn`）；
- **不能** ⇒ 可达性瓶颈成立，且目标极其具体：**只需让 L0–L2 的饿死头能越过阈值**，
  这比先前「把广度提到 75%」便宜得多。

度量与总广度同一套闭式：给定公共阈值 τ，饿死头 h 能拿到 ≥1 当且仅当
`q_max_h(τ) ≥ 1`；每个只花 1 个预算，可行性由 `Σ_S 1{q_max≥1} + Σ_{S^c} q_min ≤ B`
保证。对可行 τ 取**使目标层带覆盖率最大**的那个，**逐层**统计。

**⚠ 口径**（与其他 trace 探针相同，必须一起引用）：trace 每 (chunk,层,头) 只存
256~768 个近阈值候选，配额是候选池内的量，**绝对数不可外推到 eval**；
界必须用 trace 里**存的** `sig_h`（整块口径）——在候选池上重算 std 是撤回 46 的根因。
"""
import glob, os, sys
import numpy as np, torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                        # noqa: E402


def main():
    sd = torch.load(f"{ROOT}/varikv/d10_scalar_s0.pt/memoryless.pt", map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"], n_slots=sd.get("slots", 8),
                    d_m=sd.get("dim", 128), mode="memoryless", arch=sd["arch"])
    m.load_state_dict(sd["state"]); m.eval()
    alpha = float(m.alpha); H = int(sd["H"]); Lyr = int(sd["L"])

    # 每层累计：饿死头数、其中可被抬到 ≥1 的头数
    starv = np.zeros(Lyr); reach = np.zeros(Lyr); nchunk = 0
    sigsum = np.zeros(Lyr); signum = np.zeros(Lyr)   # 饿死头的 σ_h，用于补完因果链
    skip = 0
    for f in sorted(glob.glob(f"{ROOT}/scratch_ctrl_traces_v2_10/doc*.pt")):
        d = torch.load(f, map_location="cpu")
        for ch in d["chunks"]:
            t = float(ch["thres"])
            sc = torch.cat([pl["s0"][:, :pl["n_near"]].float() for pl in ch["layers"]], 0)
            sg = torch.cat([pl["sig_h"].float().reshape(-1) for pl in ch["layers"]])
            G, npt = sc.shape; s0f = sc.reshape(-1); B = int((s0f > t).sum())
            if B < G or B >= len(s0f) - G:
                skip += 1; continue
            vb = torch.zeros(len(s0f), dtype=torch.bool)
            vb[torch.topk(s0f, B).indices] = True
            b0 = vb.reshape(G, npt).sum(-1); S = (b0 == 0)
            if not bool(S.any()):
                skip += 1; continue
            X = sc.float(); a = alpha * sg.clamp_min(1e-6)
            ss = torch.sort(X, dim=-1).values
            hp = (X + a[:, None]).reshape(-1); lp = (X - a[:, None]).reshape(-1); N = hp.numel()
            t_hi = float(torch.topk(hp, min(B, N), largest=True).values[-1])
            t_lo = float(torch.topk(lp, min(B + 1, N), largest=True).values[-1])
            if not (t_lo < t_hi):
                skip += 1; continue
            taus = torch.linspace(t_lo, t_hi, 2048)
            qmx = npt - torch.searchsorted(ss, (taus[:, None] - a[None, :]).T.contiguous(),
                                           right=True).T
            qmn = npt - torch.searchsorted(ss, (taus[:, None] + a[None, :]).T.contiguous(),
                                           right=True).T
            feas = (qmn.sum(1) <= B) & (qmx.sum(1) >= B)
            can = (qmx[:, S] >= 1)                       # [T, |S|]
            ok = feas & ((can.sum(1) + qmn[:, ~S].sum(1)) <= B)
            if not bool(ok.any()):
                skip += 1; continue
            # **逐层各自取最优 τ**。第一版取的是「总覆盖最大」的那个 τ，
            # 那会系统性**低估**某一层单独能被抬到多少（不同层的最优 τ 不同）。
            # 只抬第 l 层的饿死头时，代价是那几个头，其余头压到 q_min，
            # 可行性条件相应放松为 `count_l(τ) + Σ_{h∉l} q_min(τ) ≤ B`。
            idx = torch.nonzero(S).flatten()             # 饿死头的全局编号
            lay = (idx // H).numpy()
            qmn_all = qmn.sum(1)                          # [T]
            for l in range(Lyr):
                sel = (lay == l)
                if not sel.any():
                    continue
                cl = can[:, torch.as_tensor(sel)].sum(1)                  # [T] 该层可抬头数
                # 其余头压到 q_min：Σ_{h∉该层饿死头} q_min = Σ q_min − Σ_{该层饿死头} q_min
                gl = idx[torch.as_tensor(sel)]
                qmn_l = qmn[:, gl].sum(1)
                afford = (cl + (qmn_all - qmn_l)) <= B
                okl = feas & afford
                best = int(cl[okl].max()) if bool(okl.any()) else 0
                starv[l] += int(sel.sum()); reach[l] += best
                sigsum[l] += float(sg[gl].sum()); signum[l] += int(sel.sum())
            nchunk += 1

    print(f"逐层可达广度（{nchunk} 个 chunk，构造性跳过 {skip} 个；α={alpha:.6f}，stored sig_h）")
    print(f"{'层':>4} {'饿死头次':>9} {'可抬起':>8} {'可达率':>8} {'饿死头 σ_h 均值':>16}")
    for l in range(Lyr):
        if starv[l] == 0:
            print(f"{l:>4} {0:>9} {'—':>8} {'—':>8} {'—':>16}"); continue
        sm = sigsum[l] / max(signum[l], 1)
        print(f"{l:>4} {int(starv[l]):>9} {int(reach[l]):>8} {reach[l]/starv[l]*100:>7.1f}%"
              f" {sm:>16.5f}")
    ee = slice(0, 3); ll = slice(20, Lyr)
    print(f"\n  σ_h（饿死头均值）：L0–L2 **{sigsum[ee].sum()/max(signum[ee].sum(),1):.5f}**"
          f"   vs  L20–L27 **{sigsum[ll].sum()/max(signum[ll].sum(),1):.5f}**"
          f"   比值 **{(sigsum[ll].sum()/max(signum[ll].sum(),1))/max(sigsum[ee].sum()/max(signum[ee].sum(),1),1e-12):.1f}×**")
    early = slice(0, 3); late = slice(20, Lyr)
    print()
    print(f"  **L0–L2（`index` 顺序实际抬的那批）可达率 "
          f"{reach[early].sum()/max(starv[early].sum(),1)*100:.1f}%**"
          f"   （{int(reach[early].sum())}/{int(starv[early].sum())} 头次）")
    print(f"  L20–L27 可达率 {reach[late].sum()/max(starv[late].sum(),1)*100:.1f}%"
          f"   全体 {reach.sum()/max(starv.sum(),1)*100:.1f}%")
    print()
    e = reach[early].sum()/max(starv[early].sum(), 1)
    if e > 0.6:
        print("  ⇒ **前两层的饿死头基本是可达的** ⇒ 瓶颈不在可达性，而在"
              "「网络没学到该抬它们」（`R_learn`）。")
    elif e < 0.2:
        print("  ⇒ **前两层的饿死头基本不可达** ⇒ 可达性瓶颈成立，且目标极具体："
              "只需让 L0–L2 的饿死头能越过阈值。")
    else:
        print(f"  ⇒ 前两层可达率 {e*100:.0f}%，居中；两条解释都不能下断言。")
    print("\n  ⚠ 口径：trace 只有近阈值候选，**绝对数不可外推 eval**；界用 stored `sig_h`。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
