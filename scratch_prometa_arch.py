#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**架构选择实验**：多 atom 的 R 个方向该由什么参数化产生？

`scratch_prometa_repr.py` 已确定两件事：
  ① 自由 q 从 R=1 的 J@0.1=0.50 涨到 R=32 的 0.87 ⇒ 多 atom 上限很高；
  ② `q_{l,h,r} = base_{l,h} + off_r`（atom 偏移跨层头共享）几乎无效（+0.009）。
     且因为那次是**逐未来单独拟合**，`off_r` 在拟合内等价于 `off_{m,r}`
     ⇒ **把 trunk 输出扩宽成 M×R×d_lat 这个直觉改法也没用**。

所以问题变成：atom 偏移要不要**逐 (层,头)**？本脚本把全部 M 个未来**联合**
拟合，于是「跨 m 共享」与「跨 (l,h) 共享」才是两个可分辨的约束：

    free    q[m,b,r]  自由                       —— 上界（不可部署，b=(层,头)）
    off_b   q[m,b,r] = base[m,b] + off[b,r]      —— atom 偏移**逐头静态**、跨未来共享
    off_m   q[m,b,r] = base[m,b] + off[m,r]      —— atom 偏移**逐未来**、跨头共享
    off_g   q[m,b,r] = base[m,b] + off[r]        —— 全局共享（已知无效，阴性对照）

`off_b` 对应的架构改动是给 Student 加 `head_atom_emb[R,L,H,d]`
（R=8 时 114,688 参数），**上下文相关的部分仍只有 `A(u_m)` 一个向量**。
若 `off_b` 能吃下 `free` 的大部分增益，架构改动就是廉价且可部署的。

⚠ 这里拟合的 `base[m,b]` 是**每篇文档自由拟合**的，真实 Student 要**从上下文
预测**它 —— 所以本脚本给的是**族的上界**，不是可达性能。这条边界必须同写。
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))
from scratch_prometa_repr import topk_jaccard, _kl        # noqa: E402
from prometa.risk import entropic_risk_torch               # noqa: E402


def global_mask_J(Ut, Up, rhos, beta=0.0):
    """**真实决策级**保留掩码的 Jaccard（外部复核 2026-08-22 指出必须改，采纳）。

    `level="pair"` 用的是**跨全部 (层,头) 的单一全局阈值**，不是每头各留 ρ。
    逐头 top-k 的 Jaccard 量的是 **per-head ranking fidelity**，
    不是最终 cache mask —— 拿它当「决策级」是第⑱类错（两个动作空间直接比）。

    Ut, Up: [M, B, n]（每个 (m,b) 已归一化）。→ {ρ: (J_mask, 逐头配额的相关)}。
    """
    out = {}
    Rt = entropic_risk_torch(Ut.permute(1, 0, 2), beta, standardized=False)  # [B,n]
    Rp = entropic_risk_torch(Up.permute(1, 0, 2), beta, standardized=False)
    B, n = Rt.shape
    for r in rhos:
        k = max(1, int(round(r * B * n)))
        it = Rt.flatten().topk(k).indices
        ip = Rp.flatten().topk(k).indices
        At = torch.zeros(B * n, dtype=torch.bool, device=Rt.device).scatter_(0, it, True)
        Ap = torch.zeros(B * n, dtype=torch.bool, device=Rp.device).scatter_(0, ip, True)
        inter = float((At & Ap).sum())
        # 逐头配额（全局阈值下每个头实际拿到多少）—— 本项目的核心动作空间
        qt = At.view(B, n).sum(-1).float()
        qp = Ap.view(B, n).sum(-1).float()
        cc = float(torch.corrcoef(torch.stack([qt, qp]))[0, 1]) if qt.std() > 0 else float("nan")
        out[r] = (inter / max(2 * k - inter, 1), cc,
                  float((qt == 0).float().mean()), float((qp == 0).float().mean()))
    return out

# `arch1`/`archR` 是**真实 Student 的参数化**，不是族上界：
#   上下文相关的部分只有 `c[m] = A(u_m)` 一个 d 维向量，**对所有 (层,头) 相同**；
#   逐头的部分 `he[b,r]` 是**静态参数**（当前 `head_emb`，R=1）。
#   前面 off_* 三个都让 `base[m,b]` 逐头自由，比真实架构宽松得多。
#   `arch1s`/`archRs`：**方向仍 L2 归一化，但乘一个逐 (层,头) 可学尺度** `s_b`。
#   实测 L2 是主瓶颈（全局掩码 J 0.6352→0.4055），但完全去掉归一化会让 logit
#   自由膨胀、softmax 饱和。这一族只放开**尺度**这一个自由度，是最小改动。
MODES = ("free", "off_b", "off_m", "off_g", "arch1", "archR", "arch1s", "archRs")


def fit_joint(target, K, R, mode, steps=1500, lr=0.05, seed=0, l2=False):
    """target: [M,B,n]；K: [B,n,d] → `Û=normalize(max_r softmax(q·K/√d))`: [M,B,n]。"""
    M, B, n = target.shape
    d = K.shape[-1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    rn = lambda *sh: (torch.randn(*sh, generator=g) * 0.5).to(K.device).requires_grad_(True)
    if mode in ("arch1", "archR", "arch1s", "archRs"):
        Reff = 1 if mode in ("arch1", "arch1s") else R
        c, he = rn(M, d), rn(B, Reff, d)
        if mode.endswith("s"):
            # 方向归一化 + 逐 (层,头) 可学尺度（log 参数化保证正）
            sc_b = torch.zeros(B, 1, 1, device=K.device).requires_grad_(True)
            prm = [c, he, sc_b]
            mk = lambda: (torch.nn.functional.normalize(
                c[:, None, None, :] + he[None], dim=-1) * sc_b.exp()[None])
            npar_extra = he.numel() + sc_b.numel()
        else:
            prm = [c, he]
            mk = lambda: c[:, None, None, :] + he[None]        # [M,B,Reff,d]
            npar_extra = he.numel()
    else:
        base = rn(M, B, d)
        off = {"free": None, "off_b": rn(B, R, d), "off_m": rn(M, R, d),
               "off_g": rn(R, d)}[mode]
        if mode == "free":
            q = rn(M, B, R, d); prm = [q]; mk = lambda: q
        else:
            prm = [base, off]
            mk = {"off_b": lambda: base[:, :, None, :] + off[None],
                  "off_m": lambda: base[:, :, None, :] + off[:, None],
                  "off_g": lambda: base[:, :, None, :] + off[None, None]}[mode]
        npar_extra = 0 if mode == "free" else off.numel()
    opt = torch.optim.Adam(prm, lr=lr)
    sc = 1.0 / (d ** 0.5)

    def fwd(qq):
        if l2 and not mode.endswith("s"):
            qq = torch.nn.functional.normalize(qq, dim=-1)
        p = torch.softmax(torch.einsum("mbrd,bnd->mbrn", qq, K) * sc, -1)
        u = p.amax(2)
        return u / u.sum(-1, keepdim=True).clamp_min(1e-30)

    traj = []
    for i in range(steps):
        loss = _kl(target, fwd(mk())).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (i + 1) % max(1, steps // 4) == 0:
            traj.append(float(loss))
    with torch.no_grad():
        pred = fwd(mk())
    return pred.detach(), float(_kl(target, pred).mean()), traj, npar_extra


def _selftest():
    """阴阳对照：造一个 **atom 方向逐 b 不同** 的目标，`off_b` 必须赢 `off_g`。"""
    torch.manual_seed(0)
    M, B, n, d, R = 2, 3, 300, 16, 2
    K = torch.randn(B, n, d)
    tg = []
    for m in range(M):
        rows = []
        for b in range(B):
            # 每个 b 有**自己的**两个 atom 方向；两个 m 只差一个全局平移
            a1 = torch.randn(d) + m * 0.3
            a2 = torch.randn(d) + m * 0.3
            p = torch.stack([torch.softmax(4 * a1 @ K[b].T / d ** .5, -1),
                             torch.softmax(4 * a2 @ K[b].T / d ** .5, -1)])
            u = p.amax(0); rows.append(u / u.sum())
        tg.append(torch.stack(rows))
    T = torch.stack(tg)                                  # [M,B,n]
    out = {}
    for md in MODES:
        if md.startswith("arch"): continue
        _, kl, _, npar = fit_joint(T, K, R, md, steps=800, lr=0.1)
        out[md] = kl
        print(f"  {md:<6} KL={kl:.5f}  额外参数 {npar}")
    assert out["free"] <= out["off_b"] + 1e-3, out
    assert out["off_b"] < out["off_g"] * 0.9, out          # 逐 b 的 atom 必须更好
    assert out["off_b"] < out["off_m"] * 0.9, out
    print("① 目标的 atom 方向逐 b 不同时，off_b 显著优于 off_m / off_g　PASS")

    # ② 阴性对照：把目标改成**全局同一对 atom**，此时 off_g 就够了
    a1, a2 = torch.randn(d), torch.randn(d)
    rows = []
    for m in range(M):
        rr = []
        for b in range(B):
            p = torch.stack([torch.softmax(4 * a1 @ K[b].T / d ** .5, -1),
                             torch.softmax(4 * a2 @ K[b].T / d ** .5, -1)])
            u = p.amax(0); rr.append(u / u.sum())
        rows.append(torch.stack(rr))
    T2 = torch.stack(rows)
    o2 = {md: fit_joint(T2, K, R, md, steps=800, lr=0.1)[1] for md in ("off_b", "off_g")}
    assert o2["off_g"] < 2.0 * o2["off_b"], o2
    # ③ arch1 ⊂ archR（同 R 下 archR 不可能更差）+ L2 归一化的代价可测
    k1 = fit_joint(T, K, R, "arch1", steps=800, lr=0.1)[1]
    kR = fit_joint(T, K, R, "archR", steps=800, lr=0.1)[1]
    kRn = fit_joint(T, K, R, "archR", steps=800, lr=0.1, l2=True)[1]
    assert kR <= k1 + 1e-3, (k1, kR)
    print(f"③ arch1 KL={k1:.5f} ≥ archR KL={kR:.5f}（嵌套单调）；archR+L2 归一化 {kRn:.5f}　PASS")
    print(f"② 目标为全局同一对 atom 时 off_g({o2['off_g']:.5f}) 与 off_b"
          f"({o2['off_b']:.5f}) 同量级 ⇒ 判据不是恒偏向 off_b　PASS")
    print("\nscratch_prometa_arch.py 自测 3 条全过")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("-g", "--gate", default="fastkvzip")
    p.add_argument("--manifest", default="prometa_data/manifest_v1_ss.jsonl")
    p.add_argument("--split", default="val")
    p.add_argument("--n_docs", type=int, default=2)
    p.add_argument("--chunk", type=int, default=16000)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--Rs", default="4,8,16")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--rhos", default="0.02,0.05,0.1,0.2")
    p.add_argument("--beta", type=float, default=0.0,
                   help="风险聚合的 β（0 = 逐未来取均值）")
    p.add_argument("--l2", action="store_true",
                   help="按真实 Student 那样对 q 做 L2 归一化（`model.py` 里是无条件做的）。"
                        "它把 logit 尺度钉死，可能本身就是一层瓶颈 —— 开/关各跑一次量它")
    p.add_argument("--out", default="scratch_prometa_arch.json")
    a = p.parse_args()
    if a.selftest:
        return _selftest()

    from model.wrapper import ModelKVzip
    from prometa.teacher import chunk_ranges, future_utility
    Rs = [int(x) for x in a.Rs.split(",")]
    rhos = [float(x) for x in a.rhos.split(",")]
    recs = [json.loads(l) for l in open(a.manifest)]
    recs = [r for r in recs if r["split"] == a.split and r["futures"]][:a.n_docs]
    model = ModelKVzip(a.model, "retain", a.gate)
    dev = model.device
    sys_len = int(model.sys_prompt_ids.shape[1])
    rows = []
    for ri, rec in enumerate(recs):
        ctx = torch.tensor([rec["ctx"]], device=dev)
        _all, usable = chunk_ranges(ctx.shape[1] + sys_len, sys_len, a.chunk, a.window)
        if not usable:
            continue
        lo, hi = usable[len(usable) // 2]; n = hi - lo
        prev = getattr(model, "varikv_train", False); model.varikv_train = True
        try:
            with torch.no_grad():
                kv = model.prefill(ctx, prefill_chunk_size=a.chunk, do_score=False, chunk_ratio=1.0)
        finally:
            model.varikv_train = prev
        L, H = len(kv.key_cache), int(kv.key_cache[0].shape[1])
        d = int(kv.key_cache[0].shape[-1])
        Kc = torch.stack([kv.key_cache[l][0, :, lo:hi, :].float() for l in range(L)]).reshape(L * H, n, d)
        tg = []
        for f in rec["futures"]:
            with torch.no_grad():
                kv.capture_q, kv._q_cap = True, {}
                model(torch.tensor([list(f["q"])], device=dev), kv, update_cache=False)
                kv.capture_q = False
                U = future_utility(kv.key_cache, [kv._q_cap[l] for l in range(L)], lo, hi, out_np=False)
            tg.append((U / U.sum(-1, keepdim=True).clamp_min(1e-30)).reshape(L * H, n))
            kv._q_cap = {}
        T = torch.stack(tg)                                  # [M, L*H, n]
        for R in Rs:
            for md in MODES:
                pred, kl, traj, npar = fit_joint(T, Kc, R, md, steps=a.steps, l2=a.l2)
                row = dict(doc=ri, R=R, mode=md, KL=kl, npar_extra=npar, n=n,
                           conv=traj, M=int(T.shape[0]))
                for r_ in rhos:
                    row[f"J@{r_}"] = topk_jaccard(T.reshape(-1, n), pred.reshape(-1, n),
                                                  max(1, int(round(r_ * n))))
                gm = global_mask_J(T, pred, rhos, beta=a.beta)
                for r_ in rhos:
                    row[f"Jmask@{r_}"], row[f"quotacorr@{r_}"], \
                        row[f"starve_t@{r_}"], row[f"starve_p@{r_}"] = gm[r_]
                rows.append(row)
                print(f"  doc{ri} R={R:<3} {md:<6} KL={kl:.4f} 逐头J@0.1={row['J@0.1']:.4f} "
                      f"**全局掩码J@0.1={row['Jmask@0.1']:.4f}** "
                      f"配额相关={row['quotacorr@0.1']:+.3f} "
                      f"饿死头 教师{row['starve_t@0.1']:.2f}/预测{row['starve_p@0.1']:.2f} "
                      f"额外参数 {npar:,}  收敛 {traj[-2]:.4f}→{traj[-1]:.4f}", flush=True)
                del pred
                torch.cuda.empty_cache()
        del kv, Kc, T; torch.cuda.empty_cache()
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\n" + "=" * 74)
    print(f"{'R':>4} {'参数化':<8} {'额外参数':>10} {'KL':>8} "
          + " ".join(f"{'逐头J@'+str(r):>10}" for r in rhos) + "  ｜ "
          + " ".join(f"{'掩码J@'+str(r):>10}" for r in rhos) + "  配额相关@0.1")
    for R in Rs:
        for md in MODES:
            g = [x for x in rows if x["R"] == R and x["mode"] == md]
            if not g: continue
            print(f"{R:>4} {md:<8} {g[0]['npar_extra']:>10,} {np.mean([x['KL'] for x in g]):>8.4f} "
                  + " ".join(f"{np.mean([x[f'J@{r}'] for x in g]):>10.4f}" for r in rhos) + "  ｜ "
                  + " ".join(f"{np.mean([x[f'Jmask@{r}'] for x in g]):>10.4f}" for r in rhos)
                  + f"  {np.mean([x['quotacorr@0.1'] for x in g]):>+8.3f}")
    print("\n  随机基线（同一 k 下两个随机集合的期望 Jaccard = ρ/(2−ρ)）："
          + " ".join(f"ρ={r}:{r/(2-r):.4f}" for r in rhos))
    g1 = [x["J@0.1"] for x in rows if x["mode"] == "arch1"]
    a1 = float(np.mean(g1)) if g1 else float("nan")
    print(f"\n**真实架构的可达上界**（上下文只给一个 c[m]、逐头部分是静态参数）")
    print(f"  当前 arch1 (R=1)：J@0.1 = {a1:.4f}")
    for md in ("arch1s", "archRs"):
        for R in Rs:
            g = [x for x in rows if x["R"] == R and x["mode"] == md]
            if g and (md == "archRs" or R == Rs[0]):
                print(f"  {md:<7} R={R if md=='archRs' else 1:<3}："
                      f"逐头 J {np.mean([x['J@0.1'] for x in g]):.4f}　"
                      f"**全局掩码 J {np.mean([x['Jmask@0.1'] for x in g]):.4f}**　"
                      f"KL {np.mean([x['KL'] for x in g]):.4f}")
    m1 = [x["Jmask@0.1"] for x in rows if x["mode"] == "arch1"]
    mm1 = float(np.mean(m1)) if m1 else float("nan")
    print(f"  当前 arch1 (R=1)：**全局掩码 J@0.1 = {mm1:.4f}**")
    for R in Rs:
        aR = [x["J@0.1"] for x in rows if x["R"] == R and x["mode"] == "archR"]
        fr = [x["J@0.1"] for x in rows if x["R"] == R and x["mode"] == "free"]
        mR = [x["Jmask@0.1"] for x in rows if x["R"] == R and x["mode"] == "archR"]
        mf = [x["Jmask@0.1"] for x in rows if x["R"] == R and x["mode"] == "free"]
        if aR:
            print(f"  archR R={R:<3}：逐头 J {np.mean(aR):.4f}（{np.mean(aR)-a1:+.4f}）"
                  f"　**全局掩码 J {np.mean(mR):.4f}（{np.mean(mR)-mm1:+.4f}）**"
                  f"　｜ free 逐头 {np.mean(fr):.4f} 掩码 {np.mean(mf):.4f}")
    print("\n⚠ **判词只看全局掩码那一列** —— 逐头 J 是 per-head ranking fidelity，"
          "不是 level=pair 的真实动作空间。")
    print("Finished.")


if __name__ == "__main__":
    main()
