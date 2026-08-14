#!/usr/bin/env python3
"""训练 VariKV-B 最终版的控制器。**同时训三臂，这才是实验本体。**

    stateful   : M_t = Write(M_{t-1}, R_t, E_t)
    memoryless : M 永远停在 M_init（可训练参数更少 ⇒ 对照不完全匹配）
    shuffled   : Write 照跑，但 retained/evicted 的**成员身份随机置换**
                 ⇒ 参数、计算、状态动力学全一致，只破坏「历史↔候选」的对应

论文的核心命题不是「比 FastKVzip 高」，而是

    stateful > shuffled ≈ memoryless ≳ FastKVzip

只有 stateful 显著优于 shuffled，才说明**历史真的提供了当前 KV 之外的增量信息**
（即 `I(U;M|X) > 0`）。只赢 FastKVzip 说明学到了一个更好的普通 scorer，那是别人
（Apple KVP 等）已经占住的问题。

--------------------------------------------------------------------------------
两个关键实现选择

1. **不对 top-k 反传。** 基线预填在 no_grad 下由 teacher 脚本采集完
   `(x, s⁰, mask, thres, U)`，这里只重放递归、算排序损失。top-k 只在评测时出现。

2. **排序损失作用在 `s' = s⁰ + Δs` 上，不是在 Δs 上。** 目标是让最终排序逼近 U，
   而 s⁰ 已经是先验；「历史增量」由 stateful−shuffled 之差度量，不该塞进损失定义。
   差值除以 σ_base 无量纲化，否则 logistic 会因为各头尺度不同而饱和。

3. **写入只用随机子样本，排序只用近阈值子样本。** teacher 存的候选里前 n_near 个是
   按 |s⁰−thres| 最近挑的（有偏），后面是均匀随机的。用有偏子集去估计
   retained/evicted 的池化摘要会让 writer 看到扭曲的历史，所以两者分开。
"""
import argparse
import glob
import os
import random
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.control_memory import ControlMemory           # noqa: E402


def _pw(du, mode):
    """成对权重。linear = |ΔU| / median|ΔU|，截到 [0,5] 防止长尾主导。"""
    if mode == "none":
        return torch.ones_like(du)
    a = du.abs()
    return (a / a.median().clamp_min(1e-12)).clamp(0.0, 5.0)


def _flat_pairs(sp, s0r, U, sigma, n_pairs, gen, pair_w="linear"):
    """一维版本，用于**跨 (层,kv头)** 的全局排序。"""
    n = sp.numel()
    i = torch.randint(0, n, (n_pairs,), generator=gen, device=sp.device)
    j = torch.randint(0, n, (n_pairs,), generator=gen, device=sp.device)
    du = U[i] - U[j]
    keep = du.abs() > 1e-6
    if not bool(keep.any()):
        z = torch.zeros((), device=sp.device)
        return sp.sum() * 0.0, z, z
    lg = (sp[i] - sp[j]) / sigma * du.sign()
    lg0 = (s0r[i] - s0r[j]) / sigma * du.sign()
    w = _pw(du, pair_w)
    return ((w * F.softplus(-lg))[keep].sum() / w[keep].sum().clamp_min(1e-6),
            (lg[keep] > 0).float().mean(), (lg0[keep] > 0).float().mean())


def pair_loss(sp, s0r, U, sigma, n_pairs, gen, pair_w="linear"):
    """sp/U [H,n]（近阈值子集），sigma [H,1] → 成对 logistic 排序损失。"""
    H, n = sp.shape
    i = torch.randint(0, n, (H, n_pairs), generator=gen, device=sp.device)
    j = torch.randint(0, n, (H, n_pairs), generator=gen, device=sp.device)
    du = torch.gather(U, 1, i) - torch.gather(U, 1, j)
    ds = (torch.gather(sp, 1, i) - torch.gather(sp, 1, j)) / sigma
    # 近似并列的对不含排序信息，只会把噪声当信号
    keep = du.abs() > 1e-6
    if not bool(keep.any()):
        z = torch.zeros((), device=sp.device)
        return sp.sum() * 0.0, z, z
    lg = ds * du.sign()
    # **按 |ΔU| 加权**：固定预算 top-B 选择的 regret 恰好是被错换的成对的 |U_i−U_j| 之和，
    # 所以加权版才是那个 regret 的可微代理；不加权等于把"两个几乎并列的候选排反"
    # 和"把最重要的和最没用的排反"惩罚成一样。`--pair_w none` 保留不加权做消融。
    w = _pw(du, pair_w)
    loss = (w * F.softplus(-lg))[keep].sum() / w[keep].sum().clamp_min(1e-6)
    acc = (lg[keep] > 0).float().mean()
    # **必须同时报 s0 自己的排序准确率**：只报 acc(s') 会把"s0 本来多好"和
    # "修正加了多少"混在一起。真正的量是 acc(s') − acc(s0)。
    ds0 = (torch.gather(s0r, 1, i) - torch.gather(s0r, 1, j)) / sigma
    acc0 = ((ds0 * du.sign())[keep] > 0).float().mean()
    return loss, acc, acc0


def run_doc(cm, doc, dev, n_pairs, gen, train=True, lam_global=1.0,
            skip_first_loss=True, shuf_gen=None, pair_w="linear"):
    """重放一篇文档的所有 chunk：读 M_{t-1} → 损失 → 写 M_t。

    **pair RNG 与 shuffle RNG 必须分开。** 若共用一个 generator，`shuffled` 臂在
    `write()` 里每层调 H 次 randperm 会额外消耗随机数，于是从第二个 chunk 起
    stateful 与 shuffled 采到的 (i,j) 就不同了 —— 那个差值里会混进"两臂看到的
    样本不一样"，而这恰恰是 `stateful − shuffled` 要排除的东西。

    **两级损失**。`level="pair"` 是跨 (层×kv头×token) 的**全局**阈值化，所以只在
    头内采样成对样本，等于完全不监督"layer 23/head 2 的 token 该不该压过
    layer 5/head 1 的"——而跨层/头的预算再分配很可能正是旧 v1 selection 效应的
    主要来源（手工版里同样的缺口叫 β_group）。
    全局项能成立是因为教师的 U 已经是 **W_O 投影后的残差流单位**，天然跨组可比；
    而且全局项**不除以逐头 σ**，因为全局阈值比的就是原始分数。
    """
    H, L = doc["H"], doc["L"]
    # 状态是 (M_gru, M_dir) 二元组，不能直接 .to()
    M = [tuple(t.to(dev) for t in cm.init_state(l)) for l in range(L)]
    tot_l, tot_a, cnt = 0.0, 0.0, 0   # tot_a 累计的是 acc(s')−acc(s0)
    tot_g, gcnt = 0.0, 0
    # **两项必须分开聚合。** 若都塞进一个 list 再取均值，每个 chunk 有 L≈28 个头内项
    # 却只有 1 个全局项 ⇒ lam_global=1 实际权重只有 1/28，"加了全局监督"名不副实。
    losses, gl_losses = [], []
    for ci, ch in enumerate(doc["chunks"]):
        new_M = []
        g_sp, g_s0, g_U = [], [], []
        # 全局尺度：跨该 chunk 所有 (层,kv头) 的基线分标准差
        # **必须用教师存的全量 σ_g**。从 ch["layers"] 的 s0 现算是错的：那是
        # 「近阈值 + 随机」的有偏子集，σ 被低估，而推理侧
        # （learned_ctrlcache.py）算的是整块 16000 个候选的 σ ⇒ margin 特征的
        # 尺度训练/部署不一致。老 trace 没存就退回旧算法，并只警告一次。
        if "gsig" in ch:
            gsig_doc = torch.tensor(float(ch["gsig"]), device=dev).clamp_min(1e-6)
        else:
            if not getattr(run_doc, "_warned", False):
                print("  [warn] trace 无 gsig（旧版教师），margin 尺度退回子集估计",
                      flush=True); run_doc._warned = True
            gsig_doc = torch.cat([p_["s0"].reshape(-1) for p_ in ch["layers"]]
                                 ).float().std().clamp_min(1e-6).to(dev)
        for l, pl in enumerate(ch["layers"]):
            k = pl["k"].to(dev).float()
            v = pl["v"].to(dev).float()
            s0 = pl["s0"].to(dev).float()
            U = pl["U"].to(dev).float()
            ret = pl["ret"].to(dev)
            nn_ = pl["n_near"]
            xr_raw = cm.raw(k, v)                              # [H,n,2d]
            x = cm.feat(xr_raw)                                # [H,n,d_m]
            q = cm.q_read(xr_raw)
            r = cm.read(M[l], xr_raw)
            # 到**全局**阈值的距离：level="pair" 决定去留的是 s0−τ，不是头内排名
            thr = pl.get("thres", None)
            mg = None if thr is None else (s0 - float(thr)) / gsig_doc
            st_ = None if "sig_h" not in pl else (
                pl["mu_h"].to(dev).float(), pl["sig_h"].to(dev).float(), gsig_doc)
            ds = cm.delta(x, r, s0, q=q, margin=mg, stats=st_)
            sp = s0 + ds
            sig = s0.std(-1, keepdim=True).clamp_min(1e-6)
            # **只在近阈值子集上算排序损失**
            # 第一个 chunk 还没有历史可读，它的监督对 B 的命题无信息
            if not (skip_first_loss and ci == 0):
                lo, a, a0 = pair_loss(sp[:, :nn_], s0[:, :nn_], U[:, :nn_],
                                      sig, n_pairs, gen, pair_w=pair_w)
                losses.append(lo)
                tot_l += float(lo); tot_a += float(a - a0); cnt += 1
                g_sp.append(sp[:, :nn_].reshape(-1))
                g_s0.append(s0[:, :nn_].reshape(-1))
                g_U.append(U[:, :nn_].reshape(-1))
            # **写入只用随机子集**（后半段），近阈值子集是有偏的
            xr = x[:, nn_:]
            rr = ret[:, nn_:]
            new_M.append(cm.write(M[l], xr, rr, ~rr, gen=shuf_gen or gen))
            del k, v, x, r, q, xr_raw
        # ---- 跨 (层,kv头) 的全局排序项 ----
        if lam_global > 0 and g_sp:
            gs = torch.cat(g_sp); g0 = torch.cat(g_s0); gu = torch.cat(g_U)
            gsig = g0.std().clamp_min(1e-6)          # 全局尺度，不用逐头 σ
            lg_, ag, ag0 = _flat_pairs(gs, g0, gu, gsig, n_pairs, gen,
                                       pair_w=pair_w)
            gl_losses.append(lg_)
            tot_g += float(ag - ag0); gcnt += 1
        M = new_M
    if not losses and not gl_losses:
        return None, 0.0, 0.0, 0.0
    tot = torch.zeros((), device=dev)
    if losses:
        tot = tot + torch.stack(losses).mean()
    if gl_losses:
        tot = tot + lam_global * torch.stack(gl_losses).mean()
    return (tot, tot_l / max(cnt, 1), tot_a / max(cnt, 1), tot_g / max(gcnt, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--d_kv", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_pairs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--pair_w", default="linear", choices=["linear", "none"],
                    help="成对损失是否按 |ΔU| 加权。top-B 的 selection regret 就是被错换"
                         "成对的 |ΔU| 之和，所以 linear 才是它的可微代理；none 做消融")
    ap.add_argument("--lam_global", type=float, default=1.0,
                    help="跨 (层,kv头) 全局排序项的权重。0 = 只学头内重排，"
                         "那样跨层/头的预算再分配完全没有监督")
    ap.add_argument("--out", default="varikv/ctrlm")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(os.path.join(ROOT, a.traces, "doc*.pt")))
    assert files, f"没有 trace，先跑 scratch_ctrl_teacher.py（{a.traces}）"
    random.Random(a.seed).shuffle(files)
    n_val = max(1, int(len(files) * a.val_frac))
    val_f, tr_f = files[:n_val], files[n_val:]
    print(f"训练 {len(tr_f)} 篇 / 验证 {len(val_f)} 篇", flush=True)
    docs_tr = [torch.load(f, map_location="cpu") for f in tr_f]
    docs_va = [torch.load(f, map_location="cpu") for f in val_f]
    L, H = docs_tr[0]["L"], docs_tr[0]["H"]

    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    hist = {}
    for mode in ("stateful", "memoryless", "shuffled"):
        # **三臂必须同一初始化、同一数据顺序、同一 pair 采样**，否则差异混进随机性
        torch.manual_seed(a.seed)
        cm = ControlMemory(a.d_kv, L, H, n_slots=a.slots, d_m=a.dim,
                           mode=mode).to(dev)
        opt = torch.optim.AdamW(cm.parameters(), lr=a.lr, weight_decay=0.01)
        print(f"\n=== {mode}　参数 {cm.n_params()/1e3:.1f}K ===", flush=True)
        for ep in range(a.epochs):
            cm.train()
            # 两个独立 generator：pair 的种子三臂完全相同，shuffle 的单独走
            g = torch.Generator(device=dev).manual_seed(a.seed * 1000 + ep)
            gs = torch.Generator(device=dev).manual_seed(a.seed * 7919 + ep)
            order = list(range(len(docs_tr)))
            random.Random(a.seed * 100 + ep).shuffle(order)
            el, ea, n = 0.0, 0.0, 0
            for di in order:
                loss, l_, acc, gacc = run_doc(cm, docs_tr[di], dev, a.n_pairs, g,
                                              lam_global=a.lam_global, shuf_gen=gs,
                                              pair_w=a.pair_w)
                if loss is None:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cm.parameters(), 1.0)
                opt.step()
                el += l_; ea += acc; n += 1
            cm.eval()
            with torch.no_grad():
                gv = torch.Generator(device=dev).manual_seed(12345)
                gvs = torch.Generator(device=dev).manual_seed(54321)
                vl, va, vg, m_ = 0.0, 0.0, 0.0, 0
                for d_ in docs_va:
                    _, l_, acc, gacc = run_doc(cm, d_, dev, a.n_pairs, gv,
                                               train=False,
                                               lam_global=a.lam_global,
                                               shuf_gen=gvs, pair_w=a.pair_w)
                    vl += l_; va += acc; vg += gacc; m_ += 1
            print(f"  ep{ep} train loss {el/max(n,1):.4f} acc {ea/max(n,1):.4f} | "
                  f"val 头内Δacc {va/max(m_,1):+.4f} **全局Δacc {vg/max(m_,1):+.4f}** | "
                  f"alpha {float(cm.alpha):.4f}", flush=True)
            hist.setdefault(mode, []).append((va / max(m_, 1), vg / max(m_, 1)))
        torch.save(dict(state=cm.state_dict(), mode=mode, slots=a.slots,
                        dim=a.dim, d_kv=a.d_kv, L=L, H=H, args=vars(a)),
                   os.path.join(ROOT, a.out, f"{mode}.pt"))

    print("\n" + "=" * 78)
    print("验证集成对排序准确率（最后一个 epoch）—— **这是本实验的判据**")
    for k, v in hist.items():
        print(f"  {k:<11} 头内 {v[-1][0]:+.4f}   全局 {v[-1][1]:+.4f}")
    if "stateful" in hist and "shuffled" in hist:
        d = hist["stateful"][-1][1] - hist["shuffled"][-1][1]
        print(f"\n  stateful − shuffled = {d:+.4f}")
        print("  >0 且稳定 ⇒ 历史提供了当前 KV 之外的增量信息（I(U;M|X)>0）")
        print("  ≈0        ⇒ 只是学了个更好的普通 scorer，B 的核心命题不成立")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
