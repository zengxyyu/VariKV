"""各臂把配额搬到了哪里：`M_b`、`R_starve`、施主/受主层分布。零 GPU。

回答的问题：`sgs` 的 +18.27 主要来自**把饿死头拉过阈值**，还是**已活跃头之间的
细粒度重分配**？这决定死区/地板那条线是主机制还是次要分析。

口径与已知偏差（必须随结论一起说）：
  * teacher trace 每 (chunk,层,头) 只存 **768 个候选**，且按 |s⁰−τ| 采样 ⇒
    **不是**该头的全部 token。因此这里的 b_h 是**候选池内**的配额，
    不是真实全局配额；`b⁰_h = 0` 表示「近阈值候选一个都没保留」，
    是「饿死」的**代理**而非定义。真值需要改 cache 把 valid 掩码落盘再跑 GPU。
  * 预算严格相等：新掩码在**同一候选池内取 top-B**，B = 基线保留数。
    这一步是必须的 —— 用固定 τ 判定会把「Δs 的全局常数偏移」这个
    **规范自由度**当成真实效应（本项目已犯过一次，见 CLAUDE.md）。
  * 因此 Σ_h Δb_h = 0 恒成立，`M_b = ½Σ|Δb_h|` 就是「搬了多少格」。
"""
import os, sys, glob, argparse, numpy as np, torch
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.calib_scorer import CalibScorer                      # noqa: E402


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu")
    m = CalibScorer(sd.get("d_kv", 128), sd["L"], sd["H"],
                    n_slots=sd.get("slots", 8), d_m=sd.get("dim", 128),
                    mode="memoryless", arch=sd["arch"],
                    scale=sd.get("scale", "head"),
                    alpha_max=sd.get("alpha_max", 1.0))
    m.load_state_dict(sd["state"])
    return m.eval(), sd["arch"], sd.get("scale", "head"), float(m.alpha)


class _Zero:
    """Δs ≡ 0 的**零对照**。

    必需，不是可选：`ret` 是真实全局阈值（对该头**全部** token）下的保留掩码，
    而本脚本在 **768 个候选**内重取 top-B。即使不加任何修正，这两者也未必一致，
    于是重构本身就会产生一个非零的 `M_b`。**不减掉这个底噪，任何 `M_b` 都不可读。**
    """
    def read(self, state, x_raw):
        return None

    def delta(self, x, r, s0, q=None, margin=None, stats=None):
        return torch.zeros_like(s0)


@torch.no_grad()
def analyse(m, traces, n_doc):
    Mb, Rst, n_chunk = [], [], 0
    lay_don = None; lay_rec = None; L_ = None
    n_starved_tot = n_starved_lifted = 0
    for f in sorted(glob.glob(os.path.join(ROOT, traces, "doc*.pt")))[:n_doc]:
        d = torch.load(f, map_location="cpu")
        L_ = L_ or d["L"]
        if lay_don is None:
            lay_don = np.zeros(d["L"]); lay_rec = np.zeros(d["L"])
        for ch in d["chunks"]:
            g, t = float(ch["gsig"]), float(ch["thres"])
            S0, DS, RET, LH = [], [], [], []
            for l, pl in enumerate(ch["layers"]):
                s0 = pl["s0"].float()                       # [H, n]
                mu, sh = pl["mu_h"].float(), pl["sig_h"].float()
                mg = (s0 - t) / g
                ds = m.delta(None, m.read((l,), None), s0, margin=mg, stats=(mu, sh, g))
                for h in range(s0.shape[0]):
                    S0.append(s0[h]); DS.append(ds[h]); RET.append(pl["ret"][h])
                    LH.append((l, h))
            S0 = torch.stack(S0); DS = torch.stack(DS); RET = torch.stack(RET)
            b0 = RET.sum(-1).numpy().astype(int)            # [G] 候选池内基线配额
            B = int(b0.sum())
            if B == 0:
                continue
            flat = (S0 + DS).reshape(-1)
            keep = torch.zeros_like(flat, dtype=torch.bool)
            keep[torch.topk(flat, B).indices] = True        # **等预算** top-B
            b1 = keep.reshape(S0.shape).sum(-1).numpy().astype(int)
            db = b1 - b0
            assert db.sum() == 0, db.sum()                  # 预算守恒
            mb = int(np.abs(db).sum()) // 2
            if mb == 0:
                continue
            starved = (b0 == 0)
            rst = int(db[starved].clip(min=0).sum()) / mb
            Mb.append(mb); Rst.append(rst); n_chunk += 1
            n_starved_tot += int(starved.sum())
            n_starved_lifted += int(((b1 > 0) & starved).sum())
            for gi, (l, h) in enumerate(LH):
                if db[gi] < 0: lay_don[l] += -db[gi]
                elif db[gi] > 0: lay_rec[l] += db[gi]
    return (np.array(Mb), np.array(Rst), n_chunk, lay_don, lay_rec,
            n_starved_tot, n_starved_lifted, L_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="scratch_ctrl_traces_v2_10")
    ap.add_argument("--n_doc", type=int, default=10)
    a = ap.parse_args()
    ARMS = [("scalar σ_h .999", "varikv/d10_scalar_s%d.pt/memoryless.pt"),
            ("sgs    σ_g .999", "varikv/sg_scalar_s%d.pt/memoryless.pt"),
            ("sg125  σ_g 1.25", "varikv/sg125_scalar_s%d.pt/memoryless.pt"),
            ("chead  σ_g 1.25", "varikv/chead_s%d.pt/memoryless.pt"),
            ("chd10  σ_g .999", "varikv/chead10_s%d.pt/memoryless.pt"),
            ("chead15 σ_g 1.5", "varikv/chead15_s%d.pt/memoryless.pt"),
            ("sg035  σ_g .349", "varikv/sg035_scalar_s%d.pt/memoryless.pt"),
            ("sh286  σ_h 2.861", "varikv/sh286_scalar_s%d.pt/memoryless.pt"),
            ("stat10 静态表", "varikv/stat10_s%d.pt/memoryless.pt"),
            ("chd0   去头嵌入", "varikv/chead0_s%d.pt/memoryless.pt")]
    print("**候选池内**配额移动（trace 口径，见文件头限定）\n")
    z_mb, z_rs, z_nc, *_ = analyse(_Zero(), a.traces, a.n_doc)
    if len(z_mb):
        print(f"  **零对照（Δs≡0）**：M_b 中位 {np.median(z_mb):.0f}，"
              f"R_starve {np.mean(z_rs):.3f}，{z_nc} 个 chunk")
        print(f"  ⇒ 各臂的 M_b **必须显著高于这个底噪**才说明它真的搬了配额。\n")
    else:
        print("  **零对照 M_b = 0**（候选池内 top-B 与 ret 逐位一致）⇒ 底噪为零，可直接读。\n")
    print(f"{'臂':<18}{'M_b 中位':>10}{'R_starve':>10}{'饿死头被抬起':>14}"
          f"{'施主 top3 层':>20}{'受主 top3 层':>20}")
    for nm, pat in ARMS:
        MB, RS, ST, LT = [], [], [], []
        don = rec = None
        for s in (0, 1, 2):
            p = os.path.join(ROOT, pat % s)
            if not os.path.exists(p):
                continue
            m, arch, sc, al = load(p)
            mb, rs, nc, ld, lr, st, li, L_ = analyse(m, a.traces, a.n_doc)
            if not len(mb):
                continue
            MB.append(np.median(mb)); RS.append(np.mean(rs))
            ST.append(li / max(st, 1))
            don = ld if don is None else don + ld
            rec = lr if rec is None else rec + lr
        if not MB:
            print(f"{nm:<18}  (缺 ckpt)"); continue
        d3 = ",".join(f"L{i}" for i in np.argsort(-don)[:3])
        r3 = ",".join(f"L{i}" for i in np.argsort(-rec)[:3])
        print(f"{nm:<18}{np.mean(MB):>10.0f}{np.mean(RS):>10.3f}"
              f"{np.mean(ST):>13.1%}{d3:>20}{r3:>20}")
