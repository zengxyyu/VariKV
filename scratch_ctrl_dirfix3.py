#!/usr/bin/env python3
"""方向传递诊断 v3：验证**双通路**架构是否把 direction 那一格救回来。

v1/v2 的定位结论：
    oracle（头直接拿真 w）  +0.2797   ← 上界
    base（纯 GRU）          −0.0015
    dm256（容量翻倍）       −0.0009   ← **容量假说被排除**
    no_gru（纯均值池化）    +0.0385   ← 跳过 GRU 才传得过去
    固定 α=1.0              +0.0102   ← α 是次因，量级小一档

据此把状态拆成 M_gru（门控递归）+ M_dir（池化均值的**线性 EMA**，不过门），
读出对二者拼接做注意力。本脚本要回答的唯一问题：

    双通路的 base 能不能达到 no_gru 的 +0.0385？

若能 ⇒ 仪器合格，可以去做真实 B；若不能 ⇒ 还有别的东西在挡。
"""
import os
import sys

import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
sys.path.insert(0, ROOT)
import importlib.util as _iu                                     # noqa: E402

_s = _iu.spec_from_file_location("D", os.path.join(ROOT, "scratch_ctrl_dirfix.py"))
D = _iu.module_from_spec(_s); _s.loader.exec_module(D)
from attention.control_memory import ControlMemory               # noqa: E402


def run(cm, doc, n_pairs, gen, skip_first=True):
    M = [cm.init_state(l) for l in range(D.L)]
    losses, accd = [], []
    for t, per in enumerate(doc):
        newM = []
        for l, pl in enumerate(per):
            raw = cm.raw(pl["k"], pl["v"])
            x, q = cm.feat(raw), cm.q_read(raw)
            r = cm.read(M[l], raw)
            ds = cm.delta(x, r, pl["s0"], q=q)
            nn_ = pl["n_near"]
            if not (skip_first and t == 0):          # chunk0 的 U 是噪声
                sp, b = (pl["s0"] + ds)[:, :nn_], pl["s0"][:, :nn_]
                sig = pl["s0"].std(-1, keepdim=True).clamp_min(1e-6)
                U = pl["U"]
                i = torch.randint(0, nn_, (D.H, n_pairs), generator=gen, device=sp.device)
                j = torch.randint(0, nn_, (D.H, n_pairs), generator=gen, device=sp.device)
                du = torch.gather(U, 1, i) - torch.gather(U, 1, j)
                lg = (torch.gather(sp, 1, i) - torch.gather(sp, 1, j)) / sig * du.sign()
                l0 = (torch.gather(b, 1, i) - torch.gather(b, 1, j)) / sig * du.sign()
                losses.append(torch.nn.functional.softplus(-lg).mean())
                accd.append(float((lg > 0).float().mean() - (l0 > 0).float().mean()))
            newM.append(cm.write(M[l], x[:, nn_:], pl["ret"][:, nn_:],
                                 ~pl["ret"][:, nn_:], gen=gen))
        M = newM
    return torch.stack(losses).mean(), sum(accd) / max(len(accd), 1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = [D.make_doc(s, dev=dev) for s in range(48)]
    va = [D.make_doc(9000 + s, dev=dev) for s in range(12)]
    print(f"device={dev}　双通路 base　direction 信号\n", flush=True)
    print(f"{'α_init':>8}{'stateful':>11}{'shuffled':>11}{'差':>10}"
          f"{'   (参照 no_gru +0.0385, 旧 base −0.0015)'}")
    for ainit in (0.05, 0.3):
        res = {}
        for mode in ("stateful", "shuffled"):
            torch.manual_seed(7)
            cm = ControlMemory(D.DKV, D.L, D.H, n_slots=8, d_m=128, mode=mode,
                               alpha_max=1.0, alpha_init=ainit).to(dev)
            opt = torch.optim.AdamW(cm.parameters(), lr=3e-3, weight_decay=0.01)
            for ep in range(150):
                g = torch.Generator(device=dev).manual_seed(ep)
                for doc in tr:
                    loss, _ = run(cm, doc, 512, g)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(cm.parameters(), 1.0); opt.step()
            with torch.no_grad():
                gv = torch.Generator(device=dev).manual_seed(999)
                res[mode] = sum(run(cm, d, 1024, gv)[1] for d in va) / len(va)
            if mode == "stateful":
                st_alpha = float(cm.alpha)
        print(f"{ainit:>8.2f}{res['stateful']:>+11.4f}{res['shuffled']:>+11.4f}"
              f"{res['stateful']-res['shuffled']:>+10.4f}   α_final={st_alpha:.3f}",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
