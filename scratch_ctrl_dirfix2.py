#!/usr/bin/env python3
"""方向传递诊断 v2：直接测「α 太小卡住长梯度路径」+ 去掉 chunk0 的噪声监督。

v1（`scratch_ctrl_dirfix.py`）已经给出关键一档：**oracle = +0.2755**。
头直接拿到真实的 w 就能拿到 0.2755，说明损失与 α 完全支持大幅改善，
**瓶颈确确实实在 writer→read 这条记忆通路上**，不在标定。

v1 没有隔离的两件事，这里补上：

1. **α 初始为 3.35e-4，会把整条长梯度路径的梯度一起乘以它。**
   oracle 的路径只有 head→orc（两层），能自举把 α 顶上去；
   base 的路径是 head→read→GRU→pool→x_proj（五段），自举不起来。
   这里把 α **固定**（冻结，不参与优化）来消除这个耦合。

2. **chunk 0 的 U 是纯随机噪声**（那时还没有历史），v1 却照样在它上面算排序损失，
   一半监督是不可学的噪声。这里 chunk 0 **只写不算损失**。

判读：若固定 α 后 `stateful − shuffled` 明显为正 ⇒ 原因是 α 的自举问题，
架构本身没坏，真实训练只需给 α 一个 warmup。若仍 ≈0 ⇒ 记忆通路本身表达/优化不了方向。
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
sys.path.insert(0, ROOT)
import importlib.util as _iu                                  # noqa: E402

_s = _iu.spec_from_file_location("D", os.path.join(ROOT, "scratch_ctrl_dirfix.py"))
D = _iu.module_from_spec(_s)
_s.loader.exec_module(D)
from attention.control_memory import ControlMemory            # noqa: E402


class R2(torch.nn.Module):
    def __init__(self, mode, d_m, alpha):
        super().__init__()
        self.cm = ControlMemory(D.DKV, D.L, D.H, n_slots=8, d_m=d_m, mode=mode)
        with torch.no_grad():                     # **冻结 α**，消除自举耦合
            self.cm.alpha_on.fill_(20.0)          # sigmoid(20) ≈ 1
            self.cm.log_alpha.fill_(float(torch.tensor(alpha).log()))
        self.cm.alpha_on.requires_grad_(False)
        self.cm.log_alpha.requires_grad_(False)
        self.mode = mode

    def forward(self, doc, n_pairs, gen, skip_first=True):
        cm = self.cm
        M = [cm.init_state(l) for l in range(D.L)]
        losses, accd = [], []
        for t, per in enumerate(doc):
            newM = []
            for l, pl in enumerate(per):
                raw = cm.raw(pl["k"], pl["v"])
                x = cm.feat(raw)
                q = cm.q_read(raw)
                r = cm.read(M[l], raw)
                ds = cm.delta(x, r, pl["s0"], q=q)
                nn_ = pl["n_near"]
                # **chunk 0 的 U 是噪声（还没有历史），只写不算损失**
                if not (skip_first and t == 0):
                    sp, b = (pl["s0"] + ds)[:, :nn_], pl["s0"][:, :nn_]
                    sig = pl["s0"].std(-1, keepdim=True).clamp_min(1e-6)
                    U = pl["U"]
                    i = torch.randint(0, nn_, (D.H, n_pairs), generator=gen,
                                      device=sp.device)
                    j = torch.randint(0, nn_, (D.H, n_pairs), generator=gen,
                                      device=sp.device)
                    du = torch.gather(U, 1, i) - torch.gather(U, 1, j)
                    lg = (torch.gather(sp, 1, i) - torch.gather(sp, 1, j)) / sig * du.sign()
                    l0 = (torch.gather(b, 1, i) - torch.gather(b, 1, j)) / sig * du.sign()
                    losses.append(torch.nn.functional.softplus(-lg).mean())
                    accd.append(float((lg > 0).float().mean() - (l0 > 0).float().mean()))
                xr, rr = x[:, nn_:], pl["ret"][:, nn_:]
                newM.append(cm.write(M[l], xr, rr, ~rr, gen=gen))
            M = newM
        return torch.stack(losses).mean(), sum(accd) / max(len(accd), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = [D.make_doc(s, dev=dev) for s in range(48)]
    va = [D.make_doc(9000 + s, dev=dev) for s in range(12)]
    print(f"device={dev}  direction 信号  chunk0 只写不算损失\n", flush=True)
    print(f"{'固定α':>8}{'stateful':>11}{'shuffled':>11}{'差':>10}")
    for alpha in (1.0, 0.3, 0.1):
        res = {}
        for mode in ("stateful", "shuffled"):
            torch.manual_seed(7)
            R = R2(mode, 128, alpha).to(dev)
            opt = torch.optim.AdamW([p for p in R.parameters() if p.requires_grad],
                                    lr=a.lr, weight_decay=0.01)
            for ep in range(a.epochs):
                g = torch.Generator(device=dev).manual_seed(ep)
                for doc in tr:
                    loss, _ = R(doc, 512, g)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
                    opt.step()
            with torch.no_grad():
                gv = torch.Generator(device=dev).manual_seed(999)
                res[mode] = sum(R(d, 1024, gv)[1] for d in va) / len(va)
        print(f"{alpha:>8.2f}{res['stateful']:>+11.4f}{res['shuffled']:>+11.4f}"
              f"{res['stateful']-res['shuffled']:>+10.4f}", flush=True)
    print("\n参照：v1 的 oracle（头直接拿真 w）= +0.2755，base（α 自学）≈ 0")
    print("若固定 α 后差值明显为正 ⇒ 是 α 自举问题，架构没坏，真实训练加 warmup 即可")


if __name__ == "__main__":
    sys.exit(main())
