#!/usr/bin/env python3
"""定位并修复「方向性历史传不过去」——最终版 B 的仪器问题。

背景：`scratch_ctrl_ladder.py` 的分级正对照测出

    local（不需历史）      +0.0310        流水线本身是通的
    scalar（历史传标量）    +0.0139        但 shuffled 也 +0.0157 —— 因为 shuffle
                                          保留集合大小，"驱逐比例"这个信号穿得过去
    direction（历史传方向） +0.0001        stateful − shuffled = −0.0016，传不过去

独立读出投影、d_m 128、非学习均值通路，三样都试过，都没用。这是个**会污染主结论**
的仪器问题：真实实验若跑出 stateful ≈ shuffled，分不清是数据没信号还是仪器传不动。

本脚本用四个变体分离原因，全部只在 direction 这一格上比：

  oracle   头直接拿到真实的 w（绕过整条记忆通路）
           ← **上界对照**。若它也≈0，说明瓶颈在 α/损失，不在记忆，
             那整个合成测试的标定就是错的，得先改测试而不是改架构。
  base     当前架构
  no_gru   M 直接等于均值池化（去掉 GRU 与注意力池化）
           ← 分离"递归门控压掉了信号"这一假设
  dm256    d_m=256（等于原始特征维度，瓶颈完全消失）
           ← 分离容量假设

判读：第一个让 `stateful − shuffled` 明显为正的变体，就是原因所在。
若四个全≈0（含 oracle），则问题在损失/α 的标定，不在架构。
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
from attention.control_memory import ControlMemory           # noqa: E402

L, H, DKV, NEAR, NRAND = 4, 4, 128, 128, 256
N = NEAR + NRAND


def make_doc(seed, n_chunk=3, dev="cpu"):
    """direction 信号：U_i = <k_i, w>，w = 上一 chunk **被驱逐集合**的均值方向。

    只有读得到"上一块具体丢了哪些"才能预测；shuffle 只保留集合大小，w 会被打乱。
    """
    g = torch.Generator(device=dev).manual_seed(seed)
    chs, prev_k, prev_ret = [], None, None
    for t in range(n_chunk):
        per, cur_k, cur_ret = [], [], []
        for l in range(L):
            k = torch.randn(H, N, DKV, generator=g, device=dev)
            v = torch.randn(H, N, DKV, generator=g, device=dev)
            s0 = torch.randn(H, N, generator=g, device=dev)
            ret = torch.rand(H, N, generator=g, device=dev) < 0.3
            if prev_k is None:
                U = torch.randn(H, NEAR, generator=g, device=dev)
                w = torch.zeros(H, DKV, device=dev)
            else:
                w = torch.stack([prev_k[l][h][~prev_ret[l][h]].mean(0)
                                 for h in range(H)])
                U = torch.einsum("hnd,hd->hn", k[:, :NEAR], w)
            per.append(dict(k=k, v=v, s0=s0, ret=ret, U=U, w=w, n_near=NEAR))
            cur_k.append(k); cur_ret.append(ret)
        prev_k, prev_ret = cur_k, cur_ret
        chs.append(per)
    return chs


class Runner(nn.Module):
    def __init__(self, variant, mode, d_m):
        super().__init__()
        self.variant, self.mode = variant, mode
        self.cm = ControlMemory(DKV, L, H, n_slots=8, d_m=d_m, mode=mode)
        if variant == "oracle":
            self.orc = nn.Linear(DKV, d_m)     # 真实 w → 读出向量

    def forward(self, doc, n_pairs, gen):
        cm = self.cm
        M = [cm.init_state(l) for l in range(L)]
        losses, acc_d = [], []
        for per in doc:
            newM = []
            for l, pl in enumerate(per):
                raw = cm.raw(pl["k"], pl["v"])
                x = cm.feat(raw)
                q = cm.q_read(raw)
                if self.variant == "oracle":
                    r = self.orc(pl["w"])[:, None].expand(-1, N, -1)
                else:
                    r = cm.read(M[l], raw)
                ds = cm.delta(x, r, pl["s0"], q=q)
                sp = pl["s0"] + ds
                sig = pl["s0"].std(-1, keepdim=True).clamp_min(1e-6)
                nn_ = pl["n_near"]
                a, b = sp[:, :nn_], pl["s0"][:, :nn_]
                U = pl["U"]
                i = torch.randint(0, nn_, (H, n_pairs), generator=gen, device=a.device)
                j = torch.randint(0, nn_, (H, n_pairs), generator=gen, device=a.device)
                du = torch.gather(U, 1, i) - torch.gather(U, 1, j)
                lg = (torch.gather(a, 1, i) - torch.gather(a, 1, j)) / sig * du.sign()
                lg0 = (torch.gather(b, 1, i) - torch.gather(b, 1, j)) / sig * du.sign()
                losses.append(torch.nn.functional.softplus(-lg).mean())
                acc_d.append(float((lg > 0).float().mean() - (lg0 > 0).float().mean()))
                if self.variant == "no_gru":
                    xr, rr = x[:, nn_:], pl["ret"][:, nn_:]
                    if self.mode == "memoryless":
                        newM.append(M[l])
                    else:
                        if self.mode == "shuffled":
                            p = torch.stack([torch.randperm(xr.shape[1], generator=gen,
                                                            device=xr.device)
                                             for _ in range(H)])
                            rr = torch.gather(rr, 1, p)
                        mu = cm._mean(xr, ~rr).expand(-1, M[l].shape[1], -1)
                        newM.append(mu)
                else:
                    xr, rr = x[:, nn_:], pl["ret"][:, nn_:]
                    newM.append(cm.write(M[l], xr, rr, ~rr, gen=gen))
            M = newM
        return torch.stack(losses).mean(), sum(acc_d) / len(acc_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n_train", type=int, default=48)
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n_pairs", type=int, default=512)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = [make_doc(s, dev=dev) for s in range(a.n_train)]
    va = [make_doc(9000 + s, dev=dev) for s in range(a.n_val)]
    print(f"device={dev}  train={len(tr)} val={len(va)}  "
          f"L={L} H={H} n={N}\n", flush=True)

    print(f"{'变体':<10}{'d_m':>5}{'stateful':>11}{'shuffled':>11}{'差':>10}")
    for variant, d_m in (("oracle", 128), ("base", 128), ("no_gru", 128),
                         ("dm256", 256)):
        res = {}
        for mode in ("stateful", "shuffled"):
            torch.manual_seed(7)
            R = Runner("base" if variant == "dm256" else variant, mode, d_m).to(dev)
            opt = torch.optim.AdamW(R.parameters(), lr=a.lr, weight_decay=0.01)
            for ep in range(a.epochs):
                g = torch.Generator(device=dev).manual_seed(ep)
                for doc in tr:
                    loss, _ = R(doc, a.n_pairs, g)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
                    opt.step()
            with torch.no_grad():
                gv = torch.Generator(device=dev).manual_seed(999)
                res[mode] = sum(R(d, 1024, gv)[1] for d in va) / len(va)
            if variant == "oracle":
                break                      # oracle 与历史无关，一臂足够
        s = res["stateful"]; h = res.get("shuffled", float("nan"))
        print(f"{variant:<10}{d_m:>5}{s:>+11.4f}{h:>+11.4f}{s-h:>+10.4f}", flush=True)
    print("\n判读：第一个让『差』明显为正的变体即原因所在。"
          "\n若 oracle 也≈0 ⇒ 瓶颈在 α/损失标定，不在记忆架构，应先改测试。")


if __name__ == "__main__":
    sys.exit(main())
