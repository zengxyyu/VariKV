#!/usr/bin/env python3
"""方向传递诊断 v4：**给每一格配误差棒**。

为什么要有这个脚本。v1–v3 的整条阶梯

    oracle +0.2797 / no_gru +0.0385 / base −0.0015 / dm256 −0.0009 / α=1.0 +0.0102
    双通路 α=0.05 +0.0145 / α=0.30 −0.0044 / base(α自学) −0.0061

每一格都是**单次训练的一次读数**，而相邻配置的差值符号来回跳、量级都压在 0.01 附近。
2026-08-14 的 v1 复现刚刚证明过同一件事的代价：同一份代码三次训练在 Retr.KV 上跨度
39 分，配对 bootstrap 的 ★ 照样给出互相矛盾的方向——**配对 bootstrap 量化的是评测集
抽样噪声，不是优化器方差**。这里的 12 篇验证文档同理：它只覆盖验证集噪声，训练轨迹
的方差完全没有被测。

所以本脚本改成每个变体 n 个种子，报**逐种子的 `stateful − shuffled` 和跨种子跨度**。
判读规则先定死，避免事后挑：

    若 |mean(diff)| < spread ⇒ 这一格与 0 不可分，不许解读符号。
    只有 mean(diff) 明显超出跨种子跨度，才算"传得过去"。

同一个种子下两臂共享初始化与 pair 采样（pair RNG 与 shuffle RNG 分离，见 P0-1），
所以逐种子的差值是配对量，比两臂各自的绝对值更可信。

变体：
  dual    当前双通路架构（GRU 递归 + 方向 EMA）
  mean    只留方向 EMA，GRU 通路冻结在初值 ⇒ 分离"递归门控压掉了信号"
  oracle  头直接拿真实 w，绕开整条记忆通路 ⇒ 上界（单臂，无 shuffled 对照）
"""
import argparse
import os
import statistics
import sys

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
sys.path.insert(0, ROOT)
import importlib.util as _iu                                     # noqa: E402

_s = _iu.spec_from_file_location("D", os.path.join(ROOT, "scratch_ctrl_dirfix.py"))
D = _iu.module_from_spec(_s); _s.loader.exec_module(D)
from attention.control_memory import ControlMemory               # noqa: E402


class Runner(nn.Module):
    def __init__(self, variant, mode, d_m=128, alpha_max=1.0, alpha_init=0.05):
        super().__init__()
        self.variant, self.mode = variant, mode
        self.cm = ControlMemory(D.DKV, D.L, D.H, n_slots=8, d_m=d_m, mode=mode,
                                alpha_max=alpha_max, alpha_init=alpha_init)
        if variant == "oracle":
            self.orc = nn.Linear(D.DKV, d_m)

    def _write(self, state, x, m_ret, m_evi, gen):
        """mean 变体：GRU 通路原样传下去，只更新方向 EMA。

        直接复用 `ControlMemory.write` 再把 M2 换回 M 是不行的——那样 GRU 仍然
        参与前向、仍然占梯度路径。这里重写这一步，让 GRU 完全不被调用。
        """
        cm = self.cm
        if self.variant != "mean":
            return cm.write(state, x, m_ret, m_evi, gen=gen)
        M, Dm = state
        if cm.mode == "memoryless":
            return state
        if cm.mode == "shuffled":
            n = x.shape[1]
            gdev = gen.device if gen is not None else x.device
            perm = torch.stack([torch.randperm(n, generator=gen, device=gdev)
                                for _ in range(x.shape[0])]).to(x.device)
            m_ret = torch.gather(m_ret, 1, perm)
            m_evi = torch.gather(m_evi, 1, perm)
        rho = torch.sigmoid(cm.dir_decay)[None, :, None]
        D2 = rho * Dm + (1.0 - rho) * torch.cat([cm._mean(x, m_ret),
                                                 cm._mean(x, m_evi)], dim=1)
        return (M, D2)

    def forward(self, doc, n_pairs, gen, shuf_gen=None, skip_first=True):
        cm = self.cm
        M = [cm.init_state(l) for l in range(D.L)]
        losses, accd = [], []
        for t, per in enumerate(doc):
            newM = []
            for l, pl in enumerate(per):
                raw = cm.raw(pl["k"], pl["v"])
                x, q = cm.feat(raw), cm.q_read(raw)
                if self.variant == "oracle":
                    r = self.orc(pl["w"])[:, None].expand(-1, D.N, -1)
                else:
                    r = cm.read(M[l], raw)
                ds = cm.delta(x, r, pl["s0"], q=q)
                nn_ = pl["n_near"]
                # chunk 0 的 U 是纯噪声（还没有历史），只写不算损失
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
                # **shuffle 用独立 generator**：共用一个的话 shuffled 臂会多消耗抽样，
                # 从第二个 chunk 起两臂看到的 pair 就不同了，配对性被破坏（P0-1）。
                newM.append(self._write(M[l], xr, rr, ~rr, shuf_gen or gen))
            M = newM
        return torch.stack(losses).mean(), sum(accd) / max(len(accd), 1)


def train_eval(variant, mode, seed, tr, va, epochs, lr, dev, alpha_init):
    torch.manual_seed(seed)
    R = Runner(variant, mode, alpha_init=alpha_init).to(dev)
    opt = torch.optim.AdamW(R.parameters(), lr=lr, weight_decay=0.01)
    for ep in range(epochs):
        # 两臂在同一 (seed, ep) 下看到**完全相同**的 pair 序列
        g = torch.Generator(device=dev).manual_seed(1000 * seed + ep)
        gsh = torch.Generator(device=dev).manual_seed(7919 + 1000 * seed + ep)
        for doc in tr:
            loss, _ = R(doc, 512, g, shuf_gen=gsh)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
            opt.step()
    with torch.no_grad():
        gv = torch.Generator(device=dev).manual_seed(999)
        gvs = torch.Generator(device=dev).manual_seed(54321)
        acc = sum(R(d, 1024, gv, shuf_gen=gvs)[1] for d in va) / len(va)
    return acc, float(R.cm.alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="dual", choices=("dual", "mean", "oracle"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--alpha_init", type=float, default=0.05)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = [D.make_doc(s, dev=dev) for s in range(48)]
    va = [D.make_doc(9000 + s, dev=dev) for s in range(12)]
    print(f"variant={a.variant}  seeds={a.seeds}  device={dev}  "
          f"direction 信号  chunk0 只写不算损失\n", flush=True)
    print(f"{'seed':>5}{'stateful':>11}{'shuffled':>11}{'差':>10}{'α_end':>8}")
    diffs = []
    for sd in a.seeds:
        res, al = {}, 0.0
        for mode in ("stateful", "shuffled"):
            res[mode], al = train_eval(a.variant, mode, sd, tr, va,
                                       a.epochs, a.lr, dev, a.alpha_init)
            if a.variant == "oracle":
                # 与历史无关，一臂足够。记 0 而不是 nan，好让下面的
                # mean±sd 直接给出**上界本身**的跨种子跨度——上界也需要误差棒。
                res["shuffled"] = 0.0
                break
        d = res["stateful"] - res["shuffled"]
        diffs.append(d)
        print(f"{sd:>5}{res['stateful']:>+11.4f}{res['shuffled']:>+11.4f}"
              f"{d:>+10.4f}{al:>8.3f}", flush=True)
    fin = [d for d in diffs if d == d]
    if len(fin) >= 2:
        m, sp = statistics.mean(fin), statistics.stdev(fin)
        print(f"\n{'mean±sd':>5}{'':>22}{m:>+10.4f} ± {sp:.4f}")
        print("判读：" + ("**与 0 不可分**，不许解读符号" if abs(m) < sp
                         else "均值超出跨种子跨度，可以解读"))
    elif fin:
        print(f"\n上界（单臂）: {fin[0]:+.4f}" if a.variant != "oracle" else "")


if __name__ == "__main__":
    sys.exit(main())
