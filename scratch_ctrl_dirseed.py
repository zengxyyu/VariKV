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

    单样本 t 检验，H0: mean(diff)=0，df=n−1，双侧 95%。
    **不要用 `|mean|<sd` 这种土规则**：它等价于 |t|>√n，n=3 时是 1.73，而临界值是
    4.303 —— 会把 t=−2.98（p≈0.10）判成"可以解读"。这不是吹毛求疵：本文件的
    `mean` 那一格正好是这个数，按土规则会得出"稳定为负"的过强结论。
    n≥5 才有起码的检出力（df=4 临界 2.776）；这仍然只是**要不要继续 debug 的
    停止规则**，论文里报效应要用更多种子 + 对数据与种子双重 bootstrap。

同一个种子下两臂共享初始化与 pair 采样（pair RNG 与 shuffle RNG 分离，见 P0-1），
所以逐种子的差值是配对量，比两臂各自的绝对值更可信。

**A→E 分段短接阶梯。** 不要只跑 oracle/mean/dual 然后归因——那样只能知道"失败了"，
不知道信号在哪一段消失。每一档只把链路的一段换成"精确给定"，其余保持真实：

  A raw_oracle   r = Linear_fresh(w)              专用投影直接拿真值 → 测损失/α/head
  B proj_oracle  r = x_proj(wc)                   换成架构**自己的**共享投影（无 bias，
                                                  差向量里 bias 本就抵消）→ 测投影瓶颈
  C ema_direct   M_dir 走真实 EMA 递归，
                 r = (D_evi − D_ret) 直接给 head  → 测递归/池化，**绕过读出注意力**
  D read_exact   M_dir 填入**精确**对比向量，
                 r = cm.read(...) 真实注意力      → 测读出注意力
  E full         真实 write（池化+GRU+EMA）→ 真实 read → 完整链路
  E_mean         同 E 但 GRU 完全不被调用（纯 EMA）

判读：**第一个从"显著为正"掉到"与 0 不可分"的档位，就是瓶颈所在。**
A 已知 +0.2503±0.0017（旧仪器）。注意 A 与 B 的差别只在投影是否专用、是否共享参数。
"""
import argparse
import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
sys.path.insert(0, ROOT)
import importlib.util as _iu                                     # noqa: E402

_s = _iu.spec_from_file_location("D", os.path.join(ROOT, "scratch_ctrl_dirfix.py"))
D = _iu.module_from_spec(_s); _s.loader.exec_module(D)
from attention.control_memory import ControlMemory               # noqa: E402


class Runner(nn.Module):
    """六档变体共用一个 forward；差别只在 `r_i` 从哪来、状态怎么更新。

    A/B 完全绕开记忆，所以 shuffled 臂与 stateful 臂**逐位相同**，只跑一臂；
    它们那一列报的是"相对 s0 排序的准确率增益"，即上界。
    C/D/E 的状态真的携带信息，shuffled 才是有效对照，报两臂之差。
    """

    ONE_ARM = ("raw_oracle", "proj_oracle")

    def __init__(self, variant, mode, d_m=128, alpha_max=1.0, alpha_init=0.05):
        super().__init__()
        self.variant, self.mode = variant, mode
        self.cm = ControlMemory(D.DKV, D.L, D.H, n_slots=8, d_m=d_m, mode=mode,
                                alpha_max=alpha_max, alpha_init=alpha_init)
        if variant == "raw_oracle":
            self.orc = nn.Linear(D.DKV, d_m)      # 专用投影，与架构无关

    # ---------------------------------------------------------------- 掩码
    def _masks(self, m_ret, m_evi, gen):
        """shuffled：随机置换成员身份。保住条数与计算量，破坏"谁被丢了"。"""
        if self.mode != "shuffled":
            return m_ret, m_evi
        n = m_ret.shape[1]
        gdev = gen.device if gen is not None else m_ret.device
        perm = torch.stack([torch.randperm(n, generator=gen, device=gdev)
                            for _ in range(m_ret.shape[0])]).to(m_ret.device)
        return torch.gather(m_ret, 1, perm), torch.gather(m_evi, 1, perm)

    # ---------------------------------------------------------------- 写
    def _write(self, state, x, m_ret, m_evi, gen):
        cm, M, Dm = self.cm, *state
        if cm.mode == "memoryless":
            return state
        mr, me = self._masks(m_ret, m_evi, gen)
        pair = torch.cat([cm._mean(x, mr), cm._mean(x, me)], dim=1)   # [H,2,d]
        if self.variant == "read_exact":
            return (M, pair)                       # 精确对比向量，不过 EMA
        if self.variant in ("ema_direct", "full_mean"):
            # 真实 EMA 递归，但 **GRU 完全不被调用**（不能算完 M2 再丢弃——
            # 那样 GRU 仍在前向图里、仍占梯度路径）
            rho = torch.sigmoid(cm.dir_decay)[None, :, None]
            return (M, rho * Dm + (1.0 - rho) * pair)
        return cm.write(state, x, m_ret, m_evi, gen=gen)   # full：整条真实通路

    # ---------------------------------------------------------------- 读
    def _read(self, state, raw, pl):
        cm = self.cm
        if self.variant == "raw_oracle":
            return self.orc(pl["w"])[:, None].expand(-1, D.N, -1)
        if self.variant == "proj_oracle":
            # 架构**自己的**共享投影，且不加 bias —— 对比向量里 bias 本就抵消
            return F.linear(pl["wc"], cm.x_proj.weight)[:, None].expand(-1, D.N, -1)
        if self.variant == "ema_direct":
            Dm = state[1]                          # 绕过读出注意力
            return (Dm[:, 1] - Dm[:, 0])[:, None].expand(-1, D.N, -1)
        return cm.read(state, raw)                 # read_exact / full / full_mean

    def forward(self, doc, n_pairs, gen, shuf_gen=None, skip_first=True):
        cm = self.cm
        M = [cm.init_state(l) for l in range(D.L)]
        losses, accd = [], []
        for t, per in enumerate(doc):
            newM = []
            for l, pl in enumerate(per):
                raw = cm.raw(pl["k"], pl["v"])
                x, q = cm.feat(raw), cm.q_read(raw)
                r = self._read(M[l], raw, pl)
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
    ap.add_argument("--variant", default="full",
                    choices=("raw_oracle", "proj_oracle", "ema_direct",
                             "read_exact", "full", "full_mean"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
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
            if a.variant in Runner.ONE_ARM:
                # A/B 完全绕开记忆 ⇒ shuffled 与 stateful 逐位相同，跑第二臂没有信息。
                # 记 0，于是"差"这一列就是**相对 s0 的准确率增益**，即该档的上界；
                # 对它做同样的 t 检验（H0: 无增益）仍然有意义。
                res["shuffled"] = 0.0
                break
        d = res["stateful"] - res["shuffled"]
        diffs.append(d)
        print(f"{sd:>5}{res['stateful']:>+11.4f}{res['shuffled']:>+11.4f}"
              f"{d:>+10.4f}{al:>8.3f}", flush=True)
    fin = [d for d in diffs if d == d]
    if len(fin) >= 2:
        m, sp = statistics.mean(fin), statistics.stdev(fin)
        n = len(fin)
        t = m / (sp / n ** 0.5) if sp > 0 else float("inf")
        tc = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 2.0)
        sig = abs(t) > tc
        what = "相对 s0 的增益" if a.variant in Runner.ONE_ARM else "stateful−shuffled"
        print(f"\n{what}  mean {m:+.4f} ± {sp:.4f} (sd)   n={n}  "
              f"t={t:+.2f}  临界(df={n-1},双侧95%)={tc}")
        print("判读：" + ("**显著**，可以解读符号" if sig else
                        "**与 0 不可分**，不许解读符号"))


if __name__ == "__main__":
    sys.exit(main())
