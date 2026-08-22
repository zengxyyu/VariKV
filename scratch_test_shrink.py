#!/usr/bin/env python3
"""`shrink`（向均匀收缩）mode 的 CPU 冒烟测试：六条不变量 + 三个阴性对照。

**为什么要有这个文件**：`shrink` 是地板的**匹配搬动量对照**。若它自己
预算不守恒、或 γ 与实际搬动量对不上，那么「地板 vs 收缩」的比较就同时
变了两个变量，整个对照作废。**本项目规矩：新 mode 上 GPU 前必过 CPU 冒烟。**

    .venv/bin/python scratch_test_shrink.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "external/FastKVzip/prefill"))
from attention.quota_project import project_quota   # noqa: E402


def mk(L=28, H=4, n=1500, seed=0):
    """造一个形状像真实 `b0` 的配额向量：重尾 + 大量零配额头。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(L * H, generator=g) ** 3
    x[torch.rand(L * H, generator=g) < 0.55] = 0.0          # 55% 饿死
    b0 = (x / x.sum() * (0.1 * n * L * H)).round().clamp(0, n)
    return b0, n, L, H


def run(b0, n, L, H, **env):
    for k in ("VARIKV_SHRINK_GAMMA", "VARIKV_SHRINK_MB"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    d = torch.zeros_like(b0)
    out = project_quota(b0.clone(), d, n, "shrink", L, H)
    for k in env:
        os.environ.pop(k, None)
    return out


def mb(bt, b0):
    return float((bt.float() - b0.float()).abs().sum()) / 2.0


def main():
    ok = True
    b0, n, L, H = mk()
    B = int(b0.sum())
    unif = B / (L * H)
    half = float((b0.float() - unif).abs().sum()) / 2.0
    print(f"夹具：L={L} H={H} n={n}  Btot={B}  饿死 {int((b0==0).sum())}/{L*H}"
          f"  unif={unif:.2f}  γ=1 时搬动量={half:.1f}\n")

    # ① 预算守恒（每个 γ 都要）
    for g in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        bt = run(b0, n, L, H, VARIKV_SHRINK_GAMMA=g)
        s = int(bt.sum())
        assert s == B, f"γ={g} 预算不守恒 {s} != {B}"
    print("① 预算守恒（6 个 γ）                      PASS")

    # ② γ=0 必须逐位等于 b0
    bt0 = run(b0, n, L, H, VARIKV_SHRINK_GAMMA=0.0)
    assert torch.equal(bt0, b0.long()), "γ=0 不是恒等"
    print("② γ=0 逐位恒等                            PASS")

    # ③ γ=1 必须收敛到均匀（整数取整内）
    bt1 = run(b0, n, L, H, VARIKV_SHRINK_GAMMA=1.0)
    dev1 = float((bt1.float() - unif).abs().max())
    assert dev1 <= 1.0, f"γ=1 偏离均匀 {dev1}"
    print(f"③ γ=1 收敛到均匀（max 偏离 {dev1:.2f} ≤ 1）    PASS")

    # ④ 搬动量必须线性于 γ
    bad = []
    for g in (0.1, 0.25, 0.5, 0.9):
        bt = run(b0, n, L, H, VARIKV_SHRINK_GAMMA=g)
        got, exp = mb(bt, b0), g * half
        if abs(got - exp) > max(2.0, 0.02 * exp):
            bad.append((g, got, exp))
    assert not bad, f"搬动量非线性：{bad}"
    print("④ 搬动量 = γ·half_l1（4 个 γ，误差 ≤2%）   PASS")

    # ⑤ MB 参数化。**判据本身写错了两次，两次都是这个测试抓的（记录在案）**：
    #   v1「相对误差 ≤2%」—— MB=200 给 195 就 FAIL，而那是整数取整不是 bug：
    #      γ=200/12200≈0.016 时 `|dev_h|` 小的头 `γ·dev_h < 0.5` 舍成 0。
    #   v2「只会欠交付」—— 也错：50→51、1000→1002 **超**交付。原因是
    #      `rebalance` 为严格守恒预算做的再分配**本身会增加搬动量**。
    # ⇒ 取整是**双侧**的，正确判据只有一条：`|实际 − 请求| ≤ 取整上界`。
    # ⚠ **由此得到一条使用规则**：做「地板 vs 收缩」的匹配搬动量对照时，
    # 必须按 `[shrink]`/`[floor]` 日志里的**实际** L1/2 匹配，不能按请求值。
    bad, rows = [], []
    bound = 0.5 * L * H / 2.0
    for target in (50.0, 200.0, 1000.0, 5000.0):
        if target > half:
            continue
        bt = run(b0, n, L, H, VARIKV_SHRINK_MB=target)
        got = mb(bt, b0)
        rows.append((target, got, target - got))
        if abs(target - got) > bound:
            bad.append(("偏差超出取整上界", target, got, bound))
    assert not bad, f"MB 反解违反判据：{bad}"
    print(f"⑤ MB 反解：|实际 − 请求| ≤ 取整上界 {bound:.0f}       PASS")
    for t_, g_, d_ in rows:
        print(f"     请求 {t_:7.0f} → 实际 {g_:7.0f}"
              f"（差 {-d_:+5.1f} = {-100*d_/t_:+.1f}%）")

    # ⑥ 饱和：请求超过上限 ⇒ 截到 γ=1，且实际 = half
    bt = run(b0, n, L, H, VARIKV_SHRINK_MB=half * 10)
    got = mb(bt, b0)
    assert abs(got - half) <= max(2.0, 0.02 * half), f"饱和后 {got} != {half}"
    assert abs(project_quota._sh_gam - 1.0) < 1e-9, "饱和时 γ 应为 1"
    print(f"⑥ 饱和截断可见（请求 {half*10:.0f} → 实际 {got:.0f}） PASS")

    # --- 阴性对照：判据本身也要能拒 ---
    def must_raise(desc, **env):
        nonlocal ok
        try:
            run(b0, n, L, H, **env)
        except AssertionError:
            print(f"   阴性 {desc:28s} 正确拒绝  PASS")
            return
        except Exception as e:
            print(f"   阴性 {desc:28s} 抛了别的：{type(e).__name__}  FAIL")
            ok = False
            return
        print(f"   阴性 {desc:28s} **没拒绝**  FAIL")
        ok = False

    print("\n阴性对照：")
    must_raise("两个参数都给", VARIKV_SHRINK_GAMMA=0.5, VARIKV_SHRINK_MB=100)
    must_raise("γ > 1", VARIKV_SHRINK_GAMMA=1.5)
    must_raise("MB 为负", VARIKV_SHRINK_MB=-5)
    # 一个都不给
    for k in ("VARIKV_SHRINK_GAMMA", "VARIKV_SHRINK_MB"):
        os.environ.pop(k, None)
    try:
        project_quota(b0.clone(), torch.zeros_like(b0), n, "shrink", L, H)
        print("   阴性 一个都不给                  **没拒绝**  FAIL")
        ok = False
    except AssertionError:
        print("   阴性 一个都不给                  正确拒绝  PASS")

    # --- 对拍：shrink 与 floor 在同一 b0 上的搬动量可匹配 ---
    print("\n⑦ 与地板的搬动量匹配演练：")
    os.environ["VARIKV_QUOTA_FLOOR"] = "8"
    bf = project_quota(b0.clone(), torch.zeros_like(b0), n, "floor", L, H)
    os.environ.pop("VARIKV_QUOTA_FLOOR")
    mbf = mb(bf, b0)
    bs = run(b0, n, L, H, VARIKV_SHRINK_MB=mbf)
    mbs = mb(bs, b0)
    print(f"   地板 b8 搬动量 {mbf:.1f} → 收缩匹配到 {mbs:.1f}"
          f"（γ={project_quota._sh_gam:.4f}）"
          f"  {'PASS' if abs(mbs-mbf) <= max(2.0, 0.02*mbf) else 'FAIL'}")
    # **判词必须由数字生成。** 首版这里写死了「动的头完全不同」，而实测交集
    # 104/106 —— 两者动的是**几乎同一批头**。真正要报的是**方向与分配**：
    df = (bf.float() - b0.float())
    ds = (bs.float() - b0.float())
    sf, ss = torch.sign(df), torch.sign(ds)
    both = (sf != 0) & (ss != 0)
    agree = int((sf[both] == ss[both]).sum())
    cos = float((df @ ds) / (df.norm() * ds.norm() + 1e-30))
    print(f"   动的头：地板 {int((df!=0).sum())}、收缩 {int((ds!=0).sum())}、"
          f"交集 {int(both.sum())}，其中**同向 {agree}/{int(both.sum())}**")
    print(f"   cos(Δb_floor, Δb_shrink) = {cos:+.4f}")
    print(f"   ⇒ 在这个合成夹具上两条路径**高度重合**（因为 55% 的头低于均匀，"
          f"收缩也把它们全抬起来）。**真实 b0 上未必如此**，"
          f"上 GPU 后要按 `[shrink]`/`[floor]` 日志逐 chunk 复核这两个量；"
          f"若真实数据上 cos 也接近 1，则这个对照分辨不出两条假说，**必须改设计**。")
    print("\n" + ("ALL PASS" if ok else "有 FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
