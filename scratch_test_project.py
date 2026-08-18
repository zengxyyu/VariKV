#!/usr/bin/env python3
"""配额投影器的不变量单测 —— 在**真实 b0** 上跑，不是合成数据。零 GPU。

为什么必须有这个文件：本项目已经两次被离散化坑掉整批实验。
  · 第一次：`within` 用固定加性表 + **全局**配平 ⇒ 层总量漂移 938.9 槽/层，作业作废。
  · 第二次：`within` 改成层内配平后不变量满足了，但 `Δ_lh = c_l` 仍不是 no-op ——
    零配额头上的 `clamp(0,n)` 打破对称性，层常数分量借 clamp 泄漏进层内再分配。

结论：**连续空间的分解不蕴含离散实现的分解**，`Π(x+y) ≠ Π(x)+Π(y)`。
任何新的投影模式都必须先过这里再上 GPU。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
L, H = 28, 4


sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
import torch                                                        # noqa: E402
from attention.quota_project import project_quota as _pq            # noqa: E402
from attention.quota_project import rebalance as _rb                # noqa: E402


def project(b0, tab, n, mode):
    """**直接调用生产实现**，不做镜像复制。

    外部复核正确指出：镜像实现会让「生产改了、测试没改」或「两边带同一个错」时
    测试仍然全绿。本文件因此只负责构造用例与断言不变量，投影逻辑一律走
    `attention/quota_project.py`，即 `learned_ctrlcache.py` 运行时用的同一个函数。
    """
    return _pq(torch.as_tensor(b0, dtype=torch.float32),
               torch.as_tensor(tab, dtype=torch.float32),
               int(n), mode, L, H).numpy().astype(float)


def cases(f, k):
    d = [json.loads(x) for x in open(os.path.join(ROOT, f))]
    d = [r for r in d if isinstance(r["b_base"], list) and len(r["b_base"]) == 112]
    for r in d[:k]:
        b0 = np.array(r["b_base"], float)
        n = int(r["hi"] - r["lo"])
        if n <= 0 or b0.max() > n:
            n = int(max(b0.max(), 1))
        yield b0, n


def main():
    rng = np.random.default_rng(0)
    tab = np.load(os.path.join(ROOT, "scratch_quota_dbh_kv02.npy"))
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    C = list(cases("scratch_quota_r02.jsonl", K))
    print(f"真实 b0：{len(C)} 个 chunk（Retr.KV @ρ=0.2）\n")
    fails = []

    def chk(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            fails.append(name)

    # T1 Δ=0 ⇒ 恒等（三种 mode）
    for m in ("full", "within", "across"):
        bad = sum(not np.array_equal(project(b0, np.zeros(112), n, m), b0)
                  for b0, n in C)
        chk(f"T1 Δ=0 恒等 [{m}]", bad == 0, f"{bad}/{len(C)} 违反")

    # T2 预算守恒 + 非负 + 上界（随机大扰动）
    bad = 0
    for b0, n in C:
        for _ in range(3):
            t = rng.normal(0, 3000, 112); t -= t.mean()
            for m in ("full", "within", "across"):
                b = project(b0, t, n, m)
                if b.sum() != b0.sum() or b.min() < 0 or b.max() > n:
                    bad += 1
    chk("T2 Σb=B 且 0≤b≤n（随机 ±3000）", bad == 0, f"{bad} 违反")

    # T3 within 下逐层常数必须 no-op（旧写法在这里 20/20 失败）
    bad = 0
    for b0, n in C:
        c = rng.normal(0, 300, L); t = np.repeat(c, H); t -= t.mean()
        if not np.array_equal(project(b0, t, n, "within"), b0):
            bad += 1
    chk("T3 within(Δ_lh=c_l) = no-op", bad == 0, f"{bad}/{len(C)} 违反")

    # T4 across 下纯层内扰动必须 no-op
    bad = 0
    for b0, n in C:
        t = rng.normal(0, 300, (L, H)); t -= t.mean(1, keepdims=True)
        if not np.array_equal(project(b0, t.reshape(-1), n, "across"), b0):
            bad += 1
    chk("T4 across(Σ_h Δ_lh=0) = no-op", bad == 0, f"{bad}/{len(C)} 违反")

    # T5 within 的定义性不变量：逐层总量 = 基线
    bad = 0
    for b0, n in C:
        b = project(b0, tab, n, "within")
        bad += int((b.reshape(L, H).sum(1) != b0.reshape(L, H).sum(1)).sum())
    chk("T5 within 逐层总量 = 基线", bad == 0, f"{bad}/{L*len(C)} 层违反")

    # T6 across 的定义性不变量：逐层总量 = 两级投影的目标（构造性，非近似）
    bad = mx = 0
    for b0, n in C:
        b0m = b0.reshape(L, H); d = b0m.sum(1) + tab.reshape(L, H).sum(1)
        # 期望值同样用**生产实现**算，保证测的是同一个函数
        Bl = _rb(torch.as_tensor(np.clip(np.round(d), 0, H * n)).long(),
                 torch.as_tensor(d, dtype=torch.float32),
                 int(b0.sum()), H * n).numpy().astype(float)
        got = project(b0, tab, n, "across").reshape(L, H).sum(1)
        bad += int((got != Bl).sum()); mx = max(mx, int(np.abs(got - Bl).max()))
    chk("T6 across 逐层总量 = 目标", bad == 0, f"{bad}/{L*len(C)} 层违反，最大偏离 {mx}")

    # T7 极端饱和：b0 全 0 / 全 n，配上巨大 Δ
    bad = 0
    for _, n in C[:4]:
        for b0 in (np.zeros(112), np.full(112, float(n)),
                   np.where(np.arange(112) % 2, float(n), 0.0)):
            for t in (np.full(112, 99999.0) - 99999.0 / 112,
                      np.full(112, -99999.0) + 99999.0 / 112):
                for m in ("full", "within", "across"):
                    b = project(b0.copy(), t, n, m)
                    if b.sum() != b0.sum() or b.min() < 0 or b.max() > n:
                        bad += 1
    chk("T7 极端饱和下仍满足全部不变量", bad == 0, f"{bad} 违反")

    print(f"\n{'全部通过' if not fails else '失败: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
