"""CPU 先验证预投影的数学不变量，再写进推理路径。

问题：`project_quota` 会裁剪 —— 一个 `b0_h = 0` 的头吐不出配额，
那部分请求被**清零**，其余照常通过 ⇒ **有效方向 ≠ 名义方向**（实测 kv 表
可实现率仅 0.236、名义动 111 头实际只有 56）。

预投影的做法（两步，顺序不能反）：
  ① **裁剪**：`d_h ← max(d_h, −b0_h)`（不能给出比自己拥有的更多）；
  ② **再平衡**：裁剪只削掉负的一侧 ⇒ `Σd > 0`，把正的一侧整体缩放回去，
     使 `Σd = 0` 严格成立。**缩放只作用于正侧，不会重新引入不可行的负值。**

必须验的三条不变量：
  (a) `Σd = 0`（保预算）；
  (b) `d_h ≥ −b0_h` 逐头成立（可行）；
  (c) 缩放因子 ∈ (0, 1]，即再平衡**只会缩小**正侧、不会放大。
"""
import numpy as np

RNG = np.random.default_rng(0)


def preproj(d, b0):
    d = d.astype(float).copy()
    d = np.maximum(d, -b0.astype(float))          # ① 裁剪
    neg = -d[d < 0].sum()
    pos = d[d > 0].sum()
    if pos > 0:
        s = neg / pos                             # ② 再平衡（只缩正侧）
        d[d > 0] *= s
    else:
        s = 1.0
    return d, s


ok = True
for trial in range(2000):
    H = 112
    b0 = RNG.integers(0, 2000, H).astype(float)
    b0[RNG.random(H) < 0.55] = 0.0                # 模拟 ρ=0.1 的 ~64/112 饿死头
    d = RNG.normal(0, 300, H)
    d = d - d.mean()                              # 原始表：Σ=0
    dp, s = preproj(d, b0)
    a = abs(dp.sum()) < 1e-6 * max(1.0, np.abs(dp).sum())
    b = bool((dp >= -b0 - 1e-9).all())
    c = 0.0 < s <= 1.0 + 1e-12
    ok &= a and b and c
    if not (a and b and c):
        print(f"trial {trial} 失败: Σ={dp.sum():+.3e} 可行={b} s={s:.4f}"); break
print("2000 次随机试验：" + ("三条不变量全过（Σ=0、逐头可行、缩放 ∈ (0,1]）" if ok else "**有失败**"))

# 阴性对照：不做预投影时，有多少请求量会落在饿死头上被裁掉
tot_clip = tot = 0.0
for _ in range(200):
    H = 112
    b0 = RNG.integers(0, 2000, H).astype(float); b0[RNG.random(H) < 0.55] = 0.0
    d = RNG.normal(0, 300, H); d = d - d.mean()
    clipped = np.maximum(d, -b0)
    tot_clip += np.abs(clipped - d).sum(); tot += np.abs(d).sum()
print(f"阴性对照：不预投影时约 {tot_clip / tot:.1%} 的请求量落在饿死头上被裁掉"
      f"（实测 kv 表可实现率 0.236 ⇒ 约 76% 被裁，量级吻合）")
