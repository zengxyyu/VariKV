"""逐层饿死剖面（零 GPU，从 `b_base` 直接算）。**跨 panel 比较 L1 之前必须先跑。**
判据：只有当某层在该 panel 上也 100% 饿死、且 4 头 × b1 的剂量占比同量级时，
「只抬 L1」才是同一个干预（否则是第⑥类错：测错对象）。"""
import json, sys
import numpy as np
L, H = 28, 4
for f, name in [("scratch_qdump_kvf01c.jsonl", "Retr.KV@0.1"),
                ("scratch_qdump_psf01c.jsonl", "PrefSuf@0.1")]:
    B = []
    for ln in open(f):
        r = json.loads(ln)
        B.append(r["b_base"])
    B = np.array(B)                          # [chunk, L*H]
    star = (B == 0).mean(0).reshape(L, H)    # 逐 (层,头) 饿死率
    per_layer = star.mean(1)
    Btot = np.median(B.sum(1))
    full = [l for l in range(L) if per_layer[l] == 1.0]
    print(f"\n=== {name}  chunk={len(B)}  Btot 中位={Btot:.0f}  全局饿死率={star.mean():.3f}")
    print(f"  100% 饿死的层: {full}")
    print(f"  >90% 的层: {[l for l in range(L) if 0.9 <= per_layer[l] < 1.0]}")
    for l in [0, 1, 2, 6, 13]:
        lift4 = 4.0                          # 4 头 × b1
        print(f"   L{l:<3} 饿死率={per_layer[l]:.3f}  该层饿死头数="
              f"{int((star[l]==1.0).sum())}/4  剂量 4/Btot={100*lift4/Btot:.4f}%")
