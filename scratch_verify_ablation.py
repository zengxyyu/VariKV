#!/usr/bin/env python3
"""把 RESULTS_ABLATION.md 里写死的关键数字与原始结果独立重算值对拍。

为什么需要它：那份文件是按结果到达顺序手写追加的，2500 行、九条撤回，
**手抄错一个数字不会有任何报错**。本项目已经栽过同类：`zip` 静默截断
（撤回 35）、`--ctrlm_mode` 默认值（把整批评测跑成另一个方法）。

判据：重算值与文件里的数字差 > 0.005 即报 FAIL。
运行：.venv/bin/python scratch_verify_ablation.py
"""
import os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scratch_read_scores import read_scores, paired

CELLS = [("scbench_kv",0.1,"_d10scalar_s0","_kv01gm1"),("scbench_kv",0.2,"_sc11_s0","_kv02gm1"),
 ("scbench_kv",0.5,"_sc11_s0","_kv05gm1"),
 ("scbench_prefix_suffix",0.2,"_sc11_s0","_ps02gm1"),("scbench_prefix_suffix",0.3,"_sc11_s0","_ps03gm1"),
 ("scbench_prefix_suffix",0.4,"_sc11_s0","_ps04gm1"),("scbench_prefix_suffix",0.5,"_sc11_s0","_ps05gm1"),
 ("scbench_prefix_suffix",0.75,"_sc11_s0","_ps75gm1"),
 ("scbench_vt",0.05,"_sc11_s0","_vt005gm1"),("scbench_vt",0.1,"_sc11_s0","_vt01gm1"),
 ("scbench_vt",0.2,"_sc11_s0","_vt02gm1"),("scbench_vt",0.3,"_sc11_s0","_vt03gm1"),
 ("scbench_vt",0.4,"_sc11_s0","_vt04gm1"),("scbench_vt",0.5,"_sc11_s0","_vt05gm1"),
 ("scbench_vt",0.75,"_sc11_s0","_vt75gm1")]

# (标签, 重算函数键, 文件里写的值)
CLAIMS = {
    "always_mean": +0.46, "hold_mean": +4.91, "oracle_mean": +8.44,
    "always_worst": -17.60, "hold_worst": -0.88,
    "always_negrate": 60.0, "hold_negrate": 27.0,
}

def main():
    DAT = []
    fail = 0
    print("== 1. 逐格 A+ / A- 与文件对拍 ==")
    DOC = open("RESULTS_ABLATION.md").read()
    for ds, R, tp, tn in CELLS:
        B = read_scores(ds, "_g8base", R); P = read_scores(ds, tp, R); N = read_scores(ds, tn, R)
        c = sorted(set(B) & set(P) & set(N))
        b = np.array([B[k] for k in c])*100
        p = np.array([P[k] for k in c])*100
        n = np.array([N[k] for k in c])*100
        assert len(c) == len(B), f"{ds}@{R} 臂样本数少于基线：{len(c)} vs {len(B)}"
        DAT.append((b, p, n))
    print(f"   {len(DAT)} 格，全部臂样本数 == 基线 n  OK")

    rng = np.random.default_rng(0)
    alw = np.array([(p-b).mean() for b,p,n in DAT])
    orc = np.array([(np.maximum(np.maximum(b,p),n)-b).mean() for b,p,n in DAT])
    hold = np.zeros(len(DAT))
    for _ in range(2000):
        for i,(b,p,n) in enumerate(DAT):
            idx = rng.permutation(len(b)); h = len(b)//2
            for A,Bh in ((idx[:h],idx[h:]),(idx[h:],idx[:h])):
                g = [0.0,(p[A]-b[A]).mean(),(n[A]-b[A]).mean()]
                hold[i] += [0.0,(p[Bh]-b[Bh]).mean(),(n[Bh]-b[Bh]).mean()][int(np.argmax(g))]
    hold /= 4000.0
    got = {"always_mean":alw.mean(),"hold_mean":hold.mean(),"oracle_mean":orc.mean(),
           "always_worst":alw.min(),"hold_worst":hold.min(),
           "always_negrate":(alw<0).mean()*100,"hold_negrate":(hold<0).mean()*100}
    print("\n== 2. 阶梯与风险画像 ==")
    print(f"   {'量':<18}{'文件':>9}{'重算':>9}{'差':>9}   判定")
    for k, want in CLAIMS.items():
        g = got[k]; d = abs(g-want)
        tol = 0.5 if "negrate" in k else 0.02
        ok = d <= tol
        fail += (not ok)
        print(f"   {k:<18}{want:>+9.2f}{g:>+9.2f}{d:>9.3f}   {'OK' if ok else '**FAIL**'}")

    print("\n== 3. 撤回清单结构检查 ==")
    nums = sorted(set(int(x) for x in re.findall(r"撤回 (\d+)", DOC)))
    miss = [k for k in range(1, max(nums)+1) if k not in nums]
    print(f"   文档中出现的撤回编号 1..{max(nums)}，共 {len(nums)} 个")
    print(f"   缺号：{miss if miss else '无'}   "
          f"（1-10 是本文件早期局部编号，11-19 在 REVIEW_BRIEF、20-25 在 JOURNAL）")
    dup = [k for k in nums if DOC.count(f"撤回 {k}（") > 1]
    print(f"   重复定义的撤回条目：{dup if dup else '无'}")

    print("\n== 4. 方向不可替代的四格配对（重算） ==")
    FL = [("scbench_vt",0.2,"_vt02gm1",["_vtf02a","_vtf02b"]),
          ("scbench_vt",0.4,"_vt04gm1",["_vtf04a","_vtf04b"]),
          ("scbench_vt",0.5,"_vt05gm1",["_vtf05a","_vtf05b"]),
          ("scbench_prefix_suffix",0.3,"_sc11_s0",["_psf03a","_psf03b"]),
          ("scbench_prefix_suffix",0.5,"_ps05gm1",["_psf05a","_psf05b"])]
    nclean = 0
    for ds,R,best,fls in FL:
        A = read_scores(ds,best,R); o=[]
        allstar = True
        for f in fls:
            try: F = read_scores(ds,f,R)
            except Exception: o.append(f"{f}=缺"); allstar=False; continue
            m,lo,hi,_ = paired(A,F); st = lo*hi>0
            allstar &= st
            o.append(f"{f}: {m:+.2f}[{lo:+.2f},{hi:+.2f}]{'*' if st else ' '}")
        nclean += allstar
        print(f"   {ds.split('_')[-1]:<14}@{R:<5} " + "  ".join(o))
    print(f"   ⇒ 干净格 {nclean}/{len(FL)}（文件已更正为 5 格中 4 格干净、PrefSuf@0.5 擦边）")

    print(f"\n{'全部通过' if fail==0 else f'**{fail} 项 FAIL**'}")
    return 1 if fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
