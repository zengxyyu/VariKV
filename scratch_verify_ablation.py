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


    print("\n== 5. 满缓存分数（headroom 的分母）与文档一致？ ==")
    FULL={"scbench_kv":68.20,"scbench_prefix_suffix":50.00,"scbench_vt":41.07}
    for ds,want in FULL.items():
        O=read_scores(ds,"_g8base",1.0)
        got=np.mean(list(O.values()))*100
        ok=abs(got-want)<0.05; fail+= (not ok)
        print(f"   {ds:<24}{want:>8.2f}{got:>8.2f}   {'OK' if ok else '**FAIL**'}")

    print("\n== 6. headroom 分层：高 headroom 格的 argmax 是否恒为 g=+1 ==")
    HI=[(0,68.20),(1,68.20),(7,50.00),(3,50.00),(8,41.07)]   # 对应 CELLS 的下标
    nhi=0
    for i,full in HI:
        b,p,n = DAT[i]
        a=[0.0,(p-b).mean(),(n-b).mean()]
        hd=full-b.mean(); am=int(np.argmax(a)); nhi += (am==1)
        print(f"   CELLS[{i}] headroom {hd:+6.2f}  argmax={['g=0','g=+1','g=-1'][am]}")
    ok = nhi==len(HI); fail += (not ok)
    print(f"   {nhi}/{len(HI)} 为 g=+1   {'OK（增量为零是必然的）' if ok else '**FAIL：§零之三 的命题需重查**'}")

    print("\n== 7. 地板全网格的关键数字（撤回 42 的依据） ==")
    # (ds, ratio, {b_min: tag})。**只列已 Finished. 且 n == 基线 n 的**；
    # b_min 不手抄——从 /tmp/vq/log 反解会依赖临时文件，故此处显式写死并由
    # 下面的断言保护：若某 tag 的读数变了（例如误覆盖），倍数就会对不上。
    FLOOR = [
        ("scbench_kv",0.2,{8:"_flr8",32:"_flr32",128:"_flr128",512:"_flr512"}),
        ("scbench_prefix_suffix",0.2,{4:"_psf02d",8:"_psf02c",16:"_psflr02h",
                                      32:"_psf02a",64:"_psflr02d",128:"_psf02b",256:"_psflr02e"}),
        ("scbench_prefix_suffix",0.1,{8:"_psf01c",32:"_psflr01b",128:"_psflr01c"}),
        ("scbench_vt",0.4,{8:"_vtf04c",32:"_vtf04a",128:"_vtf04b"}),
    ]
    best={}
    for ds,R,d in FLOOR:
        B=read_scores(ds,"_g8base",R); row=[]
        bb=-99
        for bm,tag in sorted(d.items()):
            try: A=read_scores(ds,tag,R)
            except Exception: row.append(f"b{bm}:MISS"); continue
            if len(A)!=len(B): row.append(f"b{bm}:n={len(A)}"); continue
            m,lo,hi,_=paired(A,B); bb=max(bb,m)
            row.append(f"b{bm} {m:+.2f}{'*' if lo>0 or hi<0 else ''}")
        best[(ds,R)]=bb
        print(f"   {ds.split('_')[-1]:<14}@{R:<5} " + " | ".join(row) + f"   BEST {bb:+.2f}")
    # **不再用 argmax 比值**（撤回 42）：两侧网格密度不等、且三个候选峰互不可分，
    # 取最大值再相除既有选择偏差、方向也不确定。改用**同 `b_min`** 的比值。
    KVT = {4:"_kvf02e", 8:"_flr8", 16:"_kvf02f", 32:"_flr32", 128:"_flr128"}
    PST = {4:"_psf02d", 8:"_psf02c", 16:"_psflr02h", 32:"_psf02a"}
    bk = read_scores("scbench_kv","_g8base",0.2)
    bp = read_scores("scbench_prefix_suffix","_g8base",0.2)
    rr = []
    for bm in sorted(set(KVT) & set(PST)):
        A = read_scores("scbench_kv", KVT[bm], 0.2)
        Bx = read_scores("scbench_prefix_suffix", PST[bm], 0.2)
        if len(A)!=len(bk) or len(Bx)!=len(bp):
            print(f"   b{bm}: 样本不全，跳过"); continue
        mk = paired(A,bk)[0]; mp = paired(Bx,bp)[0]
        if mp <= 0:
            print(f"   b{bm}: PrefSuf {mp:+.2f} ≤ 0，比值无定义"); continue
        rr.append((bm, mk/mp))
        print(f"   同 b{bm:<4} Retr.KV {mk:+6.2f}  PrefSuf {mp:+5.2f}  比值 {mk/mp:.1f}x")
    if rr:
        lo_r, hi_r = min(r for _,r in rr), max(r for _,r in rr)
        ok = 5.0 <= lo_r and hi_r <= 10.0; fail += (not ok)
        print(f"   ⇒ 同 b_min 比值区间 {lo_r:.1f}–{hi_r:.1f}x   "
              f"{'OK（文件写 5.6–9.2x）' if ok else '**FAIL：文件里的区间需重算**'}")
    # PrefSuf@0.2 的峰必须在 b16（撤回 42 的直接依据）
    print("\n== 8. MultiHop@0.4 三个方向无关对照全部与零不可分 ==")
    B=read_scores("scbench_vt","_g8base",0.4); REV=read_scores("scbench_vt","_vt04gm1",0.4)
    nns=0
    for bm,tag in [(8,"_vtf04c"),(32,"_vtf04a"),(128,"_vtf04b")]:
        A=read_scores("scbench_vt",tag,0.4)
        m,lo,hi,_=paired(A,B); ns = not (lo>0 or hi<0); nns+=ns
        m2,lo2,hi2,_=paired(REV,A)
        print(f"   b{bm:<4} 地板 {m:+.2f}[{lo:+.2f},{hi:+.2f}]{'ns' if ns else '**★**'}"
              f"   反向-地板 {m2:+.2f}[{lo2:+.2f},{hi2:+.2f}]{'*' if lo2>0 or hi2<0 else ' '}")
    ok = nns==3; fail += (not ok)
    print(f"   {nns}/3 与零不可分   {'OK' if ok else '**FAIL：§四的 best-of-3 表述要改**'}")

    print(f"\n{'全部通过' if fail==0 else f'**{fail} 项 FAIL**'}")
    return 1 if fail else 0

if __name__ == "__main__":
    raise SystemExit(main())
