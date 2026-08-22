# -*- coding: utf-8 -*-
"""ProMeta manifest 的**独立审计** —— 只读文件重算，schema 由文件本身决定。

    .venv/bin/python -B scratch_prometa_audit_ds.py [manifest.jsonl]

**为什么单独一份**：`prometa/dataset.py` 自己也有断言，但那是**构建期**的，
与构建代码共享同一套假设 —— 构建逻辑错了，它的断言会跟着一起错（第①类错：
判据本身会错）。这份只吃 jsonl，连字段名都是从文件里读出来的。

它抓到过的真问题：`--span qa` 会让 synth（唯一带答案的 kind）与其余 kind
用两套标签定义；训练侧真实的「不同池化输入」只有 320 个而非账面 480 个。
"""
import json, hashlib, collections, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_av = [x for x in sys.argv[1:] if not x.startswith("--")]
P = _av[0] if _av else "prometa_data/manifest_v1_ss.jsonl"
SPAN = "qa" if "--span=qa" in sys.argv else "q"   # 默认按训练脚本的默认值 q 判
R = [json.loads(l) for l in open(P)]
bad = []
def chk(n, ok, d):
    print(f"{'✓' if ok else '✗'} {n}：{d}");  bad.append(n) if not ok else None

# 字段按 kind 分：`ss_ok` 只在 selfstudy、`meta` 只在 synth —— 这是设计不是缺失。
need = {"id","ctx","futures","split","kind","band","source_ids","n_ctx","built"}
chk("① 公共字段完整 / 条数", all(need <= set(r) for r in R) and len(R)==200,
    f"{len(R)} 条；公共字段 {sorted(need)}；selfstudy 另有 ss_ok、synth 另有 meta")

# `q_text`/`grounded` 只有 LLM 生成的 selfstudy 问句才有；synth/continuation 直接给 token id。
badf = [r["id"] for r in R for f in r["futures"]
        if not {"kind","q","a","needs"} <= set(f) or len(f["q"]) < 5
        or (r["kind"]=="selfstudy" and not f.get("q_text","").strip())]
mc = collections.Counter((r["kind"], len(r["futures"])) for r in R)
chk("② futures 结构 / 非空", not badf,
    f"kind×M {dict(sorted(mc.items()))}（continuation 只有 1 个真实未来，设计如此）；坏 future {len(badf)}")

# n_ctx 与 ctx 实长一致（构建期字段不可信，重算）
mm = [r["id"] for r in R if r["n_ctx"] != len(r["ctx"])]
chk("③ n_ctx 与 ctx 实长一致", not mm, f"不一致 {len(mm)} 条")

# ④ **源文档级**划分互不相交（从 source_ids 重算）
S = collections.defaultdict(set)
for r in R: S[r["split"]].update(r["source_ids"])
it = {f"{a}∩{b}": len(S[a]&S[b]) for a,b in [("train","val"),("train","test"),("val","test")]}
chk("④ 源文档级互不相交", all(v==0 for v in it.values()),
    f"源文档数 { {k:len(v) for k,v in sorted(S.items())} }；交集 {it}")

h  = [hashlib.sha1(str(r["ctx"]).encode()).hexdigest() for r in R]
hp = [hashlib.sha1(str(r["ctx"][:4000]).encode()).hexdigest() for r in R]
chk("⑤ 上下文互不重复（全串 / 前 4000）", len(set(h))==200 and len(set(hp))==200,
    f"全串唯一 {len(set(h))}/200、前缀唯一 {len(set(hp))}/200")

B = {"8-16k":(8000,16000),"16-32k":(16000,32000),"32-64k":(32000,64000),"64-128k":(64000,131072)}
off = [(r["id"],r["band"],r["n_ctx"]) for r in R if not B[r["band"]][0]*.95 <= r["n_ctx"] <= B[r["band"]][1]*1.05]
chk("⑥ 长度落在 band 内", not off, f"越界 {len(off)}" + (f" 例{off[:3]}" if off else ""))

V=152064
ill=[r["id"] for r in R if min(r["ctx"])<0 or max(r["ctx"])>=V
     or any(min(f["q"])<0 or max(f["q"])>=V for f in r["futures"])]
chk("⑦ token id 合法（ctx 与 q）", not ill, f"越界 {len(ill)} 条（词表 {V}）")

from prometa.teacher import chunk_ranges
SYS, CH = 28, 16000
nz=[]; tot=0; per=[]
for r in R:
    _all, us = chunk_ranges(r["n_ctx"]+SYS, SYS, CH, 4096)
    tot+=len(us); per.append(len(us))
    if not us: nz.append(r["id"])
chk("⑧ 每条都有可用 chunk", not nz,
    f"零 chunk {len(nz)} 条；总 {tot}，均 {tot/len(R):.1f}/条，min {min(per)} max {max(per)}")

# ⑨ a 与 needs 是否真的全空（决定 --span 只能取 q）
na = sum(1 for r in R for f in r["futures"] if f["a"])
nn = sum(1 for r in R for f in r["futures"] if f["needs"])
nq = sum(len(r["futures"]) for r in R)
byk = collections.Counter(f["kind"] for r in R for f in r["futures"])
print(f"  {nq} 个 future：a 非空 {na}、needs 非空 {nn}；kind 分布 {dict(sorted(byk.items()))}")
# ⚠ **判据本身错过一次**（2026-08-22 外部复核指出，采纳）：首版无条件要求
# `a` 全空或全满，于是把一个**对 `--span q` 完全健康**的数据集判成失败并 exit 1。
# `a` 混合只在 `--span qa` 下才是缺陷（那时标签定义会随 kind 变）。
# 判据必须知道 span 才能判 —— 不知道就只能报告，不能定罪。第①类错。
chk(f"⑨ span 自洽（当前按 --span={SPAN} 判）", SPAN == "q" or na == 0 or na == nq,
    f"a 非空 {na}/{nq}（只有 synth 有）；span=q 时只取问句 ⇒ 无影响；"
    f"span=qa 时标签定义会随 kind 变 ⇒ 训练脚本已加硬闸拒绝")

# ⑩ grounded 与 ss_ok
g=[f["grounded"] for r in R for f in r["futures"] if r["kind"]=="selfstudy"]
so=collections.Counter(r.get("ss_ok") for r in R if r["kind"]=="selfstudy")
kk=collections.Counter(r["kind"] for r in R)
chk("⑩ selfstudy 全部生成成功", so.get(False,0)==0 and so.get(True,0)==kk["selfstudy"],
    f"selfstudy ss_ok {dict(so)} / 共 {kk['selfstudy']} 条；kind {dict(kk)}；"
    f"grounded 均 {sum(g)/len(g):.3f} min {min(g):.2f} <0.15 的 {sum(1 for x in g if x<0.15)} 个")

print(f"  split×kind {dict(sorted(collections.Counter((r['split'],r['kind']) for r in R).items()))}")
print(f"  split×band {dict(sorted(collections.Counter((r['split'],r['band']) for r in R).items()))}")
n_tr=sum(1 for r in R if r["split"]=="train")
NCH=3
used=sum(min(NCH, len(chunk_ranges(r["n_ctx"]+SYS, SYS, CH, 4096)[1])) for r in R if r["split"]=="train")
print(f"\n  train {n_tr} 篇，n_chunk={NCH} ⇒ **{used} 个不同池化输入**（旧 8×5=40 ⇒ {used/40:.1f}×）")
print(f"  记忆化判据：上下文路径 279,616 参数 ÷ 每输入 320 个输出 = **{279616//320} 个输入**才饱和；"
      f"当前 {used} < {279616//320} ⇒ **容量上仍足以背下查找表** ⇒ 致盲/打乱对照仍是必需的")
print(f"  token：全 {sum(r['n_ctx'] for r in R):,}，train {sum(r['n_ctx'] for r in R if r['split']=='train'):,}")
print(f"\nsha256 {hashlib.sha256(open(P,'rb').read()).hexdigest()}")
print("**审计全过**" if not bad else f"**{len(bad)} 条不过：{bad}**")
sys.exit(1 if bad else 0)
