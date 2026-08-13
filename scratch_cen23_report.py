import contextlib, io, json, os, sys
import numpy as np
import os, sys
ROOT = os.path.abspath(".")
sys.path.insert(0, os.path.join(ROOT, "external/FastKVzip/prefill"))
os.chdir(os.path.join(ROOT, "external/FastKVzip/prefill"))
from results.parse import parse_answer, evaluate_answer
MODEL="qwen2.5-7b-instruct-1m"; _M=contextlib.redirect_stdout(io.StringIO())
PANEL={"scbench_mf":"Math.Find","scbench_prefix_suffix":"Retr.Prefix-Suffix"}
def per_sample(data,suffix,ratio,task="qa"):
    with _M: ANSW,SUBT=parse_answer(data)
    pr,fu,i={},{},0
    while True:
        f=f"./results/{data}/{i}_{MODEL}_fastkvzip{suffix}/output-pair.json"
        if not os.path.exists(f): break
        dd=json.load(open(f)); p,q,ans=[],[],[]
        for fmt in [k for k in dd if k.startswith(task)]:
            for info,text in dd[fmt]:
                if abs(float(info[0])-ratio)<1e-9:
                    p.append(text["pruned"]); q.append(text["full__"]); ans.append(text["answer"])
        gold=ANSW[i] if ANSW else ans; sub=SUBT[i] if SUBT else None
        if p:
            with _M:
                pr[i]=float(np.mean(evaluate_answer(p,gold,data,task,subtask=sub)))
                fu[i]=float(np.mean(evaluate_answer(q,gold,data,task,subtask=sub)))
        i+=1
    return pr,fu
def boot(dif,n=10000,seed=0):
    r=np.random.default_rng(seed); dif=np.asarray(dif)
    s=dif[r.integers(0,len(dif),(n,len(dif)))].mean(1)
    return dif.mean(), float(np.quantile(s,.025)), float(np.quantile(s,.975))
print(f"\n{'='*104}\n【质心 @ ratio 0.3 / 0.2】—— 论文 Figure-11 合法区间内　★=95%CI 不含 0\n{'-'*104}")
print(f"{'论文面板':<20}{'ratio':>7}{'n':>5}{'满缓存':>8}{'基线':>8}{'headroom':>10}{'K=16 Δ':>20}{'K=1024 Δ':>20}")
for data in ["scbench_prefix_suffix","scbench_mf"]:
    for ratio in [0.3,0.2]:
        pb,fb=per_sample(data,"__full_chunk16k_w4096",ratio)
        cs=[per_sample(data,f"__c23{K}_{data}_chunk16k_w4096_cen{K}",ratio)[0] for K in (16,1024)]
        common=sorted(set(pb)&set(cs[0])&set(cs[1]))
        if not common: print(f"{PANEL[data]:<20}{ratio:>7} 无共同样本"); continue
        b=np.array([pb[i] for i in common])*100; full=np.mean([fb[i] for i in common])*100
        s=f"{PANEL[data]:<20}{ratio:>7.2f}{len(common):>5}{full:>8.2f}{b.mean():>8.2f}{full-b.mean():>+10.2f}"
        for pc in cs:
            c=np.array([pc[i] for i in common])*100; mm,lo,hi=boot(c-b)
            s+=f"{mm:>+8.2f}[{lo:>+5.1f},{hi:>+5.1f}]{'★' if (lo>0 or hi<0) else ' '}"
        print(s)
print("="*104)
