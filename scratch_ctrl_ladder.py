import torch, sys, importlib.util as iu
sys.path.insert(0,'external/FastKVzip/prefill')
spec=iu.spec_from_file_location('T','scratch_ctrl_train.py'); T=iu.module_from_spec(spec); spec.loader.exec_module(T)
from attention.control_memory import ControlMemory
L,H,d,n,NEAR = 2,2,128,192,64
DIR = torch.randn(d)
def make_doc(seed, level):
    g=torch.Generator().manual_seed(seed); chs=[]; k0=ret0=None
    for t in range(2):
        per=[]
        for l in range(L):
            k=torch.randn(H,n,d,generator=g); v=torch.randn(H,n,d,generator=g)
            s0=torch.randn(H,n,generator=g); ret=torch.rand(H,n,generator=g)<0.3
            if level=="local": U=torch.einsum('hnd,d->hn',k[:,:NEAR],DIR)
            elif t==0: U=torch.randn(H,NEAR,generator=g)
            elif level=="scalar":
                U=float((~ret0[l]).float().mean())*torch.einsum('hnd,d->hn',k[:,:NEAR],DIR)
            else:
                w=torch.stack([k0[l][h][~ret0[l][h]].mean(0) for h in range(H)])
                U=torch.einsum('hnd,hd->hn',k[:,:NEAR],w)
            per.append(dict(k=k.half(),v=v.half(),s0=s0,ret=ret,U=U,n_near=NEAR))
        if t==0: k0=[p['k'].float() for p in per]; ret0=[p['ret'] for p in per]
        chs.append(dict(layers=per))
    return dict(H=H,L=L,chunks=chs)
for level in ("local","scalar","direction"):
    tr=[make_doc(s,level) for s in range(12)]; va=[make_doc(100+s,level) for s in range(4)]
    out={}
    for mode in (("stateful","shuffled") if level!="local" else ("stateful",)):
        torch.manual_seed(7)
        cm=ControlMemory(d,L,H,n_slots=8,d_m=128,mode=mode)
        opt=torch.optim.AdamW(cm.parameters(),lr=3e-3,weight_decay=0.01)
        for ep in range(40):
            g=torch.Generator().manual_seed(ep)
            for doc in tr:
                loss,_,_=T.run_doc(cm,doc,'cpu',256,g)
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            gv=torch.Generator().manual_seed(999)
            out[mode]=sum(T.run_doc(cm,doc,'cpu',512,gv)[2] for doc in va)/len(va)
    extra = f"   shuffled {out['shuffled']:+.4f}   差 {out['stateful']-out['shuffled']:+.4f}" if 'shuffled' in out else ""
    print(f"{level:<10} stateful {out['stateful']:+.4f}{extra}", flush=True)
