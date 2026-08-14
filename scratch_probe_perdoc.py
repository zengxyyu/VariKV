#!/usr/bin/env python3
"""历史特征的最优权重在文档之间是否翻号 —— 决定池化线性探针的负结果能不能解读。

`scratch_probe_histinfo.py` 测出：加**可观测的存活集合特征**也掉 −0.012，和加驱逐侧
特征一样多，且留出损失涨 15–30%。对一个只多 3–4 个参数的线性模型这大得反常。
存活集合的特征是决策时合法可见、理论上（冗余/覆盖度）应该有用的东西，它却同样有害
⇒ 掉分很可能不是"历史没信息"，而是**池化的全局线性模型本身在加噪**。

机制假设：μ_R/μ_E 是**逐文档的方向**，`cos(k, μ_E)` 在不同文档里含义不同。一个全局
权重若与逐文档最优权重符号不一致，用到留出文档上会**整篇一致地错**。

做法：逐文档单独拟合（每篇内部再切 train/val 成对，避免自欺），报每个特征权重的
符号一致性与跨文档 sd。若历史特征的符号在文档间近似五五开 ⇒ 池化解释成立，
池化探针的负结果**不能**当作"历史无信息"的证据。
"""
import glob, os, sys, statistics as st
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as iu
_s = iu.spec_from_file_location("P", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "scratch_probe_histinfo.py"))
P = iu.module_from_spec(_s); _s.loader.exec_module(P)

traces = sys.argv[1] if len(sys.argv) > 1 else "scratch_ctrl_traces_sm_cont"
dev = "cuda" if torch.cuda.is_available() else "cpu"
files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      traces, "doc*.pt")))
NAMES = ["z", "margin", "|k|", "|v|", "s0",
         "cos(kv,μR)", "maxR", "top5R",
         "cos(kv,μE)", "cos(kv,μE−μR)", "maxE", "maxR−maxE"]
W = []
for f in files:
    Xc, Xs, Xh, U, G = P.build([f], 0.5, dev)
    X = torch.cat([Xc, Xs, Xh], 1)
    g = torch.Generator(device=dev).manual_seed(0)
    D, w = P._design(X, U, G, 400000, g, dev)
    th = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([th], max_iter=300, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10)
    def cl():
        opt.zero_grad(); l = (w * F.softplus(-(D @ th))).sum() / w.sum()
        l.backward(); return l
    opt.step(cl)
    W.append(th.detach().cpu())
    del Xc, Xs, Xh, U, G, X, D, w
W = torch.stack(W)                                   # [n_doc, F]
print(f"{traces}  逐文档拟合 {W.shape[0]} 篇\n")
print(f"{'特征':<14}{'权重均值':>10}{'跨文档 sd':>11}{'同号比例':>10}{'  判定'}")
for i, nm in enumerate(NAMES):
    v = W[:, i]
    frac = max(float((v > 0).float().mean()), float((v < 0).float().mean()))
    tag = "**符号不稳**" if frac < 0.75 else ("稳定" if frac > 0.9 else "偏稳")
    print(f"{nm:<14}{float(v.mean()):>+10.4f}{float(v.std()):>11.4f}"
          f"{frac:>10.1%}   {tag}")
cur = [max(float((W[:, i] > 0).float().mean()), float((W[:, i] < 0).float().mean()))
       for i in range(5)]
hist = [max(float((W[:, i] > 0).float().mean()), float((W[:, i] < 0).float().mean()))
        for i in range(5, 12)]
print(f"\n当前特征的平均同号比例 {st.mean(cur):.1%}   历史/集合特征 {st.mean(hist):.1%}")
print("若后者明显更低 ⇒ 逐文档最优权重方向不一致，全局线性模型必然被拖累，"
      "\n池化探针的负结果不能当作'历史无信息'的证据。")
