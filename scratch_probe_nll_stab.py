#!/usr/bin/env python3
"""`U^NLL` 的有效性对照 —— 那个 Spearman≈0 到底是"靶子错了"还是"探针在测噪声"。

`scratch_probe_nll_oracle.py` 在两个 panel 上都测到 `corr(U^NLL, U^attn) ≈ 0`
（Retr.KV +0.032 p=0.42，Retr.MultiHop −0.013 p=0.75）。在把它当成"注意力靶子与
真实预测效用无关"之前，必须先排掉一个更平庸的解释：

    翻转 16.9 万条 KV 里的**一条**，NLL 的变化（中位 2.2e-3）可能根本不是那条
    KV 的"效用"，而是一次混沌扰动 —— 换个存活集合 S 再测一遍就完全变样。

若是后者，`U^NLL` 本身没有作为"效用"的可复现性，它和**任何**东西的相关都会是 0，
包括和它自己。那这个零结果就什么也没证明。

三个对照，成本从低到高：

A. **确定性**：同一掩码连算两次 NLL。前向是确定的 ⇒ 应当逐位相同。若不同，说明有
   非确定性内核，后面所有差分都要先减掉这个底噪。

B. **块级方向性**：从存活集合里按分数**最高 / 最低 / 随机**各去掉 B 条，比较 ΔNLL。
   若"去掉高分"明显比"去掉低分"更伤，说明本探针的 NLL 差分在**聚合层面**确实能分辨
   重要性 —— 那么单条效应小只是效应量小，不是测不准。若三者无差别，探针本身失效。

C. **跨存活集合的可复现性**（决定性的那个）：同一批候选，在两个只差 1% 随机翻转的
   存活集合 `S` 与 `S'` 上各算一次 `U^NLL`，看两者的 Spearman。

     高（≳0.5）⇒ `U^NLL` 是候选自身的稳定属性，Spearman≈0 是真结论：
                 **注意力靶子与预测效用无关**，换教师有据；
     低（≲0.2）⇒ 单条翻转是混沌，`U^NLL` 不能当效用标签用，oracle 结论作废，
                 且**任何**基于单条边际效用的教师（含 `U^setmarginal`）都可疑。

注意 C 的上界不是 1：`S` 与 `S'` 本来就不同，真实边际效用本就该有点差别。所以低相关
有歧义，高相关才是干净的结论 —— 这是这个对照的固有不对称，不要反过来读。
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
_P = os.path.join(ROOT, "external/FastKVzip/prefill")
sys.path.insert(0, _P)
os.chdir(_P)

from data import DataWrapper, load_dataset_all                  # noqa: E402
from model import ModelKVzip                                    # noqa: E402
from utils import set_gen_length                                # noqa: E402


@torch.inference_mode()
def nll(model, ids, n_ans, kv):
    p = model._prob(ids, kv)[-n_ans - 1:-1].float()
    lab = ids[0, -n_ans:]
    return float(-p.gather(1, lab[:, None]).clamp_min(1e-12).log().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--n_cand", type=int, default=24)
    ap.add_argument("--block", type=int, default=256, help="对照 B 的块大小")
    ap.add_argument("--perturb", type=float, default=0.01,
                    help="对照 C 里 S' 相对 S 的随机翻转比例")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()

    m = ModelKVzip(a.model, "retain", "fastkvzip")
    ds = DataWrapper(a.data, load_dataset_all(a.data, m.tokenizer), m)
    set_gen_length(a.data, m)
    g = torch.Generator(device="cpu").manual_seed(0)
    det, blocks, pairs = [], [], []

    for si in range(min(a.num, len(ds))):
        kv_f = ds.prefill_context(si, do_score=False)
        inputs, _ = ds.generate_answer(si, kv_f, prob=False)
        task = "qa" if "qa" in inputs else list(inputs.keys())[0]
        ids = torch.cat([inputs[task][k] for k in ("q", "a")], dim=-1)
        n_ans = len(inputs[task]["a"][0])
        del kv_f
        torch.cuda.empty_cache()

        kv = ds.prefill_context(si, prefill_chunk=a.chunk, window_size=a.window,
                                chunk_ratio=a.ratio, level="pair")
        kv.valid = kv.valid.clone()      # prefill 在 inference_mode 下建的张量不可就地改
        V, sink = kv.valid, kv.sink
        base = nll(m, ids, n_ans, kv)

        # ---- A 确定性 ------------------------------------------------------
        det.append(abs(nll(m, ids, n_ans, kv) - base))

        sc = torch.stack(kv.score, 0)[:, 0]
        s_ev = sc[..., sink:sink + V.shape[-1]].float()
        L_, H_, N_ = s_ev.shape
        keep = V.reshape(-1)
        s_flat = s_ev.reshape(-1)

        # ---- B 块级方向性：只在**存活**条目里挑，去掉 B 条 -----------------
        ki = keep.nonzero(as_tuple=True)[0]
        ks = s_flat[ki]
        nb = min(a.block, len(ki) // 4)
        order = ks.argsort(descending=True)
        sel = {"top": ki[order[:nb]], "bot": ki[order[-nb:]],
               "rand": ki[torch.randperm(len(ki), generator=g)[:nb]]}
        row = {}
        for nm, idx in sel.items():
            V.view(-1)[idx] = False
            row[nm] = nll(m, ids, n_ans, kv) - base
            V.view(-1)[idx] = True
        blocks.append(row)

        # ---- C 跨存活集合的可复现性 ---------------------------------------
        # 候选取全局阈值附近（与 oracle 同口径：离阈值远的怎么改分也翻不了）
        tau = s_flat.sort(descending=True).values[
            max(int(s_flat.numel() * a.ratio) - 1, 0)]
        cand = (s_flat - tau).abs().argsort()[:a.n_cand]
        # S' = S 随机翻转 1%；**候选本身必须排除在扰动之外**，否则测的是
        # "翻转它两次"而不是"在不同背景下翻转它"
        pool = torch.ones(len(s_flat), dtype=torch.bool)
        pool[cand] = False
        pidx = pool.nonzero(as_tuple=True)[0]
        pk = pidx[torch.randperm(len(pidx), generator=g)[:int(len(pidx) * a.perturb)]]

        def u_all():
            b = nll(m, ids, n_ans, kv)
            out = []
            for c in cand.tolist():
                kept = bool(V.view(-1)[c])
                V.view(-1)[c] = not kept
                n2 = nll(m, ids, n_ans, kv)
                V.view(-1)[c] = kept
                out.append((n2 - b) if kept else (b - n2))
            return np.array(out)

        uS = u_all()
        V.view(-1)[pk] = ~V.view(-1)[pk]
        uS2 = u_all()
        V.view(-1)[pk] = ~V.view(-1)[pk]
        pairs.append((uS, uS2))
        from scipy.stats import spearmanr
        print(f"  样本 {si}: base {base:.4f}  确定性 |Δ| {det[-1]:.2e}  "
              f"块 top {row['top']:+.4f} / rand {row['rand']:+.4f} / bot {row['bot']:+.4f}  "
              f"C-corr {spearmanr(uS, uS2).statistic:+.3f}", flush=True)
        del kv
        torch.cuda.empty_cache()

    from scipy.stats import spearmanr
    np.save(os.path.join(ROOT, f"scratch_nllstab_{a.data}.npy"),
            np.array([np.concatenate([p[0], p[1]]) for p in pairs]))
    print(f"\n=== {a.data} @ ratio {a.ratio}　{len(pairs)} 篇 ===")
    print(f"A 确定性：同掩码两次 NLL 的 |Δ| 最大 {max(det):.3e}  "
          f"（应为 0；非 0 则是底噪，需与 |U^NLL| 中位 2.2e-3 比）")
    for nm in ("top", "bot", "rand"):
        v = np.array([b[nm] for b in blocks])
        print(f"B 去掉 {a.block} 条 {nm:<5} 的 ΔNLL： {v.mean():+.4f} ± {v.std():.4f}")
    print("  判读：top ≫ rand ≳ bot ⇒ 探针在聚合层面能分辨重要性，单条效应小只是效应量小")
    allc = [spearmanr(p[0], p[1]).statistic for p in pairs]
    cat = (np.concatenate([p[0] for p in pairs]),
           np.concatenate([p[1] for p in pairs]))
    print(f"C 跨存活集合 Spearman(U^NLL(S), U^NLL(S'))： 合并 "
          f"{spearmanr(*cat).statistic:+.4f}   逐样本中位 {np.median(allc):+.4f}"
          f"  （{len(allc)} 篇：" + " ".join(f"{x:+.2f}" for x in allc) + "）")
    print("  ≳0.5 ⇒ U^NLL 是候选的稳定属性，oracle 的零相关是真结论；"
          "≲0.2 ⇒ 单条翻转是混沌，oracle 与任何单条边际效用教师都作废")


if __name__ == "__main__":
    raise SystemExit(main())
