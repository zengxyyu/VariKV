#!/usr/bin/env python3
"""`U^NLL` oracle —— 教师靶子到底对不对，不训练任何模型就能判。

**为什么必须先做这个，而不是直接换教师。** 项目已经测出

    Retr.MultiHop：残差**显著更忠实**于满缓存（KL 0.2575→0.1779, t=−3.49），
                   任务分数却掉 9.96 ⇒ **保真度 ≠ 任务效用**

而现有两个教师靶子（`U^full` 与 `U^setmarginal`）的目标函数都是
`F(S) = −‖W_O(o_full − o_S)‖²` —— 最优解都在 `S = 满缓存`。所以它们在 MultiHop 上
教的方向被证明是错的。自然的修法是换成**未来预测损失**

    L(S) = −Σ_j log p(y_j | S)          U^NLL_c = L(S∖{c}) − L(S∪{c})

它的最优解**不必是满缓存**：删掉干扰项若让预测更准，教师就会奖励删除。

但换教师要重跑教师 + 训练 + 下游，代价很大。**先用几十个候选 brute-force 验证靶子
是否真的错位**：若 `U^attn` 与 `U^NLL` 在 Retr.KV 上强相关、在 MultiHop 上弱相关甚至
反号，那 −9.96 的因果链就闭合了，换教师才有依据；若两者到处都强相关，说明靶子不是
病根，换了也白换。

--------------------------------------------------------------------------------
做法

对每个样本：满缓存预填一次拿 `o_full` 与 teacher-forced 的答案串；压缩预填一次拿
存活掩码 `S`。然后在**全局阈值附近**挑 `n_cand` 个 `(层, kv头, token)` 三元组，
对每个候选各算两个量：

  `U^NLL`  = L(S∖{c}) − L(S∪{c})   翻转 `kv.valid[l,h,i]` 后重跑一次答案前向。
                                   **暴力**，每个候选一次前向，这是全部成本所在。
  `U^attn` = err(S∖{c}) − err(S∪{c})，`err(o)=‖W_O(o_full−o_S)‖²`
                                   softmax 的秩一更新，`O(d)`，与教师 set_marginal 同式。

两者符号约定一致：都是"**c 缺席的代价**"，正值 = 该留。

**为什么只挑阈值附近**：离阈值远的候选无论怎么改分都翻不了，对它们算 U 是在测一个
恒真的排序。手工版实测 β=0.5 只翻转 0.895% 的条目。
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
    """teacher-forced 答案位置上的平均 NLL。`ids` 由满缓存那遍构造、各臂共用。"""
    p = model._prob(ids, kv)[-n_ans - 1:-1].float()
    lab = ids[0, -n_ans:]
    return float(-p.gather(1, lab[:, None]).clamp_min(1e-12).log().mean())


@torch.inference_mode()
def attn_marginal(model, kv, qcap, layers, cand, keep, n_ctx, sink):
    """`U^attn`：条件于存活集合 S 的边际效用，与 scratch_ctrl_teacher 的
    `utility_setmarginal` 同一公式（秩一更新，不重算 softmax）。"""
    d = model.config.hidden_size // model.config.num_attention_heads
    H = model.config.num_key_value_heads
    out = {}
    for (l, h, i) in cand:
        Aq = qcap[l][0].float() * (d ** -0.5)          # [HQ,T,d]
        HQ, T, _ = Aq.shape
        G = HQ // H
        Aq = Aq.view(H, G, T, d)[h]                    # [G,T,d]
        K = kv.key_cache[l][0][h, :n_ctx].float()
        V = kv.value_cache[l][0][h, :n_ctx].float()
        WO = model.model.model.layers[l].self_attn.o_proj.weight.detach().float()
        W = WO[:, h * G * d:(h + 1) * G * d]
        Gram = W.T @ W
        a = torch.einsum("gtd,nd->gtn", Aq, K)
        e = (a - a.amax(-1, keepdim=True)).exp()
        o_full = torch.einsum("gtn,nd->gtd", e, V) / e.sum(-1, keepdim=True)
        # **off-by-sink**：`kv.valid` 只覆盖可驱逐区 `[sink, sink+ctx_len)`，
        # 而 `key_cache` / `e` 是绝对位置、长度 n_ctx = sink + ctx_len。
        # 前 sink 个是 attention sink，**永远保留**，必须补成 True 再对齐。
        # 冒烟直接崩在这里（169063 vs 169035，差 28 = sink），是好事：
        # 若形状恰好能广播就会静默算错。
        mv = keep[l][h].to(K.device)
        m = torch.cat([torch.ones(sink, dtype=mv.dtype, device=K.device), mv])
        assert m.numel() == n_ctx, (m.numel(), n_ctx)
        ia = i + sink                                  # 候选下标也要转成绝对位置
        eS = e * m[None, None, :]
        ZS = eS.sum(-1, keepdim=True).clamp_min(1e-30)
        NS = torch.einsum("gtn,nd->gtd", eS, V)

        def err(o):
            z = (o_full - o).permute(1, 0, 2).reshape(T, G * d)
            return ((z @ Gram) * z).sum(-1).mean()

        sg = -1.0 if bool(m[ia]) else 1.0              # 保留⇒移出，驱逐⇒加回
        Zp = (ZS + sg * e[..., ia:ia + 1]).clamp_min(1e-30)
        Np = NS + sg * e[..., ia:ia + 1] * V[ia]
        eo, es = err(Np / Zp), err(NS / ZS)
        out[(l, h, i)] = float(eo - es) if bool(m[ia]) else float(es - eo)
        del a, e, eS, NS
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--n_cand", type=int, default=32,
                    help="每个样本 brute-force 多少个近阈值候选。每个候选一次答案前向")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()

    m = ModelKVzip(a.model, "retain", "fastkvzip")
    ds = DataWrapper(a.data, load_dataset_all(a.data, m.tokenizer), m)
    set_gen_length(a.data, m)
    L = m.config.num_hidden_layers
    n = min(a.num, len(ds))
    rows = []
    for si in range(n):
        kv_f = ds.prefill_context(si, do_score=False)
        inputs, _ = ds.generate_answer(si, kv_f, prob=False)
        task = "qa" if "qa" in inputs else list(inputs.keys())[0]
        ids = torch.cat([inputs[task][k] for k in ("q", "a")], dim=-1)
        n_ans = len(inputs[task]["a"][0])
        n_ctx = kv_f.key_cache[0].shape[2]
        # post-RoPE query：用 attn.py 的 capture_q（钩 prepare 不行，满缓存那遍
        # flatten 恒为 False，见 scratch_ctrl_teacher 的说明）
        kv_f.capture_q, kv_f._q_cap = True, {}
        m.model(ids, past_key_values=kv_f)
        kv_f.capture_q = False
        qcap = {l: kv_f._q_cap[l] for l in range(L)}
        del kv_f
        torch.cuda.empty_cache()

        kv = ds.prefill_context(si, prefill_chunk=a.chunk, window_size=a.window,
                                chunk_ratio=a.ratio, level="pair")
        # **必须 clone。** `prefill` 带 `@torch.inference_mode()`，它建出来的张量是
        # inference tensor，**在推理上下文之外不允许就地修改**（这是 CLAUDE.md 记过的
        # 同一类坑）。而本探针的做法正是逐个候选翻转 `valid` 的一位再重算 NLL。
        kv.valid = kv.valid.clone()
        V = kv.valid                                   # [L,H,ctx_len]（含末尾永久窗口）
        sink = kv.sink
        assert V.shape[-1] == kv.ctx_len, (V.shape, kv.ctx_len)
        base = nll(m, ids, n_ans, kv)
        # 近阈值：`self.score` 是逐层 [1,H,n]，取全局阈值附近的三元组
        sc = torch.stack(kv.score, 0)[:, 0]            # [L,H,n_tot]
        nev = V.shape[-1]
        s_ev = sc[..., sink:sink + nev]
        # `torch.quantile` 在 >16777216 个元素时直接报错，而 s_ev 是
        # 28×4×~15万 ≈ 1680 万，正好卡在边界上。用排序取分位，与
        # `score.py:_threshold` 的做法一致（它也是 sort 后取第 int(N·ratio)−1 个）。
        _f = s_ev.reshape(-1).float()
        tau = _f.sort(descending=True).values[max(int(_f.numel() * a.ratio) - 1, 0)]
        flat = (s_ev - tau).abs().reshape(-1)
        idx = flat.argsort()[:a.n_cand]
        H_, N_ = s_ev.shape[1], s_ev.shape[2]
        cand = [(int(t // (H_ * N_)), int((t // N_) % H_), int(t % N_))
                for t in idx.tolist()]

        ua = attn_marginal(m, kv, qcap, list(range(L)), cand,
                           {l: V[l] for l in range(L)}, n_ctx, sink)
        for (l, h, i) in cand:
            V[l, h, i] = ~V[l, h, i]                   # 翻转，重算 NLL
            n2 = nll(m, ids, n_ans, kv)
            kept = not bool(V[l, h, i])                # 翻转前是否在 S 中
            V[l, h, i] = ~V[l, h, i]                   # 复原
            u_nll = (n2 - base) if kept else (base - n2)
            rows.append((si, l, h, i, u_nll, ua[(l, h, i)]))
        del kv
        torch.cuda.empty_cache()
        if (si + 1) % 5 == 0:
            arr = np.array([(r[4], r[5]) for r in rows])
            from scipy.stats import spearmanr
            rho = spearmanr(arr[:, 0], arr[:, 1]).statistic
            print(f"  {si+1}/{n}  n_cand={len(rows)}  Spearman(U^NLL, U^attn) = {rho:+.4f}",
                  flush=True)

    arr = np.array([(r[4], r[5]) for r in rows])
    np.save(os.path.join(ROOT, f"scratch_nll_{a.data}.npy"), np.array(rows))
    from scipy.stats import spearmanr
    rho = spearmanr(arr[:, 0], arr[:, 1])
    # 逐样本的 Spearman，才能报跨样本跨度（合并算一个数会把样本级方差藏起来）
    per = []
    for si in range(n):
        sub = np.array([(r[4], r[5]) for r in rows if r[0] == si])
        if len(sub) > 5:
            per.append(spearmanr(sub[:, 0], sub[:, 1]).statistic)
    print(f"\n=== {a.data} @ ratio {a.ratio}  {len(rows)} 个候选 / {n} 篇 ===")
    print(f"  合并 Spearman(U^NLL, U^attn) = {rho.statistic:+.4f}  p={rho.pvalue:.2e}")
    if per:
        print(f"  逐样本 Spearman  中位 {np.median(per):+.4f}  "
              f"均值 {np.mean(per):+.4f} ± {np.std(per):.4f}  n={len(per)}")
        print(f"  为正的样本比例 {np.mean(np.array(per) > 0):.1%}")
    print("\n判读：Retr.KV 强正、MultiHop 弱或反号 ⇒ 靶子错位是 −9.96 的原因，换教师有据；"
          "\n      两个 panel 都强正 ⇒ 靶子不是病根，换了也白换。")


if __name__ == "__main__":
    raise SystemExit(main())
