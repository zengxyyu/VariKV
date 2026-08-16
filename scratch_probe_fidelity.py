#!/usr/bin/env python3
"""残差究竟有没有"把压缩模型拉回满缓存"—— 直接测，不再靠分数曲线推断。

**这是对我自己一个说法的证伪实验。** 我根据任务分数的形状说过"这个残差做的是让压缩后
的模型表现得更像满缓存"，但那只是**行为假说**：分数曲线的形状可以由好几种机制产生。
要坐实它，必须直接测到满缓存的分布距离。

对同一批样本、同一个 query，测答案位置上

    D_base = KL( p_full ‖ p_base )        FastKVzip 压缩后
    D_ours = KL( p_full ‖ p_ours )        再加学习残差

并同时记录任务分数。于是每个 panel 落进一个 2×2：

|            | 更接近 full          | 更远离 full            |
|------------|---------------------|-----------------------|
| **分数更好** | fidelity recovery   | beneficial denoising  |
| **分数更差** | harmful restoration | destructive           |

预期（若我的假说成立）：Retr.KV 落 fidelity recovery，Retr.MultiHop 落
**harmful restoration** —— 即 `D_ours < D_base` 但分数更差。那会是一个比任何分数曲线
都强的结论：**更忠实于满缓存 ≠ 任务上更好**。若 MultiHop 上 `D_ours > D_base`，
我的假说就被否了，−9.96 得另找解释。

用 KL 而不是框架自带的 `mean|Δp|`：后者是总变差量级的量，对分布尾部不敏感，而
answer-token 的预测常常是长尾。KL(p_full‖p) 还有直接的解释——把 full 当真值时的
额外编码代价。
"""
import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
_P = os.path.join(ROOT, "external/FastKVzip/prefill")
sys.path.insert(0, _P)
os.chdir(_P)

from data import DataWrapper, load_dataset_all                  # noqa: E402
from model import ModelKVzip                                    # noqa: E402
from utils import set_gen_length                                # noqa: E402


@torch.inference_mode()
def probs(model, ids, n_ans, kv):
    """在**给定的** token 串上 teacher-force，取答案位置的下一 token 分布 [T_ans, V]。

    `ids` 必须由满缓存那一遍构造一次、两臂共用。若两边各自调 `generate_answer`，
    拿到的是各自**生成**的答案，长度都不同（实测 33 vs 35），根本没法比 —— 这也正是
    `Evaluator` 的做法：`inputs` 由满缓存建一次，压缩缓存复用同一串。
    """
    p = model._prob(ids, kv)
    return p[-n_ans - 1: -1].float()                   # 与 _cal 的切法一致


def kl(pf, pp):
    """KL(p_full ‖ p) 在答案位置上取均值。clamp 防 log(0)。"""
    pf = pf.clamp_min(1e-12); pp = pp.clamp_min(1e-12)
    return float((pf * (pf.log() - pp.log())).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", default="scbench_kv")
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--ckpt", default="../../../varikv/ctrl_b_a1_s0.pt/memoryless.pt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--chunk", type=int, default=16000)
    ap.add_argument("--window", type=int, default=4096)
    a = ap.parse_args()

    import torch as _t
    from attention.control_memory import ControlMemory

    def build(with_ctrl):
        m = ModelKVzip(a.model, "control_learned" if with_ctrl else "retain", "fastkvzip")
        if with_ctrl:
            ck = _t.load(a.ckpt, map_location="cpu")
            ns = ck.get("slots", 8)
            cm = ControlMemory(ck.get("d_kv", 128), ck["L"], ck["H"], n_slots=ns,
                               d_m=ck.get("dim", 128), mode="memoryless",
                               typed=ck["state"]["M_init"].shape[2] == 2 * ns)
            cm.load_state_dict(ck["state"])
            m.ctrl_module = cm.to(m.device).eval()
            m.ctrl_seed, m.ctrl_rho_max = 0, 1.0
        return m

    # **两个模型不能同时驻留**（各 15GB + 两份 169k 的 KV）。逐臂跑、把 p_full 存下来
    # 复用：p_full 由 ratio=1.0 的预填给出，与是否挂 controller 无关
    # （ratio=1.0 不进 prune_chunk，这是构造性的，见 learned_ctrlcache 的说明）。
    out = {}
    for arm in ("base", "ours"):
        m = build(arm == "ours")
        ds = DataWrapper(a.data, load_dataset_all(a.data, m.tokenizer), m)
        set_gen_length(a.data, m)
        n = min(a.num, len(ds))
        rec = []
        for i in range(n):
            kv_f = ds.prefill_context(i, do_score=False)
            inputs, _ = ds.generate_answer(i, kv_f, prob=False)
            task = "qa" if "qa" in inputs else list(inputs.keys())[0]
            ids = _t.cat([inputs[task][k] for k in ("q", "a")], dim=-1)
            n_ans = len(inputs[task]["a"][0])
            pf = probs(m, ids, n_ans, kv_f); del kv_f
            kv_c = ds.prefill_context(i, prefill_chunk=a.chunk,
                                      window_size=a.window, chunk_ratio=a.ratio,
                                      level="pair")
            pp = probs(m, ids, n_ans, kv_c); del kv_c
            rec.append(kl(pf, pp))
            _t.cuda.empty_cache()
            if (i + 1) % 10 == 0:
                print(f"  {arm} {i+1}/{n}  KL 均值 {sum(rec)/len(rec):.4f}", flush=True)
        out[arm] = rec
        del m; _t.cuda.empty_cache()

    import statistics as st
    b, o = out["base"], out["ours"]
    d = [o[i] - b[i] for i in range(min(len(b), len(o)))]
    print(f"\n=== {a.data} @ ratio {a.ratio}  n={len(d)} ===")
    print(f"  KL(full‖base) = {st.mean(b):.4f}")
    print(f"  KL(full‖ours) = {st.mean(o):.4f}")
    m_, sd = st.mean(d), st.stdev(d)
    t = m_ / (sd / len(d) ** 0.5)
    print(f"  差 (ours−base) = {m_:+.4f} ± {sd:.4f}   t={t:+.2f}  "
          f"{'**ours 更接近 full**' if t < -2 else ('**ours 更远离 full**' if t > 2 else '不可分')}")


if __name__ == "__main__":
    raise SystemExit(main())
