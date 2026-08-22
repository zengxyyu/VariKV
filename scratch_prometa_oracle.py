#!/usr/bin/env python3
"""ProMeta 的 **Oracle 未来使用探针** —— 不训练任何东西，只回答一个问题：

    **在同一段上下文的多个「真实未来」上，一个 KV 位置的未来效用分布，
    是否携带超出其均值的信息？**

若否，ProMeta 的「风险敏感不可逆遗忘」这个核心命题在本工作点上没有内容，
整条线当场停 —— 这是本仓库最贵的一条经验（撤回 49–63 全都是
「先建框架、后做判决实验、框架死掉」）。

**为什么这个探针在本仓库特别便宜**：`scbench_kv` 每条样本自带
**5 个 question / 5 个 answer、共享同一个 context**（`data["question"]`
是 list len=5）。⇒ ProMeta 最理想也最贵的数据结构「同一前缀 + 多个真实
未来查询」我们**免费拥有**，不需要合成未来、不需要采样 LLM。

**为什么不用改注意力代码**：`attn.py:114` 已有 `capture_q` 钩子，
把每层 post-RoPE 的 query 存进 `_q_cap`（VariKV-B 教师留下的，注释里
写明它**必须在 flatten 分支之外**，因为满缓存参照那次预填从不进
`prune_chunk`）。所以只需：预填满缓存拿 `key_cache` → 对每个未来跑一次
`_prob`（**单次 teacher-forced forward**）拿 `q` → 注意力自己在 GPU 上算。

    U[m, l, h, i] = max_{t, hq∈group(h)}  softmax_i( q[hq,t]·K[h,i]/√d )

**口径声明（必须随数字一起引用）**：softmax 只在**前缀位置**上归一化，
不含 q+a 自身 token。这定义的是「未来查询有多想要前缀位置 i」的相对量，
不是模型真实的注意力权重。换一种归一化会改变绝对值，但**不改变同一
(层,头) 内的排序**，而本探针的所有判据都只用排序。

**⚠ 面板选择会左右判据 A，方向与直觉相反，必须两个都跑**：若未来**互不相交**，
则 `mean = max/M` 是单调变换、排序必然相同 ⇒ 判据 A 平凡判否。分歧只可能来自
「被单个未来强需要」与「被全部未来弱需要」两类位置并存。
⇒ `scbench_kv`（5 个互不相交的 key 查找）**对判据 A 偏悲观**；
`scbench_qa_eng`（同一文档上 5 个自然语言问题，共享背景 + 各自证据）
**是分歧最可能出现的面板**。**只在一个面板上得到结论都不算。**

    .venv/bin/python scratch_prometa_oracle.py -d scbench_kv     --idx 0 --n 4
    .venv/bin/python scratch_prometa_oracle.py -d scbench_qa_eng --idx 0 --n 4
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "external/FastKVzip/prefill"))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("-d", "--data", default="scbench_kv")
    p.add_argument("-g", "--gate", default="fastkvzip")
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--keep", type=int, default=32768,
                   help="每个样本随机保留多少个前缀位置（含索引）。"
                        "全存的话 5x28x4x169k x fp16 = 378MB/样本。"
                        "均匀抽样保排序，本探针的判据只用排序。")
    p.add_argument("--tblock", type=int, default=32, help="query 时间维分块")
    p.add_argument("--out", default="scratch_prometa_oracle")
    return p.parse_args()


@torch.inference_mode()
def future_utility(kv, q_cap, n_prefix, tblock):
    """→ U: [L, Hkv, n_prefix] float32 (cpu)

    `q_cap[l]`: [1, Hq, T, d] post-RoPE；`kv.key_cache[l]`: [1, Hkv, N, d]。
    GQA 映射 `kv_head = q_head // (Hq // Hkv)`；组内对 q 头取 **max**
    —— 保留集是逐 (层, KV头) 的，只要组内任一 query 头需要该位置就算需要，
    这与 KVzip 族打分器「对 query 头取 max」的口径一致。
    """
    L = len(q_cap)
    out = []
    for l in range(L):
        q = q_cap[l]                      # [1,Hq,T,d]
        K = kv.key_cache[l]               # [1,Hkv,N,d]
        assert q.dim() == 4 and K.dim() == 4, (q.shape, K.shape)
        Hq, T, d = q.shape[1], q.shape[2], q.shape[3]
        Hkv, N = K.shape[1], K.shape[2]
        assert N >= n_prefix, (N, n_prefix)
        assert Hq % Hkv == 0, (Hq, Hkv)
        G = Hq // Hkv
        Kp = K[0, :, :n_prefix, :].float()          # [Hkv,n_prefix,d]
        acc = torch.zeros(Hkv, n_prefix, device=Kp.device, dtype=torch.float32)
        scale = 1.0 / (d ** 0.5)
        for h in range(Hkv):
            qh = q[0, h * G:(h + 1) * G].float()    # [G,T,d]
            for t0 in range(0, T, tblock):
                qb = qh[:, t0:t0 + tblock, :]       # [G,tb,d]
                a = torch.einsum("gtd,nd->gtn", qb, Kp[h]) * scale
                a = torch.softmax(a, dim=-1)        # 只在前缀上归一化（口径见 docstring）
                acc[h] = torch.maximum(acc[h], a.amax(dim=(0, 1)))
                del a
        out.append(acc.cpu())
        del Kp, acc
    return torch.stack(out, 0).numpy()               # [L,Hkv,n_prefix]


def main():
    args = parse()
    from data.load import load_dataset_all
    from data.wrapper import DataWrapper
    from model.wrapper import ModelKVzip
    from utils.func import set_gen_length

    model = ModelKVzip(args.model, "retain", args.gate)
    ds_raw = load_dataset_all(args.data, model.tokenizer)
    dataset = DataWrapper(args.data, ds_raw, model)
    set_gen_length(args.data, model)

    rng = np.random.default_rng(0)
    hi = min(args.idx + args.n, len(dataset))
    for idx in range(args.idx, hi):
        kv = dataset.prefill_context(idx, do_score=False)     # 满缓存，不驱逐
        n_prefix = int(kv.key_cache[0].shape[-2])
        print(f"[prometa] sample {idx}: prefix {n_prefix} tokens, "
              f"L={len(kv.key_cache)} Hkv={kv.key_cache[0].shape[1]}", flush=True)

        # 5 个真实未来（scbench 自带）；prob=False 只生成答案文本，不做额外 forward
        inputs, _ = dataset.generate_answer(idx, kv, prob=False)
        # ⚠ 不变量（补）：`generate_answer` 内部对每个 question 调 `model.generate`，
        # 后者 `update_cache=False` 并以 `kv.slice(seen_token_prev)` 回滚。若哪天默认
        # 变了，前缀会被 5 次生成污染，而 U 仍然算得出来 —— 静默失败。
        assert int(kv.key_cache[0].shape[-2]) == n_prefix, \
            f"generate_answer 改动了前缀：{kv.key_cache[0].shape[-2]} != {n_prefix}"
        tags = [t for t in inputs["eval_task"]]
        assert len(tags) >= 2, f"样本 {idx} 只有 {len(tags)} 个未来，本探针需要多个"

        keep = np.arange(n_prefix) if n_prefix <= args.keep else \
            np.sort(rng.choice(n_prefix, size=args.keep, replace=False))

        Us = []
        for tag in tags:
            kv.capture_q = True
            kv._q_cap = {}
            ids = torch.cat([inputs[tag]["q"], inputs[tag]["a"]], dim=1)
            _ = model._prob(ids, kv, device="cpu")
            kv.capture_q = False
            # ⚠ 不变量：`_prob` 用 update_cache=False，前缀必须原样回滚。
            assert int(kv.key_cache[0].shape[-2]) == n_prefix, \
                f"前缀被改动：{kv.key_cache[0].shape[-2]} != {n_prefix}"
            assert len(kv._q_cap) == len(kv.key_cache), \
                f"只捕到 {len(kv._q_cap)}/{len(kv.key_cache)} 层的 query"
            qc = [kv._q_cap[l] for l in range(len(kv.key_cache))]
            U = future_utility(kv, qc, n_prefix, args.tblock)   # [L,Hkv,n_prefix]
            print(f"  future {tag}: T={qc[0].shape[2]} "
                  f"U(max)={U.max():.4f} U(mean)={U.mean():.3e}", flush=True)
            Us.append(U[:, :, keep].astype(np.float16))
            kv._q_cap = {}
            del qc, U

        U = np.stack(Us, 0)                                     # [M,L,Hkv,keep]
        out = f"{args.out}_{args.data}_{idx}.npz"
        np.savez_compressed(out, U=U, keep=keep.astype(np.int32),
                            n_prefix=n_prefix, tags=np.array(tags))
        print(f"[prometa] saved {out}  U{U.shape}  "
              f"{os.path.getsize(out)/1e6:.1f} MB", flush=True)
        del kv, inputs, Us, U
        torch.cuda.empty_cache()
    print("Finished.")


if __name__ == "__main__":
    main()
