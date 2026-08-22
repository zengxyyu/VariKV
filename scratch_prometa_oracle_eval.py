#!/usr/bin/env python3
"""**ProMeta 的判决实验：Oracle 下游臂。** 不训练任何东西。

────────────────────────────────────────────────────────────────────────────
它回答什么
────────────────────────────────────────────────────────────────────────────
ProMeta 的**全部**新颖性等价于一句话：`ρ_β`（β>0，偏向最坏未来）优于 `β=0`
（期望未来效用，＝ LookaheadKV 一类已发表方法）。
§十一之十六 用离线代理目标（最坏未来保住多少注意力质量）测到 **β>0 单调有害**，
但那是**质量代理**，而本仓库已实测过**保真度与任务效用会背离**
（`FINDINGS_DENOISING.md`）。⇒ 必须用**真实任务分数**再判一次。

本脚本把 **oracle 未来效用**（用样本自带的真实 question/answer 算）直接当保留分
注入驱逐通路，扫 β。

**⚠ 它是「特权信息参照臂」，不是任务分数的上界（2026-08-22 撤回上一版措辞）。**
上一版写「它是任何 Student 的上界」并据此写下判据「若 oracle 上 β=0≈β>0，
则任何 Student 都不可能靠 β 赢」—— **两句都过强**。理由是本仓库自己测过的
那件事：**保真度与任务效用会背离**（`FINDINGS_DENOISING.md`：恢复得越忠实、
Retr.MultiHop 分数越低）。一个**有偏**的 Student 完全可能因为正则化或
任务对齐的偏置，在下游**超过**用真值 `U` 的 oracle。
⇒ 正确说法是 **privileged-information comparator / upper-information
reference**，不是 task-performance upper bound。
⇒ 判据也要相应放宽（见下）。

────────────────────────────────────────────────────────────────────────────
怎么做到零 upstream 改动
────────────────────────────────────────────────────────────────────────────
`eval_chunk.py` 每个样本的顺序是**固定**的：

    满缓存 prefill_context(chunk_ratio=1.0)  →  generate_answer(kv_full)
      →  del kv  →  for ratio: prefill_context(chunk_ratio=ratio)

所以 patch 三处即可：
  · `prefill_context`：`chunk_ratio>=1.0` 时**清空** stash 并记下 idx；
    `<1.0` 时**断言** stash 已就绪且 idx 对得上（顺序假设写成断言，不靠信仰）。
  · `generate_answer`：原函数返回后，用**当时还活着的满缓存 kv** + 它给出的
    真实 q/a 算 oracle `U`，塞进 stash。
  · `_init_kv`：只给「新建的、类型恰为 RetainCache 的」cache 挂上 oracle 表。

满缓存参照那次 `chunk_ratio=1.0` ⇒ 上游根本不进 `prune_chunk` ⇒
**Oracle 对参照是构造性无操作**，参照天然干净（VariKV 残差那条线栽过的洞这里没有）。

    export VARIKV_RATIOS=0.1
    .venv/bin/python -B scratch_prometa_oracle_eval.py --pm_beta 0 --pm_gamma 0.5 \\
      -- -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip -d scbench_kv --num 40 \\
         --prefill_chunk 16000 --window_size 4096 --level pair --tag _pmo
"""
import argparse
import os
import runpy
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PREFILL = os.path.join(HERE, "external/FastKVzip/prefill")
sys.path.insert(0, HERE)

STASH = {}


def build_oracle(model, kv, inputs, ranges, tblock, verbose=True):
    """→ `{(lo,hi): U[M,L,Hkv,n] fp16 on GPU}`，用样本自带的真实未来算。"""
    from prometa.teacher import future_utility
    n_prefix = int(kv.key_cache[0].shape[-2])
    Lk = len(kv.key_cache)
    tags = list(inputs["eval_task"])
    assert len(tags) >= 2, f"该样本只有 {len(tags)} 个未来，Oracle 臂需要多个"
    acc = {r: [] for r in ranges}
    with torch.no_grad():
        for tg in tags:
            kv.capture_q, kv._q_cap = True, {}
            ids = torch.cat([inputs[tg]["q"], inputs[tg]["a"]], dim=1)
            model(ids, kv, update_cache=False)
            kv.capture_q = False
            assert int(kv.key_cache[0].shape[-2]) == n_prefix, \
                f"前缀被改动：{kv.key_cache[0].shape[-2]} != {n_prefix}"
            assert len(kv._q_cap) == Lk, f"只捕到 {len(kv._q_cap)}/{Lk} 层"
            qc = [kv._q_cap[l] for l in range(Lk)]
            for lo, hi in ranges:
                acc[(lo, hi)].append(torch.as_tensor(
                    future_utility(kv.key_cache, qc, lo, hi, tblock)))
            kv._q_cap = {}
            del qc
    out = {r: torch.stack(v, 0).half().to(kv.key_cache[0].device)
           for r, v in acc.items()}
    if verbose:
        tot = sum(v.numel() * 2 for v in out.values()) / 1e6
        print(f"[oracle] M={len(tags)} 个真实未来 × {len(ranges)} 个 chunk，"
              f"前缀 {n_prefix}，表 {tot:.0f} MB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pm_beta", type=float, default=0.0)
    ap.add_argument("--pm_gamma", type=float, default=0.5)
    ap.add_argument("--pm_combine", default="resid",
                    choices=["resid", "replace", "tokonly", "quotaonly", "oracleboth"],
                    help="后三个是**匹配动作空间的分解**（见 prometa/cache.py:"
                         "_pm_rebuild_mask）：tokonly 只换头内留谁、quotaonly 只换"
                         "每头拿多少、oracleboth 是上界。γ 对它们无效。")
    ap.add_argument("--tblock", type=int, default=32)
    ap.add_argument("--pm_quiet", action="store_true")
    a, rest = ap.parse_known_args()
    rest = [x for x in rest if x != "--"]

    # 从透传参数里取 chunk / window（**必须与 eval_chunk 实际用的一致**，
    # 否则算出来的 chunk 边界对不上，`_pm_R` 会硬报错 —— 那是设计好的安全网）
    def grab(flag, default, cast):
        return cast(rest[rest.index(flag) + 1]) if flag in rest else default
    chunk = grab("--prefill_chunk", 16000, int)
    window = grab("--window_size", 4096, int)

    os.chdir(PREFILL)
    sys.path.insert(0, PREFILL)

    from attention.kvcache import RetainCache
    from data.wrapper import DataWrapper
    from model.wrapper import ModelKVzip

    from prometa.cache import make_prometa_cache
    from prometa.teacher import chunk_ranges

    PM = make_prometa_cache(RetainCache)
    cfg = dict(beta=a.pm_beta, gamma=a.pm_gamma, combine=a.pm_combine)

    o_pc, o_ga, o_init = (DataWrapper.prefill_context,
                          DataWrapper.generate_answer, ModelKVzip._init_kv)

    def pc(self, idx, do_score=False, prefill_chunk=16000, window_size=512,
           chunk_ratio=1.0, level="pair", save_hidden=False):
        if chunk_ratio >= 1.0:
            STASH.clear()
            STASH["idx"] = idx
        else:
            # 顺序假设写成断言：stash 必须已就绪、且是**这个**样本的
            assert STASH.get("U") is not None, \
                "压缩预填时 oracle 表还没算 —— eval_chunk 的调用顺序变了"
            assert STASH.get("idx") == idx, (STASH.get("idx"), idx)
        return o_pc(self, idx, do_score, prefill_chunk, window_size,
                    chunk_ratio, level, save_hidden)

    def ga(self, idx, kv, prob=True):
        out = o_ga(self, idx, kv, prob=prob)
        if STASH.get("U") is None:
            n_prefix = int(kv.key_cache[0].shape[-2])
            sys_len = int(self.model.sys_prompt_ids.shape[1])
            clen = n_prefix - sys_len
            w = window if clen >= chunk else int(0.02 * clen)   # 上游短上下文重标定
            _, use = chunk_ranges(n_prefix, sys_len, chunk, w)
            assert use, f"样本 {idx} 没有可用 chunk（clen={clen}）"
            STASH["U"] = build_oracle(self.model, kv, out[0], use, a.tblock,
                                      verbose=not a.pm_quiet)
            STASH["idx"] = idx
        return out

    def init(self, kv=None, evict_range=(0, 0)):
        c = o_init(self, kv=kv, evict_range=evict_range)
        if kv is None and type(c) is RetainCache and STASH.get("U") is not None:
            c.__class__ = PM
            c.pm_init(None, oracle=STASH["U"], verbose=not a.pm_quiet, **cfg)
        return c

    DataWrapper.prefill_context = pc
    DataWrapper.generate_answer = ga
    ModelKVzip._init_kv = init

    suf = ("_pmoR" + f"{a.pm_beta:g}".replace(".", "p").replace("-", "m")
           + "g" + f"{a.pm_gamma:g}".replace(".", "p") + a.pm_combine[:3])
    if "--tag" in rest:
        rest[rest.index("--tag") + 1] += suf
    else:
        rest += ["--tag", suf]
    print(f"[oracle-eval] cfg={cfg} chunk={chunk} window={window}\n"
          f"[oracle-eval] ⚠ **特权信息参照臂**（不是任务分数上界）：用了未来查询、"
          f"按定义泄漏，只能与其他 Oracle 臂互比、不能当方法\n"
          f"[oracle-eval] 透传：{' '.join(rest)}", flush=True)
    sys.argv = ["eval_chunk.py"] + rest
    runpy.run_path(os.path.join(PREFILL, "eval_chunk.py"), run_name="__main__")


if __name__ == "__main__":
    main()
