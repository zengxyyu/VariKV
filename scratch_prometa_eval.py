#!/usr/bin/env python3
"""用**上游 `eval_chunk.py` 本体**评 ProMeta —— 不复制它的评测循环。

为什么要 runpy 而不是自己写一个 loop：结果目录名、`set_ratios()`、
`save_result` 的口径、`--num/--idx` 的语义、`Finished.` 标记，全都必须与
基线**逐字一致**，否则读数脚本读不到、或者读成另一批样本。复制一份循环
就等于手抄第二份清单（第④类错）。

用法：本脚本自己的开关放前面，**其余参数原样透传给 `eval_chunk.py`**。

    .venv/bin/python -B scratch_prometa_eval.py \\
        --prometa_ckpt varikv/prometa_s0.pt --pm_gamma 0.5 --pm_beta 1.0 \\
        -- -m Qwen/Qwen2.5-7B-Instruct-1M -g fastkvzip -d scbench_kv \\
           --num 100 --prefill_chunk 16000 --window_size 4096 --level pair \\
           --tag _pmtest

⚠ `VARIKV_RATIOS` 要像平时一样 export（`eval.py` 与 `results/parse.py` 各有
一份 `set_ratios()`，两边都得看见）。
⚠ **`--pm_gamma 0` 是构造性零点**：应当与同参数基线逐位相同。任何一轮
ProMeta 评测都要带上这一格，理由见 `scratch_prometa_smoke.py` 的开头。
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFILL = os.path.join(HERE, "external/FastKVzip/prefill")
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--prometa_ckpt", required=True)
    ap.add_argument("--pm_beta", type=float, default=1.0)
    ap.add_argument("--pm_gamma", type=float, default=1.0)
    ap.add_argument("--pm_combine", default="resid", choices=["resid", "replace"])
    ap.add_argument("--pm_pool_layer", type=int, default=-1,
                    help="-1 = 用 ckpt 训练时的 pool_layer（默认，且是唯一正确的）")
    ap.add_argument("--pm_quiet", action="store_true")
    a, rest = ap.parse_known_args()
    rest = [x for x in rest if x != "--"]

    os.chdir(PREFILL)
    sys.path.insert(0, PREFILL)

    from prometa.integrate import install, load_student, tag_suffix

    net, ck = load_student(os.path.join(HERE, a.prometa_ckpt)
                           if not os.path.isabs(a.prometa_ckpt) else a.prometa_ckpt)
    trained_pl = int(ck.get("args", {}).get("pool_layer", 14))
    pool_layer = trained_pl if a.pm_pool_layer < 0 else a.pm_pool_layer
    if pool_layer != trained_pl:
        print(f"[prometa] ⚠ 评测用 pool_layer={pool_layer} 与训练时的 "
              f"{trained_pl} **不一致** —— 这是一个消融，不是默认路径", flush=True)
    cfg = dict(beta=a.pm_beta, gamma=a.pm_gamma, combine=a.pm_combine,
               pool_layer=pool_layer)
    print(f"[prometa] ckpt={a.prometa_ckpt} epoch={ck.get('epoch')} "
          f"arch={ck['arch']} teacher={ck.get('teacher')}", flush=True)

    # tag 必须带上配置，否则不同配置写进同一个结果目录互相覆盖
    suf = tag_suffix(cfg)
    if "--tag" in rest:
        i = rest.index("--tag")
        rest[i + 1] = rest[i + 1] + suf
    else:
        rest += ["--tag", suf]

    install(net, verbose=not a.pm_quiet, **cfg)
    sys.argv = ["eval_chunk.py"] + rest
    print(f"[prometa] 透传给 eval_chunk.py: {' '.join(rest)}", flush=True)
    runpy.run_path(os.path.join(PREFILL, "eval_chunk.py"), run_name="__main__")


if __name__ == "__main__":
    main()
