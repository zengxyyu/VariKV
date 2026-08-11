"""定位 stage1 评测全 0 的原因：打印原始 pred，并和不经过记忆的纯 HF 前向对照。

用法：
    .venv/bin/python scratch_debug_eval.py --tier 1 --n 6
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample
from varikv.evaluate import generate, exact_match
from stage1 import data as stage1_data


@torch.no_grad()
def plain_generate(model, tok, prompt: str, max_new_tokens=16):
    """完全不碰记忆模块的对照：普通 HF 前向 + 贪心解码。"""
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.cache.budget = args.budget
    cfg = cfg.ablation(args.tier)
    model, tok, mem = build(cfg)
    mem.eval()
    if args.ckpt and Path(args.ckpt).exists():
        mem.load_state_dict(torch.load(args.ckpt, map_location=cfg.device)["memory"])
        print(f"loaded {args.ckpt}")

    val = stage1_data.load(str(Path(__file__).parent / "stage1/val.jsonl"))
    # 优先挑最短的（n_distract=0，109 token，根本不触发驱逐）
    val = sorted(val, key=lambda s: s.n_distract)[: args.n]

    print(f"\n{'='*78}\ntier={args.tier} budget={cfg.cache.budget} "
          f"model={cfg.model_name}\n{'='*78}")

    for i, s in enumerate(val):
        ctx, q, a = encode_sample(tok, s, cfg.device)
        pred = generate(model, mem, tok, ctx, q)
        prompt, _ = stage1_data.render(s)
        plain = plain_generate(model, tok, prompt)

        print(f"\n--- sample {i}  kind={s.kind}  n_distract={s.n_distract}  "
              f"ctx_tokens={ctx.shape[1]} ---")
        print(f"  question   : {s.question}")
        print(f"  GOLD       : {s.answer!r}")
        print(f"  pred(mem)  : {pred!r}   match={exact_match(pred, s.answer)}")
        print(f"  pred(plain): {plain!r}   match={exact_match(plain, s.answer)}")
        if i == 0:
            print(f"\n  [prompt tail 300 chars]\n  ...{prompt[-300:]!r}")
            print(f"  [q_ids decoded] {tok.decode(q[0])!r}")
            print(f"  [a_ids decoded] {tok.decode(a[0])!r}")
            print(f"  [ctx tail decoded] {tok.decode(ctx[0, -40:])!r}")


if __name__ == "__main__":
    main()
