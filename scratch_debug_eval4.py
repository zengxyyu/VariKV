"""验证修复：q_text 去尾空格 + answer 带前导空格 + max_new_tokens 加大。

不改仓库源码，先在这里复刻 encode_sample/generate 的修复版，确认 tier1 能出非零分。
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build
from stage1 import data as stage1_data


def encode_fixed(tok, sample, device):
    ctx = tok(sample.context, return_tensors="pt", add_special_tokens=True).input_ids
    q_text = f"\n\n[QUERY] {sample.question}\n[ANSWER]"      # ← 去掉尾空格
    q = tok(q_text, return_tensors="pt", add_special_tokens=False).input_ids
    a = tok(" " + sample.answer, return_tensors="pt", add_special_tokens=False).input_ids
    return ctx.to(device), q.to(device), a.to(device)


@torch.no_grad()
def gen(model, mem, tok, ctx_ids, q_ids, max_new_tokens):
    cache = mem.prefill(model, ctx_ids)
    cur, out_ids = q_ids, []
    for _ in range(max_new_tokens):
        B, T = cur.shape
        kv_len = mem.n_mem + mem.n_seen
        pos = torch.arange(mem.n_seen, mem.n_seen + T,
                           device=cur.device).unsqueeze(0).expand(B, T)
        am = torch.ones(B, kv_len + T, device=cur.device, dtype=torch.long)
        out = model(input_ids=cur, past_key_values=cache, position_ids=pos,
                    attention_mask=am, use_cache=True)
        cache = out.past_key_values
        mem.n_seen += T
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        if nxt.item() == tok.eos_token_id:
            break
        out_ids.append(nxt.item())
        cur = nxt
    return tok.decode(out_ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 200])
    args = ap.parse_args()

    cfg = Config()
    cfg.cache.budget = args.budget
    cfg = cfg.ablation(args.tier)
    model, tok, mem = build(cfg)
    mem.eval()

    val = [s for s in stage1_data.load(str(Path(__file__).parent / "stage1/val.jsonl"))
           if s.n_distract in args.levels][: args.n]

    old_hit, new_hit = 0, 0
    buckets = defaultdict(lambda: [0, 0])
    for i, s in enumerate(val):
        # 旧写法
        ctx_o = tok(s.context, return_tensors="pt").input_ids.to(cfg.device)
        q_o = tok(f"\n\n[QUERY] {s.question}\n[ANSWER] ", return_tensors="pt",
                  add_special_tokens=False).input_ids.to(cfg.device)
        p_old = gen(model, mem, tok, ctx_o, q_o, 16)
        # 新写法
        ctx, q, _ = encode_fixed(tok, s, cfg.device)
        p_new = gen(model, mem, tok, ctx, q, args.max_new)

        o = s.answer.lower() in p_old.lower()
        n = s.answer.lower() in p_new.lower()
        old_hit += o
        new_hit += n
        buckets[(s.kind, s.n_distract)][0] += n
        buckets[(s.kind, s.n_distract)][1] += 1
        if i < 5:
            print(f"[{i}] gold={s.answer!r}\n    old={p_old!r} -> {o}\n    new={p_new!r} -> {n}")

    print(f"\n{'='*70}")
    print(f"tier {args.tier}  n={len(val)}  levels={args.levels}")
    print(f"  旧（尾空格 + 16 tok）: {old_hit}/{len(val)} = {old_hit/len(val):.3f}")
    print(f"  新（无尾空格 + {args.max_new} tok）: {new_hit}/{len(val)} = {new_hit/len(val):.3f}")
    print("  分组（新）:")
    for k in sorted(buckets, key=str):
        c, t = buckets[k]
        print(f"    {str(k):24s} {c}/{t} = {c/max(t,1):.3f}")


if __name__ == "__main__":
    main()
