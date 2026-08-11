"""三个对照实验，拆开 stage1 评测全 0 的成因。

A. max_new_tokens 是否不够（模型先说一段开场白再给答案）
B. prompt 结尾的空格是否破坏了分词（"[ANSWER] " vs "[ANSWER]"）
C. tier1（无驱逐、n_mem=0）下，记忆路径与纯 HF 路径是否真的等价
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, encode_sample
from varikv.evaluate import generate, exact_match
from stage1 import data as stage1_data


@torch.no_grad()
def hf_gen(model, tok, prompt, max_new_tokens):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    am = torch.ones_like(ids)
    out = model.generate(ids, attention_mask=am, max_new_tokens=max_new_tokens,
                         do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    cfg = Config()
    cfg.cache.budget = 256
    cfg = cfg.ablation(1)
    model, tok, mem = build(cfg)
    mem.eval()

    val = stage1_data.load(str(Path(__file__).parent / "stage1/val.jsonl"))
    val = sorted(val, key=lambda s: s.n_distract)[:8]

    print("\n" + "=" * 78)
    print("A/B  长度 & 结尾空格   (纯 HF，无记忆)")
    print("=" * 78)
    hit = {"16_sp": 0, "48_sp": 0, "48_nosp": 0}
    for i, s in enumerate(val):
        prompt, gold = stage1_data.render(s)
        nosp = prompt.rstrip()                      # 去掉 "[ANSWER] " 末尾空格
        a = hf_gen(model, tok, prompt, 16)
        b = hf_gen(model, tok, prompt, 48)
        c = hf_gen(model, tok, nosp, 48)
        for k, p in (("16_sp", a), ("48_sp", b), ("48_nosp", c)):
            hit[k] += int(exact_match(p, gold))
        print(f"\n[{i}] gold={gold!r}")
        print(f"   16tok +sp : {a!r:60s} {exact_match(a, gold)}")
        print(f"   48tok +sp : {b!r:60s} {exact_match(b, gold)}")
        print(f"   48tok -sp : {c!r:60s} {exact_match(c, gold)}")
    print(f"\n  命中率 /{len(val)}: {hit}")

    print("\n" + "=" * 78)
    print("C  tier1 记忆路径 vs 纯 HF：第一步 logits 是否一致")
    print("=" * 78)
    for i, s in enumerate(val[:4]):
        ctx, q, _ = encode_sample(tok, s, cfg.device)
        # 记忆路径：prefill(ctx) 后前向 q
        cache = mem.prefill(model, ctx)
        B, T = q.shape
        pos = torch.arange(mem.n_seen, mem.n_seen + T, device=q.device).unsqueeze(0)
        am = torch.ones(B, mem.n_mem + mem.n_seen + T, device=q.device, dtype=torch.long)
        lg_mem = model(input_ids=q, past_key_values=cache, position_ids=pos,
                       attention_mask=am, use_cache=True).logits[:, -1].float()
        # 纯 HF：一次性前向 ctx+q
        full = torch.cat([ctx, q], dim=1)
        lg_hf = model(input_ids=full,
                      attention_mask=torch.ones_like(full)).logits[:, -1].float()
        d = (lg_mem - lg_hf).abs().max().item()
        t1, t2 = lg_mem.argmax(-1).item(), lg_hf.argmax(-1).item()
        print(f"[{i}] n_mem={mem.n_mem} n_seen={mem.n_seen} ctx={ctx.shape[1]}  "
              f"max|Δlogit|={d:7.3f}  top1: mem={tok.decode([t1])!r} hf={tok.decode([t2])!r}"
              f"  {'SAME' if t1 == t2 else '**DIFF**'}")


if __name__ == "__main__":
    main()
