"""Stage 2a 驱动：真实语料（fineweb-edu）上训练/评测五档 + 随机驱逐对照。

    python scratch_stage2a.py train --tier 5
    python scratch_stage2a.py eval  --tier 1 2 3 4 5 --extra random

**随机驱逐对照从一开始就带上**。stage1 的教训：不带它就分不清「方法有效」和
「这个任务对驱逐根本没有分辨率」—— 在 stage1 上 random 赢过了包括 recency 和
已发表 Expected Attention 在内的所有判据，那才暴露出任务本身判不了驱逐。
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from varikv.config import Config
from varikv.train import build, forward_loss
from varikv import realdata


def make_random_evict(mem, seed=1234):
    """把驱逐判据替换成随机 —— 对照组，检验任务是否真的对驱逐有分辨率。"""
    def patched(k, v, keep_from, n_real):
        g = torch.Generator(device="cpu").manual_seed(seed + n_real)
        r = torch.rand(n_real, generator=g).to(k.device)
        return r.unsqueeze(0).expand(k.shape[0], n_real), {}
    return patched


def run_train(args):
    cfg = Config()
    cfg.cache.budget = args.budget
    cfg.memory.num_slots = args.num_slots
    cfg.train.max_steps = args.steps
    cfg = cfg.ablation(args.tier)
    model, tok, mem = build(cfg)

    params = mem.trainable_parameters()
    if not params:
        print(f"[tier {args.tier}] 无可训练参数，跳过")
        return
    print(f"[tier {args.tier}] evict={cfg.cache.evict_policy} "
          f"absorb={cfg.cache.absorb_mode} 可训练 {sum(p.numel() for p in params)/1e6:.2f}M")

    train_set, _ = realdata.load_fineweb(tok, n_train=args.n_train, n_val=args.n_val)
    print(f"训练文档 {len(train_set)} 篇，"
          f"长度 {min(s.n_tokens for s in train_set)}–{max(s.n_tokens for s in train_set)}")

    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=cfg.train.max_steps,
        pct_start=cfg.train.warmup_steps / max(cfg.train.max_steps, 1),
    )

    step = 0
    opt.zero_grad()
    while step < cfg.train.max_steps:
        for s in train_set:
            if step >= cfg.train.max_steps:
                break
            ctx, q, a = realdata.encode_real(
                s, cfg.device, args.target_len, cfg.train.max_train_context
            )
            if ctx.shape[1] <= cfg.cache.budget + cfg.cache.n_sink:
                continue
            lm = forward_loss(model, mem, ctx, q, a)
            aux = mem.collect_aux_loss()
            loss = lm
            if "free_energy" in aux:
                loss = loss + cfg.train.free_energy_weight * aux["free_energy"]
            if "predictor" in aux:
                loss = loss + cfg.free_energy.predictor_loss_weight * aux["predictor"]
            (loss / cfg.train.grad_accum).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                opt.step(); sched.step(); opt.zero_grad()
            if step % cfg.train.log_every == 0:
                msg = f"step {step:5d}  lm {lm.item():.4f}"
                for k_, v_ in aux.items():
                    msg += f"  {k_} {v_.item():.4f}"
                print(msg, flush=True)
            step += 1

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = out / f"real_k{args.num_slots}_tier{args.tier}.pt"
    torch.save({"memory": mem.state_dict(), "tier": args.tier,
                "num_slots": args.num_slots}, ck)
    print(f"saved {ck}")


@torch.no_grad()
def run_eval(args):
    results = {}
    for tier in args.tier:
        for mode in (["normal"] + (["random"] if args.extra_random and tier in (4, 5) else [])):
            cfg = Config()
            cfg.cache.budget = args.budget
            cfg.memory.num_slots = args.num_slots
            cfg = cfg.ablation(tier)
            model, tok, mem = build(cfg)
            mem.eval()
            ck = Path(args.out) / f"real_k{args.num_slots}_tier{tier}.pt"
            if ck.exists():
                mem.load_state_dict(torch.load(ck, map_location=cfg.device)["memory"])
            elif tier not in (1, 3):
                print(f"[warn] tier {tier} 无 ckpt，评的是随机初始化的记忆")
            if mode == "random":
                mem._evict_scores = make_random_evict(mem)

            _, val = realdata.load_fineweb(tok, n_train=args.n_train, n_val=args.n_val)
            tot, n = 0.0, 0
            for s in val[: args.limit]:
                ctx, q, a = realdata.encode_real(s, cfg.device, args.target_len, 0)
                if ctx.shape[1] <= cfg.cache.budget + cfg.cache.n_sink:
                    continue
                tot += forward_loss(model, mem, ctx, q, a).item()
                n += 1
            key = f"tier{tier}" + ("_randevict" if mode == "random" else "")
            results[key] = {"nll": tot / max(n, 1), "n": n}
            print(f"{key:22s} nll {tot/max(n,1):.4f}  (n={n})", flush=True)
            del model, mem
            torch.cuda.empty_cache()

    print("\n=== Stage 2a：真实语料 (fineweb-edu) ===")
    base = results.get("tier1", {}).get("nll")
    for k in sorted(results, key=lambda x: results[x]["nll"]):
        d = f"  Δ vs tier1 {results[k]['nll'] - base:+.4f}" if base else ""
        print(f"  {k:22s} {results[k]['nll']:.4f}{d}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("--tier", type=int, nargs="+", default=[5])
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--num_slots", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--target_len", type=int, default=256)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--extra_random", action="store_true")
    ap.add_argument("--out", type=str, default="varikv/ckpt_real")
    ap.add_argument("--json_out", type=str, default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        a.tier = a.tier[0]
        run_train(a)
    else:
        run_eval(a)
