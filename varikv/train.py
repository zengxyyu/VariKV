"""训练：冻结 LLM，只训记忆模块与摊销 F 预测器。

Loss = lm_loss + w_F·free_energy + w_pred·predictor_distill

- lm_loss 只算在 answer token 上。这一项让梯度**从最终输出回流**，
  因此失真被隐式定义在语言建模输出空间而非 KV 重建空间 —— 自动满足缺口①的精神
  （重建 KV 向量只是宽松代理，真正要保的是下游输出）。
- free_energy 是 ELBO 意义的变分自由能（重建 + KL），作小权重正则。
- predictor_distill 让摊销预测器逼近精确 F；其梯度不回流到记忆模块。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from varikv.cache import MemoryAugmentedCache
from varikv.config import Config
from stage1 import data as stage1_data


def build(cfg: Config):
    dtype = getattr(torch, cfg.dtype)
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",   # 需要能取到 q_proj 中间量做 query 统计
    ).to(cfg.device)
    model.eval()
    for p in model.parameters():          # 冻结 LLM（HANDOFF §1）
        p.requires_grad_(False)

    mem = MemoryAugmentedCache(model, cfg).to(cfg.device, dtype=dtype)
    return model, tok, mem


def encode_sample(tok, sample, device, max_context: int = 0):
    """context 走预填（会被驱逐/吸收），query+answer 走正常前向。

    max_context>0 时截断上下文的**尾部**。stage1 的 needle 在最前面、干扰项在后面，
    所以截尾保留 needle、只减少干扰强度 —— 任务结构不变，难度下降。
    这样梯度能从 lm_loss 一路回传到 needle 被吸收的那一刻（见 TrainConfig.truncate_bptt）。
    """
    ctx_ids = tok(sample.context, return_tensors="pt", add_special_tokens=True).input_ids
    if max_context and ctx_ids.shape[1] > max_context:
        ctx_ids = ctx_ids[:, :max_context]
    q_text = f"\n\n[QUERY] {sample.question}\n[ANSWER] "
    q_ids = tok(q_text, return_tensors="pt", add_special_tokens=False).input_ids
    a_ids = tok(sample.answer, return_tensors="pt", add_special_tokens=False).input_ids
    return ctx_ids.to(device), q_ids.to(device), a_ids.to(device)


def forward_loss(model, mem, ctx_ids, q_ids, a_ids):
    """预填 context → 在 query+answer 上前向 → 只在 answer token 上算 CE。"""
    cache = mem.prefill(model, ctx_ids)

    tail = torch.cat([q_ids, a_ids], dim=1)
    B, T = tail.shape
    kv_len = mem.n_mem + mem.n_seen
    position_ids = torch.arange(
        mem.n_seen, mem.n_seen + T, device=tail.device
    ).unsqueeze(0).expand(B, T)
    attention_mask = torch.ones(B, kv_len + T, device=tail.device, dtype=torch.long)

    out = model(
        input_ids=tail, past_key_values=cache, position_ids=position_ids,
        attention_mask=attention_mask, use_cache=True,
    )
    logits = out.logits[:, :-1]
    target = tail[:, 1:]
    n_ans = a_ids.shape[1]
    # 只保留 answer 段（最后 n_ans 个预测位）
    logits = logits[:, -n_ans:].float()
    target = target[:, -n_ans:]
    lm_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1)
    )
    return lm_loss


def train(cfg: Config, tier: int, out_dir: str):
    torch.manual_seed(cfg.train.seed)
    cfg = cfg.ablation(tier)
    model, tok, mem = build(cfg)

    params = mem.trainable_parameters()
    if not params:
        print(f"[tier {tier}] 无可训练参数（discard 或 training-free 的 moment 档）→ 跳过训练。")
        return
    n_param = sum(p.numel() for p in params)
    print(f"[tier {tier}] evict={cfg.cache.evict_policy} absorb={cfg.cache.absorb_mode} "
          f"可训练参数 {n_param/1e6:.2f}M")

    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.train.lr, total_steps=cfg.train.max_steps,
        pct_start=cfg.train.warmup_steps / max(cfg.train.max_steps, 1),
    )

    train_set = stage1_data.load(str(Path(__file__).parent.parent / "stage1/train.jsonl"))
    print(f"训练样本 {len(train_set)}")
    if cfg.train.truncate_bptt > 0:
        span = cfg.train.truncate_bptt * cfg.cache.prefill_chunk
        print(f"[warn] truncate_bptt={cfg.train.truncate_bptt} → 梯度只回传约 {span} token。"
              f" stage1 的 needle 在上下文最前面，其吸收过程将收不到梯度。")
    if cfg.train.max_train_context:
        print(f"训练上下文截断到 {cfg.train.max_train_context} token"
              f"（评测用完整长度；修好 RoPE 后记忆位置无关，短训长推成立）")

    step = 0
    n_skipped = 0
    opt.zero_grad()
    while step < cfg.train.max_steps:
        for sample in train_set:
            if step >= cfg.train.max_steps:
                break
            ctx, q, a = encode_sample(
                tok, sample, cfg.device, cfg.train.max_train_context
            )
            # 上下文短于预算就不会触发驱逐，记忆全程不参与，loss 连 grad_fn 都没有
            # （stage1 里 n_distract=0 的样本只有 109 token，占数据的 1/4）。
            # 这类样本对训练记忆零贡献，直接跳过；评测时仍保留，
            # 用来检查「记忆不应损害本来就放得下的短上下文」。
            min_len = cfg.cache.budget + cfg.cache.n_sink
            if ctx.shape[1] <= min_len:
                n_skipped += 1
                continue
            lm_loss = forward_loss(model, mem, ctx, q, a)

            aux = mem.collect_aux_loss()
            loss = lm_loss
            if "free_energy" in aux:
                loss = loss + cfg.train.free_energy_weight * aux["free_energy"]
            if "predictor" in aux:
                loss = loss + cfg.free_energy.predictor_loss_weight * aux["predictor"]

            (loss / cfg.train.grad_accum).backward()

            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()

            if step % cfg.train.log_every == 0:
                msg = f"step {step:5d}  lm {lm_loss.item():.4f}"
                for k, v in aux.items():
                    msg += f"  {k} {v.item():.4f}"
                print(msg, flush=True)
            step += 1

    os.makedirs(out_dir, exist_ok=True)
    ckpt = Path(out_dir) / f"tier{tier}.pt"
    torch.save({"memory": mem.state_dict(), "tier": tier}, ckpt)
    print(f"saved {ckpt}")
    if n_skipped:
        print(f"（跳过 {n_skipped} 个短于预算 {cfg.cache.budget} 的样本 —— 它们不触发驱逐，训不到记忆）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=5, help="§11.7 消融档位 1..5（5=VariKV）")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", type=str, default="varikv/ckpt")
    args = ap.parse_args()

    cfg = Config()
    if args.model:
        cfg.model_name = args.model
    if args.budget:
        cfg.cache.budget = args.budget
    if args.steps:
        cfg.train.max_steps = args.steps
    train(cfg, args.tier, args.out)
