"""Stage 2b 训练：在 Fast KVzip 管线内训练 VariKV 记忆（目标模型 7B）。

与 stage2a 的关键区别：训练走的就是**评测时的同一条路径**（真实门控的 per-head
驱逐 + MemoryEvictCache），所以没有 train/test 布局失配。

显存是主要约束：LLM 冻结，但记忆读出的 effective KV 参与后续每一次前向，
梯度要穿过这些前向，所以激活仍需保留。用两个旋钮压住：
  --max_ctx      训练上下文长度（短训长推，RoPE 修好后成立，见 CLAUDE.md）
  --detach_every 每 N 个 chunk 截断一次记忆递归（truncate BPTT）
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip          # noqa: E402
from varikv.config import Config              # noqa: E402
from varikv.memory import DistributionalMemory  # noqa: E402
from varikv import realdata                   # noqa: E402


def build(args):
    # 默认 memory_retain：Figure 11 的基线全部跑在 kv_type="retain" 上，
    # 评测端已对齐到它，训练端也必须同机制 —— 两种机制吸收的 KV 集合虽经实测
    # 逐位相同（2,075,565 条，记忆状态差 1e-3），但读回后与注意力的交互不同，
    # 会经由 hidden states → 门控分数 → 驱逐决策形成反馈回路，不宜假定等价。
    m = ModelKVzip(args.model, kv_type=args.kv_type, gate_path_or_name=args.gate)
    cfg = Config()
    cfg.memory.num_slots = args.num_slots
    H = m.config.num_key_value_heads
    L = m.config.num_hidden_layers
    hd = getattr(m.config, "head_dim",
                 m.config.hidden_size // m.config.num_attention_heads)
    mem = DistributionalMemory(2 * hd, cfg.memory, mode=args.mode,
                               n_groups=(L * H) if args.residual else 0).to(
        m.device, dtype=torch.float32          # 记忆用 fp32：bf16 下精度累加有
    )                                          # 5.98% 相对误差（CLAUDE.md）
    mem.reset(1, L * H, device=m.device, dtype=torch.float32)
    for p in mem.parameters():
        p.requires_grad_(True)
    m.varikv_memory = mem
    m.varikv_M = args.num_slots
    m.varikv_train = True                      # 绕开 prefill 的 inference_mode
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone() if rot else None
    m.varikv_detach_readback = not args.residual   # 残差模式记忆不进 cache
    m.varikv_residual = args.residual
    if args.residual and args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=m.device)["memory"]
        miss = mem.load_state_dict(sd, strict=False)
        print(f"[init] 从 {args.init_ckpt} 热启动，新增键 {list(miss.missing_keys)}")
    return m, mem, cfg, L * H



def _grad_report(mem):
    """分模块梯度统计 + 门的分布。

    今天三次踩到「loss 正常下降但某条通路梯度恒为 0」——inference_mode 截断、
    update_flatten_view 没有 backward、_swap_out 用 cat 重建时切图。
    这类失败在 loss 曲线上完全隐形，只能靠显式看梯度发现，
    所以做成训练时的常态监控，而不是事后排查。

    门的统计同样必要：残差模式下门保持关闭就等价于基线，loss 一样低，
    「loss 降了」不代表「记忆被用上了」。必须直接看 sigmoid(gate)。
    """
    import torch as _t
    groups = {
        "enc": ("encoder.", "to_mu", "to_logvar"),
        "dec": ("decoder.",),
        "slot": ("slot_mu_init", "slot_logvar_init"),
        "gate": ("residual_gate",),
        "pgate": ("point_gate_logit",),
    }
    parts, dead = [], []
    for gname, prefixes in groups.items():
        ps = [q for n, q in mem.named_parameters()
              if any(n.startswith(x) for x in prefixes)]
        if not ps:
            continue
        gmax = max((q.grad.abs().max().item() for q in ps
                    if q.grad is not None), default=0.0)
        parts.append(gname + f"{gmax:.0e}")
        if gmax == 0.0:
            dead.append(gname)
    msg = "  g[" + " ".join(parts) + "]"
    if dead:
        msg += " !零梯度:" + ",".join(dead)
    rg = getattr(mem, "residual_gate", None)
    if rg is not None:
        gg = _t.sigmoid(rg.detach()).float()
        msg += (" gate[mean%.3f max%.3f >0.1占%.2f]"
                % (gg.mean(), gg.max(), (gg > 0.1).float().mean()))
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-1M")
    ap.add_argument("--gate", default="fastkvzip")
    ap.add_argument("--obj", default="lm", choices=["lm", "gap"],
                    help="lm=语言建模CE；gap=回归注意力缺口(IndexMem式)")
    ap.add_argument("--residual", action="store_true",
                    help="输出端门控残差：记忆不进 cache")
    ap.add_argument("--init_ckpt", default="",
                    help="从已有 ckpt 热启动（残差模式下 gate 用初值）")
    ap.add_argument("--kv_type", default="memory_retain",
                    choices=["memory", "memory_retain"])
    ap.add_argument("--level", default="pair")
    ap.add_argument("--mode", default="dist", choices=["point", "dist"])
    ap.add_argument("--num_slots", type=int, default=16)
    ap.add_argument("--ratio", type=float, default=0.3,
                    help="ratio_mode=fixed 时使用的压缩比")
    # 固定比例是原始行为，保留为消融的一支：
    #   fixed  —— 只在一个压缩比上训练（此前所有轮次都是 0.3）
    #   random —— 每步随机抽一个，与评测的多比例扫描匹配
    # 之所以要 random：记忆吸收的是「被扔掉的那批 KV」，而那批是什么完全
    # 取决于压缩比 —— 0.75 时只扔最没用的 25%，0.2 时要扔掉 80%（含不少重要的）。
    # 固定 0.3 训练意味着评测的 6 个比例里有 5 个是外推；而且实测 0.3 处的
    # 注意力缺口均方只有 3.34e-03，信号本身也弱。
    # 对照：FastKVzip 的门控输出的是**比例无关**的重要性分数，比例只在推理时
    # 用来切阈值，所以它一个 ckpt 跑遍 6 个比例、不存在这个问题。
    ap.add_argument("--ratio_mode", default="fixed", choices=["fixed", "random"])
    ap.add_argument("--ratio_choices", type=float, nargs="+",
                    default=[0.75, 0.5, 0.4, 0.3, 0.2, 0.1])
    ap.add_argument("--max_ctx", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--target_len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate_lr", type=float, default=0.02,
                    help="残差门的学习率，须远大于主 lr，见 opt 处说明")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", default="varikv/ckpt_stage2b")
    ap.add_argument("--probe", action="store_true", help="只跑几步测显存")
    args = ap.parse_args()

    m, mem, cfg, G = build(args)
    params = [p for p in mem.parameters() if p.requires_grad]
    print(f"记忆可训练参数 {sum(p.numel() for p in params)/1e6:.2f}M  G={G}  "
          f"M={args.num_slots}  mode={args.mode}")

    # **直接用 FastKVzip 自己的加载器**，拿到与其门控训练逐字相同的文档。
    # 论文 §3.3 / 附录 A.1：FineWeb-Edu，10K–30K 的样本取 50 万 token，
    # 再由拼接得到的 ~100K 长样本取另外 50 万，合计 1M token。
    # 选 FineWeb-Edu 的理由论文写明：与下游评测数据集**无重叠**，
    # 所以「训练用通用语料、评测用检索任务」不是缺陷而是刻意设计。
    # 筛选确定性（np.arange + 长度过滤，无随机种子），可精确复现。
    from data.load import load_fineweb

    # 组成不是我推断的，是 prefill/feature.py 写死的：
    #     folders = [("fineweb_10k", 29), ("fineweb_10k_cat", 5)]
    # 即前 29 篇 + 前 5 篇。实测 434.9K + 547.6K = 0.98M token，
    # 与论文正文「1M training tokens」及附录「500K + 500K」一致。
    # （load_fineweb 里的 `total > 10**6` 只是加载器返回上限，
    #   训练实际只消费前 29 / 前 5 篇 —— 论文与代码并不矛盾。）
    train = []
    for src_name, n_take in (("fineweb_10k", 29), ("fineweb_10k_cat", 5)):
        for d in load_fineweb(src_name)[:n_take]:
            ids = m.encode(d["context"])[0].tolist()   # 与评测同口径 add_special_tokens=False
            train.append(realdata.RealSample(ids=ids, n_tokens=len(ids)))
    print(f"训练文档 {len(train)} 篇，共 {sum(x.n_tokens for x in train)/1e6:.2f}M token "
          f"(长度 {min(x.n_tokens for x in train)}-{max(x.n_tokens for x in train)})")

    # 门必须用单独的、大得多的学习率。
    #
    # AdamW 的步长约等于 lr（梯度被归一化），所以 lr=1e-4 跑 1500 步最多把
    # gate 的 logit 移动 0.15 —— sigmoid(-4)=0.018 到 sigmoid(-3.85)=0.021，
    # 等于门根本没开、记忆全程没被用上。实测 50 步后 sigmoid(gate) 纹丝未动。
    # FastKVzip 给它的门控用 lr=0.2（比常规大 3 个数量级）也是同一个道理：
    # 门是标量、值域被 sigmoid 压住，需要在 logit 空间上大步走。
    gate_params = [q for n, q in mem.named_parameters()
                   if n.startswith("residual_gate") and q.requires_grad]
    other_params = [q for n, q in mem.named_parameters()
                    if not n.startswith("residual_gate") and q.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr},
         {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0}],
        weight_decay=0.01)
    if gate_params:
        print(f"[opt] 门单独一组 lr={args.gate_lr}（其余 {args.lr}）")
    lossf = torch.nn.CrossEntropyLoss()
    step, t0 = 0, time.time()
    ratio_hist = {}
    peak = 0.0

    while step < (5 if args.probe else args.steps):
        for s in train:
            if step >= (5 if args.probe else args.steps):
                break
            ids = s.ids
            ctx_ids = ids[-(args.max_ctx + args.target_len):-args.target_len]
            tgt = ids[-args.target_len:]
            ctx_t = torch.tensor([ctx_ids], device=m.device)
            tgt_t = torch.tensor([tgt], device=m.device)

            if args.ratio_mode == "random":
                cur_ratio = random.choice(args.ratio_choices)
            else:
                cur_ratio = args.ratio
            mem.reset(1, G, device=m.device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.no_grad():
                    kv = m.prefill(ctx_t, prefill_chunk_size=args.chunk, do_score=True,
                                   chunk_ratio=cur_ratio, window_size=args.window,
                                   level=args.level)
                # 预填在 no_grad 下跑完（记忆状态因此也不带图），随后补一次带梯度
                # 的读回：梯度经 loss → 目标前向 → 记忆KV → decoder 回传。
                if not args.residual:
                    kv.refresh_with_grad()   # 残差模式记忆不在 cache，无需补写
                if args.obj == "gap":
                    # IndexMem 式目标：直接回归被驱逐造成的注意力缺口。
                    # 只跑 model.model（不需要 lm_head），损失由各层累加。
                    kv.collect_residual_loss = True
                    kv.residual_losses = []
                    m.model.model(tgt_t, past_key_values=kv)
                    kv.collect_residual_loss = False
                    if not kv.residual_losses:
                        step += 1
                        continue
                    loss = torch.stack(kv.residual_losses).mean()
                    out = logits = None
                else:
                    out = m.model(tgt_t, past_key_values=kv)
                    logits = out.logits[:, :-1].float()
                    loss = lossf(logits.reshape(-1, logits.size(-1)),
                                 tgt_t[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"step {step}: OOM，跳过（峰值 {torch.cuda.max_memory_allocated()/2**30:.1f} GB）")
                step += 1
                continue

            ratio_hist[cur_ratio] = ratio_hist.get(cur_ratio, 0) + 1
            peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
            if step % args.log_every == 0 or args.probe:
                g = max((p.grad.abs().max().item() for p in params if p.grad is not None),
                        default=0.0)
                print(f"step {step:4d} lm {loss.item():.4f} |g|{g:.1e} "
                      f"{peak:.0f}GB {(time.time()-t0)/(step+1):.1f}s"
                      + _grad_report(mem)
                      + (f" r{cur_ratio}" if args.ratio_mode == "random" else ""),
                      flush=True)
            step += 1
            del kv, out, logits, loss
            torch.cuda.empty_cache()

    if args.probe:
        print(f"\n探测结束：峰值显存 {peak:.1f} GB")
        return
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    ck = outd / f"s2b_{args.mode}_k{args.num_slots}.pt"
    torch.save({"memory": mem.state_dict(), "mode": args.mode,
                "num_slots": args.num_slots, "model": args.model}, ck)
    print(f"saved {ck}")
    if args.ratio_mode == "random":
        print(f"[ratio 分布] {dict(sorted(ratio_hist.items()))}")


if __name__ == "__main__":
    main()
