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

import numpy as np
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
    # **真正冻结 backbone。** 原先只有 model.eval()，参数仍是 requires_grad=True，
    # 于是目标段前向会为整个 7B 计算参数梯度（bf16 下 .grad 缓冲约 15 GB），
    # 而 optimizer 只管 mem.parameters() => 这些梯度既没人用、也不会被 zero_grad
    # 清掉。数学上不影响结果（没人读它们），但白烧显存，且与论文 frozen backbone
    # 的说法不符。
    for q in m.model.parameters():
        q.requires_grad_(False)
    for p in mem.parameters():
        p.requires_grad_(True)
    assert not any(q.requires_grad for q in m.model.parameters()), "backbone 未冻结"
    assert any(p.requires_grad for p in mem.parameters()), "记忆参数未开梯度"
    m.varikv_memory = mem
    m.varikv_M = args.num_slots
    m.varikv_train = True                      # 绕开 prefill 的 inference_mode
    rot = getattr(m.model.model, "rotary_emb", None)
    m.varikv_inv_freq = rot.inv_freq.detach().clone() if rot else None
    m.varikv_detach_readback = not args.residual   # 残差模式记忆不进 cache
    m.varikv_residual = args.residual
    m.varikv_detach_every = args.detach_every
    if args.residual and args.init_ckpt:
        sd = torch.load(args.init_ckpt, map_location=m.device)["memory"]
        miss = mem.load_state_dict(sd, strict=False)
        print(f"[init] 从 {args.init_ckpt} 热启动，新增键 {list(miss.missing_keys)}")
    return m, mem, cfg, L * H



KL_DOC = """答案端多位置 teacher KL（--obj kl）

**它回答的问题：当初 `lm` 目标的失败，有多少要归给目标本身？**（GPT 2026-08-12 的
"平反实验"。架构一个字不改，只换监督。）

原来的 `lm` 目标是对文档**最后 128 个 token** 做 CE。问题不在 CE 而在位置：
局部窗口 (window_size) 是被强制保留的，而紧邻目标的上下文正是最强的预测因子，
所以最后 128 个 token 基本不需要远处内容 —— 记忆对损失近乎无关，
optimizer 最省事的解就是把门关掉。实测 σ(gate)=0.014，低于 0.018 的初值。

teacher KL 换成的目标是：

    L = Σ_t w_t · KL( p_F(·|x_<t) ‖ p_V(·|x_<t) )

p_F 是**满缓存**的同一个冻结模型，p_V 是 FastKVzip 剪枝 + 记忆。这与研究目标
直接对齐 —— 我们要补偿的正是"驱逐让冻结 LLM 的行为发生了什么变化"，
而不是让 0.33M 参数去学 FineWeb 的语言建模。

**本实现额外做一件 GPT 没提、但决定这个实验有没有意义的事：同时前向一份
"剪枝但不带记忆"的参照 p_P，于是每步都能报出**

    gap_t  = KL(p_F ‖ p_P)          驱逐真正造成的分布损伤（记忆的靶子）
    resid_t= KL(p_F ‖ p_V)          记忆之后还剩多少
    recov  = 1 − Σw·resid / Σw·gap  **记忆补回了百分之几**

理由：`gap` 目标那次的教训是"loss 0.003"看着收敛、其实只比平凡解好 10–15%
（P0_FINDINGS §4）。KL 损失有同样的陷阱，**必须把靶子的大小并排报出来**。
而且 `gap_t` 本身就是判据：若 fineweb 文档尾部的 gap ≈ 0，那么
**这份数据上任何目标都不可能有信号**，10 步就能知道，不必跑 1500 步。

位置加权（`--kl_weight`）也来自同一诊断：`sensitive` 让权重 ∝ gap_t，
把监督压到驱逐真正破坏了预测的位置上，这正是原目标缺的东西。
"""


def _kl_rows(pF_log, q_log):
    """逐位置 KL(p_F‖q)，两个输入都是 log-softmax。返回 [T]。"""
    pF = pF_log.exp()
    return (pF * (pF_log - q_log)).sum(-1)


@torch.no_grad()
def _logprobs(m, ctx_t, tgt_t, chunk=None, ratio=None, window=None, level=None,
              absorb=None):
    """前向一段目标，返回 log_softmax [T-1, V]（预测 tgt[1:]）。

    ratio=None ⇒ 满缓存（teacher）；否则分块剪枝预填。
    absorb=False ⇒ 剪枝但**不吸收**（参照 p_P，用来量驱逐造成的损伤）。
    """
    if ratio is None:
        kv = m.prefill(ctx_t, do_score=False)
    else:
        if absorb is not None:
            m.varikv_absorb_override = absorb
        kv = m.prefill(ctx_t, prefill_chunk_size=chunk, do_score=True,
                       chunk_ratio=ratio, window_size=window, level=level)
        if absorb is not None:
            m.varikv_absorb_override = None
    lg = m.model(tgt_t, past_key_values=kv).logits[:, :-1].float()
    out = torch.log_softmax(lg[0], -1)
    del kv, lg
    torch.cuda.empty_cache()
    return out


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
    ap.add_argument("--obj", default="lm", choices=["lm", "gap", "kl"],
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
    # --- 答案端/多位置 teacher KL（2026-08-12 新增，见下方 KL_DOC） ---
    ap.add_argument("--kl_weight", default="sensitive",
                    choices=["uniform", "sensitive", "top"],
                    help="按位置加权：uniform 全等权；sensitive 权重 ∝ 驱逐造成的 "
                         "KL(p_F‖p_P)；top 只用该量最大的一半位置")
    ap.add_argument("--kl_tau", type=float, default=1.0,
                    help="sensitive 权重的温度：w ∝ d_t^tau")
    # default=None 而不是 "tail"：help 说 kl 默认 random，但默认值写的是 tail，
    # 忘记显式传参就会静默退回“永远训文档尾部”。改成按 obj 解析并打印出来。
    ap.add_argument("--ctx_pos", default=None, choices=["tail", "random"],
                    help="上下文窗口取文档尾部还是随机位置。原 lm 目标只取尾部，"
                         "监督位置因此毫无多样性；未指定时 kl 取 random、其余取 tail")
    ap.add_argument("--seed", type=int, default=42,
                    help="采样计划的种子。dist/point 两个 run 必须看到逐字相同的"
                         "(文档, 起点, ratio) 序列，否则 ablation 混入采样噪声")
    ap.add_argument("--detach_every", type=int, default=1,
                    help="每 N 个 chunk 截断一次记忆递归。1 = 旧行为（只有最后一个 "
                         "chunk 的 absorb 图连着 encoder）")
    # **默认必须是 0 = 真正不过滤。** 曾把默认设成 1，看着像"不过滤"，实际是
    # "上下文至少 1 个 chunk = 16000 token"，把 34 篇里的 20 篇短文档全滤掉了
    # （只剩 14 篇），于是 v2a 不是 v1+修复、而是 v1+修复+文档子集，
    # 复现性判据被这个默认值悄悄破坏。
    ap.add_argument("--min_chunks", type=int, default=0,
                    help="要求上下文至少能切出 N 个 chunk（0 = 不过滤）。实测 "
                         "max_ctx=32768/chunk=16000 恒定只有 1 次驱逐 => 流式长程"
                         "记忆没被测到；要测它需要 --min_chunks 4 --n_short 0 --n_long 10")
    ap.add_argument("--val_windows", type=int, default=8,
                    help="固定验证窗口数。gap_v 只在开训前算一次，之后每 --val_every "
                         "步只算 resid_v => 恢复率曲线不再被在线采样噪声淹没")
    ap.add_argument("--val_every", type=int, default=100)
    # 语料取用篇数。默认 29/5 = FastKVzip feature.py 写死的组成（论文 A.1）。
    # 实测：fineweb_10k 的 68 篇**全部 <32,256 token**，所以在 chunk=16000 下
    # 「一次以上的驱逐」只能来自 fineweb_10k_cat（10 篇，103k–122k）。
    # 要训练流式记忆就必须 --n_short 0 --n_long 10。
    ap.add_argument("--n_short", type=int, default=29)
    ap.add_argument("--n_long", type=int, default=5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate_lr", type=float, default=0.02,
                    help="残差门的学习率，须远大于主 lr，见 opt 处说明")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--out", default="varikv/ckpt_stage2b")
    ap.add_argument("--probe", action="store_true", help="只跑几步测显存")
    args = ap.parse_args()

    if args.ctx_pos is None:
        args.ctx_pos = "random" if args.obj == "kl" else "tail"
        print(f"[cfg] --ctx_pos 未指定 => 按 obj={args.obj} 取 {args.ctx_pos}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

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
    for src_name, n_take in (("fineweb_10k", args.n_short),
                             ("fineweb_10k_cat", args.n_long)):
        if n_take <= 0:
            continue
        for d in load_fineweb(src_name)[:n_take]:
            ids = m.encode(d["context"])[0].tolist()   # 与评测同口径 add_special_tokens=False
            train.append(realdata.RealSample(ids=ids, n_tokens=len(ids)))
    print(f"训练文档 {len(train)} 篇，共 {sum(x.n_tokens for x in train)/1e6:.2f}M token "
          f"(长度 {min(x.n_tokens for x in train)}-{max(x.n_tokens for x in train)})")

    # **预生成采样计划**，而不是训练循环里现场 random。
    # 原因：dist 与 point 是两个独立进程，各自的 random 状态不同，会抽到不同的
    # (文档, 起点)，于是 "dist vs point" 的差里混着采样噪声。用同一个 seed
    # 预生成同一份计划，两个 run 就看到逐字相同的数据。
    need = args.max_ctx + args.target_len
    rng = random.Random(args.seed)
    n_steps = 5 if args.probe else args.steps
    # 按 min_chunks 过滤：上下文至少要能切出 N 个 chunk，否则驱逐次数太少
    need_ctx = max(args.min_chunks * args.chunk, 1)
    pool = [i for i, s in enumerate(train) if s.n_tokens >= need_ctx + args.target_len]
    if not pool:
        raise SystemExit(f"没有文档长于 {need_ctx + args.target_len} token；"
                         f"降低 --min_chunks 或 --chunk")
    print(f"[计划] min_chunks={args.min_chunks} => 可用文档 {len(pool)}/{len(train)} 篇"
          f"（需 ≥{need_ctx + args.target_len} token）")
    sched = []
    for k in range(n_steps):
        di = pool[k % len(pool)]
        L_doc = train[di].n_tokens
        want = min(args.max_ctx, L_doc - args.target_len)
        if args.ctx_pos == "random" and L_doc > want + args.target_len:
            a = rng.randrange(0, L_doc - want - args.target_len)
        else:
            a = L_doc - want - args.target_len
        r = (rng.choice(args.ratio_choices) if args.ratio_mode == "random"
             else args.ratio)
        sched.append((di, a, want, r))
    # 固定验证窗口：gap_v 只算一次，之后每 val_every 步只算 resid_v
    vrng = random.Random(args.seed + 1)
    val_sched = []
    for _ in range(args.val_windows):
        di = pool[vrng.randrange(len(pool))]
        L_doc = train[di].n_tokens
        want = min(args.max_ctx, L_doc - args.target_len)
        a = (vrng.randrange(0, L_doc - want - args.target_len)
             if L_doc > want + args.target_len else 0)
        val_sched.append((di, a, want, args.ratio))

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

    val_gap = None
    while step < n_steps:
        for _plan in [None]:      # 单次迭代的壳子：保留原缩进，避免大范围重排
            di, a, want, cur_ratio = sched[step]
            ids = train[di].ids
            ctx_ids = ids[a:a + want]
            tgt = ids[a + want:a + want + args.target_len]
            ctx_t = torch.tensor([ctx_ids], device=m.device)
            tgt_t = torch.tensor([tgt], device=m.device)

            # ---- 固定验证窗口上的恢复率（去掉在线采样噪声）----
            if args.obj == "kl" and (step % args.val_every == 0):
                if val_gap is None:
                    val_gap = []
                    _kt0 = m.kv_type
                    m.kv_type = "retain"
                    for (vd, va, vw, vr) in val_sched:
                        vi = train[vd].ids
                        c = torch.tensor([vi[va:va + vw]], device=m.device)
                        g = torch.tensor([vi[va + vw:va + vw + args.target_len]],
                                         device=m.device)
                        pf = _logprobs(m, c, g)
                        pp = _logprobs(m, c, g, args.chunk, vr, args.window, args.level)
                        val_gap.append((c.cpu(), g.cpu(), pf.cpu(),
                                        float(_kl_rows(pf, pp).mean())))
                        del pf, pp
                    m.kv_type = _kt0
                    print(f"[val] {len(val_gap)} 个固定窗口，"
                          f"gap 均值 {np.mean([x[3] for x in val_gap]):.4f}", flush=True)
                rs, gs = [], []
                for (c, g, pf, gp) in val_gap:
                    with torch.no_grad():
                        kvv = m.prefill(c.to(m.device), prefill_chunk_size=args.chunk,
                                        do_score=True, chunk_ratio=args.ratio,
                                        window_size=args.window, level=args.level)
                        lg = m.model(g.to(m.device),
                                     past_key_values=kvv).logits[0, :-1].float()
                    rs.append(float(_kl_rows(pf.to(m.device),
                                             torch.log_softmax(lg, -1)).mean()))
                    gs.append(gp)
                    del kvv, lg
                    torch.cuda.empty_cache()
                print(f"[val] step {step:4d}  gap {np.mean(gs):.4f}  "
                      f"resid {np.mean(rs):.4f}  "
                      f"**恢复 {100*(1-np.sum(rs)/max(np.sum(gs),1e-9)):+.1f}%**",
                      flush=True)
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
                if args.obj == "kl":
                    # ---- teacher（满缓存）与参照（剪枝但无记忆），都不带梯度 ----
                    # 用 kv_type="retain" 拿 p_P：那就是评测基线本身，比加一个
                    # absorb 开关更少假设。
                    _kt = m.kv_type
                    m.kv_type = "retain"
                    pF = _logprobs(m, ctx_t, tgt_t)                    # 满缓存
                    pP = _logprobs(m, ctx_t, tgt_t, args.chunk, cur_ratio,
                                   args.window, args.level)            # 剪枝无记忆
                    m.kv_type = _kt
                    gap = _kl_rows(pF, pP)                             # [T-1]
                    if args.kl_weight == "uniform":
                        w = torch.ones_like(gap)
                    elif args.kl_weight == "top":
                        w = (gap >= gap.median()).float()
                    else:
                        w = gap.clamp_min(0).pow(args.kl_tau)
                    w = w / w.sum().clamp_min(1e-9)
                    # ---- student（带记忆，带梯度）----
                    out = m.model(tgt_t, past_key_values=kv)
                    qlog = torch.log_softmax(out.logits[0, :-1].float(), -1)
                    resid = _kl_rows(pF, qlog)
                    loss = (w * resid).sum()
                    logits = None
                    g_w = float((w * gap).sum())
                    r_w = float((w * resid).sum())
                    # 也报 uniform：**不能只用训练所用的同一套权重给自己打分**，
                    # 否则 sensitive 加权很容易只是在当前窗口上过拟合。
                    g_u = float(gap.mean()); r_u = float(resid.mean())
                    nch = kv.stats["calls"] / max(kv.n_layers, 1)
                    extra = (f" gap {g_w:.4f} resid {r_w:.4f} "
                             f"recov {100*(1-r_w/max(g_w,1e-9)):+.1f}% "
                             f"(unif {100*(1-r_u/max(g_u,1e-9)):+.1f}%) "
                             f"ch {nch:.0f}")
                    del pF, pP, qlog, resid, gap, w
                elif args.obj == "gap":
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
                    extra = ""
                else:
                    out = m.model(tgt_t, past_key_values=kv)
                    logits = out.logits[:, :-1].float()
                    loss = lossf(logits.reshape(-1, logits.size(-1)),
                                 tgt_t[:, 1:].reshape(-1))
                    extra = ""
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
                print(f"step {step:4d} {args.obj} {loss.item():.4f} |g|{g:.1e} "
                      f"{peak:.0f}GB {(time.time()-t0)/(step+1):.1f}s"
                      + extra + _grad_report(mem)
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
                "num_slots": args.num_slots, "model": args.model,
                # 训练配置必须随 ckpt 存：旧 ckpt 只有前四项，max_ctx/lr/steps/obj
                # 都只能从日志副作用反推（CLAUDE.md 记过这个缺口）
                "args": vars(args)}, ck)
    print(f"saved {ck}")
    if args.ratio_mode == "random":
        print(f"[ratio 分布] {dict(sorted(ratio_hist.items()))}")


if __name__ == "__main__":
    main()
