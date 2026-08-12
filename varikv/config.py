"""VariKV 的统一配置。

方法定位见 theory_distributional_memory.md §11「选项 3：自由能统一驱逐」：
一个自由能标量 F_i = D_i + λ·KL_i 同时决定
  决策 A：哪个 KV 降级进记忆（按 F 排序，留精确 top-B）
  决策 B：降级后写入多少（同一个 KL_i 导出写入率）

四档消融（§11.7）由下面两个正交开关组合而成，不是四套代码：
    evict_policy   absorb_mode    对应已有方法
    ----------------------------------------------------
    -              discard        KVzip / FastKVzip
    recency        point          Infini-attention / Tensor Cache
    recency        moment         MomentKV（training-free 二阶矩）← 真实门槛
    free_energy    point          IndexMem 加强
    free_energy    dist           本方法
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class MemoryConfig:
    """记忆模块本身的结构超参。"""

    # --- 槽 ---
    num_slots: int = 16            # K：每个 (layer, kv_head) 的槽数
    d_latent: int = 64             # 槽存的是 latent 高斯的维度，不是 KV 维度
    # 初始 log σ²。必须是「不确定」（≥0）：空记忆还没吸收任何东西，
    # 理应方差大、容易被写入；随着吸收精度累加才逐渐变确定、变得抗覆盖。
    # （视频版用的 -2.0 在这里是错的：那等于让空记忆一开始就很笃定，
    #   既抗拒第一次写入，又让初始 KL 被 1/σ_p² 放大到爆炸。）
    logvar_init: float = 0.0
    # 精度遗忘因子 γ：τ_new = γ·τ_old + Σᵢ ηᵢ·τ_obs。
    # 没有它，流式吸收会让精度无界累加 → 记忆过度自信 → τ_old 大到 μ 再也不更新，
    # 等于彻底拒绝新信息（对 stage1 的 update 型样本必然失败）。
    # γ<1 给旧证据打折，是贝叶斯滤波应对非平稳环境的标准手段，
    # 也对应认知科学的记忆衰减（theory §10.2）。
    precision_decay: float = 0.95
    # 单次 absorb 对**单个槽**的精度贡献上限（有效样本量 / effective sample size）。
    #
    # 不设上限就会踩「伪重复计数」：一个 chunk 里被驱逐的几百个相邻 token 的 KV
    # 高度相关，把它们当成几百个独立观测会严重高估精度。实测每槽单次就累到 ≈8，
    # 稳态 τ=8/(1−γ)=160，于是 1/τ≈0.006、方差被压到 clamp 下界，
    # 训练 60 步后 81~98% 的槽 logvar 焊死在 −4，方差彻底失去动态范围。
    # 取 1.0 表示「一次吸收最多相当于一个独立观测」，稳态 τ≈1/(1−γ)=20，
    # 对应 logvar≈−3，留出了动态范围。缩放同时作用于一阶、二阶矩，故均值不变。
    max_eff_obs: float = 1.0
    # clamp 下界同时是精度硬上界 τ_max = e^{-logvar_min}。
    # 取 -4 → τ_max≈55，即「最多相当于 55 个观测的确信度」，
    # 把 KL 中 1/σ_p² 的放大倍数限制在可控范围（§11.4.3 数值稳定）。
    logvar_min: float = -4.0
    logvar_max: float = 4.0

    # --- 编码器 / 解码器 ---
    d_hidden: int = 256
    # 缺口②：decoder 必须非线性，否则整个模型塌回线性高斯（卡尔曼可解），
    # 摊销识别网络就是多余的，理论卖点消失。
    nonlinear_decoder: bool = True

    # --- 先验 ---
    # 缺口①：get_prior 必须保留混合（逐 slot KL 按 w_k 加权求和），
    # 不能先把 K 个高斯平均成一个再算 KL —— 那会丢掉多峰性。
    keep_mixture_prior: bool = True
    prior_temperature: float = 1.0  # 相似度 → w_k 的 softmax 温度
    # 缺口 B5：用 data-dependent 的 responsibility w_k 时，完整的混合模型 ELBO
    # 还含一项「选哪个组件」的编码成本：
    #     KL(q(c|e) ‖ p(c)) = Σ_k w_k·log(w_k·K) = log K − H(w)
    # 只算 Σ_k w_k·KL(q‖p_k) 相当于默认组件分配免费，ELBO 不完整。
    # 率失真视角下这一项就是码率的一部分：把证据归到某个具体槽本身要花比特，
    # 分配越尖锐花得越多。留开关是因为它会改变行为，需要能做对照消融。
    include_assignment_kl: bool = True

    # --- 写入门控（决策 B）---
    # 分配与强度解耦：gate_ik = w_ik · η_i，其中 η_i = sigmoid(α·zscore(KL_i) − β)。
    # 于是 Σ_k gate_ik = η_i ≤ 1，「一个观测总共写入多少」有明确的概率解释，
    # 「独立观测的精度可加」这一贝叶斯前提才成立（详见 memory.absorb）。
    # KL 取 **chunk 内 z-score** 而非绝对值：它跨记忆演化会横跨约 4 个数量级，
    # 任何固定的 (α, β) 都会让 sigmoid 两头饱和。故 β 的含义是「相对 surprise 阈值」，
    # β=0 即平均水平开始写入；旧的 β=2 是配绝对 KL 的，在 z-score 下会把门控压死。
    eta_alpha: float = 2.0
    eta_beta: float = 0.0

    # --- 读出 ---
    # 缺口③：方差必须进入读出，否则不确定性在数学上空转，
    # 「分布式」就退化成「点记忆 + 一个没用的 σ 缓冲区」，实验会假阴性。
    sample_on_read: bool = True     # 重参数化采样 z ~ N(μ, σ²)
    # 槽 anchor 的初始化。"random" = 原行为（每个 seed 一套随机码本，且它学不动）；
    # "ortho" = 确定性近正交码本，跨 seed 完全一致，用来隔离"随机码本"这个方差源。
    slot_init: str = "random"
    logvar_into_decoder: bool = True  # 把 logvar 作为 decoder 的显式输入
    tokens_per_slot: int = 1        # 每个槽读出几个 effective KV


@dataclass
class FreeEnergyConfig:
    """自由能标量 F_i = D_i + λ·KL_i 及其摊销预测器。"""

    # λ：惊讶项相对失真项的权重 —— 即率失真拉格朗日乘子，扫它就得到 R-D 曲线上的
    # 不同工作点，本身就该在论文里做敏感性分析。
    # 实测（running 归一化，mom=0.9，记忆演化稳定后，两项对排序的秩相关）：
    #   λ=0   → 1.000 / 0.056   纯失真，精确退化成 Expected Attention（§11.3 第 1 行）
    #   λ=0.3 → 0.602 / 0.718   两项大致均衡  ← 默认取此
    #   λ=1   → 0.399 / 0.885
    #   λ=3   → 0.288 / 0.950   KL 主导
    lam: float = 0.3

    # F 的两项如何归一化到可比尺度：
    #   "running"（默认）—— 先做量纲归一（D 除以 E‖v‖² 成为无量纲相对失真，并用
    #       (N·ā)² 代替 ā² 消去序列长度依赖；KL 取 per-dim），再各自除以自身的
    #       **running 标准差**。running 统计是数据集级的量、不随 chunk 组成变化，
    #       因此 F_i 仍只是 KV_i 与当前记忆状态的函数，λ 保住率失真拉格朗日乘子
    #       「每 nat 码率值多少相对失真」的含义（theory §9.1-9.2），且 λ=1 恰为等权。
    #       必须除 std 而非只做量纲归一：决定排序的是离散度不是均值，实测
    #       std(D_n)≈0.69 恒定而 std(KL_n) 随记忆演化从 2e-4 长到 7e-2，
    #       只做量纲归一的话 F 的排序 99% 由 D 决定（F~KL 秩相关仅 0.09），
    #       KL 项形同虚设 —— 那正是 §11.3 退化表第一行的 Expected Attention。
    #   "zscore" —— 各自做 chunk 内 z-score。数值最稳，但 F_i 变成依赖同批其他 token
    #       的量，不再是 KV_i 的函数，λ 也退化为「两个标准分的相对权重」，
    #       率失真的解释随之失效。仅在 running 数值不稳时退回。
    f_normalize: str = "running"
    # 0.9 是折中：更大(0.95)则 running std 追不上 KL 演化、KL 项过度主导；
    # 更小(0.5)最平衡但几乎等同于用当前批统计，F_i 的绝对语义随之削弱。
    v_scale_momentum: float = 0.9

    # 未来 query 分布 p(q) 的在线估计（Expected-Attention 式）。
    # 注意：若改用「已实现注意力」代替这个期望，方法就退化成 H2O/SnapKV
    # （§11.3 退化表第二行），失去原创性 —— 所以这里必须是对 q 分布取期望。
    query_stat_momentum: float = 0.95
    query_stat_eps: float = 1e-6

    # 摊销 F 预测器（§11.4.1）：精确 F_i 每 token 太贵，
    # 训一个轻量网络预测它，完全类比 FastKVzip 用 gate 蒸馏 KVzip 分数。
    # 好处还包括：离散 top-k 不阻断梯度（预测器靠蒸馏训，梯度不穿过驱逐）。
    use_amortized_predictor: bool = True
    predictor_hidden: int = 128
    predictor_loss_weight: float = 1.0
    # 训练早期用精确 F 驱逐、同时蒸馏预测器；之后切换到预测器驱逐。
    exact_f_warmup_steps: int = 200


@dataclass
class CacheConfig:
    """KV cache 的驱逐 / 吸收 / 读出策略。"""

    evict_policy: Literal["recency", "free_energy"] = "free_energy"
    absorb_mode: Literal["discard", "point", "moment", "dist"] = "dist"

    budget: int = 512               # B：保留精确 KV 的数量
    local_window: int = 128         # 最近 window 内的 KV 永不驱逐（sink 式保护）
    n_sink: int = 4                 # 最前面 n 个 token 永不驱逐（attention sink）
    prefill_chunk: int = 512        # 分块预填的块大小

    # 记忆读出的 effective KV 是否参与注意力（关掉可做「只写不读」的对照）
    read_memory: bool = True


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 1
    grad_accum: int = 8
    max_steps: int = 2000
    warmup_steps: int = 100
    # loss = lm_loss + free_energy_weight·F + predictor_loss_weight·distill
    free_energy_weight: float = 0.01
    grad_clip: float = 1.0
    # 每 N 个 chunk 截断一次 BPTT；**0 = 不截断**（默认）。
    #
    # 关于梯度可达性，实测结论（2026-08-03，与直觉相反，勿凭想当然改回去）：
    # lm_loss 回传到记忆的**唯一路径**是「读出的 effective KV 被写进 cache、
    # 参与其后每一次前向」。把 read_memory 关掉，lm_loss 直接没有 grad_fn。
    # 这条路径不经过 self.mu 的跨 chunk 递归，所以 detach_state() **切不断它** ——
    # 即便 truncate_bptt=2，第 0 个 chunk（needle 所在处）吸收后的 mu 依然拿得到梯度。
    # 因此截断并不会像预想那样让 needle 学不到；它只影响 self.mu 递归本身的可训性。
    #
    # 默认仍取 0：递归可训是「记忆是递归状态而非参数」这一设计的一部分。
    # 显存不靠截断梯度来控，而靠 max_train_context —— 实测 13.8k 上下文不截断会
    # OOM（约 57GB），4k 则安全。
    truncate_bptt: int = 0
    # 训练时把上下文截到这个长度（0 = 不截）。评测仍用完整长度。
    # 修好 RoPE 后记忆是位置无关的，所以短训长推是成立的
    # （Infini-attention 等长上下文记忆方法的标准做法）。
    max_train_context: int = 4096
    log_every: int = 10
    eval_every: int = 200
    seed: int = 0


@dataclass
class Config:
    # 阶段 1 用小模型快速迭代；结论确认后换 Qwen3-8B 复核。
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "bfloat16"
    device: str = "cuda"

    # 记忆接在哪些层。None = 全部层。
    # 参数跨层共享（HANDOFF 红线 2「模块必须轻量」），但每层有独立的槽状态。
    layers: Optional[list] = None
    share_across_layers: bool = True

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    free_energy: FreeEnergyConfig = field(default_factory=FreeEnergyConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def ablation(self, tier: int) -> "Config":
        """返回 §11.7 杀手对比中的第 tier 档（1..5）的配置副本。

        tier 3 是 2026-08-03 新增的 MomentKV 式 training-free 二阶矩 baseline ——
        文献检索表明「打赢点均值记忆」已不构成贡献，真实门槛是打赢它。
        """
        import copy

        c = copy.deepcopy(self)
        settings = {
            # 注意：这是「滑动窗口 + 丢弃」，**不是 KVzip**。
            # 曾把这里注释成「丢弃：驱逐策略无关」，那是错的 —— 什么都不吸收时，
            # 保留哪些 KV 恰恰是唯一决定性能的因素，不是无关而是唯一变量。
            # 真正的 KVzip 用重建注意力打分，比 recency 强得多。
            # 因此「Δ vs tier1」量的是相对滑动窗口的差距，不能当作赢过 KVzip。
            1: ("recency", "discard"),
            2: ("recency", "point"),
            3: ("recency", "moment"),       # MomentKV 式，training-free
            4: ("free_energy", "point"),
            5: ("free_energy", "dist"),
        }
        if tier not in settings:
            raise ValueError(f"tier must be 1..5, got {tier}")
        c.cache.evict_policy, c.cache.absorb_mode = settings[tier]
        return c
