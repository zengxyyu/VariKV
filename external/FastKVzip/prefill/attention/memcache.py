
"""VariKV × Fast KVzip 接入（Stage 2b，2026-08-07 新增）。

本仓库的方法此前只在合成任务（stage1）和 fineweb 语言建模上跑过，从未接触真实
长上下文基准。这个文件把 `varikv` 的分布式记忆挂进 Fast KVzip 的评测管线，
从而能与已复现的 KVzip / FastKVzip / SnapKV / DuoAttn 基线用**同一份数据、
同一个指标、同一个模型**对比。

为什么继承 `EvictCache` 而不是重写：
    它已经实现了 per-head 变长布局（cache 摊平成 `[Σ_heads len_k_head, dim]`，
    边界记在 `cu_len_k`）以及 AdaKV kernel 的对接。重写等于把这些债重新欠一遍。
    这里只覆写三个方法 —— `_sample_cache`（丢弃锚点）、`_get_valid`（掩码偏移）、
    `update`（位置追踪）。

每个 head 段的不变式（记忆前缀恒在最前）：

    [ mem_0 … mem_{M-1} | real_0 … real_{L_h-1} ]

`update_flatten_view` 把新 KV 追加到各 head 段**末尾**，所以前缀不会被打散；
反过来若把记忆放段尾，下一次 update 就会插到它后面，不变式立刻破裂。

两个必须自己维护的东西：

1. **原始位置**。`_sample_cache` 会**压紧**缓存，几轮驱逐后 head 段内的下标
   不再等于 token 的原始位置。而 RoPE 逆旋转必须要绝对位置 —— 否则槽是不同
   相位 key 的加权平均，不对应任何有效位置，σ² 还会被相位噪声虚增，
   「方差编码不确定性」这个核心主张当场失效。
   （这一条读代码前没被记录，是 Stage 2b 比预想多出来的工作量。）

2. **per-head 驱逐数量不同**。各 head 扔掉的条数不一样，凑不成矩形张量，
   所以补齐到本层最大值并传 `valid` 掩码给 `absorb`。等价性已单测：
   pad 与不 pad 的 μ/logvar 差 1e-17，且与 pad 数量无关。
"""
import torch
import torch.nn.functional as F

from attention.kvcache import EvictCache


class MemoryEvictCache(EvictCache):
    """EvictCache + 分布式记忆：被驱逐的 KV 不丢弃，吸收进 (μ,σ²) 槽再读回。"""

    _STATE = ("mu", "logvar", "var_content", "pos", "_pos_tau")

    def __init__(self, model, evict_range, mem_module, rope_inv_freq=None,
                 n_mem_per_head: int = 16, absorb_enabled: bool = True):
        super().__init__(model, evict_range)
        self.mem = mem_module
        self.absorb_enabled = absorb_enabled and mem_module is not None
        self.M = n_mem_per_head if self.absorb_enabled else 0
        self.inv_freq = rope_inv_freq
        self.head_dim = model.config.hidden_size // model.config.num_attention_heads
        self.head_dim = getattr(model.config, "head_dim", self.head_dim)

        self.pos_track = [None] * self.n_layers      # 与 key_cache 同布局的原始位置
        self._seen_real = [0] * self.n_layers        # 各层已见真实 token 数
        self._mem_inserted = [False] * self.n_layers
        self.stats = {"absorbed": 0, "calls": 0, "warned_share": False}
        # 训练用：预填期间插入**detach 的**读回，使 LLM 前向完全不建图。
        # 记忆状态本身仍带梯度（那是小张量，开销可忽略），训练时在预填结束后
        # 调用 refresh_with_grad() 补一次带梯度的读回，梯度即可沿
        # loss → 目标前向 → 记忆KV → decoder → 记忆状态 → encoder 回传。
        # 不这么做的话，5 个 chunk 的 7B 前向图直接 OOM（实测 78.2 GB）。
        # 代价是丢掉「早期读回影响后续 hidden state」这个二阶效应 —— 标准的
        # 截断 BPTT 取舍，与 varikv 训练侧 truncate_bptt 的动机一致。
        self.detach_readback = False

        # 每个新 cache = 一条新序列 ⇒ 记忆状态必须归零。
        # 记忆**模块**是跨样本共享的（参数要共享），但它的**状态**(μ,σ²,pos,τ)
        # 是这条序列的。评测里每条样本都新建 cache 却复用同一个模块，不在这里
        # 重置的话，第 N 条样本的记忆里装着第 N−1 条的内容 —— 不会报错，
        # 只会让所有结果被前一条样本污染。
        if self.absorb_enabled:
            self.mem.reset(1, self.n_layers * self.n_heads_kv,
                           device=self.device,
                           dtype=next(self.mem.parameters()).dtype)


    def assert_gate_compatible(self, gates):
        """`snap` 门控与本 cache 不兼容 —— 必须显式拦住，否则是静默错位。

        原因是打分时机（attn.py）：
          gate / head  → update() **之前**，输入 hidden_states           ✓
          expect       → update() **之前**，输入本 chunk 的 k/v          ✓
          snap         → update() **之后**，输入 update() 返回的**整个扁平缓存**
                         —— 那里已经含有我们插入的记忆前缀，于是 score 的长度和
                         下标都会和真实 token 对不上，而 threshold 仍按 ctx 长度
                         切片，错位不会报错，只会让驱逐决策变成垃圾。
        """
        if gates is None:
            return
        bad = [i for i, g in enumerate(gates) if getattr(g, "name", "") == "snap"]
        if bad:
            raise NotImplementedError(
                "MemoryEvictCache 不支持 snap 门控：snap 在 update() 之后对整个"
                "扁平缓存打分，会把记忆前缀算进去导致下标错位。"
                "请改用 fastkvzip / expect / head 门控。"
            )

    def init_score(self, get_score=True):
        """拦住 KVzip 的重建打分路径（gates=None）。

        `_get_score` 假定 key_states 是**矩形** [bsz, head_kv, k, dim] 并按绝对
        位置切片（`key_states[:, :, start:end]`），而 EvictCache 系的 update()
        返回的是**扁平** [Σ_heads len_k_head, dim] —— 两者根本对不上。
        这不是我们引入的（上游 eval.py 就是靠强制 kv_type="retain" 绕开的），
        但新加的 "memory" 类型会让人自然地去试，所以给一句明确的错误。
        """
        if get_score:
            raise NotImplementedError(
                "MemoryEvictCache 不支持 KVzip 的重建打分（gates=None / gate=\"\"）："
                "_get_score 需要矩形 KV 布局，而本 cache 是 per-head 变长扁平布局。"
                "请指定 fastkvzip / expect / head 门控。"
            )
        return super().init_score(get_score)

    # ------------------------------------------------------------------ 位置

    @staticmethod
    def _flat_insert(old, new, lens, cu, H, seq):
        """update_flatten_view 的纯 PyTorch 等价实现（每个 head 段末尾插 seq 行）。

        **必须存在**：`update_flatten_view` 来自 AdaKV 的自定义 CUDA 扩展，
        没有注册 backward。记忆读出的 KV 在目标前向里要经过它，grad_fn 会在
        那里被静默切断 —— 实测表现为 loss 正常下降、`|grad|max` 恒为 0，
        看起来在训练其实什么都没学到。只在需要梯度时走这条较慢的路径。
        """
        segs = []
        for h in range(H):
            s = int(cu[h])
            segs.append(old[s: s + int(lens[h])])
            segs.append(new[h * seq: (h + 1) * seq])
        return torch.cat(segs)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=dict()):
        # 位置追踪、记忆状态按层切片、absorb 的补齐全部假定 B=1（上游评测也逐条跑）。
        # B>1 不会报错，只会静默把不同样本的 KV 混进同一批槽 —— 必须显式拦住。
        if key_states.shape[0] != 1:
            raise NotImplementedError(
                f"MemoryEvictCache 目前只支持 batch=1，收到 {key_states.shape[0]}"
            )
        seq = key_states.shape[-2]
        need_grad = torch.is_grad_enabled() and len(self.key_cache) > layer_idx and (
            self.key_cache[layer_idx].requires_grad
        )
        if need_grad:
            dim = key_states.shape[-1]
            H = self.n_heads_kv
            if layer_idx == 0:
                self._seen_tokens += cache_kwargs.get("seen_token", seq)
                self.cu_len_q = seq * self.cu_head
            lens, cu = self.info["len_k"][layer_idx], self.info["cu_len_k"][layer_idx]
            self.key_cache[layer_idx] = self._flat_insert(
                self.key_cache[layer_idx],
                key_states.contiguous().view(-1, dim), lens, cu, H, seq)
            self.value_cache[layer_idx] = self._flat_insert(
                self.value_cache[layer_idx],
                value_states.contiguous().view(-1, dim), lens, cu, H, seq)
            self.info["cu_len_k"][layer_idx] = cu + self.cu_len_q
            self.info["len_k"][layer_idx] = lens + seq
            self.info["max_len_k"][layer_idx] += seq
            out = (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        seen = self._seen_real[layer_idx]
        new_pos = torch.arange(seen, seen + seq, device=self.device, dtype=torch.long)
        lens = self.info["len_k"][layer_idx]

        if self.pos_track[layer_idx] is None:
            self.pos_track[layer_idx] = torch.cat([new_pos] * self.n_heads_kv)
        else:
            old, segs, off = self.pos_track[layer_idx], [], 0
            for h in range(self.n_heads_kv):
                keep = int(lens[h]) - seq          # 该 head 段中原有的条数
                segs.append(old[off: off + keep])
                segs.append(new_pos)
                off += keep
            self.pos_track[layer_idx] = torch.cat(segs)
        self._seen_real[layer_idx] = seen + seq
        return out

    def slice(self, seen_token_prev: int):
        """父类会把 query/生成阶段新增的 KV 从各 head 段**末尾**回滚掉，
        位置追踪与 _seen_real 必须同步回滚。

        不同步的后果不是立刻报错，而是**静默错位**：评测管线对同一个 cache 反复
        生成（每个 question 一次），每次都 slice 一下；pos_track 只增不减，
        几轮之后它和 key_cache 长度对不上，RoPE 逆旋转拿到的就是错位的位置，
        槽会变成不同相位 key 的平均 —— 正是 rope.py 全力避免的那个失效模式。
        """
        offset = self._seen_tokens - seen_token_prev
        if offset > 0:
            for l in range(self.n_layers):
                if self.pos_track[l] is None:
                    continue
                cu, lens = self.info["cu_len_k"][l], self.info["len_k"][l]
                segs = []
                for h in range(self.n_heads_kv):
                    s = int(cu[h])
                    segs.append(self.pos_track[l][s: s + int(lens[h]) - offset])
                self.pos_track[l] = torch.cat(segs)
                self._seen_real[l] = max(self._seen_real[l] - offset, 0)
        return super().slice(seen_token_prev)

    # ------------------------------------------------------- 预算记账（公平性）

    def measured_kv(self):
        """各层实际缓存条数（含记忆前缀）。**对比必须用这个数，而不是名义压缩比。**

        为什么不在裁剪流程里「扣预算」（试过两版，都废弃了）：
          v1 按 M/ctx_len 线性扣减 ratio —— `adakv-layer` 的 safeguard 跨层重分配
             预算、`threshold` 又按分数阈值而非精确计数选取，ratio→保留数是非线性的，
             实测偏差 +0.29% ~ −16.85%。
          v2 在 valid 掩码里降级 M 条最低分的 KV —— 在 1.5B(H=2)+adakv-layer 上
             精确归零，但换到 7B(H=4)+pair 就崩：账本 self.valid 差恰好 16（对），
             实际缓存却差 767 且**逐轮扩大**。上游的 valid 账本与 len_k 之间还有
             我没能完全推清的耦合，在那里做掩码手术太脆弱。

        所以改成不动上游、只如实报告：记忆带来的是 **每 head 恒定 M 条**的额外
        开销，把它计入横轴（实际缓存量）后再比曲线，结论就不会被这点开销污染。
        这比在裁剪逻辑里做手术更简单，也更容易验证。
        """
        return {
            "per_layer": [int(x.sum()) for x in self.info["len_k"]],
            "total": int(sum(int(x.sum()) for x in self.info["len_k"])),
            "mem_overhead": self.M * self.n_heads_kv * self.n_layers,
        }

    # ------------------------------------------------------------------ 掩码

    def _get_valid(self, layer_idx: int):
        """在父类基础上把记忆前缀标为恒有效（永不被驱逐）。"""
        if not self.M or not self._mem_inserted[layer_idx]:
            return super()._get_valid(layer_idx)
        valid_list = []
        for h in range(self.n_heads_kv):
            valid = self.valid[layer_idx][h]
            pad = int(self.info["len_k"][layer_idx][h]) - valid.shape[-1] - self.sink - self.M
            valid = F.pad(valid, (self.sink + self.M, pad), mode="constant", value=True)
            valid_list.append(valid)
        return valid_list

    # ------------------------------------------------------------------ 核心

    def _sample_cache(self, layer_idx, valid_list):
        """锚点：在 mask 生效**之前**把 `[~mask]` 的 KV 吸收进记忆。"""
        if not self.stats["warned_share"]:
            self.stats["warned_share"] = True
            self.assert_gate_compatible(getattr(self, "gates", None))
            if self.absorb_enabled:
                real = float(self.info["len_k"][layer_idx].float().mean()) - self.M
                share = self.M / max(real + self.M, 1.0)
                print(f"[VariKV] 每head保留真实KV≈{real:.0f}，记忆{self.M} "
                      f"→ 记忆占可见KV {share*100:.2f}%")
                print(f"[VariKV] 记忆额外开销 {self.M*self.n_heads_kv*self.n_layers} 条 "
                      f"—— 对比时请用 measured_kv() 的实际缓存量做横轴，勿用名义比例")
                if share < 0.03:
                    print("[VariKV][警告] 记忆占比 <3%，很可能小到测不出任何影响 —— "
                          "这是容量设计问题而非方法问题，建议加大 num_slots "
                          "或在更低的压缩比下比较。")


        if not self.absorb_enabled:
            # 关掉吸收时也必须同步过滤位置，否则 pos_track 与 key_cache 长度漂移。
            # 这条路径只用于「与原生 EvictCache 对拍」，但漂移会让布局自检误报。
            m = torch.cat(valid_list)
            super()._sample_cache(layer_idx, valid_list)
            if self.pos_track[layer_idx] is not None:
                self.pos_track[layer_idx] = self.pos_track[layer_idx][m]
            return

        mask = torch.cat(valid_list)
        k_all, v_all = self.key_cache[layer_idx], self.value_cache[layer_idx]
        pos_all = self.pos_track[layer_idx]
        cu, lens = self.info["cu_len_k"][layer_idx], self.info["len_k"][layer_idx]
        H = self.n_heads_kv

        ev_k, ev_v, ev_p, counts = [], [], [], []
        for h in range(H):
            s, e = int(cu[h]), int(cu[h]) + int(lens[h])
            drop = ~mask[s:e]
            # 记忆前缀不参与吸收（它本来就是记忆），从 drop 里排除
            if self._mem_inserted[layer_idx]:
                drop[: self.M] = False
            n = int(drop.sum())
            counts.append(n)
            ev_k.append(k_all[s:e][drop] if n else None)
            ev_v.append(v_all[s:e][drop] if n else None)
            ev_p.append(pos_all[s:e][drop] if (n and pos_all is not None) else None)

        if max(counts) > 0:
            self._absorb(layer_idx, ev_k, ev_v, ev_p, counts, max(counts))

        super()._sample_cache(layer_idx, valid_list)
        if pos_all is not None:
            self.pos_track[layer_idx] = pos_all[mask]

        self._refresh_memory(layer_idx)

    def _absorb(self, layer_idx, ev_k, ev_v, ev_p, counts, n_max):
        H, d = self.n_heads_kv, self.head_dim
        dev, dt = self.device, self.dtype
        kb = torch.zeros(1, H, n_max, d, device=dev, dtype=dt)
        vb = torch.zeros(1, H, n_max, d, device=dev, dtype=dt)
        pb = torch.zeros(H, n_max, device=dev, dtype=torch.float32)
        val = torch.zeros(1, H, n_max, device=dev, dtype=torch.bool)
        for h in range(H):
            n = counts[h]
            if not n:
                continue
            kb[0, h, :n], vb[0, h, :n] = ev_k[h], ev_v[h]
            if ev_p[h] is not None:
                pb[h, :n] = ev_p[h].float()
            val[0, h, :n] = True

        if self.inv_freq is not None:
            from varikv.rope import cos_sin_at, inverse_rope
            cos, sin = cos_sin_at(self.inv_freq, pb, dtype=kb.dtype)
            kb = inverse_rope(kb, cos, sin)

        # 期望注意力 ā：直接用门控分数。ELBO 的重建项必须与 F_i 的失真项共用
        # 同一个定义（注意力输出空间），否则「自由能」在代码里指了两个不同的量。
        # score[layer] 形状 [1,H,ctx_len] 且**按绝对位置索引**，pos_track 记的
        # 正是绝对位置，两者可以直接对上 —— 这也是维护 pos_track 的额外收益。
        ea = None
        score = getattr(self, "score", None)
        if score is not None and not isinstance(score, torch.Tensor):
            sl = score[layer_idx] if layer_idx < len(score) else None
            if sl is not None and sl.numel() and sl.shape[-1] > 0:
                ea = torch.zeros(1, H, n_max, device=dev, dtype=torch.float32)
                for h in range(H):
                    n = counts[h]
                    if not n or ev_p[h] is None:
                        continue
                    idx = ev_p[h].clamp(0, sl.shape[-1] - 1)
                    s = sl[0, h, idx].float().clamp_min(0)
                    tot = s.sum()
                    # ā 的语义是「在这 N 个观测上和为 1」，absorb 里再乘 N 还原
                    ea[0, h, :n] = s / tot if tot > 0 else 1.0 / max(n, 1)

        # 记忆模块的 dtype 未必等于缓存的 dtype（训练时常用 fp32 记忆 + bf16 模型，
        # 因为 CLAUDE.md 记过 bf16 下精度累加有 5.98% 相对误差）。这里显式对齐，
        # 否则 encoder 的 matmul 直接报 dtype 不匹配。
        mdt = next(self.mem.parameters()).dtype
        ev = torch.cat([kb, vb], dim=-1).to(mdt)
        gs = slice(layer_idx * H, (layer_idx + 1) * H)
        self._swap_in(gs)
        grad_on = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.mem.parameters()
        )
        with torch.set_grad_enabled(grad_on):
            self.mem.absorb(ev, expected_attn=ea, positions=pb, valid=val)
        self._swap_out(gs)
        self.stats["absorbed"] += int(val.sum())
        self.stats["calls"] += 1

    # ------------------------------------------------------------------ 读回

    def _refresh_memory(self, layer_idx):
        """把槽解码成 effective KV，写回各 head 段最前面的 M 个位置。

        首次调用需要**插入**（各 head 段长度 +M），之后是**原地覆盖**。
        不覆盖的话记忆就只写不读，等于没接上 —— 这个坑之前踩过。
        """
        if not self.M:
            return
        H, d = self.n_heads_kv, self.head_dim
        gs = slice(layer_idx * H, (layer_idx + 1) * H)
        self._swap_in(gs)
        grad_on = torch.is_grad_enabled() and any(
            p.requires_grad for p in self.mem.parameters()
        )
        with torch.set_grad_enabled(grad_on):
            eff = self.mem.read()                      # [1,H,M,d_kv]
            slot_pos = self.mem.pos.detach().clone()   # [1,H,K] 槽的位置质心
        self._swap_out(gs)

        eff = eff.reshape(1, H, -1, 2 * d)[:, :, : self.M]
        k_new, v_new = eff[..., :d], eff[..., d:]

        # 重旋到槽的位置质心：R(δ)R(p)=R(p+δ)，纯代数，不需要额外前向
        if self.inv_freq is not None:
            from varikv.rope import cos_sin_at, apply_rope
            p = slot_pos.reshape(1, H, -1)[:, :, : self.M].reshape(H, self.M)
            cos, sin = cos_sin_at(self.inv_freq, p, dtype=k_new.dtype)
            k_new = apply_rope(k_new, cos, sin)

        k_new = k_new.reshape(H, self.M, d).to(self.dtype)
        v_new = v_new.reshape(H, self.M, d).to(self.dtype)
        if self.detach_readback:
            k_new, v_new = k_new.detach(), v_new.detach()

        k_all, v_all = self.key_cache[layer_idx], self.value_cache[layer_idx]
        cu, lens = self.info["cu_len_k"][layer_idx], self.info["len_k"][layer_idx]

        if not self._mem_inserted[layer_idx]:
            ks, vs, ps = [], [], []
            for h in range(H):
                s, e = int(cu[h]), int(cu[h]) + int(lens[h])
                ks += [k_new[h], k_all[s:e]]
                vs += [v_new[h], v_all[s:e]]
                if self.pos_track[layer_idx] is not None:
                    ps += [
                        torch.full((self.M,), -1, device=self.device, dtype=torch.long),
                        self.pos_track[layer_idx][s:e],
                    ]
            self.key_cache[layer_idx] = torch.cat(ks)
            self.value_cache[layer_idx] = torch.cat(vs)
            if ps:
                self.pos_track[layer_idx] = torch.cat(ps)
            new_lens = lens + self.M
            self.info["len_k"][layer_idx] = new_lens
            self.info["max_len_k"][layer_idx] = int(new_lens.max())
            cum = new_lens.cumsum(0).int()
            self.info["cu_len_k"][layer_idx] = torch.cat([self.zero, cum])
            self._mem_inserted[layer_idx] = True
        else:
            for h in range(H):
                s = int(self.info["cu_len_k"][layer_idx][h])
                k_all[s: s + self.M] = k_new[h]
                v_all[s: s + self.M] = v_new[h]

    # ---- 记忆状态按层切片：参数各层共享，状态各层独立 ----

    def _swap_in(self, gs):
        self._full = {n: getattr(self.mem, n) for n in self._STATE}
        for n in self._STATE:
            setattr(self.mem, n, self._full[n][:, gs].clone())

    def _swap_out(self, gs):
        """用 cat 重建而非就地索引赋值 —— 后者会切断自动微分。

        `full[:, gs] = new` 是 in-place 写，训练时要么报「a view of a leaf Variable
        that requires grad is being used in an in-place operation」，要么静默地
        让梯度传不回记忆参数。cat 重建保住计算图，代价只是每层重建一个
        [1,G,K,d_z] 的小张量（G=112、K=16、d_z=64 → 约 11 万个数）。
        """
        for n in self._STATE:
            full, new = self._full[n], getattr(self.mem, n)
            setattr(
                self.mem, n,
                torch.cat([full[:, : gs.start], new, full[:, gs.stop:]], dim=1),
            )
        self._full = None

    def refresh_with_grad(self):
        """预填结束后补一次**带梯度**的读回，供训练用。

        配合 detach_readback：预填期间的读回是 detach 的（LLM 前向不建图），
        这里把各 head 段最前面的 M 条替换成带梯度的版本，并用 cat 重建而非
        就地赋值 —— 就地写会把梯度写进一个不带图的张量里，静默丢梯度。
        """
        assert all(self._mem_inserted), "记忆尚未插入，先跑一次 prefill"
        H, d = self.n_heads_kv, self.head_dim
        for l in range(self.n_layers):
            gs = slice(l * H, (l + 1) * H)
            self._swap_in(gs)
            eff = self.mem.read()
            slot_pos = self.mem.pos.detach().clone()
            self._swap_out(gs)

            eff = eff.reshape(1, H, -1, 2 * d)[:, :, : self.M]
            k_new, v_new = eff[..., :d], eff[..., d:]
            if self.inv_freq is not None:
                from varikv.rope import cos_sin_at, apply_rope
                p = slot_pos.reshape(1, H, -1)[:, :, : self.M].reshape(H, self.M)
                cos, sin = cos_sin_at(self.inv_freq, p, dtype=k_new.dtype)
                k_new = apply_rope(k_new, cos, sin)
            k_new = k_new.reshape(H, self.M, d).to(self.dtype)
            v_new = v_new.reshape(H, self.M, d).to(self.dtype)

            k_all, v_all = self.key_cache[l], self.value_cache[l]
            cu, lens = self.info["cu_len_k"][l], self.info["len_k"][l]
            ks, vs = [], []
            for h in range(H):
                s0, e0 = int(cu[h]), int(cu[h]) + int(lens[h])
                ks += [k_new[h], k_all[s0 + self.M: e0]]
                vs += [v_new[h], v_all[s0 + self.M: e0]]
            self.key_cache[l] = torch.cat(ks)
            self.value_cache[l] = torch.cat(vs)
