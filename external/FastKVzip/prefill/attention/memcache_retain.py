"""VariKV × Fast KVzip —— 建在 `RetainCache` 上的版本（2026-08-09）。

**为什么要有这一版**：Figure 11 的全部基线跑的是 `kv_type="retain"`（`args.py` 的
默认值，`run.sh` 与 `scratch_repro_full.py` 都不覆盖它），而先前的
`MemoryEvictCache` 继承自 `EvictCache`。两者虽经实测在基线上生成逐字相同，
但要做到与上游**逐项一致**，方法应当挂在基线实际使用的那套机制上。

`RetainCache` 与 `EvictCache` 的差别：
    EvictCache  物理删除被驱逐的 KV，缓存变成 per-head 变长的扁平布局
    RetainCache 什么都不删，只维护一个 `valid` 掩码，在 `prepare()` 里
                按掩码取子集喂给 varlen FlashAttention

对我们反而更简单，两点：
  1. 缓存始终是规整的 `[B, H, seq, dim]`，**不压紧**，所以「原始位置 = 序列维下标」，
     不需要 EvictCache 版里那套 `pos_track` 追踪（RoPE 逆旋转要绝对位置）。
  2. `self.valid` 是累积的、且不会被重置，所以「本轮新被驱逐的」可以直接由
     evict_range 切出来，不存在重复吸收的风险。

布局不变式：每层缓存为 `[B, H, M + seq, dim]`，记忆前缀恒在最前。
放最前而非最后的原因同 EvictCache 版：`update()` 从序列末尾追加，
放末尾会被下一次 update 打散；且 flash-attn 的因果掩码按右下对齐，
记忆作为「最旧」的 key 恒可被注意且不泄露未来。
"""
import torch
import torch.nn.functional as F

from attention.kvcache import RetainCache


class MemoryRetainCache(RetainCache):
    """RetainCache + 分布式记忆：被掩码掉的 KV 吸收进 (μ,σ²) 槽再读回。"""

    _STATE = ("mu", "logvar", "var_content", "pos", "_pos_tau")

    def __init__(self, model, evict_range, mem_module, rope_inv_freq=None,
                 n_mem_per_head: int = 16, absorb_enabled: bool = True):
        super().__init__(model, evict_range)
        self.mem = mem_module
        self.absorb_enabled = absorb_enabled and mem_module is not None
        self.M = n_mem_per_head if self.absorb_enabled else 0
        self.inv_freq = rope_inv_freq
        self.head_dim = getattr(
            model.config, "head_dim",
            model.config.hidden_size // model.config.num_attention_heads)
        self._mem_inserted = [False] * self.n_layers
        self._absorbed_upto = 0          # 已吸收到的 context 位置（绝对坐标）
        self.stats = {"absorbed": 0, "calls": 0}
        self.detach_readback = False
        # 每 N 个 chunk 才截断一次记忆递归。**原先是每 chunk 无条件 detach**，
        # 与 scratch_stage2b_train.py 顶部 docstring 承诺的 --detach_every 不符：
        # 那样只有最后一个 chunk 的 absorb 计算图还连着 encoder，跨 chunk 的
        # 「记忆如何影响后续轨迹」完全拿不到梯度。1 = 旧行为。
        self.detach_every = 1
        self._chunk_i = 0
        # 消融用：'normal' 正常读出；'zero' 把等效 KV 置零。
        # 置零后 key=0 ⇒ 所有 query 的 logit 恒为 0（仍分走 softmax 质量），
        # 而 value=0 ⇒ 对输出不贡献内容。用来把「抢注意力」与「内容有害」拆开。
        self.readout_mode = "normal"
        # residual_mode=True 时记忆**不进 cache**，改由 attn.py 在注意力输出端
        # 以 `o = o_attn + sigmoid(gate)·m(q)` 融合。这是 2026-08-09 文献扫描
        # 与零读出消融共同指向的修正：IndexMem/Tensor Cache/Infini-attention
        # 全是这个形态，而把记忆塞进 softmax 会付出与内容无关的 30~40 点固定成本。
        self.residual_mode = False
        # 是否训练「写入」通路（encoder/to_mu/to_logvar）。见 _absorb_layer 的说明。
        self.train_write = False
        # IndexMem(ICML'26) 式的记忆损失：ℒ = ‖(o_full − o_pruned) − g(q)·m(q)‖²
        # 直接回归「驱逐造成的注意力缺口」，而不是绕一圈用下游语言建模损失。
        # 后者的问题实测过：目标 token 的预测几乎全靠局部上下文，而局部窗口
        # (window_size=4096) 是被强制保留的，记忆装垃圾还是宝贝都影响不大 ——
        # 训练 loss 一直正常，评测却掉 30~40 点。
        # 这个目标还与评测指标对齐：评测算的正是「离全量缓存有多远」。
        self.collect_residual_loss = False
        self.residual_losses = []

        # 每个新 cache = 一条新序列 ⇒ 记忆状态归零。模块跨样本共享（参数要共享），
        # 状态属于当前序列；不重置会让第 N 条样本带着第 N−1 条的记忆。
        if self.absorb_enabled:
            self.mem.reset(1, self.n_layers * self.n_heads_kv, device=self.device,
                           dtype=next(self.mem.parameters()).dtype)

    def assert_gate_compatible(self, gates):
        """`snap` 门控在 update() **之后**对返回的缓存打分。RetainCache 的 update
        返回的是含记忆前缀的完整张量，会让 score 与 context 位置错位。"""
        if gates is None:
            return
        if any(getattr(g, "name", "") == "snap" for g in gates):
            raise NotImplementedError(
                "MemoryRetainCache 不支持 snap 门控（它在 update 后对整个缓存打分，"
                "会把记忆前缀算进去）。请用 fastkvzip / expect / head。")

    # ------------------------------------------------------------------ 掩码

    def _get_valid(self, layer_idx: int, n_seq: int):
        """把记忆前缀标为恒有效（永不被掩掉）。"""
        if not self.M or not self._mem_inserted[layer_idx]:
            return super()._get_valid(layer_idx, n_seq)
        valid = self.valid[layer_idx]
        pad = n_seq - valid.shape[-1] - self.sink - self.M
        return F.pad(valid, (self.sink + self.M, pad), mode="constant", value=True)

    def slice(self, seen_token_prev: int):
        """父类按 `[:, :, :seen_token_prev]` 回滚；有记忆前缀时真实长度要 +M。"""
        if not self.M or not any(self._mem_inserted):
            return super().slice(seen_token_prev)
        keep = self.M + seen_token_prev
        for i in range(self.n_layers):
            self.key_cache[i] = self.key_cache[i][:, :, :keep]
            self.value_cache[i] = self.value_cache[i][:, :, :keep]
        self._seen_tokens = seen_token_prev

    # ------------------------------------------------------------------ 核心

    def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
        """锚点：`self.valid` 算好之后，把**本轮新被掩掉**的 KV 吸收进记忆。

        RetainCache 不物理删除，所以「被驱逐」= valid==False。`self.valid` 是
        累积的且不重置，所以只需处理 evict_range 这一段新出现的区域，
        天然不会重复吸收。
        """
        out = super().prune_chunk(ratio, evict_range, level)
        if not self.absorb_enabled:
            return out
        self.assert_gate_compatible(getattr(self, "gates", None))

        lo, hi = evict_range                      # 绝对 context 坐标（含 sink 偏移）
        s, e = lo - self.sink, hi - self.sink     # self.valid 的坐标（不含 sink）
        if e <= s:
            return out
        _skip = getattr(self, "skip_first_detach", False) and self._chunk_i == 0
        if self.train_write and not _skip and (self._chunk_i % max(self.detach_every, 1) == 0):
            # 截断 BPTT：切断上一段的图，否则 11 个 chunk × 28 层的 encoder 激活
            # 会一路累积。detach_every=1 时梯度只经由**最后一个 chunk** 的 absorb
            # 回传；调大它可让 encoder 看到跨 chunk 的记忆递归，代价是显存。
            self.mem.detach_state()
        self._chunk_i += 1
        for l in range(self.n_layers):
            self._absorb_layer(l, s, e)
            self._refresh_memory(l)
        self._absorbed_upto = e
        return out

    def _absorb_layer(self, layer_idx, s, e):
        H, d = self.n_heads_kv, self.head_dim
        off = self.sink + (self.M if self._mem_inserted[layer_idx] else 0)
        k_all = self.key_cache[layer_idx]         # [1,H,seq,d]
        v_all = self.value_cache[layer_idx]
        vmask = self.valid[layer_idx][..., s:e]   # [H, e-s]（squeeze 后）
        if vmask.dim() == 3:
            vmask = vmask.squeeze(0)

        drops = [(~vmask[h]).nonzero(as_tuple=True)[0] for h in range(H)]
        counts = [int(x.numel()) for x in drops]
        n_max = max(counts) if counts else 0
        if n_max == 0:
            return

        dev, dt = self.device, self.dtype
        kb = torch.zeros(1, H, n_max, d, device=dev, dtype=dt)
        vb = torch.zeros(1, H, n_max, d, device=dev, dtype=dt)
        pb = torch.zeros(H, n_max, device=dev, dtype=torch.float32)
        val = torch.zeros(1, H, n_max, device=dev, dtype=torch.bool)
        for h in range(H):
            n = counts[h]
            if not n:
                continue
            idx = drops[h] + s                    # → self.valid 坐标
            cache_idx = idx + off                 # → 缓存序列维下标
            kb[0, h, :n] = k_all[0, h, cache_idx]
            vb[0, h, :n] = v_all[0, h, cache_idx]
            # 缓存不压紧 ⇒ 原始 token 位置 = 下标 − 记忆前缀长度
            pb[h, :n] = (cache_idx - (self.M if self._mem_inserted[layer_idx] else 0)).float()
            val[0, h, :n] = True

        if self.inv_freq is not None:
            from varikv.rope import cos_sin_at, inverse_rope
            cos, sin = cos_sin_at(self.inv_freq, pb, dtype=kb.dtype)
            kb = inverse_rope(kb, cos, sin)

        # 期望注意力用门控分数（score[layer] 形状 [1,H,ctx]，按 context 位置索引）
        ea = None
        sl = getattr(self, "score", None)
        if sl is not None and not isinstance(sl, torch.Tensor) and layer_idx < len(sl):
            sc_l = sl[layer_idx]
            if sc_l.numel():
                ea = torch.zeros(1, H, n_max, device=dev, dtype=torch.float32)
                for h in range(H):
                    n = counts[h]
                    if not n:
                        continue
                    p = (drops[h] + s).clamp(0, sc_l.shape[-1] - 1)
                    v_ = sc_l[0, h, p].float().clamp_min(0)
                    tot = v_.sum()
                    ea[0, h, :n] = v_ / tot if tot > 0 else 1.0 / max(n, 1)

        mdt = next(self.mem.parameters()).dtype
        ev = torch.cat([kb, vb], dim=-1).to(mdt)
        gs = slice(layer_idx * H, (layer_idx + 1) * H)
        self._swap_in(gs)
        # 注意这里**不**看 torch.is_grad_enabled()：预填整体跑在 no_grad 下
        # （为了让 28 层 × 11 chunk 的 LLM 前向不建图，否则 OOM），
        # 但 absorb 只对已缓存的 KV 张量做运算，建图代价很小。
        # 若跟着外层一起关梯度，encoder/to_mu/to_logvar 就永远拿不到梯度 ——
        # 记忆的**写入**通路完全不被训练，只有 decoder 和门在学。
        # 实测过：encoder 梯度恒为 0.00e+00。
        grad_on = self.train_write and any(p.requires_grad
                                           for p in self.mem.parameters())
        with torch.set_grad_enabled(grad_on):
            self.mem.absorb(ev, expected_attn=ea, positions=pb, valid=val)
        self._swap_out(gs)
        self.stats["absorbed"] += int(val.sum())
        self.stats["calls"] += 1

    # ------------------------------------------------------------------ 读回

    def _refresh_memory(self, layer_idx):
        if not self.M or self.residual_mode:
            return          # 残差模式：记忆不进 cache，读出在 attn 输出端做
        H, d = self.n_heads_kv, self.head_dim
        gs = slice(layer_idx * H, (layer_idx + 1) * H)
        self._swap_in(gs)
        grad_on = torch.is_grad_enabled() and any(p.requires_grad
                                                  for p in self.mem.parameters())
        with torch.set_grad_enabled(grad_on):
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
        k_new = k_new.reshape(1, H, self.M, d).to(self.dtype)
        v_new = v_new.reshape(1, H, self.M, d).to(self.dtype)
        if self.readout_mode == "zero":
            k_new = torch.zeros_like(k_new)
            v_new = torch.zeros_like(v_new)
        if self.detach_readback:
            k_new, v_new = k_new.detach(), v_new.detach()

        if not self._mem_inserted[layer_idx]:
            self.key_cache[layer_idx] = torch.cat([k_new, self.key_cache[layer_idx]], dim=2)
            self.value_cache[layer_idx] = torch.cat([v_new, self.value_cache[layer_idx]], dim=2)
            self._mem_inserted[layer_idx] = True
        else:
            self.key_cache[layer_idx] = torch.cat(
                [k_new, self.key_cache[layer_idx][:, :, self.M:]], dim=2)
            self.value_cache[layer_idx] = torch.cat(
                [v_new, self.value_cache[layer_idx][:, :, self.M:]], dim=2)

    def memory_residual(self, query_states, layer_idx):
        """输出端残差读出：对本层的 K 个槽单独做一次注意力，再按门缩放。

            m(q) = softmax(q·K̂ᵀ/√d)·V̂ ,   返回 sigmoid(gate)·m(q)

        与「塞进 cache」的本质差别：这里的 softmax **只在 16 个槽内部归一化**，
        完全不参与真实 KV 的那次 softmax，所以不会抢走任何注意力质量。
        gate→−∞ 时输出→0，精确退回基线。

        query_states: [B, HQ, T, d]（post-RoPE，prepare 之前）
        返回:          [B, T, HQ*d]，与两个分支 `view(bsz,q_len,-1)` 后的形状一致。
        （必须在 view 之后加：分块前的前向走非 flatten 分支，attn_output 是
          4 维 [B,T,HQ,d]；分块后走 flatten 分支，是 5 维 [B,T,H,G,d]。
          两者 view(bsz,q_len,-1) 后统一为 [B,T,HQ*d]，且查询头的排布一致 ——
          prepare 里 `view(bsz,H,G,T,d)` 决定了 head h ↔ (kv=h//G, grp=h%G)。）
        """
        # ---- P0-A guard（2026-08-11）：空记忆不得注入 ----
        # 本函数在 attn.py:149 被**无条件**调用，因此第一次吸收发生之前（槽仍是初值）
        # 也会往注意力输出里加东西。实测后果：同一 ckpt 跨独立 job 的 ratio-1.0 分数
        # 逐字相同、不同 ckpt 之间不同（68.20/66.80/68.60/67.20/67.80/70.40），
        # 即 ckpt 决定了本该与记忆无关的那一档分数 ⇒ full-cache 参照被污染。
        # 未吸收过任何东西时返回全零，形状与正常返回一致。
        if getattr(self, "_absorbed_upto", 0) <= 0:
            H = self.n_heads_kv
            B, HQ, T, d = query_states.shape
            return query_states.new_zeros(B, T, HQ * d)
        H, d = self.n_heads_kv, self.head_dim
        B, HQ, T, _ = query_states.shape
        Gq = HQ // H
        gs = slice(layer_idx * H, (layer_idx + 1) * H)
        self._swap_in(gs)
        grad_on = torch.is_grad_enabled() and any(p.requires_grad
                                                  for p in self.mem.parameters())
        with torch.set_grad_enabled(grad_on):
            eff = self.mem.read().reshape(1, H, -1, 2 * d)[:, :, : self.M]
            slot_pos = self.mem.pos.detach().clone()
        self._swap_out(gs)

        k_hat, v_hat = eff[..., :d], eff[..., d:]          # [1,H,M,d]
        if self.inv_freq is not None:
            # query 是 post-RoPE 的，所以 k̂ 也要旋到槽的位置质心才能正确内积
            from varikv.rope import cos_sin_at, apply_rope
            p = slot_pos.reshape(1, H, -1)[:, :, : self.M].reshape(H, self.M)
            cos, sin = cos_sin_at(self.inv_freq, p, dtype=k_hat.dtype)
            k_hat = apply_rope(k_hat, cos, sin)

        q = query_states.view(B, H, Gq, T, d).to(k_hat.dtype)
        logits = torch.einsum("bhgtd,bhmd->bhgtm", q, k_hat) / (d ** 0.5)
        w = torch.softmax(logits.float(), dim=-1).to(v_hat.dtype)
        m = torch.einsum("bhgtm,bhmd->bhgtd", w, v_hat)    # [B,H,Gq,T,d]

        gate = self.mem.residual_gate
        g = torch.sigmoid(gate[layer_idx * H:(layer_idx + 1) * H]) if gate is not None \
            else torch.zeros(H, device=m.device, dtype=m.dtype)
        # gate surgery（2026-08-12）：把注入幅度整体缩放。
        # 目的：point 的门学到 0.265、dist 只有 0.131，而"门越开分越低"是本项目
        # 已建立的规律。若把 point 的门缩到 dist 的水平就能从 14.60 回到 45~50，
        # 那 39.6 分的差就主要是**幅度失控**，与"方差携带信息"无关。
        gs = float(getattr(self, "gate_scale", 1.0))
        m = m * (g.view(1, H, 1, 1, 1).to(m.dtype) * gs)

        if self.collect_residual_loss:
            tgt = self._attn_gap(query_states, layer_idx)      # [B,H,Gq,T,d]，已 detach
            if tgt is not None:
                self.residual_losses.append(
                    torch.nn.functional.mse_loss(m.float(), tgt.float()))

        return m.permute(0, 3, 1, 2, 4).reshape(B, T, H * Gq * d)

    @torch.no_grad()
    def _attn_gap(self, query_states, layer_idx):
        """回归目标 = o_full − o_pruned，即「被驱逐掉的那部分注意力输出」。

        两次注意力只差一个 mask：o_full 看全部缓存，o_pruned 只看 valid 的部分。
        因为 RetainCache 物理上什么都没删，全量注意力是可以直接算出来的 ——
        这正是选 RetainCache 而非 EvictCache 作基类的意外收益。

        只在**目标 token**（T≈128 个）上算，不是整段预填：代价 T×S 而非 S²，
        而且语义上也对 —— 评测时提问的就是这些 query。
        """
        H, d = self.n_heads_kv, self.head_dim
        B, HQ, T, _ = query_states.shape
        Gq = HQ // H
        k_all = self.key_cache[layer_idx]                      # [1,H,S,d]
        v_all = self.value_cache[layer_idx]
        S = k_all.shape[2]
        if S <= T:
            return None
        valid = self._get_valid(layer_idx, S)                  # [H,S] 或 [1,H,S]
        if valid.dim() == 3:
            valid = valid.squeeze(0)

        dev = k_all.device
        # 目标 token 占据缓存最后 T 个位置 ⇒ query i 可见 key j ⇔ j ≤ S−T+i
        idx_k = torch.arange(S, device=dev).view(1, S)
        idx_q = (S - T) + torch.arange(T, device=dev).view(T, 1)
        causal = idx_k <= idx_q                                # [T,S]

        q = query_states.view(B, H, Gq, T, d).to(k_all.dtype)
        k = k_all.unsqueeze(2).expand(B, H, Gq, S, d)
        v = v_all.unsqueeze(2).expand(B, H, Gq, S, d)
        neg = torch.finfo(k_all.dtype).min

        mfull = torch.zeros(1, H, 1, T, S, device=dev, dtype=k_all.dtype)
        mfull.masked_fill_(~causal.view(1, 1, 1, T, S), neg)
        o_full = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mfull)

        mprun = mfull.clone()
        mprun.masked_fill_(~valid.view(1, H, 1, 1, S), neg)
        o_prun = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mprun)
        return (o_full - o_prun).detach()

    def refresh_with_grad(self):
        """训练用：预填后补一次**带梯度**的读回（配合 detach_readback）。

        必须临时关掉 detach_readback —— `_refresh_memory` 里那句
        `if self.detach_readback: k_new, v_new = k_new.detach(), v_new.detach()`
        会把这次补写也 detach 掉，于是 loss 正常下降而 |grad|max 恒为 0。
        实测踩过：这条路径在 RetainCache 版上直接导致「看起来在训练、其实什么都没学」。
        """
        saved, self.detach_readback = self.detach_readback, False
        try:
            for l in range(self.n_layers):
                if self._mem_inserted[l]:
                    self._refresh_memory(l)
        finally:
            self.detach_readback = saved

    def measured_kv(self):
        """实际参与注意力的 KV 数（按 valid 掩码计），含记忆前缀。"""
        if self.valid is None:
            return {"total": 0, "mem_overhead": 0}
        return {
            "total": int(self.valid.sum()) + self.M * self.n_heads_kv * self.n_layers,
            "mem_overhead": self.M * self.n_heads_kv * self.n_layers,
        }

    # ---- 记忆状态按层切片：参数各层共享，状态各层独立 ----

    def _swap_in(self, gs):
        # 必须与 _swap_out 一样在 grad 上下文里执行：外层预填跑在 no_grad 下，
        # 若这里的 clone / 下面的 cat 不开梯度，absorb 刚建好的图会被**这一步**
        # 切断（实测症状：encoder 梯度恒为 0.00e+00，而 decoder/gate 正常）。
        with torch.set_grad_enabled(self.train_write):
            self._full = {n: getattr(self.mem, n) for n in self._STATE}
            for n in self._STATE:
                setattr(self.mem, n, self._full[n][:, gs].clone())

    def _swap_out(self, gs):
        with torch.set_grad_enabled(self.train_write):
            for n in self._STATE:
                full, new = self._full[n], getattr(self.mem, n)
                setattr(self.mem, n,
                        torch.cat([full[:, : gs.start], new, full[:, gs.stop:]], dim=1))
        self._full = None
