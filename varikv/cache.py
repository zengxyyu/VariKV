"""带记忆的 KV cache 驱动：分块预填 → 按自由能驱逐 → 吸收进记忆 → 读出拼回。

关键实现决策：**per-token 统一驱逐**（一个 token 的 KV 在所有 layer/head 上同进同退），
而不是 per-head 独立驱逐。理由是后者会让每个头的 KV 长度不同，cache 退化成
per-head 变长布局（FastKVzip 的 `[Σ_heads len_k_head, dim]` + `cu_seqlens_k` 就是这么来的），
必须配套改 attention kernel。在方法本身还没验证时引入那层复杂度，会让
「方法对不对」和「张量布局写对没有」两类 bug 缠在一起。
驱逐分数按 (layer, head) 聚合成 per-token 标量，cache 全程保持规整张量。

位置编码：cache 里存的 k 已经过 RoPE。驱逐后**不重排位置** —— 保留每个 token 原本的
RoPE 相位，新 token 用其真实绝对位置，因此留下来的 token 之间相对位置关系不变
（H2O / KVzip 等驱逐类方法的标准做法）。记忆读出的 effective KV 没有位置概念，
作为 attention-sink 式的前缀拼在最前面。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import DynamicCache

from .config import Config
from .free_energy import FreeEnergyScorer
from .memory import DistributionalMemory
from .moment import MomentMemory
from .rope import apply_rope, cos_sin_at, inverse_rope


class MemoryAugmentedCache(nn.Module):
    """把冻结 LLM 的 KV cache 管起来：驱逐、吸收、读出。"""

    def __init__(self, model, cfg: Config):
        super().__init__()
        self.cfg = cfg
        mcfg = model.config
        self.n_layers = mcfg.num_hidden_layers
        self.n_kv_heads = getattr(mcfg, "num_key_value_heads", mcfg.num_attention_heads)
        self.head_dim = getattr(
            mcfg, "head_dim", mcfg.hidden_size // mcfg.num_attention_heads
        )
        self.layers = cfg.layers if cfg.layers is not None else list(range(self.n_layers))
        self.n_groups = len(self.layers) * self.n_kv_heads
        self.d_kv = 2 * self.head_dim

        absorb = cfg.cache.absorb_mode
        self.absorb_mode = absorb
        if absorb == "moment":
            # training-free baseline，无可训练参数
            self.memory = MomentMemory(self.d_kv, cfg.memory)
        elif absorb != "discard":
            self.memory = DistributionalMemory(
                self.d_kv, cfg.memory, mode=absorb
            )
        else:
            self.memory = None

        self.scorer = (
            FreeEnergyScorer(self.d_kv, cfg.free_energy, cfg.memory.d_latent)
            if cfg.cache.evict_policy == "free_energy"
            else None
        )

        # RoPE 的 inv_freq：吸收前要把 key 逆旋转回 position-free frame，
        # 读出时再按槽的位置质心重旋（见 varikv/rope.py 的完整理由）。
        rot = getattr(model.model, "rotary_emb", None)
        self.register_buffer(
            "inv_freq",
            rot.inv_freq.detach().clone() if rot is not None else torch.zeros(1),
            persistent=False,
        )
        self.has_rope = rot is not None

        # 当前 cache 中，前 n_mem 个位置是记忆读出的 effective KV
        self.n_mem = 0
        # 已见的真实 token 数（用于绝对位置）
        self.n_seen = 0
        self.aux_losses = {}

    # ------------------------------------------------------------------ 状态

    def reset(self, batch: int, device, dtype):
        if self.memory is not None:
            self.memory.reset(batch, self.n_groups, device=device, dtype=dtype)
        if self.scorer is not None:
            self.scorer.reset()
        self.n_mem = 0
        self.n_seen = 0
        self.token_pos = None      # [n_real] cache 中每个真实 KV 的绝对位置
        self.aux_losses = {"free_energy": [], "predictor": []}

    def detach_state(self):
        if self.memory is not None:
            self.memory.detach_state()

    # ------------------------------------------------------- cache <-> 分组张量

    def _gather_kv(self, cache: DynamicCache, start: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """把各层 cache 中 [start:] 的 KV 堆成 [B, G, N, d_head] 两个张量。"""
        ks, vs = [], []
        for l in self.layers:
            ks.append(cache.key_cache[l][:, :, start:, :])
            vs.append(cache.value_cache[l][:, :, start:, :])
        # [L, B, H, N, d] -> [B, L*H, N, d]
        k = torch.stack(ks, dim=1).flatten(1, 2)
        v = torch.stack(vs, dim=1).flatten(1, 2)
        return k, v

    def _scatter_kv(self, cache: DynamicCache, k: torch.Tensor, v: torch.Tensor, start: int):
        """把 [B, G, N, d_head] 写回各层 cache 的 [start:] 段。"""
        B, G, N, d = k.shape
        L, H = len(self.layers), self.n_kv_heads
        k = k.view(B, L, H, N, d)
        v = v.view(B, L, H, N, d)
        for i, l in enumerate(self.layers):
            cache.key_cache[l] = torch.cat(
                [cache.key_cache[l][:, :, :start, :], k[:, i]], dim=2
            )
            cache.value_cache[l] = torch.cat(
                [cache.value_cache[l][:, :, :start, :], v[:, i]], dim=2
            )

    # ------------------------------------------------------------------ 驱逐

    def _evict_scores(self, k, v, keep_from: int, n_real: int):
        """给每个真实 token 一个驱逐分数，**低分先降级进记忆**。

        free_energy：F_i = D_i + λ·KL_i，按 (layer, head) 聚合成 per-token 标量。
                     F 高 = 压进记忆代价大 = 该留精确。
        recency    ：位置越靠后分数越高（= 只保最近的），等价于 Infini-attention
                     / Tensor Cache 的 FIFO 驱逐，作为消融第 2 档。
        """
        B = k.shape[0]
        device = k.device
        if self.cfg.cache.evict_policy == "recency":
            pos = torch.arange(n_real, device=device, dtype=torch.float32)
            return pos.unsqueeze(0).expand(B, n_real), {}

        rel_pos = (
            torch.arange(n_real, device=device, dtype=torch.float32) / max(n_real, 1)
        ).view(1, 1, n_real).expand(B, self.n_groups, n_real)
        F, aux = self.scorer.score(self.memory, k, v, rel_pos)
        # 按组聚合成 per-token 分数（保持 cache 规整，见模块 docstring）
        return F.mean(dim=1), aux

    def _split_keep_evict(self, scores, n_real: int):
        """返回 (keep_idx, evict_idx)。sink 与最近窗口内的 token 强制保留。"""
        ccfg = self.cfg.cache
        B = scores.shape[0]
        device = scores.device
        budget = ccfg.budget

        protected = torch.zeros(n_real, dtype=torch.bool, device=device)
        protected[: min(ccfg.n_sink, n_real)] = True                 # attention sink
        if ccfg.local_window > 0:
            protected[max(0, n_real - ccfg.local_window):] = True     # 最近窗口

        n_protected = int(protected.sum().item())
        n_free = max(budget - n_protected, 0)

        s = scores.clone()
        s[:, protected] = float("inf")                                # 保证排在最前
        order = torch.argsort(s, dim=-1, descending=True)             # 高分优先保留
        keep_idx = order[:, : n_protected + n_free]
        evict_idx = order[:, n_protected + n_free:]
        # 恢复原始顺序，保持 RoPE 相位与因果性
        keep_idx, _ = torch.sort(keep_idx, dim=-1)
        evict_idx, _ = torch.sort(evict_idx, dim=-1)
        return keep_idx, evict_idx

    def _maybe_evict(self, cache: DynamicCache):
        """cache 超预算时：算分 → 分割 → 吸收被驱逐者 → 重建 cache。"""
        ccfg = self.cfg.cache
        total = cache.key_cache[self.layers[0]].shape[2]
        n_real = total - self.n_mem
        if n_real <= ccfg.budget:
            return

        k, v = self._gather_kv(cache, self.n_mem)                     # 只看真实 KV
        scores, aux = self._evict_scores(k, v, self.n_mem, n_real)
        keep_idx, evict_idx = self._split_keep_evict(scores, n_real)

        if evict_idx.shape[-1] == 0:
            return

        if self.memory is not None:
            # 被驱逐的 KV 写入记忆（决策 B）
            idx = evict_idx[0]                                        # per-token 统一
            k_ev, v_ev = k[:, :, idx, :], v[:, :, idx, :]
            pos_ev = self.token_pos[idx] if self.token_pos is not None else None
            if self.has_rope and pos_ev is not None:
                # 逆旋转回 position-free frame 再吸收。不做这一步的话，
                # 槽是不同相位 key 的加权平均，不对应任何有效位置，且会虚增 σ²
                # —— 那样「方差编码不确定性」这个核心主张就不成立。
                cos, sin = cos_sin_at(self.inv_freq, pos_ev, dtype=k_ev.dtype)
                k_ev = inverse_rope(k_ev, cos, sin)
            ev = torch.cat([k_ev, v_ev], dim=-1)
            # 把 scorer 算好的期望注意力传下去，让 ELBO 的重建失真与 F_i 的
            # 失真项共用同一个定义（注意力输出空间）。recency 档没有 scorer，传 None。
            ea = aux.get("expected_attn") if aux else None
            if ea is not None:
                ea = ea[:, :, idx]
            kl, fe = self.memory.absorb(ev, expected_attn=ea, positions=pos_ev)
            self.aux_losses["free_energy"].append(fe)
        if aux and "predictor_loss" in aux:
            self.aux_losses["predictor"].append(aux["predictor_loss"])

        # 保留的真实 KV
        kid = keep_idx[0]
        k_keep, v_keep = k[:, :, kid, :], v[:, :, kid, :]
        if self.token_pos is not None:
            self.token_pos = self.token_pos[kid]
        self._scatter_kv(cache, k_keep, v_keep, self.n_mem)
        self._refresh_memory_prefix(cache)

    # ------------------------------------------------------------------ 读出

    def _refresh_memory_prefix(self, cache: DynamicCache):
        """用当前记忆重新生成 effective KV 前缀，替换 cache 最前面的 n_mem 个位置。"""
        if self.memory is None or not self.cfg.cache.read_memory:
            return
        eff = self.memory.read()                                      # [B,G,M,d_kv]
        prec = self.memory.read_precision()                           # [B,G,M]
        d = self.head_dim
        k_mem, v_mem = eff[..., :d], eff[..., d:]

        # 槽存的是 pre-RoPE 的 key，读出时按位置质心重旋到一个**有效位置**。
        # R(δ)R(p)=R(p+δ)，所以这是纯代数重旋，不需要额外前向。
        if self.has_rope and self.memory.pos is not None:
            cos, sin = cos_sin_at(
                self.inv_freq, self.memory.pos, dtype=k_mem.dtype
            )                                                    # [B,G,K,d]
            cos = cos.repeat_interleave(self.memory.cfg.tokens_per_slot, dim=2)
            sin = sin.repeat_interleave(self.memory.cfg.tokens_per_slot, dim=2)
            k_mem = apply_rope(k_mem, cos, sin)

        # 缺口③的第三条路径：精度直接缩放 effective key 的幅度，
        # 让「不确定的槽」在 softmax 里天然获得更低的 logit。
        scale = (prec / (1.0 + prec)).unsqueeze(-1).to(k_mem.dtype)
        k_mem = k_mem * scale

        M = k_mem.shape[2]
        B, G = k_mem.shape[0], k_mem.shape[1]
        L, H = len(self.layers), self.n_kv_heads
        k_mem = k_mem.view(B, L, H, M, d)
        v_mem = v_mem.view(B, L, H, M, d)
        for i, l in enumerate(self.layers):
            rest_k = cache.key_cache[l][:, :, self.n_mem:, :]
            rest_v = cache.value_cache[l][:, :, self.n_mem:, :]
            cache.key_cache[l] = torch.cat([k_mem[:, i].to(rest_k.dtype), rest_k], dim=2)
            cache.value_cache[l] = torch.cat([v_mem[:, i].to(rest_v.dtype), rest_v], dim=2)
        self.n_mem = M

    # ------------------------------------------------------------------ 预填

    @torch.no_grad()
    def _update_query_stats(self, model, hidden_states):
        """从当前 chunk 的 hidden state 估计 query 分布（供 D_i 的期望注意力用）。"""
        if self.scorer is None:
            return
        # 用第一层的 q_proj 作为 query 分布的代理，避免把所有层都跑一遍
        layer = model.model.layers[self.layers[0]]
        q = layer.self_attn.q_proj(hidden_states)
        B, T, _ = q.shape
        q = q.view(B, T, -1, self.head_dim).transpose(1, 2)           # [B, n_q, T, d]
        # 折叠到 KV 组数：GQA 下多个 q head 共享一个 kv head
        n_q = q.shape[1]
        rep = max(n_q // self.n_kv_heads, 1)
        q = q[:, : self.n_kv_heads * rep].view(B, self.n_kv_heads, rep, T, self.head_dim)
        q = q.mean(dim=2)                                             # [B, H, T, d]
        q = q.unsqueeze(1).expand(B, len(self.layers), self.n_kv_heads, T, self.head_dim)
        self.scorer.query_stats.update(q.flatten(1, 2))

    def prefill(self, model, input_ids: torch.Tensor) -> DynamicCache:
        """分块预填长上下文，块间做驱逐与吸收。返回最终 cache。"""
        ccfg = self.cfg.cache
        B, T = input_ids.shape
        device = input_ids.device
        cache = DynamicCache()
        self.reset(B, device, next(model.parameters()).dtype)

        n_chunks = 0
        for s in range(0, T, ccfg.prefill_chunk):
            chunk = input_ids[:, s : s + ccfg.prefill_chunk]
            q_len = chunk.shape[1]
            kv_len = self.n_mem + self.n_seen
            position_ids = torch.arange(
                self.n_seen, self.n_seen + q_len, device=device
            ).unsqueeze(0).expand(B, q_len)
            attention_mask = torch.ones(B, kv_len + q_len, device=device, dtype=torch.long)

            out = model.model(
                input_ids=chunk,
                past_key_values=cache,
                position_ids=position_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            cache = out.past_key_values
            new_pos = torch.arange(
                self.n_seen, self.n_seen + q_len, device=device, dtype=torch.float32
            )
            self.token_pos = (
                new_pos if self.token_pos is None
                else torch.cat([self.token_pos, new_pos])
            )
            self.n_seen += q_len
            self._update_query_stats(model, out.last_hidden_state)

            self._maybe_evict(cache)

            n_chunks += 1
            if (self.cfg.train.truncate_bptt > 0
                    and n_chunks % self.cfg.train.truncate_bptt == 0):
                self.detach_state()
            if self.scorer is not None:
                self.scorer.step()

        return cache

    def collect_aux_loss(self) -> dict:
        """把预填过程中累积的辅助损失聚合起来。"""
        out = {}
        for name, vals in self.aux_losses.items():
            if vals:
                out[name] = torch.stack([v.float() for v in vals]).mean()
        return out

    def trainable_parameters(self):
        ps = []
        if self.memory is not None:
            ps += list(self.memory.parameters())
        if self.scorer is not None:
            ps += list(self.scorer.parameters())
        return ps
