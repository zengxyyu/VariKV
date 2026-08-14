"""LearnedControlRetainCache —— VariKV-B 最终版接进 harness 的那一层。

复用手工版（`ctrlcache.py`）已经在真机上验收通过的时序，只把「历史→修正」的那一步
从手工几何换成学出来的 `ControlMemory`：

    M_{t-1} ──read──▶ Δs_t ──▶ s⁰+Δs ──threshold──▶ R_t/E_t ──write──▶ M_t

无同 chunk 泄漏：本 chunk 的决定只用 `M_{t-1}`，`M_t` 影响的是下一个 chunk。

沿用手工版三条结构性保证（不是断言，是构造）：
  1. 预算：`threshold` 按 ratio 取全局 top-n，改分数只改"留哪些"不改"留几个"；
     `retain_delta` 记录实测差（父类用 `score > score_sort[n]` 而非严格 topk，
     阈值处并列时会少留，所以这是**经验事实**，不是构造性恒等）。
  2. ratio=1.0 不进 `prune_chunk` ⇒ 满缓存参考天然干净。
  3. `alpha=0`（初始值）⇒ Δs≡0 ⇒ 与基线**逐位相同**。

评测阶段仍派生自 `RetainCache`（逻辑掩码压缩、物理保留全部 KV），所以能回答
"memory-conditioned selection 有没有用"，**不能**声称峰值显存下降。若 B 成立，
再移植到 `EvictCache`。
"""
from typing import Tuple

import torch

from .kvcache import RetainCache


class LearnedControlRetainCache(RetainCache):
    def __init__(self, model, evict_range: Tuple[int, int], ctrl=None,
                 train_mode: bool = False, seed: int = 0):
        super().__init__(model, evict_range)
        self.ctrl = ctrl                       # ControlMemory，None ⇒ 纯基线
        self.train_mode = train_mode
        self.head_dim = getattr(
            model.config, "head_dim",
            model.config.hidden_size // model.config.num_attention_heads)
        self.M = None                          # [L][H,K,d_m]，惰性初始化
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        # 诊断
        self.flip_frac, self.retain_delta, self.delta_std = [], [], []
        # 训练用采集缓冲（train_mode 时填）
        self.trace = []

    @property
    def active(self) -> bool:
        return self.ctrl is not None and float(self.ctrl.alpha) != 0.0

    def _ensure_state(self):
        if self.M is None:
            self.M = [self.ctrl.init_state(l) for l in range(self.n_layers)]

    def _kv(self, layer_idx: int, pos: torch.Tensor):
        k = self.key_cache[layer_idx][0][:, pos]          # [H,n,d]
        v = self.value_cache[layer_idx][0][:, pos]
        return k, v

    # ------------------------------------------------------------------ 主体
    def prune_chunk(self, ratio: float, evict_range=tuple, level: str = "pair"):
        lo, hi = evict_range
        score0 = torch.stack(self.score, dim=0)[..., lo:hi]        # [L,1,H,n]
        score = score0

        if self.ctrl is not None:
            self._ensure_state()
            pos = torch.arange(lo, hi, device=self.device)
            ctx = torch.enable_grad() if self.train_mode else torch.no_grad()
            with ctx:
                delta = torch.zeros_like(score0)
                feats = []
                for l in range(self.n_layers):
                    k, v = self._kv(l, pos)
                    x = self.ctrl.feat(k, v)                        # [H,n,d_m]
                    r = self.ctrl.read(self.M[l], x)
                    delta[l, 0] = self.ctrl.delta(x, r, score0[l, 0].float())
                    # **每层用完即弃**：[H,n,d_m] 每层 16 MB，28 层留着就是 450 MB。
                    # 手工版正是在这里踩过 917 MB 的坑。写入阶段重算一次特征更划算。
                    feats.append(None)
                    del k, v, x, r
            if self.active:
                score = score0 + delta.to(score0.dtype)
                self.delta_std.append(float(delta.std()))

        valid, thres = self.threshold(score, ratio, level)          # [L,H,n]

        if self.active:
            with torch.no_grad():                                   # 自包含 flip rate
                v0, _ = self.threshold(score0, ratio, level)
                self.flip_frac.append(float((valid ^ v0).float().mean()))
                self.retain_delta.append(int(valid.sum()) - int(v0.sum()))

        if self.ctrl is not None:
            self._write(lo, hi, valid)

        if self.valid is None:
            self.valid = valid
        else:
            self.valid = torch.cat([self.valid, valid], dim=-1)
        r_ = self.valid.float().mean().item()
        self.flatten = True
        return thres, r_

    # ------------------------------------------------------------------ 写
    def _write(self, lo: int, hi: int, valid: torch.Tensor):
        pos = torch.arange(lo, hi, device=self.device)
        ctx = torch.enable_grad() if self.train_mode else torch.no_grad()
        with ctx:
            for l in range(self.n_layers):
                k, v = self._kv(l, pos)
                x = self.ctrl.feat(k, v)
                m_ret = valid[l]
                self.M[l] = self.ctrl.write(self.M[l], x, m_ret, ~m_ret,
                                            gen=self._gen)
                del k, v, x

    # ------------------------------------------------------------------ 采集
    def collect(self, lo: int, hi: int, valid: torch.Tensor, score0: torch.Tensor,
                thres: float, n_keep: int = 1024):
        """训练用：抽样候选并记录 (特征, 基线分, 掩码, 阈值距离)。

        **只抽阈值附近的**：离阈值很远的 token 无论 Δs 多大都翻不了，
        用它们做排序损失是在学一个恒真的排序。手工版的 flip rate 已经量化过
        这件事——β=0.5 只翻转 0.895% 的条目。
        """
        pos = torch.arange(lo, hi, device=self.device)
        out = []
        for l in range(self.n_layers):
            k, v = self._kv(l, pos)
            d = (score0[l, 0].float() - thres).abs()               # [H,n]
            idx = d.argsort(dim=-1)[:, :n_keep]                    # 最靠近阈值的
            out.append(dict(layer=l, idx=idx.cpu(),
                            k=torch.gather(k, 1, idx[..., None].expand(-1, -1, k.shape[-1])).cpu(),
                            v=torch.gather(v, 1, idx[..., None].expand(-1, -1, v.shape[-1])).cpu(),
                            s0=torch.gather(score0[l, 0].float(), 1, idx).cpu(),
                            ret=torch.gather(valid[l], 1, idx).cpu()))
            del k, v
        self.trace.append(dict(lo=lo, hi=hi, thres=thres, per_layer=out))
