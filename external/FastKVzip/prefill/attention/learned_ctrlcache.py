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
                 train_mode: bool = False, seed: int = 0, n_write: int = 512,
                 rho_max: float = 1.0):
        super().__init__(model, evict_range)
        self.ctrl = ctrl                       # ControlMemory，None ⇒ 纯基线
        self.train_mode = train_mode
        self.head_dim = getattr(
            model.config, "head_dim",
            model.config.hidden_size // model.config.num_attention_heads)
        # **写入端固定采样规模，与训练一致。** 训练时 writer 只看 teacher 存下的
        # 512 个随机候选，推理若把整个 chunk（≈16000）交给它，均值路虽无偏，
        # 但注意力池化的 softmax 集中度（尤其 max logit 的分布）在两种规模下差别很大
        # ⇒ M_gru 的训练/部署分布不一致。统一到同一个采样规模，顺带让写入开销
        # 与 chunk 大小解耦。
        self.n_write = int(n_write)
        # **按预算门控修正**：ratio > rho_max 时不施加 Δs。
        # 依据是实测的曲线形状：残差在宽松预算上让出 1–4 分（基线本来近乎无损），
        # 在 0.2 上补回 18.8 分 —— 也就是它"把曲线压平"。压平在紧预算侧是收益、
        # 在宽松侧是纯损失，所以不该无条件施加。
        # 关键是**这个门控可部署**：ratio 在推理时已知，而 headroom 不知道
        # （要跑满缓存才能算）。
        self.rho_max = float(rho_max)
        self.M = None                          # [L][(M_gru, M_dir)]，惰性初始化
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        # 诊断
        self.flip_frac, self.retain_delta, self.delta_std = [], [], []
        # 训练用采集缓冲（train_mode 时填）
        self.trace = []

    @property
    def active(self) -> bool:
        return self.ctrl is not None and float(self.ctrl.alpha) != 0.0

    def _ensure_state(self):
        """self.M[l] 是 (M_gru, M_dir) 二元组——read/write 的签名已按二元组定义。"""
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
            # 全局阈值要先算一次（用未修正的分数），供 margin 特征使用。
            # 这不是泄漏：τ 只依赖 s0 与 ratio，推理时同样拿得到。
            with torch.no_grad():
                _, thr_g = self.threshold(score0, ratio, level)
                f0 = score0[:, 0].float()                   # [L,H,n]，整块全量
                gsig = f0.std().clamp_min(1e-6)
                mu_h, sig_h = f0.mean(-1), f0.std(-1)       # [L,H]
            with ctx:
                delta = torch.zeros_like(score0)
                feats = []
                for l in range(self.n_layers):
                    k, v = self._kv(l, pos)
                    # **必须走 raw/feat/read 三件套**：feat() 只收一个已拼好的
                    # [k;v]，read() 要的是 raw 而不是投影后的 x（读出 query 用的是
                    # 独立于 x_proj 的投影）。此前这里还停在重构前的两参数签名，
                    # 评测路径一跑就 TypeError——trainer 同步了，cache 漏了。
                    xr_raw = self.ctrl.raw(k, v)                    # [H,n,2d]
                    x = self.ctrl.feat(xr_raw)                      # [H,n,d_m]
                    q = self.ctrl.q_read(xr_raw)
                    r = self.ctrl.read(self.M[l], xr_raw)
                    s0l = score0[l, 0].float()
                    mg = None if thr_g is None else (s0l - thr_g) / gsig
                    # **统计量显式传入**，不让 delta 从手上这批候选现算——训练侧
                    # 拿到的是有偏子集，两边口径必须一致（control_memory.delta 的说明）
                    delta[l, 0] = self.ctrl.delta(
                        x, r, s0l, q=q, margin=mg,
                        stats=(mu_h[l], sig_h[l], gsig))
                    # **每层用完即弃**：[H,n,d_m] 每层 16 MB，28 层留着就是 450 MB。
                    # 手工版正是在这里踩过 917 MB 的坑。写入阶段重算一次特征更划算。
                    feats.append(None)
                    del k, v, x, r, q, xr_raw
            if getattr(self.ctrl, "replace", False):
                score = delta.to(score0.dtype)         # 独立打分器：完全不用 s⁰
            elif self.active and ratio <= self.rho_max:
                score = score0 + delta.to(score0.dtype)
                self.delta_std.append(float(delta.std()))

        valid, thres = self.threshold(score, ratio, level)          # [L,H,n]

        if self.active and ratio <= self.rho_max:
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
        n = hi - lo
        if self.n_write and n > self.n_write:      # 与训练同规模的随机子样本
            sub = torch.stack([torch.randperm(n, generator=self._gen)[:self.n_write]
                               for _ in range(self.n_heads_kv)]).to(self.device)
        else:
            sub = torch.arange(n, device=self.device).expand(self.n_heads_kv, -1)
        pos = sub + lo
        ctx = torch.enable_grad() if self.train_mode else torch.no_grad()
        with ctx:
            for l in range(self.n_layers):
                # 逐头取各自的子样本位置
                k = torch.stack([self.key_cache[l][0][h, pos[h]]
                                 for h in range(self.n_heads_kv)])
                v = torch.stack([self.value_cache[l][0][h, pos[h]]
                                 for h in range(self.n_heads_kv)])
                x = self.ctrl.feat(self.ctrl.raw(k, v))
                m_ret = torch.gather(valid[l], 1, sub)
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
