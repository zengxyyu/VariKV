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

import json
import os

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
        # `VARIKV_CTRL_OFF=1`：**硬关**学习臂，`score` 保持 `score0` 不变 ⇒ 该次运行
        # 与不带 ckpt 的原生基线**逐位相同**，但仍保留 dump / 注入能力。
        # 为什么需要这个开关：跨方法精确移植要求捐赠方的配额是**干净**的，而
        # `--ctrlm_alpha 0` 走 logit 路径（`_p = 1e-6`）得到的是 alpha ≈ 1e-6·alpha_max
        # 而**不是精确 0**，`active` 仍为真、Δs 仍非零。逐 chunk 的微扰会改变保留集，
        # 进而污染**后续** chunk 的 `score0` —— 那正是 dump 要采的量。
        if os.environ.get("VARIKV_CTRL_OFF"):
            return False
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
                # **`VARIKV_CTRL_GAIN`：在学到的修正上乘一个实数增益 g。**
                # 需要它是因为「把注入表取负」**不是**干净的方向反号 ——
                # `project_quota` 含 clamp/round/rebalance，实测
                # `Π(b⁰−Δ)−b⁰` 与 `−[Π(b⁰+Δ)−b⁰]` 的不对称度中位 1.12、
                # 负向搬动量只有正向的 57%。在 `Δs` 上乘 g 则是**精确**的：
                # g=0 退回纯基线、g=1 原样、g=−1 严格反向。
                _g = float(os.environ.get("VARIKV_CTRL_GAIN", "1.0"))
                if _g != 1.0:
                    delta = delta * _g
                score = score0 + delta.to(score0.dtype)
                self.delta_std.append(float(delta.std()))

        valid, thres = self.threshold(score, ratio, level)          # [L,H,n]

        if self.active and ratio <= self.rho_max:
            with torch.no_grad():                                   # 自包含 flip rate
                v0, _ = self.threshold(score0, ratio, level)
                self.flip_frac.append(float((valid ^ v0).float().mean()))
                self.retain_delta.append(int(valid.sum()) - int(v0.sum()))

        # --- 评测时的保序自检（默认关闭，env 开）------------------------------
        # 「网络的选择 ≡ 它的配额向量」此前只在 fineweb trace 上验过（逆序对
        # 0/628,320、配额重放 22/22）。**评测分布上没验过** —— scbench 的 z 范围
        # 可能落到 `scratch_probe_monotone.py` 的网格证书 z∈[−14.65,87.36] 之外。
        # 这里在真实评测分数上直接比：把本臂的每头配额取出来，按 `s⁰` 原序重放，
        # 与本臂实际选出的掩码逐位 XOR。0 ⇒ 保序性在评测分布上成立。
        if os.environ.get("VARIKV_RANK_SELFCHECK") and self.active and ratio < 1:
            with torch.no_grad():
                _v0r, _ = self.threshold(score0, ratio, level)
                _s0 = score0[:, 0]; _L, _H, _n = _s0.shape
                _q = valid.reshape(_L * _H, _n).sum(-1)
                _idx = torch.argsort(_s0.reshape(_L * _H, _n), dim=-1, descending=True)
                _rp = torch.zeros(_L * _H, _n, dtype=torch.bool, device=_s0.device)
                _rp.scatter_(1, _idx,
                             torch.arange(_n, device=_s0.device)[None, :] < _q[:, None])
                _x = int((_rp.reshape(_L, _H, _n) ^ valid).sum())
                self._rank_xor = getattr(self, "_rank_xor", 0) + _x
                self._rank_tot = getattr(self, "_rank_tot", 0) + int(valid.numel())
                print(f"[rank-selfcheck] lo={lo} 配额重放与实际掩码不同 {_x}"
                      f" / {valid.numel()}  累计 {self._rank_xor}/{self._rank_tot}",
                      flush=True)

        # --- 静态配额注入：把网络整个换掉，只保留一张逐头配额表（默认关闭）------
        # 组内测得 `Δb` 有 97% 的方差由一个逐头常数解释（留出验证）。这里检验它够不够：
        # **丢掉网络、丢掉逐 token 修正、头内退回 `s⁰` 原序**，只按 `b_base + Δb_h`
        # 取 top-b。由保序重标定≡配额分配的等价定理，这与"某个保序打分器"完全等价。
        # 注意跨 panel 那张表不迁移（Retr.KV 与 MultiHop 相关 −0.204），所以这只是
        # **组内**命题的检验，不是一个通用方法。
        # --- 逐样本逐 chunk 的**绝对配额**注入（跨方法精确移植，默认关闭）---------
        # 与 `VARIKV_QUOTA_INJECT` 的区别：那个注入的是**增量表**（b⁰ + Δ），
        # 而且表是跨文档平均的。外部复核正确指出：平均表分不清「本方法排序不好」
        # 与「没给它这个文档真正的配额」—— 对同一 chunk 位置，不同文档的捐赠方
        # 配额本就不同，平均后再喂给文档 1，结果差无法归因。
        #
        # 精确移植：同一样本先跑捐赠方存下 `b^donor_{sample,chunk,l,h}`，再用本方法
        # 的排序按**完全相同**的配额重放 ⇒ **配额逐位相同，唯一变量是排序**。
        #
        # 对齐是这里最危险的地方（cache 每样本重建，样本号只能靠模块级计数），
        # 所以 npz 里同时存 `lo`/`hi`，每个 chunk 都断言匹配 —— **宁可崩，也不要
        # 静默错位**（错位会让实验看起来跑通、结果却是拿别的样本的配额）。
        _qa = os.environ.get("VARIKV_QUOTA_ABS")
        if _qa:
            with torch.no_grad():
                import numpy as _np
                g = globals()
                if not hasattr(self, "_qabs"):
                    if "_VARIKV_QABS" not in g:
                        z = _np.load(_qa)
                        g["_VARIKV_QABS"] = {k: z[k] for k in z.files}
                        g["_VARIKV_QABS_S"] = -1
                    g["_VARIKV_QABS_S"] += 1              # 新 cache = 新样本
                    self._qabs = g["_VARIKV_QABS"]
                    self._qabs_s = g["_VARIKV_QABS_S"]
                    self._qabs_c = 0
                Z, si, ci = self._qabs, self._qabs_s, self._qabs_c
                assert si < Z["quota"].shape[0], f"样本号 {si} 超出表 {Z['quota'].shape}"
                assert ci < int(Z["nchunk"][si]), f"chunk 号 {ci} 超出样本 {si}"
                assert int(Z["lo"][si, ci]) == int(lo) and int(Z["hi"][si, ci]) == int(hi), \
                    (f"配额表对齐失败：样本 {si} chunk {ci} 期望 "
                     f"lo/hi=({int(Z['lo'][si,ci])},{int(Z['hi'][si,ci])}) 实得 ({lo},{hi})")
                self._qabs_c += 1
                sc = score0[:, 0]
                L, H, n = sc.shape
                q = torch.as_tensor(Z["quota"][si, ci], dtype=torch.long,
                                    device=sc.device)
                assert q.numel() == L * H, f"配额维度 {q.numel()} != {L*H}"
                want = int(q.sum().item())
                q = q.clamp(0, n)
                assert int(q.sum().item()) == want, \
                    f"clamp 改变了总预算 {want} -> {int(q.sum().item())}（n={n} 太小）"
                idx = torch.argsort(sc.reshape(L * H, n), dim=-1, descending=True)
                nv = torch.zeros(L * H, n, dtype=torch.bool, device=sc.device)
                nv.scatter_(1, idx,
                            torch.arange(n, device=sc.device)[None, :] < q[:, None])
                valid = nv.reshape(L, H, n)

        _qi = os.environ.get("VARIKV_QUOTA_INJECT")
        if _qi:
            with torch.no_grad():
                import numpy as _np
                if not hasattr(self, "_qinj_raw"):
                    self._qinj_raw = torch.as_tensor(_np.load(_qi), dtype=torch.float32)
                    self._qinj_ci = 0
                    # 2D 表 [C, 112] = 逐 (chunk 位置, 头)。方差分解显示网络在 panel
                    # 内部 99.8% 的行为由 (头) + (头 × chunk 位置) 解释，真正依赖文档
                    # 内容的残差只有 0.1–0.2%，所以位置索引表能近乎完整复现网络。
                if self._qinj_raw.dim() == 2:
                    # score0 是 [L,1,H,n]，含全部层 ⇒ 本块**每 chunk 执行一次**，
                    # 位置计数每次 +1。`lo` 回退表示换了一条新序列，计数归零。
                    if lo < getattr(self, "_qinj_lo", 1 << 62):
                        self._qinj_ci = 0
                    self._qinj_lo = lo
                    self._qinj = self._qinj_raw[
                        min(self._qinj_ci, self._qinj_raw.shape[0] - 1)]
                    self._qinj_ci += 1
                else:
                    self._qinj = self._qinj_raw
                sc = score0[:, 0]                                  # [L,H,n]
                L, H, n = sc.shape
                vb, _ = self.threshold(score0, ratio, level)       # 基线掩码
                b0 = vb.sum(-1).reshape(-1).float()                # [L*H]
                # **总预算必须与基线严格相等**，否则比的就不是同一个压缩率了。
                # **投影逻辑的唯一实现在 attention/quota_project.py** —— 生产与
                # `scratch_test_project.py` 共用同一个函数，禁止镜像复制（镜像会让
                # "生产改了测试没改"或"两边同错"时测试仍然全绿）。非 full 模式的预算
                # 守恒是构造性的，`project_quota` 内部直接断言，**不做兜底修补** ——
                # 若投影出 bug 必须让它崩，而不是被通用修补循环悄悄改成另一个干预。
                from attention.quota_project import project_quota
                _qm = os.environ.get("VARIKV_QUOTA_MODE", "full")
                # `sc` 只有 floorproj 用得到（把地板目标投影回可达集，见
                # quota_project.reachable_project 的 docstring）；其余模式忽略它，
                # 默认路径逐字节不变。
                bt = project_quota(b0, self._qinj.to(b0.device), n, _qm, L, H, sc=sc)
                idx = torch.argsort(sc.reshape(L * H, n), dim=-1, descending=True)
                nv = torch.zeros(L * H, n, dtype=torch.bool, device=sc.device)
                ar = torch.arange(n, device=sc.device)[None, :]
                nv.scatter_(1, idx, ar < bt[:, None])
                new_valid = nv.reshape(L, H, n)
                # **进程内逐位自检**：Δb=0 时注入必须与基线阈值选出**同一批 token**，
                # 不只是同样的计数。外部复核正确指出「配额相同 ≠ 集合相同」；这里在
                # 同一进程、同一 score0 上直接比，绕开跨运行的数值不确定性。
                # 数学上二者应当恒等（|{s>τ}|=b₀ ⇒ 那 b₀ 个就是最大的 b₀ 个；平局只
                # 出现在 =τ 处、排在其后），真实分数上已离线验过 2464/2464 同集。
                if os.environ.get("VARIKV_INJECT_SELFCHECK"):
                    diff = int((new_valid ^ vb).sum())
                    self._inj_xor = getattr(self, "_inj_xor", 0) + diff
                    self._inj_tot = getattr(self, "_inj_tot", 0) + int(vb.numel())
                    print(f"[inject-selfcheck] chunk lo={lo} 掩码逐位不同 {diff}"
                          f" / {vb.numel()}  累计 {self._inj_xor}/{self._inj_tot}",
                          flush=True)
                valid = new_valid

        # --- 逐 (chunk, 层, kv头) 真实配额导出（默认关闭，env 开）---------------
        # 保序重标定 ≡ 逐头配额分配（ICLR_PLAN §四之五）已经证明并在 trace 上验过，
        # 但 trace 每 (chunk,层,头) 只存 768 个候选，`b_{c,h}` 的**绝对值**不是推理时
        # 的真实配额。context-quota shuffle 与 kv quota replay 都要真实值，所以在这里
        # 落盘 —— 这是唯一同时拿得到 `valid`（本臂）与 `v0`（基线）的地方。
        # **默认路径逐字节不变**：env 未设时下面整块不执行。
        _qd = os.environ.get("VARIKV_QUOTA_DUMP")
        if _qd:
            with torch.no_grad():
                _v0 = v0 if (self.active and ratio <= self.rho_max) else \
                    self.threshold(score0, ratio, level)[0]
                _sc = score0[:, 0]                       # [L,H,n]，与 _v0 同形
                self._qseq = getattr(self, "_qseq", 0) + 1
                with open(_qd, "a") as _f:
                    _f.write(json.dumps({
                        "seq": self._qseq, "lo": int(lo), "hi": int(hi),
                        "ratio": float(ratio), "level": level,
                        "thres": float(thres),
                        "b_arm": valid.sum(-1).flatten().tolist(),    # [L*H]
                        "b_base": _v0.sum(-1).flatten().tolist(),
                        # **被驱逐的分数质量** —— 推理时完全可得的无标签信号，
                        # 是「丢了多少信息」的代理。现有三个候选门控输入（零配额头
                        # 比例、配额熵、Gini）都只描述**配额分布的形状**，实测对
                        # 「该不该校准」没有预测力（Spearman +0.393/−0.143/+0.143）；
                        # 唯一有预测力的 slack 需要任务标签（+0.919），不能当门控。
                        # 分数可能有负值，所以同时记原始和与 softplus 和，离线再挑。
                        "s_tot": float(_sc.sum()),
                        "s_ret": float(_sc[_v0].sum()),
                        "sp_tot": float(torch.nn.functional.softplus(_sc).sum()),
                        "sp_ret": float(torch.nn.functional.softplus(_sc[_v0]).sum()),
                    }) + "\n")

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
        # **memoryless 直接短路。** `ControlMemory.write` 在 memoryless 下立刻原样
        # 返回状态，但走到那一步之前，下面已经把 28 层的 KV gather 出来、算完
        # `feat()` 了 —— 纯浪费。不是正确性问题，但每个 chunk 白算 28 次投影。
        if getattr(self.ctrl, "mode", None) == "memoryless":
            return
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
