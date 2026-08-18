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
                tgt = (b0 + self._qinj.to(b0.device)).clamp(0, n)
                # **总预算必须与基线严格相等**，否则比的就不是同一个压缩率了。
                # 单轮最大余数法**不够**：`tgt` 先被 clamp 到 [0,n]，clamp 造成的缺口
                # 可能远大于头数，而每头一轮只能移动 ±1。实测 Δb=±9999 时缺口 18、
                # 可减头只有 11，一轮补不完，总预算从 179 变成 186。改成迭代配平。
                Btot = int(b0.sum().item())
                # ---- 竞争域受限投影：`within` / `across` 消融 -------------------
                # **关键更正（2026-08-18）**：`within-only` 与 `across-only`
                # **不能**表示成固定的 112 维加性表。离线审计（复现本段逻辑，220 个
                # 真实 chunk）显示，把理论分量当加性表喂进来、再用**全局**最大余数
                # 配平，会把层内干预偷偷变成跨层干预：
                #     within 表的逐层总量漂移均值 938.9 槽/层，仅 5.83% 的层无漂移；
                #     cos(实现within, 实现across) = +0.4509（理论应为 0）。
                # 原因是 clamp(0,n) 在零配额头上截断（within 表每 chunk 有 36 格被
                # 截），缺口由配平循环在**全局**范围内补，不受层内约束。
                # 正确做法是把约束写进投影，而它依赖运行时的 `b0`：
                #     within : 每层总量 = 基线层总量，层内按表分配（层内配平）
                #     across : 每层总量 = 基线 + L_l，层内按**基线比例**分配
                # 两者不要求相加等于 full —— 离散化后可加性本就不成立，这里是
                # **各自隔离一个通道**，不是做加性分解。
                _qm = os.environ.get("VARIKV_QUOTA_MODE", "full")
                assert _qm in ("full", "within", "across"), _qm
                nL = L

                def _rebal(bt_, tgt_, tot, hi):
                    # 把整数向量配平到 sum==tot，按小数余数优先，界 [0,hi]
                    df = tot - int(bt_.sum().item())
                    while df != 0:
                        if df > 0:
                            rm = (bt_ < hi).nonzero().flatten()
                            if rm.numel() == 0:
                                break
                            tk = min(df, rm.numel())
                            pk = rm[torch.argsort((tgt_ - bt_.float())[rm],
                                                  descending=True)[:tk]]
                            bt_[pk] += 1; df -= tk
                        else:
                            rm = (bt_ > 0).nonzero().flatten()
                            if rm.numel() == 0:
                                break
                            tk = min(-df, rm.numel())
                            pk = rm[torch.argsort((tgt_ - bt_.float())[rm])[:tk]]
                            bt_[pk] -= 1; df += tk
                    return bt_

                if _qm != "full":
                    # **构造性定义（2026-08-18 二次修正）**：两个消融从**表**上就分开，
                    # 不再共用 full 表让投影去"自动消掉"多余分量。
                    #     Δ^W_{l,h} = Δ_{l,h} − mean_h Δ_{l,·}   （逐层去均值）
                    #     Δ^A_l     = Σ_h Δ_{l,h}                （层净变化）
                    # 为什么必须显式去均值：零配额头上的 clamp(0,n) 打破对称性，
                    # 层常数分量会**借 clamp 泄漏**进层内再分配。单测（Δ_lh = c_l 在
                    # within 下必须 no-op）在旧写法上 20/20 失败。实测泄漏只占搬动量
                    # 1.21%、99.1% 的格逐位相同，所以 `_p02win2` 的既有结果仍可用，
                    # 但定义现在是构造性的。
                    b0m = b0.reshape(nL, H)
                    tbm = self._qinj.to(b0.device).reshape(nL, H)
                    bt = torch.zeros(nL, H, dtype=torch.long, device=b0.device)
                    if _qm == "within":
                        tbw = tbm - tbm.mean(1, keepdim=True)
                        for _l in range(nL):
                            base_l = int(b0m[_l].sum().item())
                            t_l = (b0m[_l] + tbw[_l]).clamp(0, n)
                            bt[_l] = _rebal(t_l.round().long().clamp(0, n), t_l,
                                            base_l, n)
                        bt = bt.reshape(-1)
                    else:                                   # across，两级整数投影
                        # 一级：先把 28 个层总量整数化并**严格**配平到 Btot。
                        # 旧写法在层内 round 后再做 112 维全局配平，会让 ±1 落到
                        # 任意 (层,头) 上 —— 实测偏离 ≤2 槽/层、4 槽/chunk
                        # （占预算 0.0013%），量级可忽略但定义不干净，故改掉。
                        d_l = b0m.sum(1) + tbm.sum(1)
                        Bl = _rebal(d_l.round().long().clamp(0, H * n), d_l,
                                    Btot, H * n)
                        # 二级：层内按**基线比例**分配，逐层严格配平到 B'_l
                        for _l in range(nL):
                            base_l = float(b0m[_l].sum())
                            tot_l = int(Bl[_l].item())
                            t_l = (b0m[_l] * (tot_l / base_l) if base_l > 0
                                   else torch.full((H,), tot_l / H,
                                                   device=b0.device)).clamp(0, n)
                            bt[_l] = _rebal(t_l.round().long().clamp(0, n), t_l,
                                            tot_l, n)
                        bt = bt.reshape(-1)
                        # **不再做 112 维全局配平** —— 一级已保证 Σ B'_l = Btot，
                        # 二级逐层严格配平，故总和构造性相等。
                else:
                    bt = tgt.round().long().clamp(0, n)
                diff = Btot - int(bt.sum().item())
                while diff != 0:
                    if diff > 0:
                        room = (bt < n).nonzero().flatten()
                        if room.numel() == 0:
                            break
                        take = min(diff, room.numel())
                        pick = room[torch.argsort((tgt - bt.float())[room],
                                                  descending=True)[:take]]
                        bt[pick] += 1; diff -= take
                    else:
                        room = (bt > 0).nonzero().flatten()
                        if room.numel() == 0:
                            break
                        take = min(-diff, room.numel())
                        pick = room[torch.argsort((tgt - bt.float())[room])[:take]]
                        bt[pick] -= 1; diff += take
                # 宁可崩，也不能悄悄改变压缩率 —— 那样比出来的分数是无效的。
                assert int(bt.sum().item()) == Btot, \
                    f"配额注入后预算 {int(bt.sum().item())} != 基线 {Btot}"
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
