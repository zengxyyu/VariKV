"""质心的剩余误差是「质量」还是「方向」？—— (mass × direction) oracle 2×2 + γ-sweep

**为什么是这个参数化，而不是「分子 × 分母」。** 质心的读出是
`o = λ·o_R + (1−λ)·v̂`，`λ = D_R/(D_R+D̂_E)`，所以被驱逐贡献只由两个量决定：

    D_E  质量 —— 被驱逐集合该占多少 softmax mass
    v_E = N_E/D_E  方向 —— 只在被驱逐集合内部做 attention 会输出什么

把 `N̂_E → N_E` 而保留 `D̂_E` **不是**干净的方向 oracle：`N̂_E = D̂_E·v̂_E`，两者共用
同一个 Jensen 偏置权重 `n_j e^{aᵀk̄_j}`，替换其一会破坏分子/分母的一致归一化。
设偏置近似统一 `D̂_E=ρD_E`、`N̂_E=ρN_E`，则 `v̂_E=v_E` 精确成立而
`(N_R+ρN_E)/(D_R+D_E)` 把被驱逐项按 ρ 补、按 1 稀释 ⇒ 人为退化，得到负结果也
说明不了任何事。(mass, direction) 才是正交的两个旋钮。

**四格**（`o(D,v) = λ(D)·o_R + (1−λ(D))·v`，`λ(D)=D_R/(D_R+D)`）：

|              | 质量 `D̂_E` | 质量 `D_E` |
|---|---|---|
| 方向 `v̂_E` | 现方法 | **Oracle-Mass** |
| 方向 `v_E` | **Oracle-Direction** | = 满缓存（内建自检，误差应 ≈0） |

**γ-sweep（比二值 oracle 信息量大得多）**：保持 `v̂_E` 不动，只缩放真实质量

    o(γ) = λ(γD_E)·o_R + (1−λ(γD_E))·v̂_E,    γ ∈ {0, .05, .1, .25, .5, .75, 1}

现方法的等效 γ 是 `D̂_E/D_E`，P0 测的 log 中位 −2.57 ⇒ **γ ≈ 0.077（低估 13×）**。

**定量预注册。** 设 `v̂ = v + ε`、`ε` 与 `v` 无关，平方损失下最优收缩是
`γ* = ‖v‖²/(‖v‖²+E‖ε‖²) = 1/(1+e²)`，`e = ‖v̂−v‖/‖v‖`。P0 测 `e` 中位 0.666
（只用 μ_v，正是本方法用的）⇒ **预测 γ* ≈ 0.69**；用全协方差 e=0.438 ⇒ 0.84。

    峰值在 0.7–0.85  ⇒ 收缩理论成立，现方法**欠修正约 9 倍**，质量校正值得做；
                       E4 的 +1249% 灾难则来自它同时改了方向估计，不是来自修准质量
    峰值在 0.08 附近 ⇒ 平方损失收缩不适用 ⇒ ε 不是与 v 无关的噪声，而是**系统性
                       对齐在有害方向**，或下游被尾部主导。这才是真正有意思的结果
    单调递减到 0     ⇒ 修正整体负价值，质心的收益来自别处

**同时补一个此前只靠推断的量。** P0 §5.3 的 0.438 是朴素相对 L2 误差，
**没有记 `‖v̂‖/‖v‖`**，所以不能用 `cos=√(1−e²)` 反推余弦（那只在 `r=cosθ`
即最优缩放时成立，一般情况是 `cosθ=(r²+1−e²)/(2r)`）。这里直接同时输出
`e`、`r`、`cos`，把这个缺口填掉。

**跨头相加后的层级量也要报**：P0 测跨头相消只留下 0.25，所以逐头对齐好 ≠ 层级对齐好。

用法：CUDA_VISIBLE_DEVICES=k .venv/bin/python scratch_probe_massdir.py \
          --data scbench_kv --K 16 --n 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external/FastKVzip/prefill"))

from model.wrapper import ModelKVzip                     # noqa: E402
from attention.kvcache import RetainCache                # noqa: E402
from data.load import load_dataset_all                   # noqa: E402
from data.wrapper import DataWrapper, get_query          # noqa: E402

GAMMAS = [0.0, 0.02, 0.05, 0.077, 0.1, 0.25, 0.5, 0.75, 1.0]
_Q = {}
_orig_prepare = RetainCache.prepare


def _patched_prepare(self, q, k, v, l):
    _Q[l] = q.detach().clone()
    return _orig_prepare(self, q, k, v, l)


def cs(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="scbench_kv")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.1)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(28)))
    ap.add_argument("--mem_frac", type=float, default=0.45,
                    help="自身显存上限占比；与其他任务共卡时的保护栏，0 = 不限制")
    a = ap.parse_args()

    RetainCache.prepare = _patched_prepare
    # **与在跑的 eval 任务共卡**：给自己设显存上限，宁可本探针 OOM，也不要把邻居
    # 那个 3 小时的评估任务饿死。80 GB 卡上 0.45 ⇒ 36 GB，探针实测约 30 GB。
    if a.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    m = ModelKVzip("Qwen/Qwen2.5-7B-Instruct-1M", kv_type="centroid",
                   gate_path_or_name="fastkvzip")
    m.varikv_K = a.K
    m.varikv_rope_mode = "post"
    H = m.config.num_key_value_heads
    d = getattr(m.config, "head_dim", m.config.hidden_size // m.config.num_attention_heads)
    print(f"[cfg] {a.data} K={a.K} ratio={a.ratio} H={H} d={d} 样本 {a.n} 条", flush=True)

    ds = load_dataset_all(a.data, m.tokenizer)
    dw = DataWrapper(a.data, ds, m)
    if a.n <= 0:                      # --n 0 ⇒ 用整个数据集，与下游评估同口径
        a.n = len(ds) - a.start
        print(f"[cfg] --n 0 ⇒ 全量 {a.n} 条", flush=True)

    rows, lay_rows, gam = [], [], []
    per_sample = []          # 每条样本一行：(逐头误差中位 ×4, 该样本的最优 γ, log r_D 中位)
    for i in range(a.start, a.start + a.n):
        n_before = len(rows)
        kv = dw.prefill_context(i, prefill_chunk=16000, window_size=4096,
                                chunk_ratio=a.ratio, level="pair")
        q_ids = m.apply_template(get_query("qa", list(ds[i]["question"])[0])).to(m.device)
        _Q.clear()
        m.model(q_ids, past_key_values=kv)
        S = kv.key_cache[0].shape[2]
        for l in a.layers:
            WO = m.model.model.layers[l].self_attn.o_proj.weight.detach().float()
            kbar, vbar, logn = kv._summary(l, torch.float32)     # [H,K,d],[H,K,d],[H,K]
            v_ = kv._get_valid(l, S)
            while v_.dim() > 2:
                v_ = v_.squeeze(0)
            valid = v_.bool().to(m.device)
            kall = kv.key_cache[l][0].float(); vall = kv.value_cache[l][0].float()
            T = _Q[l].shape[2]; G = _Q[l].shape[1] // H
            agg = {k: torch.zeros(WO.shape[0], device=m.device)
                   for k in ("full", "cc", "mc", "cd", "R")}
            for h in range(H):
                ev = (~valid[h]).nonzero(as_tuple=True)[0]
                if ev.numel() < 64:
                    continue
                k_ev, v_ev = kall[h, ev], vall[h, ev]
                # 与 centroid.py:183 的分簇一致：b = (pos // W).clamp(max=K-1)。
                # **不要再加 sink**：`_get_valid` 已在前面 pad 了 sink 个 True，所以返回的
                # mask 与 key_cache 对齐 ⇒ `ev` 本身就是原始 token 位置。
                # （`_absorb_layer` 里的 `pos = drop + s + sink` 是因为它索引的是
                #  `self.valid`，那个数组相对 key_cache 少了前面的 sink 段。）
                b_of = (ev // kv.W).clamp(max=a.K - 1)
                k_rt, v_rt = kall[h, valid[h]], vall[h, valid[h]]
                for g in range(G):
                    hq = h * G + g
                    W = WO[:, hq * d:(hq + 1) * d]
                    aq = _Q[l][0].view(H, G, T, d)[h, g, -1].float() * (d ** -0.5)
                    # ---- 精确 ----
                    sR, sE = aq @ k_rt.T, aq @ k_ev.T
                    LR, LE = torch.logsumexp(sR, -1), torch.logsumexp(sE, -1)
                    oR = torch.softmax(sR, -1) @ v_rt
                    vE = torch.softmax(sE, -1) @ v_ev            # = N_E/D_E，方向
                    # ---- 质心（与 memory_correct 逐字同构）----
                    sbar = aq @ kbar[h].T                    # [K]，下面复用
                    r = sbar + logn[h]
                    LEh = torch.logsumexp(r, -1)
                    vEh = torch.softmax(r, -1) @ vbar[h]
                    # ---- 重做 P0 的 E 阶梯：那批结论只有 3 样本 × 2 问题 = 6 个 query ----
                    # 「越多矩越差且单调」「E4 修准质量是灾难」是整个机制故事的地基，
                    # 却是全仓库样本量最小的结论之一。这里在同一批样本上重新检验，
                    # 而且**把质量与方向解耦**（P0 的 E4 同时改了两者，无法归因）。
                    #
                    # **必须用 scatter，不能用外积。** 朴素写法 `k_ev[None]-kbar[h][:,None]`
                    # 是 [K, n_ev, d]：K=16 时 1.1 GB、**K=1024 时 73 GB ⇒ 必然 OOM**，
                    # 而且它不依赖 g 却在 g 循环里被重复分配 G 次。下面用 index_add 把
                    # 复杂度从 O(K·n·d) 降到 O(n·d)，全量样本 × 全 28 层才跑得动。
                    #   δ_i      = aᵀk_i − aᵀk̄_{b(i)}
                    #   Var_j    = mean_{i∈j} δ_i²
                    #   (Σ_vk a)_j = mean_{i∈j}(v_i−v̄_j)·δ_i
                    #              = (Σ_{i∈j} v_i δ_i)/n_j − v̄_j·(Σ_{i∈j} δ_i)/n_j
                    delta = sE - sbar[b_of]                            # [n]
                    z1 = torch.zeros(a.K, device=aq.device, dtype=torch.float32)
                    cnt_ = z1.clone().index_add_(0, b_of, torch.ones_like(delta))
                    cl = cnt_.clamp_min(1.0)
                    var_ = z1.clone().index_add_(0, b_of, delta * delta) / cl
                    sum_d = z1.clone().index_add_(0, b_of, delta)
                    sum_vd = torch.zeros(a.K, d, device=aq.device, dtype=torch.float32)
                    sum_vd.index_add_(0, b_of, v_ev * delta[:, None])
                    cov_a = sum_vd / cl[:, None] - vbar[h] * (sum_d / cl)[:, None]
                    if h == 0 and g == 0:
                        # **自检 off-by-sink**：这里重算的簇计数必须与缓存自己在吸收时
                        # 累计的 exp(logn) 一致；不一致说明分簇口径对不上。
                        ref = torch.where(logn[h] > -1e29, logn[h].exp(),
                                          torch.zeros_like(cnt_))
                        assert torch.allclose(cnt_, ref, atol=1.0), \
                            f"层 {l} 簇计数不符：重算 {cnt_.sum():.0f} vs 缓存 {ref.sum():.0f}"
                    logn2 = torch.where(logn[h] > -1e29, logn[h] + 0.5 * var_, logn[h])
                    r2 = sbar + logn2
                    LE2 = torch.logsumexp(r2, -1)                      # 二阶累积量质量
                    vEc = torch.softmax(r, -1) @ (vbar[h] + cov_a)     # 只换方向（一阶权重）
                    # **真正的 joint Gaussian 必须用二阶权重给簇加权。**
                    # joint 公式是  N̂_E = Σ_j e^{r2_j}(μ_v,j + Σ_vk,j a)，
                    # 所以方向是 softmax(**r2**) @ (μ_v + Σ_vk a)，不是 softmax(r)。
                    # 旧的「两者都换」格拿二阶的**总质量**配一阶的**簇间权重**，
                    # 那不是任何一个估计器：设两簇一阶 logits 相等（权重 0.5/0.5）、
                    # 而第二簇 ½aᵀΣa=3，则真 joint 应给它 0.953 的权重，旧写法仍给 0.5。
                    # 方向误差会因此很大。保留旧格（列 14）作对照，新格是列 16。
                    vEc2 = torch.softmax(r2, -1) @ (vbar[h] + cov_a)
                    # ---- exact-MGF oracle：逐簇的精确 logsumexp ----
                    # 有了它才能把 Jensen 缺口分解成「二阶吃掉多少 / 高阶累积量剩多少」：
                    #   r_ex − r0 = 一阶漏掉的全部（Jensen 缺口）
                    #   r_ex − r2 = 二阶之后还剩的（κ₃/6 + κ₄/24 + …）
                    # 注意 logsumexp_j(r_ex_j) 恒等于 LE，所以"精确质量"不是新信息；
                    # 有信息的是**逐簇**的两个残差之比。
                    _ninf = torch.full_like(z1, float("-inf"))
                    mx = _ninf.scatter_reduce(0, b_of, sE, "amax", include_self=True)
                    ex = z1.clone().index_add_(0, b_of, (sE - mx[b_of]).exp())
                    r_ex = mx + ex.clamp_min(1e-30).log()              # [K]
                    _oc = cnt_ > 0
                    # **Jensen 保证 r_ex ≥ r0**（`Σexp(aᵀk_i) ≥ n·exp(aᵀk̄)`），所以
                    # 带符号的缺口理论上非负。用 abs 会把违反掩掉，而违反恰恰说明
                    # 簇均值/分组/精度出了问题 —— 单独记 signed 与越界比例作自检。
                    gs = (r_ex - (sbar + logn[h]))[_oc]                # signed Jensen 缺口
                    g0 = gs.abs()
                    g2 = (r_ex - r2)[_oc].abs()                        # 二阶后的残差
                    jg0 = float(g0.median()) if g0.numel() else float("nan")
                    jg2 = float(g2.median()) if g2.numel() else float("nan")
                    jneg = float((gs < -1e-4).float().mean()) if gs.numel() else float("nan")
                    # **「二阶只是簇间公共平移」这个解释此前只是推断，这里直接测。**
                    # 若 shift 的跨簇散布 ≈ 0，softmax 里就被完全约掉，于是"改总质量很多、
                    # 改相对权重几乎为零"就闭环了。KL 是同一件事的直接读数。
                    sh = (0.5 * var_)[_oc]
                    w0 = torch.softmax(r, -1)[_oc]
                    w2 = torch.softmax(r2, -1)[_oc]
                    shsd = float(sh.std()) if sh.numel() > 1 else 0.0
                    shrg = float(sh.max() - sh.min()) if sh.numel() else float("nan")
                    klw = float((w2 * ((w2 + 1e-30).log()
                                       - (w0 + 1e-30).log())).sum())
                    # ---- 四格 ----
                    def cell(LEx, vx):
                        lam = torch.exp(LR - torch.logaddexp(LR, LEx))
                        return lam * oR + (1 - lam) * vx
                    o_full, o_cc = cell(LE, vE), cell(LEh, vEh)
                    o_mc, o_cd = cell(LE, vEh), cell(LEh, vE)
                    dfull = o_full - oR                          # = Δo
                    nd = (W @ dfull).norm().clamp_min(1e-12)
                    e = float((vEh - vE).norm() / vE.norm().clamp_min(1e-30))
                    rows.append((
                        l, hq, float(LEh - LE),                  # log r_D
                        e, float(vEh.norm() / vE.norm().clamp_min(1e-30)), cs(vEh, vE),
                        float((W @ (o_cc - o_full)).norm() / nd),
                        float((W @ (o_mc - o_full)).norm() / nd),
                        float((W @ (o_cd - o_full)).norm() / nd),
                        cs(W @ (o_cc - oR), W @ dfull),
                        float((W @ (o_cc - oR)).norm() / nd),
                        1.0 / (1.0 + e * e),                     # 预测 γ*
                        float((W @ (cell(LE2, vEh) - o_full)).norm() / nd),   # E4 质量
                        float((W @ (cell(LEh, vEc) - o_full)).norm() / nd),   # E2 方向
                        float((W @ (cell(LE2, vEc) - o_full)).norm() / nd),   # 14 旧「两者都换」**权重错**
                        float((vEc - vE).norm() / vE.norm().clamp_min(1e-30)),  # 15 e(+Σvk)
                        # **追加列，不前插** —— 前插会改动所有既有报告的列号
                        float((W @ (cell(LE2, vEc2) - o_full)).norm() / nd),  # 16 真 joint
                        float((vEc2 - vE).norm() / vE.norm().clamp_min(1e-30)),  # 17 e(joint)
                        jg0, jg2,                                             # 18/19 Jensen 缺口
                        (1.0 - jg2 / jg0) if jg0 > 1e-12 else float("nan"),   # 20 二阶解释率
                        shsd, shrg, klw, jneg,                        # 21-24 公共平移自检
                    ))
                    # ---- γ-sweep（保持 v̂ 不变，只缩放真实质量）----
                    errs = []
                    for gm in GAMMAS:
                        LEg = LE + np.log(gm) if gm > 0 else torch.tensor(
                            -1e30, device=LE.device, dtype=LE.dtype)
                        errs.append(float((W @ (cell(LEg, vEh) - o_full)).norm() / nd))
                    gam.append(errs)
                    for key, o_ in (("full", o_full), ("cc", o_cc), ("mc", o_mc),
                                    ("cd", o_cd), ("R", oR)):
                        agg[key] += W @ o_
            dl = agg["full"] - agg["R"]
            nl = dl.norm().clamp_min(1e-12)
            lay_rows.append((l,
                             float((agg["cc"] - agg["full"]).norm() / nl),
                             float((agg["mc"] - agg["full"]).norm() / nl),
                             float((agg["cd"] - agg["full"]).norm() / nl),
                             cs(agg["cc"] - agg["R"], dl)))
        del kv
        torch.cuda.empty_cache()
        # **统计单位是样本，不是 (层,头) 实例** —— 280 个实例嵌套在 2 条样本里并不独立，
        # 上下文内容/驱逐集合/query 都是样本级共享的。逐样本聚合后再在样本上 bootstrap。
        Ai = np.array(rows[n_before:]); Gi = np.array(gam[n_before:])
        med_i = [float(np.median(Gi[:, j])) for j in range(len(GAMMAS))]
        per_sample.append((float(np.median(Ai[:, 6])), float(np.median(Ai[:, 7])),
                           float(np.median(Ai[:, 8])), GAMMAS[int(np.argmin(med_i))],
                           float(np.median(Ai[:, 2])), float(np.median(Ai[:, 3]))))
        print(f"  样本 {i}: 现方法 {per_sample[-1][0]:.4f}  Oracle-Mass "
              f"{per_sample[-1][1]:.4f}  Oracle-Dir {per_sample[-1][2]:.4f}  "
              f"该样本最优 γ {per_sample[-1][3]}  log r_D {per_sample[-1][4]:+.3f}",
              flush=True)

    A = np.array(rows); Gm = np.array(gam); B = np.array(lay_rows)
    # **一定要存盘。** 这一跑要 40 分钟 × 20 篇 169k 预填；上一次因为算了列 16–20
    # 却忘了打印，整跑 GPU 时间全废了，只能重来。存了盘就再也不会。
    np.save(str(ROOT / f"scratch_massdir_{a.data}_K{a.K}.npy"), A)
    md = lambda c: float(np.median(A[:, c]))                     # noqa: E731
    print("\n" + "=" * 96)
    print(f"(mass × direction) oracle 2×2　{a.data} @ratio {a.ratio}　K={a.K}　"
          f"{len(A)} 个 (层,查询头)")
    print("-" * 96)
    print("【方向估计量本身】—— 填上 P0 §5.3 只有 e 没有 r/cos 的缺口")
    print(f"  e = ‖v̂−v‖/‖v‖            中位 {md(3):.4f}   P90 {np.percentile(A[:,3],90):.4f}")
    print(f"  r = ‖v̂‖/‖v‖              中位 {md(4):.4f}")
    print(f"  cos(v̂, v)                中位 {md(5):.4f}")
    print(f"  自检：cos=(r²+1−e²)/(2r) ⇒ {(md(4)**2+1-md(3)**2)/(2*md(4)):.4f}"
          f"　（错误公式 √(1−e²) 会给 {max(0,1-md(3)**2)**0.5:.4f}）")
    print(f"\n【质量估计量】log(D̂_E/D_E)  中位 {md(2):+.4f} ⇒ "
          f"D̂_E/D_E = {np.exp(md(2)):.4f}（低估 {np.exp(-md(2)):.1f}×）")
    print("-" * 96)
    print("【四格：‖o − o_full‖ / ‖Δo‖，逐头，越小越好；1.0 = 完全不修正】")
    print(f"{'':<34}{'中位':>10}{'P90':>10}")
    for nm, c in (("现方法      (D̂, v̂)", 6), ("Oracle-Mass (D , v̂)", 7),
                  ("Oracle-Dir  (D̂, v )", 8)):
        print(f"  {nm:<32}{md(c):>10.4f}{np.percentile(A[:,c],90):>10.4f}")
    print(f"  {'（第四格 (D,v) 恒等于 o_full ⇒ 误差 0，为内建自检）':<32}")
    print(f"\n  现方法的修正方向 cos(δ̂, Δo) 中位 {md(9):.4f}　"
          f"幅度 ‖δ̂‖/‖Δo‖ 中位 {md(10):.4f}")
    print("-" * 96)
    print("【层级（跨头经 W_O 相加后）—— P0 测跨头相消只留 0.25，逐头好≠层级好】")
    print(f"{'层':<8}{'现方法':>10}{'Oracle-Mass':>14}{'Oracle-Dir':>13}{'cos(现方法)':>13}")
    for l in a.layers:
        s = B[B[:, 0] == l]
        if len(s):
            print(f"  {int(l):<6}{s[:,1].mean():>10.4f}{s[:,2].mean():>14.4f}"
                  f"{s[:,3].mean():>13.4f}{s[:,4].mean():>13.4f}")
    print("-" * 96)
    print("【重做 P0 的 E 阶梯（原结论只有 6 个 query）—— 质量与方向解耦】")
    print(f"{'':<42}{'误差中位':>10}{'P90':>10}")
    for nm, c in ((f"现方法        质量 0 阶 + 方向 μ_v", 6),
                  (f"E4 式         质量**二阶 MGF** + 方向 μ_v", 12),
                  (f"E2/MomentKV 式 质量 0 阶 + 方向 +Σ_vk·a", 13),
                  (f"两者都换      二阶 MGF + Σ_vk·a", 14)):
        print(f"  {nm:<40}{md(c):>10.4f}{np.percentile(A[:,c],90):>10.4f}")
    print(f"  方向误差 e：只用 μ_v {md(3):.4f}  →  加 Σ_vk·a {md(15):.4f}"
          f"　（P0 记的是 0.666 → 0.438）")
    print("  判读：若「E4 式」明显差于「现方法」⇒ 复现 P0 的灾难；若反而更好 ⇒")
    print("        P0 的 +1249% 是 n=6 的假象，或来自它把质量与方向一起改了")
    print("-" * 96)
    # 第 14 列（旧「两者都换」）把**一阶** softmax 权重配上**二阶**质量 LE2，不是任何
    # 一个自洽的估计量。第 16 列才是真 joint：权重也用二阶 logits。两者之差量化了
    # 那个不一致到底有多大——若可忽略，之前所有基于第 14 列的读数不必改口径。
    print("【一阶权重 vs 真 joint —— 修掉「二阶质量配一阶方向」的不一致】")
    print(f"{'':<42}{'误差中位':>10}{'P90':>10}")
    for nm, c in (("旧写法  LE2 × softmax(一阶 r)  [列14]", 14),
                  ("真 joint LE2 × softmax(二阶 r2) [列16]", 16)):
        print(f"  {nm:<40}{md(c):>10.4f}{np.percentile(A[:,c],90):>10.4f}")
    print(f"  方向误差 e：一阶权重 {md(15):.4f}  →  二阶权重 {md(17):.4f}")
    # 「二阶只是公共平移」的直接证据。此前只有"结果符合该解释"，没有测过量本身。
    print(f"  二阶项 ½aᵀΣa 的**跨簇**散布：std 中位 {md(21):.4f}  极差中位 {md(22):.4f}")
    print(f"  KL(softmax(r2) ‖ softmax(r0))  中位 {md(23):.2e}  "
          f"P90 {np.percentile(A[:,23],90):.2e}")
    print("  判读：KL≈0 ⇒ 二阶在簇间近乎公共平移，被 softmax 约掉 ⇒ "
          "改总质量很多、改相对权重为零")
    print("-" * 96)
    # Jensen 缺口的分解：`r_ex` 是**精确** MGF（对簇内成员做分段 logsumexp），所以
    #   列18 = |r_ex − r0| 一阶漏掉的全部，  列19 = |r_ex − r2| 二阶之后还剩的。
    # 解释率 1 − 列19/列18 就是"二阶到底补上了 Jensen 缺口的百分之几"。这是唯一
    # 能把"二阶够不够"和"二阶方向对不对"分开的数字。
    print("【Jensen 缺口：二阶 MGF 补上了多少（对精确 MGF 的分段 logsumexp 而言）】")
    print(f"  一阶缺口 |r_ex−r0|   中位 {md(18):.4f}   P90 {np.percentile(A[:,18],90):.4f}")
    print(f"  二阶残差 |r_ex−r2|   中位 {md(19):.4f}   P90 {np.percentile(A[:,19],90):.4f}")
    _er = A[:, 20][np.isfinite(A[:, 20])]
    print(f"  二阶解释率 1−残差/缺口  中位 {np.median(_er):.3f}   "
          f"P10 {np.percentile(_er,10):.3f}   为负比例 {np.mean(_er<0):.1%}")
    # Jensen 自检：`Σexp(aᵀk_i) ≥ n·exp(aᵀk̄)` ⇒ r_ex−r0 恒 ≥ 0。越界说明簇均值、
    # 分组或精度有问题，不是"结果不好"，是"实现不对"。
    print(f"  自检 signed 缺口为负的比例 {np.nanmean(A[:,24]):.2%}"
          f"（Jensen 保证应 ≈ 0；显著为正 = 簇均值/分组/精度有 bug）")
    print("  注意措辞：这 88% 是**逐簇 log-MGF（Jensen）缺口**的百分比，"
          "既不是注意力输出误差的、也不是下游分数的")
    print("  判读：解释率高而列16 不比列6 好 ⇒ 质量估准了但下游不吃这一套（对齐问题）；")
    print("        解释率低 ⇒ 二阶不够，需要每簇一个标量校正（P0 §5.5 的提法）")
    print("-" * 96)
    print("【γ-sweep：保持 v̂ 不变，只缩放真实质量 γ·D_E】")
    print(f"{'γ':>8}{'误差中位':>12}{'误差均值':>12}")
    med = [float(np.median(Gm[:, j])) for j in range(len(GAMMAS))]
    for j, gm in enumerate(GAMMAS):
        star = " ←最小" if j == int(np.argmin(med)) else ""
        note = "  (γ=0：完全不修正)" if gm == 0 else (
            "  (≈现方法等效 γ)" if abs(gm - 0.077) < 1e-9 else (
                "  (=Oracle-Mass)" if gm == 1.0 else ""))
        print(f"{gm:>8.3f}{med[j]:>12.4f}{Gm[:,j].mean():>12.4f}{star}{note}")
    P = np.array(per_sample)
    def bt(v, n=10000, seed=0):
        r = np.random.default_rng(seed)
        s = v[r.integers(0, len(v), (n, len(v)))].mean(1)
        return v.mean(), float(np.quantile(s, .025)), float(np.quantile(s, .975))
    print("-" * 96)
    print(f"【样本级统计（n={len(P)} 条样本，bootstrap 在样本上做）】")
    for nm, c in (("现方法      (D̂, v̂)", 0), ("Oracle-Mass (D , v̂)", 1),
                  ("Oracle-Dir  (D̂, v )", 2)):
        mm, lo, hi = bt(P[:, c])
        print(f"  {nm:<26}{mm:>8.4f} [{lo:.4f},{hi:.4f}]")
    mm, lo, hi = bt(P[:, 1] - P[:, 0])
    print(f"  {'Oracle-Mass − 现方法':<26}{mm:>+8.4f} [{lo:+.4f},{hi:+.4f}]"
          f"{' ★' if (lo > 0 or hi < 0) else ' 未分离'}")
    u, cnt = np.unique(P[:, 3], return_counts=True)
    print(f"  逐样本最优 γ 的分布： " + "  ".join(f"{x:g}×{c}" for x, c in zip(u, cnt))
          + f"　中位 {np.median(P[:,3]):g}")
    print(f"  逐样本 log r_D 中位的散布： {P[:,4].mean():+.3f} ± {P[:,4].std():.3f}"
          f"　（min {P[:,4].min():+.3f} / max {P[:,4].max():+.3f}）")
    print("-" * 96)
    print("【log(D̂_E/D_E) 的离散度 —— 决定「全局常数」是否安全；过冲是 E4 的死法】")
    print(f"  全体 (层,头,样本)： 中位 {md(2):+.3f}  P10 {np.percentile(A[:,2],10):+.3f}"
          f"  P90 {np.percentile(A[:,2],90):+.3f}  std {A[:,2].std():.3f}")
    lay = sorted(set(A[:, 0].astype(int)))
    sp = [(l, float(np.median(A[A[:, 0] == l, 2]))) for l in lay]
    worst = sorted(sp, key=lambda x: x[1])
    print(f"  逐层中位的极差 {max(v for _, v in sp) - min(v for _, v in sp):.3f}"
          f"　最低 3 层 {[(int(l), round(v,2)) for l, v in worst[:3]]}"
          f"　最高 3 层 {[(int(l), round(v,2)) for l, v in worst[-3:]]}")
    print(f"  ⇒ 若逐层极差 ≫ 0，全局常数不安全，需按层（或按 P0 提的每簇一个标量）校正")
    emp = GAMMAS[int(np.argmin(med))]
    pred = md(11)
    print(f"\n  实测最优 γ ≈ {emp}　　收缩理论预测 γ* = 1/(1+e²) 中位 = {pred:.4f}")
    print("=" * 96)
    print("判读（预注册）：")
    print("  最优 γ ≈ 0.7–0.85 ⇒ 收缩理论成立，现方法欠修正约 9 倍，质量校正值得做；")
    print("     E4 的 +1249% 则来自它同时改了方向估计，不是来自修准质量")
    print("  最优 γ ≈ 0.08     ⇒ 平方损失收缩不适用 ⇒ ε 系统性对齐在有害方向，")
    print("     或下游被尾部主导 —— 「低估质量是安全机制」成立")
    print("  单调递减到 0      ⇒ 修正整体负价值，质心的收益来自别处")
    print("注意：本探针只测**局部注意力误差**。MultiHop 已证明「更接近满缓存」不等于")
    print("     「任务分数更高」，所以最优 γ 还需要一次下游验证才能当方法用。")


if __name__ == "__main__":
    main()
