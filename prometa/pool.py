"""前缀上下文的**在线**注意力池化 —— 让 Student 在分块预填中拿到「到目前为止
的上下文摘要」，而**不必把 169k×3584 的 hidden 全存下来**（28 层全存是 33 GB）。

池化定义（与 `ProMetaPredictor.latents` 里的离线版**必须逐位一致**）：

    a = softmax_N( Qp · h'ᵀ / √dp )        z = a · h'

softmax 在**全部已见位置**上归一化，所以不能简单地分块求和。用在线 softmax
（与 flash-attention 同一套 running-max 技巧）即可**精确**增量化：

    维护 m = 运行最大 logit、s = Σ exp(logit − m)、v = Σ exp(logit − m)·h'
    新块来时先把旧的 (s, v) 按 exp(m_old − m_new) 重标定，再累加。

⇒ 显存 O(K·dp)，与上下文长度无关，且结果与「一次性看全部位置」**数值等价**
（自测里对拍到 1e-6）。
"""
import torch


class OnlineAttnPool:
    def __init__(self, pool_q, device=None, dtype=torch.float32):
        """pool_q: [K, dp]（`ProMetaPredictor.pool_q`）"""
        self.q = pool_q.detach().to(device or pool_q.device, dtype)
        K, dp = self.q.shape
        self.scale = dp ** -0.5
        self.m = torch.full((K, 1), float("-inf"), device=self.q.device, dtype=dtype)
        self.s = torch.zeros((K, 1), device=self.q.device, dtype=dtype)
        self.v = torch.zeros((K, dp), device=self.q.device, dtype=dtype)
        self.n = 0

    @torch.no_grad()
    def update(self, h_proj):
        """h_proj: [n, dp]，**已投影**的本块 hidden。"""
        h = h_proj.to(self.q.dtype)
        logits = (self.q @ h.T) * self.scale                 # [K,n]
        m_new = torch.maximum(self.m, logits.max(-1, keepdim=True).values)
        rescale = torch.exp(self.m - m_new)
        rescale = torch.nan_to_num(rescale, nan=0.0, posinf=0.0)   # 首块 m=-inf
        w = torch.exp(logits - m_new)                        # [K,n]
        self.s = self.s * rescale + w.sum(-1, keepdim=True)
        self.v = self.v * rescale + w @ h
        self.m = m_new
        self.n += h.shape[0]
        return self

    def value(self):
        """→ z: [K, dp]。未见任何位置时返回全零（调用方须自行判断 `self.n`）。"""
        if self.n == 0:
            return torch.zeros_like(self.v)
        return self.v / self.s.clamp_min(1e-30)


def _selftest():
    torch.manual_seed(0)
    K, dp, N = 4, 16, 997
    q = torch.randn(K, dp)
    h = torch.randn(N, dp)

    # ① 与一次性全量池化数值等价（这是整个在线化的前提）
    a = torch.softmax(q @ h.T * dp ** -0.5, dim=-1)
    ref = a @ h
    for chunks in ([N], [1, N - 1], [100, 300, 597], [1] * 10 + [N - 10]):
        p = OnlineAttnPool(q)
        i = 0
        for c in chunks:
            p.update(h[i:i + c]); i += c
        err = (p.value() - ref).abs().max().item()
        assert err < 1e-5, (chunks[:3], err)
    print(f"① 与全量池化等价（4 种分块方式，max|差| < 1e-5）　PASS")

    # ② 极端 logit 不溢出（h 放大 1000 倍）
    p = OnlineAttnPool(q); p.update(h * 1000)
    assert torch.isfinite(p.value()).all()
    a2 = torch.softmax(q @ (h * 1000).T * dp ** -0.5, -1) @ (h * 1000)
    assert (p.value() - a2).abs().max() < 1e-2, (p.value() - a2).abs().max()
    print("② 极端 logit 不溢出、仍与全量一致　PASS")

    # ③ 空状态返回零、n 计数正确
    p = OnlineAttnPool(q)
    assert p.n == 0 and p.value().abs().max() == 0
    p.update(h[:5]); assert p.n == 5
    print("③ 空状态与计数　PASS")

    # ④ 阴性对照：故意漏掉 rescale ⇒ 必须对不上（证明 rescale 是必需的）
    p = OnlineAttnPool(q)
    p.update(h[:500])
    m_saved, s_saved, v_saved = p.m.clone(), p.s.clone(), p.v.clone()
    p.update(h[500:])
    good = (p.value() - ref).abs().max().item()
    logits = (q @ h[500:].T) * dp ** -0.5
    m_new = torch.maximum(m_saved, logits.max(-1, keepdim=True).values)
    w = torch.exp(logits - m_new)
    bad_s = s_saved + w.sum(-1, keepdim=True)          # 漏了 rescale
    bad_v = v_saved + w @ h[500:]
    bad = ((bad_v / bad_s) - ref).abs().max().item()
    assert bad > 100 * max(good, 1e-9), (good, bad)
    print(f"④ 阴性对照：漏 rescale 误差 {bad:.3e} ≫ 正确 {good:.3e}　PASS")
    print("\nprometa/pool.py 自测 4 条全过")


if __name__ == "__main__":
    _selftest()
