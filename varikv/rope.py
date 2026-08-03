"""RoPE 的正/逆旋转工具。

为什么需要（这是方法正确性问题，不是工程细节）：
LLM 的 cache 里存的 key 已经过 RoPE（在 FastKVzip 里可见于
`prefill/attention/attn.py:54` 的 apply_rotary_pos_emb 先于 `update()`）。
若直接把 post-RoPE 的 key 累加进记忆槽，会踩到 RoPE 不满足加法分配律这一事实：

    α·R_p k + (1−α)·R_p' k'  ≠  R_φ(α·k + (1−α)·k')          (MemRoPE, arXiv 2603.12513)

即**加权平均后的 key 不对应任何有效位置**。后果有两层：
  1. 同一记忆槽对不同位置的 query 响应无规律振荡（实测内积在 −17…+13 之间跳）；
  2. 更致命的是，混合相位会**虚增 σ²** —— 这部分方差来自相位不一致，
     与认知不确定性毫无关系。那样一来「方差编码不确定性」这个核心主张就不成立
     （HANDOFF 红线 1）。

因此：吸收前用 R_p^{-1} 把 key 转回 position-free frame，读出时再按指定位置重旋。
重旋位置取该槽所概括 token 的**位置质心**（EPL arXiv 2409.14364 的 UPL 最优解）。
因为 R(δ)R(p) = R(p+δ)，这一步是纯代数运算，不需要任何额外前向。

参考：Still (2606.07878 §2.2) 的 position-free frame；Landmark Attention (2305.16300)
与 StreamingLLM (2309.17453) 的「存 pre-RoPE、读时加位置」。
"""

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF 的 half-split 约定（Qwen/Llama 系）。"""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def cos_sin_at(inv_freq: torch.Tensor, positions: torch.Tensor, dtype=None):
    """任意（含**浮点**）位置处的 cos/sin。

    槽的位置质心是加权平均，一般不是整数，所以不能走 HF 的
    `rotary_emb(x, position_ids)`（它要 long）。这里直接用 inv_freq 算，
    与 HF 的 `emb = cat([freqs, freqs])` 约定保持一致。

    positions [...] → cos/sin [..., d_head]
    """
    freqs = positions.float().unsqueeze(-1) * inv_freq.float()   # [..., d/2]
    emb = torch.cat([freqs, freqs], dim=-1)                      # [..., d]
    cos, sin = emb.cos(), emb.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """正向旋转，与 HF apply_rotary_pos_emb 等价。"""
    return x * cos + rotate_half(x) * sin


def inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """逆旋转 R_p^{-1}。

    因 rotate_half(rotate_half(x)) = −x，可验证
        apply_rope(x,c,s) * c − rotate_half(apply_rope(x,c,s)) * s
      = x(c²+s²) = x
    故逆旋转即把 sin 取负。
    """
    return x * cos - rotate_half(x) * sin
