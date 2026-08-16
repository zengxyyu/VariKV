"""CentroidControlCache —— 质心读出 + 学习残差改分，同时开。

**为什么这个组合从来没测过，以及为什么值得测。**

两条线动的是流水线的不同环节，构造上正交：

    候选 ──打分 s⁰──▶ [残差: s⁰+Δs(K,V)] ──阈值──▶ 保留集 R / 驱逐集 E
                                                          │
                     attention 输出 ◀── [质心: 把 E 摘要成簇读回] ◀┘

残差只改**扔谁**，完全不碰 attention 输出；质心不改选择，只把**已经扔掉的**摘要成
每头若干簇后加回读出。`args.py` 一直把两者写成互斥，所以从未同时开过。

真正值得测的是它们之间**存在一个真实的交互**，不是简单相加：质心摘要的对象是
**驱逐集**，而残差会改变驱逐集的成分。若残差倾向于把更冗余、更好摘要的 token 扔掉，
质心的信息损失就更小 —— 超可加；若两者在修同一个"丢失质量"的问题，则次可加。

经验上也支持它们不是一回事：**峰值 ratio 不同**（RESULTS_GRID.md，11 panel 均值）

    质心 K=1024   ρ=0.1 +3.66   ρ=0.2 +1.51   ρ=0.3 −1.75
    残差 v2       ρ=0.1 +1.02   ρ=0.2 +2.12   ρ=0.3 −0.57

--------------------------------------------------------------------------------
实现靠 MRO，而不是把两边的代码抄到一起：

    CentroidRetainCache.prune_chunk   调 super() 做阈值化，再按 self.valid 吸收驱逐集
    LearnedControlRetainCache.prune_chunk  自己重写阈值化（先改分数），不调 super

所以 `CentroidControlCache(CentroidRetainCache, LearnedControlRetainCache)` 的 MRO 是

    Centroid.prune_chunk → super() → LearnedControl.prune_chunk（改分数+阈值化）
                         → 回到 Centroid 吸收驱逐集

恰好就是想要的顺序。抄代码会立刻产生两份需要同步维护的阈值化逻辑。

**`__init__` 的顺序是有讲究的**：先 `LearnedControlRetainCache.__init__`（它会跑
`RetainCache.__init__` 并设好控制字段），再 `CentroidRetainCache.__init__`（会**再跑一次**
`RetainCache.__init__`）。第二次重置的只是基类字段，而基类字段此刻还没被用过；
控制字段是 Learned 自己设的，`RetainCache.__init__` 碰不到，所以能存活。反过来写
就会把质心的字段冲掉。
"""
from typing import Tuple

from .centroid import CentroidRetainCache
from .learned_ctrlcache import LearnedControlRetainCache


class CentroidControlCache(CentroidRetainCache, LearnedControlRetainCache):

    def __init__(self, model, evict_range: Tuple[int, int], ctrl=None,
                 n_clusters: int = 1024, rope_inv_freq=None,
                 rope_mode: str = "post", seed: int = 0,
                 rho_max: float = 1.0, n_write: int = 512):
        # 顺序不能反，见模块 docstring
        LearnedControlRetainCache.__init__(
            self, model, evict_range, ctrl=ctrl, train_mode=False,
            seed=seed, n_write=n_write, rho_max=rho_max)
        CentroidRetainCache.__init__(
            self, model, evict_range, n_clusters=n_clusters,
            rope_inv_freq=rope_inv_freq, rope_mode=rope_mode)
        # CentroidRetainCache.__init__ 走了一遍 RetainCache.__init__，把
        # LearnedControlRetainCache 依赖的诊断列表连同基类字段一起重置了。
        # 控制字段（ctrl / M / rho_max / _gen / n_write）是 Learned 自己设的，
        # RetainCache 碰不到，所以还在；只有这三个诊断列表要补回来。
        self.flip_frac, self.retain_delta, self.delta_std = [], [], []
        self.trace = []
