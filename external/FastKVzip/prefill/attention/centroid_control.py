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

**`__init__` 有一个会静默失效的坑，第一版就踩了。** `super()` 按**实例的 MRO** 解析，
不按定义它的类。所以 `CentroidRetainCache.__init__` 里那句 `super().__init__(...)`，
在本类的实例上解析到的是 **`LearnedControlRetainCache.__init__`**（MRO 的下一个），
而不是 `RetainCache.__init__`。

于是「先显式调 Learned 设好 ctrl，再显式调 Centroid」这种写法，第二步会**用默认参数
重跑 Learned 的 __init__**，把 `ctrl` 覆盖回 `None` —— `prune_chunk` 里
`if self.ctrl is not None` 直接跳过整个修正块，**组合臂退化成纯质心，不报任何错**。
冒烟测试也抓不到：`[Cen+Ctrl] ...` 那行是 `eval_chunk` 在建 cache 之前打的。

正确写法是**只调一次** `CentroidRetainCache.__init__`，让它沿 MRO 自己把
Learned -> RetainCache 串起来（控制字段会拿到默认值），**之后**再把非默认的控制字段
显式设上。末尾的断言是为了让同类错误下次直接崩掉而不是静默降级。
"""
from typing import Tuple

import torch

from .centroid import CentroidRetainCache
from .learned_ctrlcache import LearnedControlRetainCache


class CentroidControlCache(CentroidRetainCache, LearnedControlRetainCache):

    def __init__(self, model, evict_range: Tuple[int, int], ctrl=None,
                 n_clusters: int = 1024, rope_inv_freq=None,
                 rope_mode: str = "post", seed: int = 0,
                 rho_max: float = 1.0, n_write: int = 512):
        # **只调这一次**：它的 super() 沿 MRO 依次走 LearnedControlRetainCache
        # -> RetainCache，两层的字段都会建好（控制字段取默认值）。
        CentroidRetainCache.__init__(
            self, model, evict_range, n_clusters=n_clusters,
            rope_inv_freq=rope_inv_freq, rope_mode=rope_mode)
        # 再把非默认的控制字段设上。**必须在上面之后**，否则会被那一步覆盖。
        self.ctrl = ctrl
        self.rho_max = float(rho_max)
        self.n_write = int(n_write)
        self._gen = torch.Generator(device="cpu").manual_seed(seed)
        # 让同类错误崩掉而不是静默降级成纯质心
        assert self.ctrl is not None, "组合臂必须有 ctrl，否则等于只跑质心"
        assert self.centroid_mode, "组合臂必须开质心，否则等于只跑残差"
