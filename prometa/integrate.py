"""把 ProMeta 挂进上游评测流水线 —— **零 upstream 文件改动**。

**为什么不改 `eval_chunk.py` / `wrapper.py` / `args.py`**：本会话有 30+ 个作业
在队列里排着，新起的作业是**现读现载**这些文件的。任何一次手滑都会同时污染
一整批正在跑的格子（第⑪类错：实验规格错了却已上 GPU）。所以走 monkey-patch，
和本仓库全部 `scratch_probe_*.py` 同一手法。

挂载点是 `ModelKVzip._init_kv`：它是**唯一**创建 cache 的地方，且
`_init_kv(kv=<已有>)` 会原样返回（生成/`_prob` 复用 cache 时不会被二次 init）。

**一条构造性的好性质**：`eval_chunk.py` 的满缓存参照走
`dataset.prefill_context(idx, do_score=False)`，`chunk_ratio` 默认 1.0 ⇒
上游 `if chunk_ratio < 1.0` 分支不进 ⇒ **`prune_chunk` 一次都不被调用** ⇒
ProMeta 在满缓存参照上是**构造性无操作**。这正是 VariKV 残差那条线栽过的坑
（空记忆仍无条件注入，把 `full__` 参照本身改掉了）——ProMeta 结构上没有这个洞。
`assert_reference_clean` 提供一个运行时复核。
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_STATE = {}


def load_student(ckpt_path, device="cuda"):
    """从 `scratch_prometa_train.py` 存的 ckpt 复原 Student。

    **架构参数一律从 ckpt 里的 `arch` 读，绝不从 CLI 默认值猜。**
    `--ctrlm_mode` 那次的教训：靠默认值加载会在某天静默跑成另一个方法。
    """
    import torch

    from prometa.model import ProMetaPredictor
    ck = torch.load(ckpt_path, map_location="cpu")
    arch = ck["arch"]
    net = ProMetaPredictor(**arch)
    net.load_state_dict(ck["state"], strict=True)      # strict：少一个键就报错
    net = net.to(device).float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ck


def install(net, *, beta=1.0, gamma=1.0, combine="resid", pool_layer=14,
            verbose=True):
    """给 `ModelKVzip._init_kv` 打补丁。可重复调用（会先卸载旧的）。"""
    from attention.kvcache import RetainCache
    from model.wrapper import ModelKVzip

    from prometa.cache import make_prometa_cache

    uninstall()
    PM = make_prometa_cache(RetainCache)
    orig = ModelKVzip._init_kv

    def patched(self, kv=None, evict_range=(0, 0)):
        out = orig(self, kv=kv, evict_range=evict_range)
        # **只在新建的、类型恰为 RetainCache 的 cache 上挂**。用 `type(...) is`
        # 而不是 isinstance：别把 Memory/Centroid/Control 那些子类也套上。
        if kv is None and type(out) is RetainCache:
            out.__class__ = PM
            out.pm_init(net, beta=beta, gamma=gamma, combine=combine,
                        pool_layer=pool_layer, verbose=verbose)
        return out

    ModelKVzip._init_kv = patched
    _STATE["orig"] = orig
    _STATE["cfg"] = dict(beta=beta, gamma=gamma, combine=combine,
                         pool_layer=pool_layer)
    print(f"[prometa/integrate] 已挂载 beta={beta} gamma={gamma} "
          f"combine={combine} pool_layer={pool_layer}", flush=True)
    return PM


def uninstall():
    if "orig" in _STATE:
        from model.wrapper import ModelKVzip
        ModelKVzip._init_kv = _STATE.pop("orig")
        _STATE.pop("cfg", None)


def tag_suffix(cfg):
    """配置必须进 tag —— 否则不同配置写进同一个结果目录互相覆盖（本仓库旧伤）。"""
    g = f"{cfg['gamma']:g}".replace(".", "p").replace("-", "m")
    b = f"{cfg['beta']:g}".replace(".", "p").replace("-", "m")
    return f"_pm{cfg['combine'][:3]}g{g}b{b}L{cfg['pool_layer']}"


def assert_reference_clean(kv):
    """满缓存参照上 ProMeta 必须一次都没动手。"""
    n = getattr(kv, "pm_nchunk", 0)
    assert n == 0, f"满缓存参照上 ProMeta 动了 {n} 次 —— 参照被污染了"
