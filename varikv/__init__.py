"""VariKV — Variational Free-Energy Eviction for KV Cache Compression.

方法：theory_distributional_memory.md §11「选项 3：自由能统一驱逐」
执行：HANDOFF.md
"""

from .config import CacheConfig, Config, FreeEnergyConfig, MemoryConfig, TrainConfig
from .cache import MemoryAugmentedCache
from .free_energy import FreeEnergyScorer
from .memory import DistributionalMemory

__all__ = [
    "Config", "MemoryConfig", "FreeEnergyConfig", "CacheConfig", "TrainConfig",
    "DistributionalMemory", "FreeEnergyScorer", "MemoryAugmentedCache",
]
