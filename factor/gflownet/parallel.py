"""
reward 并行计算（进程池）
========================

TB 训练中 batch 内 16 条轨迹的 reward 计算是主要瓶颈（深树求值 0.3-1s/次）。
本模块用进程池并行求值：worker 进程持有面板（initializer 注入），每个 worker
维护独立的子树缓存（``node_cache``），公式字符串 -> reward 并行计算。

Windows 无 fork，必须用模块级函数（spawn 可 pickle）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from factor.formula import formula_builder

__all__ = ["RewardPool", "compute_reward_worker"]

_G: dict = {}


class _BoundedCache(dict):
    """FIFO 容量限制的子树缓存（每项面板 ~6MB@520×1456，防 worker OOM）。"""

    def __init__(self, maxsize: int = 200):
        super().__init__()
        self._maxsize = maxsize
        self._order: list = []

    def __setitem__(self, key, value):
        if key in self:
            return
        super().__setitem__(key, value)
        self._order.append(key)
        while len(self._order) > self._maxsize:
            old = self._order.pop(0)
            super().pop(old, None)


def _init_worker(panel: dict, market_cap: Optional[dict],
                 rets, rets_rank, features: list, eps: float,
                 temp: Optional[float], cache_maxsize: int = 200):
    """worker 初始化：注入面板/市值/horizon 收益/收益 rank/特征。"""
    _G["panel"] = panel
    _G["mc"] = market_cap
    _G["rets"] = rets
    _G["rr"] = rets_rank
    _G["feats"] = features
    _G["eps"] = eps
    _G["temp"] = temp
    _G["node_cache"] = _BoundedCache(cache_maxsize)


def compute_reward_worker(formula: str) -> float:
    """worker 内的公式 -> reward（含市值中性化 + rank IC + 子树缓存）。"""
    from factor.gflownet.reward import neutralize_market_cap, rank_ic_series

    panel = _G["panel"]
    fp = formula_builder(formula, features=_G["feats"],
                         node_cache=_G["node_cache"])(panel)
    if fp is None or fp.empty:
        v = 0.0
    else:
        if _G["mc"] is not None:
            fp = neutralize_market_cap(fp, _G["mc"])
        ic = rank_ic_series(fp, _G["rets"], returns_rank=_G["rr"])
        v = float(ic.abs().mean())
    eps = _G["eps"]
    if _G["temp"] is not None:
        return float(np.exp(max(v, eps) / _G["temp"]))
    return v if np.isfinite(v) and v > 0 else eps


class RewardPool:
    """进程池包装：batch 内公式并行求 reward。

    Args:
        panel: 特征面板 dict（date×code）。
        market_cap: 市值面板（None = 不中性化）。
        returns: horizon 收益面板（factor[t] 对齐）。
        returns_rank: 预计算的收益 rank 面板。
        features: 特征名列表。
        eps / temp: 奖励参数（与 make_reward_fn 一致）。
        n_jobs: 进程数（1 = 串行退化，不创建进程池）。
    """

    def __init__(self, panel: dict, market_cap=None, returns=None,
                 returns_rank=None, features: list | None = None,
                 eps: float = 1e-4, temp: Optional[float] = None,
                 n_jobs: int = 4, cache_maxsize: int = 200):
        self._n_jobs = n_jobs
        self._pool = None
        if n_jobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                n_jobs, initializer=_init_worker,
                initargs=(panel, market_cap, returns, returns_rank,
                          features or [], eps, temp, cache_maxsize))
        else:
            _init_worker(panel, market_cap, returns, returns_rank,
                         features or [], eps, temp, cache_maxsize)

    def compute(self, formulas: list[str]) -> list[float]:
        if self._pool is not None:
            return self._pool.map(compute_reward_worker, formulas)
        return [compute_reward_worker(f) for f in formulas]

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
