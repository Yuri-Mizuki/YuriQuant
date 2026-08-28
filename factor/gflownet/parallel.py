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
                 temp: Optional[float], cache_maxsize: int = 200,
                 styles: Optional[dict] = None,
                 long_ir_lambda: float = 0.5, long_ir_cap: float = 1.0,
                 barra_mu: float = 0.5, top_q: float = 0.2):
    """worker 初始化：注入面板/市值/horizon 收益/收益 rank/特征/风格价差。"""
    _G["panel"] = panel
    _G["mc"] = market_cap
    _G["rets"] = rets
    _G["rr"] = rets_rank
    _G["feats"] = features
    _G["eps"] = eps
    _G["temp"] = temp
    _G["styles"] = styles or {}
    _G["ir_lambda"] = long_ir_lambda
    _G["ir_cap"] = long_ir_cap
    _G["barra_mu"] = barra_mu
    _G["top_q"] = top_q
    _G["node_cache"] = _BoundedCache(cache_maxsize)


def compute_reward_worker(formula: str) -> float:
    """worker 内的公式 -> reward（复用 reward.composed_factor_reward 完整口径）。"""
    from factor.gflownet.reward import composed_factor_reward

    panel = _G["panel"]
    fp = formula_builder(formula, features=_G["feats"],
                         node_cache=_G["node_cache"])(panel)
    v = composed_factor_reward(
        fp, _G["rets"], returns_rank=_G["rr"], market_cap=_G["mc"],
        temp=None,  # 温度在 worker 出口统一施加
        long_ir_lambda=_G["ir_lambda"], long_ir_cap=_G["ir_cap"],
        barra_mu=_G["barra_mu"], styles=_G["styles"], top_q=_G["top_q"])
    eps, temp = _G["eps"], _G["temp"]
    if temp is not None:
        return float(np.exp(max(v, eps) / temp))
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
        styles: 风格价差 dict（None 且 barra_mu>0 时由主进程从 panel 构建，
            一次构建随 initializer 广播，避免每个 worker 重复计算）。
        long_ir_lambda / long_ir_cap / barra_mu / top_q:
            奖励塑形参数（与 make_reward_fn 一致）。
        n_jobs: 进程数（1 = 串行退化，不创建进程池）。
    """

    def __init__(self, panel: dict, market_cap=None, returns=None,
                 returns_rank=None, features: list | None = None,
                 eps: float = 1e-4, temp: Optional[float] = None,
                 styles: Optional[dict] = None,
                 long_ir_lambda: float = 0.5, long_ir_cap: float = 1.0,
                 barra_mu: float = 0.5, top_q: float = 0.2,
                 n_jobs: int = 4, cache_maxsize: int = 200):
        self._n_jobs = n_jobs
        self._pool = None
        # 风格价差在主进程一次性预计算（依赖面板而非公式，可广播复用）
        want_barra = barra_mu is not None and barra_mu > 0
        if want_barra and styles is None:
            from factor.gflownet.reward import build_barra_styles
            styles = build_barra_styles(panel, returns, top_q=top_q)
        initargs = (panel, market_cap, returns, returns_rank,
                    features or [], eps, temp, cache_maxsize,
                    styles, long_ir_lambda, long_ir_cap, barra_mu, top_q)
        if n_jobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(n_jobs, initializer=_init_worker,
                                  initargs=initargs)
        else:
            _init_worker(*initargs)

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
