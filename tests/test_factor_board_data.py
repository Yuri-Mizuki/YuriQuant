"""factor_board_data 的窗口指标回归锁（2026-11-22 sharpe 修复）。

缺陷：``sharpe_arithmetic(r, PPY)`` 把 PPY 误传第二位置参数 ``rf``——
日无风险被折成 ~2.18%，全部分位 sharpe 打成 −20~−100 的假大负，
看板排序键（q5_net_all_sharpe）自建成起失效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.factors.factor_board_data import _window_metrics


def test_window_metrics_sharpe_sign_matches_return():
    """正收益分位组合的 sharpe 必须为正（rf=0 口径下符号恒同向）。"""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-02", periods=900)
    # 构造年化 ~+15%、日波 ~1.3% 的净值曲线
    r = pd.Series(rng.normal(0.0006, 0.013, len(idx)), index=idx)
    nav = (1 + r).cumprod()
    bench = (1 + pd.Series(rng.normal(0.0002, 0.012, len(idx)), index=idx)).cumprod()
    to = pd.Series(0.5, index=idx[::21])

    m = _window_metrics(nav, bench, to)
    for w, v in m.items():
        assert v["ann"] == v["ann"]  # 全窗口可算
        if v["ann"] > 0.02:
            assert v["sharpe"] > 0, f"{w}: 年化 {v['ann']:.2%} 但 sharpe {v['sharpe']}"
        assert abs(v["sharpe"]) < 10, f"{w}: sharpe {v['sharpe']} 不可能（rf 误传复发）"
