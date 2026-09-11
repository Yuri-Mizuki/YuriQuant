"""
pytest 共享 fixtures
====================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.mock import (
    MockDataSource,  # noqa: F401  （2026-09-11 实现下沉 data/mock，此处转发保 fixtures/旧引用兼容）
)


@pytest.fixture
def mock_ds() -> MockDataSource:
    return MockDataSource()


@pytest.fixture
def synth_parts():
    """多因子合成 / stacking 测试共用的确定性 mock 输入。

    返回 ``(未来一期收益面板, [CompositeInput × 3])``，三个因子分别为：
    动量类（与收益正相关）、动量带噪副本（高冗余，用于验证正交化）、
    反转类（弱信号、负 IC，用于验证符号对齐）。

    置于 conftest 是因为 2026-09-11 后合成能力横跨两层：因子层的确定性组合
    （``tests/test_synthesis.py``）与模型层的 ML stacking
    （``tests/test_stacking.py``）需要同一份输入做对照。
    """
    from factor.operators import cs_zscore, ts_rank
    from factor.preprocessing import standardize_zscore
    from factor.synthesis import CompositeInput

    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-01", periods=200, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(20)]
    # 注入 AR(1) 收益 → 动量因子有正 IC
    phi = 0.3
    rets = np.zeros((200, 20))
    for t in range(1, 200):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, 20)
    close = pd.DataFrame(10 * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    ret_panel = close.pct_change().shift(-1)

    f1 = ts_rank(close, 5)            # 动量类，与收益正相关
    # f2 是 f1 的带噪副本 → 与 f1 高度冗余（用于测试正交化消除相关性）
    noise = pd.DataFrame(rng.normal(0, 0.1, (200, 20)), idx, codes)
    f2 = standardize_zscore(f1 + noise)
    f3 = cs_zscore(ts_rank(close, 20) * -1)  # 反转类，弱信号
    comps = [
        CompositeInput("ts_rank_5", standardize_zscore(f1), ic=0.15, ir=1.2),
        CompositeInput("f1_noisy", f2, ic=0.12, ir=1.0),
        CompositeInput("rev", standardize_zscore(f3), ic=-0.05, ir=-0.4),
    ]
    return ret_panel, comps
