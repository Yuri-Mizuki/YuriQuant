"""IC 口径收口测试（``stats.ic.calc_ic_series`` 为全库唯一实现）。

背景（2026-09-10 第二批口径统一）：``factor.gflownet.reward.rank_ic_series``
此前自实现一份逐行 rank + 归一化相关，与 ``stats.ic.calc_ic_series`` 并行维护。
现在前者是后者的薄封装，本文件锁死三件事：

1. 薄封装与真源**逐元素一致**（含返回索引对齐到 ``factor_panel.index``）；
2. ``calc_ic_series`` 的默认路径**没有被动过**（corrwith 语义）；
3. ``returns_rank`` 性能捷径的**适用边界**：因子整行缺失不影响，行内散点
   缺失才会与默认路径分叉 —— 这条分歧是已知且被刻意记录的，不是 bug 回归。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.gflownet.reward import rank_ic_series
from stats.ic import calc_ic_series

N_DAYS, N_CODES = 120, 40


def _panels(seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=N_DAYS, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(N_CODES)]
    factor = pd.DataFrame(rng.normal(0, 1, (N_DAYS, N_CODES)), idx, codes)
    rets = pd.DataFrame(rng.normal(0, 0.02, (N_DAYS, N_CODES)), idx, codes)
    return factor, rets


def test_reward_wrapper_matches_canonical():
    """薄封装 == 真源（无 NaN 情形逐元素一致）。"""
    factor, rets = _panels()
    got = rank_ic_series(factor, rets)
    exp = calc_ic_series(factor, rets)
    assert got.index.equals(factor.index)          # 训练面板为主索引
    both = got.notna() & exp.notna()
    assert (got.isna() == exp.isna()).all()
    assert float((got[both] - exp[both]).abs().max()) == 0.0


def test_reward_wrapper_masks_nonfinite_to_nan():
    """常数因子（截面零方差）→ NaN，不是 ±inf。

    canonical 走 pandas corrwith，退化行可能返回 ±inf；奖励端做 abs(ic).mean()，
    inf 会直接毒化奖励，故薄封装保留 NaN 语义（收敛前自实现版本亦如此）。
    """
    factor, rets = _panels()
    factor.iloc[3] = 7.0                            # 该行因子为常数
    got = rank_ic_series(factor, rets)
    assert bool(np.isfinite(got.dropna()).all())


def test_reward_wrapper_index_covers_factor_rows():
    """收益面板缺行时，因子侧对应日取 NaN（索引对齐到 factor_panel）。"""
    factor, rets = _panels()
    got = rank_ic_series(factor, rets.iloc[5:])
    assert len(got) == N_DAYS
    assert got.iloc[:5].isna().all()


def test_returns_rank_fast_path_exact_without_extra_nan():
    """因子不比收益多出 NaN 时，性能捷径与默认路径完全一致。"""
    factor, rets = _panels()
    fast = calc_ic_series(factor, rets, returns_rank=rets.rank(axis=1))
    slow = calc_ic_series(factor, rets)
    assert (fast.isna() == slow.isna()).all()
    assert float((fast - slow).abs().max()) == 0.0


def test_returns_rank_fast_path_exact_for_whole_row_nan():
    """因子**整行**缺失（窗口预热）不影响：该行掩码全 False，两侧结果一致。"""
    factor, rets = _panels()
    factor.iloc[:15] = np.nan
    fast = calc_ic_series(factor, rets, returns_rank=rets.rank(axis=1))
    slow = calc_ic_series(factor, rets)
    assert (fast.isna() == slow.isna()).all()
    assert float((fast - slow).abs().max()) == 0.0


def test_returns_rank_fast_path_diverges_on_scattered_nan():
    """因子在收益有效处还有**行内散点缺失**时，捷径与默认路径分叉。

    这条断言是**刻意**的：它把已知分歧钉在测试里（30% 散点缺失时日均 IC 差
    约 1e-2，与 IC 量级 3e-2~5e-2 同阶）。若将来有人以为"捷径等价"而把它推广到
    入库/报告路径，本测试与 ``calc_ic_series`` docstring 的警告会同时提醒他。
    """
    factor, rets = _panels()
    rng = np.random.default_rng(1)
    factor = factor.mask(rng.random(factor.shape) < 0.30)
    fast = calc_ic_series(factor, rets, returns_rank=rets.rank(axis=1))
    slow = calc_ic_series(factor, rets)
    diff = (fast - slow).abs().dropna()
    assert diff.mean() > 1e-4                      # 确有分叉，不是等价实现
    assert fast.notna().sum() == slow.notna().sum()  # 但有效行集合不变
