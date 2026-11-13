"""`focus_sector_probe` 的两条不变量。

为什么值得测
------------
1. `shuffle_industry` 是**机制安慰剂的地基**：它必须只打乱「股票↔行业」对应，
   不能顺手改掉每日行业边际分布或 NaN 位置——否则安慰剂臂会引入
   「多/少踢掉几只」的隐藏差异，结论就从「行业信息无用」漂移成「约束强度不同」。
2. `K_EXIT` 是**无约束臂的 k 真源**：`ARMS` 里 `kw=None` 的臂靠它拿 k，
   新增一个无约束臂却忘了登记 ⇒ `k=None` 让策略构造静默退化或报错（2026-11-13
   加了 `k8_plain` 时才踩到，靠它兜住）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.focus_sector_probe import ARMS, K_EXIT, shuffle_industry


@pytest.fixture()
def ind() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    vals = np.array(["银行", "医药", "电子", "化工"])
    data = rng.choice(vals, size=(30, 40)).astype(object)
    # 造 20% 的 NaN（真实面板 28.4%），并固定一部分股票行业为常量
    mask = rng.random((30, 40)) < 0.2
    data[mask] = np.nan
    data[:, :5] = "银行"
    return pd.DataFrame(data, index=pd.bdate_range("2024-01-02", periods=30),
                        columns=[f"S{i:03d}" for i in range(40)])


def test_shuffle_keeps_nan_mask_and_marginal(ind):
    """打乱后：NaN 位置逐位不变，每行非空值的多重集不变。"""
    out = shuffle_industry(ind, seed=1)
    assert out.shape == ind.shape
    assert out.isna().to_numpy().tolist() == ind.isna().to_numpy().tolist()
    for i in range(len(ind)):
        a = sorted(ind.iloc[i].dropna().tolist())
        b = sorted(out.iloc[i].dropna().tolist())
        assert a == b, f"第 {i} 行行业边际分布被改动"


def test_shuffle_actually_changes_assignment(ind):
    """不能是恒等变换 —— 否则安慰剂等于真实臂，检验无意义。"""
    out = shuffle_industry(ind, seed=1)
    diff = (out.to_numpy() != ind.to_numpy()) & ind.notna().to_numpy()
    assert int(diff.sum()) > 0


def test_shuffle_is_reproducible(ind):
    a = shuffle_industry(ind, seed=7)
    b = shuffle_industry(ind, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_k_exit_covers_unconstrained_arms():
    """所有 `kw=None`（无约束）的臂必须在 `K_EXIT` 里有 (k, exit_mult)。"""
    missing = [n for n, _, kw in ARMS if kw is None and n not in K_EXIT]
    assert not missing, f"无约束臂缺 K_EXIT 登记: {missing}"
