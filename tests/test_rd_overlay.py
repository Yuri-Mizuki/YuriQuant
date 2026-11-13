"""`rd_overlay_probe` 的合成口径不变量 —— 钉住「池子不变」与安慰剂性质。

为什么必须有这组测试
--------------------
`rd_to_cap` 覆盖只有 67%，如果把缺失格直接交给 `caliber.weighted_rank_average`
（取**交集**），股票池会被砍掉 10~30%，那时「因子有没有增量」会与「选股池变小」
混在一起 —— 这正是 `reports/factor_ev_rd/` 里 +9.16pp 不可直接采信的原因之一。
本模块的 `composite` 用 `fill` 保住池子，**必须由测试守住**：

  * 任何 w / fill 下，输出的非空位置必须与 base 逐位相同；
  * `w=0` 必须退化为 `rank(base)`（自检臂 = 基线臂）；
  * 安慰剂必须保留 NaN 位置（否则安慰剂臂偷偷换了池子，对照失效）。

2026-11-13 的另一条教训：`rebalance_granularity_probe` 里「换手笔数被 min() 截断」
让持仓从 5.9 静默掉到 3.7、年化从 36.58% 掉到 33.52%，回测照跑不报错。
同理，池子被悄悄缩小也不会报错 —— 所以这里不测「数值对不对」，只测**结构不变量**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.rd_overlay_probe import (
    block_bootstrap,
    composite,
    placebo_panels,
)


@pytest.fixture()
def panels():
    """3 天 × 8 只：base 覆盖 7 只有值，rd 只覆盖 5 只（模拟 67% 覆盖）。"""
    idx = pd.bdate_range("2024-01-02", periods=3)
    cols = [f"{i:06d}.SZ" for i in range(8)]
    base = pd.DataFrame(
        [[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, np.nan]] * 3,
        index=idx, columns=cols, dtype="float64")
    rd = pd.DataFrame(
        [[1.0, np.nan, 0.5, np.nan, 0.2, 0.1, np.nan, np.nan]] * 3,
        index=idx, columns=cols, dtype="float64")
    return base, rd


@pytest.mark.parametrize("w,fill", [(0.0, None), (0.05, 0.5), (0.2, 0.5), (0.2, 0.0)])
def test_composite_keeps_base_universe(panels, w, fill):
    """池子不变量：非空位置必须与 base 逐位相同 —— 这是「增量」可比的前提。"""
    base, rd = panels
    out = composite(base, rd, w, fill)
    assert out.shape == base.shape
    pd.testing.assert_frame_equal(out.notna(), base.notna())


def test_composite_zero_weight_is_rank_of_base(panels):
    """w=0 必须退化为 rank(base)：否则「base 臂」不是基线，整张 Δ 表都要重算。"""
    base, rd = panels
    out = composite(base, rd, 0.0, 0.5)
    expect = base.rank(axis=1, pct=True)
    pd.testing.assert_frame_equal(out.where(base.notna()), expect.where(base.notna()),
                                  check_dtype=False)


def test_composite_fill_changes_values_but_not_universe(panels):
    """fill=0 与 fill=0.5 必须给出**不同**数值、**相同**位置。

    两者若连数值都相同，说明 fill 参数没接上（2026-11-13 solo/focus 上两者组合逐位
    相同，一度被误判为 bug —— 实际是那批股票进不了头部；这里在构造上钉住差异存在）。
    """
    base, rd = panels
    a = composite(base, rd, 0.2, 0.5)
    b = composite(base, rd, 0.2, 0.0)
    pd.testing.assert_frame_equal(a.notna(), b.notna())
    assert not a.equals(b)
    assert float((a - b).abs().max().max()) > 0


def test_composite_respects_ordering_when_rd_aligns(panels):
    """rd 与 base 同序时，加权重不得改变排序（同序因子的合成只压缩间距）。"""
    idx = pd.bdate_range("2024-01-02", periods=1)
    cols = [f"{i:06d}.SZ" for i in range(6)]
    base = pd.DataFrame([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], index=idx, columns=cols)
    out = composite(base, base.copy(), 0.2, 0.5)
    assert list(out.iloc[0].sort_values(ascending=False).index) == cols


@pytest.mark.parametrize("kind", ["rand", "shuf"])
def test_placebo_preserves_nan_mask(panels, kind):
    """安慰剂必须保留 NaN 位置 —— 否则它偷偷换了池子，「跑赢安慰剂」不成立。"""
    _, rd = panels
    pl = placebo_panels(rd)[kind]
    pd.testing.assert_frame_equal(pl.notna(), rd.notna())


def test_shuffle_placebo_preserves_marginal_distribution(panels):
    """`shuf` 只打乱次序：每行的取值集合必须与原始完全相同（边缘分布不变）。"""
    _, rd = panels
    shuf = placebo_panels(rd)["shuf"]
    for d in rd.index:
        a = np.sort(rd.loc[d].dropna().to_numpy())
        b = np.sort(shuf.loc[d].dropna().to_numpy())
        np.testing.assert_allclose(a, b)


def test_ma60_placebo_is_slower_than_raw():
    """`ma60` 用滚动均值替代日频市值：它必须比原面板**更平滑**（日变化更小）。

    注意窗口：`rolling(60, min_periods=20)` 在 3 天的 fixture 上全是 NaN，
    故这里单造 30 天。`min_periods=20` 也是为什么它只能在天数足够的真实面板上用。
    """
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-02", periods=30)
    cols = [f"{i:06d}.SZ" for i in range(6)]
    rd = pd.DataFrame(rng.normal(size=(30, 6)), index=idx, columns=cols)
    rd.iloc[:5, 0] = np.nan                    # 前 5 天缺失：滚动窗口要能跨过缺口
    ma = placebo_panels(rd)["ma60"]
    x = rd.fillna(0.0)
    y = ma.fillna(0.0)
    assert float(y.diff().abs().sum().sum()) < float(x.diff().abs().sum().sum())
    assert ma.notna().to_numpy().sum() > 0


def test_block_bootstrap_zero_series_is_null():
    """零差序列：点估计 0、p 取 1（不能因为 `means<=0`/`>=0` 各占一半而报 2）。"""
    s = pd.Series(np.zeros(300))
    r = block_bootstrap(s, block=20, n_boot=200)
    assert r["ann"] == pytest.approx(0.0)
    assert r["p"] == pytest.approx(1.0)


def test_block_bootstrap_detects_positive_drift():
    """稳定正漂移：全年化必须为正、且 p 很小（否则显著性判据形同虚设）。"""
    rng = np.random.default_rng(7)
    s = pd.Series(rng.normal(0.001, 0.002, 500))     # 日均 +10bp ≈ 年化 +25%
    r = block_bootstrap(s, block=20, n_boot=500)
    assert r["ann"] > 0.15
    assert r["p"] < 0.05
