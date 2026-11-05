"""截面有效性入库闸（``factor.validation``）的单元测试。

背景见模块文档：假面板（sd=1 但无信息）唯一可识别的指纹是**当日截面唯一值数**。
这几个用例把四类形态锁死，防止以后有人图省事改成「看 sd」或「看覆盖率」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.validation import (
    DISCRETE_MAX_UNIQ,
    check_panel,
    judge,
    panel_variation_stats,
)


def _panel(values: np.ndarray, codes: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=values.shape[0], freq="B")
    cols = [f"{i:06d}.SZ" for i in range(codes)]
    return pd.DataFrame(values, index=idx, columns=cols)


def test_true_panel_passes():
    """普通截面因子：每天各股不同 ⇒ ok。"""
    rng = np.random.default_rng(0)
    df = _panel(rng.normal(size=(25, 500)))
    ok, reason, st = check_panel(df)
    assert ok and reason == "ok"
    assert st["uniq"] > 400 and st["flat_frac"] == 0.0


def test_constant_panel_rejected():
    """定义性常数（alpha360_CLOSE0 ≡ 1）：拒收，理由是 flat。"""
    df = _panel(np.ones((25, 500)))
    ok, reason, _ = check_panel(df)
    assert not ok and reason == "flat"


def test_market_level_broadcast_rejected():
    """市场级广播：每天一个值铺满全市场（alt_macro_* 的形态）。

    🚨 关键用例：这类面板的**全局 sd 不为 0**（不同日取值不同），
    只看 sd 的检查会放过它 —— 必须按日判截面。
    """
    daily = np.linspace(-2.0, 3.0, 25)[:, None]
    df = _panel(np.repeat(daily, 500, axis=1))
    st = panel_variation_stats(df)
    assert st["sd"] > 1.0            # 全局 sd 正常 —— 所以 sd 不能当判据
    ok, reason, _ = check_panel(df)
    assert not ok and reason == "flat"


def test_float_noise_grid_rejected():
    """浮点噪声：唯一值几十个，且全部挤在同一量级内抖动（相对跨度 ~1e-6）。"""
    rng = np.random.default_rng(1)
    row = np.repeat(1.0 + rng.uniform(-1e-6, 1e-6, size=20), 250)[:5000]
    df = _panel(np.tile(row, (25, 1)), codes=5000)
    ok, reason, st = check_panel(df)
    assert not ok and reason == "noise"
    assert st["uniq"] < 50 and st["rel_span"] < 1e-3


def test_rank_normalized_discrete_factor_passes():
    """rank 归一化档位（alpha158_IMAX20 形态）：唯一值 21 个但跨度 = 1.0。

    🚨 这是首版判据的误杀现场：唯一值少 + uniq_ratio 低，但它们是**真因子**。
    护栏是相对跨度（1e0 而非 1e-6）。
    """
    grid = np.round(np.linspace(0.0, 1.0, 21), 4)
    rng = np.random.default_rng(3)
    df = _panel(rng.choice(grid, size=(25, 5000)), codes=5000)
    ok, reason, st = check_panel(df)
    assert ok and reason == "discrete"
    assert st["rel_span"] > 1.0


def test_boolean_factor_passes_as_discrete():
    """布尔因子（is_st 修复后是 0/1）：唯一值只有 2 个，但它是合法离散因子。"""
    rng = np.random.default_rng(2)
    df = _panel((rng.random((25, 500)) < 0.03).astype(float))
    ok, reason, st = check_panel(df)
    assert ok and reason == "discrete"
    assert st["uniq"] <= DISCRETE_MAX_UNIQ


def test_sparse_event_factor_not_rejected():
    """稀疏事件因子：几乎全 NaN，但不是「无截面差异」⇒ 放行并标 sparse。

    近全空是另一类问题，剔除它会改变现有 31 个因子的行为。
    """
    df = _panel(np.full((25, 500), np.nan))
    df.iloc[-1, :3] = [1.0, 2.0, 3.0]
    ok, reason, _ = check_panel(df)
    assert ok and reason == "sparse"


def test_empty_panel_is_sparse():
    ok, reason, st = check_panel(pd.DataFrame())
    assert ok and reason == "sparse" and st["uniq"] == 0


def test_judge_thresholds_are_configurable():
    """判据参数可调（供不同数据态复用），默认值不放宽。"""
    st = dict(uniq=40, n_valid=5000, uniq_ratio=0.008, flat_frac=0.0,
              valid_frac=0.9, rel_span=2.0, sd=1.0)
    assert judge(st)[0] is True                        # 相对跨度正常：放行
    assert judge({**st, "rel_span": 1e-6})[0] is False  # 抖动：拒收
    assert judge({**st, "rel_span": 1e-6}, min_rel_span=1e-9)[0] is True  # 阈值可放宽


# ---------------------------------------------------------------------------
# 段级判据：2026-11-05 全量回归的三次误杀现场
# ---------------------------------------------------------------------------
def test_all_segments_flat_is_rejected():
    """每段都恒常数（市场级广播）⇒ 拒。"""
    daily = np.linspace(-2.0, 3.0, 100)[:, None]
    df = _panel(np.repeat(daily, 500, axis=1))
    ok, reason, st = check_panel(df, segments=5)
    assert not ok and reason == "flat"
    assert all(f == 1.0 for f in st["seg_flat"])


def test_historical_flat_but_recent_differences_passes():
    """历史零膨胀、**近期**才有截面差异 ⇒ 放行（inner_netbuy_* 的形态）。

    选股用的是近期 IC 窗，历史退化不影响可用性。
    """
    rng = np.random.default_rng(4)
    vals = np.zeros((100, 500))
    vals[-20:] = rng.normal(size=(20, 500))       # 只有最后一段有差异
    ok, reason, st = check_panel(_panel(vals), segments=5)
    assert ok, st
    assert st["seg_flat"][-1] < 0.5 and st["seg_flat"][0] == 1.0


def test_recent_flat_but_history_normal_passes():
    """末段退化、多数段正常 ⇒ 放行（pledge_chg_20d 源数据尾部陈旧的形态）。"""
    rng = np.random.default_rng(5)
    vals = rng.normal(size=(100, 500))
    vals[-20:] = 0.0                              # 最近一个月源数据全 0
    ok, reason, st = check_panel(_panel(vals), segments=5)
    assert ok, st
    assert st["seg_flat"][-1] == 1.0 and st["seg_flat"][0] < 0.5


def test_sparse_days_are_not_counted_as_flat():
    """当日只有 1~2 个样本时 uniq 必然为 1，不能算「截面常数」。"""
    rng = np.random.default_rng(6)
    vals = np.full((100, 500), np.nan)
    vals[:, :2] = rng.normal(size=(100, 2))       # 每天只有 2 个样本
    vals[:60] = rng.normal(size=(60, 500))        # 前 60 天有完整截面且各股不同
    ok, reason, st = check_panel(_panel(vals), segments=5)
    assert ok, st
    # 后两段只有 2 个样本 ⇒ 不可判 ⇒ seg_flat 为 nan 而不是 1.0
    assert np.isnan(st["seg_flat"][-1])
