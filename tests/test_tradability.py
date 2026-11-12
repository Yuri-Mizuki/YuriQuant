"""可交易性口径单一模块的回归测试（2026-09-10 收敛）。

背景：此前 ``build_tradable_mask`` 住在 ``data/cache_helpers.py``，``data/tradability.py``
另有一套 ``build_executable_mask`` / ``build_directional_masks``——同一件事两个模块、
两种口径。收敛后两者都在 ``data.tradability``，本文件把"两种口径的差别"锁死，
防止将来有人把两者合并成一份而静默改变语义。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _status_multiindex(rows: dict) -> pd.DataFrame:
    """{date: {code: (high_limited, low_limited, is_st, is_suspended)}} → 长表。"""
    recs = {}
    for d, per_code in rows.items():
        for c, (hi, lo, st, susp) in per_code.items():
            recs[(d, c)] = (hi, lo, st, susp)
    df = pd.DataFrame.from_dict(
        recs, orient="index",
        columns=["high_limited", "low_limited", "is_st", "is_suspended"],
    )
    df.index = pd.MultiIndex.from_tuples(df.index, names=["date", "code"])
    return df


def test_cache_helpers_no_longer_exports_mask():
    """掩码只能从 data.tradability 拿到——cache_helpers 不得保留第二个入口。"""
    import data.cache_helpers as ch
    import data.tradability as td

    assert not hasattr(ch, "build_tradable_mask"), "cache_helpers 不应再导出 build_tradable_mask"
    assert hasattr(td, "build_tradable_mask")
    assert hasattr(td, "build_directional_masks")
    # 旧的 T 日非方向化掩码已删（run_backtest 退役后成死代码，勿复活）
    assert not hasattr(td, "build_executable_mask")


def test_two_conventions_differ_on_next_day_seal(tmp_path):
    """核心：封板发生在 T+1 时，T+1 口径剔除该行、T 日口径不剔除。

    这正是两套掩码的**唯一**语义差别，也是 GP 适应度与最终回测口径必须一致的
    原因（两者都用 T+1 口径）。
    """
    from data.tradability import build_directional_masks, build_tradable_mask

    idx = pd.date_range("2023-01-03", periods=3, freq="B")
    cols = ["A", "B"]
    close = pd.DataFrame(10.0, index=idx, columns=cols)
    d0, d1, d2 = idx

    # A 在 d1 收盘封涨停（10.0 >= high_limited 10.0）→ d0 的信号 T+1 买不进
    status = _status_multiindex({
        d0: {"A": (11.0, 9.0, False, False), "B": (11.0, 9.0, False, False)},
        d1: {"A": (10.0, 9.0, False, False), "B": (11.0, 9.0, False, False)},
        d2: {"A": (11.0, 9.0, False, False), "B": (11.0, 9.0, False, False)},
    })
    status.to_parquet(tmp_path / "history_stock_status.parquet")

    # T+1 口径：d0 行看 d1 状态 → A 被剔除
    mask_next = build_tradable_mask(close, bwd=None, cache_root=str(tmp_path))
    assert not mask_next.at[d0, "A"], "T+1 封涨停应剔除 T 日信号"
    assert mask_next.at[d0, "B"]

    # T 日口径：d0 行只看 d0 状态 → A 当日未封板 → 不剔除（差异所在）
    long = status.reset_index()
    buyable, _sellable = build_directional_masks(long, idx, pd.Index(cols), close_panel=close)
    assert bool(buyable.at[d0, "A"]), "T 日口径不应看到 T+1 的封板"
    # 而 d1 行（d1 当日已封板）在 T 日口径下被禁买
    assert not bool(buyable.at[d1, "A"])


def test_tradable_mask_missing_status_is_permissive(tmp_path):
    """状态表缺失 → 全 True 并退化为不过滤（历史行为不得改变）。"""
    from data.tradability import build_tradable_mask

    close = pd.DataFrame(np.ones((3, 2)), index=pd.date_range("2023-01-03", periods=3), columns=["A", "B"])
    mask = build_tradable_mask(close, cache_root=str(tmp_path))
    assert mask.shape == close.shape
    assert mask.dtypes.eq(bool).all()
    assert mask.all().all()


def test_tradable_mask_st_and_suspension(tmp_path):
    """ST / 停牌在 T 日或 T+1 日任一为真 → 剔除（持续状态语义）。"""
    from data.tradability import build_tradable_mask

    idx = pd.date_range("2023-01-03", periods=3, freq="B")
    close = pd.DataFrame(10.0, index=idx, columns=["A", "B"])
    d0, d1, d2 = idx
    status = _status_multiindex({
        d0: {"A": (11.0, 9.0, True, False),   # A 当日即 ST
             "B": (11.0, 9.0, False, False)},
        d1: {"A": (11.0, 9.0, True, False),
             "B": (11.0, 9.0, False, True)},  # B 次日停牌
        d2: {"A": (11.0, 9.0, False, False),
             "B": (11.0, 9.0, False, False)},
    })
    status.to_parquet(tmp_path / "history_stock_status.parquet")
    mask = build_tradable_mask(close, bwd=None, cache_root=str(tmp_path))

    assert not mask.at[d0, "A"], "当日 ST 应剔除"
    assert not mask.at[d0, "B"], "T+1 停牌应剔除"
    assert not mask.at[d1, "A"], "T+1 仍为 ST 应剔除"
    assert mask.at[d2, "A"], "d2 无异常状态且 T+1(d2 之后无数据) → 保留"


def test_tradable_mask_carries_st_forward_after_status_gap(tmp_path):
    """状态表断更后，ST **不得**被 ``fillna(False)`` 静默放行（2026-11-10 事故）。

    实测现场：``history_stock_status`` 止于 2026-10-22，面板延到 11-04；
    从 10-23 起全市场剔除率由 3.4~4.8% 掉到 **0.0000**，且日志无任何提示，
    四只 ST 直接进了 solo 持仓（Top10 里 5 只 ST，全市场基线 3.67%）。

    规则：ST / 停牌是**持续状态**，断更期沿用最近可得值。
    """
    from data.tradability import build_tradable_mask

    idx = pd.date_range("2023-01-02", periods=5, freq="B")
    close = pd.DataFrame(10.0, index=idx, columns=["A", "B"])
    d0, d1, d2, d3, d4 = idx
    # 状态表只覆盖 d0..d2；A 在最后一个状态日仍是 ST
    status = _status_multiindex({
        d0: {"A": (11.0, 9.0, False, False), "B": (11.0, 9.0, False, False)},
        d1: {"A": (11.0, 9.0, False, False), "B": (11.0, 9.0, False, False)},
        d2: {"A": (11.0, 9.0, True, False), "B": (11.0, 9.0, False, False)},
    })
    status.to_parquet(tmp_path / "history_stock_status.parquet")
    mask = build_tradable_mask(close, bwd=None, cache_root=str(tmp_path))

    assert bool(mask.at[d1, "A"]) is False, "T+1(d2) 已 ST，d1 应剔除（覆盖区内不变）"
    for d in (d2, d3, d4):
        assert not mask.at[d, "A"], f"{d.date()} 断更期必须沿用 ST 状态"
    for d in (d0, d1, d2, d3, d4):
        assert mask.at[d, "B"], "非 ST 不受影响"


def test_signal_day_tradable_fallback_keeps_st(tmp_path):
    """状态表无信号日时，降级分支必须仍剔除 ST（原来只判涨跌停/停牌）。

    2026-11-02 现场：降级后 4 只 ST 全被标 ``tradable=True`` 并进 Top8。
    """
    from scripts.pipelines.alla_daily_rank import signal_day_tradable

    idx = pd.date_range("2023-01-02", periods=3, freq="B")
    d0, d1, d2 = idx
    status = _status_multiindex({
        d0: {"A": (11.0, 9.0, True, False), "B": (11.0, 9.0, False, False)},
        d1: {"A": (11.0, 9.0, True, False), "B": (11.0, 9.0, False, False)},
    })   # ← 状态表止于 d1，信号日 d2 缺失
    status.to_parquet(tmp_path / "history_stock_status.parquet")

    # 降级路径依赖日线兜底涨跌停：造一份最小 daily_all_a
    dly = pd.DataFrame(
        {"close": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
         "volume": [1e6] * 6},
        index=pd.MultiIndex.from_tuples(
            [(d, c) for d in (d0, d1, d2) for c in ("A", "B")],
            names=["date", "code"]))
    dly.to_parquet(tmp_path / "daily_all_a.parquet")

    close_raw = pd.DataFrame([[10.0, 10.0]], index=pd.DatetimeIndex([d2]),
                             columns=["A", "B"])
    ok = signal_day_tradable(close_raw, d2, cache_root=str(tmp_path))
    assert bool(ok["A"]) is False, "降级分支必须用最近可得状态剔除 ST"
    assert bool(ok["B"]) is True
