"""行业上下文（`scripts/builders/industry_context.py`）单元测试
================================================================

只测**不依赖真实数据**的两个纯函数：`agg_industry`（个股截面 → 行业截面）
和 `rebal_dates`（调仓日）。`build_context` / `float_share_panel` 依赖
`{data_root}/parquet` 下的真实表，归 `--check` 体检（见模块 docstring），
这里不重复造桩。

重点锁两件事：
1. `agg_industry(winsor=None)` 必须与 `industry_rotation.agg_sue` 的
   `Σ(w·x)/Σ(w)` + `cnt >= 5` 实现**逐位一致** —— 那是丢失原件 `_agg` 的
   唯一结构等价参照物；
2. 缩尾只在行业内、且只在成员数够的行业里生效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.builders.industry_context import agg_industry, rebal_dates


def _bincount_ref(v, lbl, w, n_ind: int = 31, min_n: int = 5) -> np.ndarray:
    """`industry_rotation.agg_sue:107-113` 的同构实现（无缩尾）。"""
    x = np.asarray(v, float)
    L = np.asarray(lbl, int)
    W = np.asarray(w, float)
    num = np.bincount(L, weights=x * W, minlength=n_ind)
    den = np.bincount(L, weights=W, minlength=n_ind)
    cnt = np.bincount(L, minlength=n_ind)
    return np.where((cnt >= min_n) & (den > 0), num / np.where(den == 0, 1, den), np.nan)


class TestAggIndustry:
    def test_weighted_mean(self):
        v = np.array([1.0, 3.0, 10.0, 20.0, 40.0, 60.0])
        lbl = np.array([0, 0, 0, 0, 0, 1])
        w = np.array([1.0, 1.0, 1.0, 3.0, 3.0, 1.0])
        out = agg_industry(v, lbl, w, winsor=None)
        assert out[0] == (1 + 3 + 10 + 3 * 20 + 3 * 40) / 9
        assert np.isnan(out[1])          # 只有 1 只，低于 min_members
        assert out.shape == (31,)

    def test_matches_bincount_reference(self):
        rng = np.random.default_rng(0)
        n = 500
        v = rng.normal(size=n)
        lbl = rng.integers(0, 31, size=n)
        w = rng.uniform(0.1, 1e6, size=n)
        got = agg_industry(v, lbl, w, winsor=None)
        want = _bincount_ref(v, lbl, w)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_masks_bad_label_and_weight(self):
        v = np.arange(1.0, 8.0)                        # 1..7
        lbl = np.array([0, 0, 0, 0, 0, 0, -1])         # 最后一只无行业归属
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0])  # 第 6 只权重 0
        out = agg_industry(v, lbl, w, winsor=None)
        assert out[0] == (1 + 2 + 3 + 4 + 5) / 5       # 6 与 7 都不参与
        assert np.isnan(out[1:]).all()

    def test_nan_value_ignored(self):
        v = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        lbl = np.zeros(6, dtype=int)
        w = np.ones(6)
        assert agg_industry(v, lbl, w, winsor=None)[0] == (1 + 3 + 4 + 5 + 6) / 5

    def test_all_invalid_returns_nan(self):
        out = agg_industry(np.array([np.nan, np.nan]), np.array([0, 0]),
                           np.array([0.0, 0.0]))
        assert np.isnan(out).all()

    def test_shape_mismatch_returns_nan(self):
        out = agg_industry(np.array([1.0, 2.0]), np.array([0]), np.array([1.0, 1.0]))
        assert np.isnan(out).all()

    def test_winsor_clips_within_industry_only(self):
        """组 0 有一个 1e6 的离群值，组 1 全在正常量级 —— 缩尾只该动组 0。"""
        rng = np.random.default_rng(1)
        n_ind, m = 2, 200
        v = np.concatenate([rng.normal(0, 1, m), rng.normal(5, 1, m)])
        v[0] = 1e6
        lbl = np.repeat([0, 1], m)
        w = np.ones(2 * m)
        raw = agg_industry(v, lbl, w, winsor=None)
        win = agg_industry(v, lbl, w, winsor=(1.0, 99.0))
        assert win[0] < raw[0] / 10          # 离群值被压回分位内
        # 另一组无离群值：缩尾只削掉两端各 1% 的尾部，均值变化应在千分位量级
        assert abs(win[1] - raw[1]) < 0.02 * abs(raw[1])

    def test_winsor_skipped_for_small_industry(self):
        """成员 < WINSOR_MIN_N(20) 的行业不缩尾，避免分位数失真。"""
        v = np.array([0.0] * 9 + [100.0])
        lbl = np.zeros(10, dtype=int)
        w = np.ones(10)
        assert agg_industry(v, lbl, w, winsor=(1.0, 99.0))[0] == np.mean(v)

    def test_empty_input(self):
        assert np.isnan(agg_industry(np.array([]), np.array([]), np.array([]))).all()


class TestFreeFloat:
    """自由流通近似：扣减规则、去重、PIT 可用日。

    ⚠️ 去重这条不是理论洁癖：真实数据里同一份报表的股东会出现 2~4 次
    （「十大股东」与「十大流通股东」两张表并列），不去重会把 601398 扣成
    11124 亿股 > 流通股本 3564 亿股，全市场被 floor 压平。
    """

    @staticmethod
    def _patch(monkeypatch, holder, float_shares):
        """把股本桩做成**和真实现一样的前向填充**（否则 PIT 断言会测到桩的 bug）。"""
        from scripts.builders import industry_context as ic

        def fake_share_panel(dates, codes, **kw):
            want = pd.DatetimeIndex(dates)
            full = pd.DatetimeIndex(sorted(set(float_shares.index) | set(want)))
            w = float_shares.reindex(index=full).ffill()
            return w.reindex(index=want, columns=list(codes))

        monkeypatch.setattr(ic, "_share_holder", lambda: holder)
        monkeypatch.setattr(ic, "float_share_panel", fake_share_panel)
        return ic

    def test_dedupes_duplicate_holder_lists(self, monkeypatch):
        d = pd.Timestamp("2024-06-30")
        holder = pd.DataFrame({
            "code": ["A"] * 4, "ann_date": [d] * 4, "holder_end_date": [d] * 4,
            "holder_name": ["大股东", "大股东", "小散", "小散"],
            "holder_pct": [60.0, 60.0, 1.0, 1.0],
            "float_quantity": [600.0, 600.0, 10.0, 10.0],
        })
        fs = pd.DataFrame({"A": [1000.0]}, index=pd.DatetimeIndex([d]))
        ic = self._patch(monkeypatch, holder, fs)
        out = ic.free_float_share_panel([d], ["A"])
        assert out.loc[d, "A"] == 400.0        # 1000 − 600，只扣一次

    def test_floor_prevents_zero(self, monkeypatch):
        d = pd.Timestamp("2024-06-30")
        holder = pd.DataFrame({
            "code": ["A"], "ann_date": [d], "holder_end_date": [d],
            "holder_name": ["集团"], "holder_pct": [95.0], "float_quantity": [990.0],
        })
        fs = pd.DataFrame({"A": [1000.0]}, index=pd.DatetimeIndex([d]))
        ic = self._patch(monkeypatch, holder, fs)
        assert ic.free_float_share_panel([d], ["A"], floor_ratio=0.05).loc[d, "A"] == 50.0

    def test_pit_uses_announcement_date_not_period_end(self, monkeypatch):
        period_end = pd.Timestamp("2024-03-31")
        ann = pd.Timestamp("2024-04-25")
        holder = pd.DataFrame({
            "code": ["A"], "ann_date": [ann], "holder_end_date": [period_end],
            "holder_name": ["集团"], "holder_pct": [60.0], "float_quantity": [600.0],
        })
        fs = pd.DataFrame({"A": [1000.0]},
                          index=pd.DatetimeIndex([period_end, ann]))
        ic = self._patch(monkeypatch, holder, fs)
        dates = pd.DatetimeIndex(["2024-04-10", "2024-04-30"])
        out = ic.free_float_share_panel(dates, ["A"])
        assert out.loc["2024-04-10", "A"] == 1000.0   # 未公告 ⇒ 不扣减（回退流通股本）
        assert out.loc["2024-04-30", "A"] == 400.0    # 公告后 ⇒ 扣减生效

    def test_small_holders_not_deducted(self, monkeypatch):
        d = pd.Timestamp("2024-06-30")
        holder = pd.DataFrame({
            "code": ["A"] * 2, "ann_date": [d] * 2, "holder_end_date": [d] * 2,
            "holder_name": ["基金甲", "基金乙"], "holder_pct": [3.0, 4.0],
            "float_quantity": [30.0, 40.0],
        })
        fs = pd.DataFrame({"A": [1000.0]}, index=pd.DatetimeIndex([d]))
        ic = self._patch(monkeypatch, holder, fs)
        assert ic.free_float_share_panel([d], ["A"]).loc[d, "A"] == 1000.0

    def test_weekly_first_trading_day(self):
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03",
                                "2024-01-08", "2024-01-09", "2024-01-15"])
        got = rebal_dates(idx, "W")
        assert got == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08"),
                       pd.Timestamp("2024-01-15")]

    def test_monthly_first_trading_day(self):
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-31",
                                "2024-02-01", "2024-02-29"])
        got = rebal_dates(idx, "M")
        assert got == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01")]

    def test_start_filters_after_grouping(self):
        """start 落在周中时，那一周已被更早的首日占用 ⇒ 不重复计入。"""
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-08"])
        got = rebal_dates(idx, "W", start=pd.Timestamp("2024-01-03"))
        assert got == [pd.Timestamp("2024-01-08")]

    def test_daily_is_identity(self):
        """`"D"` = 每个交易日都是调仓日（日频调仓诊断用）。"""
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
        assert rebal_dates(idx, "D") == list(idx)
        assert rebal_dates(idx, "D", start=pd.Timestamp("2024-01-03")) == \
            [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]

    def test_unsorted_input_is_sorted(self):
        """输入乱序时结果必须升序（`"D"` 直接透传 idx，不排序会错位）。"""
        idx = pd.DatetimeIndex(["2024-01-08", "2024-01-02", "2024-01-03"])
        assert rebal_dates(idx, "D") == sorted(idx)
        assert rebal_dates(idx, "W") == [pd.Timestamp("2024-01-02"),
                                        pd.Timestamp("2024-01-08")]

    def test_matches_report_replica(self):
        """与 `report_replica.rebal_dates` 逐位一致（本模块是它的复刻）。"""
        from scripts.portfolio.report_replica import rebal_dates as rr
        idx = pd.bdate_range("2023-01-02", "2023-12-29")
        for freq in ("W", "M"):
            assert rebal_dates(idx, freq, pd.Timestamp("2023-06-01")) == \
                rr(idx, pd.Timestamp("2023-06-01"), freq)
