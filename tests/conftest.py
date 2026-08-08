"""
pytest 共享 fixtures
====================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.datasource import DataSource


class MockDataSource(DataSource):
    """用随机数据模拟日K线、成分股等，供无凭证环境测试。"""

    MOCK_CODES = [f"{600000+i:06d}.SH" for i in range(50)]

    def __init__(self):
        # 日历含 2022 起：历史回补类测试（拉更早区间触发数据源）依赖 2022 交易日存在
        self._cal = self._gen_calendar(20220101, 20241231)

    def _gen_calendar(self, begin: int, end: int) -> list[int]:
        dates = pd.date_range(str(begin), str(end), freq="B")
        return [int(d.strftime("%Y%m%d")) for d in dates]

    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        cal = [d for d in self._cal if d >= begin]
        if end:
            cal = [d for d in cal if d <= end]
        return cal

    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        return self.MOCK_CODES

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        rows = []
        for code in self.MOCK_CODES:
            rows.append({
                "con_code": code,
                "in_date": "2023-01-03",
                "out_date": pd.NaT,
                "INDEX_NAME": "沪深300",
            })
        return pd.DataFrame(rows)

    def get_daily_kline(self, code_list, begin_date, end_date) -> pd.DataFrame:
        codes = list(code_list)
        cal = [d for d in self._cal if begin_date <= d <= end_date]
        if not cal:
            return pd.DataFrame()
        rng = np.random.default_rng(42)
        base = 10.0 + rng.uniform(0, 50, len(codes))
        frames = []
        for ci, code in enumerate(codes):
            prices = base[ci] * (1 + rng.normal(0, 0.02, len(cal)).cumprod())
            df = pd.DataFrame({
                "code": code,
                "open": prices,
                "high": prices * (1 + rng.uniform(0, 0.03, len(cal))),
                "low": prices * (1 - rng.uniform(0, 0.03, len(cal))),
                "close": prices * (1 + rng.normal(0, 0.01, len(cal))),
                "volume": rng.integers(1e6, 1e8, len(cal)),
                "amount": rng.uniform(1e7, 1e9, len(cal)),
            }, index=pd.to_datetime([str(d) for d in cal]))
            df.index.name = "date"
            frames.append(df)
        out = pd.concat(frames).reset_index().set_index(["date", "code"])
        return out.sort_index()

    # ---- 分钟K线（mock：按 A 股交易时段生成，48 根/日 @5分钟）----
    @staticmethod
    def _intraday_times(date_int: int, period: int) -> list[pd.Timestamp]:
        """A 股交易时段 9:30-11:30 / 13:00-15:00，bar 标签 = 时段开始时刻。

        每半天 120//period 根：9:30, 9:30+period, ... （period=5 → 48 根/日）。
        """
        date = pd.Timestamp(str(date_int))
        n = 120 // period
        morning = [
            date + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=period * i)
            for i in range(n)
        ]
        afternoon = [
            date + pd.Timedelta(hours=13) + pd.Timedelta(minutes=period * i)
            for i in range(n)
        ]
        return morning + afternoon

    def get_minute_kline(self, code_list, begin_date, end_date, period=5,
                         begin_time=None, end_time=None) -> pd.DataFrame:
        codes = list(code_list)
        cal = [d for d in self._cal if begin_date <= d <= end_date]
        if not cal:
            return pd.DataFrame(
                columns=["kline_time", "code", "open", "high", "low", "close", "volume", "amount"]
            )
        bars_per_day = 240 // period
        rng = np.random.default_rng(42)
        rows = []
        for code in codes:
            for d in cal:
                times = self._intraday_times(d, period)
                # 日内时段过滤（与 AmazingDataSource 的 begin_time/end_time 行为一致）
                if begin_time is not None:
                    times = [t for t in times if t.hour * 100 + t.minute >= begin_time]
                if end_time is not None:
                    times = [t for t in times if t.hour * 100 + t.minute <= end_time]
                if not times:
                    continue
                day_open = 10.0 * (1 + rng.normal(0, 0.01))
                # 日内随机游走：首根 open = 当日开盘价，末根 close 为随机游走终点
                n = len(times)
                path = np.cumsum(rng.normal(0, 0.002, n))
                opens = day_open * (1 + np.concatenate([[0.0], path[:-1]]))
                closes = day_open * (1 + path)
                highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.003, n))
                lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.003, n))
                vols = rng.integers(1e4, 1e6, n)
                for i in range(n):
                    rows.append({
                        "kline_time": times[i],
                        "code": code,
                        "open": float(opens[i]),
                        "high": float(highs[i]),
                        "low": float(lows[i]),
                        "close": float(closes[i]),
                        "volume": float(vols[i]),
                        "amount": float(vols[i] * closes[i]),
                    })
        out = pd.DataFrame(rows).set_index(["kline_time", "code"])
        return out.sort_index()

    def get_adj_factor(self, code_list) -> pd.DataFrame:
        codes = list(code_list)
        cal = self._cal
        rng = np.random.default_rng(7)
        data = 1.0 + rng.uniform(0, 0.5, (len(cal), len(codes)))
        return pd.DataFrame(data, index=pd.to_datetime([str(d) for d in cal]), columns=codes)

    def get_backward_factor(self, code_list) -> pd.DataFrame:
        codes = list(code_list)
        cal = self._cal
        rng = np.random.default_rng(11)
        # 后复权因子是累积值，围绕 1.0 缓慢漂移，量级上比单次复权因子更接近 1
        data = 1.0 + np.cumsum(rng.normal(0, 0.001, (len(cal), len(codes))), axis=0)
        return pd.DataFrame(data, index=pd.to_datetime([str(d) for d in cal]), columns=codes)

    def get_code_info(self, security_type="EXTRA_STOCK_A") -> pd.DataFrame:
        # 列结构对齐基类 docstring：symbol/status/pre_close/high_limited/low_limited/price_tick
        n = len(self.MOCK_CODES)
        return pd.DataFrame(
            {"symbol": [c[:6] for c in self.MOCK_CODES],
             "status": ["正常"] * n,
             "pre_close": [10.0] * n,
             "high_limited": [11.0] * n,
             "low_limited": [9.0] * n,
             "price_tick": [0.01] * n},
            index=self.MOCK_CODES,
        )

    def get_history_stock_status(self, code_list, begin_date, end_date) -> pd.DataFrame:
        codes = list(code_list)
        cal = [d for d in self._cal if begin_date <= d <= end_date]
        if not cal:
            return pd.DataFrame(
                columns=["date", "code", "pre_close", "high_limited", "low_limited",
                         "is_st", "is_suspended", "is_ex_dividend", "is_ex_rights"]
            )
        rows = []
        for code in codes:
            for d in cal:
                rows.append({
                    "date": pd.Timestamp(str(d)),
                    "code": code,
                    "pre_close": 10.0,
                    "high_limited": 11.0,
                    "low_limited": 9.0,
                    "is_st": False,
                    "is_suspended": False,
                    "is_ex_dividend": False,
                    "is_ex_rights": False,
                })
        return pd.DataFrame(rows)

    def get_industry_classification(self, level: int = 1) -> pd.DataFrame:
        # 把 50 只 mock 股票均分到 3 个行业，纳入日期与成分股一致（2023-01-03）
        industries = ["食品饮料", "银行", "电子"]
        rows = []
        for i, code in enumerate(self.MOCK_CODES):
            industry = industries[i % len(industries)]
            rows.append({
                "code": code,
                "industry_code": f"IND{i % len(industries):02d}.SI",
                "industry_name": industry,
                "level": level,
                "in_date": "2023-01-03",
                "out_date": pd.NaT,
            })
        return pd.DataFrame(rows)

    def get_equity_structure(self, code_list) -> pd.DataFrame:
        codes = list(code_list)
        rng = np.random.default_rng(13)
        rows = []
        for code in codes:
            rows.append({
                "code": code,
                "change_date": pd.Timestamp("2023-01-03"),
                "tot_share": float(rng.uniform(5000, 100000)),   # 万股
                "float_share": float(rng.uniform(3000, 80000)),  # 万股
            })
        return pd.DataFrame(rows)

    # ---- 分红 / 十大股东 / 股东户数（mock：简化事件，2026-08-04 新增接口）----
    def get_dividend(self, code_list) -> pd.DataFrame:
        codes = list(code_list)
        rows = []
        for code in codes:
            rows.append({
                "code": code,
                "ann_date": pd.Timestamp("2023-06-20"),
                "record_date": pd.Timestamp("2023-06-23"),
                "ex_date": pd.Timestamp("2023-06-26"),
                "payout_date": pd.Timestamp("2023-06-26"),
                "report_period": pd.Timestamp("2022-12-31"),
                "cash_per_share_pre_tax": 0.5,
                "bonus_rate": 0.0,
                "base_share": 100000.0,   # 万股
            })
        return pd.DataFrame(rows)

    def get_share_holder(self, code_list) -> pd.DataFrame:
        codes = list(code_list)
        rows = []
        for code in codes:
            for k in range(10):
                rows.append({
                    "code": code,
                    "ann_date": pd.Timestamp("2023-04-30"),
                    "holder_end_date": pd.Timestamp("2023-03-31"),
                    "holder_name": f"机构{k}" if k < 5 else f"自然人{k}",
                    "holder_pct": 3.0 - 0.2 * k,
                    "holder_quantity": 1e6 * (10 - k),
                    "float_quantity": 1e6 * (10 - k),
                })
        return pd.DataFrame(rows)

    def get_holder_num(self, code_list) -> pd.DataFrame:
        codes = list(code_list)
        rows = []
        for code in codes:
            for q, base in zip(range(4), (20000, 21000, 20500, 22000)):
                rows.append({
                    "code": code,
                    "ann_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=90 * q),
                    "holder_end_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=90 * q),
                    "holder_num": float(base + q * 100),
                })
        return pd.DataFrame(rows)

    # ---- 财务报表（mock：返回一份简单季报）----
    def _mock_financial(self, code_list, field, rng):
        codes = list(code_list)
        rows = []
        for code in codes:
            for q in range(4):  # 4 个报告期
                rows.append({
                    "code": code,
                    "ann_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=90 * q),
                    "report_period": pd.Timestamp("2023-01-01") + pd.Timedelta(days=90 * q),
                    "statement_type": "1",
                    "report_type": "年报" if q == 0 else "季报",
                    field: float(rng.normal(1e8, 1e7)),
                })
        return pd.DataFrame(rows)

    def get_balance_sheet(self, code_list, begin_date=None, end_date=None) -> pd.DataFrame:
        df = self._mock_financial(code_list, "TOTAL_ASSETS", np.random.default_rng(21))
        # 补 TOT_SHARE / EQUITY（市值、估值因子用；2026-08-04 分红因子依赖市值）
        codes = list(code_list)
        rng = np.random.default_rng(22)
        rows = []
        for code in codes:
            for q in range(4):
                rows.append({
                    "code": code,
                    "ann_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=90 * q),
                    "report_period": pd.Timestamp("2023-01-01") + pd.Timedelta(days=90 * q),
                    "statement_type": "1",
                    "report_type": "年报" if q == 0 else "季报",
                    "TOT_SHARE": float(rng.uniform(1e8, 1e9)),      # 股
                    "TOT_SHARE_EQUITY_EXCL_MIN_INT": float(rng.uniform(1e9, 1e10)),
                })
        extra = pd.DataFrame(rows)
        cols = ["code", "ann_date", "report_period", "statement_type", "report_type",
                "TOTAL_ASSETS", "TOT_SHARE", "TOT_SHARE_EQUITY_EXCL_MIN_INT"]
        out = df.merge(extra.drop(columns=["statement_type", "report_type"]),
                       on=["code", "ann_date", "report_period"], how="left")
        return out[[c for c in cols if c in out.columns]]

    def get_cash_flow(self, code_list, begin_date=None, end_date=None) -> pd.DataFrame:
        return self._mock_financial(code_list, "WS_OPERA_ACT", np.random.default_rng(22))

    def get_income(self, code_list, begin_date=None, end_date=None) -> pd.DataFrame:
        df = self._mock_financial(code_list, "OPERA_REV", np.random.default_rng(23))
        df["NET_PRO_INCL_MIN_INT_INC"] = df["OPERA_REV"] * 0.1
        df["BASIC_EPS"] = 0.5
        return df


@pytest.fixture
def mock_ds() -> MockDataSource:
    return MockDataSource()
