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
        self._cal = self._gen_calendar(20230101, 20241231)

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
        return pd.DataFrame(
            {"symbol": [c[:6] for c in self.MOCK_CODES],
             "pre_close": [10.0] * len(self.MOCK_CODES)},
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


@pytest.fixture
def mock_ds() -> MockDataSource:
    return MockDataSource()
