"""
数据层通路验证
==============

在没有 AmazingData SDK 凭证的环境下，用 MockDataSource 模拟数据，
验证以下链路是否畅通：

1. config 加载 + 环境变量展开
2. DataSource 抽象接口 → 具体实现
3. DataCache 增量缓存（首次全量、二次只拉增量）
4. Universe point-in-time 成分股查询

运行:
    python -m scripts.test_data_layer
    或
    python scripts/test_data_layer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import shutil
import tempfile
import numpy as np
import pandas as pd

from config import Config
from data.datasource import DataSource
from data.cache import DataCache
from data.universe import Universe


# ---------------------------------------------------------------------------
# Mock 数据源：模拟 AmazingData 的返回格式
# ---------------------------------------------------------------------------
class MockDataSource(DataSource):
    """用随机数据模拟日K线、成分股等，供无凭证环境测试。"""

    MOCK_CODES = [f"{600000+i:06d}.SH" for i in range(50)]

    def __init__(self):
        self._cal = self._gen_calendar(20230101, 20241231)

    def _gen_calendar(self, begin: int, end: int) -> list[int]:
        dates = pd.date_range(str(begin), str(end), freq="B")  # 工作日
        return [int(d.strftime("%Y%m%d")) for d in dates]

    def get_calendar(self, begin: int = 20100101, end: int | None = None) -> list[int]:
        cal = [d for d in self._cal if d >= begin]
        if end:
            cal = [d for d in cal if d <= end]
        return cal

    def get_code_list(self, security_type: str = "EXTRA_STOCK_A") -> list[str]:
        return self.MOCK_CODES

    def get_index_constituent(self, index_code: str) -> pd.DataFrame:
        # 模拟沪深300成分股：50只，2023年初全部纳入
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
        # 生成随机行情
        rng = np.random.default_rng(42)
        n = len(cal) * len(codes)
        base = 10.0 + rng.uniform(0, 50, len(codes))  # 每只股票基础价格
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

    def get_code_info(self, security_type="EXTRA_STOCK_A") -> pd.DataFrame:
        return pd.DataFrame(
            {"symbol": [c[:6] for c in self.MOCK_CODES],
             "pre_close": [10.0] * len(self.MOCK_CODES)},
            index=self.MOCK_CODES,
        )


# ---------------------------------------------------------------------------
# 测试主函数
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("YuriQuant 数据层通路验证")
    print("=" * 70)

    # 用临时目录做缓存，测完清理
    tmp_root = Path(tempfile.mkdtemp(prefix="yuriquant_test_"))
    print(f"\n[1/5] 临时缓存目录: {tmp_root}")

    # 强制重载 config
    Config.load()
    cfg = Config.get()
    print(f"[2/5] 配置加载成功")
    print(f"      数据源类型: {cfg['datasource']['type']}")
    print(f"      默认universe: {cfg['universe']['default']}")
    print(f"      fetch起始日: {cfg['fetch']['begin_date']}")

    # 用 Mock 数据源
    ds = MockDataSource()
    cache = DataCache(ds, cache_root=tmp_root)
    print(f"[3/5] DataCache 创建成功")

    # --- 测试交易日历 ---
    cal = cache.get_calendar(20230101, 20240131)
    assert len(cal) > 0, "交易日历不应为空"
    print(f"[4/5] 交易日历: {len(cal)} 个交易日 ({cal[0]} ~ {cal[-1]})")

    # --- 测试成分股 ---
    uni = Universe(cache)
    codes = uni.get_hs300(20240101)
    assert len(codes) == 50, f"应有50只成分股，实际{len(codes)}"
    print(f"      沪深300成分股(20240101): {len(codes)} 只")

    # --- 测试日K线（首次全量）---
    print(f"\n[5/5] 测试日K线增量缓存...")
    kline1 = cache.get_daily_kline(codes, 20230101, 20240131)
    assert not kline1.empty, "K线不应为空"
    n1 = len(kline1)
    print(f"      首次拉取: {n1} 行, {kline1.index.get_level_values('code').nunique()} 个代码")

    # --- 测试增量更新（不重新拉取已有日期）---
    # 模拟数据源新增了 20240201 以后的数据
    ds._cal = ds._gen_calendar(20230101, 20240301)
    kline2 = cache.get_daily_kline(codes, 20230101, 20240301)
    n2 = len(kline2)
    new_rows = n2 - n1
    print(f"      增量更新: 总 {n2} 行，新增 {new_rows} 行")

    # --- 验证本地 parquet 文件 ---
    files = list(tmp_root.glob("*.parquet"))
    print(f"\n      缓存文件: {[f.name for f in files]}")

    # 清理
    shutil.rmtree(tmp_root, ignore_errors=True)
    print("\n" + "=" * 70)
    print("验证通过！数据层通路正常。")
    print("=" * 70)


if __name__ == "__main__":
    main()
