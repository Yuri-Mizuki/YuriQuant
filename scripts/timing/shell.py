"""纯择时研究线共享件：调仓壳 + 指数日K装载（2026-11-21 自 sws_regime_probe 抽取）。

调仓壳 = 申万 0922 研报 §3 的动态仓位规则（sws_timing_probe 探针中唯一正超额形态）：
每周 `weekday` 调仓，回看期收益 = 上次调仓日收盘 → 本次调仓前一交易日收盘；
信号 ≥0.5 执行追涨杀跌（趋势模式）、<0.5 执行均值回归（震荡模式），
每满 ``step_pct``% 动 ``step`` 仓，钳 [0,1]；首日以 ``init_pos`` 建仓（建仓成本不计）。

缝合纪律（TODO.md §〇·二 P2 纯择时线）：信号策略必须与退化形态（sig 恒 0/恒 1）
和持有基准同台；门在报告参数臂上判，参数网格只做敏感性不用于选峰。
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_index_daily(code: str, begin: int = 20140101, cache_dir: Path | None = None,
                     refresh: bool = False) -> pd.DataFrame:
    """指数日K（AmazingData），带 parquet 缓存（>5 天自动重拉）。"""
    cache = (cache_dir or ROOT / "reports") / "data_cache" / f"{code}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if df.index.max() >= pd.Timestamp(dt.date.today()) - pd.Timedelta(days=5):
            return df
    from data.datasource import create_datasource

    ds = create_datasource()
    raw = ds.get_daily_kline([code], begin, int(dt.date.today().strftime("%Y%m%d")))
    df = raw.xs(code, level="code").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def run_rebalance_shell(
    close: pd.Series,
    sig: pd.Series,
    cost_bps: float = 0.0,
    step_pct: float = 1.0,
    step: float = 0.1,
    init_pos: float = 0.5,
    weekday: int = 2,
) -> dict:
    """周频调仓壳（研报 §3）。``sig`` 须已延迟到 T+1 可用；收盘执行、次日起生效。

    返回 dict：ret（日收益，含成本）/ pos（每日生效仓位）/ avg_pos / turnover（双边）。
    """
    ret = close.pct_change().fillna(0.0)
    dates = close.index
    pos_eff = pd.Series(init_pos, index=dates)
    cost = pd.Series(0.0, index=dates)
    cur = init_pos
    prev_r: pd.Timestamp | None = None
    for r in [d for d in dates if d.weekday() == weekday]:
        loc = dates.get_loc(r)
        if loc == 0:
            continue
        if prev_r is not None:
            lb = float(close.iloc[loc - 1] / close.loc[prev_r] - 1.0)
        else:
            lb = 0.0  # 首个调仓日无回看期，仓位不动
        s = float(sig.loc[r])
        pct = float(np.floor(abs(lb) / (step_pct / 100.0)) * step) * np.sign(lb)
        new = float(np.clip(cur + (pct if s >= 0.5 else -pct), 0.0, 1.0))
        cost.iloc[loc] = abs(new - cur) * cost_bps * 1e-4
        cur = new
        pos_eff.iloc[loc + 1:] = cur
        prev_r = r
    sr = pos_eff * ret - cost
    return {
        "ret": sr,
        "pos": pos_eff,
        "avg_pos": float(pos_eff.mean()),
        "turnover": float(cost.sum() / max(cost_bps * 1e-4, 1e-12)) if cost_bps > 0 else float("nan"),
    }
