"""
日内特征 → 日频因子构建与入库
==============================

路径 A 的落地实现：把 5 分钟 K 线的日内信息压成"每天一个值"的日频横截面
因子，注册进 FactorLibrary（与现有挖掘/合成因子同库对比、同管线迭代）。

特征清单（14 个，全部只用当日收盘前可知的日内信息，无未来函数）：
- 波动率类：intraday_rv（已实现波动率）、intraday_rv_z（波动率 Z）、
  intraday_rsj（好坏波动率差）、intraday_range（日内振幅）
- 收益结构：overnight_ret（隔夜收益）、intraday_ret（日内收益）、
  overnight_intraday_diff（日内-隔夜差）
- 时段动量：open30_ret（开盘 30 分钟）、close30_ret（尾盘 30 分钟）、
  am_pm_diff（上午-下午差）、first_bar_ret（首根 5 分钟 bar）
- 量价/流动性：intraday_vwap_dev（收盘 VWAP 偏离）、
  intraday_vol_ratio（量比）、intraday_amihud（日内非流动性）

口径说明
--------
- 5 分钟 bar 内收益：bar_ret = close/open - 1（避免跨日拼接）。
- 已实现波动率 RV = Σ r_i²，r_0 = close_0/open_0 - 1（首根），
  r_i = close_i/close_{i-1} - 1（i>=1），无跨日。
- 除权/除息日样本剔除（分钟线不复权，除权日 bar 有跳变污染特征）。
- 每个因子面板按日截面 zscore 标准化后入库；IC 用 spearman
  （factor[t] vs returns[t+1]，returns = 日线次日收益）。
- 因子在 t 日收盘后可得（用到当日全部 5 分钟数据），预测 t+1 日收益，
  与日频因子库口径一致。

用法
----
    python -m scripts.build_intraday_factors --offline            # 读缓存（推荐）
    python -m scripts.build_intraday_factors --mock               # mock 验证
    python -m scripts.build_intraday_factors --offline --factors rv,overnight_ret  # 子集
    python -m scripts.build_intraday_factors --offline --dataset hs300_2025 --no-save  # 只算不入库
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data.cache import DataCache
from data.cache_helpers import load_daily
from data.datasource import create_datasource
from data.offline import OfflineDataSource
from data.universe import Universe
from factor.preprocessing import standardize_zscore
from research.factor_library import FactorLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_intraday_factors")

PERIOD = 5
_BARS_PER_DAY = 240 // PERIOD          # 48
_OPEN30_BARS = 6                       # 开盘 30 分钟 = 6 根
_CLOSE30_BARS = 6                      # 尾盘 30 分钟 = 6 根
_LOOKBACK = 20                         # 波动率 Z / 量比的回看窗口
_ROLL_MIN = 10                         # 回看窗口最小样本

# 特征定义：(name, 中文公式说明)
FACTOR_DEFS: dict[str, str] = {
    "intraday_rv": "已实现波动率 Σr²（5分钟收益平方和，日内）",
    "intraday_rv_z": "已实现波动率 Z：当日 RV / 过去20日平均 RV",
    "intraday_rsj": "好坏波动率差 (RV_up-RV_down)/(RV_up+RV_down)",
    "intraday_range": "日内振幅 (当日max(high)-min(low))/当日open",
    "overnight_ret": "隔夜收益 open/prev_close-1",
    "intraday_ret": "日内收益 close/open-1",
    "overnight_intraday_diff": "日内收益 - 隔夜收益",
    "open30_ret": "开盘30分钟累计收益（前6根bar）",
    "close30_ret": "尾盘30分钟累计收益（后6根bar）",
    "am_pm_diff": "上午收益 - 下午收益",
    "first_bar_ret": "首根5分钟bar收益（9:30）",
    "intraday_vwap_dev": "收盘价偏离日内VWAP close/VWAP-1",
    "intraday_vol_ratio": "量比：当日成交量/过去20日均量",
    "intraday_amihud": "日内非流动性 mean(|bar_ret|/amount)",
}


def load_data(cache, uni, index_code, begin, end):
    codes, cal, daily = load_daily(cache, uni, index_code, begin, end)
    target = end or cal[-1]
    mk = cache.get_minute_kline(codes, begin, target, period=PERIOD)
    status = cache.get_history_stock_status(codes, begin, target)
    log.info("日线 %d 行 / %d分钟 %d 行 / 状态 %d 行", len(daily), PERIOD, len(mk), len(status))
    return codes, daily, mk, status


def _ex_div_keys(status) -> set:
    """返回除权/除息日 (date, code) 集合。"""
    if status is None or status.empty:
        return set()
    st = status.copy()
    if isinstance(st.index, pd.MultiIndex):
        st = st.reset_index()
    flags = st[st["is_ex_dividend"] | st["is_ex_rights"]]
    if flags.empty:
        return set()
    return set(zip(pd.to_datetime(flags["date"]).dt.normalize(), flags["code"]))


def _minute_frame(mk, status) -> pd.DataFrame:
    """5 分钟长表：date/code/bar_ret/vol/amount/typical，剔除除权日。"""
    df = mk.reset_index()
    df["date"] = df["kline_time"].dt.normalize()
    bad = _ex_div_keys(status)
    if bad:
        mask = ~pd.Series(
            [(d, c) in bad for d, c in zip(df["date"], df["code"])], index=df.index
        )
        df = df[mask]
    df = df.dropna(subset=["open", "high", "low", "close", "volume", "amount"])
    df = df.sort_values(["date", "code", "kline_time"]).reset_index(drop=True)
    # bar 内收益（无跨日）：首根 close/open-1，其余 close/prev_close-1
    df["prev_close"] = df.groupby(["date", "code"])["close"].shift(1)
    df["base"] = df["prev_close"].fillna(df["open"])
    df["bar_ret"] = df["close"] / df["base"] - 1.0
    df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


def _panel_from_series(s: pd.Series, daily_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """(date, code) 长表 Series → date×code 宽表，与日线日期对齐。"""
    p = s.unstack("code")
    return p.reindex(index=daily_dates)


# ---- 特征计算 ----


def build_features(mf: pd.DataFrame, daily: pd.DataFrame,
                   status: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """计算全部日内特征面板（date×code）。

    mf: _minute_frame 输出的 5 分钟长表。
    daily: (date, code) 日线（用于隔夜/日内收益）。
    status: 历史状态表，用于剔除 daily 部分的除权日样本。
    返回 {name: 原始面板}。
    """
    daily_dates = pd.DatetimeIndex(sorted(daily.index.get_level_values("date").unique()))
    codes = sorted(mf["code"].unique())
    out: dict[str, pd.DataFrame] = {}

    # ---- 日线口径 ----
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    bad = _ex_div_keys(status)
    if bad:
        keep = ~pd.Series(
            [(dt, c) in bad for dt, c in zip(d["date"], d["code"])], index=d.index
        )
        d = d[keep]
    open_w = d.pivot(index="date", columns="code", values="open")
    close_w = d.pivot(index="date", columns="code", values="close")
    prev_close = close_w.shift(1)
    out["overnight_ret"] = open_w / prev_close - 1.0
    out["intraday_ret"] = close_w / open_w - 1.0
    out["overnight_intraday_diff"] = out["intraday_ret"] - out["overnight_ret"]

    # ---- 分钟口径：按 (date, code) 分组聚合 ----
    g = mf.groupby(["date", "code"], sort=True)
    bar_ret = mf.set_index(["date", "code"])["bar_ret"]
    vol = mf.set_index(["date", "code"])["volume"]
    amt = mf.set_index(["date", "code"])["amount"]
    typ = mf.set_index(["date", "code"])["typical"]
    hi = mf.set_index(["date", "code"])["high"]
    lo = mf.set_index(["date", "code"])["low"]
    op = mf.set_index(["date", "code"])["open"]
    cl = mf.set_index(["date", "code"])["close"]

    # 已实现波动率
    rv = g["bar_ret"].apply(lambda s: float((s * s).sum()))
    out["intraday_rv"] = _panel_from_series(rv, daily_dates)

    # 好坏波动率差 RSJ
    rv_up = g["bar_ret"].apply(lambda s: float(s[s > 0].pow(2).sum()))
    rv_dn = g["bar_ret"].apply(lambda s: float(s[s < 0].pow(2).sum()))
    rsj = (rv_up - rv_dn) / (rv_up + rv_dn).replace(0.0, np.nan)
    out["intraday_rsj"] = _panel_from_series(rsj, daily_dates)

    # 日内振幅
    def _range(x):
        h = float(x["high"].max()); l = float(x["low"].min()); o = float(x["open"].iloc[0])
        return np.nan if o == 0 else (h - l) / o
    out["intraday_range"] = _panel_from_series(g.apply(_range, include_groups=False), daily_dates)

    # 时段收益
    def _seg_sum(times, lo_hm, hi_hm):
        # 用时间标签过滤（字符串比较可靠）
        return lambda s: float(s[s.index.get_level_values("kline_time").strftime("%H:%M").between(lo_hm, hi_hm)].sum())
    # 前 6 根 = 9:30-9:55；后 6 根 = 14:30-14:55；上午 9:30-11:25；下午 13:00-14:55
    hm = mf["kline_time"].dt.strftime("%H:%M")
    mf2 = mf.assign(_hm=hm)
    g2 = mf2.groupby(["date", "code"], sort=True)
    open30 = g2.apply(lambda x: float(x.loc[x["_hm"].between("09:30", "09:55"), "bar_ret"].sum()), include_groups=False)
    close30 = g2.apply(lambda x: float(x.loc[x["_hm"].between("14:30", "14:55"), "bar_ret"].sum()), include_groups=False)
    am = g2.apply(lambda x: float(x.loc[x["_hm"].between("09:30", "11:25"), "bar_ret"].sum()), include_groups=False)
    pm = g2.apply(lambda x: float(x.loc[x["_hm"].between("13:00", "14:55"), "bar_ret"].sum()), include_groups=False)
    out["open30_ret"] = _panel_from_series(open30, daily_dates)
    out["close30_ret"] = _panel_from_series(close30, daily_dates)
    out["am_pm_diff"] = _panel_from_series(am - pm, daily_dates)
    out["first_bar_ret"] = _panel_from_series(
        mf2[mf2["_hm"] == "09:30"].set_index(["date", "code"])["bar_ret"], daily_dates)

    # VWAP 偏离：VWAP = Σ(typical*vol)/Σvol
    vwap = g.apply(lambda x: float((x["typical"] * x["volume"]).sum() / x["volume"].sum()), include_groups=False)
    close_day = g["close"].last()
    out["intraday_vwap_dev"] = _panel_from_series(close_day / vwap - 1.0, daily_dates)

    # 量比：当日成交量 / 过去 20 日均量（按 code 滚动）
    vol_day = g["volume"].sum()
    vol_panel = _panel_from_series(vol_day, daily_dates)
    roll_mean = vol_panel.rolling(_LOOKBACK, min_periods=_ROLL_MIN).mean()
    out["intraday_vol_ratio"] = vol_panel / roll_mean

    # 日内 Amihud：mean(|bar_ret| / amount)（amount 单位元，值很小，标准化处理）
    amihud = g.apply(
        lambda x: float((x["bar_ret"].abs() / x["amount"]).mean()), include_groups=False)
    out["intraday_amihud"] = _panel_from_series(amihud, daily_dates)

    # 波动率 Z：当日 RV / 过去 20 日平均 RV
    rv_panel = out["intraday_rv"]
    rv_roll = rv_panel.rolling(_LOOKBACK, min_periods=_ROLL_MIN).mean()
    out["intraday_rv_z"] = rv_panel / rv_roll

    # 对齐到统一日期/代码（与日线一致）
    aligned: dict[str, pd.DataFrame] = {}
    for name, p in out.items():
        p = p.reindex(index=daily_dates, columns=sorted(p.columns))
        aligned[name] = p
    return aligned


def build_returns(daily: pd.DataFrame) -> pd.DataFrame:
    """未来一期收益面板：次日收益（与挖掘/库 IC 口径一致）。"""
    d = daily.reset_index()
    close_w = d.pivot(index="date", columns="code", values="close").sort_index()
    return close_w.pct_change().shift(-1)


def main():
    parser = argparse.ArgumentParser(description="日内特征 → 日频因子构建与入库")
    parser.add_argument("--mock", action="store_true", help="mock 验证（2023-2024）")
    parser.add_argument("--offline", action="store_true", help="只读缓存，不连 SDK")
    parser.add_argument("--index", default="000300.SH")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="因子库数据集名（默认自动推导）")
    parser.add_argument("--factors", default=None, help="只构建指定特征，逗号分隔")
    parser.add_argument("--no-save", action="store_true", help="只计算不入库")
    args = parser.parse_args()

    if args.mock:
        import tempfile
        from tests.conftest import MockDataSource
        ds = MockDataSource()
        begin, end = args.begin or 20230103, args.end or 20241231
        cache = DataCache(ds, cache_root=tempfile.mkdtemp(prefix="mock_cache_"))
        dataset = args.dataset or "mock"
    elif args.offline:
        ds = OfflineDataSource()
        begin, end = args.begin or 20250101, args.end or 20251231
        cache = DataCache(ds)
        dataset = args.dataset or "hs300_2025"
    else:
        ds = create_datasource()
        begin, end = args.begin or 20250101, args.end or 20251231
        cache = DataCache(ds)
        dataset = args.dataset or "hs300_2025"
    uni = Universe(cache)

    # 特征子集
    if args.factors:
        names = [f.strip() for f in args.factors.split(",") if f.strip()]
        bad = [n for n in names if n not in FACTOR_DEFS]
        if bad:
            raise ValueError(f"未知特征 {bad}，可选 {sorted(FACTOR_DEFS)}")
    else:
        names = list(FACTOR_DEFS)

    codes, daily, mk, status = load_data(cache, uni, args.index, begin, end)
    if daily.empty or mk.empty:
        log.error("数据为空，无法构建")
        sys.exit(1)

    log.info("构建日内特征（%d 个）...", len(names))
    mf = _minute_frame(mk, status)
    features = build_features(mf, daily, status)
    returns_panel = build_returns(daily)

    if args.no_save:
        for name in names:
            p = features[name]
            log.info("  %-24s 面板 %d 日 × %d 股  | 非空率 %.0f%%",
                     name, p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
        log.info("未入库（--no-save）。完成")
        return

    lib = FactorLibrary(dataset=dataset)
    log.info("入库到数据集: %s", dataset)
    reg_rows = []
    for name in names:
        panel = features[name]
        std = standardize_zscore(panel)
        row = lib.register(
            name=name,
            panel=std,
            returns_panel=returns_panel,
            kind="raw",
            formula=FACTOR_DEFS[name],
            source=f"intraday:build_intraday_factors:{PERIOD}min",
        )
        reg_rows.append(row)
        log.info("  已入库 %s（IC=%.4f, t_nw=%.2f, best_sharpe=%.3f@%s）",
                 name, row["ic_mean"], row["t_stat_nw"], row["best_sharpe"], row["best_config"])

    # 实验记录
    try:
        from research.experiments import record_experiment
        record_experiment(
            kind="intraday_factors",
            command=" ".join(sys.argv),
            params={"index": args.index, "begin": begin, "end": end,
                    "dataset": dataset, "factors": names, "n_codes": len(codes)},
            data_fingerprint=cache.get_fingerprint(),
            result_path=str(lib.root),
            metrics={"n_factors": len(reg_rows),
                     "best_ir": max((r["ic_ir"] for r in reg_rows), default=0.0)},
            note="日内特征 → 日频因子入库",
        )
    except Exception as e:
        log.warning("实验记录写入失败: %s", e)

    log.info("完成。数据集 %s 现有 %d 个因子（新增 %d 个日内因子）",
             dataset, len(lib.list_all()), len(reg_rows))


if __name__ == "__main__":
    main()
