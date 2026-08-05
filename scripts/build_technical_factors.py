"""
技术面迭代/累积类指标 → 日频因子构建与入库
============================================

从星耀数智 ad-technical-analysis 技能的 56 个指标中挑选「现有算子空间
无法表达」的 9 个迭代/累积/EMA 类指标，作为独立信号源补入因子库：

    macd_hist   MACD 柱 = 2*(DIF-DEA)，DIF=EMA12-EMA26, DEA=EMA(DIF,9)
    rsi_12      RSI(12) = SMA(MAX(C-LC,0),12,1)/SMA(ABS(C-LC),12,1)*100
    kdj_j       KDJ 的 J = 3K-2D（K/D 用通达信 SMA 递归）
    trix_12     TRIX = (MTR-REF(MTR,1))/REF(MTR,1)*100, MTR=EMA³(C,12)
    obv_dev     OBV 偏离 30 日均线 = OBV/MA(OBV,30)-1（累积量取乖离）
    wad_dev     WAD 偏离 30 日均线（WAD 为多空累积力度线）
    asi_26      ASI = 26 日 SI 滚动和（振动升降指标）
    cho         CHO = (MA(MID,10)-MA(MID,20))/100, MID=ΣV*(2C-H-L)/(H+L)
    sar_dev     (close-SAR)/close，SAR 抛物线转向（Wilder 迭代）

口径说明
--------
- 价格一律用「后复权价」：daily 缓存为原始价，adj = raw × backward_factor。
  避免技能脚本"前复权用最新因子归一化"导致的样本末端漂移（未来信息）。
- 除权除息日由后复权价格自然衔接，无需剔除样本。
- 因子在 t 日收盘后可得，预测 t+1 收益（与库内 IC 口径一致）。
- 每因子面板 zscore 截面标准化后入库。

用法
----
    python -m scripts.build_technical_factors --offline            # 读缓存（推荐）
    python -m scripts.build_technical_factors --mock               # mock 验证
    python -m scripts.build_technical_factors --offline --no-save  # 只算不入库
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
from data.datasource import create_datasource
from data.universe import Universe
from factor.preprocessing import standardize_zscore
from research.factor_library import FactorLibrary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_technical_factors")

FACTOR_DEFS: dict[str, str] = {
    "macd_hist": "MACD柱 = 2*(DIF-DEA)，DIF=EMA12-EMA26, DEA=EMA(DIF,9)",
    "rsi_12": "RSI(12) = SMA(MAX(C-LC,0),12,1)/SMA(ABS(C-LC),12,1)*100",
    "kdj_j": "KDJ-J = 3K-2D（RSV→SMA(3,1) 递归）",
    "trix_12": "TRIX = 三重EMA(C,12)变化率",
    "obv_dev": "OBV/MA(OBV,30)-1（能量潮乖离）",
    "wad_dev": "WAD/MA(WAD,30)-1（威廉多空力度线乖离）",
    "asi_26": "ASI = 26日SI滚动和（振动升降指标）",
    "cho": "CHO = (MA(MID,10)-MA(MID,20))/100，MID=ΣV*(2C-H-L)/(H+L)",
    "sar_dev": "(close-SAR)/close，Wilder抛物线SAR",
}


class _OfflineDataSource:
    """离线模式数据源桩（同 intraday_analysis）。"""

    def _raise(self, *a, **k):
        raise RuntimeError("offline 模式不连接数据源：请先运行 scripts.update_data 拉取缓存")

    get_calendar = get_code_list = get_index_constituent = _raise
    get_daily_kline = get_minute_kline = get_adj_factor = get_backward_factor = _raise
    get_code_info = get_history_stock_status = get_industry_classification = _raise
    get_equity_structure = get_balance_sheet = get_cash_flow = get_income = _raise


# ===========================================================================
# 面板级基础算子（作用于 date×code DataFrame，时间轴在 index）
# ===========================================================================


def _ema(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """EMA：Y = (X*2 + Y'*(N-1))/(N+1)，与 ts_ema 一致。"""
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _sma_tdx(x: pd.DataFrame, n: int, m: int = 1) -> pd.DataFrame:
    """通达信 SMA(X,N,M) 递归：Y = (X*M + Y'*(N-M))/N。

    等价于 ewm(alpha=M/N, adjust=False)。KDJ/RSI 的平滑依赖此式。
    """
    alpha = m / n
    return x.ewm(alpha=alpha, adjust=False, min_periods=n).mean()


def _sma_ma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """简单移动平均（右闭窗口，前 n-1 行为 NaN）。"""
    return x.rolling(n, min_periods=n).mean()


def _sma_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).sum()


# ===========================================================================
# 逐股时序计算（输入 Series，返回 Series）
# ===========================================================================


def _calc_sar(close, high, low, n=4, step=0.02, max_af=0.2) -> pd.Series:
    """Wilder 抛物线 SAR（逐股迭代）。

    初始方向：前 n 日收盘净涨跌；多头 SAR 起点=前 n 日最低价，空头=最高价。
    AF 从 step 起，每创新极值 +step，上限 max_af；翻转时 AF 重置。
    返回值与 close 同索引的 SAR 序列（前 n 日 NaN）。
    """
    c, h, l = close.to_numpy(float), high.to_numpy(float), low.to_numpy(float)
    sar = np.full(len(c), np.nan)
    if len(c) <= n or not np.isfinite(h[:n]).any() or not np.isfinite(l[:n]).any():
        return pd.Series(sar, index=close.index)
    # 初始方向：前 n 日净涨跌
    bull = c[n - 1] >= c[0]
    if bull:
        ep = float(np.nanmax(h[:n]))
        sar_val = float(np.nanmin(l[:n]))
    else:
        ep = float(np.nanmin(l[:n]))
        sar_val = float(np.nanmax(h[:n]))
    af = step
    for i in range(n, len(c)):
        sar_val = sar_val + af * (ep - sar_val)
        if bull:
            # SAR 不高于前两日最低价（避免突进）
            lb = min(l[i - 1], l[i - 2]) if i >= 2 else l[i - 1]
            sar_val = min(sar_val, lb)
            if h[i] > ep:
                ep = h[i]
                af = min(af + step, max_af)
            if l[i] < sar_val:
                bull = False
                sar_val = ep
                ep = l[i]
                af = step
        else:
            hb = max(h[i - 1], h[i - 2]) if i >= 2 else h[i - 1]
            sar_val = max(sar_val, hb)
            if l[i] < ep:
                ep = l[i]
                af = min(af + step, max_af)
            if h[i] > sar_val:
                bull = True
                sar_val = ep
                ep = h[i]
                af = step
        sar[i] = sar_val
    return pd.Series(sar, index=close.index)


def _calc_indicators(close, high, low, open_, volume) -> dict[str, pd.Series]:
    """单只股票的 9 个指标序列（全部只用 t 及以前数据）。"""
    out: dict[str, pd.Series] = {}
    ref_c = close.shift(1)

    # MACD 柱
    dif = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(
        span=26, adjust=False, min_periods=26).mean()
    dea = dif.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = 2 * (dif - dea)

    # RSI(12)
    diff = close - ref_c
    pos = diff.clip(lower=0.0)
    up = pos.ewm(alpha=1 / 12, adjust=False, min_periods=12).mean()
    dn = diff.abs().ewm(alpha=1 / 12, adjust=False, min_periods=12).mean()
    out["rsi_12"] = (up / dn.replace(0.0, np.nan) * 100).clip(0, 100)

    # KDJ-J
    llv = low.rolling(9, min_periods=9).min()
    hhv = high.rolling(9, min_periods=9).max()
    rsv = (close - llv) / (hhv - llv).replace(0.0, np.nan) * 100
    k = rsv.ewm(alpha=1 / 3, adjust=False, min_periods=9).mean()
    d = k.ewm(alpha=1 / 3, adjust=False, min_periods=9).mean()
    out["kdj_j"] = 3 * k - 2 * d

    # TRIX(12,9)
    mtr = close.ewm(span=12, adjust=False, min_periods=12).mean()
    mtr = mtr.ewm(span=12, adjust=False, min_periods=12).mean()
    mtr = mtr.ewm(span=12, adjust=False, min_periods=12).mean()
    out["trix_12"] = (mtr - mtr.shift(1)) / mtr.shift(1).replace(0.0, np.nan) * 100

    # OBV 乖离
    direction = np.sign(close - ref_c).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv[obv.index[0]] = volume.iloc[0] if len(volume) else np.nan
    obv_ma = obv.rolling(30, min_periods=10).mean()
    out["obv_dev"] = obv / obv_ma.replace(0.0, np.nan) - 1.0

    # WAD 乖离
    mida = close - pd.concat([low, ref_c], axis=1).min(axis=1)
    midb = (close - pd.concat([ref_c, high], axis=1).max(axis=1)).where(close < ref_c, 0.0)
    wad_unit = mida.where(close > ref_c, midb)
    wad = wad_unit.cumsum()
    wad_ma = wad.rolling(30, min_periods=10).mean()
    out["wad_dev"] = wad / wad_ma.replace(0.0, np.nan) - 1.0

    # ASI(26,10)
    aa = (high - ref_c).abs()
    bb = (low - ref_c).abs()
    cc = (high - low.shift(1)).abs()
    dd = (ref_c - open_.shift(1)).abs()
    r_a = aa + bb / 2 + dd / 4
    r_b = bb + aa / 2 + dd / 4
    r_c = cc + dd / 4
    r = r_a.where((aa > bb) & (aa > cc), r_b.where((bb > cc) & (bb > aa), r_c))
    x = close - ref_c + (close - open_) / 2 + ref_c - open_.shift(1)
    si = 16 * x / r.replace(0.0, np.nan) * pd.concat([aa, bb], axis=1).max(axis=1)
    out["asi_26"] = si.rolling(26, min_periods=10).sum()

    # CHO
    mid = (volume * (2 * close - high - low) / (high + low).replace(0.0, np.nan)).cumsum()
    out["cho"] = (mid.rolling(10, min_periods=10).mean()
                  - mid.rolling(20, min_periods=10).mean()) / 100

    # SAR 偏离
    sar = _calc_sar(close, high, low)
    out["sar_dev"] = (close - sar) / close.replace(0.0, np.nan)

    return out


# ===========================================================================
# 主流程
# ===========================================================================


def load_data(cache, uni, index_code, begin, end):
    cal = cache.get_calendar(begin, end)
    if not cal:
        raise RuntimeError(f"交易日历为空（{begin}-{end}），请先更新数据")
    codes = uni.get_constituent(index_code, end or cal[-1])
    log.info("股票池: %s @ %d, %d 只", index_code, end or cal[-1], len(codes))
    daily = cache.get_daily_kline(codes, begin, end or cal[-1])
    # 复权因子 wide 表会回调数据源（_refresh_wide_table），offline 时直接读 parquet
    if isinstance(cache._ds, _OfflineDataSource):
        p = Path(cache.root) / "backward_factor.parquet"
        if p.exists():
            bf = pd.read_parquet(p)
            bf = bf[[c for c in codes if c in bf.columns]]
        else:
            bf = pd.DataFrame()
    else:
        bf = cache.get_backward_factor(codes)
    log.info("日线 %d 行 / 复权因子 %d 列", len(daily), bf.shape[1] if not bf.empty else 0)
    return codes, cal, daily, bf


def build_factor_panels(daily, bf) -> dict[str, pd.DataFrame]:
    """返回 {因子名: 原始面板(date×code)}，价格均为后复权。"""
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    codes = sorted(d["code"].unique())

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    o, h, l, c, v = _panel("open"), _panel("high"), _panel("low"), _panel("close"), _panel("volume")

    # 后复权：adj = raw × backward_factor（bf 可能缺部分股票/日期 → NaN 保持）
    if not bf.empty:
        dates = c.index
        for pnl, name in ((o, "open"), (h, "high"), (l, "low"), (c, "close")):
            f = bf.reindex(index=dates).reindex(columns=pnl.columns)
            pnl[:] = pnl.values * f.values
    else:
        log.warning("无复权因子，使用原始价（除权日指标会有跳变）")

    panels: dict[str, pd.DataFrame] = {name: pd.DataFrame(index=c.index, columns=c.columns)
                                       for name in FACTOR_DEFS}
    for code in codes:
        res = _calc_indicators(c[code], h[code], l[code], o[code], v[code])
        for name, s in res.items():
            panels[name][code] = s
    return panels


def main():
    parser = argparse.ArgumentParser(description="技术面迭代/累积类指标构建入库")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--index", default="000300.SH")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    if args.mock:
        import tempfile
        from tests.conftest import MockDataSource
        ds = MockDataSource()
        begin, end = args.begin or 20230103, args.end or 20241231
        cache = DataCache(ds, cache_root=tempfile.mkdtemp(prefix="mock_cache_"))
        dataset = args.dataset or "mock"
    elif args.offline:
        ds = _OfflineDataSource()
        begin, end = args.begin or 20250101, args.end or 20251231
        cache = DataCache(ds)
        dataset = args.dataset or "hs300_2025"
    else:
        ds = create_datasource()
        begin, end = args.begin or 20250101, args.end or 20251231
        cache = DataCache(ds)
        dataset = args.dataset or "hs300_2025"
    uni = Universe(cache)

    codes, cal, daily, bf = load_data(cache, uni, args.index, begin, end)
    if daily.empty:
        log.error("数据为空")
        sys.exit(1)

    log.info("计算技术面指标（%d 个）...", len(FACTOR_DEFS))
    panels = build_factor_panels(daily, bf)

    if args.no_save:
        for name in FACTOR_DEFS:
            p = panels.get(name)
            if p is None:
                log.info("  %-18s 缺失", name)
                continue
            log.info("  %-18s 面板 %d 日 × %d 股 | 非空率 %.0f%%",
                     name, p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
        log.info("未入库（--no-save）。完成")
        return

    lib = FactorLibrary(dataset=dataset)
    d = daily.reset_index()
    close_w = d.pivot(index="date", columns="code", values="close").sort_index()
    returns_panel = close_w.pct_change().shift(-1)

    log.info("入库到数据集: %s", dataset)
    for name in FACTOR_DEFS:
        p = panels.get(name)
        if p is None or p.notna().sum().sum() == 0:
            log.warning("跳过 %s（无有效数据）", name)
            continue
        std = standardize_zscore(p)
        row = lib.register(
            name=name,
            panel=std,
            returns_panel=returns_panel,
            kind="raw",
            formula=FACTOR_DEFS[name],
            source=f"technical:build_technical_factors:{begin}-{end}",
        )
        log.info("  已入库 %s（IC=%.4f, t_nw=%.2f, best_sharpe=%.3f@%s）",
                 name, row["ic_mean"], row["t_stat_nw"], row["best_sharpe"], row["best_config"])

    try:
        from research.experiments import record_experiment
        record_experiment(
            kind="technical_factors",
            command=" ".join(sys.argv),
            params={"index": args.index, "begin": begin, "end": end, "dataset": dataset},
            data_fingerprint=cache.get_fingerprint(),
            result_path=str(lib.root),
            metrics={"n_factors": len(FACTOR_DEFS)},
            note="技术面迭代/累积类指标入库",
        )
    except Exception as e:
        log.warning("实验记录写入失败: %s", e)

    log.info("完成。数据集 %s 现有 %d 个因子", dataset, len(lib.list_all()))


if __name__ == "__main__":
    main()
