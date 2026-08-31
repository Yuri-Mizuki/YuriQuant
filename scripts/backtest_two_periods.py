# -*- coding: utf-8 -*-
"""sig_composite_pca 两时段样本外回测：2025 全年 vs 2026 上半年。

2026H1 是严格样本外：20 个成分因子在 2026 数据上**重建**（日内/技术面/基本面
各按原公式），再用与 2025 相同的 PCA 合成流程出复合面板，月频 Top50 纯多回测。

技术面/日内 warmup：计算指标时从 begin 之前 400 天起算（保证 EMA/SAR/RSV 收敛），
只截取研究区间内的因子值——这是指标类因子的标准做法，不算偷看未来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from backtest import VectorBacktest
from data.cache import DataCache
from data.offline import OfflineDataSource
from strategy.examples import TopKLongOnly

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import setup_logging  # noqa: E402

log = setup_logging("backtest_two_periods")

from config import Config  # noqa: E402  缓存根单一真源（原硬编码 e:/data/parquet）

CACHE_ROOT = Path(str(Config.cache()["root"]))
COMPONENTS = [
    "close30_ret", "intraday_vwap_dev", "intraday_ret", "overnight_intraday_diff",
    "intraday_rsj", "sar_dev", "intraday_rv", "intraday_range", "intraday_vol_ratio",
    "rsi_12", "kdj_j", "intraday_rv_z", "macd_hist",
    "oppro_growth_sq_yoy", "np_ded_growth_ttm_yoy", "np_growth_sq_yoy",
    "np_growth_yoy", "gross_margin", "float_ratio", "netcf_yield",
]
WARMUP_DAYS = 400  # 技术面/日内指标 warmup 长度


def _load_daily(cache, codes, begin, end):
    cal = cache.get_calendar(begin, end)
    daily = cache.get_daily_kline(codes, begin, end)
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    o = d.pivot(index="date", columns="code", values="open").sort_index()
    h = d.pivot(index="date", columns="code", values="high").sort_index()
    l = d.pivot(index="date", columns="code", values="low").sort_index()
    c = d.pivot(index="date", columns="code", values="close").sort_index()
    v = d.pivot(index="date", columns="code", values="volume").sort_index()
    return cal, daily, o, h, l, c, v


def build_panels(begin: int, end: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    from data.universe import Universe
    from data.cache_helpers import _pit_universe_codes, _apply_membership_mask
    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    cal = cache.get_calendar(begin, end)
    if not cal:
        raise RuntimeError(f"日历为空 {begin}-{end}")
    # PIT 口径（2026-08-13 统一）：历史在册并集池
    codes = _pit_universe_codes(uni, "000300.SH", begin, end)

    # 技术面 warmup：从 begin-400 天起拉日线算指标，截取研究区间
    warm_begin = int((pd.Timestamp(str(begin)) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y%m%d"))
    _, dailyw, _, _, _, _, _ = _load_daily(cache, codes, warm_begin, end)
    # warmup 区间也应用 PIT mask（非在册期间行情剔除）
    dailyw = _apply_membership_mask(dailyw, uni, "000300.SH")
    dw = dailyw.reset_index()
    dw["date"] = dw["date"].dt.normalize()
    ow = dw.pivot(index="date", columns="code", values="open").sort_index()
    hw = dw.pivot(index="date", columns="code", values="high").sort_index()
    lw = dw.pivot(index="date", columns="code", values="low").sort_index()
    cw = dw.pivot(index="date", columns="code", values="close").sort_index()
    vw = dw.pivot(index="date", columns="code", values="volume").sort_index()
    bf = pd.read_parquet(CACHE_ROOT / "backward_factor.parquet")
    bf = bf.reindex(index=cw.index).reindex(columns=cw.columns).ffill()
    oa, ha, la, ca = ow * bf, hw * bf, lw * bf, cw * bf

    panels: dict[str, pd.DataFrame] = {}
    mask_dates = pd.DatetimeIndex(pd.to_datetime([str(x) for x in cal], format="%Y%m%d"))

    # ---- 技术面（warmup 后截取研究区间）----
    try:
        from factor.technical import calc_indicators
        for code in codes:
            if code not in ca.columns or ca[code].dropna().empty:
                continue
            res = calc_indicators(ca[code], ha[code], la[code], oa[code], vw[code])
            for k, s in res.items():
                if k in COMPONENTS:
                    panels.setdefault(k, pd.DataFrame(index=cw.index, columns=cw.columns))
                    panels[k][code] = s
        for k in list(panels.keys()):
            panels[k] = panels[k].reindex(index=mask_dates)
        log.info("技术面重建: %s", [k for k in COMPONENTS if k in panels])
    except Exception as e:
        log.warning("技术面重建失败: %s", e)

    # ---- 日内（分钟线 warmup 同样用更长历史；2026 研究区间内分钟线已拉）----
    try:
        from scripts.build_intraday_factors import build_features, _minute_frame
        mk = cache.get_minute_kline(codes, warm_begin, end, period=5)
        if not mk.empty:
            status = cache.get_history_stock_status(codes, warm_begin, end)
            mf = _minute_frame(mk, status)
            dailyw2 = dailyw.copy()
            intra = build_features(mf, dailyw2, status)
            for k, pnl in intra.items():
                if k in COMPONENTS:
                    panels[k] = pnl.reindex(index=mask_dates)
            log.info("日内重建: %s", [k for k in COMPONENTS if k in panels])
    except Exception as e:
        log.warning("日内重建失败: %s", e)

    # ---- 基本面（PIT 展开天然覆盖历史，无需 warmup）----
    try:
        from data.cache_helpers import load_financial_tables
        from scripts.build_fundamental_factors import build_factor_panels
        fin = load_financial_tables(cache, codes)
        fund = build_factor_panels(dailyw, cal,
                                   fin["income"], fin["balance_sheet"], fin["cash_flow"],
                                   fin["equity_structure"], fin["dividend"],
                                   fin["share_holder"], fin["holder_num"])
        for k in COMPONENTS:
            if k in fund:
                panels[k] = fund[k].reindex(index=mask_dates)
        log.info("基本面重建: %s", [k for k in COMPONENTS if k in panels])
    except Exception as e:
        log.warning("基本面重建失败: %s", e)

    # returns 面板（研究区间）
    _, _, _, _, _, c, _ = _load_daily(cache, codes, begin, end)
    returns_panel = c.pct_change().shift(-1)
    return panels, returns_panel


def run_period(begin: int, end: int, label: str):
    panels, returns_panel = build_panels(begin, end)
    missing = [k for k in COMPONENTS if k not in panels or panels[k].dropna().empty]
    if missing:
        log.warning("%s 缺因子: %s", label, missing)
    from factor.preprocessing import standardize_zscore
    from factor.synthesis import CompositeInput, synthesize_pca
    comps = [CompositeInput(name=k, panel=standardize_zscore(panels[k]), ic=0.0, ir=0.0)
             for k in COMPONENTS if k in panels and not panels[k].dropna().empty]
    if not comps:
        raise RuntimeError(f"{label} 无可用因子")
    rp = returns_panel.reindex(index=comps[0].panel.index, columns=comps[0].panel.columns)
    composite = synthesize_pca(comps, n_components=min(3, len(comps)), returns_panel=rp)
    composite = composite.reindex(index=returns_panel.index, columns=returns_panel.columns)

    bt = VectorBacktest(TopKLongOnly(k=50), rebalance_freq="M")
    result = bt.run(composite, returns_panel)
    m = result.metrics()
    log.info("%s: sharpe=%.2f 年化=%.1f%% 回撤=%.1f%% 换手=%.2f",
             label, m.get("sharpe") or 0, (m.get("annual_return") or 0) * 100,
             (m.get("max_drawdown") or 0) * 100, m.get("avg_turnover") or 0)
    return label, m, result


def main():
    out_dir = Path("reports") / "two_periods"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_period(20250101, 20251231, "2025 全年(样本内)"),
        run_period(20260101, 20260630, "2026 上半年(样本外)"),
    ]
    rows = []
    for label, m, res in results:
        rows.append({
            "期间": label,
            "年化收益": m.get("annual_return"), "夏普": m.get("sharpe"),
            "Sortino": m.get("sortino"), "最大回撤": m.get("max_drawdown"),
            "Calmar": m.get("calmar"), "胜率": m.get("win_rate"),
            "月均换手": m.get("avg_turnover"),
        })
        eq = res.equity_curve
        eq.to_csv(out_dir / f"equity_{label.split()[0].replace(' ','')}.csv")
    df = pd.DataFrame(rows)
    print("\n===== sig_composite_pca 两时段对比（月频 Top50 纯多）=====")
    with pd.option_context("display.width", 160, "display.float_format", lambda v: f"{v:.4f}"):
        print(df.to_string(index=False))
    df.to_csv(out_dir / "summary.csv", index=False)
    log.info("结果: %s", out_dir)


if __name__ == "__main__":
    main()
