"""「中国因子十强」复现与统一评估（JKP 口径榜单，2026-11-02）
================================================================

来源：夜间笔记《中国因子十强》（JKP 公开因子、美元月度市值加权多空、153 候选
过 120）。本脚本把这 10 个因子（+1 个变体）在我们自己的数据面上实现并入库到
**独立数据集**（默认 ``top10_sharpe_2026_11``），用同一套 canonical 回测
（ls_M / lo_M / ls_W）出对照表，不与生产库 ``all_a_2018_2026`` 混库。

十因子与落地口径
----------------
| # | 榜单名 | 本脚本名 | 口径 |
|---|--------|----------|------|
| 1 | Q1 ROE 改善 | ``t10_roe_improve`` | 单季净利/股东权益 的同比差（同季，PIT 公告日对齐） |
| 2 | Q2 ROA 改善 | ``t10_roa_improve`` | 单季净利/总资产 的同比差 |
| 3 | 低净经营资产扩张 | ``t10_low_noa_growth`` | -(NOA/TA 同比变化)，NOA=(TA-现金)-(负债-短借-长借-应付债券) |
| 4 | 低成交金额 | ``t10_low_amount`` | -ln(近 120 日成交额均值) |
| 5 | 残差动量 | ``t10_resid_mom`` | 36 月三因子回归 α 的 12 月 sum/std（复用 ``pap_mom_residual``） |
| 6 | 短期反转 | ``t10_rev_1m`` | -(近 21 日收益) |
| 7 | 小市值 | ``t10_smallcap`` | -ln(PIT 总股本 × 收盘) |
| 8 | 低经营营运资本增长 | ``t10_low_owc_growth`` | -ΔOWC/TA，OWC=(流动资产-现金)-(流动负债-短期借款) |
| 9 | 历史非当月反转 | ``t10_offmonth_rev`` | -(过去第 2–5 年剔除同月后的 44 个月月收益均值) |
| 10 | 去年同月收益 | ``t10_same_month_ret`` | 去年同一自然月的月度收益（正/动量） |
| 变体 | 低成交金额·短窗 | ``t10_low_amount_20d`` | -ln(近 20 日成交额均值)，对照半年窗 |

方向统一为「值越大预期收益越高」。财务量按公告日 ``ann_date`` 展开为 PIT 面板
（两张表都公告才可见 = 取 max(ann_date)）；月度量按月末定格、次月才可见。

用法
----
    python -m scripts.factors.build_top10_factors --offline --pool all_a \
        --begin 20180101 --end 20260930 --no-save          # 只算不入库
    python -m scripts.factors.build_top10_factors --offline --pool all_a \
        --begin 20180101 --end 20260930                    # 入库 + canonical 回测
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache_helpers import load_daily, load_financial_tables  # noqa: E402
from data.financials import build_pit_panel  # noqa: E402
from factor.paper_factors import (  # noqa: E402
    PaperData, month_end_panel, to_daily, pap_mom_residual,
)
from research.factor_library import FactorLibrary  # noqa: E402
from scripts.builders.common import load_close_adj  # noqa: E402
from scripts.common.cli_common import (  # noqa: E402
    add_build_args, make_data_context, print_no_save, record_experiment_safe,
    returns_from_daily, setup_logging,
)
from scripts.common.fundamental_common import add_single_quarter  # noqa: E402

log = setup_logging("build_top10_factors")

DEFAULT_LIB_DATASET = "top10_sharpe_2026_11"

FACTOR_DEFS: dict[str, str] = {
    "t10_roe_improve": "Q1 ROE改善 = 单季净利/股东权益 的同比差(同季口径, PIT公告日对齐)",
    "t10_roa_improve": "Q2 ROA改善 = 单季净利/总资产 的同比差(同季口径)",
    "t10_low_noa_growth": "低净经营资产扩张 = -Δ[NOA/TA]；NOA=(TA-现金)-(负债-短借-长借-应付债券)",
    "t10_low_amount": "低成交金额 = -ln(近120日成交额均值)，成交额非换手率",
    "t10_resid_mom": "残差动量 = 36月三因子(MKT/SMB/HML)回归α的12月sum/std",
    "t10_rev_1m": "短期反转 = -(近21日收益)，上月弱者",
    "t10_smallcap": "小市值 = -ln(PIT总股本×收盘)",
    "t10_low_owc_growth": "低经营营运资本增长 = -ΔOWC/TA；OWC=(流动资产-现金)-(流动负债-短期借款)",
    "t10_offmonth_rev": "历史非当月反转 = -(过去2-5年剔除同月后44个月月收益均值)",
    "t10_same_month_ret": "去年同月收益 = 去年同一自然月的月度收益(正/动量)",
    "t10_low_amount_20d": "[变体] 低成交金额短窗 = -ln(近20日成交额均值)",
}

FAMILY = {
    "t10_roe_improve": "基本面", "t10_roa_improve": "基本面",
    "t10_low_noa_growth": "基本面", "t10_low_owc_growth": "基本面",
    "t10_low_amount": "流动性", "t10_low_amount_20d": "流动性",
    "t10_resid_mom": "动量", "t10_rev_1m": "反转",
    "t10_offmonth_rev": "反转", "t10_same_month_ret": "动量",
    "t10_smallcap": "其他",
}

FREQ = {
    "t10_roe_improve": "季频", "t10_roa_improve": "季频",
    "t10_low_noa_growth": "季频", "t10_low_owc_growth": "季频",
    "t10_offmonth_rev": "月频", "t10_same_month_ret": "月频",
    "t10_resid_mom": "月频",
}


# ===========================================================================
# 财报长表工具
# ===========================================================================
def _dedupe(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """(code, report_period) 去重取最新公告行，只保留需要的字段。"""
    cols = ["code", "ann_date", "report_period"] + [f for f in fields if f in df.columns]
    out = df[cols].copy()
    out = out.dropna(subset=["report_period", "ann_date"])
    out["report_period"] = pd.to_datetime(out["report_period"])
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
    out = out.dropna(subset=["ann_date"])
    out = out.sort_values(["code", "report_period", "ann_date"])
    return out.drop_duplicates(["code", "report_period"], keep="last")


def _add_lag4(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """给每列加 ``_lag4``（上年同季值），按 (code, year-1, quarter) 精确对齐。"""
    d = df.copy()
    y = d["report_period"].dt.year
    q = d["report_period"].dt.quarter
    d["_cur_key"] = d["code"].astype(str) + "_" + y.astype(str) + "_" + q.astype(str)
    d["_lag_key"] = d["code"].astype(str) + "_" + (y - 1).astype(str) + "_" + q.astype(str)
    for c in cols:
        m = dict(zip(d["_cur_key"], d[c]))
        d[f"{c}_lag4"] = d["_lag_key"].map(lambda k, m=m: m.get(k, np.nan))
    return d


def _pit(df: pd.DataFrame, cal, field: str, grid: pd.DataFrame) -> pd.DataFrame:
    """长表字段 → 日频 PIT 面板，并对齐到 close 网格。"""
    p = build_pit_panel(df, cal, field)
    return p.reindex(index=grid.index, columns=grid.columns)


def build_fin_long(inc: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    """财报长表 → 带四因子列的 (code, report_period) 表；``ann_date`` 已取两表较晚者。"""
    inc_sq = add_single_quarter(inc, "NET_PRO_INCL_MIN_INT_INC", "NP_SQ")
    inc_d = _dedupe(inc_sq, ["NP_SQ"])
    bal_fields = ["TOTAL_ASSETS", "TOTAL_LIAB", "CURRENCY_CAP",
                  "TOTAL_CUR_ASSETS", "TOTAL_CUR_LIAB", "ST_BORROWING",
                  "LT_LOAN", "BONDS_PAYABLE",
                  "TOT_SHARE_EQUITY_EXCL_MIN_INT", "TOT_SHARE"]
    bal_d = _dedupe(bal, bal_fields)

    m = inc_d.merge(bal_d, on=["code", "report_period"], how="inner",
                    suffixes=("", "_bal"))
    # 两张表都公告才可见：取较晚的公告日（保守，无前视）
    m["ann_date"] = m[["ann_date", "ann_date_bal"]].max(axis=1)

    ta = m["TOTAL_ASSETS"].replace(0.0, np.nan)
    eq = m["TOT_SHARE_EQUITY_EXCL_MIN_INT"].replace(0.0, np.nan)
    cash = m["CURRENCY_CAP"].fillna(0.0)
    debt = (m["ST_BORROWING"].fillna(0.0) + m["LT_LOAN"].fillna(0.0)
            + m["BONDS_PAYABLE"].fillna(0.0))

    m["_roe_sq"] = m["NP_SQ"] / eq
    m["_roa_sq"] = m["NP_SQ"] / ta
    noa = (m["TOTAL_ASSETS"] - cash) - (m["TOTAL_LIAB"] - debt)
    m["_noa_ta"] = noa / ta
    owc = (m["TOTAL_CUR_ASSETS"] - cash) - (m["TOTAL_CUR_LIAB"] - m["ST_BORROWING"].fillna(0.0))
    m["_owc_ta"] = owc / ta

    m = _add_lag4(m, ["_roe_sq", "_roa_sq", "_noa_ta", "_owc_ta"])
    m["_roe_improve"] = m["_roe_sq"] - m["_roe_sq_lag4"]
    m["_roa_improve"] = m["_roa_sq"] - m["_roa_sq_lag4"]
    m["_noa_growth"] = m["_noa_ta"] - m["_noa_ta_lag4"]
    m["_owc_growth"] = m["_owc_ta"] - m["_owc_ta_lag4"]
    return m


def build_fin_panels(inc: pd.DataFrame, bal: pd.DataFrame, cal,
                     close_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """财务四因子 + 市值/权益面板（供残差动量使用）。"""
    m = build_fin_long(inc, bal)
    bal_fields = ["TOTAL_ASSETS", "TOTAL_LIAB", "CURRENCY_CAP",
                  "TOTAL_CUR_ASSETS", "TOTAL_CUR_LIAB", "ST_BORROWING",
                  "LT_LOAN", "BONDS_PAYABLE",
                  "TOT_SHARE_EQUITY_EXCL_MIN_INT", "TOT_SHARE"]
    bal_d = _dedupe(bal, bal_fields)

    grid = close_raw
    out: dict[str, pd.DataFrame] = {}
    out["t10_roe_improve"] = _pit(m, cal, "_roe_improve", grid)
    out["t10_roa_improve"] = _pit(m, cal, "_roa_improve", grid)
    # 方向：低扩张为正
    out["t10_low_noa_growth"] = -_pit(m, cal, "_noa_growth", grid)
    out["t10_low_owc_growth"] = -_pit(m, cal, "_owc_growth", grid)

    # 附属面板（残差动量 / 小市值需要）
    out["_equity"] = _pit(bal_d, cal, "TOT_SHARE_EQUITY_EXCL_MIN_INT", grid)
    out["_tot_share"] = _pit(bal_d, cal, "TOT_SHARE", grid)
    return out


# ===========================================================================
# 量价 / 月度因子
# ===========================================================================
def build_price_panels(cadj: pd.DataFrame, amount: pd.DataFrame,
                       market_cap: pd.DataFrame,
                       equity: pd.DataFrame, cal,
                       resid_mom: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    grid = cadj
    out: dict[str, pd.DataFrame] = {}

    out["t10_low_amount"] = -np.log(amount.rolling(120, min_periods=60).mean() + 1.0)
    out["t10_low_amount_20d"] = -np.log(amount.rolling(20, min_periods=10).mean() + 1.0)
    out["t10_rev_1m"] = -(cadj / cadj.shift(21) - 1.0)
    out["t10_smallcap"] = -np.log(market_cap.clip(lower=1.0))

    # ---- 月度因子 ----
    m_close = month_end_panel(cadj, how="last")
    m_ret = m_close.pct_change(fill_method=None)

    # 历史非当月反转：窗口 [t-60, t-13] 共 48 个月，剔除同月 4 期 ⇒ 44 期均值
    win_sum = m_ret.rolling(48, min_periods=36).sum().shift(12)
    same_m = (m_ret.shift(12) + m_ret.shift(24) + m_ret.shift(36) + m_ret.shift(48))
    mean44 = (win_sum - same_m) / 44.0
    out["t10_offmonth_rev"] = -to_daily(mean44, grid.index)

    # 去年同月收益
    out["t10_same_month_ret"] = to_daily(m_ret.shift(12), grid.index)

    if resid_mom is not None:
        out["t10_resid_mom"] = resid_mom.reindex(index=grid.index, columns=grid.columns)
    return out


# ===========================================================================
# 主构建
# ===========================================================================
def build_all(daily: pd.DataFrame, cal, inc: pd.DataFrame, bal: pd.DataFrame,
              ) -> dict[str, pd.DataFrame]:
    """在全历史网格上算因子，最后裁剪到本次评估区间。

    月度因子（残差动量 36 月回归、非当月反转 48 月窗）需要评估区间之前的
    历史；因此统一用 ``load_close_adj`` 的 KEEP_FROM 全网格计算，再 reindex
    回 ``daily``（2018+）——避免"短区间测试全 NaN / 长区间前段全 NaN"。
    """
    from config import Config

    d = daily.reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    out_close = d.pivot(index="date", columns="code", values="close").sort_index()

    cadj, close_hist = load_close_adj(with_raw=True)
    cadj = cadj.astype(float)
    root = Path(str(Config.cache()["root"]))
    d_all = pd.read_parquet(root / "daily_all_a.parquet")
    d_all.index = d_all.index.set_levels(d_all.index.levels[0].normalize(), level=0)
    amount = (d_all["amount"].unstack()
              .reindex(index=close_hist.index, columns=close_hist.columns))
    log.info("历史网格 %s ~ %s（%d 日 × %d 股），评估区间 %d 日",
             close_hist.index.min().date(), close_hist.index.max().date(),
             len(close_hist), close_hist.shape[1], len(out_close))

    hist_cal = close_hist.index
    log.info("构建财务面板（单季拆解 + 同季对齐）...")
    fin = build_fin_panels(inc, bal, hist_cal, close_hist)
    equity = fin.pop("_equity")
    tot_share = fin.pop("_tot_share")
    market_cap = tot_share.astype(float) * close_hist.astype(float)

    log.info("构建残差动量（36 月三因子回归）...")
    try:
        paper = PaperData(close=cadj, market_cap=market_cap,
                          pit={"EQUITY": equity})
        resid = pap_mom_residual(paper)
    except Exception as e:  # noqa: BLE001
        log.warning("残差动量失败（跳过）：%s", str(e)[:120])
        resid = None

    log.info("构建量价与月度面板...")
    price = build_price_panels(cadj, amount, market_cap, equity, hist_cal, resid)
    panels = {**fin, **price}

    idx = out_close.index
    cols = out_close.columns
    return {k: panels[k].reindex(index=idx, columns=cols)
            for k in FACTOR_DEFS if k in panels}


def main():
    ap = argparse.ArgumentParser(description="中国因子十强复现 + canonical 回测")
    add_build_args(ap)
    ap.add_argument("--pool", default="all_a", help="股票池（默认 all_a）")
    ap.add_argument("--lib-dataset", default=DEFAULT_LIB_DATASET,
                    help="因子库数据集名（独立目录，不碰生产库）")
    args = ap.parse_args()

    cache, uni, begin, end, _ = make_data_context(args)
    codes, cal, daily = load_daily(cache, uni, args.index, begin, end, pool=args.pool)
    fin = load_financial_tables(cache, codes)
    if daily.empty:
        log.error("数据为空")
        sys.exit(1)
    log.info("日线 %d 行 / 利润表 %d / 资产负债 %d",
             len(daily), len(fin["income"]), len(fin["balance_sheet"]))

    panels = build_all(daily, cal, fin["income"], fin["balance_sheet"])

    # 覆盖率速览
    log.info("因子覆盖率：")
    for n, p in panels.items():
        log.info("  %-22s 覆盖 %.3f  非空格数 %d", n, float(p.notna().mean().mean()),
                 int(p.notna().sum().sum()))

    if args.no_save:
        print_no_save(list(FACTOR_DEFS), panels)
        return

    lib = FactorLibrary(dataset=args.lib_dataset)
    returns_panel = returns_from_daily(daily)
    log.info("入库到独立数据集: %s（root=%s）", args.lib_dataset, lib.root)
    source = f"builders:top10_sharpe:all_a_{begin}-{end}"
    for name, formula in FACTOR_DEFS.items():
        p = panels.get(name)
        if p is None or not p.notna().any().any():
            log.warning("跳过 %s（无有效数据）", name)
            continue
        from factor.preprocessing import standardize_zscore
        std = standardize_zscore(p)
        if not std.notna().any().any():
            log.warning("跳过 %s（标准化后全 NaN）", name)
            continue
        try:
            row = lib.register(
                name=name, panel=std, returns_panel=returns_panel,
                kind="raw", formula=formula, source=source,
                family=FAMILY.get(name, ""),
                frequency=FREQ.get(name, "日频"),
                maturity="experimental",
                note="《中国因子十强》复现（2026-11-02），独立数据集评估用",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("注册失败 %s: %s", name, str(e)[:160])
            continue
        log.info("  入库 %-22s IC=%.4f t_nw=%.2f | ls_M 夏普 %.2f 年化 %.1f%%",
                 name, row.get("ic_mean", np.nan), row.get("t_stat_nw", np.nan),
                 row.get("sharpe_ls_M", np.nan), row.get("annual_return_ls_M", np.nan) * 100
                 if pd.notna(row.get("annual_return_ls_M", np.nan)) else np.nan)

    record_experiment_safe(
        kind="top10_sharpe",
        command=" ".join(sys.argv),
        params={"index": args.index, "begin": begin, "end": end,
                "pool": args.pool, "lib_dataset": args.lib_dataset},
        fingerprint=cache.get_fingerprint() if hasattr(cache, "get_fingerprint") else "",
        result_path=str(lib.root),
        metrics={"n_factors": len(panels)},
        note="中国因子十强复现评估（独立数据集）",
    )
    log.info("完成。数据集 %s 现有 %d 个因子", args.lib_dataset, len(lib.list_all()))


if __name__ == "__main__":
    main()
