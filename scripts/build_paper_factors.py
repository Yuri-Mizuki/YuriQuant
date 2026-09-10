"""论文复现因子构建与入库（awesome-systematic-trading 21 式）
==========================================================

把 awesome-systematic-trading 复现库 static/strategies/ 中可在 A 股数据面
实现的 21 个经典已发表策略翻译为截面因子入库。因子公式与方向变换的
完整说明见 ``factor/paper_factors.py`` 模块头注释。

与 build_fundamental_factors 同一套 PIT 纪律：财务值按公告日（ann_date）
展开到交易日，利润表/现金流 TTM 化，无未来函数。事件类因子依赖
ann_flag（income 表的公告日历）。

因子清单（21 个）
-----------------
量价 : pap_mom_12_1 / pap_mom_consistent / pap_rev_5d / pap_momxvol_6m /
       pap_lowvol_1y / pap_bab_beta / pap_high52w / pap_high52w_ind /
       pap_breakout_atr / pap_smallcap
财务 : pap_momxag / pap_accruals_bs / pap_asset_growth / pap_earn_quality /
       pap_fscore / pap_roa_size_adj / pap_rd_intensity / pap_value_bp /
       pap_mom_residual
事件 : pap_ann_premium / pap_ann_reversal

用法
----
    python -m scripts.build_paper_factors --offline          # 读缓存（推荐）
    python -m scripts.build_paper_factors --mock             # mock 验证
    python -m scripts.build_paper_factors --offline --no-save  # 只算不入库
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache_helpers import load_daily, load_financial_tables  # noqa: E402
from data.financials import build_pit_panel  # noqa: E402
from factor.paper_factors import PaperData, compute_paper_factors  # noqa: E402
from research.factor_library import FactorLibrary  # noqa: E402
from scripts.build_fundamental_factors import _add_ttm_yoy  # noqa: E402
from scripts.cli_common import (  # noqa: E402
    add_build_args,
    make_data_context,
    print_no_save,
    record_experiment_safe,
    register_panels,
    returns_from_daily,
    setup_logging,
)

log = setup_logging("build_paper_factors")

CFO_FIELD = "WS_OPERA_ACT"  # 现金流量表经营净额字段（data.financials 白名接口）

FACTOR_DEFS: dict[str, str] = {
    # ---- 纯量价（10）----
    "pap_mom_12_1": "经典动量UMD = close[t-21]/close[t-252]-1（12-1月，跳最近1月；多头=前20%）",
    "pap_mom_consistent": "一致动量 = min(rank(m1),rank(m2))，m1=t-7→t-1月收益，m2=t-6→t月收益",
    "pap_rev_5d": "短期反转 = -(近5日收益)；原策略做多周跌幅最大10只（多头腿）",
    "pap_momxvol_6m": "动量×波动交互 = rank(6月动量跳1周)×rank(日收益std)；原策略多'赢家∩高波'",
    "pap_lowvol_1y": "低波动 = -(252日周收益std)；原策略即纯多头（波动最低25%）",
    "pap_bab_beta": "BAB多头腿 = -(252日β，市场=全A等权)；原策略多低β空高β+杠杆，多头版无杠杆",
    "pap_high52w": "52周新高 = close/252日最高-1（George-Hwang临近度）",
    "pap_high52w_ind": "行业52周新高 = 行业内市值加权PRILAG映射回成分股；原策略做多最高6行业",
    "pap_breakout_atr": "趋势跟踪信号 = 创历史新高入场、10日ATR吊灯止损离场（时序0/1，非截面）",
    "pap_smallcap": "规模因子 = -ln(市值)；原策略多小空大，A股多头版做多小市值（重复因子对照）",
    # ---- 财务 PIT（9）----
    "pap_momxag": "资产增长×动量 = rank(11-1月动量)×1{AG前10%}；原策略1月不开仓规则不入因子",
    "pap_accruals_bs": "应计异象·资产负债表法(原版) = -[(ΔCA-ΔCash)-(ΔCL-ΔSTD)-Dep]/mean(TA)"
        "，Dep≈EBITDA-EBIT 近似",
    "pap_asset_growth": "资产增长异象 = -(总资产同比)；低增长组未来收益更高",
    "pap_earn_quality": "盈利质量四成分 = rank(-应计CF法)+rank(CFO/净利)+rank(-D/A)+rank(ROE)",
    "pap_fscore": "Piotroski FSCORE 0-9 打分，TTM 同比口径：ROA/CFO/ΔROA/CFO>NI"
        "/降杠杆/流动比率/未增发/毛利率/周转",
    "pap_roa_size_adj": "ROA规模分组内排名 = rank(ROA)-市值半分组内均值；原策略两组内做多前30%",
    "pap_rd_intensity": "研发强度 = Σ[1,.8,.6,.4,.2]×RD_{t-252k}/市值（近5年年报衰减加权）",
    "pap_value_bp": "价值HML腿 = 股东权益/市值（与库内bp同构，论文复现对照，check_dup标冗余）",
    "pap_mom_residual": "残差动量 = 36月Fama-French三因子回归α的12月sum/std（SMB/HML自建A股版）",
    # ---- 事件类（2）----
    "pap_ann_premium": "公告溢价 = rank(48月公告月成交量占比VCR)×1{本月预期公告}，稀疏面板",
    "pap_ann_reversal": "公告期反转 = -(公告前3日收益)×1{次日有公告}，稀疏面板",
}


def build_paper_panels(daily: pd.DataFrame, cal, income: pd.DataFrame,
                       balance: pd.DataFrame, cashflow: pd.DataFrame,
                       cache=None) -> dict[str, pd.DataFrame]:
    """返回 {因子名: 原始面板(date×code)}。"""
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    close = d.pivot(index="date", columns="code", values="close").sort_index()
    high = d.pivot(index="date", columns="code", values="high").sort_index()
    low = d.pivot(index="date", columns="code", values="low").sort_index()
    volume = d.pivot(index="date", columns="code", values="volume").sort_index()

    # ---- 财务长表：TTM 化 + PIT 展开 ----
    inc = income.copy()
    inc = _add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", None)
    inc = _add_ttm_yoy(inc, "LESS_OPERA_COST", "LESS_OPERA_COST_TTM", None)
    inc = _add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", None)
    if "EBIT" in inc.columns:
        inc = _add_ttm_yoy(inc, "EBIT", "EBIT_TTM", None)
    if "EBITDA" in inc.columns:
        inc = _add_ttm_yoy(inc, "EBITDA", "EBITDA_TTM", None)
    cf = cashflow.copy()
    if CFO_FIELD not in cf.columns and "NET_CASH_FLOWS_OPERA_ACT" in cf.columns:
        cf[CFO_FIELD] = cf["NET_CASH_FLOWS_OPERA_ACT"]
    cf = _add_ttm_yoy(cf, CFO_FIELD, "CFO_TTM", None)

    def _pit(report_df: pd.DataFrame, field: str) -> pd.DataFrame:
        if field not in report_df.columns:
            log.warning("字段 %s 缺失，返回空面板", field)
            return pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        pnl = build_pit_panel(report_df, cal, field)
        return pnl.reindex(index=close.index, columns=close.columns)

    pit: dict[str, pd.DataFrame] = {
        "NET_PRO_TTM": _pit(inc, "NET_PRO_TTM"),
        "OPERA_REV_TTM": _pit(inc, "OPERA_REV_TTM"),
        "LESS_OPERA_COST_TTM": _pit(inc, "LESS_OPERA_COST_TTM"),
        "CFO_TTM": _pit(cf, "CFO_TTM"),
        "TOTAL_ASSETS": _pit(balance, "TOTAL_ASSETS"),
        "TOTAL_LIAB": _pit(balance, "TOTAL_LIAB"),
        "TOTAL_CUR_ASSETS": _pit(balance, "TOTAL_CUR_ASSETS"),
        "TOTAL_CUR_LIAB": _pit(balance, "TOTAL_CUR_LIAB"),
        "CURRENCY_CAP": _pit(balance, "CURRENCY_CAP"),
        "ST_BORROWING": _pit(balance, "ST_BORROWING"),
        "LT_LOAN": _pit(balance, "LT_LOAN"),
        "EQUITY": _pit(balance, "TOT_SHARE_EQUITY_EXCL_MIN_INT"),
        "TOT_SHARE": _pit(balance, "TOT_SHARE"),
        "RD_EXP": _pit(inc, "RD_EXP"),
    }
    if "EBIT_TTM" in inc.columns:
        pit["EBIT_TTM"] = _pit(inc, "EBIT_TTM")
    if "EBITDA_TTM" in inc.columns:
        pit["EBITDA_TTM"] = _pit(inc, "EBITDA_TTM")

    # 市值 = PIT 总股本 × 收盘（PIT 安全，与 build_fundamental_factors 同口径）
    market_cap = pit["TOT_SHARE"].astype(float) * close.astype(float)

    # ---- 公告日历：income 表 (code, ann_date) → date×code bool ----
    ann_flag = pd.DataFrame(False, index=close.index, columns=close.columns)
    if not income.empty and "ann_date" in income.columns:
        ann = income[["code", "ann_date"]].dropna().drop_duplicates()
        ann["ann_date"] = pd.to_datetime(ann["ann_date"], errors="coerce").dt.normalize()
        ann = ann.dropna()
        valid = ann["ann_date"].isin(close.index) & ann["code"].isin(close.columns)
        ann = ann[valid]
        if not ann.empty:
            arr = ann_flag.values
            row_ix = close.index.get_indexer(ann["ann_date"])
            col_ix = close.columns.get_indexer(ann["code"])
            arr[row_ix, col_ix] = True
            ann_flag = pd.DataFrame(arr, index=close.index, columns=close.columns)
    log.info("公告日历：%d 个 (日, 股) 事件", int(ann_flag.values.sum()))

    # ---- 行业映射（申万一级，取 PIT 面板最新非空值）----
    industry = None
    if cache is not None:
        try:
            from data.industry import IndustryClassification
            icls = IndustryClassification(cache, level=1)
            ind_panel = icls.get_industry_panel(list(close.columns), close.index)
            industry = ind_panel.ffill().iloc[-1].dropna()
        except Exception as e:  # noqa: BLE001
            log.warning("IndustryClassification 失败，回退 rolling_grid 行业面板: %s",
                        str(e)[:80])
    if industry is None:
        # 回退：复用 rolling_grid prep 已建好的行业面板（同一股票池/日历）
        fallback = ROOT / "reports" / "alla_rolling" / "_base" / "cov_industry.parquet"
        if fallback.exists():
            cov = pd.read_parquet(fallback)
            industry = cov.ffill().iloc[-1].dropna()
            log.info("行业映射来自 %s（%d 只）", fallback.name, len(industry))
        else:
            log.warning("行业面板不可用，pap_high52w_ind 将跳过")

    data = PaperData(close=close, high=high, low=low, volume=volume,
                     market_cap=market_cap, industry=industry,
                     pit=pit, ann_flag=ann_flag)
    panels = compute_paper_factors(data)
    # 统一对齐到 close 网格（compute 内部已用 close 网格，这里兜底）
    for name in panels:
        panels[name] = panels[name].reindex(index=close.index, columns=close.columns)
    return panels


def main():
    parser = argparse.ArgumentParser(description="论文复现因子（awesome 21 式）构建入库")
    add_build_args(parser)
    parser.add_argument("--pool", default=None,
                        help="股票池（hs300/zz500/zz1000/all_a，默认取 config）")
    args = parser.parse_args()

    cache, uni, begin, end, dataset = make_data_context(args)

    codes, cal, daily = load_daily(cache, uni, args.index, begin, end, pool=args.pool)
    fin = load_financial_tables(cache, codes)
    if daily.empty:
        log.error("数据为空")
        sys.exit(1)
    log.info("日线 %d 行 / 利润表 %d / 资产负债 %d / 现金流 %d",
             len(daily), len(fin["income"]), len(fin["balance_sheet"]),
             len(fin["cash_flow"]))

    log.info("计算论文复现因子（%d 个）...", len(FACTOR_DEFS))
    panels = build_paper_panels(daily, cal, fin["income"], fin["balance_sheet"],
                                fin["cash_flow"], cache=cache)

    if args.no_save:
        print_no_save(list(FACTOR_DEFS), panels)
        return

    lib = FactorLibrary(dataset=dataset)
    returns_panel = returns_from_daily(daily)

    log.info("入库到数据集: %s", dataset)
    register_panels(
        lib, panels, FACTOR_DEFS, returns_panel,
        source=f"paper:awesome-systematic-trading:{begin}-{end}",
    )

    record_experiment_safe(
        kind="paper_factors",
        command=" ".join(sys.argv),
        params={"index": args.index, "begin": begin, "end": end, "dataset": dataset},
        fingerprint=cache.get_fingerprint(),
        result_path=str(lib.root),
        metrics={"n_factors": len([n for n in FACTOR_DEFS if n in panels])},
        note="awesome-systematic-trading 论文复现因子入库",
    )

    log.info("完成。数据集 %s 现有 %d 个因子", dataset, len(lib.list_all()))


if __name__ == "__main__":
    main()
