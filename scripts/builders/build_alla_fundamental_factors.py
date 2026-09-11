"""全A财务报表基本面因子面板（B族）→ 并入 all_a_2018_2026 数据集。

背景（2026-09-03）：A族（股东结构行为）已就位并入 all_a_2018_2026，但完整
"基本面族"还缺财务报表维度（价值/质量/成长/规模/杠杆/周转，以及分红类）。
本脚本在全A尺度（含退市，财务四表已由 backfill_financial_alla 补到 5810 主清单）
构建 B族因子，并 append 进 registry.csv 与 ic_h{1,5,10,20}.parquet，使滚动实验
select 阶段能把它们纳入特征候选池（量价之外的价值/质量/成长信息维度）。

因子清单（财务口径与 build_fundamental_factors 严格一致，PIT 到日频）：
  规模      : ln_mktcap / float_mktcap / float_ratio
  价值      : ep_ttm / bp / sp_ttm / pcf_ttm / pcf / fcf_yield / netcf_yield / peg
  分红      : div_yield / div_yield_ttm / div_payout_ratio
  质量      : roe_ttm / roa_ttm / gross_margin / net_margin / accruals / roic_ttm
               / fin_exp_ratio_ttm / asset_turnover_ttm / inv_turnover_ttm / recv_turnover_ttm
  成长      : rev_growth_yoy / np_growth_yoy / np_growth_sq_yoy / np_growth_sq_qoq
               / rev_growth_sq_yoy / rev_growth_sq_qoq / oppro_growth_sq_yoy
               / oppro_growth_sq_qoq / np_ded_growth_ttm_yoy / cfo_growth_ttm_yoy
  杠杆      : leverage
（与 A族股东结构不重叠；股东户数/十大股东因子由 A族 build_alla_holder_factors 承担）

口径：
- 财务长表加 TTM / 同比 / 单季列；用 build_pit_panel(ann_date 前向填充) 展开到交易日；
- 市值 = 期末总股本(TOT_SHARE) × 未复权收盘价（与 build_fundamental_factors 一致）；
- 面板 date×code float32，从 2016-07-01 起与现有 ic_h 索引对齐；
- IC 用次日/未来 1/5/10/20 日 Spearman（与量价 calc_ic_series 同口径），抽稀取列。

用法:
    python -m scripts.builders.build_alla_fundamental_factors            # 单机
    python -m scripts.builders.build_alla_fundamental_factors --resume   # 断点续跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_fundam")
from scripts.builders import common  # noqa: E402
from scripts.common.fundamental_common import add_single_quarter, add_ttm_yoy  # noqa: E402
from scripts.builders.common import KEEP_FROM, HORIZONS, IC_CODE_STRIDE  # noqa: E402

DATASET = "all_a_2018_2026"

# ---- 财务长表字段（all_a 全表已含，见 backfill_financial_alla）----
_INCOME_FIELDS = {
    "OPERA_REV": "营业收入",
    "LESS_OPERA_COST": "营业成本",
    "NET_PRO_INCL_MIN_INT_INC": "净利润(含少数)",
}
_BALANCE_FIELDS = {
    "TOTAL_ASSETS": "总资产",
    "TOT_SHARE_EQUITY_EXCL_MIN_INT": "股东权益(不含少数)",
    "TOT_SHARE": "期末总股本(股)",
}
_CFO_FIELD = "NET_CASH_FLOWS_OPERA_ACT"
_FREE_CF_FIELD = "FREE_CASH_FLOW"
_NET_CF_FIELD = "NET_INCR_CASH_AND_CASH_EQU"


# ---------------------------------------------------------------------------
# TTM / 同比 / 单季（长表维度，逻辑与 build_fundamental_factors 一致）
# ---------------------------------------------------------------------------
def sq_growth_long(df, sq_col, yoy_col, qoq_col):
    d = df[["code", "ann_date", "report_period", sq_col]].copy()
    d = d.dropna(subset=["report_period", sq_col])
    d["year"] = d["report_period"].dt.year
    d["quarter"] = d["report_period"].dt.quarter
    d["_key"] = (d["code"].astype(str) + "_" + d["year"].astype(str)
                 + "_" + d["quarter"].astype(str))
    d["_yoy_key"] = (d["code"].astype(str) + "_" + (d["year"] - 1).astype(str)
                     + "_" + d["quarter"].astype(str))
    prev_q = (d["quarter"] - 2) % 4 + 1
    prev_y = d["year"] - (d["quarter"] == 1).astype(int)
    d["_qoq_key"] = (d["code"].astype(str) + "_" + prev_y.astype(str)
                     + "_" + prev_q.astype(str))
    d = d.sort_values(["code", "report_period", "ann_date"])
    d = d.drop_duplicates(subset=["code", "report_period"], keep="last")
    val_map = dict(zip(d["_key"], d[sq_col]))
    base_yoy = d["_yoy_key"].map(lambda k: val_map.get(k, np.nan))
    base_qoq = d["_qoq_key"].map(lambda k: val_map.get(k, np.nan))
    d[yoy_col] = d[sq_col] / base_yoy.replace(0.0, np.nan) - 1.0
    d[qoq_col] = d[sq_col] / base_qoq.replace(0.0, np.nan) - 1.0
    return df.merge(d[["code", "ann_date", "report_period", yoy_col, qoq_col]],
                    on=["code", "ann_date", "report_period"], how="left")


def load_panels():
    """加载全A复权价/未复权价 + 财务四表 + 股本结构。返回 tuple。"""
    from config import Config

    cache_root = Path(str(Config.cache()["root"]))
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(daily.index.levels[0].normalize(), level=0)
    k0 = pd.Timestamp(KEEP_FROM)
    daily = daily[daily.index.get_level_values(0) >= k0]

    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf.index = bf.index.normalize()
    bf = bf.loc[bf.index >= k0]

    close_raw = daily["close"].unstack()
    # 只保留股票交集：bf 宽表含基金等非股票列，按 bf.columns 反向扩张会把
    # 财务表里的基金一并带进因子面板（2026-09-07 清理 32 只 ETF 列的根源）
    cols = close_raw.columns.intersection(bf.columns)
    close_raw = close_raw.reindex(close_raw.index.intersection(bf.index), columns=cols)
    bf = bf.reindex(index=close_raw.index, columns=close_raw.columns)
    close_adj = (close_raw * bf).astype(np.float32)

    income = pd.read_parquet(cache_root / "income.parquet")
    balance = pd.read_parquet(cache_root / "balance_sheet.parquet")
    cashflow = pd.read_parquet(cache_root / "cash_flow.parquet")
    equity = pd.read_parquet(cache_root / "equity_structure.parquet")
    dividend = pd.read_parquet(cache_root / "dividend.parquet")
    return close_adj, close_raw, income, balance, cashflow, equity, dividend


def _equity_float_pit(equity, cal_idx, codes) -> pd.DataFrame:
    """流通股本按 change_date 前向填充到交易日。"""
    out = pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    if equity is None or equity.empty or "float_share" not in equity.columns:
        return out
    d = equity[["code", "change_date", "float_share"]].copy()
    d["change_date"] = pd.to_datetime(d["change_date"], errors="coerce")
    d = d.dropna(subset=["change_date", "float_share"])
    for code, g in d.groupby("code"):
        if code not in codes:
            continue
        g = (g.dropna(subset=["float_share"]).sort_values("change_date")
              .drop_duplicates(subset="change_date", keep="last"))
        s = g.set_index("change_date")["float_share"]
        out[code] = s.reindex(cal_idx, method="ffill").values
    return out


def build_fundamental_panels(close_raw, cal_idx, codes, income, balance, cashflow,
                             equity, dividend) -> dict[str, pd.DataFrame]:
    """返回 {因子名: date×code 原始面板}（B族财务报表基本面）。"""
    from data.financials import build_pit_panel

    def _pit(report_df, field):
        if field not in report_df.columns:
            return pd.DataFrame(np.nan, index=cal_idx, columns=codes)
        return build_pit_panel(report_df, cal_idx, field).reindex(
            index=cal_idx, columns=codes)

    # ---- 长表加 TTM / 同比 / 单季列 ----
    inc = income.copy()
    inc = add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", "REV_YOY")
    inc = add_ttm_yoy(inc, "LESS_OPERA_COST", "LESS_OPERA_COST_TTM", None)
    inc = add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", "NP_YOY")
    inc["GROSS_PROFIT_TTM"] = (inc["OPERA_REV_TTM"] - inc["LESS_OPERA_COST_TTM"])
    inc = add_ttm_yoy(inc, "NET_PRO_AFTER_DED_NR_GL", "NP_DED_TTM", "NP_DED_TTM_YOY")
    inc = add_ttm_yoy(inc, "LESS_FIN_EXP", "LESS_FIN_EXP_TTM", None)
    inc = add_single_quarter(inc, "NET_PRO_INCL_MIN_INT_INC", "NP_SQ")
    inc = add_single_quarter(inc, "OPERA_REV", "REV_SQ")
    inc = add_single_quarter(inc, "OPERA_PROFIT", "OPPROF_SQ")
    inc = add_ttm_yoy(inc, "EBIT", "EBIT_TTM", None)

    cf = cashflow.copy()
    cf = add_ttm_yoy(cf, _CFO_FIELD, "CFO_TTM", "CFO_YOY")
    cf = add_ttm_yoy(cf, _FREE_CF_FIELD, "FREE_CF_TTM", None)
    cf = add_ttm_yoy(cf, _NET_CF_FIELD, "NET_CF_TTM", None)

    # 单季同比/环比（在长表内按 key 对齐）
    inc = sq_growth_long(inc, "NP_SQ", "NP_SQ_YOY", "NP_SQ_QOQ")
    inc = sq_growth_long(inc, "REV_SQ", "REV_SQ_YOY", "REV_SQ_QOQ")
    inc = sq_growth_long(inc, "OPPROF_SQ", "OPPROF_SQ_YOY", "OPPROF_SQ_QOQ")

    # ---- PIT 展开基础字段 ----
    pit = {}
    for f in ("OPERA_REV_TTM", "NET_PRO_TTM", "GROSS_PROFIT_TTM", "REV_YOY", "NP_YOY",
              "LESS_OPERA_COST_TTM", "NP_SQ_YOY", "NP_SQ_QOQ", "REV_SQ_YOY", "REV_SQ_QOQ",
              "OPPROF_SQ_YOY", "OPPROF_SQ_QOQ", "NP_DED_TTM_YOY", "EBIT_TTM",
              "LESS_FIN_EXP_TTM"):
        if f in inc.columns:
            pit[f] = _pit(inc, f)
    for f in ("TOTAL_ASSETS", "TOT_SHARE_EQUITY_EXCL_MIN_INT", "TOT_SHARE",
              "INV", "ACC_RECEIVABLE", "NOTES_RECEIVABLE", "ST_BORROWING",
              "LT_LOAN", "BONDS_PAYABLE", "NONCUR_LIAB_DUE_WITHIN_1Y"):
        if f in balance.columns:
            pit[f] = _pit(balance, f)
    for f in ("CFO_TTM", "CFO_YOY", "FREE_CF_TTM", "NET_CF_TTM", _CFO_FIELD):
        if f in cf.columns:
            pit[f] = _pit(cf, f)

    # 税率（ROIC 用）
    tax_panel = None
    if "INCOME_TAX" in inc.columns and "TOTAL_PROFIT" in inc.columns:
        tax_panel = (_pit(inc, "INCOME_TAX") / _pit(inc, "TOTAL_PROFIT").replace(0.0, np.nan)
                     ).clip(0.0, 0.6).fillna(0.25)

    close = close_raw.astype(float)
    cap = pit["TOT_SHARE"].astype(float) * close      # 市值（元）
    ln_cap = np.log(cap.clip(lower=1.0))

    equil = pit["TOT_SHARE_EQUITY_EXCL_MIN_INT"]
    assets = pit["TOTAL_ASSETS"]
    rev_ttm = pit["OPERA_REV_TTM"]
    np_ttm = pit["NET_PRO_TTM"]
    debt = (pit["ST_BORROWING"].fillna(0.0) + pit["LT_LOAN"].fillna(0.0)
            + pit["BONDS_PAYABLE"].fillna(0.0) + pit["NONCUR_LIAB_DUE_WITHIN_1Y"].fillna(0.0))
    invest_cap = equil + debt
    recv = pit["ACC_RECEIVABLE"].fillna(0.0) + pit["NOTES_RECEIVABLE"].fillna(0.0)

    float_share = _equity_float_pit(equity, cal_idx, codes)
    float_mktcap = float_share.astype(float) * close
    float_ratio = float_share / pit["TOT_SHARE"].replace(0.0, np.nan)

    def _safe(x, denom):
        return (x / denom.replace(0.0, np.nan))

    panels: dict[str, pd.DataFrame] = {
        "ln_mktcap": ln_cap,
        "float_mktcap": float_mktcap,
        "float_ratio": float_ratio,
        "ep_ttm": _safe(np_ttm, cap),
        "bp": _safe(equil, cap),
        "sp_ttm": _safe(rev_ttm, cap),
        "pcf_ttm": _safe(cap, pit["CFO_TTM"]),
        "pcf": _safe(cap, pit[_CFO_FIELD]),
        "fcf_yield": _safe(pit["FREE_CF_TTM"], cap),
        "netcf_yield": _safe(pit["NET_CF_TTM"], cap),
        "peg": _safe(_safe(cap, np_ttm), pit["NP_YOY"] * 100.0),
        "roe_ttm": _safe(np_ttm, equil),
        "roa_ttm": _safe(np_ttm, assets),
        "gross_margin": _safe(pit["GROSS_PROFIT_TTM"], rev_ttm),
        "net_margin": _safe(np_ttm, rev_ttm),
        "accruals": _safe(np_ttm - pit["CFO_TTM"], assets),
        "leverage": _safe(assets, equil),
        "asset_turnover_ttm": _safe(rev_ttm, assets),
        "inv_turnover_ttm": _safe(pit["LESS_OPERA_COST_TTM"], pit["INV"]),
        "recv_turnover_ttm": _safe(rev_ttm, recv),
        "fin_exp_ratio_ttm": (_safe(pit["LESS_FIN_EXP_TTM"], rev_ttm)
                              if "LESS_FIN_EXP_TTM" in pit else None),
        "rev_growth_yoy": pit["REV_YOY"],
        "np_growth_yoy": pit["NP_YOY"],
        "np_growth_sq_yoy": pit["NP_SQ_YOY"],
        "np_growth_sq_qoq": pit["NP_SQ_QOQ"],
        "rev_growth_sq_yoy": pit["REV_SQ_YOY"],
        "rev_growth_sq_qoq": pit["REV_SQ_QOQ"],
        "oppro_growth_sq_yoy": pit["OPPROF_SQ_YOY"],
        "oppro_growth_sq_qoq": pit["OPPROF_SQ_QOQ"],
        "np_ded_growth_ttm_yoy": pit["NP_DED_TTM_YOY"],
        "cfo_growth_ttm_yoy": pit["CFO_YOY"],
    }
    if tax_panel is not None and pit.get("EBIT_TTM") is not None:
        panels["roic_ttm"] = _safe(pit["EBIT_TTM"] * (1 - tax_panel), invest_cap)

    # ---- 分红类（dividend 表，PIT 按实施公告日；distinct from A族股东结构）----
    div_p = _dividend_factors(dividend, cal_idx, codes, close, cap, np_ttm)
    panels.update(div_p)

    # 移除空面板 + 统一对齐
    out = {}
    for n, p in panels.items():
        if p is None:
            continue
        out[n] = p.reindex(index=cal_idx, columns=codes)
    return out


def _dividend_factors(dividend, cal_idx, codes, close, cap, np_ttm):
    """分红类因子：div_yield / div_yield_ttm / div_payout_ratio（PIT 无未来函数）。"""
    out = {"div_yield": pd.DataFrame(np.nan, index=cal_idx, columns=codes),
           "div_yield_ttm": pd.DataFrame(np.nan, index=cal_idx, columns=codes),
           "div_payout_ratio": pd.DataFrame(np.nan, index=cal_idx, columns=codes)}
    if dividend is None or dividend.empty:
        return out
    dd = dividend.copy()
    for c in ("ann_date", "payout_date", "report_period"):
        if c in dd.columns:
            dd[c] = pd.to_datetime(dd[c], errors="coerce")
    if "cash_per_share_pre_tax" in dd.columns:
        dd["cash_per_share_pre_tax"] = pd.to_numeric(dd["cash_per_share_pre_tax"], errors="coerce")
    dd = dd.dropna(subset=["cash_per_share_pre_tax", "ann_date", "code"])

    # 1) div_yield：最近一期已实施每股派息（按 ann_date PIT）/ 收盘价
    from data.financials import build_pit_panel
    cps = build_pit_panel(dd, cal_idx, "cash_per_share_pre_tax").reindex(
        index=cal_idx, columns=codes)
    out["div_yield"] = cps / close.replace(0.0, np.nan)

    # 2) div_yield_ttm：派息日落在过去365天的累计分红 / 市值（事件级循环，稀疏可接受）
    base_share = dd["base_share"] if "base_share" in dd.columns else None
    if "base_share" in dd.columns:
        base_share = pd.to_numeric(dd["base_share"], errors="coerce").fillna(1.0)
    else:
        base_share = pd.Series(1.0, index=dd.index)
    paid = dd.dropna(subset=["payout_date"])
    for code, g in paid.groupby("code"):
        if code not in codes or code not in cap.columns:
            continue
        amt = (g["cash_per_share_pre_tax"].values * base_share.reindex(g.index).fillna(1.0).values)
        for r, a in zip(g["payout_date"].values, amt):
            win = cal_idx[(cal_idx >= r) & (cal_idx <= r + pd.Timedelta(days=365))]
            if len(win):
                col = out["div_yield_ttm"][code]
                col.loc[win] = col.loc[win].fillna(0.0) + a
    out["div_yield_ttm"] = (out["div_yield_ttm"] / cap.replace(0.0, np.nan))

    # 3) div_payout_ratio：每股分红×基准股本 / 对应报告期净利（PIT ann_date）
    if "report_period" in dd.columns:
        rows = []
        for code, g in dd.dropna(subset=["report_period"]).groupby("code"):
            if code not in codes:
                continue
            if code not in np_ttm.columns:
                continue
            np_series = np_ttm[code]
            bs = base_share.reindex(g.index).fillna(1.0).values
            for ann, cps_v, bsv in zip(g["ann_date"].values,
                                       g["cash_per_share_pre_tax"].values, bs):
                known = np_series[np_series.index <= ann]
                if known.empty:
                    continue
                np_val = float(known.iloc[-1])
                if np_val and np_val == np_val:
                    rows.append((ann, code, float(cps_v) * float(bsv) / np_val))
        if rows:
            pl = pd.DataFrame(rows, columns=["ann_date", "code", "ratio"])
            out["div_payout_ratio"] = build_pit_panel(pl, cal_idx, "ratio").reindex(
                index=cal_idx, columns=codes)
    return out


# ---------------------------------------------------------------------------
# 主流程 & 合并
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_fundamental.jsonl"
    done = {json.loads(line)["name"] for line in
            stats_path.read_text(encoding="utf-8").splitlines() if line.strip()} \
        if args.resume and stats_path.exists() else set()

    close_adj, close_raw, income, balance, cashflow, equity, dividend = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    labels = {
        "ln_mktcap": "对数市值", "float_mktcap": "流通市值", "float_ratio": "流通比例",
        "ep_ttm": "盈利收益率EP", "bp": "账面市值比", "sp_ttm": "营收市值比",
        "pcf_ttm": "市现率TTM", "pcf": "市现率", "fcf_yield": "自由现金流率",
        "netcf_yield": "净现金流率", "peg": "PEG", "div_yield": "股息率",
        "div_yield_ttm": "股息率TTM", "div_payout_ratio": "股利支付率",
        "roe_ttm": "ROE", "roa_ttm": "ROA", "gross_margin": "毛利率",
        "net_margin": "净利率", "accruals": "应计利润", "roic_ttm": "ROIC",
        "fin_exp_ratio_ttm": "财务费用率", "asset_turnover_ttm": "资产周转率",
        "inv_turnover_ttm": "存货周转率", "recv_turnover_ttm": "应收周转率",
        "leverage": "财务杠杆", "rev_growth_yoy": "营收同比", "np_growth_yoy": "净利同比",
        "np_growth_sq_yoy": "净利单季同比", "np_growth_sq_qoq": "净利单季环比",
        "rev_growth_sq_yoy": "营收单季同比", "rev_growth_sq_qoq": "营收单季环比",
        "oppro_growth_sq_yoy": "营业利润单季同比", "oppro_growth_sq_qoq": "营业利润单季环比",
        "np_ded_growth_ttm_yoy": "扣非净利增速TTM同比",
        "cfo_growth_ttm_yoy": "经营现金流增速TTM同比",
    }

    panels = build_fundamental_panels(close_raw, cal_idx, codes, income, balance,
                                      cashflow, equity, dividend)

    t0 = time.time()
    for i, name in enumerate(sorted(panels), 1):
        if name in done:
            continue
        p = panels[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan)
        p = p.clip(-1e4, 1e4)
        p.to_parquet(panels_dir / f"{name}.parquet")

        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32) for h in HORIZONS}
        row = {"name": name, "set": "fundamental",
               "label": labels.get(name, name), "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.2f ic_h1=%+.4f | %.0fs", i, len(panels),
                 name, cov, row["ic_mean_h1"], time.time() - t0)

    common.merge_outputs(ds_dir, 'fundamental', skip_existing=False, empty_log='无基本面因子统计产出')
    log.info("B族财务报表基本面因子构建完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()