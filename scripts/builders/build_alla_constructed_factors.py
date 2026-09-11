"""构造型基本面因子族（B++）→ 并入 all_a_2018_2026 数据集。

用户确认（2026-09-07）：现有 A/B/B+ 基本面因子多为"报表字段直接比值/间差"，
缺业界经典的多字段合成 / 结构分解 / 跨报告期趋势类构造因子。本脚本基于
**已有财务四表**（income/balance_sheet/cash_flow/dividend，无新增数据源）构造
16 个构造型因子，覆盖五大维度：

  ① 盈利质量      np_ded_ratio(扣非/净利,负) / main_profit_ratio(主营贡献,正)
                   ebit_margin(EBIT率,正)
  ② 增长质量      np_rev_gap(利润弹性,正) / rev_recv_gap(营收-应收增速,负)
                   cfo_to_np(经营含金量,正)
  ③ 跨期趋势      margin_delta_ttm(毛利率同比,正) / np_accel_sq(单季盈利加速,正)
                   roe_vol_pit(ROE稳定性,负)
  ④ 财务健康      altman_zscore(正) / debt_to_ebitda(负) / interest_coverage(正)
  ⑤ 资产结构风险  risky_asset_ratio(易减值资产占比,负)
  ⑥ 股息持续性    div_growth_yoy(每股分红同比,正) / div_consecutive_years(负向剔险)

口径：
- 同比/环比/加速度在**报告期维度**（长表按 report_period 错位 1 年）计算，再 PIT 到交易日；
- roe_vol_pit 用 ROE 面板过去 252 交易日变异系数（过去方向，无未来）；
- 股息类用每股现金分红面板近 1 年同比 / 连续分红年数近似。
- 面板 date×code float32，与现有 ic_h 索引对齐；IC 为未来 1/5/10/20 日 Spearman。

用法:
    python -m scripts.builders.build_alla_constructed_factors            # 单机
    python -m scripts.builders.build_alla_constructed_factors --resume   # 断点续跑
    python -m scripts.builders.build_alla_constructed_factors --only np_ded_ratio  # 调试
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

from scripts.cli_common import setup_logging  # noqa: E402

log = setup_logging("build_alla_constructed")

DATASET = "all_a_2018_2026"
KEEP_FROM = "2016-07-01"
HORIZONS = (1, 5, 10, 20)
IC_CODE_STRIDE = 3

from scripts.builders.build_alla_fundamental_factors import (  # noqa: E402
    load_panels, add_ttm_yoy, add_single_quarter, sq_growth_long,
)

_CFO_FIELD = "NET_CASH_FLOWS_OPERA_ACT"


# ---------------------------------------------------------------------------
# 报告期错位 helper（同比差 / 同比增速，长表维度，错位 1 个自然年）
# ---------------------------------------------------------------------------
def year_offset(df: pd.DataFrame, field: str, mode: str = "delta") -> pd.DataFrame:
    """返回 df 补齐 off_col：当前 report_period 与上年同期 report_period 的差/增速。

    mode: delta = 当期 - 上年同期；ratio = 当期/上年同期 - 1。
    用于毛利率同比变化、单季盈利加速度、应收同比等——必须报告期精确对齐，
    不能用日频 shift（公告日不规则）。
    """
    if field not in df.columns:
        return df
    off = "_off_" + field
    d = df[["code", "ann_date", "report_period", field]].copy()
    d = d.dropna(subset=["report_period", field])
    d["_k"] = (d["code"].astype(str) + "_" + d["report_period"].dt.strftime("%Y-%m-%d"))
    lag = d[["code", "report_period", field]].copy()
    lag["report_period"] = lag["report_period"] - pd.DateOffset(years=1)
    lag["_k"] = (lag["code"].astype(str) + "_" + lag["report_period"].dt.strftime("%Y-%m-%d"))
    lag = lag.drop_duplicates(subset=["_k"], keep="last").rename(
        columns={field: "_lag"})
    m = d.merge(lag[["_k", "_lag"]], on="_k", how="left")
    if mode == "ratio":
        m[off] = m[field] / m["_lag"].replace(0.0, np.nan) - 1.0
    else:
        m[off] = m[field] - m["_lag"]
    main = df.copy()
    return main.merge(m[["code", "ann_date", "report_period", off]],
                      on=["code", "ann_date", "report_period"], how="left")


def _pit(report_df, cal_idx, codes, field):
    from data.financials import build_pit_panel
    if field not in report_df.columns or report_df[field].isna().all():
        return pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    return build_pit_panel(report_df, cal_idx, field).reindex(index=cal_idx, columns=codes)


def _safe(x, denom):
    return (x / denom.replace(0.0, np.nan))


def build_panels(close_adj, close_raw, inc, bal, cfo, div):
    codes = close_adj.columns
    cal_idx = close_adj.index
    close = close_raw.astype(float)
    cap_pit = _pit(bal, cal_idx, codes, "TOT_SHARE").astype(float) * close
    ln_cap = np.log(cap_pit.clip(lower=1.0))

    # ---- income: TTM / 单季 / 加速度 ----
    inc = add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", "REV_YOY")
    inc = add_ttm_yoy(inc, "LESS_OPERA_COST", "LESS_OPERA_COST_TTM", None)
    inc = add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", "NP_YOY")
    inc["GROSS_PROFIT_TTM"] = inc["OPERA_REV_TTM"] - inc["LESS_OPERA_COST_TTM"]
    inc = add_ttm_yoy(inc, "NET_PRO_AFTER_DED_NR_GL", "NP_DED_TTM", None)
    inc = add_ttm_yoy(inc, "EBIT", "EBIT_TTM", None)
    inc = add_ttm_yoy(inc, "EBITDA", "EBITDA_TTM", None)
    inc = add_ttm_yoy(inc, "LESS_FIN_EXP", "LESS_FIN_EXP_TTM", None)
    inc = add_single_quarter(inc, "NET_PRO_INCL_MIN_INT_INC", "NP_SQ")
    inc = sq_growth_long(inc, "NP_SQ", "NP_SQ_YOY", "NP_SQ_QOQ")
    # 单季盈利加速度 = 本期单季同比 - 上年同期单季同比
    inc = year_offset(inc, "NP_SQ_YOY", "delta")
    # 毛利率 TTM 同比变化
    inc["GM_RATIO"] = inc["GROSS_PROFIT_TTM"] / inc["OPERA_REV_TTM"].replace(0.0, np.nan)
    inc = year_offset(inc, "GM_RATIO", "delta")

    # ---- cash: CPP TTM ----
    cfo = add_ttm_yoy(cfo, _CFO_FIELD, "CFO_TTM", None)

    def pitf(f):
        return _pit(inc, cal_idx, codes, f)

    p = {}
    for f in ("OPERA_REV_TTM", "OPERA_REV", "NP_YOY", "REV_YOY", "NET_PRO_TTM",
              "NP_DED_TTM", "EBIT_TTM", "EBITDA_TTM", "LESS_FIN_EXP_TTM",
              "OPERA_PROFIT", "TOTAL_PROFIT", "_off_NP_SQ_YOY", "_off_GM_RATIO"):
        p[f] = pitf(f)

    pc = {f: _pit(cfo, cal_idx, codes, f)
          for f in ("CFO_TTM",) if f in cfo.columns}
    pb = {f: _pit(bal, cal_idx, codes, f)
          for f in ("TOTAL_ASSETS", "TOTAL_CUR_ASSETS", "TOTAL_CUR_LIAB",
                    "SURPLUS_RESV", "UNDISTRIBUTED_PRO", "TOTAL_LIAB",
                    "GOODWILL", "ACC_RECEIVABLE", "NOTES_RECEIVABLE", "INV")}
    ta = pb.get("TOTAL_ASSETS")

    rev_ttm = p["OPERA_REV_TTM"]
    np_ttm = p["NET_PRO_TTM"]

    panels: dict[str, pd.DataFrame] = {}

    # ① 盈利质量
    panels["np_ded_ratio"] = _safe(p["NP_DED_TTM"], np_ttm)            # 负：扣非占比低→靠一次性
    panels["main_profit_ratio"] = _safe(p["OPERA_PROFIT"], p["TOTAL_PROFIT"])   # 正
    panels["ebit_margin"] = _safe(p["EBIT_TTM"], rev_ttm)              # 正
    # ④ 财务健康
    if all(k in pb for k in ("TOTAL_CUR_ASSETS", "TOTAL_CUR_LIAB", "TOTAL_LIAB",
                             "SURPLUS_RESV", "UNDISTRIBUTED_PRO")):
        wc = pb["TOTAL_CUR_ASSETS"] - pb["TOTAL_CUR_LIAB"]
        re_ = pb["SURPLUS_RESV"].fillna(0.0) + pb["UNDISTRIBUTED_PRO"]
        x1 = _safe(wc, ta)
        x2 = _safe(re_, ta)
        x3 = _safe(p["EBIT_TTM"], ta)
        x4 = _safe(cap_pit, pb["TOTAL_LIAB"])
        x5 = _safe(rev_ttm, ta)
        panels["altman_zscore"] = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
        panels["debt_to_ebitda"] = _safe(pb["TOTAL_LIAB"], p["EBITDA_TTM"])   # 负
    panels["interest_coverage"] = _safe(p["EBIT_TTM"], p["LESS_FIN_EXP_TTM"])  # 正
    # ② 增长质量
    panels["np_rev_gap"] = p["NP_YOY"] - p["REV_YOY"]   # 正：利润弹性
    if "ACC_RECEIVABLE" in pb and "NOTES_RECEIVABLE" in pb:
        recv = pb["ACC_RECEIVABLE"].fillna(0.0) + pb["NOTES_RECEIVABLE"].fillna(0.0)
        recv_growth = recv.pct_change(244, fill_method=None)  # 日频近似同比（过去，无未来）
        panels["rev_recv_gap"] = p["REV_YOY"] - recv_growth      # 负：应收涨快于营收
    if "CFO_TTM" in pc:
        panels["cfo_to_np"] = _safe(pc["CFO_TTM"], np_ttm)      # 正：经营含金量
    # ③ 跨期趋势
    panels["margin_delta_ttm"] = p["_off_GM_RATIO"]               # 正：毛利率同比升
    panels["np_accel_sq"] = p["_off_NP_SQ_YOY"]                   # 正：单季盈利加速
    roe = _safe(np_ttm, _pit(bal, cal_idx, codes,
                             "TOT_SHARE_EQUITY_EXCL_MIN_INT"))
    mu = roe.rolling(252, min_periods=40).mean()
    sd = roe.rolling(252, min_periods=40).std()
    panels["roe_vol_pit"] = (sd / mu.abs().replace(0.0, np.nan))  # 负：越稳定越好
    # ⑤ 资产结构风险
    if all(k in pb for k in ("GOODWILL", "ACC_RECEIVABLE",
                             "NOTES_RECEIVABLE", "INV")):  # 负：易减值资产占比高
        risky = (pb["GOODWILL"].fillna(0.0)
                 + pb["ACC_RECEIVABLE"].fillna(0.0)
                 + pb["NOTES_RECEIVABLE"].fillna(0.0) + pb["INV"].fillna(0.0))
        panels["risky_asset_ratio"] = _safe(risky, ta)
    # ⑥ 股息持续性
    if div is not None and not div.empty and "cash_per_share_pre_tax" in div.columns:
        div_cps = _pit(div, cal_idx, codes, "cash_per_share_pre_tax")
        panels["div_growth_yoy"] = div_cps.pct_change(244, fill_method=None)   # 正
        panels["div_consecutive_years"] = _div_consecutive(div, cal_idx, codes)  # 正：连续分红年数

    out = {}
    for n, q in panels.items():
        out[n] = q.reindex(index=cal_idx, columns=codes)
    return out


def _div_consecutive(div: pd.DataFrame, cal_idx, codes) -> pd.DataFrame:
    """每股现金分红 >0 的连续年数（长表逐年聚合，PIT 展开）。

    对每 (code, 年度) 取该年最后一笔分红的每股金额，判 >0；在同一 code 内
    从早到晚统计"连续分红年数"，停止分红则断档归零。绑定到该年分红公告日
    前向填充到交易日（无未来函数）。
    """
    from data.financials import build_pit_panel
    dv = div[["code", "ann_date", "cash_per_share_pre_tax"]].copy()
    dv["ann_date"] = pd.to_datetime(dv["ann_date"], errors="coerce")
    dv["cps"] = pd.to_numeric(dv["cash_per_share_pre_tax"], errors="coerce").fillna(0.0)
    dv = dv.dropna(subset=["ann_date"])
    dv["year"] = dv["ann_date"].dt.year
    one = (dv.sort_values(["code", "year", "ann_date"])
             .groupby(["code", "year"], as_index=False).last()[["code", "year", "ann_date", "cps"]])
    one["pay"] = one["cps"] > 0.0
    one = one.sort_values(["code", "year"])
    one["brk"] = one.groupby("code")["pay"].apply(
        lambda s: (~s).cumsum()).reset_index(drop=True) if False else \
        one.groupby("code")["pay"].transform(lambda s: (~s).cumsum())
    one["consec"] = one.groupby(["code", "brk"]).cumcount() + 1
    one["consec"] = one["consec"].where(one["pay"], 0.0).astype(np.float32)
    pit_df = one.rename(columns={"consec": "_value"})[["code", "ann_date", "_value"]]
    return build_pit_panel(pit_df, cal_idx, "_value").reindex(
        index=cal_idx, columns=codes).astype(np.float32)


# ---------------------------------------------------------------------------
# 主流程 & 合并
# ---------------------------------------------------------------------------
def merge_outputs(ds_dir: Path) -> None:
    rows = [json.loads(l) for l in
            (ds_dir / "factor_stats_constructed.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        log.warning("无 B++ 构造因子统计产出")
        return
    reg = pd.read_csv(ds_dir / "registry.csv")
    new_df = pd.DataFrame(rows)
    before = len(reg)
    reg = (pd.concat([reg, new_df], ignore_index=True)
             .drop_duplicates(subset="name", keep="last"))
    reg.to_csv(ds_dir / "registry.csv", index=False, encoding="utf-8-sig")
    log.info("registry: %d -> %d 因子", before, len(reg))
    for h in HORIZONS:
        fuse_horizon_ic(ds_dir, h)


def fuse_horizon_ic(ds_dir: Path, h: int) -> None:
    from stats.ic import calc_ic_series

    stats = [json.loads(l) for l in
             (ds_dir / "factor_stats_constructed.jsonl").read_text(
                 encoding="utf-8").splitlines() if l.strip()]
    names = [s["name"] for s in stats]
    if not names:
        return
    ic = pd.read_parquet(ds_dir / f"ic_h{h}.parquet")
    close_adj, *_ = load_panels()
    fwd = close_adj.pct_change(h, fill_method=None).shift(-h)
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]
    for n in names:
        p = pd.read_parquet(ds_dir / "panels" / f"{n}.parquet")
        if n not in ic.columns:
            ic[n] = calc_ic_series(p[ic_codes], fwd).reindex(ic.index)
    ic = ic.astype(np.float32)
    ic.to_parquet(ds_dir / f"ic_h{h}.parquet")
    log.info("ic_h%d merged: %d 因子", h, ic.shape[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="只构建指定因子")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_constructed.jsonl"
    done = {json.loads(l)["name"] for l in
            stats_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if args.resume and stats_path.exists() else set()

    cache_root = Path(str(Config.cache()["root"]))
    close_adj, close_raw, income, balance, cashflow, equity, dividend = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    names_all = [
        "np_ded_ratio", "main_profit_ratio", "ebit_margin",
        "altman_zscore", "debt_to_ebitda", "interest_coverage",
        "np_rev_gap", "rev_recv_gap", "cfo_to_np",
        "margin_delta_ttm", "np_accel_sq", "roe_vol_pit",
        "risky_asset_ratio", "div_growth_yoy", "div_consecutive_years",
    ]
    if args.only:
        names_all = [n for n in names_all if n in set(args.only)]
    labels = {
        "np_ded_ratio": "扣非/净利", "main_profit_ratio": "主营贡献度",
        "ebit_margin": "EBIT率", "altman_zscore": "Altman Z",
        "debt_to_ebitda": "负债/EBITDA", "interest_coverage": "利息保障倍数",
        "np_rev_gap": "净利-营收增速差", "rev_recv_gap": "营收-应收增速差",
        "cfo_to_np": "经营含金量", "margin_delta_ttm": "毛利率同比变化",
        "np_accel_sq": "单季盈利加速度", "roe_vol_pit": "ROE波动(负)",
        "risky_asset_ratio": "易减值资产占比", "div_growth_yoy": "每股分红同比",
        "div_consecutive_years": "连续分红年数",
    }

    panels = build_panels(close_adj, close_raw, income, balance, cashflow, dividend)
    t0 = time.time()
    for i, name in enumerate([n for n in names_all if n in panels], 1):
        if name in done:
            continue
        p = panels[name].fillna(np.nan).astype(np.float32)
        p = p.replace([np.inf, -np.inf], np.nan).clip(-1e4, 1e4)
        p.to_parquet(panels_dir / f"{name}.parquet")
        cov = float(p.notna().mean().mean())
        ic = {h: calc_ic_series(p[ic_codes], fwd[h]).astype(np.float32) for h in HORIZONS}
        row = {"name": name, "set": "fundamental",
               "label": labels.get(name, name), "coverage": cov,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.3f ic_h1=%+.4f | %.0fs",
                 i, len(names_all), name, cov, row["ic_mean_h1"], time.time() - t0)

    merge_outputs(ds_dir)
    log.info("B++ 构造型基本面因子完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()