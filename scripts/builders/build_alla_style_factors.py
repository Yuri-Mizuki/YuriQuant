"""基本面风格因子族（C·Style）→ 并入 all_a_2018_2026 数据集。

用户确认（2026-09-08）：现有 A/B/B+/B++ 基本面因子（57 个）已覆盖
价值/质量/成长/规模/周转/分红/商誉/质押/现金流/盈利质量等维度，但以下
可挖掘维度在三个构建脚本中全部空白：
  ① 研发/费用结构   RD_EXP(研发强度) / 销售费用率 / 管理费用率 / 研发占总资产
  ② 杠杆细分       负债率 / 非流动负债结构
  ③ 流动性/偿债    流动比率 / 速动比率 / 现金比率（以上均未入库）
  ④ 盈利细分       EBITDA率 / 营业利润率
  ⑤ 商誉/资产质量  商誉/净资产 / 有形资产率 / 无形资产占比
  ⑥ 现金流质量    经营现金流/总资产 / 经营现金流/营收 / 经营现金流/总负债

本脚本基于**同一批 cached 财务四表**（无新增数据源）构建 16 个 C 族风格因子，
每个因子同时产出两种形态：
  - 原始 PIT 比率面板  → panels/{name}.parquet + registry（与 B/B+/B++ 同口径）
  - Barra 式风格因子    → panels_neu/{name}.parquet（MAD 去极值 → 行业哑变量 + 市值
                            中性化 → 截面 z-score，复用 factor.preprocessing::preprocess_factor）

口径：
- 流量类（营收/利润/费用/现金流）做 TTM；库存/负债/资产类为时点值，直接 PIT；
- 同比类一律在"报告期维度"错位（复用 B++ 的 year_offset），再 PIT 到交易日；
- 中性化用 reports/alla_rolling/_base 的 market_cap + cov_industry（申万一级），
  与滚动实验 `--preproc ortho` 的组织完全一致，输出到 panels_neu/。
- 纯 PIT、无未来函数；面板 date×code float32；IC 为未来 1/5/10/20 日 Spearman。

用法:
    python -m scripts.builders.build_alla_style_factors            # 单机
    python -m scripts.builders.build_alla_style_factors --resume   # 断点续跑
    python -m scripts.builders.build_alla_style_factors --only current_ratio  # 调试
    python -m scripts.builders.build_alla_style_factors --no-neu   # 只做原始比率，跳过中性化
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

log = setup_logging("build_alla_style")

DATASET = "all_a_2018_2026"
KEEP_FROM = "2016-07-01"
HORIZONS = (1, 5, 10, 20)
IC_CODE_STRIDE = 3
BASE_DIR = Path("reports") / "alla_rolling" / "_base"

from scripts.builders.build_alla_fundamental_factors import (  # noqa: E402
    load_panels, add_ttm_yoy, add_single_quarter, sq_growth_long,
)
from scripts.builders.build_alla_constructed_factors import year_offset  # noqa: E402

_CFO_FIELD = "NET_CASH_FLOWS_OPERA_ACT"


def _pit(report_df, cal_idx, codes, field):
    from data.financials import build_pit_panel
    if field not in report_df.columns or report_df[field].isna().all():
        return pd.DataFrame(np.nan, index=cal_idx, columns=codes)
    return build_pit_panel(report_df, cal_idx, field).reindex(index=cal_idx, columns=codes)


def _safe(x, denom):
    return (x / denom.replace(0.0, np.nan))


# 新增维度因子清单（{name: (label, family)}，family 用于中性化后的 set 标注）
FACTOR_DEFS = {
    # ① 研发 / 费用结构（之前完全空白）
    "rd_exp_ratio_ttm": ("研发强度=研发TTM/营收TTM", "rnd"),
    "rd_to_assets_ttm": ("研发占资产=研发TTM/总资产", "rnd"),
    "selling_exp_ratio_ttm": ("销售费用率TTM", "rnd"),
    "admin_exp_ratio_ttm": ("管理费用率TTM", "rnd"),
    # ② 杠杆细分
    "debt_to_assets": ("资产负债率=总负债/总资产", "lev"),
    "noncur_liab_ratio": ("非流动负债占比=非流动负债/总负债", "lev"),
    # ③ 流动性 / 偿债能力
    "current_ratio": ("流动比率=流动资产/流动负债", "liq"),
    "quick_ratio": ("速动比率=(流动资产-存货)/流动负债", "liq"),
    "cash_ratio": ("现金比率=货币资金/流动负债", "liq"),
    # ④ 盈利细分
    "ebitda_margin_ttm": ("EBITDA率TTM", "pro"),
    "op_income_margin_ttm": ("营业利润率TTM", "pro"),
    # ⑤ 商誉 / 资产质量
    "goodwill_to_equity": ("商誉/净资产", "qual"),
    "tangible_asset_ratio": ("有形资产率=(总资产-商誉-无形)/总资产", "qual"),
    "intangible_ratio": ("无形资产占比=无形资产/总资产", "qual"),
    # ⑥ 现金流质量
    "cfo_to_assets_ttm": ("经营现金流/总资产", "cfq"),
    "cfo_to_rev_ttm": ("经营现金流/营收TTM", "cfq"),
    "cfo_to_debt_ttm": ("经营现金流/总负债", "cfq"),
}


def build_panels(close_adj, close_raw, inc, bal, cfo, div):
    codes = close_adj.columns
    cal_idx = close_adj.index
    close = close_raw.astype(float)

    # ---- income: TTM / 单季，补齐研发/费用/盈利细分字段 ----
    inc = add_ttm_yoy(inc, "OPERA_REV", "OPERA_REV_TTM", None)
    inc = add_ttm_yoy(inc, "LESS_OPERA_COST", "LESS_OPERA_COST_TTM", None)
    inc = add_ttm_yoy(inc, "NET_PRO_INCL_MIN_INT_INC", "NET_PRO_TTM", None)
    inc["GROSS_PROFIT_TTM"] = inc["OPERA_REV_TTM"] - inc["LESS_OPERA_COST_TTM"]
    for f in ("RD_EXP", "LESS_SELLING_EXP", "LESS_ADMIN_EXP", "EBITDA", "OPERA_PROFIT"):
        if f in inc.columns:
            inc = add_ttm_yoy(inc, f, f + "_TTM", None)

    # ---- cash: CFD/CFO TTM ----
    cfo = add_ttm_yoy(cfo, _CFO_FIELD, "CFO_TTM", None)

    def pitf(f):
        return _pit(inc, cal_idx, codes, f)

    p = {}
    for f in ("OPERA_REV_TTM", "NET_PRO_TTM", "TOTAL_ASSETS",
              "RD_EXP_TTM", "LESS_SELLING_EXP_TTM", "LESS_ADMIN_EXP_TTM",
              "EBITDA_TTM", "OPERA_PROFIT_TTM"):
        if f in inc.columns:
            p[f] = pitf(f)

    pc = {f: _pit(cfo, cal_idx, codes, f) for f in ("CFO_TTM",) if f in cfo.columns}
    pb = {f: _pit(bal, cal_idx, codes, f)
          for f in ("TOTAL_ASSETS", "TOTAL_LIAB", "TOTAL_CUR_ASSETS",
                    "TOTAL_CUR_LIAB", "TOTAL_NONCUR_LIAB", "INV",
                    "CURRENCY_CAP", "GOODWILL", "INTANGIBLE_ASSETS",
                    "TOT_SHARE_EQUITY_EXCL_MIN_INT")}

    rev_ttm = p.get("OPERA_REV_TTM")
    ta = pb.get("TOTAL_ASSETS")
    tl = pb.get("TOTAL_LIAB")
    tcl = pb.get("TOTAL_CUR_LIAB")

    panels: dict[str, pd.DataFrame] = {}

    # ① 研发 / 费用结构
    if rev_ttm is not None and "RD_EXP_TTM" in p:
        panels["rd_exp_ratio_ttm"] = _safe(p["RD_EXP_TTM"], rev_ttm)
    if ta is not None and "RD_EXP_TTM" in p:
        panels["rd_to_assets_ttm"] = _safe(p["RD_EXP_TTM"], ta)
    if rev_ttm is not None and "LESS_SELLING_EXP_TTM" in p:
        panels["selling_exp_ratio_ttm"] = _safe(p["LESS_SELLING_EXP_TTM"], rev_ttm)
    if rev_ttm is not None and "LESS_ADMIN_EXP_TTM" in p:
        panels["admin_exp_ratio_ttm"] = _safe(p["LESS_ADMIN_EXP_TTM"], rev_ttm)

    # ② 杠杆细分
    if ta is not None and tl is not None:
        panels["debt_to_assets"] = _safe(tl, ta)
    if tl is not None and "TOTAL_NONCUR_LIAB" in pb:
        panels["noncur_liab_ratio"] = _safe(pb["TOTAL_NONCUR_LIAB"].fillna(0.0), tl)

    # ③ 流动性 / 偿债能力
    if tcl is not None and "TOTAL_CUR_ASSETS" in pb:
        panels["current_ratio"] = _safe(pb["TOTAL_CUR_ASSETS"], tcl)
        quick_num = pb["TOTAL_CUR_ASSETS"].astype(float) - pb.get("INV", 0.0).fillna(0.0)
        panels["quick_ratio"] = _safe(quick_num, tcl)
    if tcl is not None and "CURRENCY_CAP" in pb:
        panels["cash_ratio"] = _safe(pb["CURRENCY_CAP"].fillna(0.0), tcl)

    # ④ 盈利细分
    if rev_ttm is not None and "EBITDA_TTM" in p:
        panels["ebitda_margin_ttm"] = _safe(p["EBITDA_TTM"], rev_ttm)
    if rev_ttm is not None and "OPERA_PROFIT_TTM" in p:
        panels["op_income_margin_ttm"] = _safe(p["OPERA_PROFIT_TTM"], rev_ttm)

    # ⑤ 商誉 / 资产质量
    eq = pb.get("TOT_SHARE_EQUITY_EXCL_MIN_INT")
    if eq is not None and "GOODWILL" in pb:
        panels["goodwill_to_equity"] = _safe(pb["GOODWILL"].fillna(0.0), eq)
    if ta is not None and "GOODWILL" in pb and "INTANGIBLE_ASSETS" in pb:
        tang = (ta.astype(float) - pb["GOODWILL"].fillna(0.0)
                - pb["INTANGIBLE_ASSETS"].fillna(0.0))
        panels["tangible_asset_ratio"] = _safe(tang, ta)
    if ta is not None and "INTANGIBLE_ASSETS" in pb:
        panels["intangible_ratio"] = _safe(pb["INTANGIBLE_ASSETS"].fillna(0.0), ta)

    # ⑥ 现金流质量
    if "CFO_TTM" in pc:
        cfo_ttm = pc["CFO_TTM"]
        if ta is not None:
            panels["cfo_to_assets_ttm"] = _safe(cfo_ttm, ta)
        if rev_ttm is not None:
            panels["cfo_to_rev_ttm"] = _safe(cfo_ttm, rev_ttm)
        if tl is not None:
            panels["cfo_to_debt_ttm"] = _safe(cfo_ttm, tl)

    out = {}
    for n, q in panels.items():
        out[n] = q.reindex(index=cal_idx, columns=codes)
    return out


def _neutralize_one(name: str, panels_dir: Path, ds_dir: Path) -> float:
    """对原始面板做 Barra 式中性化（MAD → 行业+市值 → zscore）写 panels_neu/。

    复用滚动实验 _base 的市值/行业面板（与 `--preproc ortho` 同组织），
    覆盖度低于阈值的因子同样产出（让上游按 coverage 自行筛选）。
    返回中性化后覆盖率。
    """
    from factor.preprocessing import preprocess_factor

    p = pd.read_parquet(panels_dir / f"{name}.parquet")
    base = ROOT / BASE_DIR
    mc = pd.read_parquet(base / "market_cap.parquet")
    ind = pd.read_parquet(base / "cov_industry.parquet")
    common = mc.index.intersection(p.index)
    p = p.reindex(index=common)
    mc = mc.reindex(index=common, columns=p.columns)
    ind = ind.reindex(index=common, columns=p.columns)
    x = preprocess_factor(p, market_cap_panel=mc, industry_panel=ind)
    x = x.replace([np.inf, -np.inf], np.nan).astype(np.float32).clip(-10.0, 10.0)
    out_dir = ds_dir / "panels_neu"
    out_dir.mkdir(parents=True, exist_ok=True)
    x.to_parquet(out_dir / f"{name}.parquet")
    return float(x.notna().mean().mean())


def _load_styles_file(ds_dir: Path) -> list[dict]:
    sp = ds_dir / "factor_stats_style.jsonl"
    if not sp.exists():
        return []
    return [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]


def merge_outputs(ds_dir: Path) -> None:
    rows = _load_styles_file(ds_dir)
    if not rows:
        log.warning("无 C 族风格因子统计产出")
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

    names = [s["name"] for s in _load_styles_file(ds_dir)]
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
    ap = argparse.ArgumentParser(description="基本面风格因子族（C）构建")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="只构建指定因子")
    ap.add_argument("--no-neu", action="store_true", help="只做原始比率，跳过中性化")
    args = ap.parse_args()

    from config import Config
    from stats.ic import calc_ic_series

    lib_root = Path(str(Config.get()["factor_library"]["root"]))
    ds_dir = lib_root / DATASET
    panels_dir = ds_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    stats_path = ds_dir / "factor_stats_style.jsonl"
    done = {json.loads(l)["name"] for l in
            stats_path.read_text(encoding="utf-8").splitlines() if l.strip()} \
        if args.resume and stats_path.exists() else set()

    close_adj, close_raw, income, balance, cashflow, equity, dividend = load_panels()
    codes = close_adj.columns
    cal_idx = close_adj.index
    fwd = {h: close_adj.pct_change(h, fill_method=None).shift(-h) for h in HORIZONS}
    ic_codes = close_adj.columns[::IC_CODE_STRIDE]

    names_all = list(FACTOR_DEFS)
    if args.only:
        names_all = [n for n in names_all if n in set(args.only)]

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
        neu_cov = None
        if not args.no_neu:
            neu_cov = _neutralize_one(name, panels_dir, ds_dir)
        label, family = FACTOR_DEFS[name]
        row = {"name": name, "set": "fundamental",
               "label": label, "coverage": cov,
               "neutralized_coverage": neu_cov, "family": family,
               **{f"ic_mean_h{h}": float(ic[h].mean()) for h in HORIZONS}}
        with stats_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("[%d/%d] %s cov=%.3f neu=%.3f ic_h1=%+.4f | %.0fs",
                 i, len(names_all), name, cov, neu_cov or 0.0,
                 row["ic_mean_h1"], time.time() - t0)

    merge_outputs(ds_dir)
    log.info("C 族基本面风格因子完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()