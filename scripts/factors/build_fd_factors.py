"""飞书《因子字典》候选因子 → 因子库体检入库
=============================================

对 ``factor/factor_dict_factors.py`` 里的 5 个候选跑与 ``build_intraday_stat_factors``
同一套体检：截面 zscore 入库 ``FactorLibrary``（IC / NW-t / IC 衰减 / canonical
回测 / 冗余查重全套自动算），再补一份**安慰剂对照**（``research.caliber`` 的
循环平移）。

候选来源与筛选理由见 ``reports/factor_dict_probe/README.md``。

用法
----
    python -m scripts.factors.build_fd_factors --offline
    python -m scripts.factors.build_fd_factors --offline --no-save
    python -m scripts.factors.build_fd_factors --offline --features fd_tcc_g,fd_ent30
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

from data.cache import DataCache  # noqa: E402
from data.cache_helpers import (  # noqa: E402
    load_backward_factor,
    load_daily,
    load_financial_tables,
)
from data.financials import build_pit_panel  # noqa: E402
from data.intraday import MinutePanelStore  # noqa: E402
from data.offline import OfflineDataSource  # noqa: E402
from data.universe import Universe  # noqa: E402
from factor.factor_dict_factors import (  # noqa: E402
    entropy_amount_share,
    group_abnormal_panel,
    rnbr_panel,
    tcc_panel,
)
from research.factor_criteria import (  # noqa: E402
    ADMISSION_GATES,
    cyclic_placebo,
    effect_size,
    panel_coverage,
)
from scripts.common.cli_common import (  # noqa: E402
    record_experiment_safe,
    register_panels,
    setup_logging,
)
from stats.ic import calc_ic_series, ic_stats  # noqa: E402

log = setup_logging("build_fd_factors")

FACTOR_DEFS: dict[str, str] = {
    "fd_tcc": "时间网络相对中心度(全截面) = 1/mean_21( 日收益全截面zscore² )",
    "fd_tcc_g": "时间网络相对中心度(邪修) = 1/mean_21( 日收益在[申万一级×市值三档]组内zscore² )",
    "fd_grp_abn": "分组异常收益 = max_21( |日收益 − [行业×市值三档]组内均值| )",
    "fd_rnbr": "自信溢出 = std_21( 日收益对代码邻接10只收益均值做带截距截面OLS的残差 )",
    "fd_ent30": "单一成交额占比熵(30min) = −Σ p·ln p，p=(close_t/Σclose)×(vol_t/Σvol)",
}

FAMILY = {"fd_tcc": "波动率", "fd_tcc_g": "波动率", "fd_grp_abn": "波动率",
          "fd_rnbr": "情绪", "fd_ent30": "日内"}


# ----------------------------------------------------------------------
# 数据装载
# ----------------------------------------------------------------------
def load_daily_context(begin: int, end: int):
    """全 A 日线 → (ret, market_cap, industry, returns_fwd, dates, codes)。"""
    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    codes, _cal, daily = load_daily(cache, uni, "000300.SH", begin, end,
                                    pool="all_a")
    d = daily.reset_index()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    def _panel(col: str) -> pd.DataFrame:
        return d.pivot(index="date", columns="code", values=col).sort_index()

    close_raw, volume = _panel("close"), _panel("volume")
    bf = load_backward_factor(cache, list(close_raw.columns))
    if bf.empty:
        log.warning("无后复权因子，退化为原始价（除权日收益有跳变）")
        close = close_raw
    else:
        close = close_raw * bf.reindex(index=close_raw.index,
                                       columns=close_raw.columns).ffill()

    # 市值 = PIT 总股本 × 未复权 close（与生产/回测同口径）
    fin = load_financial_tables(cache, list(close_raw.columns))
    balance = fin.get("balance_sheet", pd.DataFrame())
    tot_share = build_pit_panel(balance, _cal, "TOT_SHARE").reindex(
        index=close_raw.index, columns=close_raw.columns)
    market_cap = (tot_share * close_raw).where(close_raw.notna())

    industry = None
    try:
        from data.industry import IndustryClassification
        industry = IndustryClassification(cache, level=1).get_industry_panel(
            list(close.columns), close.index)
        if industry.isna().all().all():
            industry = None
    except Exception as e:  # noqa: BLE001
        log.warning("行业面板不可用（%s），分组类因子跳过", str(e)[:80])

    ret = close.pct_change(fill_method=None)
    returns_fwd = ret.shift(-1)
    log.info("日线 %d 日 × %d 股 / 行业%s / 市值非空 %.0f%%",
             close.shape[0], close.shape[1],
             "可用" if industry is not None else "缺失",
             100 * market_cap.notna().mean().mean())
    return ret, market_cap, industry, returns_fwd, close.index, list(close.columns)


def build_entropy(begin: int, end: int) -> pd.DataFrame:
    """5 分钟面板重采样 30 分钟 → 单一成交额占比熵（hs300 池）。"""
    store = MinutePanelStore.open(5, "hs300")
    days = [d for d in store.all_days() if begin <= d <= end]
    if not days:
        raise RuntimeError(f"分钟面板在 {begin}-{end} 无交易日")
    fields = store.load(("close", "volume"), begin_date=days[0], end_date=days[-1])
    panel = entropy_amount_share(fields, days, store.codes, period=5, target=30)
    log.info("分钟熵面板 %d 日 × %d 股（%d→%d）", panel.shape[0], panel.shape[1],
             days[0], days[-1])
    return panel


# ----------------------------------------------------------------------
# 安慰剂对照（L5）
# ----------------------------------------------------------------------
def placebo_check(name: str, panel: pd.DataFrame,
                  returns_panel: pd.DataFrame) -> dict:
    """真实 IC vs 循环平移安慰剂：报 eff_ratio / eff_gap（L2/L5）。"""
    real = ic_stats(calc_ic_series(panel, returns_panel))
    try:
        plc = cyclic_placebo(panel, seed=20261119, name=name)
        plc_ic = ic_stats(calc_ic_series(plc, returns_panel))["ic"]
    except ValueError as e:
        log.warning("安慰剂跳过 %s：%s", name, e)
        return {"real_ic": real["ic"], "plc_ic": np.nan,
                "eff_ratio": np.nan, "eff_gap": np.nan}
    return {"real_ic": real["ic"], "plc_ic": plc_ic,
            **effect_size(real["ic"], plc_ic)}


def coverage_row(panel: pd.DataFrame, returns_panel: pd.DataFrame) -> dict:
    """覆盖率/有效格：先按**本因子自己的**行列裁剪前瞻收益与掩码再统计。

    ``panel_coverage`` 内部直接做 ``x & fwd & mask``，不裁剪会让 pandas 按并集
    展开（熵因子只有 hs300，并集到全 A 后 n_valid 会被 fwd 的非空撑成 8.9M 假数）。
    """
    fwd = returns_panel.reindex(index=panel.index, columns=panel.columns)
    mask = pd.DataFrame(True, index=panel.index, columns=panel.columns)
    return panel_coverage(panel, fwd, mask)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="因子字典候选复现体检入库")
    ap.add_argument("--offline", action="store_true", default=True,
                    help="只读本地缓存（默认）")
    ap.add_argument("--begin", type=int, default=20190101)
    ap.add_argument("--end", type=int, default=20251231)
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--features", default=None)
    ap.add_argument("--no-save", action="store_true", help="只算不入库")
    ap.add_argument("--report-only", action="store_true",
                    help="跳过入库，只重算面板并出体检报告")
    args = ap.parse_args()

    ret, cap, ind, returns_fwd, _dates, _codes = load_daily_context(
        args.begin, args.end)

    panels: dict[str, pd.DataFrame] = {}
    if ind is not None:
        panels["fd_tcc"] = tcc_panel(ret, ind, cap, by_group=False)
        panels["fd_tcc_g"] = tcc_panel(ret, ind, cap, by_group=True)
        panels["fd_grp_abn"] = group_abnormal_panel(ret, ind, cap)
    else:
        log.warning("无行业面板：fd_tcc 仍可算，分组类全部跳过")
        panels["fd_tcc"] = tcc_panel(ret, ind, cap, by_group=False)
    panels["fd_rnbr"] = rnbr_panel(ret)
    panels["fd_ent30"] = build_entropy(args.begin, args.end)

    if args.features:
        want = [x.strip() for x in args.features.split(",") if x.strip()]
        panels = {k: v for k, v in panels.items() if k in want}

    if args.no_save:
        for n, p in panels.items():
            log.info("  %-10s 面板 %s 日 × %d 股 | 非空率 %.0f%%",
                     n, p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
        return

    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset=args.dataset)
    if not args.report_only:
        src_daily = f"paper:build_fd_factors:{args.begin}-{args.end}"
        rows = register_panels(
            lib, panels, {k: FACTOR_DEFS[k] for k in panels}, returns_fwd,
            source=src_daily, check_dup=True,
        )
        for r in rows:
            lib.set_tag(r["name"], family=FAMILY.get(r["name"], "其他"),
                        frequency="日内" if r["name"] == "fd_ent30" else "日频")
        # formula 即中文定义（label 口径）。不标 formula_src 会被
        # ``formula_extract --apply`` 当手写保护起来，源码改了公式永不刷新。
        if rows:
            reg0 = lib._load_registry()
            reg0.loc[reg0["name"].isin([r["name"] for r in rows]), "formula_src"] = "label"
            lib._save_registry(reg0)

    # ---- 体检报告：库指标 + 覆盖 + 安慰剂 ----
    reg = lib.list_all()
    reg = reg[reg["name"].isin(list(panels))].copy()
    extra = []
    for name, panel in panels.items():
        row = {"name": name}
        row.update(coverage_row(panel, returns_fwd))
        row.update(placebo_check(name, panel, returns_fwd))
        extra.append(row)
    rep = reg.merge(pd.DataFrame(extra), on="name", how="left")
    rep = rep[["name", "cov", "cov_eff", "n_valid", "n_days", "ic_mean",
               "ic_ir", "t_stat_nw", "best_sharpe", "best_config",
               "dup_corr_max", "dup_top", "plc_ic", "eff_ratio", "eff_gap"]]
    rep["_abs_ic"] = rep["ic_mean"].abs()
    rep = rep.sort_values("_abs_ic", ascending=False).drop(columns="_abs_ic")

    out_dir = ROOT / "reports" / "factor_dict_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "checkup.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_colwidth", 26,
                           "display.float_format", "{:.4f}".format):
        print("\n==== 因子字典候选体检（全 A 日频 / 熵为 hs300 30min） ====")
        print(rep.to_string(index=False))
    log.info("报告落盘: %s", out)

    n_pass = int((rep["eff_ratio"].abs() >= ADMISSION_GATES["eff_ratio"]).sum())
    log.info("摘要: %d/%d 满足安慰剂效应量比 ≥%.1f×（准入 L5）",
             n_pass, len(rep), ADMISSION_GATES["eff_ratio"])

    record_experiment_safe(
        kind="factor_dict_probe",
        command=" ".join(sys.argv),
        params={"begin": args.begin, "end": args.end, "dataset": args.dataset,
                "n_factors": len(panels)},
        fingerprint=None, result_path=str(out),
        metrics={"n_pass_placebo": n_pass,
                 "best_abs_ic": float(rep["ic_mean"].abs().max())},
        note="飞书因子字典候选复现体检",
    )


if __name__ == "__main__":
    main()
