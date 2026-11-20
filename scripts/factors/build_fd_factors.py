"""飞书《因子字典》候选因子 → 因子库体检入库
=============================================

对 ``factor/factor_dict_factors.py`` 里的 15 个候选跑与 ``build_intraday_stat_factors``
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
    cs_spread_panel,
    csad_std21_panel,
    entropy_amount_share,
    fear_panel,
    group_abnormal_panel,
    illiq_panel,
    mmd_asym_panel,
    net_support_panel,
    rnbr_panel,
    smart_vwap_panel,
    tcc_panel,
    turnover_resid_vol_panel,
    umr_panel,
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
    "fd_csad21": "CSAD模型 std_21 = std_21( 过去120日相关性Top10板块内 |r_i − r_self| 的截面均值 )",
    "fd_mmd21": "收益率不对称性(日频近似) = mean_21( MMD(过去60日日收益, −x+2μ) )",
    "fd_tr_resid": "特质换手波动率 = std_21( 换手率对[市场/SMB/HML/MOM换手]时序回归的残差 )",
    "fd_illiq_ds": "非流动性(DS版) = |0.6·(H−L)/O + 0.4·(O_t/C_{t−1}−1)|^1.5 / (成交额^0.7 + 5e6)",
    "fd_umr_tr": "UMR2.0(换手风险) = mean_21( 超额收益 × (mean_21(换手) − 换手) )",
    "fd_umr_trv": "UMR2.0(真实波动风险) = mean_21( 超额收益 × (mean_21(TR) − TR) )",
    "fd_fear_std": "原始惊恐因子 = std_21( 惊恐度 × 个股收益 )，惊恐度=|r−r_m|/(|r|+|r_m|+0.1)",
    "fd_hf_cs": "CS价差(分钟) = std_21( 日内 Corwin-Schultz 价差估计的均值 )",
    "fd_smart_vwap": "聪明钱(跟随vwap) = 跟随时刻vwap / 全日vwap（成交量Top10时刻的后一根）",
    "fd_net_support": "净支撑成交量 = (支撑量 − 阻力量) / 总股本，支撑=分钟收盘价低于日均价",
}

FAMILY = {"fd_tcc": "波动率", "fd_tcc_g": "波动率", "fd_grp_abn": "波动率",
          "fd_rnbr": "波动率", "fd_ent30": "日内",
          "fd_csad21": "波动率", "fd_mmd21": "波动率", "fd_tr_resid": "流动性",
          "fd_illiq_ds": "流动性", "fd_umr_tr": "量价", "fd_umr_trv": "量价",
          "fd_fear_std": "波动率",
          "fd_hf_cs": "流动性", "fd_smart_vwap": "资金流", "fd_net_support": "量价"}

#: 从 5 分钟面板出发的因子（hs300 池、2022 起；其余走日线全 A）
MINUTE_FACTORS = frozenset({"fd_ent30", "fd_hf_cs", "fd_smart_vwap",
                            "fd_net_support"})


# ----------------------------------------------------------------------
# 数据装载
# ----------------------------------------------------------------------
def load_daily_context(begin: int, end: int, *, with_prices: bool = False):
    """全 A 日线 → ``(ret, market_cap, industry, returns_fwd, dates, codes, extra)``。

    ``extra`` 含 ``turnover`` / ``pb`` / ``tot_share`` / ``adj_close`` / ``bf``；
    ``with_prices=True`` 时再补 ``open`` / ``high`` / ``low`` / ``amount``
    （非流动性、UMR 真实波动用，按需装载避免白占内存）。
    换手率的口径是 **成交量 / PIT 总股本**（缓存无自由流通股，是总股本口径的
    近似）；``pb = 市值 / PIT 股东权益(含少数股东)``。
    """
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
        close, bf_adj = close_raw, None
    else:
        # 对齐到日线面板再 ffill：原始 bf 索引含缓存全历史（8767 日），
        # 直接拿它做乘法会把面板膨胀成并集索引
        bf_adj = bf.reindex(index=close_raw.index,
                            columns=close_raw.columns).ffill()
        close = close_raw * bf_adj

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

    # 特质换手波动率所需：换手率（成交量/PIT 总股本）+ pb（市值/PIT 股东权益）
    equity = build_pit_panel(balance, _cal, "TOT_SHARE_EQUITY_INCL_MIN_INT") \
        .reindex(index=close_raw.index, columns=close_raw.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        turnover = (volume / tot_share.where(tot_share > 0)) \
            .where(volume.notna())
        pb = (market_cap / equity.where(equity > 0)).where(market_cap.notna())
    extra = {"turnover": turnover, "pb": pb, "tot_share": tot_share,
             "adj_close": close, "bf": bf_adj}
    if with_prices:
        for col in ("open", "high", "low", "amount"):
            extra[col] = _panel(col)

    log.info("日线 %d 日 × %d 股 / 行业%s / 市值非空 %.0f%% / 换手非空 %.0f%%",
             close.shape[0], close.shape[1],
             "可用" if industry is not None else "缺失",
             100 * market_cap.notna().mean().mean(),
             100 * turnover.notna().mean().mean())
    return ret, market_cap, industry, returns_fwd, close.index, list(close.columns), extra


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


def build_minute_factors(begin: int, end: int, want: set[str] | None,
                         tot_share: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """5 分钟面板 → CS 价差 / 聪明钱 vwap / 净支撑成交量（hs300 池）。"""
    store = MinutePanelStore.open(5, "hs300")
    days = [d for d in store.all_days() if begin <= d <= end]
    if not days:
        raise RuntimeError(f"分钟面板在 {begin}-{end} 无交易日")
    pick = (lambda n: want is None or n in want)
    fields_needed: set[str] = set()
    if pick("fd_hf_cs"):
        fields_needed |= {"high", "low"}
    if pick("fd_smart_vwap"):
        fields_needed |= {"volume", "amount"}
    if pick("fd_net_support"):
        fields_needed |= {"close", "volume"}
    fields = store.load(tuple(sorted(fields_needed)),
                        begin_date=days[0], end_date=days[-1])
    out: dict[str, pd.DataFrame] = {}
    if pick("fd_hf_cs"):
        out["fd_hf_cs"] = cs_spread_panel(fields, days, store.codes)
    if pick("fd_smart_vwap"):
        out["fd_smart_vwap"] = smart_vwap_panel(fields, days, store.codes)
    if pick("fd_net_support"):
        dates = pd.to_datetime([str(d) for d in days])
        shares = tot_share.reindex(index=dates, columns=list(store.codes)) \
            .ffill().to_numpy(dtype=float)
        out["fd_net_support"] = net_support_panel(
            fields, days, store.codes, shares)
    for n, p in out.items():
        log.info("  %-14s 面板 %d 日 × %d 股 | 非空率 %.0f%%", n,
                 p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
    return out


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

    want = ({x.strip() for x in args.features.split(",") if x.strip()}
            if args.features else None)

    def need(*names: str) -> bool:
        return want is None or any(n in want for n in names)

    ret, cap, ind, returns_fwd, _dates, _codes, extra = load_daily_context(
        args.begin, args.end, with_prices=need("fd_illiq_ds", "fd_umr_trv"))

    # 只算被点名的因子：CSAD 每日一个全截面相关矩阵、MMD 是逐日核矩阵，
    # 没必要为跑 3 个因子先把整套（含 30 分钟熵）算一遍
    panels: dict[str, pd.DataFrame] = {}
    if need("fd_tcc"):
        panels["fd_tcc"] = tcc_panel(ret, ind, cap, by_group=False)
    if ind is not None:
        if need("fd_tcc_g"):
            panels["fd_tcc_g"] = tcc_panel(ret, ind, cap, by_group=True)
        if need("fd_grp_abn"):
            panels["fd_grp_abn"] = group_abnormal_panel(ret, ind, cap)
    elif need("fd_tcc_g", "fd_grp_abn"):
        log.warning("无行业面板：fd_tcc_g / fd_grp_abn 跳过")
    if need("fd_rnbr"):
        panels["fd_rnbr"] = rnbr_panel(ret)
    if need("fd_csad21"):
        panels["fd_csad21"] = csad_std21_panel(ret)
    if need("fd_mmd21"):
        panels["fd_mmd21"] = mmd_asym_panel(ret)
    if need("fd_tr_resid"):
        panels["fd_tr_resid"] = turnover_resid_vol_panel(
            extra["turnover"], extra["pb"], cap, ret)

    # 非流动性 / UMR / 惊恐：价格类，复权后再算（比值在乘性复权下不变）
    if need("fd_illiq_ds", "fd_umr_tr", "fd_umr_trv", "fd_fear_std"):
        bf = extra["bf"]
        mul = (lambda x: x) if bf is None else (lambda x: x * bf)
        if need("fd_illiq_ds"):
            panels["fd_illiq_ds"] = illiq_panel(
                mul(extra["open"]), mul(extra["high"]), mul(extra["low"]),
                extra["adj_close"], extra["amount"])
        if need("fd_umr_tr", "fd_umr_trv"):
            mc = cap.reindex_like(ret)
            w = mc.div(mc.sum(axis=1, min_count=1), axis=0)
            excess = ret.sub((ret * w).sum(axis=1), axis=0)
        if need("fd_umr_tr"):
            panels["fd_umr_tr"] = umr_panel(excess, extra["turnover"])
        if need("fd_umr_trv"):
            high, low = mul(extra["high"]), mul(extra["low"])
            prev = extra["adj_close"].shift(1)
            tr = (np.maximum(np.maximum(high - low, (high - prev).abs()),
                             (low - prev).abs()) / prev)
            panels["fd_umr_trv"] = umr_panel(excess, tr)
        if need("fd_fear_std"):
            panels["fd_fear_std"] = fear_panel(ret)
    if need("fd_ent30"):
        panels["fd_ent30"] = build_entropy(args.begin, args.end)
    if need("fd_hf_cs", "fd_smart_vwap", "fd_net_support"):
        panels.update(build_minute_factors(args.begin, args.end, want,
                                           extra["tot_share"]))

    if args.no_save:
        for n, p in panels.items():
            log.info("  %-10s 面板 %s 日 × %d 股 | 非空率 %.0f%%",
                     n, p.shape[0], p.shape[1], 100 * p.notna().mean().mean())
        return

    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset=args.dataset)
    if not args.report_only:
        src_daily = f"paper:build_fd_factors:{args.begin}-{args.end}"
        # 收益面板不传：走库标准口径（复权收盘 1 日远期 + 可交易掩码）。
        # 自带一份 pct_change 只会被记成 caller_supplied，卡住晋升第 2 关。
        rows = register_panels(
            lib, panels, {k: FACTOR_DEFS[k] for k in panels}, None,
            source=src_daily, check_dup=True, families=FAMILY,
        )
        for r in rows:
            lib.set_tag(r["name"],
                        frequency="日内" if r["name"] in MINUTE_FACTORS else "日频")
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
    # 按因子名合并：分轮跑（--features 只跑子集）时不把上一轮的体检行冲掉
    if out.exists():
        old = pd.read_csv(out, encoding="utf-8-sig")
        rep = pd.concat([old[~old["name"].isin(rep["name"])], rep],
                        ignore_index=True)
        rep = rep.sort_values("name")
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
