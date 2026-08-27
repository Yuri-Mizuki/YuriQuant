"""
三段样本外验证（walk-forward）：train 挖 → valid 选 → test 验
=================================================================

严格防泄漏流程：
1. **train 段**（如 2022-2023）：在 train 的收益面板上 exhaustive 挖掘候选公式，
   并按 train IC 排名（挖掘只看 train，绝不见 valid/test）。
2. **valid 段**（2024）：候选公式在 valid 段重算 IC，做显著性筛选（|t|>门槛），
   合成权重也只用 valid IC ——"选因子"只在 valid 上做。
3. **test 段**（2025）：最终验证 —— 用 valid 选出的因子 + 权重合成复合因子，
   在 test 段算 IC 和月频 TopK 回测。test 全程不参与任何选择。

输出：reports/walk_forward/
  - candidates_train.csv      train 挖掘全部候选
  - selected_valid.csv        valid 筛选后的因子
  - summary.csv               train/valid/test 三段 IC 与回测对比
  - report.html               可视化报告

用法:
    python -m scripts.walk_forward --mode both        # exhaustive + GP
    python -m scripts.walk_forward --mode exhaustive  # 只 exhaustive
    python -m scripts.walk_forward --gp-only          # 只 GP
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("walk_forward")

CACHE_ROOT = Path("e:/data/parquet")
OUT_DIR = Path("reports") / "walk_forward"


def load_full_panels(begin: int = 20220101, end: int = 20251231):
    """构建 2022-2025 全区间因子面板（日内/技术面/基本面）与 returns。

    Returns:
        panels: dict[str, pd.DataFrame(date×code)]
        returns_panel: 全区间次日收益
    """
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from data.universe import Universe
    from data.cache_helpers import _pit_universe_codes, _apply_membership_mask
    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    # 日历直接读缓存文件（避免依赖 meta 状态）
    cal_df = pd.read_parquet(CACHE_ROOT / "calendar.parquet")
    cal = [int(x) for x in cal_df["date"].tolist()]
    cal = [c for c in cal if begin <= c <= end]
    if not cal:
        raise RuntimeError("日历为空")
    # PIT 口径（2026-08-13 统一）：历史在册并集池，非在册期间由 mask 剔除
    codes = _pit_universe_codes(uni, "000300.SH", begin, end)

    # 纯离线读 parquet（不走 _refresh_long_table 的增量判断/回源）
    def _read_daily(b, e):
        df = pd.read_parquet(CACHE_ROOT / "daily_hs300.parquet").reset_index()
        df["date"] = df["date"].dt.normalize()
        df = df[(df["date"] >= pd.Timestamp(str(b))) & (df["date"] <= pd.Timestamp(str(e)))]
        df = df[df["code"].isin(codes)]
        df = _apply_membership_mask(
            df.set_index(["date", "code"]), uni, "000300.SH"
        ).reset_index()
        return df

    d = _read_daily(begin, end)
    o = d.pivot(index="date", columns="code", values="open").sort_index()
    h = d.pivot(index="date", columns="code", values="high").sort_index()
    l = d.pivot(index="date", columns="code", values="low").sort_index()
    c = d.pivot(index="date", columns="code", values="close").sort_index()
    v = d.pivot(index="date", columns="code", values="volume").sort_index()
    amt = d.pivot(index="date", columns="code", values="amount").sort_index()
    bf = pd.read_parquet(CACHE_ROOT / "backward_factor.parquet")
    bf = bf.reindex(index=c.index).reindex(columns=c.columns).ffill()
    oa, ha, la, ca = o * bf, h * bf, l * bf, c * bf

    panels: dict[str, pd.DataFrame] = {}
    panels["close"] = c
    panels["open"] = o
    panels["high"] = h
    panels["low"] = l
    panels["volume"] = v
    panels["amount"] = amt
    # 技术面（warmup 前 400 天）
    warm_begin = int((pd.Timestamp(str(begin)) - pd.Timedelta(days=400)).strftime("%Y%m%d"))
    dw = _read_daily(warm_begin, end)
    cw = dw.pivot(index="date", columns="code", values="close").sort_index()
    hw = dw.pivot(index="date", columns="code", values="high").sort_index()
    lw = dw.pivot(index="date", columns="code", values="low").sort_index()
    ow = dw.pivot(index="date", columns="code", values="open").sort_index()
    vw = dw.pivot(index="date", columns="code", values="volume").sort_index()
    bfw = pd.read_parquet(CACHE_ROOT / "backward_factor.parquet")
    bfw = bfw.reindex(index=cw.index).reindex(columns=cw.columns).ffill()
    caw, haw, law, oaw = cw * bfw, hw * bfw, lw * bfw, ow * bfw
    from factor.technical import calc_indicators
    for code in codes:
        if code not in caw.columns or caw[code].dropna().empty:
            continue
        res = calc_indicators(caw[code], haw[code], law[code], oaw[code], vw[code])
        for k, s in res.items():
            panels.setdefault(k, pd.DataFrame(index=cw.index, columns=cw.columns))
            panels[k][code] = s
    for k in list(panels.keys()):
        if k in ("close", "open", "high", "low", "volume", "amount"):
            continue
        panels[k] = panels[k].reindex(index=c.index)

    # 日内（2022-2025 分钟线；未覆盖则跳过）
    try:
        from scripts.build_intraday_factors import build_features, _minute_frame
        mk = pd.read_parquet(CACHE_ROOT / "min5_hs300.parquet")
        if not mk.empty:
            kt = mk.index.get_level_values("kline_time")
            mk = mk[(kt >= pd.Timestamp(str(begin))) & (kt <= pd.Timestamp(str(end)) + pd.Timedelta(days=1))]
            if not mk.empty:
                status = pd.read_parquet(CACHE_ROOT / "history_stock_status.parquet")
                mf = _minute_frame(mk, status)
                dd = pd.read_parquet(CACHE_ROOT / "daily_hs300.parquet")   # (date, code) MultiIndex
                dd = dd[dd.index.get_level_values("code").isin(codes)]
                dt0 = dd.index.get_level_values("date")
                dd = dd[(dt0 >= pd.Timestamp(str(begin))) & (dt0 <= pd.Timestamp(str(end)))]
                intra = build_features(mf, dd, status)
                for k, pnl in intra.items():
                    panels[k] = pnl.reindex(index=c.index)
                log.info("日内因子 %d 个", sum(1 for k in intra if k in panels))
    except Exception as e:
        log.warning("日内因子跳过: %s", e)

    # 基本面
    try:
        from data.cache_helpers import load_financial_tables
        from scripts.build_fundamental_factors import build_factor_panels
        fin = load_financial_tables(cache, codes)
        fund = build_factor_panels(daily if "daily" in dir() else d, cal,
                                   fin["income"], fin["balance_sheet"], fin["cash_flow"],
                                   fin["equity_structure"], fin["dividend"],
                                   fin["share_holder"], fin["holder_num"])
        for k, pnl in fund.items():
            panels[k] = pnl.reindex(index=c.index)
        log.info("基本面因子 %d 个", len(fund))
    except Exception as e:
        log.warning("基本面因子跳过: %s", e)

    returns_panel = c.pct_change().shift(-1)
    return panels, returns_panel


def slice_panel(panels: dict[str, pd.DataFrame], begin: str, end: str) -> dict[str, pd.DataFrame]:
    return {k: p.loc[begin:end] for k, p in panels.items() if begin in p.index or True}


def main():
    parser = argparse.ArgumentParser(description="三段样本外验证")
    parser.add_argument("--mode", default="exhaustive", choices=["exhaustive", "both", "gp"])
    parser.add_argument("--windows", default="5,10,20,60")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--valid-t", type=float, default=1.5, help="valid 段显著门槛 |t|")
    parser.add_argument("--merge", action="store_true",
                        help="方案A：train/valid 合并为 2022-2024 一段（挖掘+筛选都在内），test 2025 验证")
    parser.add_argument("--k", type=int, default=50, help="回测 TopK")
    parser.add_argument("--freq", default="M", choices=["D", "W", "M"])
    # ---- GP 分支（--mode gp / both）----
    parser.add_argument("--gp-pop", type=int, default=200, help="GP 种群规模")
    parser.add_argument("--gp-gen", type=int, default=10, help="GP 迭代代数")
    parser.add_argument("--gp-min-depth", type=int, default=2)
    parser.add_argument("--gp-max-depth", type=int, default=5)
    parser.add_argument("--gp-tournament", type=int, default=5)
    parser.add_argument("--gp-train-frac", type=float, default=0.8,
                        help="GP 内部样本外切分（train 段内再留 20% 防进化过拟合）")
    parser.add_argument("--gp-monthly-weight", type=float, default=0.5, help="月频 IC 融合权重")
    parser.add_argument("--gp-fitness", default="tstat",
                        choices=["tstat", "rankic_mean", "mutual_info", "top_excess"])
    parser.add_argument("--gp-jobs", type=int, default=1, help="GP 种群并行进程数")
    parser.add_argument("--gp-seed", type=int, default=0)
    parser.add_argument("--gp-htai", action="store_true",
                        help="GP 华泰复现口径（环内 MAD+五因子中性化+月频目标；中性化面板 offline 构建）")
    parser.add_argument("--gp-verbose", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("加载 2022-2025 全区间面板 ...")
    panels, returns_panel = load_full_panels(20220101, 20251231)
    n_days = len(returns_panel)
    log.info("全区间 %d 交易日 × %d 股, 因子 %d 个", n_days,
             len(returns_panel.columns), len(panels))

    # 切分：方案B（默认）train 2022-23 / valid 2024 / test 2025；
    # 方案A（--merge）train=valid 2022-2024 / test 2025
    all_dates = returns_panel.index
    test_dates = all_dates[(all_dates >= "2025-01-01") & (all_dates <= "2025-12-31")]
    if args.merge:
        train_dates = all_dates[(all_dates >= "2022-01-01") & (all_dates <= "2024-12-31")]
        valid_dates = train_dates   # valid 段 = train 段（方案 A：同一段）
        log.info("[方案A] train+valid 合并 %d 日 (%s~%s) / test %d 日",
                 len(train_dates), train_dates[0].date(), train_dates[-1].date(),
                 len(test_dates))
    else:
        train_dates = all_dates[(all_dates >= "2022-01-01") & (all_dates <= "2023-12-31")]
        valid_dates = all_dates[(all_dates >= "2024-01-01") & (all_dates <= "2024-12-31")]
        log.info("[方案B] train %d 日 (%s~%s) / valid %d 日 / test %d 日",
                 len(train_dates), train_dates[0].date(), train_dates[-1].date(),
                 len(valid_dates), len(test_dates))

    # ---- Step 1: train 段挖掘（PIT 并集池面板，见 load_full_panels）----
    from factor.preprocessing import standardize_zscore

    features = ["close", "open", "high", "low", "volume", "amount"]
    # 技术面/日内/基本面按类别均衡选取（避免只取插入序前 8 个遗漏日内）
    tech_feats = [k for k in panels if k in ("sar_dev", "rsi_12", "kdj_j", "macd_hist")]
    intra_feats = [k for k in panels if k.startswith(("close30", "intraday", "overnight", "open30", "first_bar", "am_"))][:6]
    fund_feats = [k for k in panels if k in ("ep_ttm", "bp", "gross_margin", "np_growth_yoy")]
    extra_feats = [k for k in (tech_feats + intra_feats + fund_feats)
                   if k not in features and panels[k].notna().mean().mean() > 0.5]
    features = features + extra_feats
    windows = tuple(int(x) for x in args.windows.split(",") if x.strip())

    train_returns = returns_panel.loc[train_dates]
    train_panel = {k: p.loc[train_dates] for k, p in panels.items()}
    log.info("train 段挖掘: %d 特征, windows=%s", len(features), windows)

    result = None
    if args.mode in ("gp", "both"):
        from factor.genetic_mining import run_gp_mining
        gp_kwargs = dict(population=args.gp_pop, generations=args.gp_gen,
                         min_depth=args.gp_min_depth, max_depth=args.gp_max_depth,
                         tournament=args.gp_tournament, train_frac=args.gp_train_frac,
                         monthly_weight=args.gp_monthly_weight, fitness_mode=args.gp_fitness,
                         n_jobs=args.gp_jobs, seed=args.gp_seed, verbose=args.gp_verbose)
        if args.gp_htai:
            from scripts.mine_factors import _build_htai_neutral_panels
            neutral = _build_htai_neutral_panels(train_panel, real=False)
            gp_kwargs.update(htai=True, neutral_panels=neutral if neutral else None)
            log.info("GP htai 口径：中性化协变量=%s", list(neutral.keys()))
        log.info("GP train 段挖掘: pop=%d gen=%d seed=%d fitness=%s", args.gp_pop,
                 args.gp_gen, args.gp_seed, args.gp_fitness)
        gp_df, _ = run_gp_mining(train_panel, train_returns, features=features,
                                 windows=windows, **gp_kwargs)
        if gp_df is None or gp_df.empty:
            log.error("GP 未挖出有效因子")
            sys.exit(1)
        gp_res = gp_df.rename(columns={"formula": "name"})
        gp_res["significant"] = gp_res["t_stat"].abs() >= 2.0
        result = gp_res[["name", "ic_mean", "ic_std", "ir", "t_stat", "significant"]].copy()
        result["n"] = gp_res["n"].astype(int)
        log.info("GP hof 因子 %d 个（|t|>=2: %d）", len(gp_res),
                 int((gp_res["t_stat"].abs() >= 2.0).sum()))

    if args.mode in ("exhaustive", "both"):
        from factor.mining import dedup_by_formula, evaluate_candidates, generate_candidates
        cands = dedup_by_formula(generate_candidates(features=features, windows=windows,
                                                      depth=args.depth))
        if len(cands) > 1000:
            log.info("候选 %d 个，截取前 1000 个控制评估时间", len(cands))
            cands = cands[:1000]
        log.info("候选公式 %d 个（并行评估 n_jobs=4, detail_n=0 提速）", len(cands))
        res_ex = evaluate_candidates(cands, train_panel, train_returns,
                                     fdr_q=args.fdr_q, detail_n=0, robust=True,
                                     n_jobs=4)
        res_ex = res_ex[["name", "ic_mean", "ic_std", "ir", "t_stat", "significant"]]
        if args.mode == "both":
            result = pd.concat([result, res_ex], ignore_index=True).drop_duplicates("name")
        else:
            result = res_ex
        log.info("train 显著(FDR): %d 个", int(result["significant"].sum()))
    result.to_csv(OUT_DIR / "candidates_train.csv", index=False)

    # ---- Step 2: valid 段筛选 + 合成权重 ----
    # 方案A（merge）：valid= train 段，直接用 train 评估结果筛选（同一段内挖+选）
    # 方案B（默认）：候选在独立 valid 段（2024）重算 IC，做第二道防泄漏筛选
    from factor.formula import formula_builder
    from research.factor_analysis import calc_ic_series
    topk = result.head(60)   # 前 60 个候选进筛选

    if args.merge:
        # 方案A：直接按 train |t| 选 top15（同段挖选，不做独立 valid 复检）
        keep = topk.copy()
        keep = keep.reindex(keep["t_stat"].abs().sort_values(ascending=False).index).head(15)
        keep = keep.rename(columns={"t_stat": "valid_t", "ic_mean": "valid_ic", "ir": "valid_ir"})
        keep["formula"] = keep["name"]
        keep[["formula", "valid_ic", "valid_ir", "valid_t"]].to_csv(
            OUT_DIR / "selected_valid.csv", index=False)
        log.info("[方案A] 同段筛选: %d 候选 → 按 train |t| 取 15 个", len(topk))
    else:
        valid_returns = returns_panel.loc[valid_dates]
        valid_panel = {k: p.loc[valid_dates] for k, p in panels.items()}
        sel_rows = []
        for _, row in topk.iterrows():
            formula = row.get("formula", row.get("name"))
            if not formula:
                continue
            try:
                fp = formula_builder(formula, features=features)(valid_panel)
            except Exception:
                continue
            if fp is None or fp.empty:
                continue
            from research.factor_analysis import calc_ic_series
            ic = calc_ic_series(fp, valid_returns).dropna()
            if len(ic) < 20:
                continue
            m, s = float(ic.mean()), float(ic.std())
            t = m / (s / np.sqrt(len(ic))) if s > 0 else 0.0
            sel_rows.append({"formula": formula, "train_t": row.get("t_stat", np.nan),
                             "valid_ic": m, "valid_ir": m / s if s > 0 else 0.0,
                             "valid_t": t, "n": len(ic)})
        sel = pd.DataFrame(sel_rows)
        if sel.empty:
            log.error("valid 段无有效候选")
            sys.exit(1)
        sel = sel.reindex(sel["valid_t"].abs().sort_values(ascending=False).index)
        keep = sel[sel["valid_t"].abs() >= args.valid_t].head(15)
        log.info("[方案B] valid 筛选: %d 候选 → %d 个过 |t|>=%.1f", len(sel), len(keep), args.valid_t)
        keep.to_csv(OUT_DIR / "selected_valid.csv", index=False)

    # ---- Step 3: test 段验证 ----
    test_returns = returns_panel.loc[test_dates]
    test_panel = {k: p.loc[test_dates] for k, p in panels.items()}
    if args.merge:
        valid_returns = train_returns   # 方案A valid 段 = train 段
    from factor.synthesis import CompositeInput, synthesize_ic_weighted
    from research.factor_analysis import calc_ir

    comps = []
    for _, row in keep.iterrows():
        fp = formula_builder(row["formula"], features=features)(panels)
        comps.append(CompositeInput(name=row["formula"],
                                    panel=standardize_zscore(fp),
                                    ic=row["valid_ic"], ir=row["valid_ir"]))
    # 合成权重来自 valid IC
    comp_all = synthesize_ic_weighted(comps, weight_by="ic_abs")
    # 各段 IC（方案A 下 train/valid 同段，只输出 train+test 两行）
    summary_rows = []
    segments = [("train", train_dates, train_returns),
                ("valid", valid_dates, valid_returns),
                ("test", test_dates, test_returns)]
    if args.merge:
        segments = [("train+valid", train_dates, train_returns),
                    ("test", test_dates, test_returns)]
    for label, dts, rts in segments:
        cseg = comp_all.loc[dts]
        ic = calc_ic_series(cseg, rts).dropna()
        summary_rows.append({"阶段": label, "IC": ic.mean(),
                             "IR": calc_ir(ic), "t": ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0,
                             "n_days": len(ic)})

    # test 回测
    from backtest import VectorBacktest
    from strategy.examples import TopKLongOnly
    bt = VectorBacktest(TopKLongOnly(k=args.k), rebalance_freq=args.freq)
    res = bt.run(comp_all.loc[test_dates], test_returns)
    m = res.metrics()
    summary_rows.append({"阶段": "test回测", "IC": np.nan, "IR": np.nan, "t": np.nan,
                         "n_days": args.k})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\n===== 三段验证汇总（合成因子）=====")
    with pd.option_context("display.width", 120, "display.float_format", lambda v: f"{v:.4f}"):
        print(summary.to_string(index=False))
    print(f"test 回测（Top{args.k} {args.freq} 纯多）: sharpe={m.get('sharpe'):.2f} "
          f"年化={m.get('annual_return')*100:+.1f}% 回撤={m.get('max_drawdown')*100:.1f}%")
    # 净值
    eq = res.equity_curve
    eq.to_csv(OUT_DIR / "equity_test.csv")

    # 可视化报告（docstring 声明产出 report.html，早期实现缺失——2026-08-05 补齐）
    try:
        from research.html_report import generate_html_report
        summary_txt = " | ".join(
            f"{r['阶段']}: IC={r['IC']:+.4f} IR={r['IR']:.2f}"
            for _, r in summary.iterrows() if pd.notna(r.get("IC"))
        )
        meta = (f"{summary_txt} · test 回测 Top{args.k} {args.freq} 纯多: "
                f"sharpe={m.get('sharpe', 0):.2f} 年化={m.get('annual_return', 0) * 100:+.1f}% "
                f"回撤={m.get('max_drawdown', 0) * 100:.1f}%")
        generate_html_report(
            {f"test 段合成因子（Top{args.k} {args.freq} 纯多）": res}, None,
            output_path=OUT_DIR / "report.html",
            title="三段样本外验证报告（walk-forward）",
            meta=meta,
        )
        log.info("可视化报告: %s", OUT_DIR / "report.html")
    except Exception as e:
        log.warning("report.html 生成失败（不影响 CSV 输出）: %s", e)

    log.info("结果: %s", OUT_DIR)


if __name__ == "__main__":
    main()
