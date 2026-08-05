"""
因子挖掘 CLI
============

跑通「算子空间 + 候选生成 + 批量 IC 检验 + 显著性筛选」闭环。

用法:
    python scripts/mine_factors.py                         # mock 数据（自带信号注入）
    python scripts/mine_factors.py --real                  # 真实数据（需 SDK + 先 update_data）
    python scripts/mine_factors.py --depth 1 --windows 5,10,20
    python scripts/mine_factors.py --top 30 --out reports/mining.csv

mock 模式注入 AR(1) 收益（动量有预测力），用于验证挖掘流程能找到显著因子。
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("mine_factors")


# ---------------------------------------------------------------------------
# Mock 面板（注入动量信号 + 一个合成财务字段，验证挖掘流程）
# ---------------------------------------------------------------------------
def gen_mock_panel_with_signal(n_days: int = 400, n_codes: int = 50, seed: int = 0) -> dict[str, pd.DataFrame]:
    """AR(1) 收益注入动量信号：rets[t] = phi*rets[t-1] + noise，使 ts_mean/momentum 类因子有正 IC。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    phi = 0.25
    rets = np.zeros((n_days, n_codes))
    for t in range(1, n_days):
        rets[t] = phi * rets[t - 1] + rng.normal(0, 0.02, n_codes)

    base = 10.0 + rng.uniform(0, 50, n_codes)
    close = pd.DataFrame(base * np.exp(np.cumsum(rets, axis=0)), idx, codes)
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_codes)))
    high = np.maximum(close.values, open_.values) * (1 + np.abs(rng.normal(0, 0.005, (n_days, n_codes))))
    low = np.minimum(close.values, open_.values) * (1 - np.abs(rng.normal(0, 0.005, (n_days, n_codes))))
    volume = pd.DataFrame(rng.lognormal(12, 0.5, (n_days, n_codes)), idx, codes)
    amount = volume * close

    # 合成"财务字段"：慢漂移的 OPERA_REV，PIT 化（每 60 日更新一次，中间 ffill）
    rev_raw = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_codes)), axis=0)), idx, codes)
    rev = rev_raw.copy()
    rev.loc[np.arange(n_days) % 60 != 0] = np.nan
    rev = rev.ffill()

    return {
        "close": close, "open": pd.DataFrame(open_, idx, codes),
        "high": pd.DataFrame(high, idx, codes), "low": pd.DataFrame(low, idx, codes),
        "volume": volume, "amount": amount, "OPERA_REV": rev,
    }


# ---------------------------------------------------------------------------
# 真实面板（从缓存 + 财务 PIT 构建）
# ---------------------------------------------------------------------------
def build_real_panel(cfg: dict, begin: int, end: int) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    from data.cache import DataCache
    from data.datasource import create_datasource
    from data.financials import build_pit_panel
    from data.universe import Universe

    ds = create_datasource()
    cache = DataCache(ds)
    uni = Universe(cache)
    index_code = cfg["universe"]["index_code"]
    cal = cache.get_calendar(begin, end)
    target_date = end if end else cal[-1]
    codes = uni.get_constituent(index_code, target_date)
    log.info("成分股数量: %d", len(codes))

    kline = cache.get_daily_kline(codes, begin, target_date)
    close = kline["close"].unstack("code")
    high = kline["high"].unstack("code")
    low = kline["low"].unstack("code")
    open_ = kline["open"].unstack("code")
    volume = kline["volume"].unstack("code")
    amount = kline["amount"].unstack("code")

    # 后复权
    adjust = cfg.get("universe", {}).get("adjust", "backward")
    if adjust == "backward":
        backward = cache.get_backward_factor(codes).reindex(index=close.index, columns=close.columns).ffill()
        close, high, low, open_ = close * backward, high * backward, low * backward, open_ * backward

    panel = {"close": close, "open": open_, "high": high, "low": low, "volume": volume, "amount": amount}

    # 财务字段 PIT 展开（公告日对齐，无未来函数）
    log.info("构建财务 PIT 面板 ...")
    income = cache.get_income(codes)
    if not income.empty:
        for field in ("OPERA_REV", "NET_PRO_INCL_MIN_INT_INC", "BASIC_EPS"):
            if field in income.columns:
                panel[field] = build_pit_panel(income, cal, field).reindex(close.index)
    balance = cache.get_balance_sheet(codes)
    if not balance.empty:
        for field in ("TOTAL_ASSETS", "TOT_SHARE_EQUITY_EXCL_MIN_INT"):
            if field in balance.columns:
                panel[field] = build_pit_panel(balance, cal, field).reindex(close.index)

    returns_panel = close.pct_change().shift(-1)
    return panel, returns_panel


# ---------------------------------------------------------------------------
# 因子库集成：下一轮迭代（把库内因子作为特征）/ 入库
# ---------------------------------------------------------------------------
def _gp_preprocess_features(panel: dict[str, pd.DataFrame], cfg: dict | None = None
                            ) -> dict[str, pd.DataFrame]:
    """GP 特征预处理（P0-③，2026-08-04）：消除价格/成交额的风格主导。

    实测（2026-08-04 长历史 GP）：close/amount 等原始特征在 4 年历史上 |t| 达 5-7
    （A 股低价股/反转风格效应），形成"适应度高原"，深树永远追不上 → GP 退化为
    浅层特征选择器。修复：进 GP 前先做
        1) 截面 zscore（去掉量纲/价格水平，保留横截面相对排序信息）
        2) 行业市值中性化（真实数据有行业/市值面板时；用对数市值 + 行业哑变量
           回归取残差，剥离风格暴露）
    注意：close 是构造 returns 的基准，必须保持原始值返回（returns 用原始 close
    计算）；只对**参与进化的特征**做预处理。
    """
    from factor.preprocessing import neutralize, standardize_zscore
    out: dict[str, pd.DataFrame] = {}

    # 行业面板（申万一级，PIT 事件表 → 每日截面哑变量）
    industry_panel = None
    market_cap_panel = None
    try:
        from data.cache import DataCache
        from data.datasource import create_datasource
        ds = create_datasource()
        cache = DataCache(ds)
        codes = list(next(iter(panel.values())).columns)
        ind = cache.get_industry_classification(level=1)
        if not ind.empty:
            ind["in_date"] = pd.to_datetime(ind["in_date"], errors="coerce")
            ind["out_date"] = pd.to_datetime(ind["out_date"], errors="coerce")
            dates = next(iter(panel.values())).index
            # 逐日：code -> industry 映射
            rows = {}
            for d in dates:
                m = ind[(ind["in_date"] <= d) & (ind["out_date"].fillna(pd.Timestamp.max) >= d)]
                rows[d] = {r["code"]: r["industry_name"] for _, r in m.iterrows() if r["code"] in codes}
            industry_panel = pd.DataFrame(rows).T.reindex(columns=codes)
        # 市值：TOT_SHARE × close（PIT 股本 × 原始收盘价）
        balance = cache.get_balance_sheet(codes)
        if not balance.empty and "TOT_SHARE" in balance.columns and "close" in panel:
            from data.financials import build_pit_panel
            from config import Config
            c = Config.get()
            b = c["fetch"]["begin_date"]
            e = c.get("end_date") or c["fetch"]["end_date"]
            cal = cache.get_calendar(b, e if e else b)
            ts = build_pit_panel(balance, cal, "TOT_SHARE").reindex(
                index=panel["close"].index, columns=panel["close"].columns)
            market_cap_panel = ts * panel["close"]
    except Exception as exc:
        log.warning("GP 特征中性化面板构建失败（跳过中性化，仅 zscore）: %s", exc)

    for name, fp in panel.items():
        if name == "close":     # close 是 returns 基准，保持原始
            out[name] = fp
            continue
        z = standardize_zscore(fp)
        if industry_panel is not None or market_cap_panel is not None:
            z = neutralize(z, market_cap_panel=market_cap_panel,
                           industry_panel=industry_panel)
            z = standardize_zscore(z)
        out[name] = z
    return out
def _merge_library_features(panel: dict[str, pd.DataFrame], dataset: str | None = None) -> dict[str, pd.DataFrame]:
    """把因子库里已存的因子面板作为新特征并入当前面板（实现复合因子参与下一轮迭代）。"""
    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset=dataset)
    feats = lib.load_library_features()
    if not feats:
        log.warning("因子库为空，--use-library 跳过")
        return panel
    base_idx = panel["close"].index
    base_cols = panel["close"].columns
    n_add = 0
    for name, fp in feats.items():
        if name in panel:
            continue
        merged = fp.reindex(index=base_idx, columns=base_cols).ffill().bfill()
        if merged.notna().any().any():
            panel[name] = merged
            n_add += 1
    log.info("注入因子库特征 %d 个，参与下一轮挖掘", n_add)
    return panel


def _save_library(result: pd.DataFrame, panel: dict[str, pd.DataFrame],
                  returns_panel: pd.DataFrame, args, is_gp: bool,
                  dataset: str | None = None) -> int:
    """把 top-K 候选因子入库（预计算 IC + 多套回测）。返回入库数量。"""
    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset=dataset)
    topk = result.head(args.lib_top)
    saved = 0
    if is_gp:
        # 统一公式解析器还原（支持 GP 窗口编名语法，不依赖 deap pset / 模块级 prim_map）
        from factor.formula import formula_builder
        feats = list(panel.keys())
        for _, row in topk.iterrows():
            formula = row.get("formula", row.get("name"))
            if not formula:
                continue
            try:
                fp = formula_builder(formula, features=feats)(panel)
            except Exception as e:
                log.warning("GP 因子入库失败 %s: %s", formula, e)
                continue
            if fp is None or fp.empty:
                continue
            lib.register(formula, fp, returns_panel, kind="raw",
                         formula=formula, source="gp:mine_factors")
            saved += 1
    else:
        from factor.synthesis import build_components
        comps = build_components(topk, panel, features=list(panel.keys()),
                                 windows=tuple(int(x) for x in args.windows.split(",") if x.strip()),
                                 depth=args.depth)
        for c in comps:
            lib.register(c.name, c.panel, returns_panel, kind="raw",
                         formula=c.name, source="mining:mine_factors")
            saved += 1
    log.info("已入库 %d 个因子", saved)
    return saved


def _derive_dataset(args) -> str:
    """不显式给 --library-dataset 时，按数据来源推导数据集名（按数据集分库根）。"""
    if getattr(args, "real", False):
        from config import Config
        cfg = Config.get()
        idx = cfg["universe"]["index_code"].split(".")[0]  # 000300
        begin = args.begin or cfg["fetch"]["begin_date"]
        end = args.end or cfg.get("end_date") or begin
        yr_b, yr_e = str(begin)[:4], str(end)[:4]
        yr = yr_b if yr_b == yr_e else f"{yr_b}_{yr_e}"
        return f"{idx}_{yr}"
    return "mock"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YuriQuant 因子挖掘")
    parser.add_argument("--real", action="store_true", help="使用真实数据（默认 mock）")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--windows", default="5,10,20,60", help="窗口候选，逗号分隔")
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2], help="候选生成深度")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--fdr-q", type=float, default=0.05, help="BH-FDR 显著性水平")
    parser.add_argument("--top", type=int, default=20, help="打印前 N 个因子")
    parser.add_argument("--detail-n", type=int, default=50,
                        help="计算 IC衰减/自相关(turnover代理)的 top-N 因子数（默认50）")
    parser.add_argument("--jobs", type=int, default=0,
                        help="并行评估进程数（默认0=CPU核数；1=串行）")
    parser.add_argument("--out", default=None, help="结果 CSV 输出路径")
    # 遗传规划
    parser.add_argument("--gp", action="store_true", help="改用遗传规划挖掘（替代 exhaustive 枚举）")
    parser.add_argument("--pop", type=int, default=200, help="GP 种群规模")
    parser.add_argument("--gen", type=int, default=20, help="GP 迭代代数")
    parser.add_argument("--max-depth", type=int, default=5, help="GP 最大树深")
    parser.add_argument("--patience", type=int, default=6,
                        help="GP 早停：连续 N 代 hof best 无提升即提前终止（0=关闭）")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="GP 进化只用前 train_frac 时间段的 IC（样本外验证，默认0.7）")
    parser.add_argument("--monthly-weight", type=float, default=0.5,
                        help="GP 月频 IC 权重（多 horizon 融合，默认0.5；0=关闭）")
    parser.add_argument("--gp-penalty", type=float, default=0.0,
                        help="GP 与因子库去相关惩罚系数（>0 自动加载因子库面板）")
    parser.add_argument("--no-window-jitter", action="store_true",
                        help="关闭 GP 窗口 jitter 变异（默认开启）")
    parser.add_argument("--gp-dedup-corr", type=float, default=0.9,
                        help="GP hof 去相关聚类阈值（0=关闭，默认0.9）")
    parser.add_argument("--gp-nsga2", action="store_true",
                        help="GP 用 NSGA-II 多目标（IC 强度 vs 换手稳定性）")
    parser.add_argument("--gp-refine", action="store_true",
                        help="GP 后做 memetic 局部搜索（hof 公式近邻批量检验）")
    parser.add_argument("--gp-neighbors", type=int, default=10, help="memetic 每公式近邻数")
    parser.add_argument("--gp-jobs", type=int, default=1, help="GP 种群并行评估进程数（默认1=串行）")
    parser.add_argument("--gp-sample-step", type=int, default=1,
                        help="GP 进化期 IC 时间子采样步长（粗筛加速；1=全样本精算）")
    parser.add_argument("--gp-preprocess", action="store_true",
                        help="GP 特征预处理：截面 zscore + 行业市值中性化（P0-③，消除价格/成交额风格主导）")
    # 因子库集成
    parser.add_argument("--save-library", action="store_true", help="把 top-K 候选因子入库（预计算 IC + 回测）")
    parser.add_argument("--use-library", action="store_true", help="把因子库已存因子作为特征，参与本轮挖掘（迭代）")
    parser.add_argument("--library-dataset", default=None,
                        help="因子库数据集名（按数据集分库根）。不填自动推导：真实→<指数>_<年>，mock→mock")
    parser.add_argument("--lib-top", type=int, default=20, help="--save-library 入库的因子数")
    args = parser.parse_args()

    windows = tuple(int(x) for x in args.windows.split(",") if x.strip())

    if args.real:
        from config import Config
        cfg = Config.get()
        begin = args.begin or cfg["fetch"]["begin_date"]
        end = args.end or cfg.get("end_date")
        panel, returns_panel = build_real_panel(cfg, begin, end)
    else:
        log.info("使用 Mock 数据（注入 AR(1) 动量信号）...")
        panel = gen_mock_panel_with_signal()
        returns_panel = panel["close"].pct_change().shift(-1)

    lib_dataset = args.library_dataset
    if lib_dataset is None and (args.save_library or args.use_library):
        lib_dataset = _derive_dataset(args)
        log.info("因子库数据集(自动推导): %s", lib_dataset)

    if args.use_library:
        panel = _merge_library_features(panel, dataset=lib_dataset)
    if args.gp and args.gp_preprocess:
        log.info("GP 特征预处理（截面 zscore + 行业市值中性化）...")
        panel = _gp_preprocess_features(panel)
    features = list(panel.keys())
    log.info("特征: %s", features)

    if args.gp:
        lib_panels = None
        if args.gp_penalty > 0:
            from research.factor_library import FactorLibrary
            lib = FactorLibrary(dataset=lib_dataset)
            lib_panels = lib.load_library_features()
            log.info("GP 库去相关惩罚开启: 惩罚系数=%.2f, 库因子 %d 个",
                     args.gp_penalty, len(lib_panels) if lib_panels else 0)

        if args.gp_nsga2:
            from factor.genetic_mining import run_gp_nsga2
            log.info("GP(NSGA-II 多目标)：pop=%d gen=%d max_depth=%d train_frac=%.2f monthly=%.2f",
                     args.pop, args.gen, args.max_depth, args.train_frac, args.monthly_weight)
            result, hof = run_gp_nsga2(
                panel, returns_panel, features=features, windows=windows,
                population=args.pop, generations=args.gen, max_depth=args.max_depth,
                patience=args.patience, train_frac=args.train_frac,
                monthly_weight=args.monthly_weight, verbose=True,
            )
            show_cols = ["formula", "ic_mean", "t_stat", "f1", "f2", "front"]
        else:
            from factor.genetic_mining import run_gp_mining
            log.info("遗传规划挖掘：pop=%d gen=%d max_depth=%d patience=%d train_frac=%.2f "
                     "monthly=%.2f penalty=%.2f jitter=%s dedup=%.2f windows=%s",
                     args.pop, args.gen, args.max_depth, args.patience, args.train_frac,
                     args.monthly_weight, args.gp_penalty, not args.no_window_jitter,
                     args.gp_dedup_corr, windows)
            result, hof = run_gp_mining(
                panel, returns_panel, features=features, windows=windows,
                population=args.pop, generations=args.gen, max_depth=args.max_depth,
                patience=args.patience, train_frac=args.train_frac,
                monthly_weight=args.monthly_weight, library_panels=lib_panels,
                library_penalty=args.gp_penalty, window_jitter=not args.no_window_jitter,
                dedup_corr=args.gp_dedup_corr, n_jobs=args.gp_jobs,
                sample_step=args.gp_sample_step, verbose=True,
            )
            show_cols = ["formula", "ic_mean", "ic_train", "ic_oos", "t_stat", "t_oos", "height", "n"]

        log.info("GP 完成，实际代数: %d（早停 patience=%d），结果数: %d",
                 getattr(hof, "generations_run", args.gen), args.patience, len(result))

        if args.gp_refine:
            from factor.genetic_mining import refine_gp_neighbors
            log.info("memetic 局部搜索：每公式近邻 %d 个，并行 %d 进程（train 段择优）",
                     args.gp_neighbors, args.jobs)
            result = refine_gp_neighbors(result, panel, returns_panel,
                                         n_per=args.gp_neighbors, n_jobs=args.jobs,
                                         min_obs=20, train_frac=args.train_frac,
                                         verbose=True)
            show_cols = ["name", "ic_mean", "ir", "t_stat", "source", "n"]

        if len(result):
            top = result.head(args.top)
            print("\n===== GP Top {} 因子（按 |t| 排序）=====".format(args.top))
            with pd.option_context("display.max_rows", None, "display.width", 220,
                                   "display.float_format", lambda v: f"{v:.4f}"):
                print(top[show_cols].to_string(index=False))
    else:
        from factor.mining import dedup_by_formula, evaluate_candidates, generate_candidates
        cands = dedup_by_formula(generate_candidates(features=features, windows=windows, depth=args.depth))
        log.info("生成候选因子数: %d", len(cands))

        import os
        n_jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
        log.info("候选评估并行进程数: %d", n_jobs if n_jobs > 1 else 1)
        result = evaluate_candidates(
            cands, panel, returns_panel,
            method=args.method, fdr_q=args.fdr_q, verbose=True,
            detail_n=args.detail_n, n_jobs=n_jobs,
        )
        log.info("有效评估因子数: %d，显著因子数(FDR q=%.2f): %d",
                 len(result), args.fdr_q, int(result["significant"].sum()) if len(result) else 0)

        if len(result):
            top = result.head(args.top)
            cols = ["name", "ir", "ic_mean", "ic_std", "ic_win_rate",
                    "ic_decay5", "ic_decay10", "autocorr",
                    "t_stat", "p_value", "significant", "n"]
            print("\n===== Top {} 候选因子（按 |IR| 排序，Alphalens 式标准摘要）=====".format(args.top))
            with pd.option_context("display.max_rows", None, "display.width", 200,
                                   "display.float_format", lambda v: f"{v:.4f}"):
                print(top[cols].to_string(index=False))

    if args.save_library:
        _save_library(result, panel, returns_panel, args, is_gp=args.gp, dataset=lib_dataset)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)
        log.info("结果已保存: %s", out_path)
    elif len(result):
        from datetime import datetime
        tag = "gp" if args.gp else "mining"
        default_out = Path("reports") / f"factor_{tag}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        default_out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(default_out, index=False)
        log.info("结果已保存: %s", default_out)

    # 实验记录（参数 + 数据指纹 + 指标摘要，统一进 experiments.csv）
    try:
        import sys
        from data.cache import DataCache
        from data.datasource import create_datasource
        from research.experiments import record_experiment
        fingerprint = DataCache(create_datasource()).get_fingerprint()
        metrics_summary = {}
        if len(result):
            top = result.iloc[0]
            metrics_summary = {
                "n_factors": int(len(result)),
                "top_ic": float(top.get("ic_mean", 0.0) or 0.0),
                "top_t": float(top.get("t_stat", 0.0) or 0.0),
                "top_formula": str(top.get("formula", top.get("name", ""))),
            }
        record_experiment(
            kind="gp" if args.gp else "mining",
            command=" ".join(sys.argv),
            params={"real": args.real, "windows": list(windows), "depth": args.depth,
                    "gp": args.gp, "pop": args.pop, "gen": args.gen,
                    "max_depth": args.max_depth, "train_frac": getattr(args, "train_frac", None),
                    "monthly_weight": getattr(args, "monthly_weight", None),
                    "gp_penalty": getattr(args, "gp_penalty", 0.0),
                    "gp_nsga2": getattr(args, "gp_nsga2", False),
                    "gp_refine": getattr(args, "gp_refine", False),
                    "jobs": args.jobs},
            data_fingerprint=fingerprint,
            result_path=str(args.out or (default_out if len(result) else "")),
            metrics=metrics_summary,
        )
    except Exception as e:
        log.warning("实验记录写入失败（不影响结果）: %s", e)


if __name__ == "__main__":
    main()
