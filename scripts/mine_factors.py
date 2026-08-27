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
    """统一真实面板构建。

    2026-08-17 重构：收敛到 ``data.cache_helpers.build_panel``（与 run_gflownet_phase1
    的 build_real_panel 共享同一实现），消除两套 PIT 并集池 / 复权 / 财务 PIT 并行逻辑。
    返回 ``(panel, returns_panel)``，口径不变。
    """
    from data.cache_helpers import build_panel as _build_panel

    log.info("构建真实面板 %s~%s ...", begin, end)
    return _build_panel(cfg, begin, end)


def _build_htai_neutral_panels(panel: dict[str, pd.DataFrame],
                               real: bool = False) -> dict[str, pd.DataFrame]:
    """构建华泰五因子中性化协变量面板。

    对应研报报告21 适应度计算的「行业、市值、过去20日收益率、过去20日平均换手率、
    过去20日波动率」五个中性化因子：

        size:   市值 = TOT_SHARE(PIT) × 后复权 close
        industry: 申万一级行业映射（date×code，值=行业名）
        mom20:  过去 20 日收益率 = close.pct_change(20)
        vol20:  过去 20 日波动率 = 日收益的 20 日滚动 std
        turn20: 过去 20 日平均换手率 = (volume / TOT_SHARE) 的 20 日滚动均值

    ``real=False``（mock）时跳过需要 SDK 的行业/市值/换手部分，只返回 mom20/vol20；
    任一协变量构建失败时自动从返回 dict 剔除（neutralize 只回归可用的部分）。
    """
    out: dict[str, pd.DataFrame] = {}
    close = panel["close"]
    try:
        out["mom20"] = close.pct_change(20)
        out["vol20"] = close.pct_change().rolling(20).std()
    except Exception:
        pass
    if not real:
        return out
    try:
        from data.cache import DataCache
        from data.datasource import create_datasource
        from data.financials import build_pit_panel
        ds = create_datasource()
        cache = DataCache(ds)
        codes = list(close.columns)
        dates = close.index
        cal = cache.get_calendar(int(dates.min().strftime("%Y%m%d")),
                                 int(dates.max().strftime("%Y%m%d")))
        ind = cache.get_industry_classification(level=1)
        if not ind.empty:
            ind["in_date"] = pd.to_datetime(ind["in_date"], errors="coerce")
            ind["out_date"] = pd.to_datetime(ind["out_date"], errors="coerce")
            rows = {}
            for d in dates:
                m = ind[(ind["in_date"] <= d) & (ind["out_date"].fillna(pd.Timestamp.max) >= d)]
                rows[d] = {r["code"]: r["industry_name"]
                           for _, r in m.iterrows() if r["code"] in codes}
            out["industry"] = pd.DataFrame(rows).T.reindex(index=dates, columns=codes)
        balance = cache.get_balance_sheet(codes)
        if not balance.empty and "TOT_SHARE" in balance.columns:
            ts = build_pit_panel(balance, cal, "TOT_SHARE").reindex(index=dates, columns=codes)
            out["size"] = ts * close
            turn = panel["volume"] / ts
            out["turn20"] = turn.rolling(20).mean()
    except Exception as exc:
        log.warning("华泰中性化协变量面板构建不完整（缺失项自动跳过）: %s", exc)
    return out


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
    parser.add_argument("--three-period", action="store_true",
                        help="三段式纪律：train 段算 IC 排序 → valid 段重新算 IC 验证 → "
                             "只保留 valid 段显著的因子（防泄漏，与 GP/ML 同纪律）。"
                             "默认日期 train 22-23 / valid 24 / test 25（真实数据时生效）")
    parser.add_argument("--top", type=int, default=20, help="打印前 N 个因子")
    parser.add_argument("--detail-n", type=int, default=50,
                        help="计算 IC衰减/自相关(turnover代理)的 top-N 因子数（默认50）")
    parser.add_argument("--jobs", type=int, default=0,
                        help="并行评估进程数（默认0=CPU核数；1=串行）")
    parser.add_argument("--out", default=None, help="结果 CSV 输出路径")
    # 遗传规划
    parser.add_argument("--gp", action="store_true", help="改用遗传规划挖掘（替代 exhaustive 枚举）")
    parser.add_argument("--gp-htai", action="store_true",
                        help="GP 华泰复现模式：环内 MAD去极值→五因子中性化→zscore + 月频20日目标 + "
                             "平均RankIC适应度（函数集/参数对齐研报21/23；pop/gen/depth/tournament 默认按研报）")
    parser.add_argument("--gp-fitness", default="tstat",
                        choices=["tstat", "rankic_mean", "mutual_info", "top_excess"],
                        help="GP 适应度口径：tstat=按 |mean IC|/std（默认）；rankic_mean=华泰研报21 的 "
                             "全期平均 RankIC；mutual_info=华泰研报23 的互信息（挖非线性因子）；"
                             "top_excess=华泰研报23 的多头超额收益（Top/Bottom 层年化超额较大者）。"
                             "后三种仅 htai 模式生效")
    parser.add_argument("--pop", type=int, default=None, help="GP 种群规模（默认：htai=1000，否则200）")
    parser.add_argument("--gen", type=int, default=None, help="GP 迭代代数（默认：htai=3，否则20）")
    parser.add_argument("--max-depth", type=int, default=None, help="GP 最大树深（默认：htai=4，否则5）")
    parser.add_argument("--min-depth", type=int, default=None, help="GP 最小树深（默认：htai=1，否则2）")
    parser.add_argument("--gp-tournament", type=int, default=None,
                        help="GP 锦标赛选择规模（默认：htai=20，否则5）")
    parser.add_argument("--patience", type=int, default=6,
                        help="GP 早停：连续 N 代 hof best 无提升即提前终止（0=关闭）")
    parser.add_argument("--train-frac", type=float, default=None,
                        help="GP 进化只用前 train_frac 时间段的 IC（样本外验证；默认：htai=1.0 全样本，否则0.7）")
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

    # ---- 华泰复现模式：特征补全 + 参数按研报解析 + 中性化协变量面板 ----
    neutral_panels = None
    if args.gp_htai:
        if "returns" not in panel and "close" in panel:
            panel["returns"] = panel["close"].pct_change()   # 研报 RETURNS
        if "vwap" not in panel and "amount" in panel and "volume" in panel:
            panel["vwap"] = panel["amount"] / panel["volume"]  # 研报 VWAP（量加权均价）
        if args.pop is None:
            args.pop = 1000
        if args.gen is None:
            args.gen = 3
        if args.min_depth is None:
            args.min_depth = 1
        if args.max_depth is None:
            args.max_depth = 4
        if args.gp_tournament is None:
            args.gp_tournament = 20
        if args.train_frac is None:
            args.train_frac = 1.0     # 研报21 全样本；报告23 CV 口径可显式 --train-frac 0.8
        neutral_panels = _build_htai_neutral_panels(panel, real=args.real)
        log.info("华泰复现模式：特征补 returns/vwap，中性化协变量=%s，参数 pop=%d gen=%d depth=(%d,%d) tourn=%d train_frac=%.2f",
                 list(neutral_panels.keys()), args.pop, args.gen, args.min_depth,
                 args.max_depth, args.gp_tournament, args.train_frac)
    else:
        if args.pop is None:
            args.pop = 200
        if args.gen is None:
            args.gen = 20
        if args.min_depth is None:
            args.min_depth = 2
        if args.max_depth is None:
            args.max_depth = 5
        if args.gp_tournament is None:
            args.gp_tournament = 5
        if args.train_frac is None:
            args.train_frac = 0.7

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
            if args.gp_htai:
                log.warning("--gp-nsga2 暂不支持 htai 口径（华泰复现为单目标平均RankIC），忽略 htai 选项")
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
            log.info("遗传规划挖掘：pop=%d gen=%d depth=(%d,%d) tourn=%d patience=%d train_frac=%.2f "
                     "monthly=%.2f penalty=%.2f jitter=%s dedup=%.2f windows=%s htai=%s fitness=%s",
                     args.pop, args.gen, args.min_depth, args.max_depth, args.gp_tournament,
                     args.patience, args.train_frac,
                     args.monthly_weight, args.gp_penalty, not args.no_window_jitter,
                     args.gp_dedup_corr, windows, args.gp_htai, args.gp_fitness)
            result, hof = run_gp_mining(
                panel, returns_panel, features=features, windows=windows,
                population=args.pop, generations=args.gen, min_depth=args.min_depth,
                max_depth=args.max_depth, tournament=args.gp_tournament,
                patience=args.patience, train_frac=args.train_frac,
                monthly_weight=args.monthly_weight, library_panels=lib_panels,
                library_penalty=args.gp_penalty, window_jitter=not args.no_window_jitter,
                dedup_corr=args.gp_dedup_corr, n_jobs=args.gp_jobs,
                sample_step=args.gp_sample_step, verbose=True,
                htai=args.gp_htai, neutral_panels=neutral_panels,
                fitness_mode=args.gp_fitness,
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

        if args.three_period and args.real:
            # 三段式纪律：train 段评估 → valid 段验证 → 只保留 valid 显著的
            from factor.cv import split_three_periods
            tr_days, va_days, te_days = split_three_periods(returns_panel.index)
            log.info("三段式：train %d 日 / valid %d 日 / test %d 日",
                     len(tr_days), len(va_days), len(te_days))
            train_panel = {k: v.loc[tr_days] for k, v in panel.items()}
            train_returns = returns_panel.loc[tr_days]
            valid_panel = {k: v.loc[va_days] for k, v in panel.items()}
            valid_returns = returns_panel.loc[va_days]

            # 1. train 段全量评估 + 排序
            log.info("train 段评估候选因子 ...")
            result_train = evaluate_candidates(
                cands, train_panel, train_returns,
                method=args.method, fdr_q=args.fdr_q, verbose=True,
                detail_n=args.detail_n, n_jobs=n_jobs,
            )
            if len(result_train) == 0:
                log.warning("train 段无有效因子")
                result = result_train
            else:
                # 取 train 段 top 2*top（宽进严出）到 valid 段重新评估
                n_valid = min(args.top * 3, len(result_train))
                top_names = result_train.head(n_valid)["name"].tolist()
                cands_valid = [c for c in cands if c.name in set(top_names)]
                log.info("valid 段重新评估 %d 个候选 ...", len(cands_valid))
                result = evaluate_candidates(
                    cands_valid, valid_panel, valid_returns,
                    method=args.method, fdr_q=args.fdr_q, verbose=True,
                    detail_n=args.detail_n, n_jobs=n_jobs,
                )
                # 标注 valid 段是否显著
                if len(result):
                    result["ic_train"] = result_train.set_index("name")["ic_mean"] \
                        .reindex(result["name"]).values
                    result["t_train"] = result_train.set_index("name")["t_stat"] \
                        .reindex(result["name"]).values
                    n_sig = int(result["significant"].sum()) if "significant" in result else 0
                    log.info("三段式完成：valid 段显著因子 %d/%d", n_sig, len(result))
        else:
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
            if args.three_period and args.real and "ic_train" in result.columns:
                cols = ["name", "ic_train", "ic_mean", "t_train", "t_stat",
                        "ir", "ic_win_rate", "significant", "n"]
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
                    "gp": args.gp, "gp_htai": getattr(args, "gp_htai", False),
                    "gp_fitness": getattr(args, "gp_fitness", None),
                    "pop": args.pop, "gen": args.gen,
                    "max_depth": args.max_depth, "min_depth": getattr(args, "min_depth", None),
                    "gp_tournament": getattr(args, "gp_tournament", None),
                    "train_frac": getattr(args, "train_frac", None),
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
