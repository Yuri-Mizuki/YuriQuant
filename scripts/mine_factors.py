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
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402


log = setup_logging("mine_factors")


# ---------------------------------------------------------------------------
# 国君研报口径：应用基准预设（次日 VWAP 执行链收益率见 factor.gtja）
# ---------------------------------------------------------------------------
def _apply_gtja_preset(args, panel: dict[str, pd.DataFrame],
                       real: bool) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """应用国君研报基准预设（2023 解构系列之一，表1），返回 (gp 用面板, gp 用收益面板)。

    - 终端集裁剪为研报六量价字段 O/H/L/C/VWAP/V（财务字段等一律移出终端）；
    - 收益率切换为次日 VWAP 执行链（费后夏普适应度与研报同口径）；
    - 表1 超参：pop=500 / gen=10 / patience=5 / windows={1,5,10,20,40,60} /
      train_frac=1.0（样本内全量挖）/ min_fitness=0.5（二次筛选·限制一）/
      dedup_corr=0.5（限制二）/ separated_mutation=True（表1 变异概率）；
      beam/family/crowding 不动（消融项，由用户显式叠加）。
    只填用户未显式指定的项（None 哨兵），显式传参优先。
    """
    from factor.gtja import build_gtja_tradable, build_vwap_exec_returns

    missing = [f for f in ("amount", "volume") if f not in panel]
    if missing:
        raise SystemExit(f"--gp-gtja 需要 {missing} 字段构建 VWAP，当前面板缺失")

    # 0) 可交易性掩码（真实模式）：T+1 停牌/封板、当日 ST/停牌 剔除进适应度。
    #    须在终端裁剪前构建（封板判定用全量 close + 复权因子）。
    tradable = build_gtja_tradable(panel) if real else None
    args._gtja_tradable = tradable      # 主流程取用后传入 run_gp_mining

    # 1) 收益率先于终端裁剪构建（helper 需要 amount/volume/close）
    returns_gp = build_vwap_exec_returns(panel)

    # 2) 终端裁剪：六量价字段
    six = {"open", "high", "low", "close", "volume", "vwap"}
    if "vwap" not in panel:
        panel = dict(panel)
        panel["vwap"] = panel["amount"] / panel["volume"]
    dropped = sorted(set(panel) - six)
    panel = {k: panel[k] for k in six & set(panel)}
    log.info("gtja 终端集 = %s（移出 %d 个非量价终端: %s）",
             sorted(panel), len(dropped), dropped)

    # 3) 表1 超参（仅填未显式指定的项）
    if args.pop is None:
        args.pop = 500
    if args.gen is None:
        args.gen = 10
    if args.patience is None:
        args.patience = 5
    if args.train_frac is None:
        args.train_frac = 1.0     # 研报基准样本内全量挖；内部 CV 口径可显式传 0.8
    if args.gp_tournament is None:
        args.gp_tournament = 5
    if args.gp_min_fitness == 0.0:
        args.gp_min_fitness = 0.5     # 二次筛选·限制一：扣费后夏普>0.5 才入池
    if args.gp_dedup_corr is None:
        args.gp_dedup_corr = 0.5      # 二次筛选·限制二：与池内因子相关 ≤50%
    # separated_mutation 对应表1 变异概率（子树0.2/结点0.2/提升0.05），预设强制开启；
    # beam/family/crowding 是研报消融项，保持用户显式设定
    args.gp_separated_mutation = True
    if args.gp_fitness not in (None, "sharpe"):
        log.warning("gtja 预设基准 fitness 为 sharpe，当前显式指定 %s（保留你的选择）",
                    args.gp_fitness)
    elif args.gp_fitness is None:
        args.gp_fitness = "sharpe"
    log.info("gtja 预设参数：pop=%d gen=%d patience=%d train_frac=%.2f tourn=%d "
             "min_fitness=%.2f dedup=%.2f fitness=%s separated_mut=%s",
             args.pop, args.gen, args.patience, args.train_frac,
             args.gp_tournament, args.gp_min_fitness, args.gp_dedup_corr,
             args.gp_fitness, args.gp_separated_mutation)
    return panel, returns_gp


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
# CLI 解析与主流程
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.Namespace:
    """构建 CLI 参数解析器并解析（挖掘 / GP / GTJA / HTAI / 因子库集成全部参数）。"""
    parser = argparse.ArgumentParser(description="YuriQuant 因子挖掘")
    add_real_mock_args(parser, offline=True, real_help="使用真实数据（默认 mock）")
    parser.add_argument("--begin", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--universe", default=None,
                        choices=["hs300", "zz500", "zz1000", "all_a"],
                        help="覆盖 config.universe.default 股票池（国君研报复现用 all_a；"
                             "不改全局配置，仅影响本次运行）")
    parser.add_argument("--windows", default=None,
                        help="窗口候选，逗号分隔（默认：gp-gtja 模式 1,5,10,20,40,60，否则 5,10,20,60）")
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2], help="候选生成深度")
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--fdr-q", type=float, default=0.05, help="BH-FDR 显著性水平")
    parser.add_argument("--three-period", action="store_true",
                        help="三段式纪律：train 段算 IC 排序 → valid 段重新算 IC 验证 → "
                             "只保留 valid 段显著的因子（防泄漏，与 GP/ML 同纪律）。"
                             "默认日期 train 22-23 / valid 24 / test 25（真实数据时生效）")
    parser.add_argument("--rolling", type=int, default=None, metavar="N_SPLITS",
                        help="滚动复核挖因子（walk-forward mining）：用 forward_folds 切 "
                             "expanding 前推折，每折 train 段评估候选 → test 段复核 top 候选，"
                             "要求跨折 IC 方向一致且 test 段显著占比达标才标 rolling_significant。"
                             "N_SPLITS>=2（实际折数 N_SPLITS-1）。与 --three-period 互斥。"
                             "可用 --rolling-top-train/--rolling-min-consistent/--rolling-min-sig 微调")
    parser.add_argument("--rolling-top-train", type=int, default=50,
                        help="滚动复核：每折 train 段取 top N 候选进入 test 复核（默认50）")
    parser.add_argument("--rolling-min-consistent", type=float, default=0.6,
                        help="滚动复核：跨折 IC 方向一致率下限（默认0.6）")
    parser.add_argument("--rolling-min-sig", type=float, default=0.3,
                        help="滚动复核：跨折 test 段显著折占比下限（默认0.3）")
    parser.add_argument("--rolling-min-folds", type=int, default=2,
                        help="滚动复核：有效折数下限（默认2）")
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
    parser.add_argument("--gp-gtja", action="store_true",
                        help="GP 国君研报复现模式（2023 解构系列之一，基准=表1）：终端裁剪为研报六量价字段 "
                             "(O/H/L/C/VWAP/V) + 收益率改用次日 VWAP 执行链（后复权，费用口径双边千三）；"
                             "默认 pop=500/gen=10/patience=5/fitness=sharpe/train_frac=1.0/"
                             "min_fitness=0.5(费后夏普入池门槛)/dedup_corr=0.5/separated_mutation。"
                             "束搜索/家庭竞争/排挤为研报消融项，不随预设开启，按需显式加参")
    parser.add_argument("--gp-fitness", default=None,
                        choices=["tstat", "rankic_mean", "mutual_info", "top_excess",
                                 "sharpe", "annual_return", "ret_minus_dd"],
                        help="GP 适应度口径：tstat=按 |mean IC|/std（默认）；rankic_mean=华泰研报21 的 "
                             "全期平均 RankIC；mutual_info=华泰研报23 的互信息（挖非线性因子）；"
                             "top_excess=华泰研报23 的多头超额收益。"
                             "sharpe/annual_return/ret_minus_dd=国君研报(2023)的费后多空净值型适应度："
                             "年化夏普（研报基准，双边千三费用内嵌）/ 年化收益 / 收益-最大回撤。"
                             "前四种仅 htai 模式生效；gtja 模式默认 sharpe")
    parser.add_argument("--pop", type=int, default=None, help="GP 种群规模（默认：htai=1000，否则200）")
    parser.add_argument("--gen", type=int, default=None, help="GP 迭代代数（默认：htai=3，否则20）")
    parser.add_argument("--max-depth", type=int, default=None, help="GP 最大树深（默认：htai=4，否则5）")
    parser.add_argument("--min-depth", type=int, default=None, help="GP 最小树深（默认：htai=1，否则2）")
    parser.add_argument("--gp-tournament", type=int, default=None,
                        help="GP 锦标赛选择规模（默认：htai=20，否则5）")
    parser.add_argument("--patience", type=int, default=None,
                        help="GP 早停：连续 N 代 hof best 无提升即提前终止（0=关闭；默认：gtja=5，否则 6）")
    parser.add_argument("--train-frac", type=float, default=None,
                        help="GP 进化只用前 train_frac 时间段的 IC（样本外验证；默认：htai/gtja=1.0 全样本，否则0.7）")
    parser.add_argument("--monthly-weight", type=float, default=0.5,
                        help="GP 月频 IC 权重（多 horizon 融合，默认0.5；0=关闭）")
    parser.add_argument("--gp-penalty", type=float, default=0.0,
                        help="GP 与因子库去相关惩罚系数（>0 自动加载因子库面板）")
    parser.add_argument("--no-window-jitter", action="store_true",
                        help="关闭 GP 窗口 jitter 变异（默认开启）")
    parser.add_argument("--gp-dedup-corr", type=float, default=None,
                        help="GP hof 去相关聚类阈值（0=关闭；默认：gtja=0.5（研报二次筛选·限制二），否则 0.9）")
    parser.add_argument("--gp-nsga2", action="store_true",
                        help="GP 用 NSGA-II 多目标（IC 强度 vs 换手稳定性）")
    parser.add_argument("--gp-refine", action="store_true",
                        help="GP 后做 memetic 局部搜索（hof 公式近邻批量检验）")
    parser.add_argument("--gp-neighbors", type=int, default=10, help="memetic 每公式近邻数")
    parser.add_argument("--gp-jobs", type=int, default=1, help="GP 种群并行评估进程数（默认1=串行）")
    parser.add_argument("--seed", type=int, default=0, help="GP 随机种子（研报多轮挖掘=多种子取并集）")
    parser.add_argument("--gp-sample-step", type=int, default=1,
                        help="GP 进化期 IC 时间子采样步长（粗筛加速；1=全样本精算）")
    # 国君研报（2023 解构系列之一）对齐参数
    parser.add_argument("--gp-beam-mult", type=int, default=0,
                        help="束搜索初始化：初始种群 = population×N 后按适应度截断（0=关闭；研报消融最佳组合成员）")
    parser.add_argument("--gp-family-competition", action="store_true",
                        help="家庭竞争：子代适应度超越父代则父代从选择池剔除（防单一父代过度繁衍）")
    parser.add_argument("--gp-crowding", default=None, choices=[None, "supplant", "sharing"],
                        help="种内相似度调整：supplant=排挤（相似>阈值者低适应减半，研报结论优于 sharing）"
                             " / sharing=共享适应度（按相似度和归一化）。默认关闭")
    parser.add_argument("--gp-crowd-corr-thr", type=float, default=0.8,
                        help="排挤算法的相似度判定阈值（默认 0.8）")
    parser.add_argument("--gp-min-fitness", type=float, default=0.0,
                        help="hof 准入门槛：fitness>=该值才进精英池（研报二次筛选·限制一，"
                             "如费后夏普>0.5 时传 0.5；0=关闭）")
    parser.add_argument("--gp-separated-mutation", action="store_true",
                        help="四类变异概率分离：子树0.2/结点0.2/提升0.05（研报基准），剩余走窗口jitter")
    parser.add_argument("--gp-preprocess", action="store_true",
                        help="GP 特征预处理：截面 zscore + 行业市值中性化（P0-③，消除价格/成交额风格主导）")
    # 因子库集成
    parser.add_argument("--save-library", action="store_true", help="把 top-K 候选因子入库（预计算 IC + 回测）")
    parser.add_argument("--use-library", action="store_true", help="把因子库已存因子作为特征，参与本轮挖掘（迭代）")
    parser.add_argument("--library-dataset", default=None,
                        help="因子库数据集名（按数据集分库根）。不填自动推导：真实→<指数>_<年>，mock→mock")
    parser.add_argument("--lib-top", type=int, default=20, help="--save-library 入库的因子数")
    return parser.parse_args()


def main():
    args = _build_parser()

    # windows 哨兵化：gtja 预设用研报时序参数集 {1,5,10,20,40,60}，否则沿用项目默认
    win_str = args.windows or ("1,5,10,20,40,60" if args.gp_gtja else "5,10,20,60")
    windows = tuple(int(x) for x in win_str.split(",") if x.strip())

    if args.real:
        from config import Config
        from data.cache_helpers import build_real_panel
        cfg = Config.get()
        if args.universe:
            cfg["universe"]["default"] = args.universe
            log.info("股票池覆盖: %s（仅本次运行）", args.universe)
        begin = args.begin or cfg["fetch"]["begin_date"]
        end = args.end or cfg.get("end_date")
        panel, returns_panel = build_real_panel(cfg, begin, end, offline=args.offline)
    else:
        from data.mock import gen_mock_panel_with_signal
        log.info("使用 Mock 数据（注入 AR(1) 动量信号）...")
        panel = gen_mock_panel_with_signal()
        returns_panel = panel["close"].pct_change().shift(-1)

    lib_dataset = args.library_dataset
    if lib_dataset is None and (args.save_library or args.use_library):
        lib_dataset = _derive_dataset(args)
        log.info("因子库数据集(自动推导): %s", lib_dataset)

    if args.use_library:
        panel = _merge_library_features(panel, dataset=lib_dataset)

    # ---- 国君复现模式：六量价终端 + 次日VWAP执行链收益 + 表1 超参 ----
    returns_gp = returns_panel
    tradable_gp = None
    if args.gp_gtja:
        if args.begin is None or args.end is None:
            raise SystemExit("--gp-gtja 复现需显式指定 --begin/--end（如 20220101/20221231），"
                             "避免默认区间偏离研报样本")
        panel, returns_gp = _apply_gtja_preset(args, panel, real=args.real)
        tradable_gp = getattr(args, "_gtja_tradable", None)

    # ---- 华泰复现模式：特征补全 + 参数按研报解析 + 中性化协变量面板 ----
    neutral_panels = None
    if args.gp_htai:
        from data.cache_helpers import build_htai_neutral_panels
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
        neutral_panels = build_htai_neutral_panels(panel, real=args.real)
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
    # 通用哨兵兜底（三种模式共用）
    if args.patience is None:
        args.patience = 6
    if args.gp_fitness is None:
        args.gp_fitness = "tstat"
    if args.gp_dedup_corr is None:
        args.gp_dedup_corr = 0.9

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
                panel, returns_gp, features=features, windows=windows,
                population=args.pop, generations=args.gen, min_depth=args.min_depth,
                max_depth=args.max_depth, tournament=args.gp_tournament,
                patience=args.patience, train_frac=args.train_frac,
                monthly_weight=args.monthly_weight, library_panels=lib_panels,
                library_penalty=args.gp_penalty, window_jitter=not args.no_window_jitter,
                dedup_corr=args.gp_dedup_corr, n_jobs=args.gp_jobs,
                sample_step=args.gp_sample_step, verbose=True,
                htai=args.gp_htai, neutral_panels=neutral_panels,
                fitness_mode=args.gp_fitness,
                beam_mult=args.gp_beam_mult,
                family_competition=args.gp_family_competition,
                crowding=args.gp_crowding,
                crowd_corr_thr=args.gp_crowd_corr_thr,
                min_fitness=args.gp_min_fitness,
                separated_mutation=args.gp_separated_mutation,
                seed=args.seed,
                tradable=tradable_gp,
            )
            show_cols = ["formula", "ic_mean", "ic_train", "ic_oos", "t_stat", "t_oos", "height", "n"]
            if args.gp_fitness in ("sharpe", "annual_return", "ret_minus_dd"):
                show_cols = ["formula", "fitness"] + show_cols[1:]

        log.info("GP 完成，实际代数: %d（早停 patience=%d），结果数: %d",
                 getattr(hof, "generations_run", args.gen), args.patience, len(result))

        if args.gp_refine:
            from factor.genetic_mining import refine_gp_neighbors
            log.info("memetic 局部搜索：每公式近邻 %d 个，并行 %d 进程（train 段择优）",
                     args.gp_neighbors, args.jobs)
            result = refine_gp_neighbors(result, panel, returns_gp,
                                         n_per=args.gp_neighbors, n_jobs=args.jobs,
                                         min_obs=20, train_frac=args.train_frac,
                                         verbose=True)
            show_cols = ["name", "ic_mean", "ir", "t_stat", "source", "n"]

        if len(result):
            top = result.head(args.top)
            sort_note = ("按 fitness（费后表现）" if args.gp_fitness in
                         ("sharpe", "annual_return", "ret_minus_dd") else "按 |t|")
            print(f"\n===== GP Top {args.top} 因子（{sort_note}排序）=====")
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
        elif args.rolling:
            # 滚动复核挖因子：forward_folds 前推折，每折 train 评估 → test 复核，
            # 跨折 IC 方向一致 + test 显著占比达标才标 rolling_significant
            if args.gp or args.gp_gtja:
                log.warning("--rolling 与 GP 模式互斥，忽略 GP 开关，走候选枚举滚动复核")
            from factor.mining import rolling_evaluate_candidates
            log.info("滚动复核：n_splits=%d, embargo=5, top_train=%d, "
                     "min_consistent=%.2f, min_sig=%.2f, min_folds=%d",
                     args.rolling, args.rolling_top_train, args.rolling_min_consistent,
                     args.rolling_min_sig, args.rolling_min_folds)
            result = rolling_evaluate_candidates(
                cands, panel, returns_panel,
                n_splits=args.rolling, embargo_days=5,
                method=args.method, fdr_q=args.fdr_q,
                n_jobs=n_jobs, top_train=args.rolling_top_train,
                min_consistent_frac=args.rolling_min_consistent,
                min_sig_frac=args.rolling_min_sig,
                min_folds=args.rolling_min_folds,
                verbose=True,
            )
            log.info("滚动复核完成：候选 %d，rolling_significant %d",
                     len(result), int(result["rolling_significant"].sum())
                     if len(result) else 0)
        else:
            result = evaluate_candidates(
                cands, panel, returns_panel,
                method=args.method, fdr_q=args.fdr_q, verbose=True,
                detail_n=args.detail_n, n_jobs=n_jobs,
            )
        if args.rolling:
            log.info("滚动复核因子数: %d，rolling_significant %d", len(result),
                     int(result["rolling_significant"].sum()) if len(result) else 0)
        else:
            log.info("有效评估因子数: %d，显著因子数(FDR q=%.2f): %d",
                     len(result), args.fdr_q,
                     int(result["significant"].sum()) if len(result) else 0)

        if len(result):
            top = result.head(args.top)
            if args.rolling:
                cols = ["name", "ir_fold", "ic_mean", "n_folds", "consistent_frac",
                        "significant_frac", "direction", "rolling_significant"]
                print("\n===== Top {} 滚动复核因子（按 |IR_fold| 排序，跨折一致）=====".format(args.top))
                with pd.option_context("display.max_rows", None, "display.width", 200,
                                       "display.float_format", lambda v: f"{v:.4f}"):
                    print(top[cols].to_string(index=False))
            else:
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
                "top_t": float(top.get("t_stat", top.get("ir_fold", 0.0)) or 0.0),
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
                    "gp_beam_mult": getattr(args, "gp_beam_mult", 0),
                    "gp_family_competition": getattr(args, "gp_family_competition", False),
                    "gp_crowding": getattr(args, "gp_crowding", None),
                    "gp_min_fitness": getattr(args, "gp_min_fitness", 0.0),
                    "jobs": args.jobs},
            data_fingerprint=fingerprint,
            result_path=str(args.out or (default_out if len(result) else "")),
            metrics=metrics_summary,
        )
    except Exception as e:
        log.warning("实验记录写入失败（不影响结果）: %s", e)


if __name__ == "__main__":
    main()
