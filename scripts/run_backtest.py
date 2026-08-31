"""
回测入口脚本
============

一键跑完: 数据 → 因子 → 策略 → 回测 → 报告

支持多因子批量回测，统一对比。

用法（安装为可编辑包后，python scripts/run_backtest.py 与
python -m scripts.run_backtest 都可以在任意 cwd 下运行；参见 README 的安装说明）:
    # 单因子（Mock）
    python scripts/run_backtest.py --factor momentum_20

    # 多因子对比（Mock）
    python scripts/run_backtest.py --factors momentum_20,volatility_20,turnover_20

    # 全部因子
    python scripts/run_backtest.py --factors all

    # 用真实数据
    python scripts/run_backtest.py --real --factors all

    # 指定策略和调仓
    python scripts/run_backtest.py --factors all --strategy topk_ls --k 30 --freq M

    # 跳过因子预处理（去极值/中性化/标准化），直接用原始因子值
    python scripts/run_backtest.py --factors all --no-preprocess

因子预处理由 config/settings.yaml 的 preprocessing 段控制：
真实数据模式下自动构建行业面板 + 市值面板做中性化（需先跑 update_data
拉取行业分类与股本结构）；Mock 模式无行业/市值数据，中性化自动跳过，
仅做去极值 + 标准化。
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import add_real_mock_args, setup_logging  # noqa: E402


from backtest import VectorBacktest  # noqa: E402
from backtest.costs import ShortCostModel  # noqa: E402
from factor import ALL_FACTORS  # noqa: E402
from factor.preprocessing import preprocess_factor  # noqa: E402
from research import generate_excel_report  # noqa: E402
from research.factor_analysis import factor_summary  # noqa: E402
from research.html_report import generate_html_report  # noqa: E402
from strategy.examples import build_strategy  # noqa: E402

log = setup_logging("backtest")

# ---------------------------------------------------------------------------
# Mock 数据
# ---------------------------------------------------------------------------
def gen_mock_data(begin: int = 20200101, end: int = 20231231, n_codes: int = 100) -> dict[str, pd.DataFrame]:
    """生成 mock 日频面板数据。"""
    dates = pd.date_range(str(begin), str(end), freq="B")
    codes = [f"{600000+i:06d}.SH" for i in range(n_codes)]
    rng = np.random.default_rng(42)

    base = 10.0 + rng.uniform(0, 50, n_codes)
    rets = rng.normal(0, 0.02, (len(dates), n_codes))
    close = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(n_codes):
        close.iloc[:, i] = base[i] * np.exp(np.cumsum(rets[:, i]))
    high = close * (1 + rng.uniform(0, 0.03, (len(dates), n_codes)))
    low = close * (1 - rng.uniform(0, 0.03, (len(dates), n_codes)))
    volume = pd.DataFrame(rng.integers(1e6, 1e8, (len(dates), n_codes)), index=dates, columns=codes)
    amount = volume * close

    return {"close": close, "high": high, "low": low, "volume": volume, "amount": amount}

# ---------------------------------------------------------------------------
# 单因子回测
# ---------------------------------------------------------------------------
def run_single_factor(
    factor_name: str,
    panel: dict[str, pd.DataFrame],
    strategy_name: str,
    k: int,
    freq: str,
    returns_panel: pd.DataFrame,
    executable_mask: pd.DataFrame | None = None,
    preprocess_cfg: dict | None = None,
    market_cap_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    short_costs: ShortCostModel | None = None,
    deleverage: bool = False,
) -> tuple[str, object, dict]:
    """跑单个因子回测，返回 (因子名, BacktestResult, factor_summary)。

    preprocess_cfg 非空且 enabled 时，因子值先经过标准预处理流程
    （去极值 -> 行业/市值中性化 -> 标准化）再进入策略与因子分析；
    market_cap_panel / industry_panel 为 None 时对应中性化自动跳过。
    short_costs / deleverage 透传给回测引擎（空头腿成本模型）。
    """
    factor_cls = ALL_FACTORS[factor_name]()
    factor_values = factor_cls.calc(panel)
    factor_panel = factor_values if isinstance(factor_values, pd.DataFrame) else factor_values.unstack("code")

    if preprocess_cfg is not None and preprocess_cfg.get("enabled", True):
        factor_panel = preprocess_factor(
            factor_panel,
            market_cap_panel=market_cap_panel,
            industry_panel=industry_panel,
            winsorize=preprocess_cfg.get("winsorize", "mad"),
            neutralize_industry=preprocess_cfg.get("neutralize_industry", True),
            neutralize_size=preprocess_cfg.get("neutralize_size", True),
            standardize=preprocess_cfg.get("standardize", "zscore"),
        )

    strat = build_strategy(strategy_name, k)
    bt = VectorBacktest(
        strategy=strat, rebalance_freq=freq,
        short_costs=short_costs, deleverage=deleverage,
    )
    result = bt.run(factor_panel, returns_panel, executable_mask=executable_mask)

    # 因子分析
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]
    fs = factor_summary(fp, rp)

    return factor_name, result, fs

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YuriQuant 回测")
    add_real_mock_args(parser, real_help="使用真实本地数据")
    parser.add_argument("--factor", default=None, help="单因子名称")
    parser.add_argument("--factors", default=None, help="多因子名称(逗号分隔)或 all")
    parser.add_argument("--strategy", default="topk_ls", choices=["topk_ls", "topk_lo", "quantile"], help="策略")
    parser.add_argument("--k", type=int, default=30, help="持仓数")
    parser.add_argument("--freq", default="M", choices=["D", "W", "M"], help="调仓频率")
    parser.add_argument("--no-preprocess", action="store_true", help="跳过因子预处理，直接用原始因子值")
    parser.add_argument("--no-short-cost", action="store_true",
                        help="关闭空头腿成本（借券费=0，旧口径；默认启用 8%% 年化）")
    parser.add_argument("--borrow-rate", type=float, default=None, help="年化借券费率（默认读配置 0.08）")
    parser.add_argument("--margin-ratio", type=float, default=None, help="融券保证金比例（默认读配置 1.0）")
    parser.add_argument("--deleverage", action="store_true",
                        help="1 倍资金约束：总保证金需求（多头+空头×保证金比例）>1 时按比例降杠杆")
    parser.add_argument("--no-html", action="store_true",
                        help="跳过交互式 HTML 报告（默认与 Excel 同时生成）")
    args = parser.parse_args()

    # 空头腿成本模型：默认从配置读并启用（修正空头腿乐观偏差）
    from config import Config as _CfgMain
    _cfg_bt = dict(_CfgMain.get().get("backtest", {}))
    _cfg_br = _cfg_bt.get("short_borrow_rate", 0.08)
    _cfg_mr = _cfg_bt.get("short_margin_ratio", 1.0)
    short_costs = ShortCostModel(
        borrow_rate=0.0 if args.no_short_cost else (args.borrow_rate if args.borrow_rate is not None else _cfg_br),
        margin_ratio=args.margin_ratio if args.margin_ratio is not None else _cfg_mr,
    )
    if short_costs.borrow_rate > 0:
        log.info("空头腿成本: 借券费年化 %.2f%% 按日计提, 保证金比例 %.1f%% (--no-short-cost 可关闭)",
                 short_costs.borrow_rate * 100, short_costs.margin_ratio * 100)
    else:
        log.info("空头腿成本: 已关闭（借券费=0，旧口径）")

    # 确定因子列表
    if args.factors:
        if args.factors == "all":
            factor_list = list(ALL_FACTORS.keys())
        else:
            factor_list = [f.strip() for f in args.factors.split(",")]
    elif args.factor:
        factor_list = [args.factor]
    else:
        factor_list = ["momentum_20"]

    log.info("=== YuriQuant 回测 ===")
    log.info("因子: %s", factor_list)

    # 1. 数据
    executable_mask = None
    if args.real:
        log.info("加载真实本地数据 ...")
        from data import get_cache
        cache = get_cache()
        from config import Config
        cfg = Config.get()
        begin = cfg["fetch"]["begin_date"]
        # `end_date: null` means "latest available trading day".  Do not use
        # a fixed date here: it can be earlier than `begin` as the configuration
        # rolls forward, which turns a real-data run into an invalid range.
        end = cfg["fetch"].get("end_date")
        if end is None:
            calendar = cache.get_calendar(begin)
            if not calendar:
                raise RuntimeError(f"No trading days are available on or after {begin}.")
            end = calendar[-1]
        end = int(end)
        if begin > end:
            raise ValueError(
                f"fetch.begin_date ({begin}) must not be later than fetch.end_date ({end})."
            )
        from data.tradability import build_executable_mask
        from data.universe import Universe
        from data.industry import IndustryClassification
        from data.market_cap import build_market_cap_panel
        from data.cache_helpers import _pit_universe_codes, _apply_membership_mask
        uni = Universe(cache)
        # PIT 口径（2026-08-13 统一）：历史在册成分并集池 + 按日成分归属，
        # 消除「当前成分回看」的幸存者偏差。
        codes = _pit_universe_codes(uni, cfg["universe"]["index_code"], begin, end)
        log.info("PIT 并集池: %d 只（%s~%s 期间在册）", len(codes), begin, end)
        kline = cache.get_daily_kline(codes, begin, end)
        kline = _apply_membership_mask(kline, uni, cfg["universe"]["index_code"])
        close_raw = kline["close"].unstack("code")
        high = kline["high"].unstack("code")
        low = kline["low"].unstack("code")
        volume = kline["volume"].unstack("code")
        amount = kline["amount"].unstack("code")

        # 复权：按 config.universe.adjust 决定是否把原始价转成后复权价。
        # 注意：市值面板用【未复权】收盘价计算，避免后复权价把历史市值放大。
        adjust_mode = cfg.get("universe", {}).get("adjust", "backward")
        close = close_raw
        if adjust_mode == "backward":
            log.info("拉取后复权因子并应用到价格 ...")
            backward = cache.get_backward_factor(codes)
            backward = backward.reindex(index=close.index, columns=close.columns).ffill()
            close = close * backward
            high = high * backward
            low = low * backward
            log.info("后复权因子行数: %d, 列数: %d", len(backward), backward.shape[1])
        elif adjust_mode != "none":
            log.warning("未知的 universe.adjust=%s，跳过复权", adjust_mode)

        panel = {"close": close, "high": high, "low": low, "volume": volume, "amount": amount}

        # 中性化所需的行业面板与市值面板（仅当配置开启且数据可用时构建）。
        market_cap_panel: pd.DataFrame | None = None
        industry_panel: pd.DataFrame | None = None
        pre_cfg = cfg.get("preprocessing", {})
        need_neutralize = (
            not args.no_preprocess
            and pre_cfg.get("enabled", True)
            and (pre_cfg.get("neutralize_industry", True) or pre_cfg.get("neutralize_size", True))
        )
        if need_neutralize:
            if pre_cfg.get("neutralize_size", True):
                log.info("构建市值面板（流通/总市值，基于未复权收盘价）...")
                equity_structure = cache.get_equity_structure(codes)
                if equity_structure is None or equity_structure.empty:
                    log.warning("本地无股本结构数据，市值中性化将被跳过（请先跑 update_data）")
                else:
                    market_cap_panel = build_market_cap_panel(
                        equity_structure, close_raw, share_field=pre_cfg.get("market_cap_field", "tot_share")
                    )
                    if market_cap_panel.dropna(how="all").empty:
                        log.warning("市值面板全为空（股本与行情日期无重叠？），市值中性化将被跳过")
                        market_cap_panel = None
            if pre_cfg.get("neutralize_industry", True):
                log.info("构建行业分类面板 ...")
                ind = IndustryClassification(cache, level=int(pre_cfg.get("industry_level", 1)))
                industry_panel = ind.get_industry_panel(codes, close.index)
                if industry_panel is None or industry_panel.dropna(how="all").empty:
                    log.warning("本地无行业分类数据，行业中性化将被跳过（请先跑 update_data）")
                    industry_panel = None

        # 可执行性掩码：涨跌停/停牌（成分归属已由 _apply_membership_mask 处理）
        log.info("拉取历史涨跌停/停牌状态 ...")
        status = cache.get_history_stock_status(codes, begin, end)
        executable_mask = build_executable_mask(
            status, close.index, close.columns, close_panel=close
        )
        log.info(
            "可执行性掩码: 平均每日可交易 %.1f / %d 只",
            executable_mask.sum(axis=1).mean(), len(close.columns),
        )
    else:
        log.info("使用 Mock 数据 ...")
        panel = gen_mock_data()
        market_cap_panel = None
        industry_panel = None

    # 预处理配置（mock 模式下没有行业/市值数据，中性化会自动跳过）
    from config import Config as _Config
    pre_cfg = dict(_Config.get().get("preprocessing", {}))
    if args.no_preprocess:
        pre_cfg["enabled"] = False
    if pre_cfg.get("enabled", True):
        log.info(
            "因子预处理: winsorize=%s, neutralize=[industry=%s, size=%s], standardize=%s",
            pre_cfg.get("winsorize", "mad"),
            industry_panel is not None and pre_cfg.get("neutralize_industry", True),
            market_cap_panel is not None and pre_cfg.get("neutralize_size", True),
            pre_cfg.get("standardize", "zscore"),
        )
    else:
        log.info("因子预处理: 已关闭（使用原始因子值）")

    # 2. 收益率面板
    returns_panel = panel["close"].pct_change().shift(-1)

    # 3. 批量回测
    results = {}
    factor_summaries = {}
    for fn in factor_list:
        log.info("回测因子: %s (策略: %s, 调仓: %s)", fn, args.strategy, args.freq)
        name, result, fs = run_single_factor(
            fn, panel, args.strategy, args.k, args.freq, returns_panel, executable_mask,
            preprocess_cfg=pre_cfg,
            market_cap_panel=market_cap_panel,
            industry_panel=industry_panel,
            short_costs=short_costs,
            deleverage=args.deleverage,
        )
        results[name] = result
        factor_summaries[name] = fs
        log.info("  年化收益: %.2f%%, 夏普: %.4f, IC: %.4f",
                 result.metrics()["annual_return"] * 100,
                 result.metrics()["sharpe"],
                 fs.get("ic_mean", 0))

    # 4. 报告（Excel + 交互式 HTML）
    report_dir = Path("reports")
    xlsx_path = report_dir / "yuriquant_report.xlsx"
    generate_excel_report(results, factor_summaries, output_path=xlsx_path)
    log.info("Excel 报告: %s", xlsx_path.resolve())

    if not args.no_html:
        html_path = report_dir / "yuriquant_report.html"
        generate_html_report(
            results, factor_summaries, output_path=html_path,
            title="YuriQuant 因子回测报告",
            meta=f"策略={args.strategy} k={args.k} 调仓={args.freq} · "
                 f"借券费={short_costs.borrow_rate*100:.0f}%/年 保证金={short_costs.margin_ratio*100:.0f}%"
                 + (" · 1倍资金降杠杆" if args.deleverage else ""),
        )
        log.info("HTML 报告: %s", html_path.resolve())

    # 5. 实验记录
    try:
        import sys
        from data.cache import DataCache
        from data.datasource import create_datasource
        from research.experiments import record_experiment
        fingerprint = DataCache(create_datasource()).get_fingerprint()
        metrics_summary = {
            fn: {"annual_return": round(results[fn].metrics().get("annual_return", 0.0), 4),
                 "sharpe": round(results[fn].metrics().get("sharpe", 0.0), 4),
                 "ic_mean": round(factor_summaries[fn].get("ic_mean", 0.0), 4)}
            for fn in factor_list if fn in results
        }
        record_experiment(
            kind="backtest",
            command=" ".join(sys.argv),
            params={"real": args.real, "factors": factor_list,
                    "strategy": args.strategy, "k": args.k, "freq": args.freq,
                    "no_preprocess": args.no_preprocess,
                    "short_borrow_rate": short_costs.borrow_rate,
                    "short_margin_ratio": short_costs.margin_ratio,
                    "deleverage": args.deleverage,
                    "html_report": not args.no_html},
            data_fingerprint=fingerprint,
            result_path=str(xlsx_path),
            metrics=metrics_summary,
        )
    except Exception as e:
        log.warning("实验记录写入失败（不影响结果）: %s", e)

    log.info("=== 完成 ===")

if __name__ == "__main__":
    main()