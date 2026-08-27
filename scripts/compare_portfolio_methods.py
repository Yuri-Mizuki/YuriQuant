"""
组合优化方法对比脚本（P2，2026-08-14）
=====================================

五种组合构建方法统一对比（同一因子/收益/行业/风格面板）：

  1. projection   启发式投影（optimize_weights factor_weighted + 约束）—— 现状基线
  2. min_var      QP 最小方差（solver.optimize_weights_qp，滚动 Ledoit-Wolf Σ）
  3. tev          QP 跟踪误差（基准=等权，λ=1.0）
  4. risk_parity  QP 风险平价（等风险预算）
  5. hrp          HRP（层次聚类+递归二分，免逆矩阵）

评估口径（与回测引擎一致，防前视）：
  - weights[t] 在 t 日设定，赚 returns[t]（t→t+1 收益）；
  - Σ 估计只用 < t 的收益（solver 内部保证）；
  - 跳过前 window 天（协方差窗口冷启动期）。
指标：年化收益/波动/夏普/最大回撤/换手 + 因子 rank IC（与因子研究口径一致）。

用法：
    python scripts/compare_portfolio_methods.py                 # mock 默认
    python scripts/compare_portfolio_methods.py --methods projection,min_var,hrp
    python scripts/compare_portfolio_methods.py --n-days 500 --n-codes 60 --seed 7
    python scripts/compare_portfolio_methods.py --out reports/portfolio_methods_compare.csv

    # 真实数据（需先 python -m scripts.update_data 拉取日线/行业/股本；PIT 并集池口径）
    python scripts/compare_portfolio_methods.py --real --begin 20230101 --end 20241231
    # 四窗口对照（2019-2020 / 2021-2022 / 2023-2024 / 2025-2026）：
    for w in "20190101 20201231" "20210101 20221231" "20230101 20241231" "20250101 20261231"; do
      set -- $w
      python scripts/compare_portfolio_methods.py --real --begin $1 --end $2 \
        --out reports/portfolio_methods_real_$1_$2.csv
    done
    真实因子默认用「滞后收益（动量代理）+ MAD 去极值」，可自行替换为因子库合成因子。
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import calc_all_metrics
from factor.preprocessing import winsorize_mad
from optimize.portfolio import optimize_weights
from optimize.solver import optimize_weights_hrp, optimize_weights_qp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("compare_portfolio_methods")
# OSQP/SCS 的 "Solution may be inaccurate" 是数值公差提示，对比脚本里属正常噪音
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

WINDOW = 120
MIN_PERIODS = 60
OUT_DEFAULT = Path("reports/portfolio_methods_compare.csv")


# ---------------------------------------------------------------------------
# Mock 数据：收益 = 行业因子 + 风格因子 + AR(1) 信号 + 噪声
# ---------------------------------------------------------------------------
def gen_mock_panel(n_days: int = 400, n_codes: int = 50, seed: int = 7) -> dict[str, pd.DataFrame]:
    """生成含可挖掘信号的面板：因子（滞后收益）与收益有真实相关性。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    n_ind = 3
    ind_map = {c: f"ind{i % n_ind}" for i, c in enumerate(codes)}
    ind_f = rng.normal(0, 0.01, (n_days, n_ind))  # 行业因子收益
    style = pd.DataFrame(  # 市值风格暴露（随机游走，截面 zscore）
        rng.normal(0, 1, (n_days, n_codes)).cumsum(axis=0), idx, codes,
    )
    style = style.sub(style.mean(axis=1), axis=0).div(style.std(axis=1), axis=0)
    style_panels = {"mktcap": style}
    idc = np.array([ind_map[c][3:] for c in codes], dtype=int)
    idc = np.array([int(i) for i in idc])
    sig = np.zeros((n_days, n_codes))
    sig[1:] = 0.15 * (style.iloc[:-1].values * 0.0 + rng.normal(0, 1, (n_days - 1, n_codes)))
    # AR(1) 可预测成分：sig_t = 0.2*sig_{t-1} + eps
    for t in range(1, n_days):
        sig[t] = 0.2 * sig[t - 1] + rng.normal(0, 1, n_codes) * 0.3
    rets = (
        ind_f[:, idc] * 0.003
        + style.values * 0.002
        + 0.008 * sig
        + rng.normal(0, 0.015, (n_days, n_codes))
    )
    returns = pd.DataFrame(rets, idx, codes)
    factor = returns.shift(1).div(returns.std().clip(lower=1e-6), axis=1)  # 滞后收益因子
    # 合成未复权收盘价（10×累积收益），仅供信号层换算股数（mock 演示，不反映真实价）
    close = 10.0 * (1.0 + returns).cumprod()
    return {"returns": returns, "factor": factor, "industry_map": ind_map,
            "style_panels": style_panels, "close": close}


# ---------------------------------------------------------------------------
# 真实数据（--real）：PIT 并集池 + 行业/市值面板（复用 run_backtest 口径）
# ---------------------------------------------------------------------------
def load_real_data(
    begin: int, end: int, index_code: str,
    sdk_cache: str | None = None, cache_root: str | None = None,
) -> dict[str, pd.DataFrame]:
    """加载真实 HS300 数据：收益面板（未复权）、行业面板（PIT）、市值风格面板。

    sdk_cache: SDK 本地缓存目录（覆盖 settings.yaml 的 sdk_local_path）。
    cache_root: parquet 缓存目录（覆盖默认 e:/data/parquet）。
        e:\\data 被外部进程锁定时（2026-08-14 实测 h5/parquet 写 PermissionError），
        传 workspace 内新目录（如 ./.cache_p2，先复制所需表过去），
        SDK 元数据下载会写入新目录而不再撞锁。
    """
    from config import Config
    from data.cache import DataCache
    from data.cache_helpers import _apply_membership_mask, _pit_universe_codes
    from data.datasource import create_datasource
    from data.industry import IndustryClassification
    from data.market_cap import build_market_cap_panel
    from data.universe import Universe

    cfg = Config.get()
    ds_cfg = dict(Config.datasource())  # {type, amazing_data, csv}
    if sdk_cache:
        ds_cfg["amazing_data"]["sdk_local_path"] = sdk_cache
    ds = create_datasource(ds_cfg)
    cache = DataCache(ds, cache_root=cache_root) if cache_root else DataCache(ds)
    uni = Universe(cache)
    codes = _pit_universe_codes(uni, index_code, begin, end)
    log.info("PIT 并集池: %d 只（%s~%s）", len(codes), begin, end)
    kline = cache.get_daily_kline(codes, begin, end)
    kline = _apply_membership_mask(kline, uni, index_code)
    close_raw = kline["close"].unstack("code").sort_index()
    # 对比脚本用【未复权】收盘价算收益：目的在方法差异，复权与否影响极小；
    # 且 get_backward_factor 会触发 SDK 写 sdk_cache h5（e:\data 在 workspace 外，
    # 2026-08-14 实测 h5 被锁 PermissionError）。真实复权口径请走 run_backtest。
    close = close_raw
    returns = close.pct_change()
    # 停牌/缺失 → 0 收益（ffill 停牌段价格不变；fillna 处理上市初期）。
    # PIT 并集池必然含停牌/新股/退市异常价（pct_change 可能 inf），
    # 若不处理，dropna(how='any') 后共同交易日过少导致协方差估计失效
    # （QP 方法全 0），inf 会让 OSQP 直接 NaN 崩溃。
    returns = returns.ffill().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    # 剔除整个窗口零波动列（上市前/长期停牌被 fillna(0) 污染 → 协方差奇异，
    # LedoitWolf 产出 NaN 让 OSQP 崩溃；这些股票本就不该进入组合）。
    returns = returns.loc[:, returns.std() > 1e-10]

    # 行业面板（PIT，date×code → 行业名）
    ind = IndustryClassification(
        cache, level=int(cfg.get("preprocessing", {}).get("industry_level", 1)),
    )
    industry_panel = ind.get_industry_panel(codes, close.index)

    # 市值风格面板：log 市值逐日 zscore（风格中性化用）
    # 股本结构直接读本地 parquet（绕开 SDK：get_equity_structure 每次全量下载
    # 写 h5/parquet，2026-08-14 实测 e:\data 间歇性写锁 PermissionError）。
    pre_cfg = cfg.get("preprocessing", {})
    es_path = (Path(cache_root) if cache_root else Path("e:/data/parquet")) / "equity_structure.parquet"
    if es_path.exists():
        equity_structure = pd.read_parquet(es_path)
    else:
        equity_structure = pd.DataFrame()
    mcap = build_market_cap_panel(
        equity_structure, close_raw,
        share_field=pre_cfg.get("market_cap_field", "tot_share"),
    )
    style = np.log(mcap.clip(lower=1e6))
    std = style.std(axis=1).replace(0, np.nan)
    style = style.sub(style.mean(axis=1), axis=0).div(std, axis=0).fillna(0.0)
    return {
        "returns": returns,
        "industry_panel": industry_panel,
        "style_panels": {"mktcap": style},
        "codes": pd.Series(codes, name="code"),
        "close_raw": close_raw,
    }


# ---------------------------------------------------------------------------
# 各方法权重（统一面板输出 date×code）
# ---------------------------------------------------------------------------
def run_methods(data: dict, methods: list[str]) -> dict[str, pd.DataFrame]:
    factor, returns = data["factor"], data["returns"]
    ind_map = data.get("industry_map")
    ind_panel = data.get("industry_panel")
    styles = data.get("style_panels")
    bench = pd.Series(1.0 / len(factor.columns), index=factor.columns)
    # 统一换手约束（0.5·Σ|Δw| ≤ 0.5）：所有方法公平对比，避免日频重估换手失控
    prev0 = pd.Series(0.0, index=factor.columns)
    out: dict[str, pd.DataFrame] = {}
    common = {"window": WINDOW, "min_periods": MIN_PERIODS,
              "prev_weights": prev0, "max_turnover": 0.5,
              "industry_panel": ind_panel}
    for m in methods:
        if m == "projection":
            w = optimize_weights(
                factor, method="factor_weighted",
                industry_map=ind_map, max_weight=0.1,
                prev_weights=prev0, max_turnover=0.5,
            )
        elif m == "min_var":
            w = optimize_weights_qp(
                factor, returns, method="min_var", industry_map=ind_map,
                style_exposures=styles, **common,
            )
        elif m == "tev":
            w = optimize_weights_qp(
                factor, returns, method="tev", risk_aversion=1.0,
                benchmark_weights=bench, industry_map=ind_map,
                style_exposures=styles, **common,
            )
        elif m == "risk_parity":
            w = optimize_weights_qp(
                factor, returns, method="risk_parity", industry_map=ind_map, **common,
            )
        elif m == "hrp":
            # HRP 为低频风险结构方法（免逆矩阵），不接换手约束
            w = optimize_weights_hrp(returns, window=WINDOW, min_periods=MIN_PERIODS)
        else:
            raise ValueError(f"未知方法 {m!r}")
        out[m] = w
    return out


# ---------------------------------------------------------------------------
# 评估（与回测口径一致：weights[t] 赚 returns[t]）
# ---------------------------------------------------------------------------
def evaluate(weights: pd.DataFrame, data: dict) -> dict[str, float]:
    returns, factor = data["returns"], data["factor"]
    w = weights.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    w = w.iloc[MIN_PERIODS:]  # 跳过协方差窗口冷启动期
    r = returns.reindex(index=w.index).values
    daily = pd.Series((w.values * r).sum(axis=1), index=w.index)
    m = calc_all_metrics(daily, weights_history=w)
    turnover = float(0.5 * w.diff().abs().sum(axis=1).mean())
    f = factor.reindex(index=w.index)
    ic = float(
        f.rank(axis=1).corrwith(returns.reindex(index=w.index).rank(axis=1), axis=1).mean()
    )
    return {
        "annual_return": m["annual_return"],
        "annual_volatility": m["annual_volatility"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "turnover": turnover,
        "rank_ic": ic,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="组合优化方法对比（P2）")
    ap.add_argument("--methods", default="projection,min_var,tev,risk_parity,hrp",
                    help="逗号分隔方法列表")
    ap.add_argument("--n-days", type=int, default=400)
    ap.add_argument("--n-codes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--real", action="store_true", help="真实本地数据（需先 update_data）")
    ap.add_argument("--begin", type=int, default=None, help="真实数据起始日 YYYYMMDD")
    ap.add_argument("--end", type=int, default=None, help="真实数据结束日 YYYYMMDD")
    ap.add_argument("--index", default="000300.SH", help="指数代码（默认沪深300）")
    ap.add_argument("--sdk-cache", default=None,
                    help="SDK 本地缓存目录（默认 settings.yaml 的 sdk_local_path；"
                         "e:\\data\\sdk_cache 被锁定时传新目录，如 ./.sdk_cache_p2）")
    ap.add_argument("--cache-root", default=None,
                    help="parquet 缓存目录（默认 e:/data/parquet；被锁定时传 workspace 内"
                         "新目录，如 ./.cache_p2，需先复制所需表过去）")
    args = ap.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.real:
        from config import Config
        cfg = Config.get()
        begin = args.begin or cfg["fetch"]["begin_date"]
        end = args.end
        if end is None:
            from data import get_cache
            cal = get_cache().get_calendar(begin)
            end = cal[-1] if cal else begin
        data = load_real_data(begin, int(end), args.index,
                              sdk_cache=args.sdk_cache, cache_root=args.cache_root)
        returns = data["returns"]
        # 真实因子：滞后收益（动量代理）+ MAD 去极值；可替换为因子库合成因子
        factor = winsorize_mad(returns.shift(1))
        factor = factor.reindex(index=returns.index, columns=returns.columns)
        data["factor"] = factor
        log.info("真实数据: %d 天 × %d 只（%s~%s），因子=滞后收益动量代理",
                 len(returns), len(returns.columns), begin, end)
    else:
        data = gen_mock_panel(n_days=args.n_days, n_codes=args.n_codes, seed=args.seed)
        log.info("mock 面板: %d 天 × %d 股（%s）", args.n_days, args.n_codes, ", ".join(methods))

    weights = run_methods(data, methods)
    rows = {}
    for m, w in weights.items():
        rows[m] = evaluate(w, data)
    df = pd.DataFrame(rows).T
    df.index.name = "method"
    df["annual_return"] = df["annual_return"] * 100
    df["annual_volatility"] = df["annual_volatility"] * 100
    df["max_drawdown"] = df["max_drawdown"] * 100
    df["rank_ic"] = df["rank_ic"] * 100
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, encoding="utf-8-sig")
    log.info("已输出: %s", out_path)
    print("\n组合优化方法对比（年化收益/波动/回撤单位 %%，IC 单位 %%）")
    print(df.round(2).to_string())


if __name__ == "__main__":
    main()
