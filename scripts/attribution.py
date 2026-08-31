"""
收益归因 CLI
============

三大归因框架的命令行入口（研究导向，纯 pandas/numpy 实现）：

    python scripts/attribution.py fm --mock                  # Fama-MacBeth（mock 演示）
    python scripts/attribution.py brinson --mock             # Brinson 归因（mock 演示）
    python scripts/attribution.py ab --mock                  # α/β 分解（mock 演示）

真实面板输入（均为 date×code 面板 CSV，index=date, columns=code）：

    python scripts/attribution.py fm \
        --factor-csv reports/factor_panel.csv --returns-csv reports/returns_future.csv
    python scripts/attribution.py brinson \
        --returns-csv reports/returns.csv --port-weights-csv reports/port_w.csv \
        --bench-weights-csv reports/bench_w.csv --industry-csv reports/industry.csv
    python scripts/attribution.py ab \
        --portfolio-csv reports/port_returns.csv --benchmark-csv reports/bench_returns.csv \
        [--factor-csv reports/factor_returns.csv] [--rf 0.02]

mock 模式：生成含已知因子溢价的合成数据，验证三条归因链路能正确恢复信号。
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

from scripts.cli_common import setup_logging  # noqa: E402


log = setup_logging("attribution")

# ---------------------------------------------------------------------------
# Mock 数据：含已知因子溢价 + 行业结构，验证归因能恢复信号
# ---------------------------------------------------------------------------
def gen_mock_attribution_data(
    n_days: int = 300, n_codes: int = 60, seed: int = 7,
) -> dict:
    """合成含已知结构的市场数据：

    - 60 只股票分 3 个行业（食品饮料/银行/电子），行业基准收益不同（电子最高）；
    - 个股收益 = 市场 + 行业漂移 + size 溢价×暴露 + 特质噪声；
    - size 溢价 = +0.001/日（Fama-MacBeth 应恢复 ≈0.001）；
    - 组合 = 每月调仓、选 size 暴露最大 20 只等权；基准 = 全市场等权。
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    industries = ["食品饮料", "银行", "电子"]
    cat = {c: industries[i % 3] for i, c in enumerate(codes)}
    ind_ret = {"食品饮料": 0.0004, "银行": -0.0001, "电子": 0.0008}

    size_i = rng.uniform(0.5, 1.5, n_codes)          # 市值因子暴露
    alpha_i = rng.normal(0, 0.0005, n_codes)
    size_prem = 0.001                                 # 每单位暴露的日溢价（已知真值）

    market = rng.normal(0.0005, 0.01, n_days)
    rets = np.zeros((n_days, n_codes))
    for t in range(n_days):
        for i in range(n_codes):
            rets[t, i] = (market[t] + ind_ret[cat[codes[i]]] + size_prem * size_i[i]
                          + alpha_i[i] + rng.normal(0, 0.008))
    returns = pd.DataFrame(rets, idx, codes)          # 当期收益（Brinson/αβ 用）

    # 因子面板（FM 用）：size 暴露（标准化）+ 动量（滞后收益）
    size_panel = pd.DataFrame(np.tile(size_i, (n_days, 1)), idx, codes)
    size_panel = size_panel.sub(size_panel.mean(axis=1), axis=0).div(size_panel.std(axis=1), axis=0)
    mom_panel = returns.shift(1)
    returns_future = returns.shift(-1)                # 未来一期收益（FM/IC 口径）

    # 组合权重：每月首个交易日选 size 暴露最大的 20 只等权；基准 = 全市场等权
    port_w = pd.DataFrame(np.nan, index=idx, columns=codes)
    bench_w = pd.DataFrame(np.nan, index=idx, columns=codes)
    months = pd.Series(idx).groupby(idx.to_period("M")).first()
    for d in months:
        top = size_panel.loc[d].nlargest(20).index
        port_w.loc[d, top] = 1.0 / 20
        bench_w.loc[d, :] = 1.0 / n_codes
    # 组合收益序列（α/β 用）：按权重×当期收益
    w_ff = port_w.ffill().fillna(0.0)
    port_ret = (returns * w_ff).sum(axis=1)
    bench_ret = returns.mean(axis=1)

    return {
        "returns": returns, "returns_future": returns_future,
        "size": size_panel, "momentum": mom_panel,
        "port_weights": port_w, "bench_weights": bench_w,
        "category": cat, "portfolio_returns": port_ret, "benchmark_returns": bench_ret,
        "size_premium_true": size_prem,
    }

# ---------------------------------------------------------------------------
# 面板 IO
# ---------------------------------------------------------------------------
def _load_panel(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

def _load_series(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]

def _load_category(path: str) -> dict:
    df = pd.read_csv(path)
    col_c = "code" if "code" in df.columns else df.columns[0]
    col_i = "industry" if "industry" in df.columns else df.columns[1]
    return dict(zip(df[col_c].astype(str), df[col_i].astype(str)))

# ---------------------------------------------------------------------------
# 子命令：Fama-MacBeth
# ---------------------------------------------------------------------------
def cmd_fm(args) -> int:
    if args.mock:
        d = gen_mock_attribution_data()
        panels = {"size": d["size"], "momentum": d["momentum"]}
        returns = d["returns_future"]
        log.info("mock 数据：size 溢价真值 = %.4f/日", d["size_premium_true"])
    else:
        panels = {}
        for spec in args.factor_csv.split(","):
            p = spec.strip()
            name = Path(p).stem
            panels[name] = _load_panel(p)
        returns = _load_panel(args.returns_csv)
        if not returns.index.equals(panels[list(panels)[0]].index):
            log.warning("returns 与 factor 面板日期不完全一致，将自动对齐")

    from research.attribution import fama_macbeth
    res = fama_macbeth(panels, returns, lag=args.lag)
    if res.empty:
        log.error("无有效横截面观测，无法估计（检查面板对齐）")
        return 1

    print("\n===== Fama-MacBeth 因子溢价（Newey-West 稳健推断）=====")
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.5f}"):
        print(res.round(5).to_string())
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out)
        log.info("结果已保存: %s", out)
    return 0

# ---------------------------------------------------------------------------
# 子命令：Brinson 归因
# ---------------------------------------------------------------------------
def cmd_brinson(args) -> int:
    if args.mock:
        d = gen_mock_attribution_data()
        returns, pw, bw, cat = d["returns"], d["port_weights"], d["bench_weights"], d["category"]
    else:
        returns = _load_panel(args.returns_csv)
        pw = _load_panel(args.port_weights_csv)
        bw = _load_panel(args.bench_weights_csv)
        cat = _load_category(args.industry_csv)

    from research.attribution import brinson_attribution
    df, summary = brinson_attribution(returns, pw, bw, cat, freq=args.freq)
    if summary["n_periods"] == 0:
        log.error("无有效归因周期（检查权重/收益面板对齐）")
        return 1

    print("\n===== Brinson-Fachler 归因（Carino 链接，%s 频）=====" % args.freq)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.5f}"):
        print(df.round(5).to_string())
    s = summary
    print("\n组合收益: {:.4f}  基准收益: {:.4f}  超额: {:.4f}".format(
        s["portfolio_return"], s["benchmark_return"], s["active_return"]))
    print("配置效应: {:.4f}  选择效应: {:.4f}  交互效应: {:.4f}  还原残差: {:.2e}".format(
        s["allocation"], s["selection"], s["interaction"], s["recon_error"]))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out)
        log.info("结果已保存: %s", out)
    return 0

# ---------------------------------------------------------------------------
# 子命令：α/β 分解
# ---------------------------------------------------------------------------
def cmd_ab(args) -> int:
    if args.mock:
        d = gen_mock_attribution_data()
        pr, br = d["portfolio_returns"], d["benchmark_returns"]
        log.info("mock 数据：真实 beta = 1.0（组合为市值加权近似）")
    else:
        pr = _load_series(args.portfolio_csv)
        br = _load_series(args.benchmark_csv)

    from research.attribution import alpha_beta
    fdf = None
    if args.factor_csv:
        fdf = _load_panel(args.factor_csv)
    res = alpha_beta(pr, br, rf=args.rf, factor_returns=fdf, lag=args.lag)
    if res["n"] < 3:
        log.error("有效观测过少，无法估计")
        return 1

    print("\n===== α/β 分解（Newey-West 稳健推断）=====")
    print(f"alpha(日):   {res['alpha']:+.6f}    alpha(年化): {res['alpha_annual']:+.4f}")
    print(f"alpha t_nw:  {res['alpha_t_nw']:+.3f}   (p={res['alpha_p_nw']:.4f})")
    print(f"beta(市场):  {res['beta']:+.4f}    beta t_nw: {res['beta_t_nw']:+.3f}")
    if not res["factors"].empty:
        print("\n额外因子:")
        with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:.5f}"):
            print(res["factors"].round(5).to_string())
    print(f"R²: {res['r2']:.4f}   观测: {res['n']}   NW滞后: {res['lag_used']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = pd.DataFrame([{
            "alpha": res["alpha"], "alpha_annual": res["alpha_annual"],
            "alpha_t_nw": res["alpha_t_nw"], "alpha_p_nw": res["alpha_p_nw"],
            "beta": res["beta"], "beta_t_nw": res["beta_t_nw"],
            "r2": res["r2"], "n": res["n"],
        }])
        rows.to_csv(out, index=False)
        log.info("结果已保存: %s", out)
    return 0

# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="YuriQuant 收益归因")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fm = sub.add_parser("fm", help="Fama-MacBeth 因子溢价")
    p_fm.add_argument("--mock", action="store_true", help="mock 演示数据")
    p_fm.add_argument("--factor-csv", help="因子面板 CSV（逗号分隔多个）")
    p_fm.add_argument("--returns-csv", help="未来一期收益面板 CSV")
    p_fm.add_argument("--lag", type=int, default=None, help="Newey-West 滞后（默认自动）")
    p_fm.add_argument("--out", default=None)
    p_fm.set_defaults(func=cmd_fm)

    p_br = sub.add_parser("brinson", help="Brinson 收益归因")
    p_br.add_argument("--mock", action="store_true")
    p_br.add_argument("--returns-csv", help="当期收益面板 CSV")
    p_br.add_argument("--port-weights-csv", help="组合期初权重面板 CSV")
    p_br.add_argument("--bench-weights-csv", help="基准期初权重面板 CSV")
    p_br.add_argument("--industry-csv", help="行业映射 CSV（code, industry 两列）")
    p_br.add_argument("--freq", default="M", choices=["D", "W", "M"])
    p_br.add_argument("--out", default=None)
    p_br.set_defaults(func=cmd_brinson)

    p_ab = sub.add_parser("ab", help="α/β 分解（CAPM/多因子）")
    p_ab.add_argument("--mock", action="store_true")
    p_ab.add_argument("--portfolio-csv", help="组合日收益 CSV")
    p_ab.add_argument("--benchmark-csv", help="基准日收益 CSV")
    p_ab.add_argument("--factor-csv", help="额外因子收益面板 CSV（可选）")
    p_ab.add_argument("--rf", type=float, default=0.0, help="年化无风险利率")
    p_ab.add_argument("--lag", type=int, default=None)
    p_ab.add_argument("--out", default=None)
    p_ab.set_defaults(func=cmd_ab)

    args = parser.parse_args()
    rc = args.func(args)
    # 实验记录（attribution 分析跑完写一行，便于回溯）
    try:
        import sys
        from research.experiments import record_experiment
        record_experiment(
            kind="attribution",
            command=" ".join(sys.argv),
            params={"cmd": args.cmd, "mock": args.mock,
                    "freq": getattr(args, "freq", None),
                    "lag": getattr(args, "lag", None),
                    "rf": getattr(args, "rf", 0.0)},
            result_path=args.out or "",
            note=f"{args.cmd} 归因",
        )
    except Exception:
        pass
    raise SystemExit(rc)

if __name__ == "__main__":
    main()