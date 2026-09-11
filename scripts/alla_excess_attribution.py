"""全A主策略超额收益归因：α/β 分解 + Brinson 行业归因。

复跑滚动网格主策略（gbdt × h1 × M × raw × TopFrac 10%）拿到每日权重
（stage_backtest 未落盘权重，引擎内部有记录，重跑一次是唯一无损路径），
然后回答两个问题：
  1. 超额收益有多少来自选股 α、多少只是市场敞口 β？
     —— alpha_beta() 对上证指数与全A等权各回归一次，α 带 Newey-West t 检验；
  2. 超额在行业内部（选股）还是行业之间（配置）？
     —— brinson_attribution() vs 全A等权、按申万一级行业分组（Carino 链接）。

输出 reports/alla_attribution/：
  alpha_beta.csv / brinson_industry.csv / summary.md / attribution.html

用法：uv run python -m scripts.alla_excess_attribution [--frac 0.10]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.cli_common import setup_logging
from research.html_report import page

log = setup_logging("alla_excess_attribution")

OUT = Path("reports") / "alla_attribution"


# ---------------------------------------------------------------------------
# 复跑主策略回测（口径严格对照 rolling_grid_alla.stage_backtest h=1 路径）
# ---------------------------------------------------------------------------
def run_main_strategy_backtest(frac: float = 0.10):
    """重跑单组合回测，返回 (日收益, 权重历史, base dict, OOS 交易日)。"""
    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest
    from scripts.rolling_grid_alla import load_base
    from strategy.examples import TopFracLongOnly

    base = load_base()
    close = base["close"]
    pred_path = Path("reports") / "alla_rolling" / "pred" / "gbdt__h1.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} 不存在，先跑 rolling_grid_alla --stage predict")
    pred = pd.read_parquet(pred_path)
    oos_days = pred.index
    fwd = close.pct_change(fill_method=None).loc[oos_days]  # h=1：未 shift 口径
    mask_oos = base["mask"].astype(bool).reindex(
        index=oos_days, columns=close.columns).fillna(True)

    strat = TopFracLongOnly(frac=frac, weight_mode="equal")
    bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                        initial_capital=1_000_000.0, costs=default_costs())
    res = bt.run(pred, fwd, executable_mask=mask_oos, horizon=1)
    dr = res.daily_returns.dropna()
    ann = (1 + dr).prod() ** (252 / len(dr)) - 1
    log.info("回测复跑完成：%d 日（%s ~ %s），年化 %.2f%%",
             len(dr), dr.index[0].date(), dr.index[-1].date(), ann * 100)
    return dr, res.weights_history, base, oos_days


# ---------------------------------------------------------------------------
# 基准权重与行业映射
# ---------------------------------------------------------------------------
def equal_weight_bench(returns_panel: pd.DataFrame) -> pd.DataFrame:
    """全A等权基准的期初权重矩阵：当日有收益的股票各 1/N。

    与 bench_eqw（close.pct_change().mean(axis=1)，NaN 自动跳过）口径一致。
    """
    held = returns_panel.notna()
    return held.div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def industry_mapping(cov_industry: pd.DataFrame) -> pd.Series:
    """date×code 行业面板 → code → 行业映射（取每只股票最近非空值）。"""
    last = cov_industry.ffill().iloc[-1].dropna()
    if last.empty:  # 面板可能只有历史段，回退取全期首个非空
        last = cov_industry.bfill().iloc[-1].dropna()
    return last


# ---------------------------------------------------------------------------
# 归因主计算（可测：输入全部为内存面板）
# ---------------------------------------------------------------------------
def compute_attribution(dr: pd.Series, weights: pd.DataFrame,
                        base: dict, oos_days: pd.Index) -> dict:
    """α/β（vs 指数 + vs 全A等权）与 Brinson（vs 等权、行业分组）一次性计算。

    Args:
        dr: 组合日收益（已 dropna）。
        weights: 引擎输出的每日权重（res.weights_history）。
        base: load_base() 的 dict（close/cov/bench_index/bench_eqw）。
        oos_days: 回测 OOS 交易日（信号面板 index）。

    Returns:
        dict: ab_df（DataFrame，index=基准名）、br_df / br_summary（Brinson）、
              cmp_summary（vs 指数对照指标）。
    """
    from research.attribution import alpha_beta, brinson_attribution
    from research.benchmarks import compare_to_benchmark

    close = base["close"]
    bench_idx = base["bench_index"].reindex(dr.index).dropna()
    bench_eqw = base["bench_eqw"].reindex(dr.index).dropna()

    # --- α/β：组合 vs 上证指数 / vs 全A等权 ---
    ab_rows = {}
    for bname, bench in (("vs_上证指数", bench_idx), ("vs_全A等权", bench_eqw)):
        ab = alpha_beta(dr.reindex(bench.index), bench)
        ab_rows[bname] = {
            "alpha_annual": ab["alpha_annual"], "alpha_t_nw": ab["alpha_t_nw"],
            "alpha_p_nw": ab["alpha_p_nw"], "beta": ab["beta"],
            "beta_t_nw": ab["beta_t_nw"], "r2": ab["r2"], "n": ab["n"],
        }
    ab_df = pd.DataFrame(ab_rows).T

    # --- Brinson：vs 全A等权、申万一级 ---
    rets_panel = close.pct_change(fill_method=None).loc[oos_days]
    port_w = weights.reindex(index=oos_days, columns=close.columns).fillna(0.0)
    bench_w = equal_weight_bench(rets_panel)
    category = industry_mapping(base["cov"]["industry"])
    log.info("行业映射覆盖 %d / %d 只股票", len(category), port_w.shape[1])
    br_df, br_summary = brinson_attribution(
        rets_panel, port_w, bench_w, category.to_dict(), freq="M")

    # --- 与指数超额对照（跟踪误差/相关/β）---
    cmp_idx = compare_to_benchmark(dr.reindex(bench_idx.index), bench_idx)
    cmp_summary = {
        "excess_idx": cmp_idx.get("excess_annual", np.nan),
        "te_idx": cmp_idx.get("tracking_error", np.nan),
        "corr_idx": cmp_idx.get("correlation", np.nan),
        "beta_idx": cmp_idx.get("beta", np.nan),
    }
    return {"ab_df": ab_df, "br_df": br_df, "br_summary": br_summary,
            "cmp_summary": cmp_summary}


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
# 报告主题 CSS（经 research.html_report.page 外壳注入；本文件不再自拼 HTML 外壳）
_CSS = """
body{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#222;max-width:960px}
h1{font-size:20px} h2{font-size:16px;margin-top:28px}
table{border-collapse:collapse;margin:12px 0;font-size:13px}
th,td{border:1px solid #ddd;padding:5px 10px;text-align:right}
th{background:#f5f5f5} td:first-child,th:first-child{text-align:left}
.pos{color:#c0392b}.neg{color:#27ae60}
.note{color:#666;font-size:12px;margin:6px 0}
"""


def _fmt_pct(v: float, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    cls = "pos" if v >= 0 else "neg"
    return f"<span class='{cls}'>{v * 100:+.{digits}f}%</span>"


def _fmt_f(v: float, digits: int = 2) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.{digits}f}"


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def render_html(ab_df: pd.DataFrame, br_df: pd.DataFrame,
                br_summary: dict, cmp_summary: dict, out_path: Path):
    ab_rows = []
    for name, r in ab_df.iterrows():
        ab_rows.append([
            name, _fmt_pct(r["alpha_annual"]), _fmt_f(r["alpha_t_nw"]),
            f"{r['alpha_p_nw']:.4f}", _fmt_f(r["beta"]), _fmt_f(r["beta_t_nw"]),
            _fmt_f(r["r2"], 3), str(int(r["n"])),
        ])
    ab_tbl = _table(["基准", "年化α", "α t值(NW)", "α p值", "β", "β t值(NW)", "R²", "N"],
                    ab_rows)

    br_rows = []
    for cat, r in br_df.iterrows():
        br_rows.append([
            str(cat), _fmt_pct(r["allocation"]), _fmt_pct(r["selection"]),
            _fmt_pct(r["interaction"]), _fmt_pct(r["total"]),
        ])
    br_tbl = _table(["行业", "配置效应", "选择效应", "交互效应", "合计"], br_rows)

    body = f"""<h1>全A主策略超额收益归因（gbdt × h1 × 月频 × raw × Top10%）</h1>
<p class="note">组合 vs 上证指数：年化超额 {_fmt_pct(cmp_summary.get('excess_idx'))}，
跟踪误差 {_fmt_pct(cmp_summary.get('te_idx'))}，相关系数 {_fmt_f(cmp_summary.get('corr_idx'), 3)}。
Brinson 基准为全A等权，效应为 Carino 链接累计值</p>
<p class="note">（算术口径 recon_error={br_summary.get('recon_error', 0):.2e}）。</p>

<h2>α/β 分解（Newey-West t 检验）</h2>
{ab_tbl}
<p class="note">α 为日度回归的年化截距；α t 值 ≥ 1.96 表示剔除市场敞口后选股收益显著非零。</p>

<h2>Brinson 行业归因（vs 全A等权）</h2>
{br_tbl}
<p class="note">配置效应=行业间权重偏离的贡献；选择效应=行业内选股贡献；
两者合计应近似等于主动收益。</p>
"""
    html = page("全A主策略超额归因", header="", body=body, css=_CSS)
    out_path.write_text(html, encoding="utf-8")
    log.info("HTML 报告：%s", out_path)


def render_summary(ab_df: pd.DataFrame, br_summary: dict,
                   cmp_summary: dict, out_path: Path):
    lines = [
        "# 全A主策略超额归因摘要",
        "",
        "## α/β 分解",
        "",
        "| 基准 | 年化α | α t(NW) | α p | β | β t(NW) | R² |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in ab_df.iterrows():
        lines.append(
            f"| {name} | {r['alpha_annual'] * 100:+.2f}% | {r['alpha_t_nw']:.2f} "
            f"| {r['alpha_p_nw']:.4f} | {r['beta']:.3f} | {r['beta_t_nw']:.2f} "
            f"| {r['r2']:.3f} |")
    lines += [
        "",
        "## Brinson（vs 全A等权，申万一级）",
        "",
        f"- 主动收益（算术口径）: {br_summary['active_return'] * 100:+.2f}%"
        f"（配置 {br_summary['allocation'] * 100:+.2f}% + 选择 "
        f"{br_summary['selection'] * 100:+.2f}% + 交互 {br_summary['interaction'] * 100:+.2f}%）",
        f"- 重建误差 recon_error: {br_summary['recon_error']:.2e}（应≈0）",
        "",
        "## 与指数超额对照",
        "",
        f"- vs 上证指数：年化超额 {cmp_summary['excess_idx'] * 100:+.2f}%，"
        f"跟踪误差 {cmp_summary['te_idx'] * 100:.2f}%，β {cmp_summary['beta_idx']:.3f}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown 摘要：%s", out_path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frac", type=float, default=0.10, help="TopFrac 集中度（默认 0.10）")
    args = ap.parse_args()

    dr, weights, base, oos_days = run_main_strategy_backtest(args.frac)
    out = compute_attribution(dr, weights, base, oos_days)
    ab_df = out["ab_df"]
    br_df = out["br_df"]
    br_summary = out["br_summary"]
    cmp_summary = out["cmp_summary"]

    OUT.mkdir(parents=True, exist_ok=True)
    ab_df.to_csv(OUT / "alpha_beta.csv", encoding="utf-8-sig")
    br_df.to_csv(OUT / "brinson_industry.csv", encoding="utf-8-sig")
    render_summary(ab_df, br_summary, cmp_summary, OUT / "summary.md")
    render_html(ab_df, br_df, br_summary, cmp_summary, OUT / "attribution.html")

    # 控制台速览
    print("\n=== α/β ===")
    print(ab_df.to_string(float_format=lambda v: f"{v:,.4f}"))
    print("\n=== Brinson 汇总（vs 全A等权）===")
    for k in ("portfolio_return", "benchmark_return", "active_return",
              "allocation", "selection", "interaction", "recon_error"):
        print(f"  {k:>20s}: {br_summary.get(k, float('nan')):+.4f}")
    log.info("归因完成 %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
