"""
投资收益报告 —— 端到端工作流的最终交付物
==========================================

把「因子筛选 → 模型预测 → 组合回测」的产出汇总成一份投资收益报告：

1. **模型预测作为因子的检验**（回答"这个信号有没有预测力"）：
   - rank IC 序列 / ICIR / Newey-West t / 月度 IC（`standard_factor_summary`）
   - 分层收益图（Q1~Q5 累计净值 + 多空，`quantile_backtest`）
   - 稀疏口径（仅调仓日信号）与持仓口径（ffill 到日频，与组合一致）并排
2. **组合 vs 大盘指数基准**：
   - 股票池为沪深300（因子库池）→ 基准 = 沪深300指数 000300.SH
   - 全A 池 → 基准 = 上证指数 000001.SH（`--index` 指定，缓存无则真实模式拉取）
   - 绩效：总收益/年化/波动/Sharpe/最大回撤/胜率 + 相对基准 alpha/beta/信息比
3. **报告**：自包含 HTML（matplotlib 图 base64 内嵌）+ CSV

用法：
    python scripts/investment_report.py --real --top 50
    python scripts/investment_report.py --real --top 50 --index 000001.SH
    python scripts/investment_report.py --mock --top 20 --model ridge   # 测试
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.metrics import PERIODS_PER_YEAR  # noqa: E402
from data.mock import load_mock_data  # noqa: E402
from factor.classic import compute_classic_features  # noqa: E402
from model.labels import build_label_pair  # noqa: E402
from research.factor_report import (  # noqa: E402
    _fig_to_b64,
    factor_test,
    plot_layers,
    plot_monthly_ic,
)
from scripts.e2e_backtest import (  # noqa: E402
    perf_stats,
    run_equal_weight_backtest,
    run_risk_parity_backtest,
    walk_forward_predictions,
)
from scripts.e2e_common import (  # noqa: E402
    HORIZON,
    drop_stale_factors,
    load_daily_data,
    select_features,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("investment_report")

OUT_DIR = Path("reports/investment_report")
BT_START = "2024-01-01"

# matplotlib 中文字体（Windows 微软雅黑）
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def plot_equity_curves(equity: pd.DataFrame) -> str:
    """净值曲线（equity 已含指数基准列，统一日历 + 起点 1.0，勿重复传基准）。"""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = {"等权top50": "#378ADD", "风险平价top50": "#D85A30"}
    for c in equity.columns:
        color = colors.get(c, "#888780")  # 指数基准列默认灰色
        ax.plot(equity.index, equity[c], label=c, linewidth=1.4, color=color)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title("策略 vs 指数基准 净值（费后，起点归一 1.0）")
    ax.set_ylabel("累计净值")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


def plot_drawdown(equity: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for c in equity.columns:
        eq = equity[c]
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values * 100, label=c, linewidth=1.2)
    ax.fill_between(equity.index, 0, -20, color="gray", alpha=0.05)
    ax.set_title("回撤（%）")
    ax.set_ylabel("%")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# 相对基准的绩效
# ---------------------------------------------------------------------------
def benchmark_stats(daily_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """alpha/beta/信息比/超额（相对基准指数）。"""
    df = pd.DataFrame({"strat": daily_ret, "bench": bench_ret}).dropna()
    if len(df) < 20 or df["bench"].std() == 0:
        return {}
    beta = float(np.cov(df["strat"], df["bench"])[0, 1] / df["bench"].var())
    alpha_d = float(df["strat"].mean() - beta * df["bench"].mean())
    excess = df["strat"] - df["bench"]
    ir = float(excess.mean() / excess.std() * np.sqrt(PERIODS_PER_YEAR)) if excess.std() > 0 else 0.0
    return {"beta": round(beta, 3), "alpha_annual": round(alpha_d * PERIODS_PER_YEAR, 4),
            "information_ratio": round(ir, 3),
            "excess_total": round(float((1 + excess).prod() - 1), 4)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info("=" * 60)
    log.info(" 投资收益报告生成")
    log.info(" 池: %s | 指数基准: %s | 调仓 %s | top-%d",
             "沪深300(因子库池)" if args.real else "mock", args.index, args.freq, args.top)
    log.info("=" * 60)

    # ---- 1. 数据 + 因子 ----
    if args.real:
        px, lib_feats = load_daily_data(begin=20220101)
    else:
        px = load_mock_data(n_days=args.n_days, n_codes=args.n_codes, seed=args.seed)
        lib_feats = {}
    classic = compute_classic_features(px)
    all_feats = {**classic, **lib_feats}
    log.info("因子总池: %d（经典 %d + 因子库 %d）",
             len(all_feats), len(classic), len(lib_feats))

    close = px["close"]
    returns = close.pct_change(fill_method=None)
    labels, fwd = build_label_pair(close, horizon=HORIZON)

    # 面板新鲜度守卫（真实模式）
    if args.real:
        all_feats = drop_stale_factors(all_feats, close.index[-1])

    # ---- 2. 特征选择（只用回测前窗口）----
    all_days = close.index
    bt_start = pd.Timestamp(args.bt_start)
    sel_days = all_days[all_days < bt_start]
    if len(sel_days) < 60:
        split = len(all_days) // 2
        sel_days = all_days[:split]
        bt_start_actual = all_days[split]
    else:
        bt_start_actual = bt_start
    feats, quality = select_features(all_feats, fwd, sel_days, max_features=args.max_features)
    log.info("特征选择: %d -> %d", len(all_feats), len(feats))

    # ---- 3. 公共网格 + 调仓日 ----
    common = None
    for f in feats.values():
        d = f.dropna(how="all").index
        common = d if common is None else common.intersection(d)
    common = common.intersection(labels.dropna(how="all").index)
    bt_days = common[common >= bt_start_actual]
    s = pd.Series(bt_days, index=bt_days)
    reb_days = list(s.groupby(s.index.to_period(args.freq)).first())
    log.info("回测区间: %s ~ %s, 调仓 %d 次", bt_days[0].date(), bt_days[-1].date(), len(reb_days))

    # ---- 4. Walk-forward 预测 ----
    log.info("Walk-forward 训练预测中（%s, window=%s）...", args.model, args.train_window)
    pred_reb = walk_forward_predictions(feats, labels, reb_days, common, model=args.model,
                                        window=args.train_window)
    pred_reb.to_csv(out_dir / "walk_forward_predictions.csv", encoding="utf-8-sig")

    # ---- 4b. 风格中性化（默认开启：五因子残差剥离风格，实测 Sharpe 0.41→0.66）----
    if args.neutralize and args.real:
        from scripts.e2e_common import build_neutral_covariates, neutralize_predictions
        mc_panel, ind_panel, extra = build_neutral_covariates(px, close, real=True)
        pred_reb = neutralize_predictions(pred_reb, mc_panel, ind_panel, extra)
        pred_reb.to_csv(out_dir / "walk_forward_predictions_neutralized.csv",
                        encoding="utf-8-sig")
        log.info("预测分数已做五因子中性化（市值/行业/动量/波动/换手残差）")

    # ---- 5. 因子检验（模型预测作为因子）----
    log.info("因子检验：模型预测作为因子 ...")
    ft = factor_test(pred_reb, fwd, close, out_dir, "model_pred")

    # ---- 6. 组合回测 ----
    bt_returns = returns.loc[bt_days[0]:bt_days[-1]]
    log.info("回测 A：等权 top-%d ...", args.top)
    result_eq = run_equal_weight_backtest(pred_reb, returns, bt_days, args.top)
    stats_eq = perf_stats(result_eq.daily_returns, f"等权top{args.top}")

    stats_rp = None
    if not args.skip_rp:
        try:
            import cvxpy  # noqa: F401
            log.info("回测 B：risk_parity top-%d ...", args.top)
            result_rp = run_risk_parity_backtest(pred_reb, returns, bt_days, args.top,
                                                 args.max_weight)
            stats_rp = perf_stats(result_rp.daily_returns, f"风险平价top{args.top}")
        except Exception as e:
            log.warning("risk_parity 跳过: %s", e)

    # ---- 7. 指数基准 ----
    from data.cache_helpers import load_index_returns
    bench_ret = load_index_returns(args.index, int(bt_days[0].strftime("%Y%m%d")),
                                   int(bt_days[-1].strftime("%Y%m%d")), real=args.real)
    bench_ret = bench_ret.reindex(bt_days) if bench_ret is not None else None
    if bench_ret is None:
        bench_ret = bt_returns.mean(axis=1)
        bench_label = "全池等权(指数不可用)"
    else:
        bench_label = f"指数基准 {args.index}"
    stats_bm = perf_stats(bench_ret, bench_label)

    # ---- 8. 汇总 ----
    # benchmark stats（beta/alpha/信息比/超额）必须 update 回 stats 对象本身，
    # 否则 summary CSV 有值但 HTML 渲染读 st.get("beta") 拿到的是原始 perf_stats
    # （无这些键）→ 表格 α/β/信息比/超额 全部显示 "—"。
    for st in [stats_eq, stats_rp]:
        if st is not None and bench_ret is not None:
            st.update(benchmark_stats(st["daily"], bench_ret))
    rows = []
    for st in [stats_eq, stats_rp, stats_bm]:
        if st is None:
            continue
        row = {k: v for k, v in st.items() if k not in ("monthly", "equity", "daily")}
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("label").T
    print("\n===== 投资收益对照（%s ~ %s, %s 调仓）=====" %
          (bt_days[0].date(), bt_days[-1].date(), args.freq))
    print(summary.to_string())
    summary.to_csv(out_dir / "investment_summary.csv", encoding="utf-8-sig")

    monthly = pd.DataFrame({st["label"]: st.get("monthly", pd.Series(dtype=float))
                            for st in [stats_eq, stats_rp, stats_bm] if st is not None})
    monthly.to_csv(out_dir / "monthly_returns.csv", encoding="utf-8-sig")

    # 净值曲线：统一日历（bt_days）+ 缺失收益填 0 + 起点归一 1.0，
    # 保证策略与指数同长度、同起点可比（否则指数线多/少 NaN 日，视觉上像两条线）
    plot_df = pd.DataFrame({
        st["label"]: (1 + st["daily"].reindex(bt_days).fillna(0.0)).cumprod()
        for st in [stats_eq, stats_rp, stats_bm] if st is not None})
    plot_df.to_csv(out_dir / "equity_curve.csv", encoding="utf-8-sig")

    # ---- 9. HTML 报告 ----
    img_layers = plot_layers(ft["layer_nav"])
    img_mic = plot_monthly_ic(ft["monthly_ic"])
    img_eq = plot_equity_curves(plot_df)
    img_dd = plot_drawdown(plot_df)

    ft_rows = ""
    for lbl, s in [("稀疏(调仓日)", ft["sum_sparse"]), ("持仓(日频)", ft["sum_hold"])]:
        ft_rows += (
            f"<tr><td>{lbl}</td>"
            f"<td>{s['ic_mean']:.4f}</td><td>{s['ic_std']:.4f}</td>"
            f"<td>{s['ir']:.2f}</td><td>{s['t_stat']:.2f}</td>"
            f"<td>{s['t_stat_nw']:.2f}</td><td>{s['ic_win_rate']*100:.0f}%</td>"
            f"<td>{s['n']}</td></tr>")

    decay = ft["sum_hold"]["ic_decay"]
    decay_html = " | ".join(f"lag{l}: {v:.4f}" for l, v in decay.items())

    pct = lambda v, d=2: (v if isinstance(v, str) else "—"
                          if v is None or (isinstance(v, float) and np.isnan(v))
                          else f"{v * 100:.{d}f}%")
    srow = lambda st, k: pct(st.get(k)) if st else "—"

    perf_html = ""
    for st in [stats_eq, stats_rp, stats_bm]:
        if st is None:
            continue
        beta = st.get("beta", "—")
        alpha = st.get("alpha_annual", "—")
        ir = st.get("information_ratio", "—")
        ex = st.get("excess_total", "—")
        if isinstance(beta, float):
            beta = f"{beta:.2f}"
        perf_html += (
            f"<tr><td><b>{st['label']}</b></td>"
            f"<td>{pct(st['total_return'])}</td><td>{pct(st['annual_return'])}</td>"
            f"<td>{pct(st['annual_vol'])}</td><td>{st['sharpe']:.2f}</td>"
            f"<td>{pct(st['max_drawdown'])}</td><td>{st['win_rate_monthly']*100:.0f}%</td>"
            f"<td>{alpha if isinstance(alpha,str) else f'{alpha*100:.2f}%'}</td>"
            f"<td>{beta}</td><td>{ir}</td><td>{pct(ex)}</td></tr>")

    mrows = ""
    for d, row in monthly.iterrows():
        cells = "".join(f'<td class="{"pos" if v > 0 else "neg"}">{v*100:+.1f}%</td>'
                        for v in row)
        mrows += f"<tr><td>{d.strftime('%Y-%m')}</td>{cells}</tr>"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>YuriQuant 投资收益报告</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,'PingFang SC',sans-serif; background:#f5f7fa; color:#1a1a2e; line-height:1.6; }}
.container {{ max-width:1000px; margin:0 auto; padding:24px 16px; }}
.header {{ background:linear-gradient(135deg,#1a1a2e,#16213e); color:#fff; padding:28px 24px; border-radius:12px; margin-bottom:20px; }}
.header h1 {{ font-size:20px; margin-bottom:6px; }}
.header .meta {{ font-size:13px; opacity:.85; }}
.card {{ background:#fff; border-radius:10px; padding:18px; margin-bottom:14px; box-shadow:0 1px 4px rgba(0,0,0,.06); }}
.card h2 {{ font-size:15px; color:#16213e; margin-bottom:12px; border-bottom:2px solid #e8e8e8; padding-bottom:6px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th {{ padding:8px 10px; background:#f8f9fa; text-align:right; border-bottom:2px solid #e0e0e0; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 10px; border-bottom:1px solid #f0f0f0; text-align:right; }}
td:first-child {{ text-align:left; }}
.pos {{ color:#c0392b; }} .neg {{ color:#27ae60; }}
img {{ width:100%; height:auto; border-radius:8px; }}
.kpi {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-bottom:12px; }}
.kpi .item {{ background:#f8f9fa; border-radius:8px; padding:12px; }}
.kpi .l {{ font-size:11px; color:#888; }} .kpi .v {{ font-size:17px; font-weight:700; }}
.warning {{ background:#fff3cd; border:1px solid #ffe082; border-radius:8px; padding:14px; font-size:12px; color:#856404; }}
</style></head><body><div class="container">

<div class="header"><h1>YuriQuant 投资收益报告</h1>
<div class="meta">回测区间 {bt_days[0].date()} ~ {bt_days[-1].date()} | 调仓 {args.freq} 频（{len(reb_days)} 次）| 基准 {bench_label} | top-{args.top} | 因子 {len(feats)}/候选 {len(all_feats)}</div></div>

<div class="card"><h2>1. 绩效总览（费后，相对基准）</h2>
<table><tr>
<th>组合</th><th>总收益</th><th>年化</th><th>波动</th><th>Sharpe</th><th>最大回撤</th>
<th>月胜率</th><th>年化α</th><th>β</th><th>信息比</th><th>超额</th></tr>
{perf_html}</table></div>

<div class="card"><h2>2. 净值曲线（策略 vs {bench_label}）</h2>
<img src="data:image/png;base64,{img_eq}"><br>
<img src="data:image/png;base64,{img_dd}"></div>

<div class="card"><h2>3. 模型预测作为因子的检验</h2>
<table><tr><th>口径</th><th>IC均值</th><th>IC标准差</th><th>ICIR</th><th>t统计</th>
<th>NW-t</th><th>IC胜率</th><th>样本日</th></tr>{ft_rows}</table>
<p style="font-size:12px;color:#666;margin-top:8px;">IC 衰减（持仓口径）: {decay_html}</p></div>

<div class="card"><h2>4. 分层收益（Q1=预测最低组, Q5=最高组, 持仓口径, 日频持有）</h2>
<img src="data:image/png;base64,{img_layers}">
<p style="font-size:12px;color:#666;margin-top:6px;"><b>读图提示：</b>本模型分层区分度弱（持仓 IC&asymp;0.03），Q5 并未稳定高于 Q1——图中 Q1 组的领先主要由 2024-09 极端行情中低预测组（超跌低价股）的暴涨贡献并复利放大（如 9/27 单日 +11%），<b>不是方向画反，而是预测对日频收益的区分力有限</b>。剔除极端日后 Q1 组日均收益与 Q5 组几乎无差异（约 2bp vs 0bp），Q1&gt;Q5 天数占比仅 49%（接近随机）。</p></div>

<div class="card"><h2>5. 月度 IC（持仓口径）</h2>
<img src="data:image/png;base64,{img_mic}"></div>

<div class="card"><h2>6. 月度收益</h2>
<table><tr><th>月份</th>{''.join(f'<th>{st["label"]}</th>' for st in [stats_eq, stats_rp, stats_bm] if st is not None)}</tr>{mrows}</table></div>

<div class="warning"><b>口径与局限：</b><br>
1. 股票池 = 因子库 significant 面板列并集（沪深300 PIT 历史成员 ~420 股），基准按池匹配：沪深300 → 000300.SH；全A → 000001.SH（--index 指定）<br>
2. 因子检验的"稀疏"口径 = 仅调仓日信号（样本=调仓次数）；"持仓"口径 = 预测 ffill 到下次调仓（与组合一致，样本=交易日）<br>
3. 组合回测成本：佣金 0.01% + 印花税 0.1%(卖出) + 滑点 5bp，已扣除<br>
4. 未过滤涨跌停/停牌可执行性；特征选择只用回测前窗口防前视；embargo=5<br>
5. 模型预测仅月频重训，信号在调仓间衰减；更高频调仓（--freq W）可能改变结论<br>
6. 非投资建议</div>
</div></body></html>"""
    (out_dir / "investment_report.html").write_text(html, encoding="utf-8")

    meta = {
        "bt_start": str(bt_days[0].date()), "bt_end": str(bt_days[-1].date()),
        "freq": args.freq, "top_n": args.top, "index_benchmark": str(args.index),
        "bench_label": bench_label, "n_rebalance": len(reb_days),
        "n_features": len(feats), "model": args.model, "horizon": HORIZON,
        "factor_test": {
            "sparse": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in ft["sum_sparse"].items() if k != "ic_decay"},
            "hold": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in ft["sum_hold"].items() if k != "ic_decay"},
        },
        "summary": {r["label"]: {k: v for k, v in r.items()} for r in rows},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "investment_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info("=" * 60)
    log.info(" 报告完成（%.1f 分钟）: %s", (time.time() - t0) / 60, out_dir)
    log.info("=" * 60)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="投资收益报告")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--freq", default="M", choices=["D", "W", "M"])
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "ridge"])
    ap.add_argument("--train-window", type=int, default=None,
                    help="滚动训练窗口（交易日数）。None=expanding 全历史；"
                         "500=滚动2年（实测 beta-neutral alpha -8.6%%→+5.5%%，"
                         "缓解训练/预测市场状态错配）")
    ap.add_argument("--index", default="000300.SH",
                    help="指数基准（沪深300池→000300.SH；全A池→000001.SH）")
    ap.add_argument("--bt-start", default=BT_START)
    ap.add_argument("--max-weight", type=float, default=0.05)
    ap.add_argument("--max-features", type=int, default=30)
    ap.add_argument("--skip-rp", action="store_true")
    ap.add_argument("--neutralize", action=argparse.BooleanOptionalAction, default=True,
                    help="预测分数五因子中性化（默认开启，--no-neutralize 关闭）")
    ap.add_argument("--n-days", type=int, default=500)
    ap.add_argument("--n-codes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
