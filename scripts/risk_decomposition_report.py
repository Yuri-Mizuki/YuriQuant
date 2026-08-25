"""
组合级风险分解报告脚本
========================

对回测结果做组合级风险分解，输出自包含 HTML 报告：

  1. 风险贡献表（边际/成分/占比，Top 10 风险贡献个股）
  2. 风格因子方差贡献（B'ΣB 分解，特质风险占比）
  3. 行业方差贡献（按行业分组求 CR 之和）
  4. VaR/CVaR 成分分解（历史模拟法，左尾 5%）
  5. 逐期组合波动率 + 风格暴露时序

用法：
    # mock 模式（无需数据源凭证）
    python scripts/risk_decomposition_report.py --mock

    # 真实数据（需先跑回测产出 BacktestResult 权重历史）
    python scripts/risk_decomposition_report.py --real --weights-path reports/xxx/weights_history.parquet

    # 自定义参数
    python scripts/risk_decomposition_report.py --mock --n-days 300 --n-codes 30 --var-quantile 0.05

输出：
    reports/risk_decomposition/risk_decomposition.html
    reports/risk_decomposition/risk_contributions.csv
    reports/risk_decomposition/variance_decomp.json
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from optimize.risk import risk_decomposition

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("risk_decomposition_report")


# ---------------------------------------------------------------------------
# Mock 数据
# ---------------------------------------------------------------------------
def gen_mock_data(n_days: int = 300, n_codes: int = 30, seed: int = 42):
    """生成 mock 权重历史 + 收益面板 + 风格暴露 + 行业面板。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    # 收益面板
    rets = pd.DataFrame(
        rng.normal(0, 0.02, (n_days, n_codes)), idx, codes,
    )

    # 月度调仓等权权重
    w = pd.DataFrame(0.0, idx, codes)
    s = pd.Series(idx, index=idx)
    monthly_dates = s.groupby(pd.Grouper(freq="ME")).last()
    for dt in monthly_dates:
        if dt in w.index:
            w.loc[dt] = 1.0 / n_codes

    # 风格暴露：动量（过去 20 日收益）+ 波动率（20 日 std）
    mom = rets.rolling(20).mean()
    vol = rets.rolling(20).std()
    style = {"mom": mom, "vol": vol}

    # 行业面板（3 个行业）
    ind_names = ["银行", "电子", "食品"]
    ind_arr = np.array([ind_names[i % 3] for i in range(n_codes)])
    industry = pd.DataFrame(
        np.tile(ind_arr, (n_days, 1)), index=idx, columns=codes,
    )

    return {
        "weights": w,
        "returns": rets,
        "style": style,
        "industry": industry,
    }


# ---------------------------------------------------------------------------
# HTML 报告渲染
# ---------------------------------------------------------------------------
def render_html(result: dict, out_dir: Path) -> Path:
    """自包含 HTML 报告。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "risk_decomposition.html"

    summary = result["summary"]
    vd = result["variance_decomp"]
    cr_df = result["risk_contributions"]
    vc = result["var_cvar"]
    exposure = result["exposure"]

    # 风险贡献 Top 10 表
    top10 = cr_df.head(10)
    cr_rows = ""
    for code, row in top10.iterrows():
        cr_rows += (
            f"<tr><td>{code}</td>"
            f"<td>{row['weight']:.4f}</td>"
            f"<td>{row['MRC']:.6f}</td>"
            f"<td>{row['CR']:.6f}</td>"
            f"<td>{row['CR_pct']*100:.2f}%</td></tr>"
        )

    # 风格因子贡献
    style_rows = ""
    for name, val in vd["style_factor_contrib"].items():
        pct = val / vd["total_variance"] * 100 if vd["total_variance"] > 0 else 0
        style_rows += (
            f"<tr><td>{name}</td><td>{val:.6f}</td><td>{pct:.2f}%</td></tr>"
        )

    # 行业贡献
    ind_rows = ""
    for name, val in vd["industry_contrib"].items():
        pct = val / vd["total_variance"] * 100 if vd["total_variance"] > 0 else 0
        ind_rows += (
            f"<tr><td>{name}</td><td>{val:.6f}</td><td>{pct:.2f}%</td></tr>"
        )

    # VaR/CVaR 成分 Top 5
    var_rows = ""
    if "component_VaR" in vc:
        top_var = vc["component_VaR"].head(5)
        for code, val in top_var.items():
            var_rows += f"<tr><td>{code}</td><td>{val:.6f}</td></tr>"

    # 逐期波动率时序数据
    vol_dates = [d.strftime("%Y-%m-%d") for d in exposure.index]
    vol_values = [float(v) for v in exposure["portfolio_vol"].values]
    vol_json = json.dumps(list(zip(vol_dates, vol_values)))

    # 方差分解饼图数据
    pie_data = []
    for name, val in vd["style_factor_contrib"].items():
        pie_data.append({"name": name, "value": float(val)})
    if vd["idiosyncratic_risk"] > 0:
        pie_data.append({"name": "特质风险", "value": float(vd["idiosyncratic_risk"])})
    pie_json = json.dumps(pie_data)

    explained_pct = vd["explained_variance"] / vd["total_variance"] * 100 if vd["total_variance"] > 0 else 0
    idio_pct = vd["idiosyncratic_risk"] / vd["total_variance"] * 100 if vd["total_variance"] > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>组合级风险分解报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1a1a1a; border-bottom: 3px solid #d32f2f; padding-bottom: 10px; }}
  h2 {{ color: #d32f2f; margin-top: 30px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.08); text-align: center; }}
  .card .label {{ font-size: 12px; color: #888; margin-bottom: 5px; }}
  .card .value {{ font-size: 24px; font-weight: bold; color: #d32f2f; }}
  .card .sub {{ font-size: 11px; color: #aaa; margin-top: 3px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.08); margin: 10px 0; }}
  th {{ background: #d32f2f; color: white; padding: 10px; text-align: left; font-size: 13px; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 13px; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
  .chart {{ height: 300px; position: relative; }}
  .note {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 15px 0; font-size: 12px; color: #856404; }}
</style>
</head>
<body>
<div class="container">
  <h1>组合级风险分解报告</h1>

  <div class="cards">
    <div class="card">
      <div class="label">平均组合波动率</div>
      <div class="value">{summary['avg_portfolio_vol']:.4f}</div>
      <div class="sub">日频</div>
    </div>
    <div class="card">
      <div class="label">风格解释占比</div>
      <div class="value">{explained_pct:.1f}%</div>
      <div class="sub">B'ΣB 分解</div>
    </div>
    <div class="card">
      <div class="label">特质风险占比</div>
      <div class="value">{idio_pct:.1f}%</div>
      <div class="sub">残差</div>
    </div>
    <div class="card">
      <div class="label">组合 VaR (5%)</div>
      <div class="value">{vc.get('portfolio_VaR', 0):.4f}</div>
      <div class="sub">历史模拟法</div>
    </div>
  </div>

  <div class="note">
    口径：Euler 分解 σ²_p = Σ w_k·(Σw)_k；协方差 Ledoit-Wolf 收缩（防前视）；
    VaR/CVaR 历史模拟法；风格暴露截面 zscore 归一化。共 {summary['n_periods']} 期截面。
  </div>

  <h2>1. 风险贡献 Top 10</h2>
  <table>
    <tr><th>证券代码</th><th>权重</th><th>边际风险贡献 MRC</th><th>成分贡献 CR</th><th>占比</th></tr>
    {cr_rows}
  </table>

  <h2>2. 方差分解</h2>
  <div class="chart-container">
    <div class="chart"><canvas id="pieChart"></canvas></div>
  </div>
  <table>
    <tr><th>风格因子</th><th>方差贡献</th><th>占比</th></tr>
    {style_rows}
    <tr style="font-weight:bold;background:#f0f0f0;">
      <td>特质风险</td><td>{vd['idiosyncratic_risk']:.6f}</td><td>{idio_pct:.2f}%</td>
    </tr>
    <tr style="font-weight:bold;background:#e0e0e0;">
      <td>总方差</td><td>{vd['total_variance']:.6f}</td><td>100%</td>
    </tr>
  </table>

  <h2>3. 行业风险贡献</h2>
  <table>
    <tr><th>行业</th><th>方差贡献</th><th>占比</th></tr>
    {ind_rows}
  </table>

  <h2>4. VaR/CVaR 成分分解（最大贡献 Top 5）</h2>
  <div class="cards" style="grid-template-columns: repeat(2, 1fr);">
    <div class="card">
      <div class="label">组合 VaR (5%)</div>
      <div class="value">{vc.get('portfolio_VaR', 0):.4f}</div>
    </div>
    <div class="card">
      <div class="label">组合 CVaR (5%)</div>
      <div class="value">{vc.get('portfolio_CVaR', 0):.4f}</div>
    </div>
  </div>
  <table>
    <tr><th>证券代码</th><th>成分 VaR</th></tr>
    {var_rows}
  </table>

  <h2>5. 逐期组合波动率</h2>
  <div class="chart-container">
    <div class="chart"><canvas id="volChart"></canvas></div>
  </div>
</div>

<script>
const volData = {vol_json};
const pieData = {pie_json};

// 波动率时序
new Chart(document.getElementById('volChart'), {{
  type: 'line',
  data: {{
    labels: volData.map(d => d[0]),
    datasets: [{{
      label: '组合波动率',
      data: volData.map(d => d[1]),
      borderColor: '#d32f2f',
      backgroundColor: 'rgba(211,47,47,0.1)',
      fill: true,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ ticks: {{ maxTicksLimit: 12 }} }} }}
  }}
}});

// 方差分解饼图
new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{
    labels: pieData.map(d => d.name),
    datasets: [{{
      data: pieData.map(d => Math.max(d.value, 0)),
      backgroundColor: ['#d32f2f', '#1976d2', '#388e3c', '#f57c00', '#7b1fa2', '#0097a7', '#757575'],
    }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false }}
}});
</script>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="组合级风险分解报告")
    parser.add_argument("--mock", action="store_true", help="用 mock 数据")
    parser.add_argument("--real", action="store_true", help="用真实数据（需 --weights-path）")
    parser.add_argument("--weights-path", type=str, help="权重历史 parquet 路径（--real 模式）")
    parser.add_argument("--returns-path", type=str, help="收益面板 parquet 路径（--real 模式）")
    parser.add_argument("--n-days", type=int, default=300)
    parser.add_argument("--n-codes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-window", type=int, default=120)
    parser.add_argument("--min-periods", type=int, default=60)
    parser.add_argument("--var-quantile", type=float, default=0.05)
    parser.add_argument("--freq", type=str, default="ME")
    parser.add_argument("--out", type=str, default="reports/risk_decomposition")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.real:
        if not args.weights_path:
            log.error("--real 模式需要 --weights-path")
            return
        weights = pd.read_parquet(args.weights_path)
        returns = pd.read_parquet(args.returns_path) if args.returns_path else None
        if returns is None:
            log.error("--real 模式需要 --returns-path")
            return
        # 风格暴露和行业面板可选
        style = None
        industry = None
        # TODO: 从缓存加载 build_style_covariates 和 industry panel
    else:
        data = gen_mock_data(args.n_days, args.n_codes, args.seed)
        weights = data["weights"]
        returns = data["returns"]
        style = data["style"]
        industry = data["industry"]

    log.info("权重面板: %s, 收益面板: %s", weights.shape, returns.shape)

    result = risk_decomposition(
        weights_history=weights,
        returns_panel=returns,
        style_exposures=style,
        industry_panel=industry,
        covariance_window=args.cov_window,
        min_periods=args.min_periods,
        var_quantile=args.var_quantile,
        freq=args.freq,
    )

    if "error" in result:
        log.error("风险分解失败: %s", result["error"])
        return

    # 渲染 HTML
    html_path = render_html(result, out_dir)
    log.info("HTML 报告: %s", html_path)

    # CSV 导出
    result["risk_contributions"].to_csv(out_dir / "risk_contributions.csv", encoding="utf-8-sig")
    result["exposure"].to_csv(out_dir / "exposure_timeseries.csv", encoding="utf-8-sig")

    # JSON 汇总
    json_data = {
        "summary": result["summary"],
        "variance_decomp": result["variance_decomp"],
        "var_cvar": {
            k: v for k, v in result["var_cvar"].items()
            if not isinstance(v, pd.Series)
        },
    }
    with open(out_dir / "variance_decomp.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    log.info("CSV: %s", out_dir / "risk_contributions.csv")
    log.info("JSON: %s", out_dir / "variance_decomp.json")

    # 控制台摘要
    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"组合级风险分解摘要（{s['n_periods']} 期）")
    print(f"{'='*60}")
    print(f"平均组合波动率:     {s['avg_portfolio_vol']:.6f}")
    print(f"平均组合方差:       {s['avg_portfolio_var']:.6f}")
    print(f"风格解释方差占比:   {s['explained_variance_ratio']*100:.2f}%")
    print(f"特质风险占比:       {s['idiosyncratic_ratio']*100:.2f}%")
    print(f"组合 VaR ({s['var_quantile']*100:.0f}%):  {s['portfolio_VaR']:.6f}")
    print(f"组合 CVaR ({s['var_quantile']*100:.0f}%): {s['portfolio_CVaR']:.6f}")
    print(f"\nTop 5 风险贡献个股:")
    for code, pct in s["top5_risk_contributors"].items():
        print(f"  {code}: {pct*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
