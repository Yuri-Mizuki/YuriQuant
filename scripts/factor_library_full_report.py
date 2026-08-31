"""全因子库检验报告：821 个因子的标准检验汇总（自包含 HTML）。

覆盖 alpha101/158/191/360（借用公开库）+ GP 挖掘 + 模型预测，每因子输出：
- IC 系列：IC 均值/ICIR/普通 t/NW-t（Newey-West 自相关稳健）/IC 胜率/IC 衰减
- 分层回测：LS（多空）与 LO（多头）月频+周频的收益/Sharpe/回撤/换手
- 借用的公开库因子明确标注"样本内检验、无挖-验分离"的局限提示

用法:
    python scripts/factor_library_full_report.py [--dataset hs300_2022_2025]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402


log = setup_logging("factor_library_report")

from research.factor_library import FactorLibrary  # noqa: E402
from research.html_report import (  # noqa: E402
    embed_image_b64 as _fig_to_b64,
    page,
)


def plot_ls_equity(name: str, ev: pd.DataFrame) -> str:
    """多空/多头分层净值曲线（evals 里已有 equity_ls_M / equity_lo_M）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for col, lab, color in [("equity_ls_M", "多空LS(月)", "#A32D2D"),
                            ("equity_lo_M", "多头LO(月)", "#185FA5")]:
        if col in ev.columns:
            s = ev[col].dropna()
            if len(s):
                ax.plot(s.index, s.values, label=lab, linewidth=1.2, color=color)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(f"{name} 分层净值（多空/多头）")
    ax.set_ylabel("累计净值")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64

def plot_monthly_ic(name: str, ev: pd.DataFrame) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ic = ev["ic"].dropna() if "ic" in ev.columns else pd.Series(dtype=float)
    monthly = ic.groupby(ic.index.to_period("M")).mean()
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    colors = ["#A32D2D" if v > 0 else "#3B6D11" for v in monthly.values]
    ax.bar(monthly.index.astype(str), monthly.values, color=colors, width=0.7)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set_title(f"{name} 月度 IC")
    ax.set_ylabel("IC")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# 全因子库报告主题样式（page 外壳 + 自定义 CSS）
_CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,'PingFang SC',sans-serif; background:#f5f7fa; color:#1a1a2e; line-height:1.6; }
.container { max-width:1200px; margin:0 auto; padding:24px 16px; }
.header { background:linear-gradient(135deg,#1a1a2e,#16213e); color:#fff; padding:28px 24px; border-radius:12px; margin-bottom:20px; }
.header h1 { font-size:20px; margin-bottom:6px; }
.header .meta { font-size:13px; opacity:.85; }
.card { background:#fff; border-radius:10px; padding:18px; margin-bottom:14px; box-shadow:0 1px 4px rgba(0,0,0,.06); }
.card h2 { font-size:15px; color:#16213e; margin-bottom:12px; border-bottom:2px solid #e8e8e8; padding-bottom:6px; }
.card h3 { font-size:14px; color:#16213e; margin-bottom:6px; }
.fam-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; }
.fam-card { background:#f8f9fa; border-radius:8px; padding:12px; border-left:3px solid #ccc; }
.fam-dot { width:10px; height:10px; border-radius:50%; display:inline-block; margin-right:6px; }
.fam-name { font-weight:600; font-size:14px; }
.fam-name .cnt { color:#888; font-weight:400; font-size:12px; }
.fam-metric { font-size:12px; color:#555; margin-top:2px; }
.fam-note { font-size:11px; color:#999; margin-top:4px; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { padding:7px 8px; background:#f0f2f5; text-align:right; border-bottom:2px solid #ddd; position:sticky; top:0; cursor:pointer; user-select:none; }
th:first-child, td:first-child { text-align:left; }
td { padding:6px 8px; border-bottom:1px solid #f0f0f0; text-align:right; }
tr:hover { background:#f8f9ff; }
.sig { color:#A32D2D; font-weight:600; }
.ic { color:#888; font-size:12px; font-weight:400; }
.formula { font-family:monospace; font-size:11px; color:#666; background:#f8f9fa; padding:6px 10px; border-radius:6px; margin-bottom:8px; overflow-x:auto; }
.img-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
.img-grid img { width:100%; height:auto; border-radius:6px; }
.warning { background:#fff3cd; border:1px solid #ffe082; border-radius:8px; padding:14px; font-size:12px; color:#856404; margin-bottom:14px; }
.search { width:100%; padding:8px 12px; border:1px solid #ddd; border-radius:8px; margin-bottom:10px; font-size:13px; }
.scroll { max-height:600px; overflow-y:auto; border:1px solid #eee; border-radius:8px; }
@media print { .scroll { max-height:none; overflow:visible; } }
"""

_JS = """
const tb = document.getElementById('ftable');
const tbody = tb.tBodies[0];
const heads = tb.tHead.rows[0].cells;
for (let i = 0; i < heads.length; i++) {
  heads[i].onclick = () => {
    const rows = [...tbody.rows].sort((a, b) => {
      const av = a.cells[i].innerText, bv = b.cells[i].innerText;
      const an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return an - bn;
      return av.localeCompare(bv);
    });
    rows.forEach(r => tbody.appendChild(r));
  };
}
document.getElementById('fsearch').oninput = (e) => {
  const q = e.target.value.toLowerCase();
  for (const r of tbody.rows) r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none';
};
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="全因子库检验报告")
    ap.add_argument("--dataset", default="hs300_2022_2025")
    ap.add_argument("--out", default=None, help="输出 HTML 路径")
    ap.add_argument("--top-n", type=int, default=15, help="画图 top 因子数（按 |IC|）")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    reg = lib.list_all()
    log.info("因子库 %s: %d 个因子", args.dataset, len(reg))

    # 排序键
    reg = reg.copy()
    reg["abs_ic"] = reg["ic_mean"].abs()
    reg = reg.sort_values("abs_ic", ascending=False).reset_index(drop=True)

    # ---- 分组统计 ----
    def src_family(s: str) -> str:
        if pd.isna(s):
            return "unknown"
        s = str(s)
        if s.startswith("alpha101"): return "alpha101"
        if s.startswith("alpha158"): return "alpha158"
        if s.startswith("alpha191"): return "alpha191"
        if s.startswith("alpha360"): return "alpha360"
        if s.startswith("gp"): return "gp"
        if s.startswith("model"): return "model"
        return s.split(":")[0]

    reg["family"] = reg["source"].map(src_family)
    fam_style = {
        "alpha101": "#378ADD", "alpha158": "#1D9E75", "alpha191": "#BA7517",
        "alpha360": "#534AB7", "gp": "#D85A30", "model": "#E24B4A",
    }
    fam_stat = []
    for fam, grp in reg.groupby("family"):
        sig = int(grp["significant"].sum()) if "significant" in grp else 0
        ic = grp["ic_mean"].dropna()
        fam_stat.append({
            "family": fam, "n": len(grp), "n_sig": sig,
            "sig_ratio": f"{sig/len(grp)*100:.0f}%",
            "ic_mean": f"{ic.abs().mean():.4f}" if len(ic) else "—",
            "top_ic": f"{grp['abs_ic'].max():.4f}" if "abs_ic" in grp else "—",
            "note": "借用公开库（WorldQuant/101公式）" if fam.startswith("alpha")
                    else ("遗传规划挖掘" if fam == "gp" else "模型预测分数"),
            "color": fam_style.get(fam, "#888780"),
        })
    fam_df = pd.DataFrame(fam_stat)

    # ---- 全因子表行 ----
    def fmt(v, nd=4, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if pct:
            return f"{v*100:.2f}%"
        return f"{v:.{nd}f}"

    rows = []
    for _, r in reg.iterrows():
        sig_tag = "显著" if r.get("significant") else "—"
        sig_cls = "sig" if r.get("significant") else ""
        rows.append(
            f"<tr><td><b>{r['name']}</b></td>"
            f"<td>{r.get('family','')}</td>"
            f"<td class='{sig_cls}'>{sig_tag}</td>"
            f"<td>{fmt(r.get('ic_mean'))}</td>"
            f"<td>{fmt(r.get('ic_ir'))}</td>"
            f"<td>{fmt(r.get('t_stat_nw'))}</td>"
            f"<td>{fmt(r.get('ic_win_rate'), 2, True)}</td>"
            f"<td>{fmt(r.get('annual_return_ls_M'), 2, True)}</td>"
            f"<td>{fmt(r.get('sharpe_ls_M'))}</td>"
            f"<td>{fmt(r.get('annual_return_lo_M'), 2, True)}</td>"
            f"<td>{fmt(r.get('avg_turnover_ls_M'), 2, True)}</td>"
            f"<td title='{str(r.get('formula',''))[:200]}'>{str(r.get('formula',''))[:60]}</td></tr>")

    # ---- Top 因子详情图 ----
    top_imgs = ""
    top_n = min(args.top_n, len(reg))
    for i in range(top_n):
        r = reg.iloc[i]
        ev = lib._load_eval(r["name"])
        if ev is None or len(ev) == 0:
            continue
        img1 = plot_ls_equity(r["name"], ev)
        img2 = plot_monthly_ic(r["name"], ev)
        top_imgs += (
            f"<div class='card'><h3>#{i+1} {r['name']} "
            f"<span class='ic'>(IC={fmt(r.get('ic_mean'))}, NW-t={fmt(r.get('t_stat_nw'))}, "
            f"LS月={fmt(r.get('annual_return_ls_M'),2,True)})</span></h3>"
            f"<div class='formula'>{str(r.get('formula',''))[:300]}</div>"
            f"<div class='img-grid'><div><img src='data:image/png;base64,{img1}'></div>"
            f"<div><img src='data:image/png;base64,{img2}'></div></div></div>")

    # ---- HTML ----
    fam_cards = "".join(
        f"<div class='fam-card'><div class='fam-dot' style='background:{f['color']}'></div>"
        f"<div class='fam-name'>{f['family']} <span class='cnt'>{f['n']}个</span></div>"
        f"<div class='fam-metric'>{f['n_sig']} 显著（{f['sig_ratio']}）</div>"
        f"<div class='fam-metric'>平均|IC| {f['ic_mean']} | Top {f['top_ic']}</div>"
        f"<div class='fam-note'>{f['note']}</div></div>"
        for _, f in fam_df.iterrows())

    n_sig_total = int(reg["significant"].sum()) if "significant" in reg else 0
    body = f"""<div class="container">
<div class="header">
<h1>YuriQuant 全因子库检验报告 — {args.dataset}</h1>
<div class="meta">{len(reg)} 个因子 | {n_sig_total} 个显著 | 面板 2022-01 ~ 2026-08 | 逐日 IC + 分层回测（多空/多头，月频+周频）</div>
</div>

<div class="warning">
<b>借用的公开因子库无挖-验分离：</b>alpha101/158/191/360 为公开量价公式的本地实现，未经过"挖掘段-验证段"
切分的样本外检验；下方 IC/分层均为<b>全样本（2022-2026）内</b>统计，存在过拟合幸存者偏差。
GP 因子为遗传规划挖掘（同样样本内评估）。判断因子是否可用，建议额外做滚动/分段验证。</div>

<div class="card"><h2>按来源统计</h2><div class="fam-grid">{fam_cards}</div></div>

<div class="card"><h2>Top {top_n} 因子详情（按 |IC| 排序）</h2>{top_imgs}</div>

<div class="card"><h2>全因子检验表（{len(reg)} 行，点击表头排序 / 搜索过滤）</h2>
<input class="search" id="fsearch" placeholder="搜索因子名 / 来源 / 公式关键词...">
<div class="scroll"><table id="ftable">
<thead><tr><th data-k="0">因子名</th><th data-k="1">来源</th><th>显著</th><th>IC</th><th>ICIR</th>
<th>NW-t</th><th>IC胜率</th><th>LS月收益</th><th>LS月Sharpe</th><th>LO月收益</th><th>换手(LS月)</th><th>公式</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></div>

<div class="warning">生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 指标口径与 standard_factor_summary / quantile_backtest 一致（NW 自相关稳健 t、多空多头双口径）</div>
</div>"""
    html = page(
        f"YuriQuant 全因子库检验报告 — {args.dataset}",
        header="",
        body=body,
        css=_CSS,
        scripts=f"<script>\n{_JS}\n</script>",
    )
    out = Path(args.out) if args.out else Path("reports") / f"factor_library_report_{args.dataset}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log.info("报告已生成: %s (%d KB)", out, len(html) // 1024)

    # 同时导出全表 CSV
    csv_out = out.with_suffix(".csv")
    cols = ["name", "family", "significant", "ic_mean", "ic_ir", "t_stat_nw",
            "ic_win_rate", "annual_return_ls_M", "sharpe_ls_M", "annual_return_lo_M",
            "avg_turnover_ls_M", "formula"]
    reg[cols].to_csv(csv_out, index=False, encoding="utf-8-sig")
    log.info("全表 CSV: %s", csv_out)

if __name__ == "__main__":
    main()