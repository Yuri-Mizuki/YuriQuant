"""
ETF 轮动 champion 生产信号板（P5，2026-10-27）
=============================================

把 P2/P4 定版的主策略 `champ`（`def_mix_crowd_f20`：12-1 动量 + 拥挤度前 20% 过滤
+ 逆波动 + 15% 波动目标 + 债金防御腿，月频）落成**可直接照单执行**的信号板：

  1. **目标持仓**：用同一套 `VectorBacktest` + 策略在最新数据上重跑到最近一个调仓日，
     取该日权重；`DEF` 合成腿按 50/50 展开成 511010.SH（国债）+ 518880.SH（黄金）。
  2. **下单与容量**：`|目标权重 − 当前权重| × 资金` 为单只下单金额，按 60 日均成交额
     计参与率；> 5% 标红（方案 §三 P3 容量口径）。
  3. **执行时点检查**：20 日折溢价 > 1% 标红（S3 结论：只作执行时点辅助，
     不做截面过滤）。
  4. **监控**：持仓与候选的拥挤度分位（份额 250 日）、溢价、成交额；
     候选榜单标注「拥挤剔除 / 动量非正 / 备选」。
  5. 输出 `reports/etf_champ/`：`signals.csv`、`monitor.csv`、`board.html`、`meta.json`。

用法:
    python -m scripts.portfolio.etf_champ_signals                       # 1000 万、空仓建仓
    python -m scripts.portfolio.etf_champ_signals --capital 5000000
    python -m scripts.portfolio.etf_champ_signals --positions 当前持仓.csv

    # 当前持仓 CSV：两列 code,weight（权重为占总资产比例，可含现金行 CASH）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.cache import DataCache
from data.datasource import create_datasource
from research.html_report import page
from scripts.portfolio.etf_rotation_p2 import (
    P1_BASE,
    build_defense,
    build_extra_panels,
    level_from_returns,
)
from scripts.portfolio.etf_rotation_probe import (
    BENCH,
    MIN_AMOUNT,
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_champ"

CHAMP = {**P1_BASE, "name": "champ", "signal": "mom_crowd_f20", "defense": "mix",
         "dedup_corr": 0.90}
# 两个月比赛版：TopK=3、周频、等权、满仓（去掉波动目标与防御腿）。
CONTEST = {"name": "contest_2m", "signal": "mom_crowd_f20", "freq": "W", "top_k": 3,
           "absolute": True, "weighting": "equal", "commission_bp": 1.0,
           "slippage_bp": 5.0, "dedup_corr": 0.90}
DEF_PARTS = {"511010.SH": 0.5, "518880.SH": 0.5}
PARTICIPATION = 0.05      # 单只单边下单 / 60 日均成交额
PREMIUM_WARN = 0.01       # 20 日折溢价告警线
CASH_CODES = {"CASH", "DEF"}

_CSS = """
  body { font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
         background:#fafbfc; color:#17233b; line-height:1.65; padding:2rem 1rem 3rem; }
  .page { max-width:1020px; margin:0 auto; }
  h1 { font-size:1.5rem; margin-bottom:.3rem; }
  h2 { font-size:1.1rem; margin:1.8rem 0 .6rem; border-bottom:2px solid #1559b6;
       padding-bottom:.3rem; }
  .meta { color:#5a6b85; font-size:.82rem; margin-bottom:1rem; }
  .meta span { margin-right:1rem; }
  table { width:100%; border-collapse:collapse; font-size:.86rem; }
  th,td { padding:.45rem .6rem; text-align:right; border-bottom:1px solid #dfe6f0; }
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left; }
  thead th { background:#eef2f7; }
  .warn { color:#b42318; font-weight:600; }
  .ok { color:#0f8a80; }
  .note { margin:.8rem 0; padding:.7rem .9rem; border-left:4px solid #b45309;
          background:#eef2f7; font-size:.84rem; }
"""


def _expand_defense(tw: pd.Series) -> pd.Series:
    """把合成防御腿 DEF 拆成两只真实 ETF，并去掉现金/零权重。"""
    out = tw.astype(float).copy()
    if "DEF" in out.index:
        v = float(out["DEF"])
        out = out.drop("DEF")
        for c, share in DEF_PARTS.items():
            out[c] = float(out.get(c, 0.0)) + share * v
    out = out[~out.index.isin(CASH_CODES)]
    return out[out > 1e-9].sort_values(ascending=False)


def _next_rebalance(idx: pd.DatetimeIndex, asof: pd.Timestamp,
                    freq: str = "M") -> pd.Timestamp | None:
    """freq="M"：asof 之后第一个「月份变化」的交易日；"W"：第一个「ISO 周变化」的交易日。"""
    later = idx[idx > asof]
    if len(later) == 0:
        return None
    if freq == "W":
        cand = later[later.isocalendar().week.to_numpy()
                     != asof.isocalendar().week]
    else:
        cand = later[later.month != asof.month]
    return cand[0] if len(cand) else None


def _load_positions(path: str | None) -> pd.Series:
    if not path:
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return pd.Series(df.set_index("code")["weight"].astype(float))


def _table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{label}</th>" for _, label in cols)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{r.get(key, '')}</td>" for key, _ in cols) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 轮动 champion 生产信号板（P5）")
    ap.add_argument("--capital", type=float, default=1e7, help="总资金（元）")
    ap.add_argument("--positions", default=None, help="当前持仓 CSV（code,weight）")
    ap.add_argument("--profile", choices=("champ", "contest"), default="champ",
                    help="champ=长线定版；contest=两个月比赛版（Top3/周频/满仓）")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认 champ→reports/etf_champ，contest→reports/etf_contest）")
    args = ap.parse_args(argv)
    out = Path(args.out or (ROOT / "reports" / (
        "etf_contest" if args.profile == "contest" else "etf_champ")))
    out.mkdir(parents=True, exist_ok=True)
    spec = CHAMP if args.profile == "champ" else CONTEST
    label = ("长线定版 champ" if args.profile == "champ"
             else "两个月比赛版（Top3/周频/满仓）")

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    uni, close, amount, _bench = bundle
    raw_close = (cache.read_daily("etf")["close"].unstack("code")
                 .reindex(index=close.index, columns=close.columns))
    panels = build_extra_panels(close, build_panels(close, amount), raw_close)
    extra = (pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})
             if spec.get("defense") else None)

    res = run_variant(bundle, panels, spec, None, None, extra_close=extra)
    rb = res["rebalance_dates"]
    asof = rb[-1]
    target = _expand_defense(res["weights"].loc[asof])
    sig_date = close.index[-1]
    amt60 = amount.rolling(60, min_periods=40).mean().loc[sig_date]
    crowd = panels["crowd"].loc[sig_date]
    prem = panels["prem20"].loc[sig_date] if "prem20" in panels else pd.Series(dtype=float)
    mask = panels["mask"].loc[sig_date]

    held = _load_positions(args.positions)
    codes = target.index.union(held.index)
    cur = held.reindex(codes).fillna(0.0)
    dw = target.reindex(codes).fillna(0.0) - cur
    order_amt = dw.abs() * args.capital
    part = order_amt / amt60.reindex(codes).replace(0.0, float("nan"))

    sig_rows, warn_n = [], 0
    for c in target.index:
        p, pr = float(part.get(c, np.nan)), float(prem.get(c, np.nan))
        flags = []
        if np.isfinite(p) and p > PARTICIPATION:
            flags.append(f"参与率>{PARTICIPATION:.0%}")
        if np.isfinite(pr) and abs(pr) > PREMIUM_WARN:
            flags.append(f"溢价{pr:+.1%}")
        warn_n += len(flags)
        sig_rows.append({
            "code": c,
            "name": uni.label(c),
            "category": uni.category(c),
            "weight": f"{target[c]:.2%}",
            "amount_wan": f"{target[c] * args.capital / 1e4:,.0f}",
            "order_wan": f"{dw.get(c, 0.0) * args.capital / 1e4:,.0f}",
            "amt60_yi": f"{float(amt60.get(c, np.nan)) / 1e8:.2f}"
                        if np.isfinite(amt60.get(c, np.nan)) else "—",
            "participation": f"{p:.2%}" if np.isfinite(p) else "—",
            "crowd_pct": f"{float(crowd.get(c, np.nan)):.0%}"
                         if np.isfinite(crowd.get(c, np.nan)) else "—",
            "prem20": f"{pr:+.2%}" if np.isfinite(pr) else "—",
            "flag": "；".join(flags) or "OK",
        })

    # 候选榜单：按原始 12-1 动量（仅可交易）排序，标注被哪条规则挡掉。
    raw_mom = panels["mom12_1"].loc[sig_date].where(mask)
    cand = raw_mom.dropna().sort_values(ascending=False).head(15)
    board_rows = []
    for rank, (c, mom) in enumerate(cand.items(), start=1):
        if c in target.index:
            status = "选中"
        elif np.isfinite(float(crowd.get(c, np.nan))) and float(crowd[c]) > 0.80:
            status = "拥挤剔除"
        elif mom <= 0:
            status = "动量非正"
        else:
            status = f"备选 #{rank}"
        board_rows.append({
            "rank": rank, "code": c, "name": uni.label(c), "category": uni.category(c),
            "mom12_1": f"{mom:+.1%}",
            "crowd_pct": f"{float(crowd.get(c, np.nan)):.0%}"
                         if np.isfinite(crowd.get(c, np.nan)) else "—",
            "prem20": f"{float(prem.get(c, np.nan)):+.2%}"
                      if np.isfinite(prem.get(c, np.nan)) else "—",
            "status": status,
        })

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(out / "signals.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(board_rows).to_csv(out / "candidates.csv", index=False,
                                    encoding="utf-8-sig")
    nxt = _next_rebalance(close.index, asof, spec.get("freq", "M"))
    nxt_txt = str(nxt.date()) if nxt is not None else (
        "下周首个交易日" if spec.get("freq") == "W" else "下月首个交易日")
    meta = {
        "profile": args.profile, "asof_rebalance": str(asof.date()),
        "signal_date": str(sig_date.date()),
        "next_rebalance": str(nxt.date()) if nxt is not None else None,
        "capital": args.capital, "pool_size": len(close.columns),
        "min_amount": MIN_AMOUNT, "participation_limit": PARTICIPATION,
        "premium_warn": PREMIUM_WARN, "warnings": warn_n,
        "strategy": spec, "bench": BENCH,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    risk_weight = float(target.drop(index=[c for c in DEF_PARTS if c in target.index],
                                    errors="ignore").sum())
    alloc = (f"风险腿 {risk_weight:.1%} + 防御腿 {1 - risk_weight:.1%}"
             if spec.get("defense") else f"满仓 {risk_weight:.1%}（无防御腿）")
    freq_txt = "每周第一个交易日" if spec.get("freq") == "W" else "每月第一个交易日"
    rule_txt = ("TopK=3；等权；无波动目标；拥挤度 = 份额 PIT 的 250 日分位（&gt; 0.80 剔除），"
                "仅作选标过滤" if args.profile == "contest" else
                "TopK=5；逆波动加权；15% 年化波动目标；防御腿 = 0.5·511010.SH + 0.5·518880.SH")
    body = f"""<div class="page">
<header>
  <h1>ETF 轮动 · 目标持仓与执行清单</h1>
  <div class="meta">
    <span>策略 {label}</span>
    <span>调仓日 {asof.date()}</span>
    <span>信号日 {sig_date.date()}</span>
    <span>下次调仓 {nxt_txt}</span>
    <span>资金 {args.capital / 1e4:,.0f} 万</span>
  </div>
</header>

<h2>目标持仓（{alloc}）</h2>
{_table(sig_rows, [("code", "代码"), ("name", "名称"), ("category", "类别"),
                    ("weight", "目标权重"), ("amount_wan", "目标金额(万)"),
                    ("order_wan", "下单(万)"), ("amt60_yi", "60日均额(亿)"),
                    ("participation", "参与率"), ("crowd_pct", "拥挤分位"),
                    ("prem20", "20日溢价"), ("flag", "提示")])}
<div class="note">参与率 = |下单金额| / 60 日均成交额，超过 {PARTICIPATION:.0%} 需拆单或分日执行；
20 日折溢价绝对值超过 {PREMIUM_WARN:.0%} 时，当日不在该价位追高（S3 结论：只作执行时点辅助）。
本次告警 {warn_n} 项。</div>

<h2>候选榜单（按 12-1 动量，仅可交易标的）</h2>
{_table(board_rows, [("rank", "#"), ("code", "代码"), ("name", "名称"),
                      ("category", "类别"), ("mom12_1", "12-1 动量"),
                      ("crowd_pct", "拥挤分位"), ("prem20", "20日溢价"),
                      ("status", "状态")])}
<div class="note">状态说明：选中=进入目标持仓；拥挤剔除=份额 250 日分位 &gt; 80%；
动量非正=绝对动量过滤；备选=满足条件但排名在 TopK 之外。</div>

<h2>口径</h2>
<div class="note">{freq_txt}调仓；{rule_txt}。
样本外结论见 <code>reports/etf_p4/README.md</code>：
OOS Sharpe 1.02、MDD 16.1%（2025-01–2026-10）；两个月比赛版依据
<code>reports/etf_contest/README.md</code>（80 个滚动 2 月窗口：中位 +2.44%、p10 −8.1%）。</div>
</div>"""

    html_path = out / "board.html"
    html_path.write_text(page("ETF 轮动 · 目标持仓", body=body, css=_CSS), encoding="utf-8")

    print(f"[champ] 调仓日 {asof.date()} · 信号日 {sig_date.date()} · "
          f"下次 {nxt_txt} · 资金 {args.capital / 1e4:,.0f} 万")
    print(sig_df[["code", "name", "weight", "order_wan", "participation", "prem20", "flag"]]
          .to_string(index=False))
    print(f"告警 {warn_n} 项；写入 {out}")


if __name__ == "__main__":
    main()
