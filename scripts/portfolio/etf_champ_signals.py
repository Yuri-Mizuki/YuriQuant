"""
ETF 轮动生产信号板（monthly / balanced）
========================================

两个 profile 共用同一套面板与回测引擎，参数全部来自 `etf_contest_spec`
（方案唯一真源：`reports/etf_contest5/README.md`）。2026-11-19 去比赛化：方案改英文
功能名，池子恢复 50 只，删除原 `contest5`（合规补位档）。

  - `monthly`「月频稳健」：**默认档**（月频 Top5 + 12-1 动量 + 拥挤度剔除 + 逆波动 +
    15% 波动目标 + 债金防御腿 + 去重 0.90 + buffer 15/40）。50 只池 T+1 open：
    13.18% / Sharpe 1.04 / MDD 13.42%，Calmar 0.98；
  - `balanced`「均衡轮动」：同选标、Top5 + HRP，中间档。5 相位均值 12.53% / 0.59 /
    MDD 31.3%，相位极差 5.14pp；
  - 🚨 原 `core`「集中轮动」已于 2026-11-19 **退役**：5 相位均值 14.27%/0.52/MDD 39.2%、
    最差相位 MDD 46.7%、Calmar 0.36；参数保留在 `etf_contest_spec.RETIRED_CORE`，
    仅供历史复现与探针基准。

落成可直接照单执行的信号板：

  1. **目标持仓**：用同一套 `VectorBacktest` + 策略在最新数据上重跑到最近一个调仓日，
     取该日权重；`DEF` 合成腿按 50/50 展开成 511010.SH（国债）+ 518880.SH（黄金）。
  2. **下单与容量**：`|目标权重 − 当前权重| × 资金` 为单只下单金额，按 60 日均成交额
     计参与率；> 5% 标红（方案 §三 P3 容量口径）。
  3. **执行时点检查**：20 日折溢价 > 1% 标红（S3 结论：只作执行时点辅助，
     不做截面过滤）。
  4. **监控**：持仓与候选的拥挤度分位（份额 250 日）、溢价、成交额；
     候选榜单标注「拥挤剔除 / 动量非正 / 备选」。
  5. 输出：`balanced` → `reports/etf_contest5/`，`monthly` → `reports/etf_champ/`
     （目录名为历史遗留，产物均为 `signals.csv`、`monitor.csv`、`board.html`、`meta.json`）。

用法:
    python -m scripts.portfolio.etf_champ_signals                       # monthly 档、1000 万、空仓建仓
    python -m scripts.portfolio.etf_champ_signals --profile balanced
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
from scripts.portfolio.etf_contest_spec import (
    DEFAULT_EXEC_CALIBER,
    EXEC_NOTE,
    RETIRED_CORE,
)
from scripts.portfolio.etf_rotation_panels import (
    CHAMP_BASE,
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

MONTHLY = {**CHAMP_BASE, "name": "monthly", "signal": "mom_crowd_f20", "defense": "mix",
           "dedup_corr": 0.90,
           # A3（2026-10-27 邻域检验）：去重后每个调仓日的有效候选清单很短，
           # 两个缓冲区宽度都不 binding，规则退化为「已持有且仍在清单里就不卖」。
           # 月换手 39.8%→32.1%（−19%）、成本 1.01%→0.80%/年，Sharpe +0.019。
           "enter_rank": 15, "exit_rank": 40}
# 集中轮动（原默认档「三锋」/ contest3）：🚨 **2026-11-19 退役**，不再进 PROFILES。
# 退役依据：50 只池 5 相位均值 14.27%/SR 0.52/MDD 39.2%、最差相位 MDD 46.7%、Calmar 0.36；
# 24.20% 的 k=1 是五个相位里唯一的正偏（其余 10.14%~13.31%）。参数见
# ``etf_contest_spec.RETIRED_CORE``（历史复现与探针基准仍用，不得删改）。
# 均衡轮动（原 contest5h「五衡」）：同选标、Top5 + HRP 风险平价（60 日协方差）。
# 50 只池 5 相位：12.53%/SR 0.59/MDD 31.3%（最差相位 36.2%），相位极差 5.14pp，
# 比 core 的 14.06pp 稳得多 ⇒ 现在的中间档。
# 🚨 2026-10-29 PIT 修复后：五衡 20.65% / SR 0.89 / MDD 33.27%，
# 三锋的收益、p10、最差窗口、MDD **四项全优**，配对差 −1.36%/窗（t=−1.97）
# **显著落后**。原「降档预案」用途已于 **2026-10-30 拍板取消**（处置 A）：
# 全样本 −9.98pp/年、贴近触发线 −2.62pp、**触发之后 −3.46pp（胜率仅 16.7%）**，
# 连"深跌时换防守"这个最后理由也不成立（README §8.2）⇒ 保留仅为对照。
# 🚨 同日 buffer 档位经**三轮修正**（见 README §4.1–4.2）：
# ① `enter_rank` 是死参数（3/5/8/10/15 净值逐位相同），buffer 只有 `exit_rank`
#    一个自由度；
# ② 非重叠 41 期下 `exit=20` 尾部确实更好（p10 −4.86% vs −7.54%、胜率 70.7%
#    vs 63.4%），但 **PIT 修复后年化不再打平**：25.43% vs 30.63%（落后 5.2pp），
#    期 Sharpe 0.46 vs 0.45 几乎相同；日 Sharpe 1.30 含 √252 对低换手的高估。
# ③ **样本外验证失败**：前 20 期 buffer +52.4% vs 三锋 +131.2%，优势只在 2024 之后，
#    逐年滚动选档 6 年出手 1 次亏 40pp ⇒ 事前不可识别，是付费的降波选项。
# ⇒ 保留为**对照档**。见 reports/etf_contest5/README.md §4.1、§8.1、§8.2。
BALANCED = {**RETIRED_CORE, "name": "balanced", "top_k": 5, "weighting": "hrp"}
#: 生产方案（2026-11-19 起两档）：``monthly`` 默认，``balanced`` 中间档。
#: 原 ``core`` 已退役，参数保留在 ``etf_contest_spec.RETIRED_CORE``。
PROFILES = {"balanced": BALANCED, "monthly": MONTHLY}
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


def _is_rebalance_day(idx: pd.DatetimeIndex, freq: str = "M") -> bool:
    """最新一根 bar 是否本身就是（周/月）首个交易日，即当日该调仓。"""
    if len(idx) < 2:
        return False
    period = "W" if freq == "W" else "M"
    return idx[-1].to_period(period) != idx[-2].to_period(period)


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
    ap = argparse.ArgumentParser(description="ETF 轮动生产信号板（balanced / monthly）")
    ap.add_argument("--capital", type=float, default=1e7, help="总资金（元）")
    ap.add_argument("--positions", default=None, help="当前持仓 CSV（code,weight）")
    ap.add_argument("--profile", choices=tuple(PROFILES), default="monthly",
                    help="monthly=月频稳健（默认）；balanced=均衡轮动（Top5/HRP，中间档）。"
                         "原 core（Top3 集中档）已于 2026-11-19 退役")
    ap.add_argument("--out", default=None,
                    help="输出目录（默认按 profile 落到 reports/etf_*）")
    ap.add_argument("--exclude", default=None,
                    help="不可交易的代码（逗号分隔）。默认不排除任何标的（50 只全池）；"
                         "复现比赛期 45 只口径时传 etf_contest_spec.EXCLUDED_5 的五个代码")
    ap.add_argument("--max-premium", type=float, default=None,
                    help="选标层折溢价闸门：|prem20| 超过该值（如 0.01）视为当日不可买。"
                         "PIT：prem20 只用当日及以前数据；避免补位时反复补进溢价品")
    ap.add_argument("--max-weight", type=float, default=None,
                    help="覆盖 profile 的单只权重上限（如 0.30）；传 0 表示不设上限，"
                         "用于观察 HRP 原始权重形状")
    args = ap.parse_args(argv)
    out = Path(args.out or (ROOT / "reports" / (
        {"balanced": "etf_contest5", "monthly": "etf_champ"}[args.profile])))
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(PROFILES[args.profile])
    if args.max_weight is not None:
        if args.max_weight <= 0:
            spec.pop("max_weight", None)
        else:
            spec["max_weight"] = args.max_weight
    label = {"balanced": "均衡轮动（Top5/HRP，中间档）",
             "monthly": "月频稳健（月度/波动目标/债金防御，默认档）"}[args.profile]

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    uni, close, amount, _bench = bundle
    raw_close = (cache.read_daily("etf")["close"].unstack("code")
                 .reindex(index=close.index, columns=close.columns))
    panels = build_extra_panels(close, build_panels(close, amount), raw_close)
    excluded = [c.strip() for c in (args.exclude or "").split(",") if c.strip()]
    if excluded:
        # 只把 mask 置 False：run_variant 会把因子 where(mask)，等价于这些标的
        # 在选标与建仓时不可交易（不影响已算好的因子历史，无前视）。
        miss = [c for c in excluded if c not in panels["mask"].columns]
        if miss:
            raise SystemExit(f"--exclude 含池外代码：{miss}")
        for c in excluded:
            panels["mask"].loc[:, c] = False
        print(f"[{args.profile}] 排除不可交易 {excluded}", flush=True)

    if args.max_premium is not None:
        prem20 = panels["prem20"]
        ok = (prem20.abs() <= args.max_premium) | prem20.isna()
        panels["mask"] = panels["mask"] & ok
        print(f"[{args.profile}] 折溢价闸门 |prem20| <= {args.max_premium:.2%}", flush=True)
    extra = (pd.DataFrame({"DEF": level_from_returns(build_defense(close)["mix"])})
             if spec.get("defense") else None)

    res = run_variant(bundle, panels, spec, None, None, extra_close=extra)
    rb = res["rebalance_dates"]
    asof = rb[-1]
    sig_date = close.index[-1]
    if asof != sig_date and _is_rebalance_day(close.index, spec.get("freq", "M")):
        # 最新交易日本身就是调仓日（周频=周一）时，引擎因「最后一根 bar 不能建仓」
        # 还停留在上一周。这里直接用当日因子/掩码取权重，否则实盘指令会落后一周。
        fac = panels[spec["signal"]].where(panels["mask"]).loc[sig_date].dropna()
        target = _expand_defense(res["strategy"].get_weights_at(sig_date, fac))
        print(f"[{args.profile}] 信号日 {sig_date.date()} 即调仓日，按当日权重出单"
              f"（次日开盘成交）", flush=True)
        asof = sig_date
    else:
        target = _expand_defense(res["weights"].loc[asof])
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
    uses_crowd = "crowd" in str(spec.get("signal", ""))
    board_rows = []
    for rank, (c, mom) in enumerate(cand.items(), start=1):
        if c in target.index:
            status = "选中"
        elif uses_crowd and np.isfinite(float(crowd.get(c, np.nan))) \
                and float(crowd[c]) > 0.80:
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
    # 下一次调仓：从**信号日**（最新交易日）往后找，避免周频时把已过去的调仓日
    # 当成"下次"。数据止于信号日时返回 None，由文案给出规则口径。
    nxt = _next_rebalance(close.index, sig_date, spec.get("freq", "M"))
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
        # 执行口径（2026-10-29 切换）：信号按信号日收盘数据算出，**次日开盘**成交。
        # 对外成绩一律按 open 口径引用（etf_contest_spec.EXEC_CALIBER[DEFAULT_EXEC_CALIBER]）。
        "exec_caliber": DEFAULT_EXEC_CALIBER, "exec_note": EXEC_NOTE,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")

    risk_weight = float(target.drop(index=[c for c in DEF_PARTS if c in target.index],
                                    errors="ignore").sum())
    alloc = (f"风险腿 {risk_weight:.1%} + 防御腿 {1 - risk_weight:.1%}"
             if spec.get("defense") else f"满仓 {risk_weight:.1%}（无防御腿）")
    freq_txt = "每周第一个交易日" if spec.get("freq") == "W" else "每月第一个交易日"
    exec_txt = ("收盘后按当日截面批量出信号 → **次日开盘**按目标权重下单成交"
                "（T+1 open；对外成绩口径）")
    rule_txt = {
        "balanced": ("TopK=5；12-1 动量 + 绝对动量过滤；"
                     "120 日相关 ≥0.90 去重；HRP 风险平价（60 日协方差）；无波动目标"),
        "monthly": "TopK=5；逆波动加权；15% 年化波动目标；防御腿 = 0.5·511010.SH + 0.5·518880.SH",
    }[args.profile]
    evidence_txt = {
        "balanced": ("中间档依据 <code>reports/etf_contest5/README.md</code> §四、§14.7。"
                     "50 只正常口径、T+1 开盘成交：<b>5 相位均值年化 12.53%、Sharpe 0.59、"
                     "MDD 31.3%（最差相位 36.2%）</b>，相位极差 5.14pp；"
                     "k=1 单相位为 14.79%/0.70/MDD 26.36%。"
                     "⚠️ 原「−15% 降档」用途已于 2026-10-30 取消（§8.2）。"),
        "monthly": ("<b>T+1 开盘口径 13.18% / Sharpe 1.04 / MDD 13.42%</b>"
                    "（2020-01 起，80 个 2 月窗口）；收盘成交口径 13.52%/1.07"
                    " ⇒ <b>执行税 −0.34pp/年</b>（月频 12 次/年 + 单只 17.6%，远低于集中档）。"
                    "月内调仓日摆动见 §五，勿线性外推。"),
    }[args.profile]
    body = f"""<div class="page">
<header>
  <h1>ETF 轮动 · 目标持仓与执行清单</h1>
  <div class="meta">
    <span>策略 {label}</span>
    <span>调仓日 {asof.date()}</span>
    <span>信号日 {sig_date.date()}</span>
    <span>下次调仓 {nxt_txt}</span>
    <span>执行 次日开盘（T+1 open）</span>
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
<div class="note">{freq_txt}调仓；{rule_txt}。<b>执行</b>：{exec_txt}。{evidence_txt}</div>
</div>"""

    html_path = out / "board.html"
    html_path.write_text(page("ETF 轮动 · 目标持仓", body=body, css=_CSS), encoding="utf-8")

    print(f"[{args.profile}] 调仓日 {asof.date()} · 信号日 {sig_date.date()} · "
          f"下次 {nxt_txt} · 资金 {args.capital / 1e4:,.0f} 万 · 执行 次日开盘")
    print(sig_df[["code", "name", "weight", "order_wan", "participation", "prem20", "flag"]]
          .to_string(index=False))
    print(f"告警 {warn_n} 项；写入 {out}")


if __name__ == "__main__":
    main()
