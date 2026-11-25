"""策略首页（历史回测详情）+ 模拟持仓页

页面结构
--------
``reports/strategies/index.html``
    策略首页：每条策略一张卡片（名称 / 规则一句话 / 净值火花线 / 年化·Sharpe·MDD），
    **点击只进该策略的历史回测详情页**；同时给因子看板入口。
``reports/strategies/etf_monthly.html``（ETF 月频稳健，默认档）
``reports/strategies/etf_balanced.html``（ETF 均衡轮动，中间档）
    详情页四段：
      ① 关键指标卡（年化 / Sharpe / MDD / Calmar / 波动 / 换手 / 调仓次数）
      ② 历史回测：策略 vs 沪深300 净值 + 回撤、分年度收益、2 月窗口收益分布
      ③ 当前目标持仓（信号日权重 + 1000 万资金下的下单金额）
      ④ 历史调仓记录（每个调仓日的持仓只数、区间收益、前三大权重）

``reports/strategies/inc_pure.html``（事件驱动 · 董监高增持）
    详情页五段：关键指标 / 净值与回撤（对市值匹配基准）/ 分年度 /
    模拟持仓（未走完持有窗）/ 逐笔记录。数字全部现算自
    ``strategy.event_driven.run``，不读缓存产物。

``reports/strategies/yqcore-focus-k8.html`` / ``yqcore-focus-k15.html`` /
``yqcore-breadth.html``（股票线三档）
    回测正文复用 ``scripts.reporting.yqcore_dashboard`` 的
    ``build_backtest_payload`` + ``render_backtest_body``（同一份产物不在两处各画一遍），
    本模块只加页首标题与共享顶栏。

``reports/strategies/live.html``（模拟持仓首页）
    只列**已开立纸面账户**的策略（真源 ``LIVE_KEYS``，当前仅 K8 定版）：
    信号日 / 持仓只数 / 期间收益 / 估值日，点开 ``live_<策略键>.html``。
    没有账户的档（K15 / 广度）不上这一页 —— 不给没有账户的档编持仓。

口径
----
与 ``reports/etf_contest5/README.md`` §零 一致：50 只池、2020-01 起单次连续回测、
T+1 开盘成交（``build_execution_split`` + ``execution_shift``）、成本按引擎默认。
**页面曲线是单相位 k=1（每月/每周首个交易日）**；对外引用的五相位均值写在各档
README 与卡片注记里，两者不可混。

ETF 的持仓历史直接取自同一遍回测的权重快照，不另存 CSV —— 持仓与曲线同源，
不会出现「页面持仓与回测权重不一致」。股票线的模拟持仓读生产账户
（``reports/yq_core/live/holdings_<信号日>.csv``），与回测口径分开标注。

用法::

    python -m scripts.reporting.strategy_home            # 首页 + 全部详情 + 模拟持仓
    python -m scripts.reporting.strategy_home --only etf_monthly
    python -m scripts.reporting.strategy_home --only inc_pure
    python -m scripts.reporting.strategy_home --only live
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import calendar as CAL
from backtest.engine import build_execution_split
from backtest.metrics import drawdown_series
from data.cache import DataCache
from data.datasource import create_datasource
from research.html_report import CHROME_CSS, page, site_nav
from scripts.portfolio.etf_champ_signals import (
    CASH_CODES,
    PROFILES,
    expand_defense,
    next_rebalance,
    sleeve_target,
)
from scripts.portfolio.etf_contest5_eval import two_month_windows, yearly
from scripts.portfolio.etf_rotation_panels import (
    build_defense,
    build_extra_panels,
    level_from_returns,
)
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)
from scripts.reporting import yqcore_dashboard as yd
from strategy import event_driven as evd
from strategy.catalog import CATALOG, cards

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "strategies"
YQCORE = ROOT / "reports" / "yq_core"
ECHARTS_SRC = "../docs/design/_shared/js/echarts.min.js"
#: 股票线详情页的 echarts 用外链脚本（与 ETF 详情页一致），不把 1MB 库塞进每页
ECHARTS_TAG = f'<script src="{ECHARTS_SRC}"></script>'
BEGIN = 20200101
CAPITAL = 1e7
ANNUAL = 252

#: 卡片顺序 / 名称 / 入口的真源 = ``strategy/catalog.py``；这里只放**页面文案**
#: （一句话规则 + 对账用的数字），改策略参数或新增档位不用碰本表。
PROFILE_META = {
    "etf_monthly": {
        "badge": "默认档",
        "desc": "每月首个交易日 · Top5 · 12-1 动量 · 逆波动加权 + 12% 波动目标",
        "evidence": ("月内五相位均值 <b>10.65% / Sharpe 0.94 / MDD 15.97%</b>"
                     "（Calmar 0.67）；k=1 单相位 13.08% / 1.13 / 13.97%。"
                     "方案与判据见 <code>reports/etf_contest5/README.md</code> §五、§14"),
    },
    "etf_balanced": {
        "badge": "中间档",
        "desc": "每周首个交易日 · Top5 · 12-1 动量 · ERC 等风险贡献",
        "evidence": ("5 相位均值 <b>13.33% / Sharpe 0.64 / MDD 29.89%</b>"
                     "（Calmar 0.45）；k=1 单相位 13.23% / 0.64 / 25.54%。"
                     "权重规则 2026-11-19 由 HRP 改 ERC，见 §14.11"),
    },
}
PROFILE_KEY = {k: e.profile for k, e in CATALOG.items() if e.kind == "etf"}
ETF_KEYS = tuple(e.key for e in sorted(CATALOG.values(), key=lambda x: x.order)
                 if e.kind == "etf")
#: 股票线三档：回测正文由 ``yqcore_dashboard`` 渲染（同一份产物不在两处各画一遍）
ALPHA_KEYS = tuple(e.key for e in sorted(CATALOG.values(), key=lambda x: x.order)
                   if e.kind == "alpha")
#: 已开立纸面模拟账户的策略。生产出单只跑 K8 定版（``--preset dingban``），
#: 其余档没有账户，因此不上「模拟持仓」页 —— 不要给它们编持仓。
LIVE_KEYS = ("yqcore-focus-k8",)

#: 股票线三档的一句话文案；**数字不写在这里**，一律读 ``reports/yq_core`` 的产物。
ALPHA_META = {
    "yqcore-focus-k8": ("定版 k=8",
                        "全 A 多因子 TopK8 · buffer 换手控制 · 行业去重"),
    "yqcore-focus-k15": ("k=15",
                         "同一组 alpha，持仓放宽到 15 只"),
    "yqcore-breadth": ("持仓 620+",
                       "同一组 alpha，宽基口径（持仓 620+ 只）"),
}

#: 事件驱动页的文案（数字由 ``strategy.event_driven.run`` 现算）
EVENT_META = {
    "badge": "事件驱动",
    "desc": "董监高本人增持 ≥0.1% 总股本 · 次日开盘买入 · 持有 20 个交易日",
    "evidence": ("归档报告（2026-11-15，旧面板）：毛超额 <b>+14.53%</b> / 扣费 +11.95% / "
                 "NW t 3.23 / 分年 9/9。当前面板（<code>_base</code> 2026-11-17 重建）重算见上方指标："
                 "年化绝对收益与 Sharpe 不变，<b>超额与分年正年数随市值分组基准小幅漂移</b>，"
                 "判据（t≥2、分年多数为正）不变。"),
}

_CSS = """
  body{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
       background:#f6f8fb;color:#17233b;line-height:1.6;margin:0;padding:0 0 3rem}
  .wrap{max-width:1180px;margin:0 auto;padding:0 1rem}
  .hero{padding:1.6rem 0 .6rem}
  .hero h1{margin:0 0 .3rem;font-size:1.6rem}
  .hero .sub{color:#4d5f7a;font-size:.92rem;max-width:900px}
  .tags{margin:.7rem 0 0;display:flex;flex-wrap:wrap;gap:.4rem}
  .tag{background:#e6eefb;color:#1b4b8f;border-radius:20px;padding:.15rem .6rem;
       font-size:.76rem}
  .tag.warn{background:#fdf0e3;color:#a35a00}
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
         gap:1rem;margin:.8rem 0 1.6rem}
  .grp{margin:1.5rem 0 0;font-size:.95rem;color:#123a6b;letter-spacing:.02em}
  .card{background:#fff;border:1px solid #e2e9f2;border-radius:12px;padding:1rem 1.1rem;
        box-shadow:0 1px 3px rgba(20,45,90,.05);transition:.15s}
  a.card{text-decoration:none;color:inherit;display:block}
  a.card:hover{border-color:#9db9e2;box-shadow:0 4px 14px rgba(20,45,90,.12);
               transform:translateY(-1px)}
  .card-h{display:flex;align-items:center;gap:.5rem;margin-bottom:.25rem}
  .card-h h3{margin:0;font-size:1.06rem}
  .card .desc{color:#5a6b85;font-size:.83rem;min-height:2.6em}
  .kpis{display:flex;gap:1rem;margin-top:.5rem;font-size:.8rem;color:#5a6b85}
  .kpis b{display:block;font-size:1.02rem;color:#17233b}
  .pos{color:#c0392b}.neg{color:#1e8449}
  .sect{margin-top:1.6rem;background:#fff;border:1px solid #e2e9f2;border-radius:12px;
        padding:1rem 1.1rem}
  .sect h2{margin:.1rem 0 .8rem;font-size:1.05rem;color:#123a6b}
  .kgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
         gap:.6rem;margin-bottom:.4rem}
  .kbox{background:#f7f9fc;border:1px solid #e6edf6;border-radius:9px;padding:.55rem .7rem}
  .kbox .lb{font-size:.75rem;color:#64748b}
  .kbox .vl{font-size:1.15rem;font-weight:600}
  table{width:100%;border-collapse:collapse;font-size:.84rem}
  th,td{padding:.4rem .55rem;border-bottom:1px solid #e6edf6;text-align:right}
  th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
  thead th{background:#f1f5fa;color:#334155;font-weight:600}
  tbody tr:hover{background:#fafcff}
  .chart{width:100%}
  .note{color:#64748b;font-size:.78rem;margin:.4rem 0 0}
  .scroll{max-height:420px;overflow:auto}
  .pos-txt{color:#c0392b}.neg-txt{color:#1e8449}
"""


# ---------------------------------------------------------------------------
# 数据装载与回测
# ---------------------------------------------------------------------------
def load_env() -> dict:
    """装载 ETF 面板（与 ``etf_contest5_eval`` 同源）。"""
    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    _uni, close, amount, _bench = bundle
    raw = cache.read_daily("etf")
    rc = raw["close"].unstack("code").reindex(index=close.index, columns=close.columns)
    ro = raw["open"].unstack("code").reindex(index=close.index, columns=close.columns)
    open_adj = (ro * (close / rc)).astype(np.float32)
    panels = build_extra_panels(close, build_panels(close, amount), rc)
    mfs = CAL.month_firsts(close.index)
    return {"bundle": bundle, "close": close, "panels": panels,
            "open_adj": open_adj, "mfs": mfs}


def run_profile(env: dict, key: str) -> tuple[dict, dict]:
    """跑单个 ETF 档：返回（spec, T+1 open 口径的结果）。

    ``monthly`` 的 DEF 合成腿必须同时补进 close 与执行价面板，否则该腿的权重
    会被吞（年化掉约 3.5pp，见 ``etf_contest5_eval.DEF_VARIANTS`` 注）。
    """
    spec = dict(PROFILES[PROFILE_KEY[key]])
    ex = None
    if spec.get("defense"):
        ex = pd.DataFrame({"DEF": level_from_returns(build_defense(env["close"])["mix"])})
    cl = env["close"].join(ex) if ex is not None else env["close"]
    od = env["open_adj"].join(ex) if ex is not None else env["open_adj"]
    r0 = run_variant(env["bundle"], env["panels"], spec, BEGIN, None, extra_close=ex)
    spl = build_execution_split(od, cl, set(pd.DatetimeIndex(r0["rebalance_dates"])))
    r1 = run_variant(env["bundle"], env["panels"], spec, BEGIN, None, extra_close=ex,
                     execution_split=spl, execution_shift=True)
    return spec, r1


def _r(v, nd: int = 6):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) \
        else round(float(v), nd)


def _snapshot(w: pd.Series, uni) -> dict:
    """一条调仓快照 → 可展示的持仓行（DEF 展开成两只真实 ETF，现金单独留一行）。"""
    tw = expand_defense(w)
    rows, cash = [], 1.0
    for c, v in tw.items():
        rows.append({"code": c, "name": uni.label(c), "category": uni.category(c),
                     "weight": _r(float(v))})
        cash -= float(v)
    if cash > 0.005:
        rows.append({"code": "CASH", "name": "现金", "category": "现金",
                     "weight": _r(cash)})
    return {"rows": rows, "cash": cash}


def build_etf_payload(env: dict, key: str) -> dict:
    spec, r1 = run_profile(env, key)
    uni = env["bundle"][0]
    eq: pd.Series = r1["equity"].dropna()
    idx = eq.index
    bench = env["bundle"][3].reindex(idx).fillna(0.0)
    nav_s = eq / eq.iloc[0]
    nav_b = (1.0 + bench).cumprod()
    dd = drawdown_series(nav_s)
    ann = float(r1["annual_return"])
    mdd = float(r1["max_drawdown"])
    vol = float(eq.pct_change().dropna().std() * np.sqrt(ANNUAL))

    # 分年度
    yr_s = yearly(eq)
    yr_b = bench.groupby(bench.index.year).apply(lambda x: float((1 + x).prod() - 1))
    years = [str(int(y)) for y in yr_s.index]

    # 2 月窗口（与评测同法：连续回测后按月首切窗）
    win = two_month_windows(eq, env["mfs"])

    # 调仓快照（信号日 → 下一信号日区间收益）
    rb = pd.DatetimeIndex(r1["rebalance_dates"])
    w = r1["weights"]
    eq_at = eq.reindex(rb).ffill()
    hist = []
    for i, d in enumerate(rb):
        nxt = rb[i + 1] if i + 1 < len(rb) else eq.index[-1]
        ret = float(eq_at.iloc[i + 1] / eq_at.iloc[i] - 1.0) if i + 1 < len(eq_at) \
            else float(eq.loc[nxt] / eq_at.iloc[i] - 1.0)
        snap = _snapshot(w.loc[d], uni)
        top = sorted(snap["rows"], key=lambda r: -r["weight"])[:3]
        top_txt = " · ".join(f"{x['name']} {x['weight'] * 100:.1f}%" for x in top
                             if x["code"] != "CASH")
        hist.append({
            "date": d.strftime("%Y-%m-%d"),
            "next": nxt.strftime("%Y-%m-%d"),
            "n": sum(1 for r in snap["rows"] if r["code"] not in CASH_CODES),
            "cash": _r(snap["cash"]),
            "ret": _r(ret),
            "top": top_txt,
        })
    cur = _snapshot(w.loc[rb[-1]], uni)

    # 最新一个调仓日的目标权重：引擎不允许在最后一根 bar 建仓，回测的最后一个
    # 调仓快照会停在上一期（monthly 是 10 月）。这里走生产同一条路径
    # ``sleeve_target`` 取最新信号日权重，保证页面持仓 == 信号板 signals.csv。
    sig_date = env["close"].index[-1]
    try:
        target, asof = sleeve_target(env["bundle"], env["panels"], env["close"],
                                      spec, sig_date)
        cur = _snapshot(target, uni)
        hold_asof = asof.strftime("%Y-%m-%d")
    except Exception:                                       # noqa: BLE001
        hold_asof = rb[-1].strftime("%Y-%m-%d")
    for r in cur["rows"]:
        r["amount_wan"] = _r(r["weight"] * CAPITAL / 1e4, 1)
    nxt = next_rebalance(env["close"].index, sig_date, spec.get("freq", "M"))
    nxt_txt = (nxt.strftime("%Y-%m-%d") if nxt is not None
               else ("下周首个交易日" if spec.get("freq") == "W" else "下月首个交易日"))

    meta = PROFILE_META[key]
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kind": "etf",
        "key": key,
        "name": CATALOG[key].label,
        "badge": meta["badge"],
        "desc": meta["desc"],
        "evidence": meta["evidence"],
        "profile": PROFILE_KEY[key],
        "freq": "月频" if spec.get("freq", "M") == "M" else "周频",
        "weighting": {"inverse_vol": "逆波动（σ≥10%）", "erc": "ERC 等风险贡献"}
                     .get(spec.get("weighting", ""), spec.get("weighting", "")),
        "start": idx[0].strftime("%Y-%m-%d"),
        "end": idx[-1].strftime("%Y-%m-%d"),
        "caliber": ("50 只 ETF 池 · T+1 开盘成交 · 单相位 k=1（调仓周期首个交易日）"
                    "· 成本按引擎默认（佣金万1双边 + 滑点 5bp）"),
        "kpi": {
            "ann": _r(ann), "sharpe": _r(r1["sharpe"], 3), "mdd": _r(mdd),
            "calmar": _r(ann / abs(mdd)) if mdd else None, "vol": _r(vol),
            "turnover": _r(r1["avg_turnover"]), "n_rb": int(len(rb)),
            "avg_hold": _r(r1["avg_holdings"], 1), "avg_cash": _r(r1["avg_cash"]),
        },
        "nav": {
            "dates": [d.strftime("%Y-%m-%d") for d in idx],
            "strat": [_r(v) for v in nav_s.to_numpy()],
            "bench": [_r(v) for v in nav_b.to_numpy()],
            "dd": [_r(v) for v in dd.reindex(idx).fillna(0.0).to_numpy()],
        },
        "yearly": {"years": years,
                   "strat": [_r(float(yr_s.loc[int(y)])) for y in years],
                   "bench": [_r(float(yr_b.get(int(y), np.nan))) for y in years]},
        "windows": {
            "dates": [d.strftime("%Y-%m") for d in win.index],
            "values": [_r(float(v)) for v in win.to_numpy()],
            "stats": {"mean": _r(float(win.mean())), "p10": _r(float(win.quantile(.1))),
                      "worst": _r(float(win.min())),
                      "win_rate": _r(float((win > 0).mean()))},
        },
        "asof": hold_asof,
        "next_rb": nxt_txt,
        "hold": {"capital": CAPITAL, "rows": cur["rows"], "cash": _r(cur["cash"])},
        "hist": hist,
    }


# ---------------------------------------------------------------------------
# 事件驱动（inc_pure）
# ---------------------------------------------------------------------------
def build_event_payload() -> dict:
    """事件驱动详情页数据：全部现算自 ``strategy.event_driven.run``。

    与 ETF 详情页的差别：这里没有「本期目标权重」这个概念，持仓由
    「最近 20 个交易日内还有哪些事件没走完」决定，所以持仓表给的是
    **建仓日 / 剩余交易日 / 浮动盈亏**，不是调仓权重。
    """
    r = evd.run()
    k = r["kpi"]
    rows = [{
        "code": o["code"], "name": o["name"], "trigger": o["trigger"],
        "buy": o["buy"], "held": o["held"], "left": o["left"],
        "strength": _r(o["strength"], 2), "n_ev": o["n_ev"],
        "ret": _r(o["ret_net"]),
    } for o in r["open"]]
    trades = [{
        "trigger": t["trigger"], "code": t["code"], "name": t["name"],
        "buy": t["buy"], "sell": t["sell"], "days": t["days"],
        "strength": _r(t["strength"], 2), "ret": _r(t["ret_net"]),
    } for t in r["trades"]]
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kind": "event",
        "key": "inc_pure",
        "name": CATALOG["inc_pure"].label,
        "badge": EVENT_META["badge"],
        "desc": EVENT_META["desc"],
        "evidence": EVENT_META["evidence"],
        "start": r["start"],
        "end": r["end"],
        "hold_days": r["hold"],
        "n_events": r["n_events"],
        "caliber": (f"全 A · 持有 {r['hold']} 个交易日 · T+1 开盘买入 / 到期日开盘卖出"
                    f" · 单边成本 {evd.COST_ONE_WAY * 1e4:.1f}bp（固定）"
                    " · 基准 = 市值 5 分组等权（前一日市值）"),
        "kpi": {
            "ann": _r(k["ann_ret"]), "ex": _r(k["ann_ex"]),
            "ex_net": _r(k["ann_ex_net"]), "t": _r(k["t"], 2),
            "sharpe": _r(k["sharpe"], 3), "mdd": _r(k["maxdd"]),
            "turnover": _r(k["turnover"], 1), "cost": _r(k["cost"]),
            "avg_hold": _r(k["avg_hold"], 1), "n_pos": int(k["n_pos"]),
            "yr_pos": k["yr_pos"],
        },
        "nav": {
            "dates": r["dates"],
            "strat": [_r(v) for v in r["nav"]],
            "bench": [_r(v) for v in r["bench_nav"]],
            "dd": [_r(v) for v in r["dd"]],
            "count": [int(v) for v in r["count"]],
        },
        "yearly": {"years": r["years"],
                   "strat": [_r(float(v)) for v in r["yr_strat"]],
                   "bench": [_r(float(v)) for v in r["yr_bench"]]},
        "hold": {"asof": r["end"], "rows": rows},
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# 页面片段
# ---------------------------------------------------------------------------
def _fmt(v, pct: bool = True, nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.{nd}f}%" if pct else f"{v:.{nd}f}"


def _cls(v) -> str:
    if v is None:
        return ""
    return "pos-txt" if v >= 0 else "neg-txt"


def _spark_svg(values: list, w: int = 300, h: int = 52, color: str = "#1b5fbf") -> str:
    """净值火花线（纯 SVG，首页不引 ECharts）。"""
    pts = [v for v in values if v is not None]
    if len(pts) < 2:
        return ""
    step = max(1, len(pts) // 160)
    pts = pts[::step] + [pts[-1]]
    lo, hi = min(pts), max(pts)
    rng = (hi - lo) or 1.0
    xy = [(i * w / (len(pts) - 1), h - (v - lo) / rng * (h - 4) - 2)
          for i, v in enumerate(pts)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    area = f"0,{h} " + line + f" {w},{h}"
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'style="width:100%;height:{h}px;margin:.5rem 0 .2rem">'
            f'<polygon points="{area}" fill="{color}" opacity="0.08"/>'
            f'<polyline points="{line}" fill="none" stroke="{color}" '
            f'stroke-width="1.6"/></svg>')


def _nav(active: str, generated: str) -> str:
    """站点顶栏（共享实现：``research.html_report.site_nav``）。"""
    return site_nav(active, generated=generated, root="..",
                    local={"home": "index.html", "live": "live.html"})


def _kbox(label: str, value: str, cls: str = "") -> str:
    return f"<div class='kbox'><div class='lb'>{label}</div><div class='vl {cls}'>{value}</div></div>"


def _hold_table(rows: list[dict], with_amount: bool) -> str:
    amt = "<th>金额（万元）</th>" if with_amount else ""
    body = "".join(
        f"<tr><td>{r['code']}</td><td>{r['name']}</td><td>{r['category']}</td>"
        f"<td>{r['weight'] * 100:.2f}%</td>"
        + (f"<td>{r['amount_wan']:,.0f}</td>" if with_amount else "")
        + "</tr>" for r in rows)
    return (f"<table><thead><tr><th>代码</th><th>名称</th><th>类别</th><th>权重</th>"
            f"{amt}</tr></thead><tbody>{body}</tbody></table>")


def _hist_table(hist: list[dict], limit: int = 60) -> str:
    rows = list(reversed(hist[-limit:]))
    body = "".join(
        f"<tr><td>{h['date']}</td><td>{h['next']}</td><td>{h['n']}</td>"
        f"<td>{h['cash'] * 100:.1f}%</td>"
        f"<td class='{_cls(h['ret'])}'>{h['ret'] * 100:+.2f}%</td>"
        f"<td>{h['top']}</td></tr>" for h in rows)
    return ("<table><thead><tr><th>调仓日</th><th>至</th><th>持仓数</th><th>现金</th>"
            "<th>区间收益</th><th>前三大</th></tr></thead><tbody>"
            + body + "</tbody></table>")


def render_detail(p: dict) -> str:
    k = p["kpi"]
    ws = p["windows"]["stats"]
    data = json.dumps(p, ensure_ascii=False).replace("</", "<\\/")
    kpis = "".join([
        _kbox("年化收益", _fmt(k["ann"]), _cls(k["ann"])),
        _kbox("Sharpe", _fmt(k["sharpe"], False, 2)),
        _kbox("最大回撤", _fmt(k["mdd"]), "neg-txt"),
        _kbox("Calmar", _fmt(k["calmar"], False, 2)),
        _kbox("年化波动", _fmt(k["vol"])),
        _kbox("月/周换手", _fmt(k["turnover"], True, 1)),
        _kbox("调仓次数", str(k["n_rb"])),
        _kbox("平均持仓", f"{k['avg_hold']:.1f} 只"),
        _kbox("平均现金", _fmt(k["avg_cash"], True, 1)),
    ])
    return f"""
{_nav("home", p["generated"])}
<div class="wrap">
  <div class="hero">
    <h1>{p['name']} <span class="tag">{p['badge']}</span></h1>
    <div class="sub">{p['desc']}</div>
    <div class="tags">
      <span class="tag">profile <code>{p['profile']}</code></span>
      <span class="tag">{p['freq']}调仓</span>
      <span class="tag">加权 {p['weighting']}</span>
      <span class="tag">{p['start']} ~ {p['end']}</span>
      <span class="tag warn">口径：{p['caliber']}</span>
    </div>
  </div>

  <div class="sect">
    <h2>关键指标</h2>
    <div class="kgrid">{kpis}</div>
    <div class="note">{p['evidence']}</div>
  </div>

  <div class="sect">
    <h2>历史回测 · 净值与回撤</h2>
    <div id="c_nav" class="chart" style="height:420px"></div>
    <div class="note">策略净值（左轴，对数）对沪深300（左轴）；下方为回撤。
      成本与成交口径见页首注记，曲线为单相位 k=1。</div>
  </div>

  <div class="sect">
    <h2>分年度收益</h2>
    <div id="c_year" class="chart" style="height:280px"></div>
  </div>

  <div class="sect">
    <h2>2 月窗口收益分布</h2>
    <div class="note">窗口均值 <b>{_fmt(ws['mean'])}</b> · p10 {_fmt(ws['p10'])} ·
      最差 {_fmt(ws['worst'])} · 胜率 {_fmt(ws['win_rate'], True, 1)}</div>
    <div id="c_win" class="chart" style="height:260px"></div>
  </div>

  <div class="sect">
    <h2>当前目标持仓 · 信号日 {p['asof']}</h2>
    <div class="note">按 {(p['hold']['capital'] / 1e4):,.0f} 万资金换算下单金额；
      现金行 = 未满仓部分。下次调仓：{p['next_rb']}（收盘后出信号，次日开盘成交）。</div>
    {_hold_table(p['hold']['rows'], True)}
  </div>

  <div class="sect">
    <h2>历史调仓记录（最近 {min(60, len(p['hist']))} 期）</h2>
    <div class="note">区间收益 = 本调仓日净值 → 下一调仓日净值；共
      {len(p['hist'])} 期。</div>
    <div class="scroll">{_hist_table(p['hist'])}</div>
  </div>
</div>
<script src="{ECHARTS_SRC}"></script>
<script>window.__P__ = {data};</script>
<script>{_DETAIL_JS}</script>
"""


_DETAIL_JS = """
const P = window.__P__;
const RED = '#c0392b', GREEN = '#1e8449', BLUE = '#1b5fbf', GREY = '#94a3b8';
function mk(id, opt) {
  const el = document.getElementById(id); if (!el) return;
  const c = echarts.init(el); c.setOption(opt);
  window.addEventListener('resize', () => c.resize());
}
const nav = P.nav;
mk('c_nav', {
  animation: false,
  tooltip: {trigger: 'axis'},
  legend: {data: ['策略', '沪深300', '回撤'], top: 4},
  grid: [{left: 58, right: 24, top: 34, height: '52%'},
         {left: 58, right: 24, top: '68%', height: '20%'}],
  xAxis: [{type: 'category', data: nav.dates, boundaryGap: false, axisLabel: {show: false}},
          {type: 'category', gridIndex: 1, data: nav.dates, boundaryGap: false,
           axisLabel: {fontSize: 10}}],
  yAxis: [{type: 'log', name: '净值', scale: true, axisLabel: {fontSize: 10}},
          {gridIndex: 1, name: '回撤', max: 0, axisLabel: {
            formatter: v => (v * 100).toFixed(0) + '%', fontSize: 10}}],
  dataZoom: [{type: 'inside', xAxisIndex: [0, 1]},
             {type: 'slider', xAxisIndex: [0, 1], height: 16, bottom: 2}],
  series: [
    {name: '策略', type: 'line', data: nav.strat, showSymbol: false,
     lineStyle: {width: 1.8, color: BLUE}, itemStyle: {color: BLUE}},
    {name: '沪深300', type: 'line', data: nav.bench, showSymbol: false,
     lineStyle: {width: 1.2, color: GREY}, itemStyle: {color: GREY}},
    {name: '回撤', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: nav.dd,
     showSymbol: false, areaStyle: {color: 'rgba(192,57,43,.18)'},
     lineStyle: {width: 1, color: RED}, itemStyle: {color: RED}}
  ]
});
const y = P.yearly;
mk('c_year', {
  animation: false,
  tooltip: {trigger: 'axis', valueFormatter: v => v == null ? '-' : (v * 100).toFixed(2) + '%'},
  legend: {data: ['策略', '沪深300'], top: 2},
  grid: {left: 52, right: 20, top: 30, bottom: 24},
  xAxis: {type: 'category', data: y.years},
  yAxis: {type: 'value', axisLabel: {formatter: v => (v * 100).toFixed(0) + '%'}},
  series: [
    {name: '策略', type: 'bar', data: y.strat, barWidth: '32%',
     itemStyle: {color: p => p.value >= 0 ? RED : GREEN}},
    {name: '沪深300', type: 'bar', data: y.bench, barWidth: '32%',
     itemStyle: {color: p => p.value >= 0 ? 'rgba(192,57,43,.35)' : 'rgba(30,132,73,.35)'}}
  ]
});
mk('c_win', {
  animation: false,
  tooltip: {trigger: 'axis', valueFormatter: v => v == null ? '-' : (v * 100).toFixed(2) + '%'},
  grid: {left: 52, right: 20, top: 20, bottom: 40},
  xAxis: {type: 'category', data: P.windows.dates,
          axisLabel: {fontSize: 9, interval: Math.ceil(P.windows.dates.length / 12)}},
  yAxis: {type: 'value', axisLabel: {formatter: v => (v * 100).toFixed(0) + '%'}},
  series: [{type: 'bar', data: P.windows.values, barWidth: '72%',
            itemStyle: {color: p => p.value >= 0 ? RED : GREEN}}]
});
"""


# ---------------------------------------------------------------------------
# 事件驱动详情页
# ---------------------------------------------------------------------------
def _event_hold_table(rows: list[dict]) -> str:
    if not rows:
        return "<div class='note'>截至回测末日没有未走完持有窗的持仓。</div>"
    body = "".join(
        f"<tr><td>{r['code']}</td><td>{r['name']}</td><td>{r['trigger']}</td>"
        f"<td>{r['buy']}</td><td>{r['held']}</td><td>{r['left']}</td>"
        f"<td>{r['strength']:.2f}</td><td>{r['n_ev']}</td>"
        f"<td class='{_cls(r['ret'])}'>{r['ret'] * 100:+.2f}%</td></tr>"
        for r in rows)
    return ("<table><thead><tr><th>代码</th><th>名称</th><th>事件可见日</th>"
            "<th>建仓日（开盘）</th><th>已持有</th><th>剩余交易日</th>"
            "<th>增持合计(%)</th><th>事件数</th><th>浮动盈亏(扣费)</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def _event_trade_table(trades: list[dict], limit: int = 80) -> str:
    rows = trades[:limit]
    body = "".join(
        f"<tr><td>{t['trigger']}</td><td>{t['code']}</td><td>{t['name']}</td>"
        f"<td>{t['buy']}</td><td>{t['sell']}</td><td>{t['days']}</td>"
        f"<td>{t['strength']:.2f}</td>"
        f"<td class='{_cls(t['ret'])}'>{t['ret'] * 100:+.2f}%</td></tr>"
        for t in rows)
    return ("<table><thead><tr><th>事件可见日</th><th>代码</th><th>名称</th>"
            "<th>买入(开盘)</th><th>卖出(开盘)</th><th>持有交易日</th>"
            "<th>增持合计(%)</th><th>收益(扣费)</th></tr></thead>"
            f"<tbody>{body}</tbody></table>")


def render_event_detail(p: dict) -> str:
    k = p["kpi"]
    data = json.dumps(p, ensure_ascii=False).replace("</", "<\\/")
    kpis = "".join([
        _kbox("年化收益", _fmt(k["ann"]), _cls(k["ann"])),
        _kbox("超额（市值匹配）", _fmt(k["ex"]), _cls(k["ex"])),
        _kbox("超额（扣费）", _fmt(k["ex_net"]), _cls(k["ex_net"])),
        _kbox("NW t", _fmt(k["t"], False, 2)),
        _kbox("Sharpe", _fmt(k["sharpe"], False, 2)),
        _kbox("最大回撤", _fmt(k["mdd"]), "neg-txt"),
        _kbox("年化换手（双边）", f"{k['turnover']:.1f}×"),
        _kbox("年化成本", _fmt(k["cost"])),
        _kbox("平均持仓", f"{k['avg_hold']:.1f} 只"),
        _kbox("回测笔数", str(k["n_pos"])),
    ])
    return f"""
{_nav("home", p["generated"])}
<div class="wrap">
  <div class="hero">
    <h1>{p['name']} <span class="tag">{p['badge']}</span></h1>
    <div class="sub">{p['desc']}</div>
    <div class="tags">
      <span class="tag">持有 {p['hold_days']} 个交易日</span>
      <span class="tag">事件 {p['n_events']:,} 条</span>
      <span class="tag">{p['start']} ~ {p['end']}</span>
      <span class="tag warn">口径：{p['caliber']}</span>
    </div>
  </div>

  <div class="sect">
    <h2>关键指标</h2>
    <div class="kgrid">{kpis}</div>
    <div class="note">{p['evidence']}</div>
  </div>

  <div class="sect">
    <h2>历史回测 · 净值与回撤</h2>
    <div id="c_nav" class="chart" style="height:420px"></div>
    <div class="note">策略逐日等权持有全部在窗股票，无事件日空仓（收益记 0）；
      基准为持仓股当日所在市值组的等权收益。下方为回撤。</div>
  </div>

  <div class="sect">
    <h2>分年度收益</h2>
    <div id="c_year" class="chart" style="height:280px"></div>
    <div class="note">分年超额为正 <b>{k['yr_pos']}</b> 年（见页首判据与证据注记）。</div>
  </div>

  <div class="sect">
    <h2>模拟持仓 · 截至 {p['hold']['asof']}</h2>
    <div class="note">最近 {p['hold_days']} 个交易日内触发、尚未走完持有窗的仓位；
      浮动盈亏按最后收盘价、扣往返成本估算，不是成交回报。</div>
    {_event_hold_table(p['hold']['rows'])}
  </div>

  <div class="sect">
    <h2>逐笔记录（最近 {min(80, len(p['trades']))} 笔）</h2>
    <div class="note">共 {len(p['trades'])} 笔已了结；收益 = 卖出开盘 / 买入开盘 − 1 − 往返成本。</div>
    <div class="scroll">{_event_trade_table(p['trades'])}</div>
  </div>
</div>
<script src="{ECHARTS_SRC}"></script>
<script>window.__P__ = {data};</script>
<script>{_EVENT_JS}</script>
"""


_EVENT_JS = """
const P = window.__P__;
const RED = '#c0392b', GREEN = '#1e8449', BLUE = '#1b5fbf', GREY = '#94a3b8';
function mk(id, opt) {
  const el = document.getElementById(id); if (!el) return;
  const c = echarts.init(el); c.setOption(opt);
  window.addEventListener('resize', () => c.resize());
}
const nav = P.nav;
mk('c_nav', {
  animation: false,
  tooltip: {trigger: 'axis'},
  legend: {data: ['策略', '市值匹配基准', '回撤'], top: 4},
  grid: [{left: 58, right: 24, top: 34, height: '52%'},
         {left: 58, right: 24, top: '68%', height: '20%'}],
  xAxis: [{type: 'category', data: nav.dates, boundaryGap: false, axisLabel: {show: false}},
          {type: 'category', gridIndex: 1, data: nav.dates, boundaryGap: false,
           axisLabel: {fontSize: 10}}],
  yAxis: [{type: 'log', name: '净值', scale: true, axisLabel: {fontSize: 10}},
          {gridIndex: 1, name: '回撤', max: 0, axisLabel: {
            formatter: v => (v * 100).toFixed(0) + '%', fontSize: 10}}],
  dataZoom: [{type: 'inside', xAxisIndex: [0, 1]},
             {type: 'slider', xAxisIndex: [0, 1], height: 16, bottom: 2}],
  series: [
    {name: '策略', type: 'line', data: nav.strat, showSymbol: false,
     lineStyle: {width: 1.8, color: BLUE}, itemStyle: {color: BLUE}},
    {name: '市值匹配基准', type: 'line', data: nav.bench, showSymbol: false,
     lineStyle: {width: 1.2, color: GREY}, itemStyle: {color: GREY}},
    {name: '回撤', type: 'line', xAxisIndex: 1, yAxisIndex: 1, data: nav.dd,
     showSymbol: false, areaStyle: {color: 'rgba(192,57,43,.18)'},
     lineStyle: {width: 1, color: RED}, itemStyle: {color: RED}}
  ]
});
const y = P.yearly;
mk('c_year', {
  animation: false,
  tooltip: {trigger: 'axis', valueFormatter: v => v == null ? '-' : (v * 100).toFixed(2) + '%'},
  legend: {data: ['策略', '市值匹配基准'], top: 2},
  grid: {left: 52, right: 20, top: 30, bottom: 24},
  xAxis: {type: 'category', data: y.years},
  yAxis: {type: 'value', axisLabel: {formatter: v => (v * 100).toFixed(0) + '%'}},
  series: [
    {name: '策略', type: 'bar', data: y.strat, barWidth: '32%',
     itemStyle: {color: p => p.value >= 0 ? RED : GREEN}},
    {name: '市值匹配基准', type: 'bar', data: y.bench, barWidth: '32%',
     itemStyle: {color: p => p.value >= 0 ? 'rgba(192,57,43,.35)' : 'rgba(30,132,73,.35)'}}
  ]
});
"""


# ---------------------------------------------------------------------------
# 股票线详情页（回测正文复用 yqcore_dashboard）
# ---------------------------------------------------------------------------
_ALPHA_CSS = """
  .sh-head{background:#fff;border-bottom:1px solid #e4e7ec;padding:16px 20px}
  .sh-head h1{margin:0 0 .25rem;font-size:1.35rem;color:#1b2029}
  .sh-head .sub{color:#5a6b85;font-size:.88rem;max-width:1000px}
  .sh-head .tags{margin:.55rem 0 0;display:flex;flex-wrap:wrap;gap:.4rem}
  .sh-head .tag{background:#eaf2ff;color:#2f6fdb;border:1px solid #cfe0fb;
                border-radius:2px;padding:2px 7px;font-size:11px}
"""


def build_alpha_payload(key: str) -> dict:
    """股票线单档回测数据（正文由 ``yqcore_dashboard`` 渲染，不重复实现）。"""
    return yd.build_backtest_payload(YQCORE, yd.BASE_DIR, tier=key)


def render_alpha_detail(p: dict) -> str:
    e = CATALOG[p["tier"]]
    badge, desc = ALPHA_META.get(p["tier"], ("股票线", ""))
    return f"""
{_nav("home", p["generated"])}
<div class="sh-head">
  <h1>{e.label} <span class="tag">{badge}</span></h1>
  <div class="sub">{desc}</div>
  <div class="tags">
    <span class="tag">窗口 {p['window']}</span>
    <span class="tag">成本：{p['cost_note']}</span>
    <span class="tag">信号 {p['signal']}</span>
  </div>
</div>
{yd.render_backtest_body(p, with_nav=False, with_cross=False)}
"""


# ---------------------------------------------------------------------------
# 模拟持仓（独立首页 + 单策略详情）
# ---------------------------------------------------------------------------
def build_live_payload(key: str) -> dict:
    """纸面账户当期持仓（K 线只取前 12 只大仓位，页面不至于过大）。"""
    return yd.build_live_payload(YQCORE, YQCORE / "live", tier=key, kline_max=12)


def _live_card(entry, p: dict) -> str:
    t = p["hold"]["totals"]
    chk = "通过" if p["check"]["ok"] else "未通过"
    return f"""
<a class="card" href="live_{entry.key}.html">
  <div class="card-h"><h3>{entry.label}</h3><span class="tag">模拟持仓</span></div>
  <div class="desc">纸面账户 · 信号日 {p['hold']['signal_date']} · 成交 {p['hold']['exec_date']}
    · T+1 开盘 · 目标 {p['hold']['k']} 只</div>
  <div class="kpis">
    <span>持仓<b>{len(p['hold']['rows'])} 只</b></span>
    <span>期间收益<b class="{_cls(t['ret'])}">{_fmt(t['ret'])}</b></span>
    <span>浮动盈亏<b class="{_cls(t['pnl'])}">{t['pnl']:,.0f} 元</b></span>
    <span>估值日<b>{p['hold']['latest_date']}</b></span>
  </div>
  <div class="note">自检{chk}：股数×收盘重算 {_fmt(p['check']['computed'], True, 3)}
    vs 落盘 meta {_fmt(p['check']['claimed'], True, 3)}</div>
</a>"""


def render_live_home(payloads: dict[str, dict]) -> str:
    parts = "".join(_live_card(CATALOG[k], payloads[k]) for k in LIVE_KEYS
                    if k in payloads)
    if not parts:
        parts = ("<div class='note'>当前没有已开立的模拟账户（缺 "
                 "<code>reports/yq_core/live/holdings_*.csv</code>）。</div>")
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
{_nav("live", gen)}
<div class="wrap">
  <div class="hero">
    <h1>模拟持仓</h1>
    <div class="sub">只列<b>已开立纸面账户</b>的策略，点开看当前持仓、持仓期表现与个股 K 线。
      没有账户的档不出现在这里 —— 不给没有账户的档编持仓。</div>
  </div>
  <div class="cards">{parts}</div>
  <div class="note">账户来自生产出单链（<code>reports/yq_core/live/holdings_&lt;信号日&gt;.csv</code>）；
    市值按股数 × 最新收盘重算，与回测净值是两套口径，不混用。</div>
</div>
"""


def render_live_detail(p: dict) -> str:
    return f"""
{_nav("live", p["generated"])}
{yd.render_live_body(p, with_nav=False, with_cross=False)}
"""


# ---------------------------------------------------------------------------
# 首页
# ---------------------------------------------------------------------------
def _kpis(ann, sharpe, mdd) -> str:
    """首页三张卡片统一的指标：年化 / Sharpe / MDD（数值缺失显示 -）。"""
    return f"""
  <div class="kpis">
    <span>年化<b class="{_cls(ann)}">{_fmt(ann)}</b></span>
    <span>Sharpe<b>{_fmt(sharpe, False, 2)}</b></span>
    <span>MDD<b>{_fmt(mdd)}</b></span>
  </div>"""


def _etf_card(p: dict) -> str:
    k = p["kpi"]
    return f"""
<a class="card" href="{p['key']}.html">
  <div class="card-h"><h3>{p['name']}</h3><span class="tag">{p['badge']}</span></div>
  <div class="desc">{p['desc']}</div>
  {_spark_svg(p['nav']['strat'])}
  {_kpis(k['ann'], k['sharpe'], k['mdd'])}
</a>"""


def _alpha_card(entry) -> str:
    """股票线卡片：读 ``reports/yq_core`` 的定版产物；缺文件时数字显示 -。"""
    rep_p = YQCORE / "report_native.json"
    nav_p = YQCORE / "nav_native.csv"
    tier = entry.key
    badge, desc = ALPHA_META.get(tier, ("股票线", ""))
    kpi, spark = {}, ""
    if rep_p.exists():
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
        t = rep.get("tiers", {}).get(tier, {})
        kpi = {"ann": t.get("ann"), "sharpe": t.get("sharpe"), "mdd": t.get("max_dd")}
    if nav_p.exists():
        try:
            s = pd.read_csv(nav_p, index_col=0, parse_dates=True)[tier].dropna()
            spark = _spark_svg([_r(v) for v in (s / s.iloc[0]).to_numpy()])
        except Exception:                                   # noqa: BLE001
            spark = ""
    return f"""
<a class="card" href="{entry.key}.html">
  <div class="card-h"><h3>{entry.label}</h3><span class="tag">{badge}</span></div>
  <div class="desc">{desc}</div>
  {spark}
  {_kpis(kpi.get('ann'), kpi.get('sharpe'), kpi.get('mdd'))}
</a>"""


def _event_card(p: dict) -> str:
    k = p["kpi"]
    return f"""
<a class="card" href="{p['key']}.html">
  <div class="card-h"><h3>{p['name']}</h3><span class="tag">{p['badge']}</span></div>
  <div class="desc">{p['desc']}</div>
  {_spark_svg(p['nav']['strat'])}
  {_kpis(k['ann'], k['sharpe'], k['mdd'])}
</a>"""


#: 首页分组（顺序即展示顺序）。组名是页面文案，不放进 catalog。
HOME_GROUPS = (
    ("alpha", "横截面策略"),
    ("event", "事件驱动策略"),
    ("etf", "ETF 策略"),
)


def render_home(payloads: dict[str, dict]) -> str:
    by_kind = {"etf": [], "alpha": [], "event": []}
    for e in cards():
        if e.kind == "etf" and e.key in payloads:
            by_kind["etf"].append(_etf_card(payloads[e.key]))
        elif e.kind == "alpha":
            by_kind["alpha"].append(_alpha_card(e))
        elif e.kind == "event" and e.key in payloads:
            by_kind["event"].append(_event_card(payloads[e.key]))
    sections = "".join(
        f'<h2 class="grp">{label}</h2><div class="cards">{"".join(by_kind[kind])}</div>'
        for kind, label in HOME_GROUPS if by_kind[kind])
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
{_nav("home", gen)}
<div class="wrap">
  <div class="hero">
    <h1>策略总览</h1>
    <div class="sub">点击卡片查看历史回测。</div>
  </div>
  {sections}
</div>
"""


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="策略首页（历史回测）+ 模拟持仓页")
    all_keys = ETF_KEYS + ALPHA_KEYS + ("inc_pure", "live")
    ap.add_argument("--only", default=None,
                    help="只生成指定页（逗号分隔，可选 %s）" % ",".join(all_keys))
    ap.add_argument("--out-dir", default=None, help="输出目录（缺省 reports/strategies）")
    ap.add_argument("--open", action="store_true", help="生成后打开首页")
    args = ap.parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else OUT
    out.mkdir(parents=True, exist_ok=True)
    keys = list(all_keys) if not args.only else \
        [s.strip() for s in args.only.split(",") if s.strip()]
    unknown = [k for k in keys if k not in all_keys]
    if unknown:
        raise SystemExit(f"未知策略键 {unknown}；可选 {list(all_keys)}")

    env = None
    payloads = {}
    live_payloads = {}
    for key in keys:
        if key == "live":
            continue                       # 模拟持仓页在循环后统一出（首页 + 详情）
        if key in ALPHA_KEYS:
            p = build_alpha_payload(key)
            f = out / f"{key}.html"
            f.write_text(page(
                title=f"{CATALOG[key].label} · 历史回测", header="",
                body=render_alpha_detail(p),
                extra_css=CHROME_CSS + yd._EXTRA_CSS + _ALPHA_CSS,
                head_extra=ECHARTS_TAG,
                scripts=yd.render_backtest_script(p)), encoding="utf-8")
            c = p["cur"]
            print(f"[回测] {f}  {p['window']}  α {c['alpha'] * 100:+.2f}%  "
                  f"SR {c['sharpe_geo']:.3f}  MDD {c['mdd']['mdd'] * 100:.2f}%")
            continue
        if key == "inc_pure":
            p = build_event_payload()
            body, css, scripts = render_event_detail(p), CHROME_CSS + _CSS, ""
        else:
            env = env if env is not None else load_env()
            p = build_etf_payload(env, key)
            body, css, scripts = render_detail(p), CHROME_CSS + _CSS, ""
        payloads[key] = p
        f = out / f"{key}.html"
        f.write_text(page(title=f"{p['name']} · 历史回测", header="", body=body,
                          css=css, scripts=scripts), encoding="utf-8")
        k = p["kpi"]
        if key == "inc_pure":
            print(f"[详情] {f}  {p['start']}~{p['end']}  年化 {k['ann'] * 100:+.2f}%  "
                  f"超额 {k['ex'] * 100:+.2f}%  t {k['t']:.2f}  MDD {k['mdd'] * 100:.2f}%  "
                  f"在持 {len(p['hold']['rows'])} 只 · 事件 {p['n_events']} 条")
        else:
            print(f"[详情] {f}  {p['start']}~{p['end']}  年化 {k['ann'] * 100:+.2f}%  "
                  f"SR {k['sharpe']:.2f}  MDD {k['mdd'] * 100:.2f}%  "
                  f"持仓 {len(p['hold']['rows'])} 行 · 调仓 {k['n_rb']} 次")

    if "live" in keys:
        for key in LIVE_KEYS:
            try:
                lp = build_live_payload(key)
            except SystemExit as exc:      # 没有持仓 CSV 就不出这一张，不编造
                print(f"[模拟持仓] 跳过 {key}：{exc}")
                continue
            f = out / f"live_{key}.html"
            f.write_text(page(
                title=f"{CATALOG[key].label} · 模拟持仓", header="",
                body=render_live_detail(lp),
                extra_css=CHROME_CSS + yd._EXTRA_CSS + _ALPHA_CSS,
                head_extra=ECHARTS_TAG,
                scripts=yd.render_live_script(lp)), encoding="utf-8")
            live_payloads[key] = lp
            t = lp["hold"]["totals"]
            print(f"[模拟持仓] {f}  信号日 {lp['hold']['signal_date']} · "
                  f"估值 {lp['hold']['latest_date']} · 实持 {len(lp['hold']['rows'])} 只 · "
                  f"期间 {t['ret'] * 100:+.2f}% · "
                  f"自检 {'通过' if lp['check']['ok'] else '未通过'}")
        home_live = out / "live.html"
        home_live.write_text(page(title="模拟持仓", header="",
                                  body=render_live_home(live_payloads),
                                  css=CHROME_CSS + _CSS), encoding="utf-8")
        print(f"[模拟持仓] {home_live}")

    if not args.only:
        home = out / "index.html"
        home.write_text(page(title="策略总览", header="", body=render_home(payloads),
                             css=CHROME_CSS + _CSS), encoding="utf-8")
        print(f"[首页] {home}")
        if args.open:
            import webbrowser
            webbrowser.open(home.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
