"""事件驱动策略 `inc_pure`：董监高主动增持 → T+1 开盘买入 → 持有 H 个交易日

为什么单独成模块（2026-11-19 统一到策略层）
-------------------------------------------
此前这套口径只在一次性探针里（`scripts/evaluation/event_driven_probe.py`，已删），
实盘出单器 `scripts/live/inc_pure_live.py` 又抄了一份事件定义。两处各写一遍，
改条件必漏一处。现在：

- **事件定义与回测**在本模块，是唯一真源；
- 实盘出单器从这里取 ``ACTIVE_REASONS`` / ``MIN_RATIO`` 等常量与代码归一化函数；
- 前端详情页（``scripts/reporting/strategy_home.py``）直接调 :func:`run` 取曲线与持仓。

它**不是** ``Strategy`` 子类：``strategy.base.Strategy`` 的输入是「单日因子截面 →
权重」，而事件驱动的输入是**事件表**，权重由「谁在持有窗内」决定；持有窗是
open-to-open 的固定日历段，与 ``backtest/`` 引擎的调仓-权重接口不同构。强行套
基类只会把两边的判据弄脏（同 ``strategy/catalog.py`` 的注释）。

口径（与 2026-11-15 的事件驱动报告逐条一致）
--------------------------------------------
| 环节 | 口径 |
|---|---|
| 事件 | ``direction=增持`` + ``reason ∈ ACTIVE_REASONS`` + ``relation=本人`` + 合计 ≥0.1% 总股本 |
| 聚合 | 同 (股票, 可见日) 多条记录合并，比例求和 |
| 可见日 | 公告日当天或其后第一个交易日；早于样本起点的公告**丢弃**（不能钳到首日） |
| 建仓 | 可见日的次一交易日**开盘**（T+1 集合竞价） |
| 卖出 | 建仓日 + H 个交易日的**开盘** |
| 重叠 | 同股在持有窗内再出事件 ⇒ 延长持有，不加仓 |
| 组合 | 逐日等权持有全部在窗股票；无事件日空仓，收益记 **0**（不是 NaN） |
| 基准 | 同市值分 5 组，取持仓股当日所在组的等权收益 |
| 成本 | 单边 13.5bp（往返 27bp），固定值、非时变 |

边界：回测允许无限资金、无限并发，不扣整手/停牌/ST；这些是实盘出单器的闸门，
两者口径不同，**不可把出单器的成交数当回测成交数**。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"

#: 持有交易日数（与实盘出单器一致；H=20 是唯一经过执行层检验的档）
HOLD = 20
#: 单边成本：往返 27bp = 佣金 2bp + 印花 5bp + 滑点 20bp，固定值非时变
COST_ONE_WAY = 0.00135
#: 主动买入的成交方式（剔除分红送转 / 股权激励实施）
ACTIVE_REASONS = ("竞价交易", "二级市场买卖", "大宗交易")
#: 合计增持占总股本下限（%）
MIN_RATIO = 0.1
#: 回测起点（面板起点；更早的公告在 ``avail_day`` 里置 NaT）
START = "2018-01-01"
#: 市值分组数
NQ = 5
ANNUAL = 250


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def cache_root() -> Path:
    return Path(str(Config.paths()["cache_root"]))


def normalize_code(code: pd.Series) -> pd.Series:
    """6 位代码 → ``000001.SZ``；已带后缀的原样返回。"""
    c = code.astype(str).str.strip()
    has = c.str.contains(r"\.", regex=True)
    six = c.str.zfill(6)
    p = six.str[0]
    suf = np.select([p.eq("6"), p.isin(["0", "3"]), p.isin(["4", "8", "9"])],
                    [".SH", ".SZ", ".BJ"], default="")
    return pd.Series(np.where(has, c, six + suf), index=c.index)


def avail_day(cal: pd.DatetimeIndex, d: pd.Series) -> pd.Series:
    """日期 → 日历上「当天或之后的第一个交易日」。

    ⚠️ 早于样本起点的事件必须置 NaT，不能顺延到首日：``searchsorted`` 对
    ``d < cal[0]`` 一律返回 0，若不拦，样本前的历史公告会被全部堆到首日建仓。
    """
    dv = pd.DatetimeIndex(pd.to_datetime(d, errors="coerce"))
    pos = cal.searchsorted(dv.values, side="left")
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    ok = (pos < len(cal)) & dv.notna() & (dv.values >= cal[0].to_datetime64())
    out.loc[ok] = cal[pos[ok]]
    return out


def load_base(base_dir: Path | None = None, start: str = START) -> dict:
    """装载回测面板（前复权价 / 开盘价 / 可交易掩码 / 全A等权基准 / 市值）。"""
    b = Path(base_dir) if base_dir else BASE_DIR
    close = pd.read_parquet(b / "close_adj.parquet").loc[start:]
    openp = pd.read_parquet(b / "open_adj.parquet").loc[start:]
    mask = pd.read_parquet(b / "tradable_mask.parquet").loc[start:]
    bench = pd.read_parquet(b / "bench_eqw.parquet")["ret"].loc[start:]
    mcap = pd.read_parquet(b / "market_cap.parquet").loc[start:]
    close = close.reindex(columns=openp.columns)
    mask = mask.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    mcap = mcap.reindex(index=close.index, columns=close.columns)
    return {"close": close, "open": openp, "mask": mask, "bench": bench,
            "mcap": mcap, "cal": pd.DatetimeIndex(close.index)}


# ---------------------------------------------------------------------------
# 事件
# ---------------------------------------------------------------------------
def inc_pure_events(cal: pd.DatetimeIndex, source: Path | None = None) -> pd.DataFrame:
    """`inc_pure` 事件：返回 ``DataFrame[code, avail, strength, name]``。

    `strength` = 同 (股票, 可见日) 的增持比例合计（%）。
    """
    p = (Path(source) if source else cache_root()) / "alt_cninfo_holder.parquet"
    df = pd.read_parquet(p, columns=["code", "sec_name", "ann_date", "direction",
                                     "chg_ratio", "reason", "relation"])
    df = df[df["direction"].astype(str).str.contains("增持")]
    df = df[df["reason"].astype(str).isin(ACTIVE_REASONS)]
    df = df[df["relation"].astype(str) == "本人"]
    df = df.assign(code=normalize_code(df["code"]),
                   r=pd.to_numeric(df["chg_ratio"], errors="coerce").abs(),
                   avail=avail_day(cal, pd.to_datetime(
                       df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")))
    df = df.dropna(subset=["avail", "r"])
    df = df[df["r"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["code", "avail", "strength", "name"])
    g = (df.groupby(["code", "avail"])
           .agg(strength=("r", "sum"), name=("sec_name", "first"))
           .reset_index())
    return g[g["strength"] >= MIN_RATIO].reset_index(drop=True)


def build_positions(ev: pd.DataFrame, cal: pd.DatetimeIndex, hold: int,
                    close: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """事件 → 持仓区间 ``[i0, i1]``（日历下标）。

    两道硬闸门（回测踩过的坑）：

    1. ``tradable_mask`` **不含上市日**判断，次新股上市前的 mask 可能为 True；
    2. 建仓日与整个持有窗的收盘价必须齐备（缺任一 ⇒ 该笔不成立，不是补 0）。

    同股在上一笔持有窗内再出事件 ⇒ 合并延长（不叠加仓位）。
    """
    out_cols = ["code", "i0", "i1", "t0", "strength", "n_ev"]
    if ev.empty:
        return pd.DataFrame(columns=out_cols)
    pos = cal.get_indexer(pd.DatetimeIndex(ev["avail"]))
    keep = (pos >= 0) & (pos + 1 + hold < len(cal))
    ev = ev.loc[keep].assign(i0=pos[keep] + 1, i1=pos[keep] + 1 + hold)
    ev = ev.sort_values(["code", "i0"])
    if ev.empty:
        return pd.DataFrame(columns=out_cols)

    cv, mv = close.values, mask.values
    col = {c: i for i, c in enumerate(close.columns)}
    ok_rows = []
    for code, i0, i1 in ev[["code", "i0", "i1"]].itertuples(index=False):
        j = col.get(code)
        if j is None:
            ok_rows.append(False)
            continue
        price_ok = bool(np.isfinite(cv[i0, j]) and np.isfinite(cv[i1, j])
                        and np.isfinite(cv[i0:i1 + 1, j]).all())
        ok_rows.append(price_ok and bool(mv[i0, j]))
    ev = ev.loc[np.array(ok_rows)]

    rows: list[dict] = []
    for code, sub in ev.groupby("code", sort=False):
        cur = None
        for i0, i1, strength in sub[["i0", "i1", "strength"]].itertuples(index=False):
            if cur is None or i0 > cur["i1"]:
                if cur is not None:
                    rows.append(cur)
                cur = {"code": code, "i0": int(i0), "i1": int(i1), "t0": int(i0) - 1,
                       "strength": float(strength), "n_ev": 1}
            else:                                   # 重叠 ⇒ 延长持有
                cur["i1"] = max(cur["i1"], int(i1))
                cur["strength"] += float(strength)
                cur["n_ev"] += 1
        if cur is not None:
            rows.append(cur)
    return pd.DataFrame(rows, columns=out_cols)


# ---------------------------------------------------------------------------
# 日频组合
# ---------------------------------------------------------------------------
def run_portfolio(pos: pd.DataFrame, env: dict) -> tuple[pd.Series, pd.Series]:
    """事件组合日收益：逐日等权持有全部在窗股票。

    phase 0 = 建仓日（开盘买 → 收盘）、1 = 持有（收盘→收盘）、2 = 卖出日（昨收 → 开盘卖）。

    ⚠️ 持有期**不做** mask 过滤：停牌持仓卖不掉，收益应记 0 而不是剔除；
    剔除等于假设「停牌时可以从组合里消失」，会系统性抬高收益（实测差 7pp/年）。
    """
    cv, ov = env["close"].values, env["open"].values
    cal = env["cal"]
    n = len(cal)
    col = {c: i for i, c in enumerate(env["close"].columns)}
    live: dict[int, list[tuple[int, int]]] = {}
    for code, i0, i1 in pos[["code", "i0", "i1"]].itertuples(index=False):
        j = col.get(code)
        if j is None:
            continue
        live.setdefault(i0, []).append((j, 0))
        for i in range(i0 + 1, i1):
            live.setdefault(i, []).append((j, 1))
        live.setdefault(i1, []).append((j, 2))

    ret = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=int)
    for i in range(1, n):
        arr = live.get(i)
        if not arr:
            continue
        j = np.array([a[0] for a in arr], dtype=int)
        ph = np.array([a[1] for a in arr], dtype=int)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cv[i, j] / cv[i - 1, j] - 1.0            # 持有
            r = np.where(ph == 0, cv[i, j] / ov[i, j] - 1.0, r)
            r = np.where(ph == 2, ov[i, j] / cv[i - 1, j] - 1.0, r)
        r = np.where(np.isfinite(r), r, np.nan)      # 0 价 / 缺价 ⇒ 不可得收益
        if np.isfinite(r).any():
            ret[i] = float(np.nanmean(r))
            cnt[i] = int(np.isfinite(r).sum())
    return pd.Series(ret, index=cal), pd.Series(cnt, index=cal)


def matched_groups(close: pd.DataFrame, mcap: pd.DataFrame, nq: int = NQ) -> pd.DataFrame:
    """每日按市值分 ``nq`` 组，返回各组等权日收益（宽表，index=日期）。"""
    q = mcap.where(mcap.notna()).rank(axis=1, pct=True)
    g = (q * nq).clip(upper=nq - 1e-9).apply(np.floor)
    # 显式 pad：停牌日沿用前收（0 收益），与归档报告的基准口径一致
    r1 = close.ffill().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    long = pd.DataFrame({"g": g.stack(), "r": r1.stack()}).dropna()
    if long.empty:
        return pd.DataFrame()
    d = long.index.get_level_values(0)
    out = long.groupby([d, "g"])["r"].mean().unstack()
    out.columns = [int(c) for c in out.columns]
    return out.reindex(columns=range(nq)).sort_index()


def bench_for_holdings(pos: pd.DataFrame, env: dict, grp: pd.DataFrame,
                       nq: int = NQ) -> pd.Series:
    """持仓股所在市值组的等权收益（分组用**前一日**市值，避免用当日信息）。"""
    if grp.empty:
        return pd.Series(np.nan, index=env["close"].index)
    n = len(env["close"])
    col = {c: i for i, c in enumerate(env["close"].columns)}
    live: dict[int, list[int]] = {}
    for code, i0, i1 in pos[["code", "i0", "i1"]].itertuples(index=False):
        j = col.get(code)
        if j is None:
            continue
        for i in range(i0, i1 + 1):
            live.setdefault(i, []).append(j)

    out = np.full(n, np.nan)
    qs = env["mcap"].where(env["mcap"].notna()).rank(axis=1, pct=True).values
    for i in range(1, n):
        js = live.get(i)
        d = env["close"].index[i]
        if not js or d not in grp.index:
            continue
        q = qs[i - 1, np.array(js, dtype=int)]
        ok = np.isfinite(q)
        if not ok.any():
            continue
        g = np.clip(np.floor(q[ok] * nq), 0, nq - 1).astype(int)
        v = grp.loc[d].reindex(g).values
        if np.isfinite(v).any():
            out[i] = float(np.nanmean(v))
    return pd.Series(out, index=env["close"].index)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
#: NW t 唯一真源在 :func:`stats.robust_stats.nw_t`（2026-11-20 删本地副本——第二份
#: 实现触发 ``test_criteria_have_single_definition_site``）。⚠️ **口径差异留痕**：
#: 旧本地版自协分母为 ``/n``，真源为 ``/(n-j)``，同序列 t 值偏差 ~1–2% ⇒ 事件线
#: 2026-11-20 前引用的 t（如 H=20 毛超额 3.23）为旧口径，**跨口径不可比、旧值
#: 不再引用**（同 2026-11-19 指标收口对 pit_intraday_probe 的处理）；
#: ``lag=5`` 显式传参保持「日频固定 5 滞后」的原设定。
from stats.robust_stats import nw_t  # noqa: E402


def summarize(ret: pd.Series, bench: pd.Series, pos: pd.DataFrame, cal: pd.DatetimeIndex,
              cnt: pd.Series) -> dict:
    """年化 / 超额 / Sharpe / 分年 / 换手 / 扣费。

    ⚠️ 空仓日必须记 0，不能 dropna：事件驱动是「有事件才持仓」，只对持仓日求均值
    再 ×250 等于假设没事件的日子也在赚钱，会系统性高估（实测差 6pp/年）。
    """
    r = ret.reindex(cal).fillna(0.0)
    b = bench.reindex(cal).fillna(0.0)
    first = ret.first_valid_index()
    if first is None or len(r.loc[first:]) < 250:
        return {}
    r, b = r.loc[first:], b.loc[first:]
    days = len(r)
    avg_h = float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 1.0
    # 年化换手（双边）：每日新建仓笔数 / 平均持仓笔数 × 2 × 250
    ann_turn = float(len(pos) / days / max(avg_h, 1.0) * 2 * ANNUAL)
    cost = ann_turn * COST_ONE_WAY
    ex = r - b
    ann_ex = float(ex.mean() * ANNUAL)
    eq = (1 + r).cumprod()
    eqb = (1 + b).cumprod()
    dd = (eq / eq.cummax() - 1.0)
    yr_s = (1 + r).groupby(r.index.year).prod() - 1.0
    yr_b = (1 + b).groupby(b.index.year).prod() - 1.0
    return {
        "days": days,
        "ann_ret": float(r.mean() * ANNUAL),
        "ann_ex": ann_ex,
        "ann_ex_net": ann_ex - cost,
        "t": nw_t(ex.values, lag=5),
        "sharpe": float(r.mean() / r.std() * np.sqrt(ANNUAL)) if r.std() > 0 else float("nan"),
        "maxdd": float(dd.min()),
        "turnover": ann_turn,
        "cost": cost,
        "yr_pos": f"{int((yr_s - yr_b).gt(0).sum())}/{len(yr_s)}",
        "avg_hold": avg_h,
        "n_pos": int(len(pos)),
        "eq": eq, "eqb": eqb, "dd": dd,
        "years": [int(y) for y in yr_s.index],
        "yr_s": yr_s, "yr_b": yr_b,
        "ex": ex,
    }


# ---------------------------------------------------------------------------
# 当前持仓（尚未走完持有窗）
# ---------------------------------------------------------------------------
def open_positions(ev: pd.DataFrame, env: dict, hold: int = HOLD) -> list[dict]:
    """截至最后一个交易日仍在持有窗内的仓位。

    ``build_positions`` 要求「整窗都在样本内」才成立（否则无法计完整收益），
    所以最近 ``hold`` 个交易日的事件会被它整体丢掉。展示「模拟持仓」必须单独
    按同一口径前推：建仓日 = 可见日次一交易日开盘，卖出日 = 建仓日 + H 个交易日。
    """
    cal, last = env["cal"], len(env["cal"]) - 1
    pos = cal.get_indexer(pd.DatetimeIndex(ev["avail"]))
    ok = (pos >= 0) & (pos + 1 <= last) & (pos + 1 + hold > last)
    sub = ev.loc[ok].assign(i0=pos[ok] + 1, i1=pos[ok] + 1 + hold)
    if sub.empty:
        return []
    col = {c: i for i, c in enumerate(env["close"].columns)}
    ov, cv, mv = env["open"].values, env["close"].values, env["mask"].values
    rows = []
    for code, g in sub.groupby("code", sort=False):
        j = col.get(code)
        if j is None:
            continue
        i0 = int(g["i0"].min())
        entry = float(ov[i0, j])
        last_px = float(cv[last, j])
        if not (np.isfinite(entry) and entry > 0 and np.isfinite(last_px) and mv[i0, j]):
            continue
        rows.append({
            "code": code,
            "name": g["name"].iloc[0] or "—",
            "trigger": cal[i0 - 1].strftime("%Y-%m-%d"),
            "buy": cal[i0].strftime("%Y-%m-%d"),
            "sell": None,                              # 卖出日在样本外，按剩余交易日数处理
            "days": int(hold),
            "held": int(last - i0 + 1),
            "left": int(i0 + hold - last),
            "strength": float(g["strength"].sum()),
            "n_ev": int(len(g)),
            "ret_gross": last_px / entry - 1.0,
            "ret_net": last_px / entry - 1.0 - 2 * COST_ONE_WAY,
            "open": True,
        })
    return sorted(rows, key=lambda r: r["trigger"], reverse=True)


# ---------------------------------------------------------------------------
# 一次跑完
# ---------------------------------------------------------------------------
def run(hold: int = HOLD, base_dir: Path | None = None,
        source: Path | None = None, start: str = START) -> dict:
    """跑完整回测，返回曲线 / 指标 / 当前持仓 / 逐笔（全部 JSON-ready 由调用方转换）。

    返回键：

    ``dates / ret / bench / count / nav / bench_nav / dd`` —— 逐日序列（nav 起点 = 1）
    ``kpi`` —— 年化、超额、扣费超额、NW t、Sharpe、MDD、换手、成本、平均持仓、笔数
    ``years / yr_strat / yr_bench`` —— 分年度收益
    ``open`` —— 截至最后一个交易日仍在持有窗内的持仓
    ``trades`` —— 已了结 / 在持的逐笔（含触发日、买卖日、收益）
    """
    env = load_base(base_dir, start)
    cal = env["cal"]
    ev = inc_pure_events(cal, source)
    pos = build_positions(ev, cal, hold, env["close"], env["mask"])
    ret, cnt = run_portfolio(pos, env)
    grp = matched_groups(env["close"], env["mcap"])
    bench = bench_for_holdings(pos, env, grp)
    s = summarize(ret, bench, pos, cal, cnt)
    if not s:
        raise RuntimeError("回测样本不足（有效交易日 < 250）")

    r = ret.reindex(cal).fillna(0.0).loc[s["eq"].index]
    b = bench.reindex(cal).fillna(0.0).loc[s["eq"].index]
    names = (ev.drop_duplicates("code").set_index("code")["name"].to_dict())

    last = len(cal) - 1
    ov = env["open"].values
    col = {c: i for i, c in enumerate(env["close"].columns)}
    trades = []
    for row in pos.sort_values("t0", ascending=False).itertuples(index=False):
        j = col.get(row.code)
        if j is None:
            continue
        buy, sell = int(row.i0), int(row.i1)
        entry = float(ov[buy, j])
        exit_px = float(ov[sell, j])
        if not (np.isfinite(entry) and entry > 0) or not np.isfinite(exit_px):
            continue
        gross = exit_px / entry - 1.0
        trades.append({
            "code": row.code,
            "name": names.get(row.code) or "—",
            "trigger": cal[int(row.t0)].strftime("%Y-%m-%d"),
            "buy": cal[buy].strftime("%Y-%m-%d"),
            "sell": cal[sell].strftime("%Y-%m-%d"),
            "days": int(row.i1 - row.i0),
            "strength": float(row.strength),
            "n_ev": int(row.n_ev),
            "ret_gross": gross,
            "ret_net": gross - 2 * COST_ONE_WAY,
            "open": False,
        })
    open_rows = open_positions(ev, env, hold)

    return {
        "hold": hold,
        "start": r.index[0].strftime("%Y-%m-%d"),
        "end": r.index[-1].strftime("%Y-%m-%d"),
        "dates": [d.strftime("%Y-%m-%d") for d in r.index],
        "ret": r.to_numpy(),
        "bench": b.to_numpy(),
        "count": cnt.reindex(r.index).fillna(0).astype(int).to_numpy(),
        "nav": s["eq"].to_numpy(),
        "bench_nav": s["eqb"].to_numpy(),
        "dd": s["dd"].reindex(r.index).fillna(0.0).to_numpy(),
        "kpi": {k: s[k] for k in ("days", "ann_ret", "ann_ex", "ann_ex_net", "t",
                                  "sharpe", "maxdd", "turnover", "cost", "yr_pos",
                                  "avg_hold", "n_pos")},
        "years": [str(y) for y in s["years"]],
        "yr_strat": s["yr_s"].to_numpy(),
        "yr_bench": s["yr_b"].to_numpy(),
        "n_events": int(len(ev)),
        "open": open_rows,
        "trades": trades,
        "base_bench": "market_cap_5group",
    }


__all__ = ["HOLD", "COST_ONE_WAY", "ACTIVE_REASONS", "MIN_RATIO", "START", "NQ",
           "cache_root", "normalize_code", "avail_day", "load_base", "inc_pure_events",
           "build_positions", "run_portfolio", "matched_groups", "bench_for_holdings",
           "nw_t", "summarize", "run"]
