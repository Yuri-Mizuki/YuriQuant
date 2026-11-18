"""事件族「决策触及度 + 上帝视角天花板」一次性诊断（2026-11-15）
=====================================================================

为什么写这个脚本（以及它不做什么）
----------------------------------
`rating_reach_probe` 在评级上得到一条比「IC 太小」更硬的判据：

> **不是信号不够强，是没有决策可以被它改动。**

那条结论只落在评级上。而项目里其余事件族（解禁 / 减持 / 大宗 / 质押 /
股东户数 / 预告为负）全部只被当成**截面选股信号**测过——altf 三族吃 0/36 席位、
事件族 31 因子 mean|IC| 0.0068、PEAD 组合层 +0.06pp。它们的失败理由因此在
证据上是含糊的：既可能是「信号弱」，也可能是「K15 里挑不出几只可改」。

本脚本把 `rating_reach_probe` 的四问原样平移到六个事件族，**一次性判生死**：

A. **触及度 reach**：K15 的 15 个仓位里，被该事件标到的有几只？
B. **组合内区分度**：被标到的持仓，次月超额真的更差吗（不换股、不加权）？
C. **上帝视角天花板**：假设 100% 知道「下月会出事」，把这些股票从 K15 剔掉，
   月均超额最多提多少 pp？——任何真实信号都不可能超过它。
D. **硬否决（PIT 合法）**：只用当时可见的事件做剔除，实际能提多少 pp？

C 的意义：天花板 ≈ 0 ⇒ 这条线在**结构上**关闭，与信号强弱无关，不必再花预算。
D 的意义：这是「排除型」信号的**自然用法**；此前六族全部按「软加权选入」测，
而 K15 每期只有 15 个位置，软加权几乎改不动排名（评级那次的教训）。

本脚本**不产出因子、不进因子库、不写生产链**。它只回答「值不值得立项」。

口径（与主实验对齐）
--------------------
K15 + buffer30 执行、信号 = gbdt_mid13（h1:h5:h10:h20 = 2:1:1:2 截面秩加权）、
基准 = 当期可交易池等权、T+1 开盘进出、单边成本不在此计（诊断只比 Δ，不比绝对）。

PIT 纪律
--------
- 事件「可见日」= 公告日/交易日映射到**当天或之后的第一个交易日**（含当日收盘后
  可知的事件，实际下单在 T+1 开盘，故不构成未来函数）。
- 解禁例外：`list_date` 自公告起即为已知的**未来**日期，允许前视到事件日之前，
  这正是「解禁前减仓」的合法依据；因此它同时出现在 `fwd`（未来坏事）口径里。
- 大宗折价的成交价是**收盘后公告**，可见日 = 成交日 + 1 交易日。

用法
----
    python -m scripts.evaluation.event_reach_probe              # 全量
    python -m scripts.evaluation.event_reach_probe --look 63    # 观察窗（自然日）
    python -m scripts.evaluation.event_reach_probe --only unlock,reduce
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"
PRED_DIR = ROOT / "reports" / "alla_rolling_ortho920fundind" / "pred"
OUT = ROOT / "reports" / "event_reach_probe"
PQ = Path("D:/data/parquet")

K, BUF = 15, 30
H_W = {1: 2.0, 5: 1.0, 10: 1.0, 20: 2.0}   # gbdt_mid13 权重
DEFAULT_LOOK = 63                           # 可见窗：一个季度（自然日）


# ------------------------------------------------------------------ 基础数据
def _dt(s: pd.Series) -> pd.Series:
    """项目内事件表日期有三种编码，统一到 Timestamp：

    - 微秒 epoch 整数（``1453334400000000``，profit_notice / equity_restricted 等）
    - ``YYYYMMDD`` 整数（``20100104``，alt_cninfo_holder 等 akshare 系）
    - 字符串 / datetime
    """
    if pd.api.types.is_integer_dtype(s):
        v = s.astype("int64")
        med = float(v.dropna().median()) if v.notna().any() else 0.0
        if 1.9e7 < med < 2.2e7:            # YYYYMMDD
            return pd.to_datetime(v.astype(str), format="%Y%m%d", errors="coerce")
        if med > 1e14:                     # 微秒 epoch
            return pd.to_datetime(v, unit="us", errors="coerce")
        if med > 1e11:                     # 毫秒 epoch
            return pd.to_datetime(v, unit="ms", errors="coerce")
        return pd.to_datetime(v, unit="s", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _sym(code: pd.Series) -> pd.Series:
    """6 位代码 → ``000001.SZ``；已带后缀的原样返回。"""
    c = code.astype(str).str.strip()
    has = c.str.contains(r"\.", regex=True)
    six = c.str.zfill(6)
    p = six.str[0]
    suf = np.select(
        [p.eq("6"), p.isin(["0", "3"]), p.isin(["4", "8", "9"])],
        [".SH", ".SZ", ".BJ"], default="",
    )
    return pd.Series(np.where(has, c, six + suf), index=c.index)


def load_base() -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close = pd.read_parquet(BASE_DIR / "close_raw.parquet")
    openp = pd.read_parquet(BASE_DIR / "open_adj.parquet")
    mask = pd.read_parquet(BASE_DIR / "tradable_mask.parquet")
    return pd.DatetimeIndex(close.index), close, openp, mask


def load_signal(mask: pd.DataFrame) -> pd.DataFrame:
    """gbdt_mid13：与 focus / 生产口径一致。"""
    num = val = None
    for h, w in H_W.items():
        p = pd.read_parquet(PRED_DIR / f"gbdt__h{h}.parquet")
        v = p.notna() & mask.reindex(index=p.index, columns=p.columns).fillna(False).astype(bool)
        r = p.where(v).rank(axis=1, pct=True)
        num = r * w if num is None else num + r * w
        val = v if val is None else (val & v)
    return (num / sum(H_W.values())).where(val)


def month_pairs(cal: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.Series(cal, index=cal)
    rb = list(s.groupby([cal.year, cal.month]).last().values)
    return [(a, b) for a, b in zip(rb[:-1], rb[1:])]


def nw_t(x: np.ndarray, lag: int = 3) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    e = x - x.mean()
    S = (e * e).sum() / n
    for l in range(1, lag + 1):
        S += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).sum() / n
    se = np.sqrt(S / n)
    m = float(x.mean())
    return (m, float(m / se)) if se > 0 else (m, np.nan)


# ------------------------------------------------------------------ 持仓路径
def holdings_path(sig: pd.DataFrame, pairs, cal, openp, mask,
                  exclude: dict | None = None, tag: str = "base") -> pd.DataFrame:
    """复现 K15 + buffer30，返回 long 表 [t, t2, code, ret, uni, ex]。

    ``exclude``: {t: set(code)} —— 当期禁止入选的标的（硬否决口径）。
    给出时，否决在**选股阶段**生效并从候选池补位（K 仍为 15），
    这与「只在最终 15 只里删掉几个」是两回事：后者会顺带降低集中度，
    把「换股能力」和「降仓效应」混在一起。
    """
    rows, hold = [], None
    for t, t2 in pairs:
        if t not in sig.index:
            continue
        i, j = cal.get_loc(t), cal.get_loc(t2)
        if j + 1 >= len(cal):
            continue
        r = (openp.loc[cal[j + 1]] / openp.loc[cal[i + 1]] - 1).replace([np.inf, -np.inf], np.nan)
        s = sig.loc[t]
        ok = s.notna() & r.notna() & mask.loc[t].reindex(s.index).fillna(False).astype(bool)
        if ok.sum() < K:
            continue
        s2 = s.where(ok)
        ban = exclude.get(t, frozenset()) if exclude else frozenset()
        if ban:
            s2 = s2.drop(index=[c for c in ban if c in s2.index], errors="ignore")
        if s2.notna().sum() < K:
            continue
        uni = float(r[ok].mean())
        rank = s2.rank(ascending=False)
        keep = [c for c in (hold or []) if rank.get(c, 1e9) <= BUF]
        add = [c for c in s2.nlargest(K + BUF).index if c not in keep][: K - len(keep)]
        pk = keep + add
        hold = pk
        for c in pk:
            rows.append((t, t2, c, float(r[c]), uni, (float(r[c]) - uni) * 100))
    return pd.DataFrame(rows, columns=["t", "t2", "code", "ret", "uni", "ex"])


# ------------------------------------------------------------------ 事件族
def _avail(cal: pd.DatetimeIndex, d: pd.Series) -> pd.Series:
    """日期 → 日历上「当天或之后的第一个交易日」。"""
    pos = cal.searchsorted(pd.DatetimeIndex(d).values, side="left")
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    ok = pos < len(cal)
    out.loc[ok] = cal[pos[ok]]
    return out


def family_unlock(cal, close):
    """解禁：share_ratio ≥ 1%（占总股本）。avail = 解禁日本身（公告即已知）。"""
    df = pd.read_parquet(PQ / "equity_restricted.parquet",
                         columns=["code", "list_date", "share_ratio"])
    df["code"] = _sym(df["code"])
    df = df[pd.to_numeric(df["share_ratio"], errors="coerce") >= 1.0]
    d = _dt(df["list_date"])
    df = df.assign(avail=_avail(cal, d)).dropna(subset=["avail"])
    return df[["code", "avail"]]


def family_reduce(cal, close):
    """减持：董监高/股东公告减持，占总股本 ≥ 0.1%。"""
    df = pd.read_parquet(PQ / "alt_cninfo_holder.parquet",
                         columns=["code", "ann_date", "direction", "chg_ratio"])
    df = df[df["direction"].astype(str).str.contains("减持")]
    df = df[pd.to_numeric(df["chg_ratio"], errors="coerce").abs() >= 0.1]
    df["code"] = _sym(df["code"])
    df = df.assign(avail=_avail(cal, _dt(df["ann_date"]))).dropna(subset=["avail"])
    return df[["code", "avail"]]


def family_block(cal, close):
    """大宗折价：成交价 / 当日收盘 − 1 ≤ −5%，可见日 = 成交日 + 1 交易日。"""
    df = pd.read_parquet(PQ / "block_trading.parquet",
                         columns=["trade_date", "code", "b_share_price"])
    df["code"] = _sym(df["code"])
    d = _dt(df["trade_date"])
    idx = close.index.get_indexer(pd.DatetimeIndex(d))
    col = close.columns.get_indexer(df["code"])
    px = np.full(len(df), np.nan)
    ok = (idx >= 0) & (col >= 0)
    px[ok] = close.values[idx[ok], col[ok]]
    disc = df["b_share_price"].to_numpy() / px - 1.0
    df = df.assign(disc=disc)
    df = df[df["disc"] <= -0.05].copy()
    d = d.loc[df.index]
    nxt = cal.searchsorted(pd.DatetimeIndex(d).values, side="right")
    keep = nxt < len(cal)
    df = df.loc[keep].assign(avail=cal[nxt[keep]])
    return df[["code", "avail"]]


def family_pledge(cal, close):
    """质押：新增质押 ≥ 1% 总股本（冻结/质押公告日可见）。"""
    df = pd.read_parquet(PQ / "equity_pledge_freeze.parquet",
                         columns=["code", "ann_date", "fro_shr_to_total_ratio"])
    df = df[pd.to_numeric(df["fro_shr_to_total_ratio"], errors="coerce") >= 1.0]
    df["code"] = _sym(df["code"])
    df = df.assign(avail=_avail(cal, _dt(df["ann_date"]))).dropna(subset=["avail"])
    return df[["code", "avail"]]


def family_holdernum(cal, close):
    """股东户数环比 +10% 以上 ⇒ 筹码分散（偏空）。"""
    df = pd.read_parquet(PQ / "holder_num.parquet",
                         columns=["code", "ann_date", "holder_end_date", "holder_num"])
    df["code"] = _sym(df["code"])
    df = df.dropna(subset=["holder_num"])
    df = df.sort_values(["code", "holder_end_date"])
    df["chg"] = df.groupby("code")["holder_num"].pct_change()
    df = df[df["chg"] >= 0.10]
    df = df.assign(avail=_avail(cal, _dt(df["ann_date"]))).dropna(subset=["avail"])
    return df[["code", "avail"]]


def family_badfc(cal, close):
    """业绩预告为负：变动上限 < 0（确定同比下降），事件日 = 首次公告日。"""
    df = pd.read_parquet(PQ / "profit_notice.parquet",
                         columns=["code", "ann_date", "first_ann_date", "p_change_max"])
    df = df[pd.to_numeric(df["p_change_max"], errors="coerce") < 0]
    df["code"] = _sym(df["code"])
    d = _dt(df["first_ann_date"])
    d = d.where(d.notna(), _dt(df["ann_date"]))
    df = df.assign(avail=_avail(cal, d)).dropna(subset=["avail"])
    return df[["code", "avail"]]


FAMILIES = {
    "unlock": family_unlock,
    "reduce": family_reduce,
    "block": family_block,
    "pledge": family_pledge,
    "holdernum": family_holdernum,
    "badfc": family_badfc,
}


# ------------------------------------------------------------------ 触及度统计
def _index_events(ev: pd.DataFrame) -> dict[str, np.ndarray]:
    d = {}
    for code, sub in ev.groupby("code", sort=False):
        d[code] = np.sort(sub["avail"].values)
    return d


def _hit(ev_idx: dict[str, np.ndarray], code: str, lo, hi) -> bool:
    a = ev_idx.get(code)
    if a is None:
        return False
    i = np.searchsorted(a, np.datetime64(lo), side="right")
    j = np.searchsorted(a, np.datetime64(hi), side="right")
    return j > i


def build_exclude(ev: pd.DataFrame, pairs, sig: pd.DataFrame,
                  topn: int = 300, look: int = DEFAULT_LOOK) -> dict:
    """每个调仓日的否决清单：候选池（信号前 topn）里「可见窗内被事件标到」的标的。"""
    ev_idx = _index_events(ev)
    out = {}
    for t, _ in pairs:
        if t not in sig.index:
            continue
        s = sig.loc[t].dropna()
        cand = s.nlargest(topn).index
        lo, hi = t - pd.Timedelta(days=look), t
        out[t] = frozenset(c for c in cand if _hit(ev_idx, c, lo, hi))
    return out


def random_exclude(pairs, sig, n_by_t: dict, topn: int = 300, seed: int = 0) -> dict:
    """安慰剂：每期随机否决**同样只数**的候选股。

    rerank 的 Δ 里混着两个来源：① 剔除坏票 ② 换进来的新票。
    ② 与事件无关，纯粹由「换股」产生，量级可能远大于①。
    因此必须用它做零假设——真实否决若跑不赢同规模的随机否决，就只是换股噪声。
    """
    rng = np.random.default_rng(seed)
    out = {}
    for t, _ in pairs:
        if t not in sig.index:
            continue
        cand = sig.loc[t].dropna().nlargest(topn).index.to_numpy()
        k = min(int(n_by_t.get(t, 0)), len(cand))
        out[t] = frozenset(rng.choice(cand, size=k, replace=False)) if k else frozenset()
    return out


def _monthly_excess(h: pd.DataFrame) -> pd.Series:
    return h.groupby("t")["ex"].mean()


def probe_one(name: str, ev: pd.DataFrame, hold: pd.DataFrame, look: int) -> dict:
    ev_idx = _index_events(ev)
    vis, fwd = [], []
    for t, t2, code in hold[["t", "t2", "code"]].itertuples(index=False):
        vis.append(_hit(ev_idx, code, t - pd.Timedelta(days=look), t))
        fwd.append(_hit(ev_idx, code, t, t2))
    hold = hold.assign(hit_vis=vis, hit_fwd=fwd)

    g = hold.groupby("t")
    base = g["ex"].mean()

    def delta(flag: str) -> tuple[float, float, float]:
        """剔除 flag=True 的持仓后的月均超额 Δ（pp）与其 NW t。"""
        d, nd = [], []
        for t, sub in hold.groupby("t"):
            keep = sub[~sub[flag]]
            nd.append(int(sub[flag].sum()))
            if len(keep) >= 1:
                d.append(keep["ex"].mean() - sub["ex"].mean())
            else:
                d.append(np.nan)
        m, t_ = nw_t(np.asarray(d))
        return float(np.nanmean(nd)), m, t_

    # B：组合内区分度
    a = hold.loc[hold["hit_vis"], "ex"]
    b = hold.loc[~hold["hit_vis"], "ex"]
    spread = (a.mean() - b.mean()) if len(a) and len(b) else np.nan
    se = np.sqrt(a.var(ddof=1) / max(len(a), 1) + b.var(ddof=1) / max(len(b), 1)) if len(a) > 1 and len(b) > 1 else np.nan
    t_spread = spread / se if se and se > 0 else np.nan

    n_fwd, d_oracle, t_oracle = delta("hit_fwd")
    n_vis, d_veto, t_veto = delta("hit_vis")

    return dict(
        family=name,
        n_events=int(len(ev)),
        p_vis=float(hold["hit_vis"].mean()),
        p_fwd=float(hold["hit_fwd"].mean()),
        ex_hit=float(a.mean()) if len(a) else np.nan,
        ex_no=float(b.mean()) if len(b) else np.nan,
        spread_pp=float(spread),
        spread_t=float(t_spread) if np.isfinite(t_spread) else np.nan,
        n_drop_fwd=n_fwd, oracle_pp=d_oracle, oracle_t=t_oracle,
        n_drop_vis=n_vis, veto_pp=d_veto, veto_t=t_veto,
        base_pp=float(base.mean()),
        n_month=int(base.size),
    )


def verdict(r: dict) -> str:
    """判定顺序：先看能不能改（rerank），再看改得对不对（比随机否决强多少）。"""
    rr = r.get("rerank_pp", np.nan)
    if not np.isfinite(rr):
        return "数据不足"
    edge = r.get("edge_over_plc", np.nan)
    plp = r.get("plc_p", np.nan)
    tt = r.get("rerank_t", np.nan)
    if np.isfinite(edge) and (edge < 0.05 or (np.isfinite(plp) and plp > 0.20)):
        return "换股噪声（跑不赢随机否决）"
    if np.isfinite(edge) and edge >= 0.10 and np.isfinite(tt) and tt >= 1.5:
        return "候选（需独立样本复验）"
    if np.isfinite(edge) and edge >= 0.05 and np.isfinite(plp) and plp <= 0.05:
        return "边缘候选（仅跑赢安慰剂）"
    if not np.isfinite(r.get("oracle_pp", np.nan)) or abs(r["oracle_pp"]) < 0.10:
        return "结构性关闭（天花板≈0）"
    return "天花板存在但实施不足"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", type=int, default=DEFAULT_LOOK, help="可见窗（自然日）")
    ap.add_argument("--placebo", type=int, default=20, help="安慰剂抽样次数（0=关闭）")
    ap.add_argument("--only", default="", help="逗号分隔的事件族子集")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cal, close, openp, mask = load_base()
    sig = load_signal(mask)
    pairs = month_pairs(cal)
    hold = holdings_path(sig, pairs, cal, openp, mask)
    base_ex = _monthly_excess(hold)
    print(f"[base] 月数={len(pairs)} 持仓记录={len(hold)} 平均仓位={len(hold)/hold['t'].nunique():.1f} "
          f"月均超额={base_ex.mean():.3f}pp")
    hold.to_parquet(OUT / "holdings.parquet")

    series: dict[str, pd.Series] = {}

    def evaluate(name: str, ev: pd.DataFrame) -> dict:
        r = probe_one(name, ev, hold, args.look)
        try:
            ban = build_exclude(ev, pairs, sig, look=args.look)
            h2 = holdings_path(sig, pairs, cal, openp, mask, exclude=ban, tag=name)
            d = (_monthly_excess(h2) - base_ex).dropna()
            r["rerank_pp"], r["rerank_t"] = nw_t(d.values)
            series[name] = d
            a = h2.groupby("t")["code"].apply(set)
            b = hold.groupby("t")["code"].apply(set)
            r["rerank_turn"] = float((a != b).mean())
            r["n_ban"] = float(np.mean([len(v) for v in ban.values()]))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {name} rerank 失败: {type(exc).__name__}: {exc}")
            r["rerank_pp"] = r["rerank_t"] = r["rerank_turn"] = r["n_ban"] = np.nan

        # 安慰剂：同规模随机否决 M 次，看真实 Δ 在零分布里的位置
        if args.placebo > 0 and np.isfinite(r.get("n_ban", np.nan)):
            n_by_t = {t: len(v) for t, v in ban.items()}
            sims = []
            for s in range(args.placebo):
                rb = random_exclude(pairs, sig, n_by_t, seed=1000 + s)
                h3 = holdings_path(sig, pairs, cal, openp, mask, exclude=rb, tag=f"{name}_pl{s}")
                dd = (_monthly_excess(h3) - base_ex).dropna()
                sims.append(float(dd.mean()))
            sims = np.asarray(sims)
            r["plc_mean"] = float(np.mean(sims))
            r["plc_sd"] = float(np.std(sims, ddof=1)) if len(sims) > 1 else np.nan
            r["plc_p"] = float((sims >= r["rerank_pp"]).mean())
            r["edge_over_plc"] = float(r["rerank_pp"] - np.mean(sims))
        else:
            r["plc_mean"] = r["plc_sd"] = r["plc_p"] = r["edge_over_plc"] = np.nan

        r["verdict"] = verdict(r)
        print(f"[ok  ] {name:<10} n={r['n_events']:>7} p_vis={r['p_vis']:.1%} "
              f"spread={r['spread_pp']:+.2f}pp(t={r['spread_t']:+.2f}) "
              f"oracle={r['oracle_pp']:+.2f}pp "
              f"rerank={r['rerank_pp']:+.2f}pp(t={r['rerank_t']:+.2f}) "
              f"placebo={r['plc_mean']:+.2f}pp(p={r['plc_p']:.2f}) → {r['verdict']}")
        return r

    names = [n for n in (args.only.split(",") if args.only else FAMILIES) if n]
    rows, pool = [], []
    for name in names:
        fn = FAMILIES.get(name)
        if fn is None:
            print(f"[skip] 未知族 {name}")
            continue
        try:
            ev = fn(cal, close)
        except Exception as exc:  # noqa: BLE001 - 单族失败不阻断其余
            print(f"[err ] {name}: {type(exc).__name__}: {exc}")
            continue
        if ev.empty:
            print(f"[skip] {name}: 无事件")
            continue
        pool.append(ev)
        rows.append(evaluate(name, ev))

    # 合并族：六族「坏事件」的并集。这是排除型用法的真实形态——
    # 单族覆盖太窄，但若并集仍报不出天花板，则整条「事件当滤网」的路都可以收口。
    if len(pool) >= 2:
        ev_all = pd.concat(pool, ignore_index=True).drop_duplicates()
        rows.append(evaluate("union_bad", ev_all))

    if not rows:
        print("无结果")
        return
    out = pd.DataFrame(rows).sort_values("oracle_pp", ascending=False)
    out.to_csv(OUT / "reach.csv", index=False, encoding="utf-8-sig")

    if series:
        ser = pd.concat(series, axis=1)
        ser.index.name = "t"
        ser.to_csv(OUT / "rerank_series.csv", encoding="utf-8-sig")
        yr = ser.groupby(ser.index.year).mean()
        print("\n[分年] 硬否决 rerank Δ(pp/月，未扣安慰剂)")
        print(yr.round(3).to_string())

    print("\n" + out.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n[out] {OUT/'reach.csv'}")


if __name__ == "__main__":
    main()
