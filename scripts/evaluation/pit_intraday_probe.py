"""PIT 动态池（421 只）上重算 im_* 因子并验证 1+2
================================================

背景：静态池的幸存者偏差
------------------------
此前全部日频/日内结论都在 ``min5_hs300.parquet`` 上做，而它只覆盖 **339 只**
——正好是**当前** HS300 名单。HS300 在 2022-01 ~ 2026-06 窗口内**曾属**的成分
共 **421 只**，缺的 82 只（19.5%）是被剔除的差股票（中位累计收益 **−51.2%**
vs 在库 +4.3%，p=5.7e−15）。所以旧结论带幸存者偏差。

已补齐分钟数据（``scripts/ingest/fetch_minute_gap.py``）并重建面板，现在
面板为 **421 码**。本脚本回答三件事：

**Q0 池审计**：421 池 vs 339 池，逐因子 IC 变化多少（是不是所有结论都要下修？）

**Q1（建议 1）复合方式修正**：上一轮真因不是「信息同质」（n_eff=22.14），
而是 **rank 平均稀释 Q1**——`im_mom_last_bar` 单因子分层完美单调
（Q1 +37.2% → Q10 −11.1%），但 10 因子 rank 平均后复合 top10 只有 **38.3%**
落在单因子 Q1 内，与单因子 top10 重合率仅 **16.2%**。改成「锚因子切 Q1 →
Q1 内二级排序」能否救回来？

**Q2（建议 2）lag 类因子**：现有 95 个全是**同日截面**的量价绝对量，
缺 T-1/T-2 的日内结构进 T 日。构造 lag 版（对面板做 ``shift(1)`` /
``shift(2)``）并测其 IC 是否与当日版互补（相关低 + IC 独立）。

口径
----
- PIT 池：``load_daily`` 已按 ``in_date/out_date`` 展开并 ``apply_membership_mask``
  把非在册期置 NaN ⇒ 每个交易日只用当时在册成分。
- 收益：日线 **T+1 open 买 → T+2 open 卖**（``buy=op.shift(-1)``，
  ``sell=op.shift(-(1+1))``），与 ``intraday_daily_probe`` 一致。
- 成本：27bp 往返 ⇒ 单边 13.5bp。
- IC：spearman，``factor[t]`` vs ``ret[t]``（ret 已是 t→t+1）。

用法::

    python -m scripts.evaluation.pit_intraday_probe --stage all
    python -m scripts.evaluation.pit_intraday_probe --stage audit     # 只做池审计
    python -m scripts.evaluation.pit_intraday_probe --stage lag       # 只做 lag 因子
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache import DataCache  # noqa: E402
from data.intraday import MinutePanelStore  # noqa: E402
from data.offline import OfflineDataSource  # noqa: E402
from data.universe import Universe  # noqa: E402
from factor.intraday_features import extract_all  # noqa: E402
from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("pit_intraday_probe")

OUT = ROOT / "reports" / "intraday_daily"
CACHE = Path("E:/data/parquet")
PERIOD = 5
ONE_WAY_BP = 13.5           # 27bp 往返 / 2
MIN_IC_DAYS = 100


# ----------------------------------------------------------------------
# 数据层
# ----------------------------------------------------------------------
def load_context(begin: int = 20220101, end: int = 20260630):
    """PIT 池上下文：分钟面板 + PIT 日线（已在册掩码）+ 收益。"""
    store = MinutePanelStore.open(PERIOD, "hs300")
    days = store.all_days()
    begin = max(begin, days[0])
    end = min(end, days[-1])

    from data.cache_helpers import load_daily
    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    codes, _cal, daily = load_daily(cache, uni, "000300.SH", begin, end)
    log.info("PIT 池 %d 只 / 面板 %d 只 / 交易日 %d", len(codes), len(store.codes),
             len(days))

    # T+1 open 买 → T+2 open 卖
    op = daily.reset_index().pivot(index="date", columns="code",
                                   values="open").sort_index()
    buy = op.shift(-1)
    sell = op.shift(-2)
    fwd = (sell / buy - 1.0).replace([np.inf, -np.inf], np.nan)
    return store, codes, daily, fwd


def build_features(store: MinutePanelStore, begin: int, end: int,
                   with_lag: bool = True) -> dict[str, pd.DataFrame]:
    """当日特征 + lag 版（T-1 / T-2 的日内结构进 T 日）。"""
    feats = extract_all(store, begin_date=begin, end_date=end)
    log.info("当日特征 %d 个 × %s", len(feats), next(iter(feats.values())).shape)
    if not with_lag:
        return feats

    # 🚨 lag 必须在**面板层**做 shift：extract_all 内部按 block 流式拼接，
    # 在 block 内 shift 会让每个 block 首日变 NaN（静默丢 128 天一次）。
    # 面板已经拼好，这里 shift 才是整条时间轴连续位移。
    lagged: dict[str, pd.DataFrame] = {}
    for name, p in feats.items():
        lagged[f"{name}_lag1"] = p.shift(1)
        lagged[f"{name}_lag2"] = p.shift(2)
    log.info("lag 特征 %d 个（lag1+lag2）", len(lagged))
    return {**feats, **lagged}


# ----------------------------------------------------------------------
# IC
# ----------------------------------------------------------------------
def daily_ic(sig: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 30) -> pd.Series:
    """逐日截面 spearman IC。"""
    idx = sig.index.intersection(fwd.index)
    cols = sig.columns.intersection(fwd.columns)
    s, r = sig.loc[idx, cols], fwd.loc[idx, cols]
    ic = {}
    for dt in idx:
        sv, rv = s.loc[dt], r.loc[dt]
        m = sv.notna() & rv.notna()
        if int(m.sum()) < min_names:
            continue
        ic[dt] = sv[m].corr(rv[m], method="spearman")
    return pd.Series(ic).dropna()


def ic_stats(ic: pd.Series) -> dict:
    if len(ic) < MIN_IC_DAYS:
        return {}
    m, sd = float(ic.mean()), float(ic.std(ddof=1))
    if sd == 0:
        return {}
    t = m / sd * np.sqrt(len(ic))
    return {"ic": m, "ic_std": sd, "t": t, "n": len(ic),
            "win": float((ic > 0).mean())}


# ----------------------------------------------------------------------
# 组合回测
# ----------------------------------------------------------------------
def run(sig: pd.DataFrame, fwd: pd.DataFrame, top: int = 10,
        buffer_mult: float = 2.0, cost_bp: float = ONE_WAY_BP,
        min_names: int = 30) -> dict:
    """日频等权 TopN + buffer。🚨 列对齐必须先做（见 daily_survivorship_probe）。"""
    sig = sig.reindex(columns=fwd.columns)
    idx = sig.index.intersection(fwd.index)
    s, r = sig.loc[idx], fwd.loc[idx]

    prev: set[str] = set()
    rets, turns = [], []
    for dt in idx:
        sv = s.loc[dt].dropna()
        if len(sv) < min_names:
            rets.append(np.nan)
            turns.append(0.0)
            continue
        rank = sv.rank(ascending=False, method="first")
        head = rank.sort_values()
        keep = head[head <= top * buffer_mult].index
        held = [c for c in prev if c in keep]
        new = [c for c in head.index if c not in held]
        picked = (held + new)[:top]
        if not picked:
            rets.append(np.nan)
            turns.append(0.0)
            continue
        w = pd.Series(1.0 / len(picked), index=picked)
        wp = (pd.Series(1.0 / len(prev), index=sorted(prev))
              if prev else pd.Series(dtype=float))
        allc = w.index.union(wp.index)
        dw = w.reindex(allc, fill_value=0) - wp.reindex(allc, fill_value=0)
        turns.append(float(dw.abs().sum() / 2))
        rr = r.loc[dt, picked].dropna()
        rets.append(float(rr.mean()) if len(rr) else np.nan)
        prev = set(picked)
    gross = pd.Series(rets, index=idx)
    net = gross - pd.Series(turns, index=idx) * cost_bp / 1e4
    return {"gross": gross.dropna(), "net": net.dropna(),
            "turn": pd.Series(turns, index=idx)}


def stats(x: pd.Series, ppy: int = 252) -> dict:
    x = x.dropna()
    if len(x) < 30:
        return {}
    nav = (1 + x).cumprod()
    cagr = float(nav.iloc[-1] ** (ppy / len(x)) - 1)
    dd = float((nav / nav.cummax() - 1).min())
    return {"ann": cagr, "sr": float(x.mean() / x.std(ddof=1) * np.sqrt(ppy)),
            "mdd": dd, "n": len(x)}


# ----------------------------------------------------------------------
# 建议 1：两级聚合（锚因子切 Q1 → Q1 内二级排序）
# ----------------------------------------------------------------------
def composite_flat(names: list[str], feats: dict[str, pd.DataFrame],
                   ics: dict[str, float]) -> pd.DataFrame:
    """旧口径：全截面定向 rank 平均（会稀释 Q1）。"""
    panels = [orient(feats[n], ics[n]) for n in names]
    idx = panels[0].index
    cols = panels[0].columns
    for p in panels[1:]:
        idx = idx.intersection(p.index)
        cols = cols.intersection(p.columns)
    num = cnt = None
    for p in panels:
        r = p.loc[idx, cols].rank(axis=1, pct=True)
        num = r if num is None else num.add(r, fill_value=0)
        c = r.notna().astype(np.int16)
        cnt = c if cnt is None else cnt.add(c)
    cnt = cnt.where(cnt >= 2)
    return (num / cnt).where(cnt.notna())


def composite_2stage(anchor: str, pool: list[str],
                     feats: dict[str, pd.DataFrame], ics: dict[str, float],
                     q: float = 0.1) -> pd.DataFrame:
    """新口径：锚因子切 Q1（保留尾部），Q1 内用其余因子二级排序。

    q=1.0 退化回全截面平均；q=0.1 对应「top10% 候选池」（421 只约 42 只）。
    """
    a = orient(feats[anchor], ics[anchor])
    others = [orient(feats[n], ics[n]) for n in pool if n != anchor]
    idx = a.index
    cols = a.columns
    for p in others:
        idx = idx.intersection(p.index)
        cols = cols.intersection(p.columns)

    a = a.loc[idx, cols]
    # 统一到 rank pct 量纲：下面 `sec` 是 [0,1] 的 rank 均值，若回退值用原始
    # 因子值（量级可能 ±1e-2），两者混在一张面板里排序会让「二级因子缺失的
    # 股票」被系统性顶到最前/最后。`run()` 反正要再 rank 一次，故 pct 等价。
    a_rank = a.rank(axis=1, pct=True)
    # 🚨 定向后「越大越好」⇒ 好的一端是**高分位**，候选集取 rank >= 1-q。
    # 写成 <= q 会取到最差的那一端（与单因子 top10 重合 0.0% 就是这么来的）。
    gate = a_rank >= (1.0 - q)
    if not others:
        return a_rank.where(gate)
    num = cnt = None
    for p in others:
        r = p.loc[idx, cols].rank(axis=1, pct=True).where(gate)
        num = r if num is None else num.add(r, fill_value=0)
        c = r.notna().astype(np.int16)
        cnt = c if cnt is None else cnt.add(c)
    # 🚨 门控阈值必须随 other 的数量缩放。写死 `cnt >= 2` 时，
    # **只有 1 个二级因子**的调用会整片变 NaN，随后被下面的 `sec.where`
    # 静默回退到锚因子本身 ⇒ 该臂等于没跑（实测 9 个 gate+单因子臂
    # 与 gate_only 逐位相同、毛收益都是 46.94%，就是这个 bug）。
    min_votes = min(2, len(others))
    cnt = cnt.where(cnt >= min_votes)
    sec = (num / cnt).where(gate)
    return sec.where(sec.notna(), a_rank.where(gate))


def orient(p: pd.DataFrame, ic: float) -> pd.DataFrame:
    """按 IC 符号定向 ⇒ 之后一律「越大越好」。"""
    return -p if ic < 0 else p


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="PIT 池重算 im_* 并验证 1+2")
    ap.add_argument("--stage", default="all",
                    choices=["all", "audit", "lag", "composite"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--buffer", type=float, default=2.0)
    ap.add_argument("--q", type=float, default=0.1, help="两级聚合的 Q1 分位")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    store, codes, daily, fwd = load_context()
    days = store.all_days()
    feats = build_features(store, days[0], days[-1], with_lag=True)

    # 对齐池与收益
    pool_cols = [c for c in store.codes if c in fwd.columns]
    log.info("可用列 %d（面板 %d ∩ 日线 %d）", len(pool_cols),
             len(store.codes), len(fwd.columns))

    # ---- 全因子 IC ----
    ic_cache = OUT / "pit_ic_table.parquet"
    if ic_cache.exists():
        tab = pd.read_parquet(ic_cache)
        log.info("复用 IC 表缓存（%d 个因子）", len(tab))
    else:
        rows = []
        for name, p in feats.items():
            p = p.reindex(columns=pool_cols)
            ic = daily_ic(p, fwd)
            st = ic_stats(ic)
            if st:
                rows.append({"name": name, **st})
        tab = pd.DataFrame(rows).sort_values("t", key=np.abs, ascending=False)
        tab.to_parquet(ic_cache)
    tag = "pit" if args.stage in ("all", "audit") else "stage"
    tab.to_csv(OUT / f"pit_ic_{tag}.csv", index=False, encoding="utf-8-sig")
    log.info("IC 表 %d 个因子，|t|>1.96 的 %d 个", len(tab),
             int((tab["t"].abs() > 1.96).sum()))

    base = tab[~tab["name"].str.contains("_lag[12]$")]
    print("\n=== Q0 PIT 池 IC（当日特征，按 |t| 排序前 20）===")
    with pd.option_context("display.width", 160, "display.float_format", "{:.4f}".format):
        print(base.head(20).to_string(index=False))

    if args.stage in ("all", "lag"):
        lag = tab[tab["name"].str.contains("_lag[12]$")]
        if len(lag):
            print("\n=== Q2 lag 因子 IC（按 |t| 排序前 20）===")
            with pd.option_context("display.width", 160,
                                   "display.float_format", "{:.4f}".format):
                print(lag.head(20).to_string(index=False))
            n_sig = int((lag["t"].abs() > 1.96).sum())
            log.info("lag 因子 %d 个，|t|>1.96 的 %d 个（%.1f%%）",
                     len(lag), n_sig, 100 * n_sig / len(lag))
            # lag vs 当日版共线性（lag 本质是当日版延迟一天，必须先剔共线）
            base_ix = base.set_index("name")
            for k in ("lag1", "lag2"):
                s = lag[lag["name"].str.endswith("_" + k)].copy()
                s["orig"] = s["name"].str.replace("_" + k + "$", "", regex=True)
                s = s[s["orig"].isin(base_ix.index)]
                s["ic_orig"] = s["orig"].map(base_ix["ic"])
                s["t_orig"] = s["orig"].map(base_ix["t"])
                both = (s["t"].abs() > 1.96) & (s["t_orig"].abs() > 1.96)
                only_lag = (s["t"].abs() > 1.96) & (s["t_orig"].abs() <= 1.96)
                print(f"  {k}: IC相关 {s['ic'].corr(s['ic_orig']):.3f} / "
                      f"同号 {100 * (np.sign(s['ic']) == np.sign(s['ic_orig'])).mean():.0f}% / "
                      f"两侧显著 {int(both.sum())} / 仅lag显著 {int(only_lag.sum())}")

    if args.stage in ("all", "composite"):
        run_composite(feats, fwd, base, pool_cols, tab, args)

    json.dump({"n_pool": len(pool_cols), "n_factors": len(tab),
               "top": args.top, "q": args.q,
               "n_sig": int((tab["t"].abs() > 1.96).sum())},
              open(OUT / "pit_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def run_composite(feats, fwd, base, pool_cols, tab, args) -> None:
    """建议 1 验证：旧口径全截面平均 vs 新口径「锚切 Q1 + Q1 内二级排序」。"""
    ics = dict(zip(tab["name"], tab["ic"], strict=True))
    sig = base[base["t"].abs() > 1.96].copy()
    sig = sig.reindex(sig["t"].abs().sort_values(ascending=False).index)
    anchor = sig.iloc[0]["name"]
    t10 = sig.head(10)["name"].tolist()
    log.info("锚因子 %s（t=%.2f）/ 前10 %d 个", anchor, sig.iloc[0]["t"], len(t10))

    def _prep(p: pd.DataFrame) -> pd.DataFrame:
        return p.reindex(columns=pool_cols).reindex(index=fwd.index)

    feats_a = {k: _prep(v) for k, v in feats.items()}
    arms: dict[str, pd.DataFrame] = {}
    arms["single"] = orient(feats_a[anchor], ics[anchor])
    arms["flat10"] = composite_flat(t10, feats_a, ics)
    arms["flat_sig"] = composite_flat(sig["name"].tolist(), feats_a, ics)
    arms["q1_2stage10"] = composite_2stage(anchor, t10, feats_a, ics, q=args.q)
    arms["q1_2stage_sig"] = composite_2stage(anchor, sig["name"].tolist(),
                                             feats_a, ics, q=args.q)

    rows = []
    for name, s in arms.items():
        res = run(s, fwd, top=args.top, buffer_mult=args.buffer)
        g, n = stats(res["gross"]), stats(res["net"])
        if not g or not n:
            continue
        rows.append({"arm": name, "gross_ann": g["ann"], "net_ann": n["ann"],
                     "sr": n["sr"], "mdd": n["mdd"], "turn": float(res["turn"].mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "pit_composite_arms.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 建议1 复合方式对照 · PIT池 · top{args.top} buffer{args.buffer} · 27bp ===")
    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(t.to_string(index=False))

    # ---- Q1 命中率：为什么旧口径会稀释 ----
    a = arms["single"]
    for name in ("flat10", "q1_2stage10"):
        s = arms[name]
        hit, cover = [], []
        for dt in s.index.intersection(a.index):
            av, sv = a.loc[dt].dropna(), s.loc[dt].dropna()
            if len(av) < 30 or len(sv) < args.top:
                continue
            a_top = set(av.nlargest(args.top).index)
            s_top = set(sv.nlargest(args.top).index)
            if not s_top:
                continue
            hit.append(len(s_top & a_top) / len(s_top))
            # 单因子 Q1 = 定向后得分最高的 q 分位（a 已定向，越大越好）
            thr = av.quantile(1 - args.q)
            cover.append(len(s_top & set(av[av >= thr].index)) / len(s_top))
        print(f"  {name}: 与单因子top{args.top}重合 {100 * np.mean(hit):.1f}% / "
              f"落在单因子Q1 {100 * np.mean(cover):.1f}%")

    run_lag_arms(feats_a, fwd, base, tab, args)


def run_lag_arms(feats, fwd, base, tab, args) -> None:
    """建议 2 验证：lag 因子能否在**组合层**提供增量（不只是 IC 层）。

    lag 因子与当日版同号率 76%~80%，所以 IC 层面「都显著」毫无意义——
    真正的检验是：把 lag 加进复合后，组合是否比只用当日版更好。
    三个臂：仅当日 / 当日+lag1 / 当日+lag1+lag2，全部用 corrected 的
    两级聚合（上一轮已证 flat 口径会稀释尾部）。
    """
    ics = dict(zip(tab["name"], tab["ic"], strict=True))
    anchor = base.reindex(base["t"].abs().sort_values(ascending=False).index
                          ).iloc[0]["name"]
    t10 = base.reindex(base["t"].abs().sort_values(ascending=False).index
                       ).head(10)["name"].tolist()

    def _with_lags(names: list[str], level: int) -> list[str]:
        out = list(names)
        for n in names:
            for k in range(1, level + 1):
                cand = f"{n}_lag{k}"
                if cand in feats:
                    out.append(cand)
        return out

    rows = []
    for label, lv in (("cur_only", 0), ("cur+lag1", 1), ("cur+lag1+lag2", 2)):
        pool = _with_lags(t10, lv)
        s = composite_2stage(anchor, pool, feats, ics, q=args.q)
        res = run(s, fwd, top=args.top, buffer_mult=args.buffer)
        g, n = stats(res["gross"]), stats(res["net"])
        if not g or not n:
            continue
        rows.append({"arm": label, "n_factor": len(pool),
                     "gross_ann": g["ann"], "net_ann": n["ann"],
                     "sr": n["sr"], "mdd": n["mdd"],
                     "turn": float(res["turn"].mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "pit_lag_arms.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 建议2 lag 因子组合层检验 · PIT池 · top{args.top} · 27bp ===")
    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(t.to_string(index=False))


if __name__ == "__main__":
    main()