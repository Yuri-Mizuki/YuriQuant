"""分钟频因子的月频可兑现性（2026-11-15）
==========================================

背景与要解决的问题
------------------
`reports/intraday_source/README.md` 的结论是「日频路线被成本墙否掉」：
`ilate`（尾盘 30 分钟收益）取反毛 +14.96bp/日、换手 **85.5%/日**，
打平单边仅 **8.8bp < 13.5bp**，扣费后 −8.12bp/日。

那个报告里还有一行「换到月度框架：多空 −151.8bp/月（取反 +151.8，IC −0.0386，n=55）」，
**但 `intraday_factor_probe.py` 里没有月度代码，这个数字不可复现**。若它成立，
月频换手远低于日频，扣费后可能仍有 +100bp/月以上——那「分钟频不能做」就得改。
所以本脚本把它重算，并把口径固定下来。

同时补两个日频报告没量过的东西：

1. **月频是否真的降低换手**（如果月频信号每月换 60%+，成本墙依旧）；
2. **触及度**——分钟数据只有 HS300（421 只），而 K15 持仓市值分位仅 37.8%
   （见 `em_rating_probe/reach/`），两个集合是否重叠。

预注册口径（先定后看，三种聚合全报，不挑）
------------------------------------------
=================  ==================================================================
``mean``           当月日均 ``ilate``（最稳，主口径）
``sum``            当月 ``ilate`` 累计（≈ 尾盘动量的月内累加）
``eom``            月末当日 ``ilate``（点值，信息最新最噪声）
=================  ==================================================================

三个都报，避免「挑一个能过口径的」。评价四层：

1. **月频截面 IC**（HS300 内，T+1 开盘→下月末 T+1 开盘，与主项目月频口径一致）；
2. **多空**：D10−D1 等权，毛/换手/扣费净；
3. **只做多 D10 超额**（相对截面均值）——这才是能兑现的口径，多空需要做空；
4. **K15 叠加**：`mid13 + λ·dev`，未覆盖 = 0（与评级探针同口径）。

用法
----
    python -m scripts.evaluation.minute_monthly_probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation.intraday_factor_probe import build_intraday
from scripts.evaluation.rating_probe import (
    BASE, K, COST_BP, factor_dev, focus_excess, load_main_signal, month_pairs, nw_t,
)
from scripts.evaluation.rating_reach_probe import holdings

OUT = Path("reports/intraday_source/monthly")
AGGS = ("mean", "sum", "eom")
LAMBDAS = (0.1, 0.2, 0.4)
FACTOR = "ilate"


def monthly_signal(intra: pd.DataFrame, rebal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """日频 → 月频信号。用「该月全部日频值」聚合，严格 ≤ 月末调仓日，PIT 安全。"""
    mon = pd.Index(intra.index).to_period("M")
    tmap = pd.Series(rebal, index=pd.PeriodIndex(rebal, freq="M"))
    out: dict[str, pd.DataFrame] = {}
    for name, how in (("mean", "mean"), ("sum", "sum"), ("eom", "last")):
        m = getattr(intra.groupby(mon), how)()
        m = m.reindex(tmap.index)
        m.index = pd.DatetimeIndex(tmap.values)
        out[name] = m.astype("float32")
    return out


def fwd(cal: pd.DatetimeIndex, openp: pd.DataFrame, t: pd.Timestamp,
        t2: pd.Timestamp) -> pd.Series | None:
    i, j = cal.get_loc(t), cal.get_loc(t2)
    if j + 1 >= len(cal):
        return None
    return (openp.loc[cal[j + 1]] / openp.loc[cal[i + 1]] - 1).replace([np.inf, -np.inf], np.nan)


def eval_monthly(sig: pd.DataFrame, pairs: list[tuple], cal: pd.DatetimeIndex,
                 openp: pd.DataFrame, mask: pd.DataFrame,
                 ctrl: pd.DataFrame | None = None) -> dict:
    """月频：截面 IC + D10−D1 + D10 多头超额，含换手与成本。

    ctrl 非空时，每月先在可交易日截面内把信号的秩对 ctrl 的秩做 OLS 取残差，
    再用残差排序——用来回答「这个信号是不是已知因子的伪装」。
    """
    ics, ls, lo = [], [], []
    prev_top, prev_bot = set(), set()
    turns = []
    years, years_lo = {}, {}
    for t, t2 in pairs:
        if t not in sig.index:
            continue
        r = fwd(cal, openp, t, t2)
        if r is None:
            continue
        s = sig.loc[t]
        ok = s.notna() & r.notna() & mask.loc[t].reindex(s.index).fillna(False).astype(bool)
        if ok.sum() < 100:
            continue
        ss, rr = s.where(ok), r.where(ok)
        if ctrl is not None:
            cdfs = ctrl if isinstance(ctrl, (list, tuple)) else [ctrl]
            cols, ok2 = [], ss.notna()
            for cd in cdfs:
                if t not in cd.index:
                    ok2 = ok2 & False
                    continue
                c = cd.loc[t].reindex(ss.index)
                ok2 = ok2 & c.notna()
                cols.append(c)
            if ok2.sum() >= 100 and len(cols) == len(cdfs):
                X = np.column_stack([np.ones(int(ok2.sum()))]
                                    + [c[ok2].rank().to_numpy(float) for c in cols])
                y = ss[ok2].rank().to_numpy(float)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                res = pd.Series(np.nan, index=ss.index, dtype="float64")
                res[ok2] = y - X @ beta
                ss = res.where(ok)
        ics.append(ss.rank().corr(rr.rank()))
        n = int(ok.sum())
        k = max(1, n // 10)
        top = set(ss.nlargest(k).index)
        bot = set(ss.nsmallest(k).index)
        # 收益是小数（0.05 = 5%），×1e4 → bp，与 COST_BP 单位一致
        ls_v = (rr[list(top)].mean() - rr[list(bot)].mean()) * 1e4
        lo_v = (rr[list(top)].mean() - rr[ok].mean()) * 1e4
        ls.append(ls_v)
        lo.append(lo_v)
        years.setdefault(pd.Timestamp(t).year, []).append(ls_v)
        years_lo.setdefault(pd.Timestamp(t).year, []).append(lo_v)
        turns.append(((len(top - prev_top) / max(len(top), 1))
                      + (len(bot - prev_bot) / max(len(bot), 1))) / 2)
        prev_top, prev_bot = top, bot
    ic, t_ic = nw_t(np.asarray(ics))
    g, t_g = nw_t(np.asarray(ls))
    l, t_l = nw_t(np.asarray(lo))
    turn = float(np.mean(turns)) if turns else np.nan
    yr = pd.Series({y: np.mean(v) for y, v in years.items()})
    return dict(n_month=len(ics), ic=ic, ic_t=t_ic, ls_gross_bp=g, ls_t=t_g,
                lo_gross_bp=l, lo_t=t_l, turn=turn,
                ls_net_bp=g - 2 * turn * COST_BP, lo_net_bp=l - turn * COST_BP,
                yr_pos=int((yr > 0).sum()), yr_n=len(yr))


def reach(h: pd.DataFrame, pool: pd.Index) -> dict:
    """K15 持仓中「有分钟数据」的比例。"""
    tot = hit = 0
    for rec in h.itertuples():
        for c in rec.ret:
            tot += 1
            hit += int(c in pool)
    return dict(n_hold=tot, n_cover=hit, p_cover=hit / tot if tot else np.nan)


def main() -> None:
    close = pd.read_parquet(f"{BASE}/close_adj.parquet")
    openp = pd.read_parquet(f"{BASE}/open_adj.parquet")
    mask = pd.read_parquet(f"{BASE}/tradable_mask.parquet").fillna(False).astype(bool)
    mid13 = load_main_signal(mask)
    cal = close.index

    intra = build_intraday()[FACTOR]
    pool = intra.columns
    print(f"分钟数据：{intra.index.min().date()} ~ {intra.index.max().date()}，"
          f"{len(intra)} 日 × {len(pool)} 只")

    pairs = [(a, b) for a, b in month_pairs(cal)
             if a >= intra.index.min() and a <= intra.index.max()]
    pairs_c = [(a, b) for a, b in pairs if a in mid13.index]
    rebal = pd.DatetimeIndex([a for a, _ in pairs])
    sig = monthly_signal(intra, rebal)

    # ---- 触及度：K15 与 HS300 池的重叠
    base_ex, h = holdings(mid13, pairs_c, cal, openp, mask)
    m0, t0 = nw_t(base_ex)
    rc = reach(h, pool)
    print(f"\n触及度：K15 持仓 {rc['n_hold']} 个持仓月，其中在有分钟数据的池内 "
          f"{rc['n_cover']}（{rc['p_cover']:.1%}）")
    print(f"基线 mid13 K15 月均超额 {m0:+.3f}%  t {t0:+.2f}  n={len(base_ex)}")

    # ---- 月频评价（原始方向 + 取反，因为日频是反转）
    #     另外做残差版：控制「当月自身收益」（月度反转），看 eom 是不是它的伪装
    mend = close.loc[rebal]
    rev_m = mend / mend.shift(1) - 1
    day_ret = close.pct_change().loc[rebal]      # 月末当日全天收益

    rows = []
    for name in AGGS:
        s = sig[name]
        for sign, tag in ((1, "raw"), (-1, "rev")):
            d = eval_monthly(s * sign, pairs, cal, openp, mask)
            d.update(agg=name, dir=tag, ctrl="none")
            rows.append(d)
    for name in AGGS:
        d = eval_monthly(sig[name] * -1, pairs, cal, openp, mask, ctrl=rev_m)
        d.update(agg=name, dir="rev", ctrl="mom")
        rows.append(d)
    # eom 的额外关键检验：控制「月末当日全天收益」，看尾盘 30 分钟是否还有独立信息
    d = eval_monthly(sig["eom"] * -1, pairs, cal, openp, mask, ctrl=day_ret)
    d.update(agg="eom", dir="rev", ctrl="day")
    rows.append(d)
    # 双控制：月反转 + 月末当日收益（最严）
    d = eval_monthly(sig["eom"] * -1, pairs, cal, openp, mask, ctrl=[rev_m, day_ret])
    d.update(agg="eom", dir="rev", ctrl="both")
    rows.append(d)
    res = pd.DataFrame(rows)[["agg", "dir", "ctrl", "n_month", "ic", "ic_t",
                              "ls_gross_bp", "ls_t", "turn", "ls_net_bp",
                              "lo_gross_bp", "lo_t", "lo_net_bp", "yr_pos", "yr_n"]]
    print("\n月频评价（多空 D10−D1 / 多头 D10 超额，单位 bp/月；ctrl=mom 为控制当月收益后的残差）")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- K15 叠加（只测主口径 mean，两个方向）
    add = []
    for sign, tag in ((1, "raw"), (-1, "rev")):
        dev = factor_dev(sig["mean"] * sign)
        for lam in LAMBDAS:
            e = focus_excess(mid13 + lam * dev, pairs_c, cal, openp, mask)
            n = min(len(e), len(base_ex))
            dm, dt = nw_t(e[:n] - base_ex[:n])
            add.append(dict(dir=tag, lam=lam, mean_ex=float(np.mean(e[:n])),
                            delta_pp=dm, t=dt, n=n))
    a = pd.DataFrame(add)
    print("\nK15 叠加（mid13 + λ·dev，λ=0.1/0.2/0.4）")
    print(a.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "monthly_eval.csv", index=False, encoding="utf-8")
    a.to_csv(OUT / "k15_overlay.csv", index=False, encoding="utf-8")
    pd.DataFrame([dict(base_ex=m0, base_t=t0, n=len(base_ex), **rc)]).to_csv(
        OUT / "reach.csv", index=False, encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
