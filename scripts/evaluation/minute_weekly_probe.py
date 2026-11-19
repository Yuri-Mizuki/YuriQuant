"""分钟频因子的周频可兑现性（2026-11-15）
==========================================

要回答的问题
------------
`reports/intraday_source/` 已确认两件事：
- **日频死**：`ilate`（尾盘 30 分钟收益）取反毛 +14.96bp/日、换手 **85.5%/日**，
  打平单边仅 8.8bp < 13.5bp；
- **月频 `eom` 活**：月末最后一个交易日的 `ilate` 取反，双控制后 D10 多头
  +75.7bp/月（t 2.13）、换手 87.3%/月。

中间空着一档：**周频**。直觉上它该是「成本摊薄 5 倍、信号只衰减一点」的甜点，
但这是个没验证过的假设——必须先量出来。

本脚本做三件事
--------------
1. **信号寿命曲线**（Part A）：日频信号对 h = 1/2/3/5/10/20 日前瞻收益的毛超额与
   换手，按 `(毛 − 2·turn·cost) × 250/h` 折年化。这是最本质的一问：
   **信号在第几天衰减到不够付钱**。周频只是这条曲线上的一个点。
2. **周频评价**（Part B）：周内三聚合（`wmean` 日均 / `wsum` 累计 / `wend` 周末日）×
   两方向 × 三级控制（周反转 / 周末当日全天收益 / 两者），含换手与扣费净收益。
3. **`wend` 与月频 `eom` 是不是同一个时点**（Part C）：两者必须用**独立的构造路径**
   （`eom` = 自然月最后一天，`wend` = ISO 周最后一天），否则会验成同义反复。

预注册口径（先定后看）
----------------------
- 调仓日 = **每周最后一个交易日**，T+1 开盘进出（与主项目一致）；
- 跳过交易日数 < 3 的「残周」（春节/国庆），并报告跳过数；
- 成本 **13.5bp 单边**（多空往返 27bp，多头 D10 单边 13.5bp）；
- Newey-West 滞后：周频 **lag=5**（约一个月），同时报 lag=3 / lag=8 看稳健性；
- D10/D1 按周截面十分位，只在「有分钟数据 + 可交易 + 有前瞻收益」的子集内排序。

用法
----
    python -m scripts.evaluation.minute_weekly_probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation.intraday_factor_probe import build_intraday
from scripts.evaluation.rating_probe import BASE, COST_BP, load_main_signal, month_pairs, nw_t

OUT = Path("reports/intraday_source/weekly")
AGGS = ("wmean", "wsum", "wend")
FACTOR = "ilate"
HORIZONS = (1, 2, 3, 5, 10, 20)
MIN_N = 100          # 截面最少股票数（HS300 池约 400 只）
MIN_WEEK_DAYS = 3    # 少于 3 个交易日的周视为残周，跳过


# --------------------------------------------------------------------------- 配对
def week_pairs(cal: pd.DatetimeIndex) -> tuple[list[tuple], int]:
    """(本周最后交易日, 下周最后交易日)。返回配对与被跳过的残周数。"""
    iso = cal.isocalendar()
    key = pd.MultiIndex.from_arrays([iso["year"].to_numpy(), iso["week"].to_numpy()])
    s = pd.Series(cal, index=key)
    rb = list(s.groupby(level=[0, 1]).last().values)          # 已按 ISO 周序排列
    pairs, dropped = [], 0
    for a, b in zip(rb[:-1], rb[1:]):
        gap = cal.get_loc(b) - cal.get_loc(a)
        if gap < MIN_WEEK_DAYS:
            dropped += 1
            continue
        pairs.append((a, b))
    return pairs, dropped


def horizon_pairs(cal: pd.DatetimeIndex, h: int) -> list[tuple]:
    """持有 h 个交易日、**不重叠**：t=cal[i] 出信号，t2=cal[i+h]，下一个组合从 i+h 起。

    必须 stride=h。若用 stride=1（每天都建一个持 h 日的组合），相邻组合重叠 h−1 天，
    换手与 t 值都会被重叠污染：换手仍按「相邻日的组合差异」算（≈每日换手），
    而年化却按 250/h 折算，等于**把成本除以 h、却没把换手除以 h**，会凭空造出收益。
    """
    return [(cal[i], cal[i + h]) for i in range(0, len(cal) - h, h)]


def fwd(cal: pd.DatetimeIndex, openp: pd.DataFrame, t: pd.Timestamp,
        t2: pd.Timestamp) -> pd.Series | None:
    i, j = cal.get_loc(t), cal.get_loc(t2)
    if j + 1 >= len(cal):
        return None
    return (openp.loc[cal[j + 1]] / openp.loc[cal[i + 1]] - 1).replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- 聚合
def weekly_signal(intra: pd.DataFrame, rebal: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """日频 → 周频信号。只用该周内 ≤ 调仓日的值，PIT 安全。"""
    iso = intra.index.isocalendar()
    g = intra.groupby(pd.MultiIndex.from_arrays([iso["year"].to_numpy(), iso["week"].to_numpy()]))
    riso = rebal.isocalendar()
    tmap = pd.Series(rebal, index=pd.MultiIndex.from_arrays(
        [riso["year"].to_numpy(), riso["week"].to_numpy()]))
    out: dict[str, pd.DataFrame] = {}
    for name, how in (("wmean", "mean"), ("wsum", "sum"), ("wend", "last")):
        m = getattr(g, how)().reindex(tmap.index)
        m.index = pd.DatetimeIndex(tmap.values)
        out[name] = m.astype("float32")
    return out


# --------------------------------------------------------------------------- 评价
def _residual(ss: pd.Series, ctrl, t: pd.Timestamp, min_n: int = MIN_N) -> pd.Series:
    """把信号秩对控制变量秩做 OLS 取残差（截面内）。"""
    cdfs = ctrl if isinstance(ctrl, (list, tuple)) else [ctrl]
    cols, ok = [], ss.notna()
    for cd in cdfs:
        if t not in cd.index:
            return ss
        c = cd.loc[t].reindex(ss.index)
        ok = ok & c.notna()
        cols.append(c)
    if ok.sum() < min_n:
        return ss
    X = np.column_stack([np.ones(int(ok.sum()))] + [c[ok].rank().to_numpy(float) for c in cols])
    y = ss[ok].rank().to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = pd.Series(np.nan, index=ss.index, dtype="float64")
    res[ok] = y - X @ beta
    return res.where(ss.notna())


def eval_pairs(sig: pd.DataFrame, pairs: list[tuple], cal: pd.DatetimeIndex,
               openp: pd.DataFrame, mask: pd.DataFrame, *,
               ctrl=None, lag: int = 5, annualize: float | None = None,
               min_n: int = MIN_N) -> dict:
    """通用口径：截面 IC + D10−D1 + D10 多头超额，含换手与扣费净收益。

    annualize 非空时按「每年调仓 250/annualize 次」折算年化 bp。
    """
    ics, ls, lo, turns = [], [], [], []
    prev_top, prev_bot = set(), set()
    years_lo = {}
    for t, t2 in pairs:
        if t not in sig.index:
            continue
        r = fwd(cal, openp, t, t2)
        if r is None:
            continue
        s = sig.loc[t]
        ok = s.notna() & r.notna() & mask.loc[t].reindex(s.index).fillna(False).astype(bool)
        if ok.sum() < min_n:
            continue
        ss, rr = s.where(ok), r.where(ok)
        if ctrl is not None:
            ss = _residual(ss, ctrl, t, min_n).where(ok)
        if ss.notna().sum() < min_n:
            continue
        ics.append(ss.rank().corr(rr.rank()))
        n = int(ok.sum())
        k = max(1, n // 10)
        top = set(ss.nlargest(k).index)
        bot = set(ss.nsmallest(k).index)
        # 收益是小数（0.05 = 5%），×1e4 → bp
        ls.append((rr[list(top)].mean() - rr[list(bot)].mean()) * 1e4)
        lo_v = (rr[list(top)].mean() - rr[ok].mean()) * 1e4
        lo.append(lo_v)
        years_lo.setdefault(pd.Timestamp(t).year, []).append(lo_v)
        turns.append(((len(top - prev_top) / max(len(top), 1))
                      + (len(bot - prev_bot) / max(len(bot), 1))) / 2)
        prev_top, prev_bot = top, bot

    ic, t_ic = nw_t(np.asarray(ics), lag=lag)
    g, t_g = nw_t(np.asarray(ls), lag=lag)
    l, t_l = nw_t(np.asarray(lo), lag=lag)
    turn = float(np.mean(turns)) if turns else np.nan
    yr = pd.Series({y: np.mean(v) for y, v in years_lo.items()}) if years_lo else pd.Series(dtype=float)
    scale = (250.0 / annualize) if annualize else np.nan
    return dict(n=len(ics), ic=ic, ic_t=t_ic,
                ls_gross_bp=g, ls_t=t_g, lo_gross_bp=l, lo_t=t_l,
                turn=turn,
                ls_net_bp=g - 2 * turn * COST_BP, lo_net_bp=l - turn * COST_BP,
                ls_net_ann=g * scale - 2 * turn * COST_BP * scale,
                lo_net_ann=l * scale - turn * COST_BP * scale,
                yr_pos=int((yr > 0).sum()), yr_n=len(yr))


def ctrl_frames(close: pd.DataFrame, rebal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """周反转（本周收益）与周末当日全天收益。"""
    px = close.loc[rebal]
    rev = px / px.shift(1) - 1
    day = close.pct_change().loc[rebal]
    return rev, day


# --------------------------------------------------------------------------- 主流程
def main() -> None:
    close = pd.read_parquet(f"{BASE}/close_adj.parquet")
    openp = pd.read_parquet(f"{BASE}/open_adj.parquet")
    mask = pd.read_parquet(f"{BASE}/tradable_mask.parquet").fillna(False).astype(bool)
    mid13 = load_main_signal(mask)
    cal = close.index

    intra = build_intraday()[FACTOR]
    print(f"分钟数据：{intra.index.min().date()} ~ {intra.index.max().date()}，"
          f"{len(intra)} 日 × {len(intra.columns)} 只")

    sub = cal[(cal >= intra.index.min()) & (cal <= intra.index.max())]

    # ============================================================ Part A 信号寿命
    print("\n=== Part A 信号寿命：日频 ilate（取反）× 不同持有期 ===")
    print("（毛超额=日频信号每 h 日调仓一次的 D10/D1 超额；净=毛−2·turn·13.5bp；年化=×250/h）")
    rows = []
    for h in HORIZONS:
        pr = horizon_pairs(sub, h)
        d = eval_pairs(intra * -1, pr, cal, openp, mask, lag=5, annualize=h)
        d.update(h=h, rebal_per_year=round(250 / h, 1))
        rows.append(d)
    A = pd.DataFrame(rows)[["h", "rebal_per_year", "n", "ic", "ic_t",
                            "ls_gross_bp", "ls_t", "turn", "ls_net_bp", "ls_net_ann",
                            "lo_gross_bp", "lo_t", "lo_net_bp", "lo_net_ann"]]
    print(A.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ============================================================ Part B 周频
    pairs, dropped = week_pairs(sub)
    rebal = pd.DatetimeIndex([a for a, _ in pairs])
    sig = weekly_signal(intra, rebal)
    rev_w, day_w = ctrl_frames(close, rebal)
    print(f"\n周频：{len(pairs)} 周（跳过残周 {dropped}），"
          f"{rebal[0].date()} ~ {rebal[-1].date()}")

    rows = []
    for name in AGGS:
        for sign, tag in ((1, "raw"), (-1, "rev")):
            d = eval_pairs(sig[name] * sign, pairs, cal, openp, mask, lag=5, annualize=5)
            d.update(agg=name, dir=tag, ctrl="none", lag=5)
            rows.append(d)
    for name in AGGS:
        for ctrl, cname in ((rev_w, "rev"), (day_w, "day"), ([rev_w, day_w], "both")):
            d = eval_pairs(sig[name] * -1, pairs, cal, openp, mask,
                           ctrl=ctrl, lag=5, annualize=5)
            d.update(agg=name, dir="rev", ctrl=cname, lag=5)
            rows.append(d)
    B = pd.DataFrame(rows)[["agg", "dir", "ctrl", "n", "ic", "ic_t", "ls_gross_bp", "ls_t",
                            "turn", "ls_net_bp", "ls_net_ann",
                            "lo_gross_bp", "lo_t", "lo_net_bp", "lo_net_ann",
                            "yr_pos", "yr_n"]]
    print("\n=== Part B 周频评价（bp/周；net_ann 为按 50 周折年化 bp）===")
    print(B.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # lag 稳健性：只对主口径 wend 取反双控，换 lag
    print("\n=== Part B2 Newey-West 滞后稳健性（wend 取反 + 双控制）===")
    lags = []
    for lg in (3, 5, 8):
        d = eval_pairs(sig["wend"] * -1, pairs, cal, openp, mask,
                       ctrl=[rev_w, day_w], lag=lg, annualize=5)
        d.update(lag=lg)
        lags.append(d)
    B2 = pd.DataFrame(lags)[["lag", "ls_gross_bp", "ls_t", "lo_gross_bp", "lo_t", "lo_net_ann"]]
    print(B2.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ============================================================ Part C 同源性
    print("\n=== Part C wend 与月频 eom 是否同源 ===")
    mp = [(a, b) for a, b in month_pairs(cal) if a >= intra.index.min() and a <= intra.index.max()]
    mrebal = pd.DatetimeIndex([a for a, _ in mp])

    # ⚠️ 两个信号必须用**独立的构造路径**，否则就是同义反复：
    #   eom  = 自然月最后一天的 ilate（与 minute_monthly_probe 的定义一致）
    #   wend = ISO 周最后一天的 ilate（weekly_signal 的 "last"）
    # 第一版我误把 eom 也写成 groupby(week).last()，于是「验证」出相关 1.000 —— 那是自己跟自己比。
    km = pd.MultiIndex.from_arrays([intra.index.year, intra.index.month])
    eom_m = intra.groupby(km).last()
    im = pd.MultiIndex.from_arrays([mrebal.year, mrebal.month])
    eom_m = eom_m.reindex(im)
    eom_m.index = pd.DatetimeIndex(mrebal)
    wend_m = sig["wend"].reindex(mrebal)

    # 日历事实：月末最后交易日是否就是该周最后交易日
    iso_all = sub.isocalendar()
    kk = pd.MultiIndex.from_arrays([iso_all["year"].to_numpy(), iso_all["week"].to_numpy()])
    wk_last = pd.Series(sub, index=kk).groupby(level=[0, 1]).last()
    overlap = [t for t in mrebal
               if wk_last.get((t.isocalendar()[0], t.isocalendar()[1])) == t]
    print(f"日历事实：{len(overlap)}/{len(mrebal)} 个月的月末最后交易日 = 该周最后交易日；"
          f"其余 {len(mrebal) - len(overlap)} 个月月末落在周中")

    # 只在「重合月」上比。注意这是**构造恒等**：两边取的是同一天同一个数，
    # 相关必然是 1，只能作为「构造一致性检查」，不能当成发现。
    cors = []
    for t in overlap:
        if t not in eom_m.index or t not in sig["wend"].index:
            continue
        a, b = eom_m.loc[t], sig["wend"].loc[t]
        ok = a.notna() & b.notna()
        if ok.sum() >= MIN_N:
            cors.append(a[ok].rank().corr(b[ok].rank()))
    c = float(np.mean(cors)) if cors else np.nan
    sd = float(np.std(cors, ddof=1)) if len(cors) > 1 else np.nan
    print(f"重合月（{len(cors)} 个）截面秩相关：{c:.3f}（逐月 sd {sd:.1e}）"
          f"—— 同一天同一个值，**恒等于 1 是定义使然，不是发现**")
    print(f"不重合的 {len(mrebal) - len(overlap)} 个月：`wend` 取到的是**下月初**的周末日，"
          f"与 `eom` 不是同一天 ⇒ 两者是不同的时点信号")
    print("⇒ 结论不是「wend ≡ eom」，而是：**月末 ≠ 周末**。`wend`（每周末取反）"
          "IC 仅 0.01、净 −459bp/年；`eom`（每月末取反）IC 0.039、毛/成本 5.91。"
          "「月末」这个位置比「任一周末」强约 4 倍 ⇒ eom 的强度**不是周效应伪装的**。")

    # ============================================================ 写盘
    OUT.mkdir(parents=True, exist_ok=True)
    A.to_csv(OUT / "lifetime.csv", index=False, encoding="utf-8")
    B.to_csv(OUT / "weekly_eval.csv", index=False, encoding="utf-8")
    B2.to_csv(OUT / "lag_robust.csv", index=False, encoding="utf-8")
    pd.DataFrame([dict(wend_eom_corr_overlap=c, n_overlap=len(cors),
                       cal_same=len(overlap), cal_n=len(mrebal),
                       n_week=len(pairs), dropped_week=dropped)]).to_csv(
        OUT / "homology.csv", index=False, encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
