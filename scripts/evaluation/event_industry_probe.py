"""L4 事件聚合：行业内增持「密度」本身是不是信号（2026-11-16）
==============================================================

为什么做这一层
--------------
L1（单事件）已通过：董监高自购有 alpha。L2/L3（事件×状态、事件链）已双否。
剩下一个完全不同的问题：**把视角从个股抬到行业，密度本身有没有信息？**

一个行业一个月内冒出 5 家公司董监高增持，和 5 家分散在 5 个行业，语义完全不同：
前者可能是行业景气拐点的早期集体判断，后者只是个体行为。而且行业口径的容量大一个数量级。

信号定义
--------
1. 事件 → 行业（申万一级，**PIT**：用 `avail` 当日的 `in_date ≤ d < out_date`）
2. 行业日事件数 `n[日, 行业]`，滚动 `K` 日累计 `roll = n.rolling(K).sum()`
3. **边缘触发**：`roll ≥ M` 且前一日 `roll < M` ⇒ 该行业进入密集期，T+1 开盘建仓
4. 持有 `H` 日（同行业重叠则延长，不重复建仓），等权买入**该行业全部成分股**（等权模拟）
5. 基准 = 全A等权

判据（先定后算）
----------------
1. 扣费后年化超额 ≥ **+3%**
2. NW(5) t ≥ **2.0**
3. 分年 ≥ **7/9**
4. 平均同时持有 ≥ **2 个行业**（否则等于押单一行业）
5. 🚨 **两个安慰剂都要过**：
   - **P1 行业打乱**：保持每日事件总数不变、把事件的行业标签随机重分配 ⇒ 检验「聚集在某个行业」
     这个结构本身有没有信息（不是行业在涨）。
   - **P2 日期打乱**：保持每个行业的事件总数不变、把日期随机重分配 ⇒ 检验「时点」有没有信息。
   要求 Δ ≥ **+1.0pp** 且 p < 0.10，两个都要。

用法
----
    python -m scripts.evaluation.event_industry_probe
    python -m scripts.evaluation.event_industry_probe --k 20 --m 3 --hold 20
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation import event_driven_probe as ED
from scripts.evaluation.event_chain_probe import raw_inc_pure

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "event_industry_probe"
PQ = Path("D:/data/parquet")

MIN_AVG_IND = 2.0        # 平均同时持有行业数下限
MIN_DELTA = 0.010        # 安慰剂净提升门槛


# ============================================================ 行业标签（PIT）
def industry_label(cal: pd.DatetimeIndex, codes: pd.Index) -> pd.DataFrame:
    """date × code → 申万一级行业名；按 in_date/out_date 逐段填充（PIT）。"""
    ind = pd.read_parquet(PQ / "industry_classification_level1.parquet",
                          columns=["code", "industry_name", "in_date", "out_date"])
    ind["code"] = ED._sym(ind["code"])
    ind["in_d"] = pd.to_datetime(ind["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    ind["out_d"] = pd.to_datetime(ind["out_date"].astype(str), format="%Y%m%d", errors="coerce")
    ind = ind.dropna(subset=["in_d"]).sort_values("in_d")

    lab = pd.DataFrame(np.nan, index=cal, columns=codes, dtype=object)
    colset = set(codes)
    for code, name, i_d, o_d in ind[["code", "industry_name", "in_d", "out_d"]].itertuples(index=False):
        if code not in colset:
            continue
        m = (cal >= i_d) & (cal < o_d if pd.notna(o_d) else True)
        if m.any():
            lab.loc[m, code] = name
    return lab


def industry_daily_ret(close: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """date × industry → 该行业成分股等权日收益（用 PIT 标签）。"""
    r = close.pct_change()
    long = pd.DataFrame({"r": r.stack(), "ind": lab.stack()}).dropna()
    if long.empty:
        return pd.DataFrame()
    d = long.index.get_level_values(0)
    out = long.groupby([d, "ind"])["r"].mean().unstack()
    return out.sort_index()


# ============================================================ 信号
def event_tags(ev: pd.DataFrame, lab: pd.DataFrame) -> pd.Series:
    """每个事件在 avail 当日所属的申万一级行业（PIT）。"""
    tags = []
    for a, c in zip(ev["avail"], ev["code"]):
        try:
            tags.append(lab.at[a, c])
        except KeyError:
            tags.append(np.nan)
    return pd.Series(tags, index=ev.index, name="ind")


def signals(ev: pd.DataFrame, lab: pd.DataFrame, cal: pd.DatetimeIndex,
            k: int, m: int) -> pd.DataFrame:
    """返回 DataFrame[industry, i_sig]：边缘触发的行业密集期起点。

    若 ``ev`` 已带 ``ind`` 列（安慰剂用），直接用它，不再查标签——
    这样安慰剂可以保持事件总数不变，只替换「事件→行业」的配对。
    """
    if "ind" in ev.columns:
        e2 = ev.dropna(subset=["ind"])
    else:
        e2 = ev.assign(ind=event_tags(ev, lab)).dropna(subset=["ind"])
    if e2.empty:
        return pd.DataFrame(columns=["industry", "i_sig"])

    daily = (e2.groupby(["avail", "ind"]).size()
                .unstack(fill_value=0).reindex(cal).fillna(0.0))
    roll = daily.rolling(k, min_periods=k).sum()
    trig = (roll >= m) & (roll.shift(1) < m)

    out = []
    for ind in trig.columns:
        ii = np.flatnonzero(trig[ind].values)
        out += [{"industry": ind, "i_sig": int(i)} for i in ii]
    return pd.DataFrame(out)


def build_windows(sig: pd.DataFrame, cal: pd.DatetimeIndex, hold: int) -> pd.DataFrame:
    """信号 → 持有区间 [i0, i1]；同行业重叠延长、不重复建仓。"""
    rows = []
    last = {}
    for ind, i in sig.sort_values("i_sig").itertuples(index=False):
        i0 = i + 1
        if i0 >= len(cal):
            continue
        i1 = min(i0 + hold, len(cal) - 1)
        if ind in last and i0 <= last[ind]["i1"]:
            last[ind]["i1"] = max(last[ind]["i1"], i1)
            continue
        w = {"industry": ind, "i0": i0, "i1": i1}
        rows.append(w)
        last[ind] = w
    return pd.DataFrame(rows)


def run_ind_portfolio(win: pd.DataFrame, ind_ret: pd.DataFrame,
                      cal: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """行业篮子日收益：每日对所有活跃行业等权。"""
    live = defaultdict(list)
    cols = {c: j for j, c in enumerate(ind_ret.columns)}
    for ind, i0, i1 in win[["industry", "i0", "i1"]].itertuples(index=False):
        j = cols.get(ind)
        if j is None:
            continue
        for i in range(i0, i1 + 1):
            live[i].append(j)
    v = ind_ret.reindex(cal).values
    ret = np.full(len(cal), np.nan)
    cnt = np.zeros(len(cal), dtype=int)
    for i in range(1, len(cal)):
        js = live.get(i)
        if not js:
            continue
        x = v[i, np.array(js, dtype=int)]
        x = x[np.isfinite(x)]
        if x.size:
            ret[i] = float(x.mean())
            cnt[i] = int(x.size)
    return pd.Series(ret, index=cal), pd.Series(cnt, index=cal)


def evaluate(win: pd.DataFrame, ind_ret: pd.DataFrame, cal: pd.DatetimeIndex,
             bench: pd.Series, hold: int) -> dict:
    if len(win) == 0:
        return {}
    ret, cnt = run_ind_portfolio(win, ind_ret, cal)
    pos = win.assign(code=win["industry"])[["code", "i0", "i1"]]
    st = ED.summarize(ret, pos, cal, hold, bench.reindex(cal).fillna(0.0), cnt)
    if not st:
        return {}
    st["avg_ind"] = float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 0.0
    st["n_win"] = len(win)
    return st


# ============================================================ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20, help="滚动窗口（交易日）")
    ap.add_argument("--m", type=int, default=3, help="窗口内最少事件数（行业·次）")
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--placebo", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    close, openp, mask, bench, mcap = ED.load_base()
    cal = pd.DatetimeIndex(close.index)
    ev = raw_inc_pure(cal)
    print(f"[base] 事件 {len(ev)} 条 / {ev['code'].nunique()} 只   {cal[0].date()} ~ {cal[-1].date()}")

    lab = industry_label(cal, close.columns)
    cover = lab.iloc[-1].notna().mean()
    print(f"[base] 行业标签覆盖率（末期）{cover:.1%}   行业数 {lab.iloc[-1].nunique()}")
    ind_ret = industry_daily_ret(close, lab)
    print(f"[base] 行业日收益矩阵 {ind_ret.shape}")

    # ---- 主信号
    sig = signals(ev, lab, cal, args.k, args.m)
    win = build_windows(sig, cal, args.hold)
    print(f"[sig] K={args.k} M={args.m} → 信号 {len(sig)} 个 → 持有窗口 {len(win)} 段")
    st = evaluate(win, ind_ret, cal, bench, args.hold)
    if not st:
        print("[FATAL] 无有效窗口")
        return
    print(f"[sig] 扣费 {st['ann_ex_net']:+.2%}/年  t {st['t']:.2f}  Sharpe {st['sharpe']:.2f}  "
          f"分年 {st['yr_pos']}  均持行业 {st['avg_ind']:.1f}  换手 {st['turnover']:.1f}x")

    # ---- 参数敏感性（不选最好，只看是否单调/稳定）
    print("\n" + "=" * 84)
    print("一、参数敏感性（K × M，扣费年化超额）")
    print("=" * 84)
    grid_rows = []
    print(f"{'K\\M':>6}" + "".join(f"{m:>10}" for m in (2, 3, 5, 8)))
    for k in (10, 20, 40):
        cells = []
        for m in (2, 3, 5, 8):
            s2 = signals(ev, lab, cal, k, m)
            w2 = build_windows(s2, cal, args.hold)
            e2 = evaluate(w2, ind_ret, cal, bench, args.hold)
            if e2:
                cells.append(f"{e2['ann_ex_net']:>10.2%}")
                grid_rows.append({"k": k, "m": m, **e2})
            else:
                cells.append(f"{'—':>10}")
        print(f"{k:>6}" + "".join(cells))

    # ---- 安慰剂
    print("\n" + "=" * 84)
    print(f"二、安慰剂（{args.placebo} 次）")
    print("=" * 84)
    rng = np.random.default_rng(20261116)

    # P1 行业打乱：事件总数、日期、行业分布都不变，只把「事件→行业」的配对随机重排。
    # 这是核心安慰剂——检验「聚集在某个行业」这个结构本身有没有信息。
    ev_ind = ev.assign(ind=event_tags(ev, lab))
    p1 = []
    for _ in range(args.placebo):
        sh = ev_ind.copy()
        sh["ind"] = rng.permutation(sh["ind"].values)
        s2 = signals(sh, lab, cal, args.k, args.m)
        w2 = build_windows(s2, cal, args.hold)
        e2 = evaluate(w2, ind_ret, cal, bench, args.hold)
        if e2:
            p1.append(e2["ann_ex_net"])
    p1 = np.array(p1)

    # P2 日期平移：整批事件在时间轴上随机平移 60~480 个交易日（保持事件结构与密集形态，
    # 只改变「什么时候发生」）。检验信号是否只是「某些年份的行业行情」。
    n_cal = len(cal)
    ci0 = cal.get_indexer(ev["avail"])
    p2 = []
    for _ in range(args.placebo):
        sh = rng.integers(60, 481) * rng.choice([-1, 1])
        ci = ci0 + sh
        keep = (ci >= 0) & (ci < n_cal)
        if keep.sum() < 0.8 * len(ci0):        # 平移到样本外的太多，重抽一次
            continue
        sh_df = ev.loc[keep].assign(avail=cal[ci[keep]])
        s2 = signals(sh_df, lab, cal, args.k, args.m)
        w2 = build_windows(s2, cal, args.hold)
        e2 = evaluate(w2, ind_ret, cal, bench, args.hold)
        if e2:
            p2.append(e2["ann_ex_net"])
    p2 = np.array(p2)

    rows = []
    for name, arr in (("P1 行业打乱", p1), ("P2 日期打乱", p2)):
        if len(arr) < 5:
            print(f"  {name:<12} 安慰剂次数不足")
            continue
        mu = float(arr.mean())
        delta = st["ann_ex_net"] - mu
        pv = float((arr >= st["ann_ex_net"]).mean())
        ok = (delta >= MIN_DELTA) and (pv < 0.10)
        print(f"  {name:<12} 真实 {st['ann_ex_net']:+.2%}  安慰剂均值 {mu:+.2%}  "
              f"Δ {delta:+.2%}  p {pv:.2f}  ⇒ {'✅ 过' if ok else '❌ 未过'}")
        rows.append({"placebo": name, "real": st["ann_ex_net"], "mean": mu,
                     "delta": delta, "p": pv, "pass": bool(ok)})

    ok_n = st["avg_ind"] >= MIN_AVG_IND
    ok_t = st["t"] >= 2.0
    ok_y = int(st["yr_pos"].split("/")[0]) / int(st["yr_pos"].split("/")[1]) >= 7 / 9 - 1e-9
    ok_p = all(r["pass"] for r in rows) if rows else False
    print(f"\n  判据：扣费 {st['ann_ex_net']:+.2%}（≥3%）{'✅' if st['ann_ex_net']>=0.03 else '❌'}  "
          f"t {st['t']:.2f}（≥2）{'✅' if ok_t else '❌'}  分年 {st['yr_pos']}（≥7/9）"
          f"{'✅' if ok_y else '❌'}  均持行业 {st['avg_ind']:.1f}（≥2）{'✅' if ok_n else '❌'}  "
          f"安慰剂 {'✅' if ok_p else '❌'}")
    print(f"  ⇒ 总判定：{'通过' if (st['ann_ex_net']>=0.03 and ok_t and ok_y and ok_n and ok_p) else '不通过'}")

    pd.DataFrame([{**st, "k": args.k, "m": args.m}]).to_csv(
        OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(grid_rows).to_csv(OUT / "grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT / "placebo.csv", index=False, encoding="utf-8-sig")
    win.to_parquet(OUT / "windows.parquet")
    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
