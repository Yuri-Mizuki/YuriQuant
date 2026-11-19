"""事件链与事件前状态：增持的「来历」是否决定收益（2026-11-16）
================================================================

为什么做这一层
--------------
`inc_pure` 只问了「增持这件事有没有信息」，没有问「这个增持是怎么来的」。
同一句「董监高增持 3,000 万元」，在两种情形下语义完全不同：

- 某股东三个月前刚减持过、现在又买回来 ⇒ **波段操作/换手**，他看的是价差不是价值；
- 某股东五年没动过、突然在下跌 30% 后自掏腰包 ⇒ **真金白银的表态**。

本脚本检验两件事（L3 事件链 + L2 事件前状态，一次跑完）：

**A. 事件链：增持前有没有减持**
  A1/A2  前 180 日同公司有任何减持 / 无减持
  A3     前 365 日同公司无减持（更严格的「干净」口径）
  B1     前 365 日**同一股东**减持过（最精确的「同一人来回做」）

**B. 事件前状态：公告前价格在哪**
  C1~C4  前 60 日收益分四档（跌>20% / 跌0~20% / 涨0~20% / 涨>20%）
  C5     距 250 日高点回撤 >30%（深跌区）

判据（先定后算，不做事后挑格子）
--------------------------------
1. 扣费后年化超额 ≥ **+3%**
2. NW(5) t ≥ **2.0**
3. 分年 ≥ **7/9** 为正
4. 子集持仓 ≥ **200 笔** 且平均同时持仓 ≥ **5 只**（否则不可操作）
5. 🚨 **相对 base 的提升必须超过「同规模随机剔除」的安慰剂基线**（默认 20 次）。
   任何"剔除/换股"类改造都会因为减少样本而改变年化，不减安慰剂就是把换股噪声当 alpha。
   判据 = 子集超额 − 安慰剂均值超额 ≥ **+1.0pp** 且安慰剂 p 值 < 0.10。

用法
----
    python -m scripts.evaluation.event_chain_probe                  # 全部子集，H=20
    python -m scripts.evaluation.event_chain_probe --hold 10
    python -m scripts.evaluation.event_chain_probe --placebo 40
"""
from __future__ import annotations

import argparse
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation import event_driven_probe as ED

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "event_chain_probe"
PQ = Path("D:/data/parquet")

def avail_session(cal: pd.DatetimeIndex, d: pd.Series) -> pd.Series:
    """日期 → 日历上「当天或之后第一个交易日」。

    与 ``ED._avail`` 逻辑一致，但先把精度对齐到 ``cal.unit``——
    pandas 2.x 读 parquet 得到 datetime64[us]，而 ``to_datetime`` 默认 ns，
    直接 searchsorted 会抛 "Cannot losslessly cast ... ns to us"。

    ⚠️ 早于样本起点的事件必须置 NaT（不能顺延到首日），同 ``ED._avail``。
    """
    dv = pd.DatetimeIndex(pd.to_datetime(d, errors="coerce")).as_unit(cal.unit)
    pos = cal.searchsorted(dv.values, side="left")
    out = pd.Series(pd.NaT, index=d.index, dtype=f"datetime64[{cal.unit}]")
    ok = (pos < len(cal)) & dv.notna() & (dv.values >= cal[0].to_datetime64())
    out.loc[ok] = cal[pos[ok]]
    return out


MIN_N_POS = 200          # 子集最少持仓笔数
MIN_AVG_NAME = 5.0       # 子集平均同时持仓下限
MIN_DELTA = 0.010        # 相对安慰剂的净提升门槛（+1.0pp）


# ============================================================ 事件明细（保留 person）
def raw_inc_pure(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """与 `fam_inc_pure` 完全同口径，但保留 person/ann 以构造事件链。

    口径：方向含「增持」+ 成交方式∈竞价/二级/大宗 + relation=本人 + 比例 ≥0.1%。
    """
    df = pd.read_parquet(PQ / "alt_cninfo_holder.parquet",
                         columns=["code", "ann_date", "direction", "chg_ratio",
                                  "reason", "relation", "person"])
    df = df[df["direction"].astype(str).str.contains("增持")]
    df = df[df["reason"].astype(str).str.contains("|".join(ED._ACTIVE), na=False)]
    df = df[df["relation"].astype(str).isin(["本人"])]
    df["code"] = ED._sym(df["code"])
    df["r"] = pd.to_numeric(df["chg_ratio"], errors="coerce").abs()
    df["ann"] = pd.to_datetime(df["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df[df["r"] >= 0.1].copy()
    df = df.assign(avail=avail_session(cal, df["ann"])).dropna(subset=["avail", "ann"])
    df["person"] = df["person"].astype(str).str.strip()
    return df[["code", "ann", "avail", "person", "r"]]


def red_index(by_person: bool) -> dict:
    """({code 或 (code,person)} → 排序后的减持公告日列表)。"""
    red = pd.read_parquet(PQ / "alt_cninfo_holder.parquet",
                          columns=["code", "ann_date", "direction", "person"])
    red = red[red["direction"].astype(str).str.contains("减持")]
    red["code"] = ED._sym(red["code"])
    red["ann"] = pd.to_datetime(red["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
    red = red.dropna(subset=["ann"])
    if by_person:
        red["person"] = red["person"].astype(str).str.strip()
        keys = list(zip(red["code"], red["person"]))
    else:
        keys = list(red["code"])
    out: dict = {}
    for k, d in zip(keys, red["ann"]):
        out.setdefault(k, []).append(d)
    for k in out:
        out[k].sort()
    return out


def _hit(idx: dict, key, ann: pd.Timestamp, days: int) -> bool:
    """key 在 (ann - days, ann) 区间内是否有记录（严格早于本次增持）。"""
    arr = idx.get(key)
    if not arr:
        return False
    lo = ann - pd.Timedelta(days=days)
    i = bisect.bisect_left(arr, lo)
    return i < len(arr) and arr[i] < ann


# ============================================================ 事件前价格状态
def price_state(ev: pd.DataFrame, close: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """公告前的价格位置，全部用 ``avail`` **之前**的收盘价（PIT，零前视）。

    ``ret60``  = 前 60 个交易日累计收益
    ``dd250``  = 相对前 250 日最高收盘的回撤（≤0）
    """
    ci = cal.get_indexer(ev["avail"])
    cj = close.columns.get_indexer(ev["code"])
    cv = close.values
    n = len(cal)
    r60 = np.full(len(ev), np.nan)
    dd = np.full(len(ev), np.nan)
    for k, (i, j) in enumerate(zip(ci, cj)):
        if i <= 0 or j < 0:
            continue
        end = i - 1                       # avail 的前一交易日
        beg60 = max(0, end - 60)
        if end - beg60 >= 50:             # 至少 50 个观测才算
            p0, p1 = cv[beg60, j], cv[end, j]
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                r60[k] = p1 / p0 - 1.0
        beg250 = max(0, end - 250)
        win = cv[beg250:end + 1, j]
        win = win[np.isfinite(win)]
        if len(win) >= 200 and win.size and win.max() > 0:
            dd[k] = cv[end, j] / win.max() - 1.0
    return ev.assign(ret60=r60, dd250=dd)


# ============================================================ 回测一个事件子集
def eval_subset(ev: pd.DataFrame, close, openp, mask, mcap, cal, hold: int) -> dict:
    if len(ev) == 0:
        return {}
    pos = ED.build_positions(ev[["code", "avail", "strength"]] if "strength" in ev else
                             ev.assign(strength=1.0)[["code", "avail", "strength"]],
                             cal, hold, close.columns, close, mask)
    if len(pos) == 0:
        return {}
    ret, cnt = ED.run_portfolio(pos, close, openp, mask, cal)
    bh = ED.bench_for_holdings(pos, close, mcap).reindex(cal).fillna(0.0)
    st = ED.summarize(ret, pos, cal, hold, bh, cnt)
    if not st:
        return {}
    st["n_pos"] = len(pos)
    st["avg_names"] = float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 0.0
    return st


# ============================================================ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--placebo", type=int, default=20, help="随机剔除安慰剂次数")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    close, openp, mask, bench, mcap = ED.load_base()
    cal = pd.DatetimeIndex(close.index)
    ev = raw_inc_pure(cal)
    print(f"[base] 事件明细 {len(ev)} 条 / {ev['code'].nunique()} 只   "
          f"{cal[0].date()} ~ {cal[-1].date()}  H={args.hold}")

    # ---- 构造分组
    idx_all = red_index(by_person=False)
    idx_per = red_index(by_person=True)
    ev = ev.assign(
        red180=[_hit(idx_all, c, a, 180) for c, a in zip(ev["code"], ev["ann"])],
        red365=[_hit(idx_all, c, a, 365) for c, a in zip(ev["code"], ev["ann"])],
        per365=[_hit(idx_per, (c, p), a, 365)
                for c, p, a in zip(ev["code"], ev["person"], ev["ann"])],
    )
    ev = price_state(ev, close, cal)
    ev = ev.assign(strength=ev["r"])

    print(f"[base] 前 180 日有减持 {ev.red180.mean():.1%}  前 365 日有减持 {ev.red365.mean():.1%}  "
          f"同一股东 365 日减持 {ev.per365.mean():.1%}")
    print(f"[base] ret60 中位 {ev.ret60.median():+.1%}   dd250 中位 {ev.dd250.median():+.1%}")

    groups = {
        "base 全体": ev,
        "A1 前180日无减持": ev[~ev["red180"]],
        "A2 前180日有减持": ev[ev["red180"]],
        "A3 前365日无减持": ev[~ev["red365"]],
        "B1 同一股东365日减持过": ev[ev["per365"]],
        "B2 同一股东365日未减持": ev[~ev["per365"]],
        "C1 ret60 ≤ −20%": ev[ev["ret60"] <= -0.20],
        "C2 ret60 (−20%, 0]": ev[(ev["ret60"] > -0.20) & (ev["ret60"] <= 0)],
        "C3 ret60 (0, +20%]": ev[(ev["ret60"] > 0) & (ev["ret60"] <= 0.20)],
        "C4 ret60 > +20%": ev[ev["ret60"] > 0.20],
        "C5 dd250 ≤ −30%": ev[ev["dd250"] <= -0.30],
    }

    # ---- 逐组回测
    rows = []
    for name, sub in groups.items():
        st = eval_subset(sub, close, openp, mask, mcap, cal, args.hold)
        if not st:
            print(f"  {name:<24} 无有效持仓，跳过")
            continue
        rows.append({"subset": name, "n_event": len(sub), **st})
    base = rows[0]
    print("\n" + "=" * 96)
    print("一、各子集表现（扣费口径：单边 13.5bp）")
    print("=" * 96)
    print(f"{'子集':<24}{'事件':>6}{'持仓':>6}{'均持':>6}{'毛超额':>9}{'扣费':>9}{'t':>7}"
          f"{'Sharpe':>8}{'换手':>7}{'分年':>7}")
    for r in rows:
        print(f"{r['subset']:<24}{r['n_event']:>6}{r['n_pos']:>6}{r['avg_names']:>6.1f}"
              f"{r['ann_ex']:>9.2%}{r['ann_ex_net']:>9.2%}{r['t']:>7.2f}"
              f"{r['sharpe']:>8.2f}{r['turnover']:>7.1f}{r['yr_pos']:>7}")

    # ---- 安慰剂：把 base 随机剔到同样规模
    print("\n" + "=" * 96)
    print(f"二、相对 base 的提升 vs 同规模随机剔除（{args.placebo} 次安慰剂）")
    print("=" * 96)
    rng = np.random.default_rng(20261116)
    placebos = {}
    for r in rows[1:]:
        k = r["n_event"]
        if k >= len(ev):
            continue
        sims = []
        for _ in range(args.placebo):
            pick = rng.choice(len(ev), size=k, replace=False)
            st = eval_subset(ev.iloc[pick], close, openp, mask, mcap, cal, args.hold)
            if st:
                sims.append(st["ann_ex_net"])
        placebos[r["subset"]] = np.array(sims)

    print(f"{'子集':<24}{'扣费超额':>10}{'安慰剂均值':>12}{'Δ':>9}{'p':>8}  判定")
    out_rows = []
    for r in rows[1:]:
        p = placebos.get(r["subset"])
        if p is None or len(p) < 5:
            print(f"{r['subset']:<24}{r['ann_ex_net']:>10.2%}{'—':>12}{'—':>9}{'—':>8}"
                  f"  安慰剂次数不足（{0 if p is None else len(p)} < 5）")
            continue
        pm = float(np.mean(p))
        delta = r["ann_ex_net"] - pm
        pv = float((p >= r["ann_ex_net"]).mean())
        ok_pl = (delta >= MIN_DELTA) and (pv < 0.10)
        ok_n = (r["n_pos"] >= MIN_N_POS) and (r["avg_names"] >= MIN_AVG_NAME)
        ok_t = r["t"] >= 2.0
        ok_y = int(r["yr_pos"].split("/")[0]) / int(r["yr_pos"].split("/")[1]) >= 7 / 9 - 1e-9
        good = ok_pl and ok_n and ok_t and ok_y
        tag = ("✅ 通过" if good else
               "边缘" if (delta > 0 and ok_n) else
               "无效" if not ok_n else "无增量")
        print(f"{r['subset']:<24}{r['ann_ex_net']:>10.2%}{pm:>12.2%}{delta:>9.2%}{pv:>8.2f}  "
              f"{tag}" + ("" if ok_n else "（样本不足）"))
        out_rows.append({"subset": r["subset"], "n_event": r["n_event"], "n_pos": r["n_pos"],
                         "avg_names": r["avg_names"], "ann_ex": r["ann_ex"],
                         "ann_ex_net": r["ann_ex_net"], "t": r["t"], "sharpe": r["sharpe"],
                         "turnover": r["turnover"], "yr_pos": r["yr_pos"],
                         "placebo_mean": pm, "delta": delta, "p": pv, "judge": tag})

    pd.DataFrame(rows).to_csv(OUT / "subsets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(out_rows).to_csv(OUT / "placebo.csv", index=False, encoding="utf-8-sig")
    ev.to_parquet(OUT / "events.parquet")

    # ---- 单调性：ret60 四档
    print("\n" + "=" * 96)
    print("三、ret60 单调性（若真有效，应随「跌得越深」单调更强，而不是某一格突然好）")
    print("=" * 96)
    for r in rows:
        if r["subset"].startswith("C") and "ret60" in r["subset"]:
            print(f"  {r['subset']:<24} n={r['n_event']:>5}  扣费 {r['ann_ex_net']:>7.2%}  "
                  f"t {r['t']:>5.2f}")

    print(f"\n[out] {OUT}")


if __name__ == "__main__":
    main()
