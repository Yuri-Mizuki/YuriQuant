"""`im_*` 短周期因子的**周频**组合检验（PIT 动态池）
==================================================

为什么必须新跑
--------------
既有 `reports/solo_weekly/` 的「周频判负」测的是 **`solo` 的月级慢信号按周调仓**
（`gbdt_mid13` 是 882 个中长期因子、horizon 2:1:1:2，刷新率本身是月级）——
那是**期限错配**，不是「短周期因子在周频失效」。该报告自己写明：

> 周频方向从因子层封死：现成 944 个中性化因子里没有短周期原料

`im_*` 恰恰是 **h1 短周期**因子（5 分钟日内结构 → 次日收益），**从未在周频测过**。
现在 PIT 池（421 只）分钟数据已齐，可以正面回答：

**Q：把 h1 短周期因子放到周频持有，是比日频更好还是更差？**

为什么预期可能更好
------------------
日频的死刑判在**成本**：0.86 日换手，往返 54bp 即转负（`pit_intraday_probe` §6.5）。
周频换手天然只有日频的 1/5 ⇒ 同样的信号强度，成本从 ~29%/年 降到 ~6%/年。
代价是信号衰减：h1 因子的 IC 衰减若很快，周频拿不到 5 天的收益。

所以本实验要同时看**两个数**：换手降了多少、毛收益掉了多少。净收益是二者之差。

口径
----
- 池：PIT 动态池（`load_daily` + `apply_membership_mask`，每日在册 300 只）
- 信号：**调仓日 T 收盘可得**的日内特征（只用 T 日及之前的 5 分钟数据）
- 收益：T+1 open 买 → T+1+H open 卖（H ∈ {1, 5, 10, 20} 交易日）
   ⇒ 周频 H=5、双周 H=10、月频 H=20；日频 H=1 作对照
- 调仓日：每 H 个交易日一次（等间隔，不用自然周以避开节假日错位）
- 成本：单边 13.5bp（往返 27bp），按调仓日扣
- 判据（**跑之前定**）：周频要算「可用」，需净年化 ≥ 日频的 9.87% **且** 换手显著更低；
  若净年化反而更低，则「周频救不了」得到证实

用法::

    python -m scripts.evaluation.pit_weekly_probe --stage all
    python -m scripts.evaluation.pit_weekly_probe --stage compare   # 只跑频率对照
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

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation.pit_intraday_probe import (  # noqa: E402
    ONE_WAY_BP,
    _orient,
    build_features,
    ic_stats,
    load_context,
)

log = setup_logging("pit_weekly_probe")

OUT = ROOT / "reports" / "intraday_daily"
PPY = 252


def daily_ic(p: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """逐日截面 spearman IC（**向量化**）。

    🚨 必须用向量化版：``pit_intraday_probe.daily_ic`` 是逐日 Python 循环，
    单次约 8s；本轮「20 个 horizon × 95 因子」= 1900 次 ⇒ **4 小时**（实测跑满
    20 分钟才出 0 行）。这里用 ``rank + corrwith``，单次 0.07s，实测与循环版
    逐位相等（max|Δ|=2.8e-17）。
    """
    idx = p.index.intersection(fwd.index)
    cols = p.columns.intersection(fwd.columns)
    a, b = p.loc[idx, cols], fwd.loc[idx, cols]
    m = a.notna() & b.notna()
    ok = m.sum(axis=1) >= 30
    c = a.where(m).rank(axis=1).corrwith(b.where(m).rank(axis=1), axis=1)
    return c.where(ok).dropna()


# ----------------------------------------------------------------------
# 多期限收益
# ----------------------------------------------------------------------
def fwd_returns(daily: pd.DataFrame, horizons: list[int]) -> dict[int, pd.DataFrame]:
    """T+1 open 买 → T+1+H open 卖，H ∈ horizons。"""
    op = daily.reset_index().pivot(index="date", columns="code",
                                   values="open").sort_index()
    buy = op.shift(-1)
    out = {}
    for h in horizons:
        sell = op.shift(-(1 + h))
        out[h] = (sell / buy - 1.0).replace([np.inf, -np.inf], np.nan)
    return out


# ----------------------------------------------------------------------
# 等间隔调仓回测
# ----------------------------------------------------------------------
def run_periodic(sig: pd.DataFrame, fwd: pd.DataFrame, hold: int,
                 top: int = 10, buffer_mult: float = 2.0,
                 cost_bp: float = ONE_WAY_BP, min_names: int = 30) -> dict:
    """每 ``hold`` 个交易日调仓一次的 TopN 等权 + buffer。

    🚨 关键点：**只在调仓日看信号**（T 日收盘），持有 hold 个交易日。
    两次调仓之间的空档用**上一次的持仓**，不能每天重新排序——否则就退化成日频。

    单期收益取 hold 日累计（fwd[hold] 已是 T+1→T+1+hold 的 open-to-open），
    所以每个调仓日的收益直接就是那一期的持有收益。
    """
    sig = sig.reindex(columns=fwd.columns)
    idx = sig.index.intersection(fwd.index)
    if len(idx) < hold * 2:
        return {"gross": pd.Series(dtype=float), "net": pd.Series(dtype=float),
                "turn": pd.Series(dtype=float), "rebal": pd.DatetimeIndex([])}

    s, r = sig.loc[idx], fwd.loc[idx]
    rebal = idx[::hold]           # 等间隔调仓日
    prev: set[str] = set()
    rets, turns = [], []
    for dt in rebal:
        sv = s.loc[dt].dropna()
        if len(sv) < min_names:
            continue
        rank = sv.rank(ascending=False, method="first")
        head = rank.sort_values()
        keep = head[head <= top * buffer_mult].index
        held = [c for c in prev if c in keep]
        new = [c for c in head.index if c not in held]
        picked = (held + new)[:top]
        if not picked:
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

    gross = pd.Series(rets, index=rebal[:len(rets)])
    net = gross - pd.Series(turns, index=gross.index) * cost_bp / 1e4
    return {"gross": gross.dropna(), "net": net.dropna(),
            "turn": pd.Series(turns, index=gross.index),
            "rebal": gross.index}


def stats_periodic(x: pd.Series, hold: int) -> dict:
    """按持有期折算年化：每年 252/hold 期。"""
    x = x.dropna()
    if len(x) < 12:
        return {}
    ppy = PPY / hold
    nav = (1 + x).cumprod()
    cagr = float(nav.iloc[-1] ** (ppy / len(x)) - 1)
    dd = float((nav / nav.cummax() - 1).min())
    sd = float(x.std(ddof=1))
    sr = float(x.mean() / sd * np.sqrt(ppy)) if sd > 0 else float("nan")
    return {"ann": cagr, "sr": sr, "mdd": dd, "n_period": len(x)}


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="im_* 短周期因子的周频检验")
    ap.add_argument("--stage", default="all", choices=["all", "compare", "ic"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--buffer", type=float, default=2.0)
    ap.add_argument("--horizons", default="1,5,10,20")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    horizons = [int(x) for x in args.horizons.split(",")]
    store, codes, daily, _fwd1 = load_context()
    days = store.all_days()
    feats = build_features(store, days[0], days[-1], with_lag=False)

    fwds = fwd_returns(daily, horizons)
    pool_cols = [c for c in store.codes if c in fwds[horizons[0]].columns]
    log.info("可用列 %d", len(pool_cols))

    feats = {k: v.reindex(columns=pool_cols) for k, v in feats.items()}

    # ---- 各期限下的因子 IC（信号衰减的直接度量）----
    # 🚨 缓存必须**按 horizon 校验覆盖**，不能只看文件在不在：
    # 加了新 horizon（如 2/3/4）但复用旧缓存时，新 horizon 的行根本不存在，
    # 下游 B 臂 `ic_tab[ic_tab["horizon"]==h]` 会取到空表 → `continue` 静默跳过，
    # 表现为「加了 horizon 但输出里没有它」，不报错。
    ic_path = OUT / "pit_weekly_ic.csv"
    cached = pd.read_csv(ic_path) if ic_path.exists() else None
    have = set(cached["horizon"].unique()) if cached is not None else set()
    if cached is not None and set(horizons) <= have:
        ic_tab = cached
        log.info("复用 IC 表缓存（%d 行，覆盖 %s）", len(ic_tab), sorted(have))
    else:
        if cached is not None:
            log.info("IC 缓存缺 horizon %s ⇒ 全量重算", sorted(set(horizons) - have))
        ic_rows = []
        for h in horizons:
            fwd = fwds[h]
            for name, p in feats.items():
                st = ic_stats(daily_ic(p, fwd))
                if st:
                    ic_rows.append({"horizon": h, "name": name, **st})
        ic_tab = pd.DataFrame(ic_rows)
        ic_tab.to_csv(ic_path, index=False, encoding="utf-8-sig")

    # 锚因子（H=1 最强，与日频实验同源，保证可比）
    base1 = ic_tab[ic_tab["horizon"] == 1].copy()
    base1 = base1.reindex(base1["t"].abs().sort_values(ascending=False).index)
    anchor = base1.iloc[0]["name"]
    log.info("锚因子 %s（H=1 t=%.2f）", anchor, base1.iloc[0]["t"])

    # 信号衰减表：锚因子 + 前 10 个在 4 个期限上的 IC
    top10 = base1.head(10)
    print("\n=== 信号衰减：同一批因子的 IC 随持有期变化 ===")
    decay = []
    for name in top10["name"]:
        row = {"name": name}
        for h in horizons:
            sub = ic_tab[(ic_tab["horizon"] == h) & (ic_tab["name"] == name)]
            row[f"H{h}"] = float(sub.iloc[0]["ic"]) if len(sub) else np.nan
        decay.append(row)
    dtab = pd.DataFrame(decay)
    with pd.option_context("display.width", 180,
                           "display.float_format", "{:.4f}".format):
        print(dtab.to_string(index=False))

    if args.stage == "ic":
        return

    # ---- 频率对照：单因子，各持有期 ----
    # 🚨 必须定向：IC<0 是反转因子，直接「买最高分」等于买反了。
    # （第一版漏了这步，日频毛 −33.36% 与 §6.3 的 +47.21% 完全对不上——方向反了。）
    ic_anchor = float(base1.iloc[0]["ic"])
    siga = _orient(feats[anchor], ic_anchor)
    log.info("锚因子定向: IC=%.4f ⇒ %s", ic_anchor,
             "取负（反转）" if ic_anchor < 0 else "原样（延续）")
    rows = []
    for h in horizons:
        res = run_periodic(siga, fwds[h], hold=h, top=args.top,
                           buffer_mult=args.buffer)
        g = stats_periodic(res["gross"], h)
        n = stats_periodic(res["net"], h)
        if not g or not n:
            continue
        rows.append({"horizon": h, "label": {1: "日频", 5: "周频", 10: "双周",
                                             20: "月频"}.get(h, f"H{h}"),
                     "gross_ann": g["ann"], "net_ann": n["ann"], "sr": n["sr"],
                     "mdd": n["mdd"], "n_period": n["n_period"],
                     "turn_per": float(res["turn"].mean()),
                     "turn_ann": float(res["turn"].mean()) * PPY / h})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "pit_weekly_freq.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 频率对照 A · 固定用 H1 锚因子 {anchor} · top{args.top} · 27bp ===")
    print("   （这一臂故意固定因子，只变持有期 ⇒ 度量「信号衰减」，不是最优周频）")
    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(t.to_string(index=False))

    # ---- 对照 B：**每个期限各用自己最强的因子**（这才是公平的周频检验）----
    # 上面 A 臂固定 H1 锚因子，但 H1 的强因子在 H5+ 会衰减（mom_last_bar IC −61%），
    # 而 H5/H10/H20 各有自己的强因子（波动率结构 / 流动性结构）。用错因子会把
    # 「因子选错」误判成「周频不行」——这正是记忆里「期限错配」的同型错误。
    rows_b = []
    for h in horizons:
        sub = ic_tab[ic_tab["horizon"] == h].copy()
        if sub.empty:
            continue
        sub = sub.reindex(sub["t"].abs().sort_values(ascending=False).index)
        best = sub.iloc[0]
        sig_h = _orient(feats[best["name"]], float(best["ic"]))
        res = run_periodic(sig_h, fwds[h], hold=h, top=args.top,
                           buffer_mult=args.buffer)
        g = stats_periodic(res["gross"], h)
        n = stats_periodic(res["net"], h)
        if not g or not n:
            continue
        rows_b.append({"horizon": h,
                       "label": {1: "日频", 5: "周频", 10: "双周",
                                 20: "月频"}.get(h, f"H{h}"),
                       "factor": best["name"], "ic": float(best["ic"]),
                       "gross_ann": g["ann"], "net_ann": n["ann"], "sr": n["sr"],
                       "mdd": n["mdd"], "turn_ann": float(res["turn"].mean()) * PPY / h})
    tb = pd.DataFrame(rows_b)
    tb.to_csv(OUT / "pit_weekly_freq_best.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 频率对照 B · 各期限用各自最强因子（公平版） · top{args.top} · 27bp ===")
    with pd.option_context("display.width", 220,
                           "display.float_format", "{:.4f}".format):
        print(tb.to_string(index=False))

    # ---- 复合在各频率下是否成立（Q1 内二级排序）----
    from scripts.evaluation.pit_intraday_probe import composite_2stage
    ics1 = dict(zip(base1["name"], base1["ic"], strict=True))
    t10 = base1.head(10)["name"].tolist()
    rows2 = []
    for h in horizons:
        s = composite_2stage(anchor, t10, feats, ics1, q=0.1)
        res = run_periodic(s, fwds[h], hold=h, top=args.top,
                           buffer_mult=args.buffer)
        g = stats_periodic(res["gross"], h)
        n = stats_periodic(res["net"], h)
        if not g or not n:
            continue
        rows2.append({"horizon": h, "gross_ann": g["ann"], "net_ann": n["ann"],
                      "sr": n["sr"], "mdd": n["mdd"],
                      "turn_ann": float(res["turn"].mean()) * PPY / h})
    t2 = pd.DataFrame(rows2)
    t2.to_csv(OUT / "pit_weekly_composite.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 复合（Q1 内二级排序，锚+H1前10）在各频率下 · top{args.top} · 27bp ===")
    print("   （注意：这里锚因子和成员都是 H1 挑的，会让长持有期吃亏——见下 C 臂）")
    with pd.option_context("display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(t2.to_string(index=False))

    # ---- 对照 C：**每个期限各用自己期限的强因子做复合**（最公平）----
    rows_c = []
    for h in horizons:
        sub = ic_tab[ic_tab["horizon"] == h].copy()
        if sub.empty:
            continue
        sub = sub[sub["t"].abs() > 1.96]
        sub = sub.reindex(sub["t"].abs().sort_values(ascending=False).index)
        if len(sub) < 3:
            continue
        ics_h = dict(zip(sub["name"], sub["ic"], strict=True))
        anchor_h = sub.iloc[0]["name"]
        members = sub.head(10)["name"].tolist()
        s = composite_2stage(anchor_h, members, feats, ics_h, q=0.1)
        res = run_periodic(s, fwds[h], hold=h, top=args.top,
                           buffer_mult=args.buffer)
        g = stats_periodic(res["gross"], h)
        n = stats_periodic(res["net"], h)
        if not g or not n:
            continue
        rows_c.append({"horizon": h,
                       "label": {1: "日频", 5: "周频", 10: "双周",
                                 20: "月频"}.get(h, f"H{h}"),
                       "anchor": anchor_h, "n_sig": int(len(sub)),
                       "gross_ann": g["ann"], "net_ann": n["ann"], "sr": n["sr"],
                       "mdd": n["mdd"], "turn_ann": float(res["turn"].mean()) * PPY / h})
    tc = pd.DataFrame(rows_c)
    tc.to_csv(OUT / "pit_weekly_composite_best.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== 对照 C · 各期限用各自强因子复合（最公平） · top{args.top} · 27bp ===")
    with pd.option_context("display.width", 230,
                           "display.float_format", "{:.4f}".format):
        print(tc.to_string(index=False))

    json.dump({"anchor": anchor, "top": args.top, "horizons": horizons},
              open(OUT / "pit_weekly_meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()