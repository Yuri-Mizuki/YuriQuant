"""日频策略幸存者偏差对照 —— 同一因子、同一执行，只换评估池。

背景
----
`reports/intraday_daily/` 的日频结论（净 16.5% / SR 0.721）跑在
`factor_library/hs300_2022_2025` 的 **339 只静态快照**上。报告自己标注了
「含幸存者偏差，真实全 A 动态池会更差」，但**从没量过差多少**。本脚本量它。

为什么不用 `im_*` 直接补
-----------------------
`min5_hs300.parquet` **只有 339 只**（就是那个静态池），**全 A 分钟数据不存在**
⇒ 95 个 `im_*` 日内因子在物理上无法在全 A 重算。硬试过日线代理
`close/open-1`，与 `im_mom_last_bar` 逐日秩相关仅 **0.081**（`proxy_check.json`）
⇒ 代理不成立，弃用。

正确设计：换因子，不换问题
------------------------
`factor_library/all_a_2018_2026` 有 **879 个日频因子**，面板本就覆盖全 A。
选经济含义与 `im_mom_last_bar`（日内反转）同类的因子，在**同一策略**上跑三个池：

- `alpha158_LOW0`（h1 IC **+0.0719**，量价族最强之一）= 当日最低价在日内区间的
  相对位置，本质是「日内超跌 → 次日回补」，与 `im_mom_last_bar` 同族。
- `alpha191_070`（h1 IC **−0.0649**）= 另一支日内反转，用于稳健性。

⚠️ **能回答与不能回答**
- ✅ 能答：**「日频反转」这个策略类型**在静态池 vs 动态池上差多少 pp。
- ❌ 不能答：95 个 `im_*` 因子在全 A 上还剩多少 —— 那需要全 A 分钟数据。

三个池（同时间窗 / 同执行 / 同成本）
----------------------------------
1. `static339`：库里那 339 只静态快照 ⇒ **偏差的起点**
2. `pit_hs300`：**PIT 动态成分**（`index_constituent_000300_SH` 按 in/out_date 逐日
   展开，含 82 只被剔除股）⇒ 隔离「静态快照」这一层
3. `all_a`：全 A 可交易池（`tradable_mask`）⇒ 偏差方向的上界

用法:
    python -m scripts.evaluation.daily_survivorship_probe --check-proxy
    python -m scripts.evaluation.daily_survivorship_probe --factor alpha158_LOW0
    python -m scripts.evaluation.daily_survivorship_probe --scan
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:/YuriQuant")
BASE = ROOT / "reports/alla_rolling/_base"
LIB_STATIC = Path(r"E:/data/factor_library/hs300_2022_2025")
LIB_ALLA = Path(r"E:/data/factor_library/all_a_2018_2026")
CONS = Path(r"E:/data/parquet/index_constituent_000300_SH.parquet")
OUT = ROOT / "reports/intraday_daily"

ONE_WAY_BP = 13.5          # 与 intraday_daily_probe 一致（27bp 往返）

DEFAULT_FACTORS = ["alpha158_LOW0", "alpha191_070", "alpha191_095", "alpha158_KLEN"]


def load_panels() -> dict[str, pd.DataFrame]:
    return {
        "open": pd.read_parquet(BASE / "open_adj.parquet").sort_index(),
        "close": pd.read_parquet(BASE / "close_adj.parquet").sort_index(),
        "mask": pd.read_parquet(BASE / "tradable_mask.parquet").sort_index(),
        "mcap": pd.read_parquet(BASE / "market_cap.parquet").sort_index(),
    }


def build_pools(static_codes: list[str], days: pd.DatetimeIndex,
                mask: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """三个池的逐日布尔掩码（index=days, columns=全A 代码）。

    ⚠️ 注意 `all_a` 包含 4800+ 只（含仙股/微盘），与 HS300 的 top10 在
    **容量意义上不是同一策略**（实测换手 0.975、净 −82%）。公平对照见
    `--mcap-q`（按当日市值分位截断）。
    """
    cols = mask.columns
    msk = mask.reindex(index=days, columns=cols).fillna(False).astype(bool)
    pools = {}

    # 1) 静态 339 —— 偏差起点
    keep = [c for c in static_codes if c in cols]
    m = pd.DataFrame(False, index=days, columns=cols)
    m[keep] = True
    pools["static339"] = m & msk

    # 2) PIT HS300 —— 按 in/out_date 逐日展开（含被剔除股）
    cons = pd.read_parquet(CONS)
    cons["in_dt"] = pd.to_datetime(cons["in_date"], format="%Y%m%d", errors="coerce")
    cons["out_dt"] = pd.to_datetime(cons["out_date"], format="%Y%m%d", errors="coerce")
    m2 = pd.DataFrame(False, index=days, columns=cols)
    for _, r in cons.iterrows():
        c = r["con_code"]
        if c not in cols:
            continue
        a = max(r["in_dt"], days.min()) if pd.notna(r["in_dt"]) else days.min()
        b = min(r["out_dt"], days.max()) if pd.notna(r["out_dt"]) else days.max()
        sel = (days >= a) & (days <= b)
        if sel.any():
            m2.loc[sel, c] = True
    pools["pit_hs300"] = m2 & msk

    # 3) 全 A —— 偏差上界（含仙股，容量不可比）
    pools["all_a"] = msk
    return pools


def apply_mcap_floor(pool: pd.DataFrame, mcap: pd.DataFrame,
                     q: float = 0.5) -> pd.DataFrame:
    """按当日市值分位截断 —— 让全 A 池与 HS300 在「容量」上可比。

    HS300 约等于市值前 300/5000 ≈ 前 6%，但日频 top10 在小盘股上冲击成本极大，
    直接跑全 A 会退化成「买最便宜的仙股」（实测净 −82%）。这里用 ``q`` 截断
    下界，``q=0.5`` 保留中位以上，``q=0.8`` 保留前 20%。
    """
    mc = mcap.reindex(index=pool.index, columns=pool.columns)
    thr = mc.quantile(q, axis=1)
    keep = mc.ge(thr, axis=0)
    return pool & keep.fillna(False)


def fwd_return(op: pd.DataFrame, period_days: int = 1) -> pd.DataFrame:
    """T+1 open 买 → T+1+period open 卖（与 intraday_daily 完全一致）。

    🚨 前视边界：信号 T 日收盘才可得 ⇒ 买入价必须是 T+1 开盘。
    """
    buy = op.shift(-1)
    sell = op.shift(-(1 + period_days))
    return (sell / buy - 1.0).replace([np.inf, -np.inf], np.nan)


def load_factor(name: str) -> tuple[pd.DataFrame, bool]:
    """返回 (因子面板, higher_is_better)。

    🚨 **必须全 A 库优先（2026-11-16 踩坑）**：`alpha191_070` 在**两个库里都有** ——
    静态库 `hs300_2022_2025` 的 1123×**421**，全 A 库 `all_a_2018_2026` 的 2511×**5832**。
    先查静态库会拿到 421 列的小面板，`uni_mean` 被压到 297，池对照直接失效。

    全 A 库的 ``panel_path`` 列**全为空**（888/888），实际落在 ``panels/<name>.parquet``；
    静态库有 ``panel_path``。两条通道分别处理。
    """
    reg_a = pd.read_csv(LIB_ALLA / "registry.csv")
    row_a = reg_a[reg_a["name"] == name]
    p_a = LIB_ALLA / "panels" / f"{name}.parquet"
    if not row_a.empty and p_a.exists():
        hib = bool(row_a.iloc[0].get("ic_mean_h1", np.nan) > 0)
        return pd.read_parquet(p_a), hib

    reg = pd.read_csv(LIB_STATIC / "registry.csv")
    row = reg[reg["name"] == name]
    if not row.empty:
        p = Path(str(row.iloc[0]["panel_path"]))
        return pd.read_parquet(p), bool(row.iloc[0].get("ic_mean_h1", np.nan) > 0)

    raise KeyError(f"未知因子 {name!r}")


def run(sig: pd.DataFrame, fwd: pd.DataFrame, pool: pd.DataFrame,
        top: int = 10, buffer_mult: float = 2.0,
        cost_bp: float = ONE_WAY_BP, min_names: int = 30,
        higher_is_better: bool = False) -> dict:
    """日频等权 TopN + buffer。higher_is_better 由因子 h1 IC 符号决定。

    🚨 **列对齐（2026-11-16 踩坑）**：全 A 因子面板 5832 列、`tradable_mask` 5826 列，
    列序不同。直接 `sig.loc[dt].where(pool.loc[dt])` 会**按位置**对齐 ⇒ 掩码张冠李戴，
    冠冕堂皇地跑出 `uni_mean=297`（应为 5000+）。必须先 `reindex(columns=pool.columns)`。
    """
    idx = sig.index
    asc = not higher_is_better
    sig = sig.reindex(columns=pool.columns)
    prev: set[str] = set()
    rets, turns, nhold, nuni = [], [], [], []
    for dt in idx:
        avail = pool.loc[dt]
        sv = sig.loc[dt].where(avail).dropna()
        nuni.append(len(sv))
        if len(sv) < min_names:
            rets.append(np.nan)
            turns.append(0.0)
            nhold.append(0)
            continue
        rank = sv.rank(ascending=asc, method="first")
        head = rank.sort_values(ascending=True)      # 名次 1 = 最优
        keep = head[head <= top * buffer_mult].index
        held = [c for c in prev if c in keep]
        new = [c for c in head.index if c not in held]
        picked = (held + new)[:top]
        if not picked:
            rets.append(np.nan)
            turns.append(0.0)
            nhold.append(0)
            continue
        w = pd.Series(1.0 / len(picked), index=picked)
        wp = (pd.Series(1.0 / len(prev), index=sorted(prev))
              if prev else pd.Series(dtype=float))
        allc = w.index.union(wp.index)
        dw = w.reindex(allc, fill_value=0) - wp.reindex(allc, fill_value=0)
        turn = float(dw.abs().sum() / 2)
        rr = fwd.loc[dt, picked].dropna()
        rets.append(float(rr.mean()) if len(rr) else np.nan)
        turns.append(turn)
        nhold.append(len(picked))
        prev = set(picked)
    gross = pd.Series(rets, index=idx)
    net = gross - pd.Series(turns, index=idx) * cost_bp / 1e4
    return {"gross": gross.dropna(), "net": net.dropna(),
            "turn": pd.Series(turns, index=idx),
            "nhold": pd.Series(nhold, index=idx),
            "nuni": pd.Series(nuni, index=idx)}


def stats(x: pd.Series, ppy: int = 252) -> dict:
    x = x.dropna()
    if len(x) < 30:
        return {}
    nav = (1 + x).cumprod()
    cagr = nav.iloc[-1] ** (ppy / len(x)) - 1
    vol = x.std(ddof=1) * np.sqrt(ppy)
    dd = (nav / nav.cummax() - 1).min()
    return {"ann": cagr, "vol": vol, "sr": x.mean() / x.std(ddof=1) * np.sqrt(ppy),
            "mdd": dd, "n": len(x), "calmar": cagr / abs(dd) if dd < 0 else np.nan}


def pool_audit(static_codes: list[str]) -> dict:
    """量化静态池漏掉了什么 —— 这是偏差的来源证据。"""
    cons = pd.read_parquet(CONS)
    cons["in_dt"] = pd.to_datetime(cons["in_date"], format="%Y%m%d", errors="coerce")
    cons["out_dt"] = pd.to_datetime(cons["out_date"], format="%Y%m%d", errors="coerce")
    d0, d1 = pd.Timestamp("2022-01-04"), pd.Timestamp("2026-06-30")
    act = cons[(cons["in_dt"] <= d1) & (cons["out_dt"].isna() | (cons["out_dt"] >= d0))]
    hist = set(act["con_code"])
    have = set(static_codes)
    missing = sorted(hist - have)

    # 漏掉股的收益 vs 在库股收益（累计，同窗口）
    cl = pd.read_parquet(BASE / "close_adj.parquet").sort_index()
    cl = cl.loc[(cl.index >= d0) & (cl.index <= d1)]

    def cum(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 20 else np.nan

    r_in = cl[[c for c in have if c in cl.columns]].apply(cum).dropna()
    r_miss = cl[[c for c in missing if c in cl.columns]].apply(cum).dropna()
    out = {
        "hist_codes": len(hist), "in_lib": len(have & hist), "missing": len(missing),
        "missing_share": len(missing) / len(hist) if hist else np.nan,
        "ret_in_median": float(r_in.median()), "ret_in_mean": float(r_in.mean()),
        "ret_miss_median": float(r_miss.median()), "ret_miss_mean": float(r_miss.mean()),
        "gap_median_pp": float((r_in.median() - r_miss.median()) * 100),
        "n_in": int(len(r_in)), "n_miss": int(len(r_miss)),
    }
    try:
        from scipy import stats as st
        out["mannwhitney_p"] = float(st.mannwhitneyu(r_in, r_miss).pvalue)
    except Exception:
        out["mannwhitney_p"] = None
    return out


def check_proxy() -> None:
    """日线代理可信度（存档证据，结论：不成立，弃用）。"""
    reg = pd.read_csv(LIB_STATIC / "registry.csv")
    sig = pd.read_parquet(reg[reg["name"] == "im_mom_last_bar"].iloc[0]["panel_path"])
    p = load_panels()
    proxy = (p["close"] / p["open"] - 1.0).replace([np.inf, -np.inf], np.nan)
    idx = sig.index.intersection(proxy.index)
    cols = sig.columns.intersection(proxy.columns)
    a = sig.loc[idx, cols].rank(axis=1)
    b = proxy.loc[idx, cols].rank(axis=1)
    d = pd.DataFrame({"a": a.stack(), "b": b.stack()}).dropna()
    daily = d.groupby(level=0).apply(lambda g: g["a"].corr(g["b"]))
    out = {"daily_mean": float(daily.mean()), "daily_median": float(daily.median()),
           "pooled": float(d["a"].corr(d["b"], method="spearman")),
           "n_days": int(len(daily)),
           "verdict": "代理不成立（相关性过低），弃用；改用全A库日频同类因子"}
    print("=== 日线代理可信度（close/open-1 vs im_mom_last_bar） ===")
    print(f"  逐日截面秩相关 均值 {out['daily_mean']:.4f} 中位 {out['daily_median']:.4f}")
    print(f"  pooled spearman {out['pooled']:.4f}  n_days {out['n_days']}")
    print(f"  ⇒ {out['verdict']}")
    json.dump(out, open(OUT / "proxy_check.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", default="alpha158_LOW0")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--buffer", type=float, default=2.0)
    ap.add_argument("--start", default="2022-01-04")
    ap.add_argument("--end", default="2026-06-30")
    ap.add_argument("--check-proxy", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--mcap-q", type=float, default=None,
                    help="全A 池按当日市值分位截断（0.5=中位以上，用于容量可比）")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.check_proxy:
        check_proxy()
        return

    reg_s = pd.read_csv(LIB_STATIC / "registry.csv")
    static_codes = list(pd.read_parquet(
        reg_s[reg_s["name"] == "im_mom_last_bar"].iloc[0]["panel_path"]).columns)

    if args.audit:
        a = pool_audit(static_codes)
        json.dump(a, open(OUT / "pool_audit.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("=== 静态池漏掉了什么 ===")
        print(f"  窗口内曾属 HS300：{a['hist_codes']} 只；库里有 {a['in_lib']} 只；"
              f"漏掉 {a['missing']} 只（{a['missing_share']*100:.1f}%）")
        print(f"  在库股累计收益 中位 {a['ret_in_median']*100:.2f}% / "
              f"均值 {a['ret_in_mean']*100:.2f}%  (n={a['n_in']})")
        print(f"  漏掉股累计收益 中位 {a['ret_miss_median']*100:.2f}% / "
              f"均值 {a['ret_miss_mean']*100:.2f}%  (n={a['n_miss']})")
        print(f"  中位差 {a['gap_median_pp']:+.2f}pp   "
              f"Mann-Whitney p = {a['mannwhitney_p']}")
        return

    p = load_panels()
    days = p["open"].index
    days = days[(days >= pd.Timestamp(args.start)) & (days <= pd.Timestamp(args.end))]
    op = p["open"].loc[days]
    fwd = fwd_return(op, 1)
    pools = build_pools(static_codes, days, p["mask"])
    if args.mcap_q is not None:
        pools["all_a"] = apply_mcap_floor(pools["all_a"], p["mcap"], args.mcap_q)
        pools["pit_hs300"] = apply_mcap_floor(pools["pit_hs300"], p["mcap"], args.mcap_q)

    factors = DEFAULT_FACTORS if args.scan else [args.factor]
    allrows = []
    for fac in factors:
        try:
            sig, hib = load_factor(fac)
        except KeyError as e:
            print(f"  跳过 {fac}: {e}")
            continue
        sig = sig.reindex(index=days)
        rows = []
        for name, pool in pools.items():
            res = run(sig, fwd, pool, top=args.top, buffer_mult=args.buffer,
                      higher_is_better=hib)
            g, n = stats(res["gross"]), stats(res["net"])
            if not g or not n:
                continue
            rows.append({"factor": fac, "pool": name,
                         "uni_mean": float(res["nuni"].mean()),
                         "gross_ann": g["ann"], "net_ann": n["ann"],
                         "sr": n["sr"], "mdd": n["mdd"],
                         "turn_day": float(res["turn"].mean()),
                         "hold": float(res["nhold"].replace(0, np.nan).mean())})
        t = pd.DataFrame(rows)
        allrows.append(t)
        base = t[t["pool"] == "static339"]
        if not base.empty:
            b = base.iloc[0]
            print(f"\n=== {fac} (higher_is_better={hib}) · top{args.top} "
                  f"buffer{args.buffer} · 27bp ===")
            print(t.to_string(index=False, float_format=lambda x: "%.4f" % x))
            for _, r in t.iterrows():
                if r["pool"] == "static339":
                    continue
                print(f"  {r['pool']:>11}: 净 {(r['net_ann']-b['net_ann'])*100:+6.2f}pp  "
                      f"SR {r['sr']-b['sr']:+.3f}  MDD {r['mdd']*100:+.1f}%")

    if allrows:
        res_all = pd.concat(allrows, ignore_index=True)
        tag = "scan" if args.scan else args.factor
        res_all.to_csv(OUT / f"survivorship_{tag}.csv", index=False,
                       encoding="utf-8-sig")
        json.dump({"start": args.start, "end": args.end, "top": args.top,
                   "buffer": args.buffer, "factors": factors},
                  open(OUT / "survivorship_meta.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()