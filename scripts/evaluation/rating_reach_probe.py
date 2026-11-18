"""卖方评级的「决策触及度」与天花板（诊断，不是新因子，2026-11-15）
=====================================================================

背景
----
`reports/em_rating_probe/README.md` 把 8 个评级构造全判负，理由都是**截面 IC 太小**
（最好残差 IC 0.0096，t 1.74）。但那套测法隐含一个假设：评级应当像量价因子一样
在全市场做截面排序。它本身是「每季约 120 只股票发生净变动」的稀疏事件，用月度截面
秩 IC 去测，命中率天然被稀释。

本脚本换一个问法，**不做新因子、不挑格子**：

> 就算评级能被完美预测，它对 K15 主模型最多值多少？

四个诊断，从「能不能用」问到「值多少」：

A. **触及度 reach**：K15 的 15 个仓位里，评级数据能说上话的有几个？
   看板：有覆盖比例、有净变动比例、净下调比例。若持仓里几乎无人被覆盖，
   那所有 IC 讨论都是空的。

B. **组合内区分度**：已经持有的股票里，当时可见「被下调」的那几只，下月是不是真的更差？
   这是存量检验——不换股、不加权，只问评级在 K15 内部有没有排序能力。

C. **oracle 天花板**：假设能 100% 预测「下月会被净下调」（上帝视角），
   把这些股票从 K15 里剔除，历史月均超额能提升多少 pp？
   这是**上限**：任何真实信号都不可能超过它。天花板接近 0 ⇒ 这条线在结构上关闭，
   不是「信号不够强」，而是「没有决策可以被它改动」。

D. **硬否决（PIT 合法）**：把软加权换成硬否决——凡当时可见净下调的股票直接从候选池剔除。
   之前测的是 `mid13 + λ·dev` 软加权（λ=0.1/0.2），覆盖 28%、变动 9.4%，软加权
   根本改不动排名，等于没测。否决法才是这种「排除型」信号的自然用法。

输出
----
``reports/em_rating_probe/reach/``：``holdings.csv``（持仓级明细）、
``reach.csv``（A）、``inside.csv``（B）、``ceiling.csv``（C/D）。

用法
----
    python -m scripts.evaluation.rating_reach_probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation.rating_probe import (
    BASE, W, K, BUF, load_events, map_to_trading_days, load_main_signal,
    month_pairs, nw_t,
)

OUT = Path("reports/em_rating_probe/reach")


# ------------------------------------------------------------------ 持仓路径
def holdings(sig: pd.DataFrame, pairs: list[tuple], cal: pd.DatetimeIndex,
             openp: pd.DataFrame, mask: pd.DataFrame,
             k: int = K, buf: int = BUF) -> tuple[np.ndarray, pd.DataFrame]:
    """复现 focus 执行口径（K15 + buffer30），并保留每期持仓与个股收益。"""
    ex, hold, recs = [], None, []
    for t, t2 in pairs:
        if t not in sig.index:
            continue
        i, j = cal.get_loc(t), cal.get_loc(t2)
        if j + 1 >= len(cal):
            continue
        r = (openp.loc[cal[j + 1]] / openp.loc[cal[i + 1]] - 1).replace([np.inf, -np.inf], np.nan)
        s = sig.loc[t]
        ok = s.notna() & r.notna() & mask.loc[t].reindex(s.index).fillna(False).astype(bool)
        if ok.sum() < k:
            continue
        s2 = s.where(ok)
        uni = r[ok].mean()
        rank = s2.rank(ascending=False)
        keep = [c for c in (hold or []) if rank.get(c, 1e9) <= buf]
        add = [c for c in s2.nlargest(k + buf).index if c not in keep][:k - len(keep)]
        pk = keep + add
        ex.append((r[pk].mean() - uni) * 100)
        recs.append(dict(t=t, t2=t2, excess=(r[pk].mean() - uni) * 100, uni=uni,
                         n_cand=int(ok.sum()), ret={c: float(r[c]) for c in pk}))
        hold = pk
    return np.asarray(ex), pd.DataFrame(recs)


# ------------------------------------------------------------------ 窗口聚合
def agg(ev: pd.DataFrame, avail: np.ndarray, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """avail ∈ (lo, hi] 的事件按 code 聚合为 (n, up, dn)。"""
    a = np.searchsorted(avail, np.datetime64(lo), side="right")
    b = np.searchsorted(avail, np.datetime64(hi), side="right")
    w = ev.iloc[a:b]
    if w.empty:
        return pd.DataFrame(columns=["n", "up", "dn"])
    w = w.assign(up=(w["delta"] > 0).astype("int8"), dn=(w["delta"] < 0).astype("int8"))
    return w.groupby("code").agg(n=("delta", "size"), up=("up", "sum"), dn=("dn", "sum"))


def long_holdings(h: pd.DataFrame, ev: pd.DataFrame, avail: np.ndarray) -> pd.DataFrame:
    """持仓 long 化，并挂上「t 时可见」与「t→t2 未来」两组评级计数。"""
    rows = []
    for rec in h.itertuples():
        for c, rr in rec.ret.items():
            rows.append((rec.t, rec.t2, c, rr, rec.uni))
    hl = pd.DataFrame(rows, columns=["t", "t2", "code", "ret", "uni"])
    hl["ex"] = (hl["ret"] - hl["uni"]) * 100

    vis, nxt = [], []
    for t, t2 in hl[["t", "t2"]].drop_duplicates().itertuples(index=False):
        a = agg(ev, avail, t - pd.Timedelta(days=W), t)
        b = agg(ev, avail, t, t2)
        for d, store in ((a, vis), (b, nxt)):
            d = d.copy()
            d["t"] = t
            store.append(d.reset_index())
    v = pd.concat(vis, ignore_index=True) if vis else pd.DataFrame(columns=["t", "code", "n", "up", "dn"])
    n = pd.concat(nxt, ignore_index=True) if nxt else pd.DataFrame(columns=["t", "code", "n", "up", "dn"])
    v = v.rename(columns={"n": "n_vis", "up": "up_vis", "dn": "dn_vis"})
    n = n.rename(columns={"n": "n_next", "up": "up_next", "dn": "dn_next"})
    hl = hl.merge(v, on=["t", "code"], how="left").merge(n, on=["t", "code"], how="left")
    for col in ["n_vis", "up_vis", "dn_vis", "n_next", "up_next", "dn_next"]:
        hl[col] = hl[col].fillna(0)
    hl["net_vis"] = hl["up_vis"] - hl["dn_vis"]
    hl["net_next"] = hl["up_next"] - hl["dn_next"]
    return hl


# ------------------------------------------------------------------ A/B/C/D
def reach_tab(hl: pd.DataFrame) -> pd.DataFrame:
    """A：每个调仓日的触及度。"""
    g = hl.groupby("t")
    out = pd.DataFrame({
        "n_hold": g.size(),
        "n_cover": g["n_vis"].apply(lambda s: (s > 0).sum()),
        "n_move": g["net_vis"].apply(lambda s: (s != 0).sum()),
        "n_down": g["dn_vis"].apply(lambda s: (s > 0).sum()),
        "n_up": g["up_vis"].apply(lambda s: (s > 0).sum()),
        "excess": g["ex"].mean(),
    })
    out["p_cover"] = out["n_cover"] / out["n_hold"]
    out["p_move"] = out["n_move"] / out["n_hold"]
    out["p_down"] = out["n_down"] / out["n_hold"]
    return out


def inside_tab(hl: pd.DataFrame) -> pd.DataFrame:
    """B：K15 内部按可见评级净变动分组，比较个股次月超额。"""
    def bucket(r) -> str:
        if r["n_vis"] == 0:
            return "nocover"
        if r["net_vis"] < 0:
            return "down"
        if r["net_vis"] > 0:
            return "up"
        return "flat"

    hl = hl.copy()
    hl["grp"] = hl.apply(bucket, axis=1)
    rows = []
    for name, sub in hl.groupby("grp"):
        m, t = nw_t(sub["ex"].values)
        rows.append(dict(grp=name, n_obs=len(sub), n_month=sub["t"].nunique(),
                         ex_bp=m * 100, t=t))
    # 配对：down 月均 vs up 月均 / flat 月均
    pm = hl.pivot_table(index="t", columns="grp", values="ex", aggfunc="mean")
    for a, b in (("down", "up"), ("down", "flat"), ("up", "flat")):
        if a in pm and b in pm:
            d = (pm[a] - pm[b]).dropna()
            m, t = nw_t(d.values)
            rows.append(dict(grp=f"{a}−{b}", n_obs=len(d), n_month=len(d),
                             ex_bp=m * 100, t=t))
    return pd.DataFrame(rows)


def ceiling(hl: pd.DataFrame, mid13: pd.DataFrame, pairs: list[tuple],
            cal: pd.DatetimeIndex, openp: pd.DataFrame, mask: pd.DataFrame,
            base_ex: np.ndarray, k: int = K) -> pd.DataFrame:
    """C/D：把「下月将被下调」（oracle）或「当时可见已下调」（PIT）设为惩罚项，重跑 K15。"""
    cols = mid13.columns

    def penalized(flag: pd.Series, col: str) -> pd.DataFrame:
        """flag: MultiIndex (t, code) bool → 惩罚矩阵。"""
        m = pd.DataFrame(False, index=mid13.index, columns=cols)
        idx = flag[flag].index
        for t, c in idx:
            if t in m.index and c in m.columns:
                m.at[t, c] = True
        return mid13 - 1e6 * m.astype(float)

    down_next = hl.set_index(["t", "code"])["dn_next"].gt(0)
    down_vis = hl.set_index(["t", "code"])["dn_vis"].gt(0)
    up_next = hl.set_index(["t", "code"])["up_next"].gt(0)

    rows = []
    pairs_c = [(a, b) for a, b in pairs if a in mid13.index]
    for name, flag in (("C1_oracle_drop_down", down_next),
                       ("C2_oracle_add_up", up_next),
                       ("D_pit_veto_down", down_vis)):
        sig = penalized(flag, name)
        e = holdings(sig, pairs_c, cal, openp, mask, k)[0]
        n = min(len(e), len(base_ex))
        d = e[:n] - base_ex[:n]
        m, t = nw_t(d)
        rows.append(dict(arm=name, mean_ex=float(np.mean(e[:n])), delta_pp=m, t=t, n=n,
                         n_flag=int(flag.sum())))
    return pd.DataFrame(rows)


def cap_gap(hl: pd.DataFrame, mcap: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """E：持仓股与「有评级覆盖股」的市值分位差异——解释为什么触及度这么低。"""
    rows = []
    for t, sub in hl.groupby("t"):
        mc = mcap.loc[t].reindex(close.columns)
        q = mc.rank(pct=True)
        held = q.reindex(sub["code"]).dropna()
        cov = q.reindex(sub.loc[sub["n_vis"] > 0, "code"]).dropna()
        rows.append(dict(t=t, q_hold=held.mean(), q_cover=cov.mean() if len(cov) else np.nan,
                         n_cover=len(cov)))
    d = pd.DataFrame(rows).dropna()
    m, tt = nw_t((d["q_hold"] - d["q_cover"]).values)
    return pd.DataFrame([dict(q_hold=d["q_hold"].mean(), q_cover=d["q_cover"].mean(),
                              gap_pp=m * 100, t=tt, n_month=len(d))])


def main() -> None:
    close = pd.read_parquet(f"{BASE}/close_adj.parquet")
    openp = pd.read_parquet(f"{BASE}/open_adj.parquet")
    mask = pd.read_parquet(f"{BASE}/tradable_mask.parquet").fillna(False).astype(bool)
    mcap = pd.read_parquet(f"{BASE}/market_cap.parquet")
    mid13 = load_main_signal(mask)
    cal = close.index

    ev = map_to_trading_days(load_events(), cal).sort_values("avail_day").reset_index(drop=True)
    avail = ev["avail_day"].values

    pairs = [(a, b) for a, b in month_pairs(cal) if a >= pd.Timestamp("2018-01-01")]
    pairs_c = [(a, b) for a, b in pairs if a in mid13.index]
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- K 扫描：触及度是否随仓位放宽而变得可用
    kscan = []
    for kk in (15, 25, 45):
        b_ex, hk = holdings(mid13, pairs_c, cal, openp, mask, kk)
        hlk = long_holdings(hk, ev, avail)
        Ak = reach_tab(hlk)
        Ck = ceiling(hlk, mid13, pairs_c, cal, openp, mask, b_ex, kk)
        c1 = Ck[Ck.arm == "C1_oracle_drop_down"].iloc[0]
        m, t = nw_t(b_ex)
        kscan.append(dict(k=kk, base_ex=m, base_t=t, n=len(b_ex),
                          n_cover=Ak["n_cover"].mean(), p_cover=Ak["p_cover"].mean(),
                          n_move=Ak["n_move"].mean(), n_down=Ak["n_down"].mean(),
                          p_down=Ak["p_down"].mean(),
                          c1_hits=int(c1["n_flag"]), c1_delta=c1["delta_pp"], c1_t=c1["t"]))
        if kk == K:
            base_ex, h, A, C = b_ex, hk, Ak, Ck
    ks = pd.DataFrame(kscan)
    print("\nK 扫描（触及度 vs 仓位数）")
    print(ks.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    ks.to_csv(OUT / "kscan.csv", index=False, encoding="utf-8")

    m0, t0 = nw_t(base_ex)
    print(f"\n基线 mid13 K15 月均超额 {m0:+.3f}%  t {t0:+.2f}  n={len(base_ex)}")

    hl = long_holdings(h, ev, avail)
    hl.to_csv(OUT / "holdings.csv", index=False, encoding="utf-8", float_format="%.6f")

    A = reach_tab(hl)
    print("\nA 触及度（K15 持仓内，均值/中位）")
    for c in ["n_hold", "n_cover", "n_move", "n_down", "n_up", "p_cover", "p_move", "p_down"]:
        print(f"  {c:<10} {A[c].mean():.3f}   {A[c].median():.3f}")

    B = inside_tab(hl)
    print("\nB 组合内区分度（个股次月超额，bp）")
    print(B.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nC/D 换用法")
    print(C.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # oracle 个股层面：下月被下调的持仓股，次月是不是真的更差
    d = hl[hl["dn_next"] > 0]["ex"].dropna()
    o = hl[hl["dn_next"] == 0]["ex"].dropna()
    mm, tt = nw_t(
        (hl[hl["dn_next"] > 0].groupby("t")["ex"].mean()
         - hl[hl["dn_next"] == 0].groupby("t")["ex"].mean()).dropna().values)
    print(f"\noracle 个股层：下月被下调的持仓 {len(d)} 个观测，超额 {d.mean()*100:+.1f}bp；"
          f"其余 {o.mean()*100:+.1f}bp；月均差 {mm*100:+.1f}bp  t {tt:+.2f}")

    A.to_csv(OUT / "reach.csv", encoding="utf-8")
    B.to_csv(OUT / "inside.csv", index=False, encoding="utf-8")
    C.to_csv(OUT / "ceiling.csv", index=False, encoding="utf-8")

    E = cap_gap(hl, mcap, close)
    print("\nE 市值分位（解释触及度）：持仓 vs 持仓中有覆盖者")
    print(E.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    E.to_csv(OUT / "cap_gap.csv", index=False, encoding="utf-8")
    pd.DataFrame([dict(arm="base_mid13", mean_ex=m0, delta_pp=0.0, t=t0, n=len(base_ex),
                       n_flag=0)]).to_csv(OUT / "base.csv", index=False, encoding="utf-8")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
