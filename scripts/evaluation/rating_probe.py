"""卖方评级族因子探针（预注册判据，2026-11-15）
=================================================

背景
----
`reports/new_source_scan/README.md` 判定：本地原始数据已榨干、量价/文本/分钟频全部判负，
**唯一还站着的「新数据源」是东财个股研报的评级族**（评级 / 评级变动 / 覆盖度）。
2026-10-13 的 `consensus_availability` 只否掉了 EPS 预测字段（``predict*`` 仅 2024-04 起），
评级字段实测 2017 起、逐年覆盖 0.86~1.00，被连带否掉。

本探针回答一个问题：**评级族在现有 882/920 因子之外，有没有可交易的增量？**

因子（全部月度、PIT 滞后 1 交易日、无事件记 NaN）
-------------------------------------------------
======================  ==========================================================  ==========
因子                     定义（窗口 W=91 自然日，回溯端 t）                                 稀疏度
======================  ==========================================================  ==========
``rtg_rev_net``          (n_up − n_down) / n_rep，n_up/n_down 按 (机构,股票) 自算 delta      中
``rtg_delta_mean``       窗口内 delta 的均值（只统计有前值的报告）                            中
``rtg_first_share``      首次覆盖占比（该机构对该股无前值）                                  低
``rtg_level``            窗口内评级序数均值（买入 3 / 增持 2 / 持有 1 / 中性 0）              低
``rtg_disp``             窗口内跨机构评级序数标准差（需 ≥2 家）                              高
``cov_level``            log1p(窗口内报告数)                                                 低
``cov_chg``              log1p(近 91 日报告数) − log1p(前 91 日报告数)                       中
``tgt_upside``           平均目标价 / 收盘 − 1（覆盖仅 5~31%）                              极高
======================  ==========================================================  ==========

🚨 **评级变动方向一律自算，不用接口的 ``ratingChange``**
实测该列编码与 ``lastEmRatingName → emRatingName`` 的方向对不上（2024-06：``ratingChange=2``
有 166/166 条是「上次为空」即首次覆盖；``ratingChange=1`` 出现 买入→增持 这类下调）。
按 ``(org_code, code)`` 全历史序列 ``rating_value.diff()`` 重算，口径自洽且 PIT 安全。

评价（预注册，三条全过才进池）
------------------------------
1. **残差 IC**：月度截面把因子秩对 ``gbdt__h1`` 秩回归取残差，再与前瞻月收益秩相关，
   均值 |t(NW)| ≥ 3.0；
2. **分年同号**：逐年残差 IC 符号一致 ≥ 7/9 年（2018–2026）；
3. **组合层**：十分位多空月均毛超额 ≥ +30bp，且扣 27bp 往返后仍 > 0，配对 t ≥ 1.5。

三条任一不过 ⇒ 该因子收口，不再换第 5 个构造（防止「挑格子」）。

用法
----
    python -m scripts.evaluation.rating_probe            # 全因子
    python -m scripts.evaluation.rating_probe --factors rtg_rev_net,cov_chg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.altdata import source_em_report as src

BASE = "reports/alla_rolling/_base"
PRED_DIR = "reports/alla_rolling_ortho920fundind/pred"
PRED = f"{PRED_DIR}/gbdt__h1.parquet"
OUT = Path("reports/em_rating_probe")

W = 91          # 因子窗口（自然日）
W_PREV = 91     # cov_chg 的对照窗口
COST_BP = 13.5  # 项目统一单边成本（往返 27bp）
H_W = {1: 2.0, 5: 1.0, 10: 1.0, 20: 2.0}   # gbdt_mid13 权重（2:1:1:2）
LAMBDAS = (0.1, 0.2)                       # 预注册的组合层权重
K, BUF = 15, 30                            # focus 执行：K15 + buffer30
FACTORS = [
    "rtg_rev_net", "rtg_delta_mean", "rtg_first_share", "rtg_level",
    "rtg_disp", "cov_level", "cov_chg", "tgt_upside",
]


def load_main_signal(mask: pd.DataFrame) -> pd.DataFrame:
    """gbdt_mid13：h1:h5:h10:h20 = 2:1:1:2 的截面秩加权（与 focus 生产口径一致）。"""
    num = None
    val = None
    for h, w in H_W.items():
        p = pd.read_parquet(f"{PRED_DIR}/gbdt__h{h}.parquet")
        v = p.notna() & mask.reindex(index=p.index, columns=p.columns).fillna(False).astype(bool)
        r = p.where(v).rank(axis=1, pct=True)
        num = r * w if num is None else num + r * w
        val = v if val is None else (val & v)
    return (num / sum(H_W.values())).where(val)


# ---------------------------------------------------------------- 事件准备
def _to_symbol(code6: pd.Series) -> pd.Series:
    """6 位代码 → 项目面板口径 ``000001.SZ``。

    实测 close 面板前缀分布：6→SH(2421) / 0→SZ(1600) / 3→SZ(1449) / 9→BJ(346)，
    无歧义（无 4、8 开头标的）。4/8 开头按北交所兜底。
    """
    p = code6.str[0]
    suf = np.select(
        [p.eq("6"), p.isin(["0", "3"]), p.isin(["4", "8", "9"])],
        [".SH", ".SZ", ".BJ"], default="",
    )
    return code6 + suf


def load_events() -> pd.DataFrame:
    ev = src.load_cache()
    ev = ev.dropna(subset=["code", "pub_date"]).copy()
    ev["code"] = _to_symbol(ev["code"].astype(str).str.zfill(6))
    ev = ev.sort_values(["code", "org_code", "pub_date"])

    # PIT：只有日期无时点 ⇒ +1 自然日，再映射到当天或之后的第一个交易日
    ev["_avail_raw"] = ev["pub_date"] + pd.Timedelta(days=src.AVAIL_LAG_TRADING_DAYS)

    g = ev.groupby(["code", "org_code"], sort=False)["rating_value"]
    ev["prev_rating"] = g.shift(1)
    ev["delta"] = ev["rating_value"] - ev["prev_rating"]
    ev["is_first"] = ev["prev_rating"].isna()
    return ev


def map_to_trading_days(ev: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """把 _avail_raw 映射到「当天或之后的第一个交易日」，超出日历的丢弃。"""
    pos = cal.searchsorted(ev["_avail_raw"].values, side="left")
    ok = pos < len(cal)
    ev = ev.loc[ok].copy()
    ev["avail_day"] = cal[pos[ok]]
    return ev


# ---------------------------------------------------------------- 因子构建
def build_factors(ev: pd.DataFrame, rebal: pd.DatetimeIndex,
                  close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """对每个调仓日 t，用 avail_day ∈ (t−W, t] 的事件构造截面因子。"""
    codes = close.columns
    out = {f: pd.DataFrame(np.nan, index=rebal, columns=codes, dtype="float32")
           for f in FACTORS}
    ev = ev.sort_values("avail_day")
    avail = ev["avail_day"].values

    for t in rebal:
        lo = np.searchsorted(avail, np.datetime64(t - pd.Timedelta(days=W)), side="right")
        hi = np.searchsorted(avail, np.datetime64(t), side="right")
        win = ev.iloc[lo:hi]
        if win.empty:
            continue
        lo2 = np.searchsorted(avail, np.datetime64(t - pd.Timedelta(days=W + W_PREV)),
                              side="right")
        prev = ev.iloc[lo2:lo]

        a = win.groupby("code").agg(
            n_rep=("rating_value", "size"),
            n_up=("delta", lambda s: float((s > 0).sum())),
            n_down=("delta", lambda s: float((s < 0).sum())),
            n_delta=("delta", "count"),
            sum_delta=("delta", "sum"),
            n_first=("is_first", "sum"),
            mean_rating=("rating_value", "mean"),
            std_rating=("rating_value", "std"),
            n_rating=("rating_value", "count"),
            tp=("target_price", "mean"),
            n_tp=("target_price", "count"),
        )
        n_prev = prev.groupby("code").size().rename("n_prev")

        idx = a.index.intersection(codes)
        a = a.loc[idx]
        n_prev = n_prev.reindex(idx).fillna(0.0)

        rev = (a["n_up"] - a["n_down"]) / a["n_rep"]
        delta_mean = a["sum_delta"] / a["n_delta"].replace(0, np.nan)
        first_share = a["n_first"] / a["n_rep"]
        disp = a["std_rating"].where(a["n_rating"] >= 2)
        cov_lv = np.log1p(a["n_rep"])
        cov_ch = np.log1p(a["n_rep"]) - np.log1p(n_prev.reindex(idx).values)
        px = close.loc[t].reindex(idx)
        upside = (a["tp"] / px - 1.0).where(a["n_tp"] >= 1)

        vals = {
            "rtg_rev_net": rev, "rtg_delta_mean": delta_mean,
            "rtg_first_share": first_share, "rtg_level": a["mean_rating"],
            "rtg_disp": disp, "cov_level": cov_lv, "cov_chg": cov_ch,
            "tgt_upside": upside,
        }
        for k, v in vals.items():
            out[k].loc[t, v.index] = v.values.astype("float32")
    return out


# ---------------------------------------------------------------- 评价
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
    return x.mean(), (x.mean() / se if se > 0 else np.nan)


def month_pairs(cal: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(建仓信号日 = 月末, 下一月末)，两者都 +1 交易日开盘进出。"""
    s = pd.Series(cal, index=cal)
    rb = list(s.groupby([cal.year, cal.month]).last().values)
    return [(a, b) for a, b in zip(rb[:-1], rb[1:])]


def evaluate(fac: dict[str, pd.DataFrame], cal: pd.DatetimeIndex, close: pd.DataFrame,
             openp: pd.DataFrame, mask: pd.DataFrame,
             pred: pd.DataFrame, mcap: pd.DataFrame | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = month_pairs(cal)
    rows_ic, rows_ls = [], []
    prev_pick: dict[str, tuple[set, set]] = {}
    any_fac = next(iter(fac.values()))
    pairs = [(a, b) for a, b in pairs if a in any_fac.index]

    for t, t2 in pairs:
        i, j = cal.get_loc(t), cal.get_loc(t2)
        if j + 1 >= len(cal):
            continue
        d0, d1 = cal[i + 1], cal[j + 1]
        r = (openp.loc[d1] / openp.loc[d0] - 1).replace([np.inf, -np.inf], np.nan)
        tradable = mask.loc[t].astype(bool) if t in mask.index else pd.Series(True, index=r.index)
        pr = pred.loc[t] if t in pred.index else None

        for name, f in fac.items():
            sig = f.loc[t]
            ok = sig.notna() & r.notna() & tradable.reindex(sig.index).fillna(False).astype(bool)
            if ok.sum() < 50:
                continue
            sv = sig[ok].rank()
            rv = r[ok].rank()
            ic = sv.corr(rv)
            rec = dict(date=t, factor=name, ic=ic, n=int(ok.sum()))

            # 残差 IC：因子秩对 mid13 秩回归后的残差 vs 收益秩（在主信号也有值的子集上）
            if pr is not None:
                pv = pr.reindex(sig.index)
                ok2 = ok & pv.notna()
                if int(ok2.sum()) >= 50:
                    y = sig[ok2].rank().values
                    x = pv[ok2].rank().values
                    X = np.column_stack([np.ones(len(x)), x])
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    resid = y - X @ beta
                    rec["resid_ic"] = pd.Series(resid).corr(
                        pd.Series(r[ok2].rank().values))
                    rec["n_resid"] = int(ok2.sum())

                    # 加市值控制的稳健性诊断（不进判据）：rank(y) ~ 1 + mid13 + logmcap
                    if mcap is not None and t in mcap.index:
                        mv = mcap.loc[t].reindex(sig.index)
                        ok3 = ok2 & mv.notna()
                        if int(ok3.sum()) >= 50:
                            y3 = sig[ok3].rank().values
                            X3 = np.column_stack([
                                np.ones(len(y3)),
                                pv[ok3].rank().values,
                                np.log(mv[ok3]).rank().values,
                            ])
                            b3, *_ = np.linalg.lstsq(X3, y3, rcond=None)
                            res3 = y3 - X3 @ b3
                            rec["resid_ic_sz"] = pd.Series(res3).corr(
                                pd.Series(r[ok3].rank().values))
            rows_ic.append(rec)

            # 十分位多空（毛）+ 两腿换手
            q = pd.qcut(sv.rank(method="first"), 10, labels=False)
            uni = r[ok].mean()
            top, bot = r[ok][q == 9], r[ok][q == 0]
            s_top, s_bot = set(top.index), set(bot.index)
            pt, pb = prev_pick.get(name, (None, None))
            turn = np.nan
            if pt is not None:
                turn = ((len(s_top - pt) / max(len(s_top), 1)) +
                        (len(s_bot - pb) / max(len(s_bot), 1))) / 2
            prev_pick[name] = (s_top, s_bot)
            rows_ls.append(dict(date=t, factor=name,
                                gross=(top.mean() - bot.mean()) * 1e4,
                                top_exc=(top.mean() - uni) * 1e4,
                                bot_exc=(bot.mean() - uni) * 1e4,
                                n=int(ok.sum()), turn=turn))
    return pd.DataFrame(rows_ic), pd.DataFrame(rows_ls)


def factor_dev(f: pd.DataFrame) -> pd.DataFrame:
    """因子 → 中性化偏离：覆盖股 (2·pct−1) 去均值，未覆盖股 = 0（预注册口径）。

    未覆盖不填 0 会造成「有覆盖 / 无覆盖」两个子集的系统性错位，填 0 表示
    「不改变主信号对这 只股票的排序判断」，等价于把因子只在覆盖集合内使用。
    """
    pct = f.rank(axis=1, pct=True)
    dev = 2 * pct - 1
    dev = dev.sub(dev.mean(axis=1), axis=0)
    return dev.fillna(0.0)


def focus_excess(sig: pd.DataFrame, pairs: list[tuple], cal: pd.DatetimeIndex,
                 openp: pd.DataFrame, mask: pd.DataFrame) -> np.ndarray:
    """focus 执行口径：K15 + buffer30，月频，T+1 开盘 → 下一 T+1 开盘，超额 vs 全体均值。"""
    ex, hold = [], None
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
        uni = r[ok].mean()
        rank = s2.rank(ascending=False)
        keep = [c for c in (hold or []) if rank.get(c, 1e9) <= BUF]
        add = [c for c in s2.nlargest(K + BUF).index if c not in keep][:K - len(keep)]
        pk = keep + add
        ex.append((r[pk].mean() - uni) * 100)
        hold = pk
    return np.asarray(ex)


def summarize(ic: pd.DataFrame, ls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in sorted(ic["factor"].unique()):
        a = ic[ic.factor == name].set_index("date")
        b = ls[ls.factor == name].set_index("date")
        m_ic, t_ic = nw_t(a["ic"].values)
        m_r, t_r = nw_t(a["resid_ic"].values) if "resid_ic" in a else (np.nan, np.nan)
        yr = a["ic"].groupby(a.index.year).mean()
        yr_r = a["resid_ic"].groupby(a.index.year).mean() if "resid_ic" in a else yr * np.nan
        gross, t_g = nw_t(b["gross"].values)
        tn = float(b["turn"].mean())
        net = gross - 2 * tn * COST_BP
        m_s, t_s = nw_t(a["resid_ic_sz"].values) if "resid_ic_sz" in a else (np.nan, np.nan)
        rows.append(dict(
            factor=name, n_month=len(a), cover_pct=a["n"].mean() / 4300,
            ic=m_ic, ic_t=t_ic, resid_ic=m_r, resid_t=t_r,
            resid_sz=m_s, resid_sz_t=t_s,
            yr_pos=int((yr > 0).sum()), yr_pos_resid=int((yr_r > 0).sum()), yr_n=len(yr),
            ls_gross_bp=gross, ls_t=t_g, turn=tn, ls_net_bp=net,
        ))
    df = pd.DataFrame(rows)
    return df.sort_values("resid_ic", ascending=False)


def verdict(row: pd.Series) -> str:
    fails = []
    if not abs(row["resid_t"]) >= 3.0:
        fails.append("残差t")
    if not max(row["yr_pos_resid"], row["yr_n"] - row["yr_pos_resid"]) >= 7:
        fails.append("分年")
    ok3 = any(row[f"d_{lam}"] >= 0.3 and row[f"t_{lam}"] >= 1.5 for lam in LAMBDAS)
    if not ok3:
        fails.append("组合")
    return "通过" if not fails else "未过:" + "+".join(fails)


def main() -> None:
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default="", help="逗号分隔，默认全部")
    ap.add_argument("--window", type=int, default=W, help="因子窗口（自然日），默认 91")
    ap.add_argument("--tag", default="", help="输出子目录后缀（如 w250）")
    args = ap.parse_args()
    W = args.window
    keep = [x for x in args.factors.split(",") if x] or FACTORS
    out_dir = OUT if not args.tag else OUT.parent / f"{OUT.name}_{args.tag}"

    close = pd.read_parquet(f"{BASE}/close_adj.parquet")
    openp = pd.read_parquet(f"{BASE}/open_adj.parquet")
    mask = pd.read_parquet(f"{BASE}/tradable_mask.parquet").fillna(False).astype(bool)
    mcap = pd.read_parquet(f"{BASE}/market_cap.parquet")
    mid13 = load_main_signal(mask)

    cal = close.index
    ev = map_to_trading_days(load_events(), cal)
    print(f"事件 {len(ev):,} 条，{ev['code'].nunique():,} 只股票，"
          f"{ev['org_code'].nunique():,} 家机构，{ev['pub_date'].min().date()} ~ {ev['pub_date'].max().date()}")

    pairs_all = month_pairs(cal)
    pairs = [(a, b) for a, b in pairs_all if a >= pd.Timestamp("2018-01-01")]
    rebal = pd.DatetimeIndex([a for a, _ in pairs])

    fac = build_factors(ev, rebal, close)
    fac = {k: v for k, v in fac.items() if k in keep}
    cover = {k: float(v.notna().mean(axis=1).mean()) for k, v in fac.items()}
    print("平均截面覆盖：" + "  ".join(f"{k} {v:.1%}" for k, v in cover.items()))

    ic, ls = evaluate(fac, cal, close, openp, mask, mid13, mcap)
    s = summarize(ic, ls)

    # 组合层：mid13 + λ·因子偏离，K15+buffer30 月均超额
    # pairs 需同时落在 mid13 覆盖内，否则基线/复合两侧期数不等
    pairs_c = [(a, b) for a, b in pairs if a in mid13.index]
    base_ex = focus_excess(mid13, pairs_c, cal, openp, mask)
    m0, t0 = nw_t(base_ex)
    print(f"\n基线 mid13 K15 月均超额 {m0:+.3f}%  t {t0:+.2f}  n={len(base_ex)}")
    for lam in LAMBDAS:
        s[f"d_{lam}"] = np.nan
        s[f"t_{lam}"] = np.nan
    for name, f in fac.items():
        dev = factor_dev(f)
        for lam in LAMBDAS:
            e = focus_excess(mid13 + lam * dev, pairs_c, cal, openp, mask)
            n = min(len(e), len(base_ex))
            d = e[:n] - base_ex[:n]
            m, t = nw_t(d)
            s.loc[s.factor == name, f"d_{lam}"] = m
            s.loc[s.factor == name, f"t_{lam}"] = t

    s["verdict"] = s.apply(verdict, axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    ic.to_parquet(out_dir / "daily_ic.parquet", index=False)
    ls.to_parquet(out_dir / "decile_ls.parquet", index=False)
    s.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    yr_tab = (ic.pivot_table(index=ic["date"].dt.year, columns="factor",
                             values="resid_ic", aggfunc="mean").round(4))
    yr_tab.to_csv(out_dir / "resid_ic_by_year.csv", encoding="utf-8")

    pd.set_option("display.width", 240)
    show = s.copy()
    show["cover_pct"] = (show["cover_pct"] * 100).round(1)
    for c in ("ic", "resid_ic", "resid_sz"):
        show[c] = show[c].round(4)
    for c in ("ic_t", "resid_t", "resid_sz_t", "ls_t", "t_0.1", "t_0.2"):
        show[c] = show[c].round(2)
    show["turn"] = (show["turn"] * 100).round(1)
    for c in ("ls_gross_bp", "ls_net_bp", "d_0.1", "d_0.2"):
        show[c] = show[c].round(2)
    print("\n=== 汇总（残差 IC 降序；cover% 平均截面覆盖；d/t 为并入 mid13 的 K15 超额增量）===")
    print(show.to_string(index=False))
    print(f"\n成本口径：单边 {COST_BP}bp，多空两腿往返 {2 * COST_BP}bp，扣在 ls_net_bp")
    (out_dir / "meta.json").write_text(json.dumps(
        dict(factors=keep, window_days=W, cost_bp=COST_BP, n_events=int(len(ev)),
             start=str(pd.Timestamp(pairs[0][0]).date()),
             end=str(pd.Timestamp(pairs[-1][0]).date())),
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
