"""`inc_pure` 的优化实测：金额分层 / 延迟建仓 / 负面排除 / 单日限额 / 与主线叠加。

存在的理由：`event_driven_probe` 只回答「这一族能不能做」，不回答「做了之后还能不能更好」。
本脚本固定 `inc_pure` 的定义与判据，只在其之上加**筛选条件**（不改事件定义本身），
用来区分「真的有增量」与「换股噪声」——后者必须靠安慰剂或对照臂识别。

五个变体，每个都与 base 做**同口径配对**（同样的 base 数据、基准、成本）：

| 变体 | 假设 | 若成立说明什么 |
|---|---|---|
| `amt_hi` | 增持金额越大越强 | 「承诺成本」是信号强度的主轴，可按金额加权 |
| `amt_lo` | 小额增持更弱 | 反向验证：若两者一样，金额无用 |
| `delay2` | 信息扩散需 1~2 天，晚一天买更好 | 若更差 ⇒ 公告日已被消化，抢不到 |
| `no_neg` | 同期有减持/解禁的增持是伪信号 | 把「行为方向」做净额判断 |
| `cap10` | 单日建仓不设上限会被稀疏事件放大 | 容量与成本约束的量化 |

另附 `overlay_k15`：`inc_pure` 与主线 K15 的日超额按 50/50 叠加，看 Sharpe 能否提升。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation import event_driven_probe as M
from scripts.evaluation.event_reach_probe import (holdings_path, load_signal,
                                                  month_pairs)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "event_opt_probe"
PQ = M.PQ

_ACTIVE = ("竞价交易", "二级市场买卖", "大宗交易")


# ============================================================ 事件（带金额）
def inc_events(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """`inc_pure` 的原始记录级事件（未按 code×avail 聚合，保留 amount 供分层）。"""
    df = pd.read_parquet(PQ / "alt_cninfo_holder.parquet",
                         columns=["code", "ann_date", "direction", "chg_ratio",
                                  "amount", "reason", "relation"])
    df = df[df["direction"].astype(str).str.contains("增持")]
    df = df[df["reason"].astype(str).str.contains("|".join(_ACTIVE), na=False)]
    df = df[df["relation"].astype(str).isin(["本人"])]
    df["code"] = M._sym(df["code"])
    r = pd.to_numeric(df["chg_ratio"], errors="coerce").abs()
    df = df.assign(r=r, amt=pd.to_numeric(df["amount"], errors="coerce").abs())
    df = df[r >= 0.1]
    df = df.assign(avail=M._avail(cal, M._dt(df["ann_date"]))).dropna(subset=["avail"])
    g = (df.groupby(["code", "avail"])
           .agg(strength=("r", "sum"), amt=("amt", "sum"), n=("r", "size"))
           .reset_index())
    return g


# ============================================================ 变体
def shift_avail(ev: pd.DataFrame, cal: pd.DatetimeIndex, n: int) -> pd.DataFrame:
    """把可见日往后挪 n 个交易日（模拟「等 n 天才建仓」）。"""
    pos = cal.get_indexer(pd.DatetimeIndex(ev["avail"])) + n
    keep = (pos >= 0) & (pos < len(cal))
    return ev.loc[keep].assign(avail=cal[pos[keep]])


def neg_flags(cal: pd.DatetimeIndex, close: pd.DataFrame) -> pd.DataFrame:
    """每 (code, 交易日) 是否处于「负面事件窗」：近 60 自然日内有减持或解禁。"""
    frames = []
    for fn in (M.fam_red, M.fam_unlock):
        e = fn(cal, close)[["code", "avail"]]
        frames.append(e.assign(kind=fn.__name__))
    e = pd.concat(frames, ignore_index=True)
    e["d"] = cal.get_indexer(pd.DatetimeIndex(e["avail"]))
    e = e[e["d"] >= 0]
    idx = pd.MultiIndex.from_arrays([e["code"], e["d"]])
    return pd.DataFrame({"neg": True}, index=idx.unique())


def apply_cap(pos: pd.DataFrame, ev: pd.DataFrame, cal: pd.DatetimeIndex,
              cap: int) -> pd.DataFrame:
    """单日建仓上限：同一天最多开 cap 笔，按事件强度取前 cap。

    ``pos`` 只有 [code, i0, i1]，事件强度需按 ``avail = cal[i0-1]`` 反查
    （``build_positions`` 里 ``i0 = pos(avail)+1``）。
    """
    if cap <= 0:
        return pos
    key = ev.set_index(["code", "avail"])["strength"]
    av = pd.DatetimeIndex(cal[np.maximum(pos["i0"].to_numpy() - 1, 0)])
    s = pd.Series([float(key.get((c, a), 0.0))
                   for c, a in zip(pos["code"].to_numpy(), av)], index=pos.index)
    tmp = pos.assign(avail=av, _s=s)
    keep = (tmp.sort_values(["avail", "_s"], ascending=[True, False])
               .groupby("avail").head(cap).index)
    return pos.loc[keep].sort_values(["code", "i0"])


# ============================================================ 主线 K15 日收益
def k15_daily(close: pd.DataFrame, openp: pd.DataFrame, mask: pd.DataFrame,
              cal: pd.DatetimeIndex) -> pd.Series:
    """K15 + buffer30 主线的日收益（用于叠加检验）。"""
    sig = load_signal(mask)
    pairs = month_pairs(cal)
    hp = holdings_path(sig, pairs, cal, openp, mask)
    col = {c: i for i, c in enumerate(close.columns)}
    cv = close.values
    live: dict[int, list[int]] = {}
    for t, t2, c in hp[["t", "t2", "code"]].itertuples(index=False):
        i, j = cal.get_loc(t), cal.get_loc(t2)
        for k in range(i + 1, min(j + 2, len(cal))):
            live.setdefault(k, []).append(col[c])
    out = np.full(len(cal), np.nan)
    for i, js in live.items():
        j = np.array(js, dtype=int)
        with np.errstate(invalid="ignore"):
            rr = cv[i, j] / cv[i - 1, j] - 1.0
        if np.isfinite(rr).any():
            out[i] = float(np.nanmean(rr))
    return pd.Series(out, index=cal)


# ============================================================ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--amt-cut", type=float, default=1000.0, help="金额分档阈值（万元）")
    ap.add_argument("--cap", type=int, default=10, help="单日建仓上限（笔）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    close, openp, mask, bench, mcap = M.load_base()
    cal = pd.DatetimeIndex(close.index)
    ev = inc_events(cal).drop_duplicates(subset=["code", "avail"])
    print(f"[base] {cal[0].date()} ~ {cal[-1].date()}  事件={len(ev)}  H={args.hold}")

    amt_w = ev["amt"].fillna(0.0) / 1e4          # 万元
    neg = neg_flags(cal, close)

    arms: dict[str, pd.DataFrame] = {"base": ev}
    arms["amt_hi"] = ev[amt_w >= args.amt_cut]
    arms["amt_lo"] = ev[amt_w < args.amt_cut]
    arms["delay2"] = shift_avail(ev, cal, 2)
    arms["delay3"] = shift_avail(ev, cal, 3)

    rows = []
    for name, sub in arms.items():
        if sub.empty:
            print(f"[skip] {name}: 无事件")
            continue
        pos = M.build_positions(sub, cal, args.hold, close.columns, close, mask)
        if pos.empty:
            continue
        ret, cnt = M.run_portfolio(pos, close, openp, mask, cal)
        mb = M.bench_for_holdings(pos, close, mcap)
        s = M.summarize(ret, pos, cal, args.hold, mb, cnt)
        if not s:
            continue
        s.update(arm=name, n_ev=len(sub))
        rows.append(s)
        pd.DataFrame({"ret": ret, "bench": mb, "n": cnt}).to_parquet(
            OUT / f"daily_{name}_H{args.hold}.parquet")
        print(f"[ok  ] {name:<8} n={len(sub):>5} 持仓={s['avg_hold']:>5.1f} "
              f"超额={s['ann_ex']:+.2%}(扣费{s['ann_ex_net']:+.2%}) t={s['t']:+.2f} "
              f"SR={s['sharpe']:.2f} 分年={s['yr_pos']} 换手={s['turnover']:.1f}x")

    # no_neg：建仓日 (code, i0) 不在负面窗
    pos = M.build_positions(ev, cal, args.hold, close.columns, close, mask)
    hit = np.array([bool(neg.loc[(c, i)].get("neg", False))
                    if (c, i) in neg.index else False
                    for c, i in pos[["code", "i0"]].itertuples(index=False)])
    pos2 = pos[~hit]
    ret, cnt = M.run_portfolio(pos2, close, openp, mask, cal)
    mb = M.bench_for_holdings(pos2, close, mcap)
    s = M.summarize(ret, pos2, cal, args.hold, mb, cnt)
    if s:
        s.update(arm="no_neg", n_ev=len(ev))
        rows.append(s)
        print(f"[ok  ] {'no_neg':<8} 剔掉 {int(hit.sum())}/{len(pos)} 笔  超额={s['ann_ex']:+.2%}"
              f"(扣费{s['ann_ex_net']:+.2%}) t={s['t']:+.2f} SR={s['sharpe']:.2f} 分年={s['yr_pos']}")
        pd.DataFrame({"ret": ret, "bench": mb, "n": cnt}).to_parquet(
            OUT / f"daily_no_neg_H{args.hold}.parquet")

    # cap：单日建仓上限
    pos3 = apply_cap(pos, ev, cal, args.cap)
    ret, cnt = M.run_portfolio(pos3, close, openp, mask, cal)
    mb = M.bench_for_holdings(pos3, close, mcap)
    s = M.summarize(ret, pos3, cal, args.hold, mb, cnt)
    if s:
        s.update(arm=f"cap{args.cap}", n_ev=len(ev))
        rows.append(s)
        print(f"[ok  ] cap{args.cap:<5} 笔数 {len(pos)}→{len(pos3)}  超额={s['ann_ex']:+.2%}"
              f"(扣费{s['ann_ex_net']:+.2%}) t={s['t']:+.2f} SR={s['sharpe']:.2f} 分年={s['yr_pos']}")
        pd.DataFrame({"ret": ret, "bench": mb, "n": cnt}).to_parquet(
            OUT / f"daily_cap{args.cap}_H{args.hold}.parquet")

    # ================================================ 与主线叠加
    # ⚠️ 两层口径陷阱（2026-11-15 踩过）：
    # ① 必须从**首个有效持仓日**起算，否则开头「事件组合空仓、基准仍在动」的
    #    日子会以 −bench 计入超额，把 +14.53% 抬成 +15.86%；
    # ② Sharpe / MDD 一律用**绝对收益**算（不依赖基准选择），超额另列一栏，
    #    否则和 `summarize` 的市值匹配基准混在一起，两边数字对不上。
    k15 = k15_daily(close, openp, mask, cal)
    d0 = pd.read_parquet(OUT / f"daily_base_H{args.hold}.parquet")
    first = d0["ret"].first_valid_index()
    idx = cal[cal >= first] if first is not None else cal
    ip = d0["ret"].reindex(idx).fillna(0.0)
    k = k15.reindex(idx).fillna(0.0)
    b = bench.reindex(idx).fillna(0.0)
    ov = pd.DataFrame({"inc_only": ip, "k15_only": k,
                       "mix_50_50": 0.5 * ip + 0.5 * k,
                       "mix_30_70": 0.3 * ip + 0.7 * k})
    st = pd.DataFrame({
        "ann_ret": ov.mean() * 250,
        "ann_ex_vs_eqw": (ov.sub(b, axis=0)).mean() * 250,
        "sharpe": ov.mean() / ov.std() * np.sqrt(250),
        "mdd": ov.add(1).cumprod().div(ov.add(1).cumprod().cummax()).sub(1).min(),
        "corr_inc": [ov[c].corr(ov["inc_only"]) for c in ov.columns],
    })
    st.to_csv(OUT / "overlay.csv", encoding="utf-8-sig")
    print("\n[叠加] 日超额的收益/风险")
    print(st.to_string(float_format=lambda x: f"{x:,.4f}"))

    out = pd.DataFrame(rows).sort_values("ann_ex_net", ascending=False)
    out.to_csv(OUT / "arms.csv", index=False, encoding="utf-8-sig")
    print("\n" + out[["arm", "n_ev", "avg_hold", "ann_ex", "ann_ex_net", "t",
                      "sharpe", "maxdd", "turnover", "yr_pos"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print(f"\n[out] {OUT/'arms.csv'}")


if __name__ == "__main__":
    main()
