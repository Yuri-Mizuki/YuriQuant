"""事件驱动策略的执行可行性检验（2026-11-15）
================================================

为什么单独做这一层
------------------
`event_opt_probe` 测出 `delay2` 使扣费超额从 +11.95% 掉到 +4.89%（−7.1pp），
说明**超额高度集中在建仓后第一个交易日**。这把整个策略押在一个假设上：

> T+1 开盘能不能买到？

如果不能（一字涨停、停牌、开盘即封板），回测里那 +11.95% 有一大半是纸面收益。
故本脚本只做一件事：**把这个假设当作待检验对象，而不是当作前提。**

四个问题
--------
1. **可成交性**：增持公告次日开盘，跳空分布如何？有多少比例开盘即涨停封死？
   卖出日有多少比例跌停封死（卖不掉）？停牌比例？
2. **执行价敏感性**：开盘价 / 收盘价 / 当日 VWAP 三种真实可得的成交价，
   超额各剩多少？差的那部分说明什么？
3. **流动性容量**：单笔要买多少钱才买得进、买进去会推动多少价格？
   按建仓日成交额分档，超额是否集中在「买不进的小票」上？
4. **综合**：给定摩擦后，这个策略还剩多少可实现的超额。

三个价格口径的差别（这是本脚本的核心）
--------------------------------------
- `open`：回测基准口径。真实可执行、且能吃到建仓日日内涨幅。
- `close`：更容易执行（尾盘买入），但**丢掉建仓日日内那段**。
- `vwap`：最接近真实成交均价（全天分批），但需要算法单。

判据
----
- 可成交性：`封涨停（open==high==low 且跳空≥4.5%）` 比例 **<5%** 才算可执行
- 执行价：三种口径的扣费超额**同号且 VWAP 口径 ≥ +5%** 才算稳健
- 容量：单笔不超过建仓日成交额 **1%** 的前提下，可承载资金 ≥ 100 万

用法
----
    python -m scripts.evaluation.exec_fill_probe                 # inc_pure, H=20
    python -m scripts.evaluation.exec_fill_probe --slip 0.001,0.002
    python -m scripts.evaluation.exec_fill_probe --hold 10       # → reports/exec_fill_probe_h10
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation import event_driven_probe as ED

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "exec_fill_probe"
BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"
PQ = Path("D:/data/parquet")

COST_ONE_WAY = ED.COST_ONE_WAY
SEALED_GAP = 0.045          # 跳空 ≥4.5% 且全天零振幅 ⇒ 任何板块都只能是封板
LIMIT_GAP = 0.095           # 主板/创业板涨停近似阈值（用于「接近涨停」统计）
ABSORB = 0.01               # 单笔最大占建仓日成交额比例


# ============================================================ 价格层
def load_prices():
    """返回 close/open/vwap（复权）与 raw 高低开收（未复权，用于判封板）。"""
    close, openp, mask, bench, mcap = ED.load_base()
    cal = pd.DatetimeIndex(close.index)
    vwap = (pd.read_parquet(BASE_DIR / "vwap_adj.parquet")
              .loc[ED.START:ED.END].reindex(index=cal, columns=close.columns))
    raw = pd.read_parquet(PQ / "daily_all_a.parquet",
                          columns=["open", "high", "low", "close", "amount"])
    raw = raw.reset_index()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["pc"] = raw.groupby("code")["close"].shift(1)
    raw["gap"] = raw["open"] / raw["pc"] - 1.0
    # 全天零振幅 = high==low==open；配合跳空幅度即可判定「封板」
    raw["flat"] = (raw["high"] == raw["low"]) & (raw["high"] == raw["open"])
    flat_up = raw["flat"] & (raw["gap"] >= SEALED_GAP)
    flat_dn = raw["flat"] & (raw["gap"] <= -SEALED_GAP)
    sealed_up = (raw.assign(v=flat_up.astype(np.int8))
                    .pivot_table(index="date", columns="code", values="v",
                                 aggfunc="max").reindex(index=cal, columns=close.columns))
    sealed_dn = (raw.assign(v=flat_dn.astype(np.int8))
                    .pivot_table(index="date", columns="code", values="v",
                                 aggfunc="max").reindex(index=cal, columns=close.columns))
    gap = (raw.pivot_table(index="date", columns="code", values="gap", aggfunc="first")
              .reindex(index=cal, columns=close.columns))
    amt = (raw.pivot_table(index="date", columns="code", values="amount", aggfunc="first")
              .reindex(index=cal, columns=close.columns))
    for d in (sealed_up, sealed_dn):
        d.fillna(0.0, inplace=True)
    return close, openp, vwap, mask, bench, mcap, gap, amt, sealed_up, sealed_dn


# ============================================================ 组合层（可换成交价）
def run_px(pos: pd.DataFrame, close: pd.DataFrame, buy: pd.DataFrame, sell: pd.DataFrame,
           mask: pd.DataFrame, cal: pd.DatetimeIndex, buy_lag: int = 0,
           slip: float = 0.0) -> tuple[pd.Series, pd.Series]:
    """按给定「买入价矩阵 / 卖出价矩阵」计算日频等权组合收益。

    每笔持仓的路径被切成三段：
      建仓日 ``i0``：``buy → close[i0]``   （buy_lag>0 时改为 ``i0+buy_lag`` 建仓）
      持有期：      ``close[t-1] → close[t]``
      卖出日 ``i1``：``close[i1-1] → sell[i1]``

    ``slip`` 为单边冲击成本，买入价上浮、卖出价下调。
    """
    n = len(cal)
    cv = close.values
    bv = buy.values * (1.0 + slip)
    sv = sell.values * (1.0 - slip)
    mv = mask.values
    col = {c: i for i, c in enumerate(close.columns)}

    live = defaultdict(list)
    for code, i0, i1 in pos[["code", "i0", "i1"]].itertuples(index=False):
        j = col.get(code)
        if j is None:
            continue
        e = i0 + buy_lag
        if e >= i1:                      # 延迟后已无持有期
            continue
        if not bool(mv[i0, j]):
            continue
        live[e].append((j, 0, e, i1))
        for i in range(e + 1, i1):
            live[i].append((j, 1, e, i1))
        live[i1].append((j, 2, e, i1))

    ret = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=int)
    for i in range(1, n):
        arr = live.get(i)
        if not arr:
            continue
        j = np.array([a[0] for a in arr], dtype=int)
        ph = np.array([a[1] for a in arr], dtype=int)
        e = np.array([a[2] for a in arr], dtype=int)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cv[i, j] / cv[i - 1, j] - 1.0
            r0 = cv[i, j] / bv[i, j] - 1.0
            r2 = sv[i, j] / cv[i - 1, j] - 1.0
        r = np.where(ph == 0, r0, r)
        r = np.where(ph == 2, r2, r)
        if np.isfinite(r).any():
            ret[i] = float(np.nanmean(r))
            cnt[i] = int(np.isfinite(r).sum())
    return pd.Series(ret, index=cal), pd.Series(cnt, index=cal)


def shift_pos(pos: pd.DataFrame, lag: int, keep_hold: bool = False,
              n_cal: int = 10 ** 9) -> pd.DataFrame:
    """建仓日整体后移 ``lag`` 个交易日。

    ``keep_hold=True`` 时卖出日同步后移（持有期仍为原长度），这是「买晚了但仍持有
    同样天数」的口径；``False`` 时卖出日不动，持有期被压缩。两者语义不同，
    **延迟的代价差 3pp/年**，报告必须说清用的是哪一个。
    """
    if lag == 0:
        return pos
    out = pos.assign(i0=pos["i0"] + lag)
    if keep_hold:
        out = out.assign(i1=out["i1"] + lag)
    out = out[out["i1"] < n_cal]
    return out[out["i1"] > out["i0"]]


def ann_stats(ret: pd.Series, bench: pd.Series, pos: pd.DataFrame,
              cal: pd.DatetimeIndex, cnt: pd.Series) -> dict:
    """年化 / 超额 / t / Sharpe / 回撤 / 换手 / 净超额（简单版，够对比用）。"""
    r = ret.dropna()
    if len(r) < 250:
        return {}
    ex = (r - bench.reindex(r.index)).dropna()
    m, t = ED.nw_t(ex.values)
    avg_h = float(cnt[cnt > 0].mean()) if (cnt > 0).any() else 1.0
    turn = float(len(pos) / len(r) / max(avg_h, 1.0) * 2 * 250)
    eq = (1 + r).cumprod()
    return {
        "days": len(r),
        "ann_ret": float(r.mean() * 250),
        "ann_ex": float(m * 250),
        "t": t,
        "sharpe": float(r.mean() / r.std() * np.sqrt(250)) if r.std() > 0 else np.nan,
        "maxdd": float((eq / eq.cummax() - 1).min()),
        "turnover": turn,
        "ann_ex_net": float(m * 250) - turn * COST_ONE_WAY,
        "yr_pos": "%d/%d" % (sum(v > 0 for v in ex.groupby(ex.index.year).mean().values),
                             ex.index.year.nunique()),
    }


# ============================================================ 逐笔层
def trade_table(pos: pd.DataFrame, close, openp, vwap, gap, amt,
                sealed_up, sealed_dn, cal) -> pd.DataFrame:
    """每笔持仓的执行属性：跳空、封板、成交额、各口径收益。"""
    col = {c: i for i, c in enumerate(close.columns)}
    cv, ov, vv = close.values, openp.values, vwap.values
    gv, av = gap.values, amt.values
    su, sd = sealed_up.values, sealed_dn.values
    rows = []
    for code, i0, i1 in pos[["code", "i0", "i1"]].itertuples(index=False):
        j = col.get(code)
        if j is None:
            continue
        rows.append({
            "code": code, "i0": i0, "i1": i1,
            "gap_in": gv[i0, j], "gap_out": gv[i1, j],
            "sealed_up_in": bool(su[i0, j]), "sealed_dn_out": bool(sd[i1, j]),
            "amt_in": av[i0, j],
            "r_open": ov[i1, j] / ov[i0, j] - 1.0,
            "r_close": cv[i1, j] / cv[i0, j] - 1.0,
            "r_vwap": vv[i1, j] / vv[i0, j] - 1.0,
            "intraday0": cv[i0, j] / ov[i0, j] - 1.0,
            "overnight0": ov[i0 + 1, j] / cv[i0, j] - 1.0
            if i0 + 1 < len(cal) else np.nan,
        })
    d = pd.DataFrame(rows)
    d["dd"] = cal[d["i0"]].values
    return d


def bench_for_trades(d: pd.DataFrame, close, mcap) -> np.ndarray:
    """每笔持仓在持有期内、其建仓时所属市值组的累计基准收益。"""
    pos = d[["code", "i0", "i1"]].copy()
    mb = ED.bench_for_holdings(pos, close, mcap).fillna(0.0).values
    return np.array([np.prod(1 + mb[i0 + 1:i1 + 1]) - 1
                     for i0, i1 in d[["i0", "i1"]].itertuples(index=False)])


# ============================================================ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=20)
    ap.add_argument("--family", default="inc_pure")
    ap.add_argument("--slip", default="0,0.001,0.002,0.005",
                    help="单边冲击成本候选，逗号分隔")
    ap.add_argument("--out", default=None,
                    help="产物目录（默认 reports/exec_fill_probe；--hold≠20 时自动加 _h<hold> 后缀）")
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        OUT if args.hold == 20 else OUT.with_name(f"{OUT.name}_h{args.hold}"))
    out.mkdir(parents=True, exist_ok=True)

    close, openp, vwap, mask, bench, mcap, gap, amt, su, sd = load_prices()
    cal = pd.DatetimeIndex(close.index)
    fn = ED.FAMILIES[args.family]
    ev = fn(cal, close).drop_duplicates(subset=["code", "avail"])
    pos = ED.build_positions(ev, cal, args.hold, close.columns, close, mask)
    print(f"[base] {args.family} 事件={len(ev)} 持仓={len(pos)} H={args.hold}")
    print(f"[base] 日历 {cal[0].date()} ~ {cal[-1].date()} ({len(cal)} 天)")

    # ---------------------------------------------------- 1 可成交性
    d = trade_table(pos, close, openp, vwap, gap, amt, su, sd, cal)
    d["ex_open"] = d["r_open"] - bench_for_trades(d, close, mcap)
    d.to_parquet(out / "trades.parquet")

    print("\n" + "=" * 78)
    print("1 可成交性（建仓日 = 公告可见日的下一交易日开盘）")
    print("=" * 78)
    n = len(d)
    print(f"  跳空分布  p5 {np.nanpercentile(d.gap_in,5):+.2%}  "
          f"p25 {np.nanpercentile(d.gap_in,25):+.2%}  "
          f"中位 {np.nanmedian(d.gap_in):+.2%}  "
          f"p75 {np.nanpercentile(d.gap_in,75):+.2%}  "
          f"p95 {np.nanpercentile(d.gap_in,95):+.2%}  p99 {np.nanpercentile(d.gap_in,99):+.2%}")
    for th, label in [(0.03, "≥3%"), (0.05, "≥5%"), (LIMIT_GAP, "≥9.5%")]:
        k = int((d.gap_in >= th).sum())
        print(f"  开盘跳空 {label:<6} {k:>4} 笔 ({k/n:>5.2%})")
    k = int(d.sealed_up_in.sum())
    print(f"  **开盘即封涨停（无法买入）** {k} 笔 ({k/n:.2%})")
    k2 = int(d.sealed_dn_out.sum())
    print(f"  卖出日封跌停（无法卖出）     {k2} 笔 ({k2/n:.2%})")
    print(f"  建仓日成交额 中位 {np.nanmedian(d.amt_in)/1e8:.2f} 亿  "
          f"p10 {np.nanpercentile(d.amt_in,10)/1e8:.2f} 亿  "
          f"p90 {np.nanpercentile(d.amt_in,90)/1e8:.2f} 亿")

    # 跳空分档看超额是否集中在「买不到」的那一档
    print("\n  按建仓日跳空分档 → 整段超额（检验「好机会是否都在高开组」）")
    q = pd.qcut(d["gap_in"].rank(method="first"), 5, labels=False)
    for g in range(5):
        s = d[q == g]
        print(f"    档{g} 跳空中位 {s.gap_in.median():+.2%}  n={len(s):>4}  "
              f"超额 {s.ex_open.mean():+.3%}  t={s.ex_open.mean()/s.ex_open.std(ddof=1)*np.sqrt(len(s)):+.2f}")

    # ---------------------------------------------------- 2 执行价敏感性
    print("\n" + "=" * 78)
    print("2 执行价敏感性（同一批持仓，只换成交价）")
    print("=" * 78)
    ncal = len(cal)
    variants = [
        ("开盘买/开盘卖", openp, openp, 0, False),
        ("开盘买/收盘卖", openp, close, 0, False),
        ("收盘买/收盘卖", close, close, 0, False),
        ("VWAP买/VWAP卖", vwap, vwap, 0, False),
        # 延迟：迟买一天但仍持满 20 日（口径 C，最保守也最贴近实盘的「买晚了」）
        ("迟1天买/持满", openp, openp, 1, True),
        ("迟2天买/持满", openp, openp, 2, True),
        # 延迟：迟买但仍按原计划日卖出（口径 A，持有期被压缩）
        ("迟1天买/按期卖", openp, openp, 1, False),
    ]
    rows = []
    for name, bp, sp, lag, keep in variants:
        pv = shift_pos(pos, lag, keep_hold=keep, n_cal=ncal)
        ret, cnt = run_px(pv, close, bp, sp, mask, cal)
        bhv = ED.bench_for_holdings(pv, close, mcap).reindex(cal).fillna(0.0)
        st = ann_stats(ret, bhv, pv, cal, cnt)
        st.update({"variant": name, "buy_lag": lag, "n_pos": len(pv)})
        rows.append(st)
        print(f"  {name:<14} 年化 {st['ann_ret']:>7.2%}  超额 {st['ann_ex']:>7.2%}  "
              f"t {st['t']:>5.2f}  净超额 {st['ann_ex_net']:>7.2%}  "
              f"Sharpe {st['sharpe']:>5.3f}  回撤 {st['maxdd']:>6.2%}  ({len(pv)} 笔)")
        if name.startswith("开盘买/开盘"):
            ret.to_frame("ret").assign(cnt=cnt).to_parquet(out / "daily_base.parquet")
            base_cnt = cnt
            base_ret = ret

    # 保守可成交口径：把跳空 ≥5% 的建仓全部视为「买不到」，直接丢弃
    print("\n  保守成交假设：跳空 ≥5% 视为买不到（丢弃）")
    keep5 = d.loc[d["gap_in"] < 0.05, ["code", "i0", "i1"]]
    ret, cnt = run_px(keep5, close, openp, openp, mask, cal)
    bhv = ED.bench_for_holdings(keep5, close, mcap).reindex(cal).fillna(0.0)
    st = ann_stats(ret, bhv, keep5, cal, cnt)
    rows.append({**st, "variant": "剔除跳空≥5%", "buy_lag": 0, "n_pos": len(keep5)})
    print(f"  {'剔除跳空≥5%':<14} 超额 {st['ann_ex']:>7.2%}  t {st['t']:>5.2f}  "
          f"净超额 {st['ann_ex_net']:>7.2%}  ({len(keep5)} 笔，丢 {len(pos)-len(keep5)} 笔)")

    # 滑点阶梯（开盘口径）
    print("\n  滑点阶梯（开盘买/开盘卖，单边）")
    bh0 = ED.bench_for_holdings(pos, close, mcap).reindex(cal).fillna(0.0)
    for s in [float(x) for x in args.slip.split(",") if x]:
        ret, cnt = run_px(pos, close, openp, openp, mask, cal, slip=s)
        st = ann_stats(ret, bh0, pos, cal, cnt)
        rows.append({**st, "variant": f"开盘+滑点{s*1e4:.0f}bp", "buy_lag": 0, "n_pos": len(pos)})
        print(f"    单边 {s*1e4:>4.0f}bp  超额 {st['ann_ex']:>7.2%}  "
              f"t {st['t']:>5.2f}  净超额 {st['ann_ex_net']:>7.2%}")
    pd.DataFrame(rows).to_csv(out / "variants.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------- 3 容量
    print("\n" + "=" * 78)
    print("3 流动性容量（单笔占建仓日成交额 ≤ %.0f%%）" % (ABSORB * 100))
    print("=" * 78)
    a = d["amt_in"].values
    med = float(np.nanmedian(a))
    avg_names = float(base_cnt[base_cnt > 0].mean())
    print(f"  平均同时持仓 {avg_names:.1f} 只（中位 {float(base_cnt[base_cnt>0].median()):.0f}，"
          f"最多 {int(base_cnt.max())}）⇒ 单笔 = 资金 / {avg_names:.0f}")
    for cap in (1e5, 5e5, 1e6, 5e6, 1e7, 5e7):
        per = cap / max(avg_names, 1)
        need = per / ABSORB                     # 该笔需要多大的成交额才吃得下
        ok10 = need <= np.nanpercentile(a, 10)
        ok50 = need <= med
        print(f"    资金 {cap/1e4:>6.0f} 万 → 单笔 {per/1e4:>6.1f} 万，"
              f"需成交额 ≥ {need/1e8:>5.2f} 亿  中位可容纳 "
              f"{'✅' if ok50 else '❌'}  最差10%分位 {'✅' if ok10 else '❌'}")
    print("\n  按建仓日成交额分档 → 超额（若超额集中在低成交额档，容量就是真约束）")
    q2 = pd.qcut(d["amt_in"].rank(method="first"), 5, labels=False)
    for g in range(5):
        s = d[q2 == g]
        print(f"    档{g} 成交额中位 {s.amt_in.median()/1e8:>5.2f} 亿  n={len(s):>4}  "
              f"超额 {s.ex_open.mean():+.3%}  "
              f"t={s.ex_open.mean()/s.ex_open.std(ddof=1)*np.sqrt(len(s)):+.2f}")

    # ---------------------------------------------------- 4 综合
    print("\n" + "=" * 78)
    print("4 结论")
    print("=" * 78)
    base = [r for r in rows if r["variant"] == "开盘买/开盘卖"][0]
    vw = [r for r in rows if r["variant"].startswith("VWAP")][0]
    cc = [r for r in rows if r["variant"] == "收盘买/收盘卖"][0]
    dfill = [r for r in rows if r["variant"] == "剔除跳空≥5%"][0]
    print(f"  可成交性：封涨停 {d.sealed_up_in.mean():.2%}（门槛 <5%）、"
          f"封跌停 {d.sealed_dn_out.mean():.2%} ⇒ "
          f"{'可执行' if d.sealed_up_in.mean() < 0.05 else '不可执行'}")
    print(f"  执行价：开盘 {base['ann_ex_net']:+.2%} / 收盘 {cc['ann_ex_net']:+.2%} / "
          f"VWAP {vw['ann_ex_net']:+.2%}")
    print(f"  ⇒ 用 VWAP 替代开盘，损耗 {base['ann_ex_net'] - vw['ann_ex_net']:+.2%}/年；"
          f"用收盘替代开盘，损耗 {base['ann_ex_net'] - cc['ann_ex_net']:+.2%}/年")
    print(f"  剔除买不到的（跳空≥5%）后净超额 {dfill['ann_ex_net']:+.2%}/年 "
          f"⇒ 损耗 {base['ann_ex_net'] - dfill['ann_ex_net']:+.2%}/年")
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
