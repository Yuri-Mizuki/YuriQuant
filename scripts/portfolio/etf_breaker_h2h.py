"""ETF 轮动 · 熔断触发后「降档五衡」能不能救：条件前瞻头对头

**决策问题**（README §8.2 / TODO P0）：全样本上五衡比三锋落后 **9.98pp/年**，但赛内纪律
只在「回撤触 −15%」时才切五衡。所以真正要问的不是全样本谁强，而是：

    在**已经出现深回撤**的那一天往后看 42 个交易日（≈ 一个赛制计分期），
    五衡是不是反而更强？

若答案是「是」，降档就是买保险（付全样本的 10pp，换深跌段的相对优势）；
若答案是「不」，那这道闸在任何状态下都是纯亏，应当取消。

方法：两档各跑一次 T+1 open 连续回测（20200101→末日，同一批信号口径），
以**三锋自己的账户回撤**划分状态（纪律本来就是看组合回撤），比较同期未来
h 个交易日的收益差 Δ = 五衡 − 三锋。

🚨 重叠陷阱：逐日起点会产生强重叠（h=42 时相邻 41 天共享 41 天）⇒ t 值严重虚高。
主表只报**点估计**，另给「每 h 日取 1」的非重叠敏感性；不要用 t 检验下结论。

用法::

    python -m scripts.portfolio.etf_breaker_h2h            # 主表 h=42
    python -m scripts.portfolio.etf_breaker_h2h --h 21,42,63 --thr 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import build_execution_split
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_champ_signals import PROFILES
from scripts.portfolio.etf_contest_spec import EXCLUDED_5
from scripts.portfolio.etf_rotation_panels import build_extra_panels
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

BEGIN = 20200101
ROOT = Path(__file__).resolve().parents[2]
BASE = dict(PROFILES["contest3"])
ALT = dict(PROFILES["contest5h"])


def load_env() -> tuple:
    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    _uni, close, amount, _bench = bundle
    raw = cache.read_daily("etf")
    rc = raw["close"].unstack("code").reindex(index=close.index, columns=close.columns)
    ro = raw["open"].unstack("code").reindex(index=close.index, columns=close.columns)
    open_adj = (ro * (close / rc)).astype(np.float32)
    panels = build_extra_panels(close, build_panels(close, amount), rc)
    for x in EXCLUDED_5:
        panels["mask"].loc[:, x] = False
    return bundle, panels, close, open_adj


def run_open(bundle, panels, close, open_adj, spec: dict) -> dict:
    """先跑收盘口径拿调仓日，再按 T+1 open 重跑（与 README §零 一致）。"""
    r0 = run_variant(bundle, panels, spec, BEGIN, None)
    rb = set(pd.DatetimeIndex(r0["rebalance_dates"]))
    spl = build_execution_split(open_adj, close, rb)
    return run_variant(bundle, panels, spec, BEGIN, None,
                       execution_split=spl, execution_shift=True)


def fwd_returns(eq: pd.Series, h: int) -> pd.Series:
    """每一起点的未来 h 个交易日收益（末 h 天为 NaN）。"""
    v = eq.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) > h:
        out[:-h] = v[h:] / v[:-h] - 1.0
    return pd.Series(out, index=eq.index)


def drawdown(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1.0


def bucket_of(dd: float, thr: float) -> str:
    if dd > -0.08:
        return "① 回撤 > −8%"
    if dd > -thr:
        return "② −8% ~ −15%"
    return "③ 回撤 ≤ −15%"


def report(dd: pd.Series, r3: pd.Series, r5: pd.Series, h: int, thr: float,
           step: int = 1) -> pd.DataFrame:
    idx = dd.index[::step]
    sub = pd.DataFrame({"dd": dd.reindex(idx), "r3": r3.reindex(idx),
                        "r5": r5.reindex(idx)}).dropna()
    sub["label"] = [bucket_of(x, thr) for x in sub["dd"]]
    sub["delta"] = sub["r5"] - sub["r3"]
    rows = []
    for name, g in [("全样本", sub)] + [(k, sub[sub["label"] == k])
                                        for k in sorted(sub["label"].unique())]:
        if len(g) == 0:
            continue
        d = g["delta"].to_numpy()
        rows.append({
            "状态": name, "n": len(g),
            "三锋": float(g["r3"].mean()), "五衡": float(g["r5"].mean()),
            "Δ均值": float(d.mean()), "Δ中位": float(np.median(d)),
            "五衡胜率": float((d > 0).mean()),
            "Δp10": float(np.percentile(d, 10)),
            "Δ最差": float(d.min()),
        })
    return pd.DataFrame(rows).set_index("状态")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="熔断触发的条件前瞻：三锋 vs 五衡")
    ap.add_argument("--h", default="21,42,63", help="前瞻交易日（逗号分隔）")
    ap.add_argument("--thr", type=float, default=0.15, help="熔断线（默认 0.15）")
    ap.add_argument("--csv", default=str(ROOT / "reports/etf_contest5/breaker_h2h.csv"))
    args = ap.parse_args(argv)
    hs = [int(x) for x in args.h.split(",")]

    bundle, panels, close, open_adj = load_env()
    r3 = run_open(bundle, panels, close, open_adj, BASE)
    r5 = run_open(bundle, panels, close, open_adj, ALT)
    print(f"[base] 三锋 {r3['annual_return']:.2%} SR {r3['sharpe']:.2f} "
          f"MDD {r3['max_drawdown']:.2%} | 五衡 {r5['annual_return']:.2%} "
          f"SR {r5['sharpe']:.2f} MDD {r5['max_drawdown']:.2%}", flush=True)

    e3 = r3["equity"].astype(float)
    e5 = r5["equity"].astype(float).reindex(e3.index)
    dd = drawdown(e3)
    print(f"末日 {e3.index[-1]:%Y-%m-%d} 三锋账户回撤 {dd.iloc[-1]:+.2%}"
          f"（触发线 {args.thr:.0%}）")

    out = []
    for h in hs:
        f3, f5 = fwd_returns(e3, h), fwd_returns(e5, h)
        for step, tag in ((1, "逐日"), (h, f"每{h}日")):
            t = report(dd, f3, f5, h, args.thr, step=step)
            print(f"\n== h={h} 交易日 · {tag}起点（Δ = 五衡 − 三锋，同起点同区间）==")
            print(f"{'状态':<16}{'n':>5}{'三锋':>9}{'五衡':>9}{'Δ均值':>9}"
                  f"{'Δ中位':>9}{'五衡胜率':>9}{'Δp10':>9}{'Δ最差':>9}")
            for k, row in t.iterrows():
                print(f"{k:<16}{row['n']:>5}{row['三锋']:>+9.2%}{row['五衡']:>+9.2%}"
                      f"{row['Δ均值']:>+9.2%}{row['Δ中位']:>+9.2%}"
                      f"{row['五衡胜率']:>9.1%}{row['Δp10']:>+9.2%}"
                      f"{row['Δ最差']:>+9.2%}")
                out.append({"h": h, "step": tag, "state": k, **row.to_dict()})

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n落盘 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
