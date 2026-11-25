"""微盘对照臂：把「小盘 beta」与「模型 alpha」分开（2026-11-25 建）。

三个臂共用同一套执行层（``run_once(open)`` + 27bp 时变成本 + 整手 100 股 +
掩码前置 + 月频调仓），**唯一变化是信号源**：

===================  ==========================================  ============
臂                    信号                                        等效问题
===================  ==========================================  ============
``model-k8``          定版复合信号 ``gbdt_mid13``                 模型选股
``micro-k8``          负市值（``-market_cap``）                   只换信号，其余全同
``micro-wide``        负市值 + 等权 10%                           宽口径风格尺子
===================  ==========================================  ============

两个微盘臂的用途不同：

- ``micro-k8`` 是**控制变量对照**——与 ``model-k8`` 的唯一差别是「信号从模型换成市值」，
  所以 ``model-k8 − micro-k8`` 就是模型的净增量（模型税已含在两边同口径里）。
- ``micro-wide`` 是**归因尺子**——420 只等权，方差小、噪声低，用于判断「策略收益里有多少
  只是小盘 beta」。8 只口径的数字不可引用：极端值主导，2024 年微盘崩盘那年它反而报 +68%。

⚠️ 两个微盘臂都**不是可投产品**：未排 ST/退市整理、涨停买不进、容量几十万级。它们是
尺子，不是替代品。结论见 ``reports/microcap_ref/README.md``。

用法::

    python -m scripts.portfolio.microcap_ref --arm model-k8
    python -m scripts.portfolio.microcap_ref --arm micro-k8
    python -m scripts.portfolio.microcap_ref --arm micro-wide
    python -m scripts.portfolio.microcap_ref --summary

一次只跑一个臂、分进程：本机提交内存常年只剩 ~3G，两臂同时驻留会 OOM。
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import caliber as cal  # noqa: E402
from strategy.examples import TopFracLongOnly  # noqa: E402
from strategy.registry import get_tier  # noqa: E402

END = 20260907
OUT = ROOT / "reports" / "microcap_ref"
#: 臂名 → (资金, 策略构造方式)
ARMS = {
    "model-k8": (1.0e5, "tier"),
    "micro-k8": (1.0e5, "tier"),
    "micro-wide": (2.0e6, "wide"),
}
LOT = 100


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    """未复权收盘 ``ffill().shift(1)``——引擎在执行日 E 取值，故不可前视。"""
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    out = raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns).ffill().shift(1)
    del raw
    gc.collect()
    return out.astype("float32")


def panel_for(name: str, ctx: cal.CaliberContext) -> pd.DataFrame:
    df = pd.read_parquet(cal.BASE_DIR / f"{name}.parquet")
    out = df.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
    del df
    gc.collect()
    return out


def _composite_lean(signal: str) -> pd.DataFrame:
    """``caliber._composite_panel`` 的省内存版（逐分量算完即释放，全程 float32）。

    语义与 :func:`caliber.weighted_rank_average` 一致：``Σ w·rank_pct / Σw``，
    只在各分量都有值的格上输出。本机提交上限常年只剩 1~2G，原版一次性载入
    4 份面板会 OOM。
    """
    spec = cal.SIGNALS[signal]
    total, valid, wsum = None, None, 0.0
    for key, wt in spec["components"].items():
        fp = cal.ROOT / cal.SIGNALS[key]["path"]
        df = pd.read_parquet(fp).astype("float32")
        r = df.rank(axis=1, pct=True).astype("float32") * float(wt)
        total = r if total is None else total.add(r)
        v = df.notna()
        valid = v if valid is None else (valid & v)
        wsum += float(wt)
        del df, r, v
        gc.collect()
    return (total / wsum).where(valid)


def lean_ctx() -> cal.CaliberContext:
    """只装 ``run_once(open)`` 用到的五块面板（同 ``lean_context``，但 float32 分批）。"""
    pred = _composite_lean("gbdt_mid13")
    idx = pred.index
    keep = idx[(idx >= pd.Timestamp(cal.WINDOWS["prod"]))
               & (idx <= pd.Timestamp(str(END)))]
    cols = pred.columns
    pred = pred.reindex(index=keep)
    gc.collect()

    def load(name: str, dtype: str | None = "float32") -> pd.DataFrame:
        df = pd.read_parquet(cal.BASE_DIR / f"{name}.parquet")
        out = df.reindex(index=keep, columns=cols)
        del df
        gc.collect()
        return out if dtype is None else out.astype(dtype)

    close = load("close_adj")
    # 布尔掩码不能走 float32（``DataFrame.where`` 要 bool dtype），缺失按可交易处理
    mask = load("tradable_mask", None).astype("boolean").fillna(True).astype(bool)
    fill = load("open_adj")
    fwd = (close.pct_change(fill_method=None)
           .replace([np.inf, -np.inf], np.nan).fillna(0.0)).astype("float32")
    return cal.CaliberContext(
        window="prod", signal="gbdt_mid13", begin=cal.WINDOWS["prod"], pred=pred,
        close=close, fwd=fwd, mask=mask, mask_same_day=mask, amt=None,
        vol60=None, ret_hist=fwd, fingerprint=cal.fingerprint_panel(pred),
        execution="open", fill=fill)


def neg_mktcap(ctx: cal.CaliberContext) -> pd.DataFrame:
    """负市值面板：值越大 = 市值越小（``TopK`` 取最大 ⇒ 取最小市值）。"""
    cap = pd.read_parquet(cal.BASE_DIR / "market_cap.parquet")
    cap = cap.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
    return -cap


def run_arm(arm: str) -> dict:
    capital, kind = ARMS[arm]
    ctx = lean_ctx()
    pp = price_panel(ctx)

    if kind == "tier":
        spec = get_tier("yqcore-focus-k8")
        strat = spec.build(sectors=panel_for("industry", ctx))
    else:
        strat = TopFracLongOnly(frac=0.10, weight_mode="equal")

    signal = ctx.pred if arm == "model-k8" else neg_mktcap(ctx)

    bt = cal.VectorBacktest(
        strategy=strat, rebalance_freq="M", initial_capital=float(capital),
        costs=cal.cost_profile("production"),
    )
    res = cal.run_once(bt, ctx, "open", signal=signal, price_panel=pp, lot_size=LOT)

    nav = res.equity_curve
    # 统一走 caliber.metrics_row（sharpe 是几何口径，t 用算术 NW）
    m = cal.metrics_row(res, float(capital))
    n_hold = (res.weights_history > 1e-12).sum(axis=1)
    last_w = res.weights_history.iloc[-1]
    last_w = last_w[last_w > 1e-12]
    cap_row = pd.read_parquet(cal.BASE_DIR / "market_cap.parquet").loc[
        res.weights_history.index[-1]
    ]
    rec = {
        "arm": arm,
        "window": f"{nav.index[0].date()}..{nav.index[-1].date()}",
        "days": int(len(nav)),
        "capital": capital,
        "ann": m["ann"],
        "sharpe": m["sharpe"],
        "sharpe_arith": m["sharpe_arith"],
        "t_stat": m["sharpe_t_stat"],
        "vol": m["vol"],
        "max_dd": m["max_dd"],
        "calmar": m["calmar"],
        "turnover": m["turnover"],
        "n_hold_med": float(n_hold.median()),
        "n_hold_mean": m["n_active"],
        "cap_med_yi": float(cap_row.reindex(last_w.index).median()) / 1e8,
        "cost_bp": m["cost_bp"],
        "impact_bp": m["impact_bp"],
    }
    print(f"=== {arm}  {rec['window']}  {rec['days']} 日")
    for k, v in rec.items():
        if isinstance(v, float):
            print(f"  {k:12s} {v:,.4f}")
        else:
            print(f"  {k:12s} {v}")

    daily = pd.DataFrame({
        "nav": nav,
        "ret": nav.pct_change(fill_method=None),
        "weight_sum": res.weights_history.sum(axis=1).reindex(nav.index).ffill(),
    })
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / f"daily_{arm}.csv")
    pd.DataFrame([rec]).to_csv(OUT / f"stat_{arm}.csv", index=False)
    return rec


def summary() -> int:
    files = sorted(OUT.glob("daily_*.csv"))
    if len(files) < 2:
        print(f"只有 {len(files)} 个臂，先跑 --arm")
        return 1
    navs = {}
    for f in files:
        arm = f.stem.replace("daily_", "")
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        navs[arm] = d["nav"]
    df = pd.DataFrame(navs).dropna(how="all")
    df = df.ffill().dropna()
    # 归一到同一起点
    df = df / df.iloc[0]

    monthly = df.resample("ME").last()
    mret = monthly.pct_change().dropna()
    mret.to_csv(OUT / "monthly_returns.csv")

    print(f"月度序列 {len(mret)} 期  {mret.index[0].date()} → {mret.index[-1].date()}")
    print(mret.tail(3).round(4).to_string())

    if "model-k8" in mret.columns:
        rows = []
        for arm in mret.columns:
            if arm == "model-k8":
                continue
            diff = mret["model-k8"] - mret[arm]
            rows.append({
                "vs": arm,
                "ann_diff": float((1 + diff).prod() ** (12 / len(diff)) - 1),
                "win_rate": float((diff > 0).mean()),
                "n": len(diff),
            })
        d = pd.DataFrame(rows)
        d.to_csv(OUT / "alpha_vs_micro.csv", index=False)
        print("\n模型相对微盘的净增量：")
        print(d.round(4).to_string(index=False))

    stats = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("stat_*.csv"))])
    stats.to_csv(OUT / "summary.csv", index=False)
    print("\n" + stats.round(4).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    if not a.arm:
        ap.error("需要 --arm 或 --summary")
    run_arm(a.arm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
