"""focus 集中档信号臂对照：882 登记 vs f50（21 相位均值 + 逐相位配对）。

单相位不能下结论：k8 只有 6 只持仓、年换手 ~85%，起点换一天年化就能差几个点。
``k8_buffer_scan`` 已立规矩 —— 比方案用 **21 相位均值**，t 用「每个相位一个差」
再对 21 个值算（**不**拼日频样本：21 个相位不独立，拼起来 t 会虚高 √21）。

执行层固定为**登记口径**，只换信号：

====  ==========  ==========================================
档位   资金/整手    对应登记读数
====  ==========  ==========================================
k15   1e6 无整手   30.13% / 1.144（uniform 口径）
k8    1e5 + 整手   35.01% / 1.229（native 口径，exit_mult 5.0）
====  ==========  ==========================================

用法::

    python -m scripts.portfolio.focus_signal_compare            # 21 相位
    python -m scripts.portfolio.focus_signal_compare --phases 5

产物：``reports/focus_signal_compare/{phase_scan.csv,summary.csv,paired.csv}``。
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
from strategy.registry import get_tier  # noqa: E402

OUT = ROOT / "reports" / "focus_signal_compare"
END = 20260907
#: 臂名 → 集中档信号面板（None = 登记口径，caliber 现场复合 ortho 的四份 pred）
ARMS: dict[str, Path | None] = {
    "reg": None,
    "f50": ROOT / "reports" / "alla_rolling_ab_f50" / "pred" / "gbdt_mid13__h1.parquet",
}
CAPS: dict[str, tuple[float, int | None]] = {
    "yqcore-focus-k15": (1.0e6, None),
    "yqcore-focus-k8": (1.0e5, 100),
}


def lean_ctx(pred_path: Path | None = None) -> cal.CaliberContext:
    """只装 ``run_once(open)`` 用到的五块（float32；省 ~1G，两臂可同时驻留）。"""
    pred = (pd.read_parquet(pred_path) if pred_path is not None
            else cal._composite_panel("gbdt_mid13"))
    keep = pred.index[(pred.index >= pd.Timestamp(cal.WINDOWS["prod"]))
                      & (pred.index <= pd.Timestamp(str(END)))]
    cols = pred.columns
    pred = pred.reindex(index=keep)

    def load(name: str, dtype: str | None = "float32") -> pd.DataFrame:
        raw = pd.read_parquet(cal.BASE_DIR / f"{name}.parquet")
        out = raw.reindex(index=keep, columns=cols)
        del raw
        gc.collect()
        return out if dtype is None else out.astype(dtype)

    close = load("close_adj")
    mask = load("tradable_mask", None).astype("boolean").fillna(True).astype(bool)
    fill = load("open_adj")
    fwd = (close.pct_change(fill_method=None)
           .replace([np.inf, -np.inf], np.nan).fillna(0.0)).astype("float32")
    return cal.CaliberContext(
        window="prod", signal="gbdt_mid13", begin=cal.WINDOWS["prod"], pred=pred,
        close=close, fwd=fwd, mask=mask, mask_same_day=mask, amt=None,
        vol60=None, ret_hist=fwd, fingerprint=cal.fingerprint_panel(pred),
        execution="open", fill=fill)


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns).ffill().shift(1)


def panel_for(name: str, ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / f"{name}.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phases", type=int, default=21)
    a = ap.parse_args(argv)

    cost = cal.cost_profile("production")
    rows = []
    for arm, pred_path in ARMS.items():
        ctx = lean_ctx(pred_path)
        pp = price_panel(ctx)
        ind = panel_for("industry", ctx)
        print(f"\n=== 臂 {arm}  指纹 {ctx.fingerprint}  "
              f"{len(ctx.pred)} 日 × {ctx.pred.shape[1]} 只")
        for tier, (cap, lot) in CAPS.items():
            spec = get_tier(tier)
            overrides = {spec.panel: ind} if spec.panel else {}
            for ph in range(a.phases):
                strat = spec.build(**overrides)
                bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                        initial_capital=float(cap), costs=cost)
                res = cal.run_once(bt, ctx, "open", signal=ctx.pred,
                                   price_panel=pp if lot else None,
                                   lot_size=lot, phase=ph)
                m = cal.metrics_row(res, cap)
                rows.append({"arm": arm, "tier": tier, "phase": ph,
                             "ann": m["ann"], "sharpe": m["sharpe"],
                             "max_dd": m["max_dd"], "turnover": m["turnover"]})
                print(f"  {tier[-3:]} phase {ph:2d}  ann {m['ann']:7.2%}  "
                      f"SR {m['sharpe']:6.3f}  MDD {m['max_dd']:7.2%}")
        del ctx, pp, ind
        gc.collect()

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "phase_scan.csv", index=False)

    summ = (df.groupby(["tier", "arm"])
            .agg(ann=("ann", "mean"), sharpe=("sharpe", "mean"),
                 max_dd=("max_dd", "mean"), turnover=("turnover", "mean"),
                 n_phase=("phase", "size")).reset_index())
    summ.to_csv(OUT / "summary.csv", index=False)
    print("\n=== 相位均值")
    print(summ.round(4).to_string(index=False))

    paired = []
    for tier in CAPS:
        w = df[df.tier == tier].pivot(index="phase", columns="arm", values="ann")
        d = (w["f50"] - w["reg"]).dropna()
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 else np.nan
        paired.append({"tier": tier, "n_phase": int(len(d)),
                       "reg_ann": float(w["reg"].mean()),
                       "f50_ann": float(w["f50"].mean()),
                       "diff_ann": float(d.mean()), "t_paired": t,
                       "n_win": int((d > 0).sum())})
    pdf = pd.DataFrame(paired)
    pdf.to_csv(OUT / "paired.csv", index=False)
    print("\n=== 逐相位配对（f50 − 882）")
    print(pdf.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
