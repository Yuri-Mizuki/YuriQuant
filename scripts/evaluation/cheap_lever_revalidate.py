"""cheap_lever_scan 候选的多窗口复验（2026-10-14）。

首窗（2018+ 全窗）扫出的两个候选 —— ``buf(0.08/0.40)`` +0.78pp、``h1w0.25`` +0.31pp ——
自带网格 selection bias，**单窗口领先不算数**（back_union 教训）。本脚本在
production 口径下把窗口切成 2019+/2021+/2023+/2025+ 四段，比
锚点 vs 候选 vs 叠加臂的逐窗一致性。

只截 ``pred`` 面板；fwd/mask/close/amt/vol60/ret_hist 保留全历史（Σ 暖机不截断）。
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from scripts.evaluation import caliber
from scripts.evaluation.portfolio_weighting import ROOT

log = logging.getLogger("cheap_lever_reval")

PRED_DIR = ROOT / "reports" / "alla_rolling_ortho" / "pred"
BEGINS = (2019, 2021, 2023, 2025)
ARMS = (  # (tag, frac_entry, frac_exit, w_h1)
    ("anchor_buf15_40_h1w0.5", 0.15, 0.40, 0.5),
    ("buf08_40_h1w0.5", 0.08, 0.40, 0.5),
    ("buf05_30_h1w0.5", 0.05, 0.30, 0.5),
    ("anchor_h1w0.25", 0.15, 0.40, 0.25),
    ("anchor_h1w0.0(纯h5)", 0.15, 0.40, 0.0),
    ("buf08_40_h1w0.25(叠加)", 0.08, 0.40, 0.25),
)


def _blend(p1: pd.DataFrame, p2: pd.DataFrame, w: float) -> pd.DataFrame:
    r1, r2 = p1.rank(axis=1, pct=True), p2.rank(axis=1, pct=True)
    return (w * r1 + (1.0 - w) * r2).where(r1.notna() & r2.notna()).astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    from strategy.examples import BufferedTopFracLongOnly

    t0 = time.time()
    cost = caliber.cost_profile("production")
    cap = caliber.PROFILES["production"].capital
    ctx = caliber.load_context(signal="ens_h1h5", window="prod", execution="open")
    h1 = pd.read_parquet(PRED_DIR / "gbdt__h1.parquet").reindex(
        index=ctx.pred.index, columns=ctx.pred.columns)
    h5 = pd.read_parquet(PRED_DIR / "gbdt__h5.parquet").reindex(
        index=ctx.pred.index, columns=ctx.pred.columns)

    rows = []
    for begin in BEGINS:
        pred_full = ctx.pred
        pred_cut = pred_full.loc[f"{begin}-01-01":]
        if len(pred_cut) < 250:
            log.warning("窗口 %d+ 只有 %d 日，跳过", begin, len(pred_cut))
            continue
        for tag, fe, fx, w in ARMS:
            sig = pred_full if abs(w - 0.5) < 1e-9 else _blend(h1, h5, w)
            # 只截 pred 窗口；fwd/mask/close/amt/vol60/ret_hist 保留全历史（暖机不截断）
            import dataclasses
            ctx_cut = dataclasses.replace(ctx, pred=sig.loc[f"{begin}-01-01":])
            m = caliber.evaluate_strategy(
                BufferedTopFracLongOnly(frac_entry=fe, frac_exit=fx),
                ctx_cut, cost, cap, execution="open")
            rows.append(dict(begin=begin, arm=tag, ann=round(m["ann"] * 100, 2),
                             sharpe=round(m["sharpe"], 3),
                             turnover=round(m["turnover"] * 100, 1)))
            log.info("%d+ %-26s 年化=%+6.2f%% Sharpe=%.3f 换手=%5.1f%%",
                     begin, tag, rows[-1]["ann"], rows[-1]["sharpe"],
                     rows[-1]["turnover"])

    df = pd.DataFrame(rows)
    out = ROOT / "reports" / "cheap_lever_scan" / "revalidate.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, out)
    for begin, g in df.groupby("begin"):
        log.info("[%d+]\n%s", begin, g.drop(columns="begin").to_string(index=False))


if __name__ == "__main__":
    main()
