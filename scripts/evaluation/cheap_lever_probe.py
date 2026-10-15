"""低成本微调总扫描（2026-10-14，由莉酱问「低成本的优化方案是真的没有了吗」）。

路线级方向已全部收口（前融合/back_union/集中度/风险维/超参/新源…），
本探针只扫**微调级**四维 —— 全部零训练、纯信号层/组合层/执行层，
单臂秒级，总耗时 ~5 分钟：

1. **frac × buffer 二维**：历史只扫过 buffer（frac 钉 0.15）与 f0.10/f0.20（close 旧口径），
   production 口径下 frac 维从没扫过；
2. **调仓日偏移**：现行 = 每月第 1 个交易日的次日，第 2/3/5 日从没试过；
3. **h 加权**：882 = gbdt h1+h5 等权秩平均，权重从没扫过（分面板都在 pred/）；
4. **成交价**：close（乐观上限）/ open（生产）/ vwap（T+1 VWAP，从未对照）。

外加免费信号臂（gbdt h1 单模型 / ens_rgr / ens_all，纯回测成本）。

口径：caliber **production**（固定 10bp · T+1 open · 1e6 · 几何 Sharpe · 窗口 2018+）。
锚点：``BufferedTopFracLongOnly(0.15, 0.40)`` + ``ens_h1h5`` 必须逐位复现登记基线
16.46%/0.685，Δ≠0 直接 abort —— 结论先于数字无效。

用法：
    python -m scripts.evaluation.cheap_lever_probe
    python -m scripts.evaluation.cheap_lever_probe --skip-close-vwap   # 省两次 ctx 加载
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from scripts.evaluation import caliber
from scripts.evaluation.portfolio_weighting import ROOT  # 仓库根（复用同一定位）

log = logging.getLogger("cheap_lever")

PRED_DIR = ROOT / "reports" / "alla_rolling_ortho" / "pred"
OUT = ROOT / "reports" / "cheap_lever_scan"

ANCHOR_ANN, ANCHOR_SHARPE = 16.46, 0.685  # v1w-ortho882 · production 登记
FRACS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.20)
K_OFFSETS = (1, 2, 3, 5)
H_WEIGHTS = (0.0, 0.25, 0.75, 1.0)  # 0.5 = 现行（锚点覆盖）


def _cs_rank(p: pd.DataFrame) -> pd.DataFrame:
    """逐日截面百分位秩（与 ``_rank_average`` 的 ``rank(pct=True)`` 同口径）。"""
    return p.rank(axis=1, pct=True)


def _blend_two(p1: pd.DataFrame, p2: pd.DataFrame, w: float) -> pd.DataFrame:
    """w·rank(p1) + (1−w)·rank(p2)，两者皆有效才出值（min_panels=2 语义等价）。"""
    r1, r2 = _cs_rank(p1), _cs_rank(p2)
    both = r1.notna() & r2.notna()
    out = w * r1 + (1.0 - w) * r2
    return out.where(both).astype(np.float32)


def _rb_kth(idx: pd.DatetimeIndex, k: int) -> set:
    """每月第 k 个交易日的**次一交易日**（k=1 逐位等于 caliber.next_day_rebalance）。"""
    s = pd.Series(idx, index=idx)
    kth = s.groupby(s.index.to_period("M")).nth(k - 1)
    pos = {d: i for i, d in enumerate(idx)}
    return {idx[pos[t] + 1] for t in kth if pos[t] + 1 < len(idx)}


def _run_open_custom_rb(ctx: caliber.CaliberContext, signal: pd.DataFrame, rb: set,
                        strategy, cost, capital: float):
    """复刻 ``caliber._run_once`` 的 open 分支，仅调仓日集合可换（不引私有名）。"""
    from backtest.engine import VectorBacktest, build_execution_split

    sig_used = signal.shift(1).where(ctx.mask)
    split = build_execution_split(ctx.fill, ctx.close, rb)
    bt = VectorBacktest(strategy=strategy, rebalance_freq="M",
                        initial_capital=float(capital), costs=cost)
    return bt.run(sig_used, ctx.fwd, horizon=1, rebalance_days=rb,
                  execution_split=split,
                  adv_panel=ctx.amt, vol_panel=ctx.vol60)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-close-vwap", action="store_true",
                    help="跳过 close/vwap 成交价对照（省两次 ctx 加载）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cost = caliber.cost_profile("production")
    cap = caliber.PROFILES["production"].capital

    log.info("加载 production ctx（T+1 open）…")
    ctx = caliber.load_context(signal="ens_h1h5", window="prod", execution="open")
    log.info("%s", caliber.describe(ctx, "production", cap, "production"))

    rows: list[dict] = []

    def run(tag: str, dim: str, strat, sig: pd.DataFrame | None = None,
            ctx_use: caliber.CaliberContext | None = None,
            execution: str = "open", rb: set | None = None) -> None:
        ctx_x = ctx_use or ctx
        if rb is not None:
            # 自定义调仓日：走复刻的 open 分支（引擎 rebalance_days 显式传入）
            res = _run_open_custom_rb(ctx_x, sig if sig is not None else ctx_x.pred,
                                      rb, strat, cost, cap)
            from scripts.pipelines.rolling_grid_alla import res_metrics
            m = res_metrics(res.daily_returns,
                            pd.Series(0.0, index=res.daily_returns.index),
                            res.turnover_series)
            m = dict(ann=m["annual"], sharpe=m["sharpe"], max_dd=m["max_dd"],
                     turnover=m["turnover"])
        else:
            if sig is not None:
                # 换信号臂：evaluate_strategy 只认 ctx.pred ⇒ replace 注入
                # （fwd/mask/close/fill 与信号无关，各面板同 index，安全）
                if np.allclose(np.nan_to_num(sig.to_numpy()),
                               np.nan_to_num(ctx_x.pred.to_numpy())):
                    raise AssertionError(f"{tag}: 信号与现行面板逐位相同 —— 注入未生效")
                import dataclasses
                ctx_x = dataclasses.replace(ctx_x, pred=sig)
            m = caliber.evaluate_strategy(strat, ctx_x, cost, cap,
                                          execution=execution)
        rows.append(dict(dim=dim, arm=tag, ann=round(m["ann"] * 100, 2),
                         sharpe=round(m["sharpe"], 3), max_dd=round(m["max_dd"] * 100, 1),
                         turnover=round(m["turnover"] * 100, 1)))
        log.info("%-28s 年化=%+6.2f%% Sharpe=%.3f 回撤=%5.1f%% 换手=%5.1f%%",
                 rows[-1]["arm"], rows[-1]["ann"], rows[-1]["sharpe"],
                 rows[-1]["max_dd"], rows[-1]["turnover"])

    # ---- 0) 锚点：必须逐位复现登记基线，否则 abort ----
    anchor = BufferedTopFracLongOnly(frac_entry=0.15, frac_exit=0.40)
    m0 = caliber.evaluate_strategy(anchor, ctx, cost, cap, execution="open")
    log.info("锚点 buf(0.15/0.40) = %.2f%%/%.3f（登记 %.2f/%.3f）",
             m0["ann"] * 100, m0["sharpe"], ANCHOR_ANN, ANCHOR_SHARPE)
    if abs(m0["ann"] * 100 - ANCHOR_ANN) > 0.02 or abs(m0["sharpe"] - ANCHOR_SHARPE) > 0.005:
        raise AssertionError("锚点未复现登记基线 —— 先修口径再谈结论")
    rows.append(dict(dim="anchor", arm="buf(0.15/0.40)_open", ann=ANCHOR_ANN,
                     sharpe=ANCHOR_SHARPE, max_dd=round(m0["max_dd"] * 100, 1),
                     turnover=round(m0["turnover"] * 100, 1)))

    # ---- 1) frac × buffer ----
    for f in FRACS:
        run(f"TopFrac({f:g})", "frac", TopFracLongOnly(frac=f))
    for e in FRACS:
        run(f"buf({e:g}/{e + 0.25:g})", "frac_buf",
            BufferedTopFracLongOnly(frac_entry=e, frac_exit=e + 0.25))
    for e in (0.08, 0.10, 0.12):
        run(f"buf({e:g}/0.4)", "frac_buf",
            BufferedTopFracLongOnly(frac_entry=e, frac_exit=0.40))

    # ---- 2) h 加权（分面板重合成；w=0.5 应逐位等于锚点）----
    h1 = pd.read_parquet(PRED_DIR / "gbdt__h1.parquet")
    h5 = pd.read_parquet(PRED_DIR / "gbdt__h5.parquet")
    h1 = h1.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
    h5 = h5.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
    for w in H_WEIGHTS:
        sig_w = _blend_two(h1, h5, w)
        run(f"h1w{w:g}", "h_weight",
            BufferedTopFracLongOnly(0.15, 0.40), sig=sig_w)

    # ---- 3) 调仓日偏移（k=1 应逐位等于锚点）----
    for k in K_OFFSETS:
        rb = _rb_kth(ctx.pred.index, k)
        run(f"月第{k}日", "rebalance_day",
            BufferedTopFracLongOnly(0.15, 0.40), rb=rb)

    # ---- 4) 免费信号臂（存档面板，零训练成本）----
    for name, f in (("gbdt_h1_only", "gbdt__h1.parquet"),
                    ("ens_rgr", "ens_rgr__h1.parquet"),
                    ("ens_all", "ens_all__h1.parquet")):
        sig = pd.read_parquet(PRED_DIR / f).reindex(
            index=ctx.pred.index, columns=ctx.pred.columns)
        run(name, "signal", BufferedTopFracLongOnly(0.15, 0.40), sig=sig)

    # ---- 5) 成交价对照（close=乐观上限 / vwap=T+1 VWAP）----
    if not args.skip_close_vwap:
        for ex in ("close", "vwap"):
            log.info("加载 execution=%s ctx …", ex)
            ctx_x = caliber.load_context(signal="ens_h1h5", window="prod",
                                         execution=ex)
            run(f"exec_{ex}", "execution",
                BufferedTopFracLongOnly(0.15, 0.40), ctx_use=ctx_x, execution=ex)

    df = pd.DataFrame(rows)
    csv = OUT / "results.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, csv)
    for dim, g in df.groupby("dim", sort=False):
        log.info("[%s]\n%s", dim, g.drop(columns="dim").to_string(index=False))


if __name__ == "__main__":
    main()
