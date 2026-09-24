"""批次 8：表格基础模型对照（TabPFN-3.5 / TabICL 2.2 vs LightGBM）滚动 runner
========================================================================

立项与版本口径真源：``GOOD_MACHINE_TASKS.md`` 批次 8（tabpfn==9.0.0 默认模型
= TabPFN-3.5；tabicl==2.2.0；载荷包 pandas 2.x 钉版警告见彼处）。
冒烟校准（09-22/23，``reports/tabpfn3_smoke/``）：CPU 上 tabpfn3 pred
≈ 27.7min/窗口 → **全A 全量是 GPU 硬前置**（``--device cuda``，单卡 ≥8GB）；
hs300 小池 CPU 可跑（本 runner 默认形态）。

滚动协议（沿用 2026-08-08 v1 实测的「ICL × 滚动短窗口」方向假设）：

- **逐月重训、月内逐日预测**（test_step=1 全量测试日——v1 的遗留高估项
  直接修掉）；训练窗 = 调仓月前最近 ``W`` 个月（W∈{2,3} 扫描；08-08：
  2m 强趋势更强、3m 唯一全正 IR，须复验）；
- 防前视：训练窗尾部截 ``horizon`` 天标签实现期（embargo，与
  ``rolling_oos``/``forward_roll_folds`` 同纪律）；
- 预测器全部走 ``model.predictor.PREDICTORS`` 注册（gbdt / tabpfn /
  tabicl 同一 fit/predict 接口，零分叉）。

评估口径（批次 8 定稿）：

- **可交易 IC**（T+1 掩码防纸面三判据，``data.tradability`` 标准口径）+
  裸 IC 对照；OOS IC / IR / Newey-West t；
- **组合层**：Top10% 等权月频，**T+1 open 执行价**（正名口径，
  ``build_execution_split`` + ``default_costs``，与批次 1 一致）；
- 臂 4（``tabpfn_gbdt``）= 两模型族预测的**秩平均**；同时报告两臂逐日
  预测秩相关均值（<0.3 才有集成增量空间——臂 4 的价值判据）。

预期管理（诚实标注）：TabArena Elo 是通用表格基准 ≠ A 股截面有优势；
v1 在 HS300 静态口径已被证伪过一次（0.029 < gbdt 0.051），本对照检验的
是「滚动短窗口协议下 ICL 优势能否复现」——**负结果同样归档**。

用法::

    P=D:/Python/Python312/python
    # 本机（hs300 小池，CPU）
    $P -m scripts.evaluation.tabpfn_rolling_compare --pool hs300 \
        --begin 20220101 --end 20260921 --test-begin 20240101
    # 好机器（GPU，全A）
    $P -u scripts/evaluation/tabpfn_rolling_compare.py --pool all_a \
        --device cuda --windows 2,3 --out reports/tabpfn_rolling_compare
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import default_costs
from backtest.engine import VectorBacktest, build_execution_split
from scripts.pipelines import rolling_grid_alla as RG
from stats.ic import calc_ic_series
from stats.robust_stats import nw_tstat
from strategy.examples import TopFracLongOnly

log = logging.getLogger("tabpfn_rolling_compare")

SIX = ("open", "high", "low", "close", "volume", "amount")

#: 臂名 → (PREDICTORS 键, 构造 kwargs)。tabpfn/tabicl 缺库时该臂报错退出
#: （Predictor 自带 pip 提示），不用静默跳过——对照实验缺臂即失真。
ARM_SPECS: dict[str, tuple[str, dict]] = {
    "gbdt": ("gbdt", {"learning_rate": 0.03, "num_leaves": 31, "seed": 42}),
    "tabpfn": ("tabpfn", {}),          # TabPFN-3.5（tabpfn 9.0.0 默认权重）
    "tabicl": ("tabicl", {}),          # TabICL 2.2.0
}
ARM_DEFAULTS = {"tabpfn": {"max_context_samples": 6000, "n_estimators": 2},
                "tabicl": {"max_context_samples": 6000, "n_estimators": 2}}


def load_pool_panel(pool: str, begin: int, end: int, offline: bool = True
                    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """复权 OHLCV 宽面板 + 后复权因子（可交易掩码封板判定用）。"""
    from data.cache_helpers import build_real_panel
    from config import Config

    cfg = Config.get()
    cfg["universe"]["default"] = pool
    panel, _ = build_real_panel(cfg, begin, end, offline=offline)
    bwd = None
    try:
        from data.cache import DataCache
        from data.cache_helpers import load_backward_factor
        from data.datasource import create_datasource

        bwd = load_backward_factor(
            DataCache(create_datasource(Config.datasource())),
            list(panel["close"].columns))
    except Exception as exc:
        log.warning("后复权因子读取失败（%s）→ 封板判定退化为复权价口径", exc)
    return panel, bwd


def month_grid(all_days: pd.DatetimeIndex, test_begin, end
               ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """测试段的 (月首日, 月末日) 列表。"""
    days = all_days[(all_days >= pd.Timestamp(test_begin))
                    & (all_days <= pd.Timestamp(end))]
    s = pd.Series(days, index=days)
    return list(zip(s.groupby(days.to_period("M")).first(),
                    s.groupby(days.to_period("M")).last()))


def rolling_train_days(all_days: pd.DatetimeIndex, m_first: pd.Timestamp,
                       window_months: int, embargo: int) -> pd.DatetimeIndex:
    """月 m 的训练窗：[m_first − W 月, m_first) 再截尾部 embargo 天标签实现期。

    ⚠️ 用 ``DateOffset(months=W)`` 而非 ``MonthBegin(W)``——后者从月内日期
    回退会先滚到当月月初，W 个月窗实际只覆盖 W−1 个月（实测 2m 窗只开出
    17~20 个交易日，触发"训练窗不足"跳月）。
    """
    start = m_first - pd.DateOffset(months=window_months)
    days = all_days[(all_days >= start) & (all_days < m_first)]
    if embargo > 0 and len(days) > embargo:
        days = days[:-embargo]
    return days


def run_arm(pred_key: str, pred_kw: dict, feats: dict[str, pd.DataFrame],
            labels: pd.DataFrame, all_days: pd.DatetimeIndex,
            months: list, window_months: int, embargo: int
            ) -> tuple[pd.DataFrame, dict]:
    """单臂逐月滚动：返回 (逐日预测面板, 计时统计)。"""
    from model.predictor import PREDICTORS

    cls = PREDICTORS[pred_key]
    codes = labels.columns
    test_days = all_days[(all_days >= months[0][0]) & (all_days <= months[-1][1])]
    out = pd.DataFrame(np.nan, index=test_days, columns=codes)
    t_fit = t_pred = 0.0
    for m_first, m_last in months:
        tr = rolling_train_days(all_days, m_first, window_months, embargo)
        te = all_days[(all_days >= m_first) & (all_days <= m_last)]
        if len(tr) < 20:
            log.warning("%s %s 训练窗 %d 日不足，跳月", pred_key, m_first.date(), len(tr))
            continue
        p = cls(**pred_kw)
        t0 = time.perf_counter()
        p.fit({k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
        t_fit += time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = p.predict({k: v.loc[te] for k, v in feats.items()})
        t_pred += time.perf_counter() - t0
        out.loc[te] = pred.reindex(index=te, columns=codes).values
    return out, {"fit_s": round(t_fit, 1), "pred_s": round(t_pred, 1),
                 "n_retrain": len(months)}


def tradable_ic_series(pred: pd.DataFrame, fwd: pd.DataFrame,
                       trad: pd.DataFrame | None) -> pd.Series:
    """逐日截面 RankIC（可交易掩码口径；掩码外股票逐日置 NaN）。"""
    if trad is None:
        return calc_ic_series(pred, fwd, method="spearman")
    p = pred.where(trad.reindex(index=pred.index, columns=pred.columns).fillna(False))
    return calc_ic_series(p, fwd, method="spearman")


def rank_blend(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """两臂预测的日截面秩平均（同日同列对齐，双缺记 NaN）。"""
    ra = a.rank(axis=1, pct=True)
    rb = b.rank(axis=1, pct=True)
    return (ra.add(rb, fill_value=np.nan) / 2.0)[
        ra.notna() & rb.notna()]


def evaluate_pred(pred: pd.DataFrame, fwd: pd.DataFrame, trad, close: pd.DataFrame,
                  open_px: pd.DataFrame, test_days: pd.DatetimeIndex,
                  bench_eqw: pd.Series | None = None,
                  rets: pd.DataFrame | None = None) -> dict:
    """单臂单窗的完整指标行：可交易 IC + NW t + Top10% 月频 open 组合。

    口径分层：``fwd`` = shift(-h) 贴标签口径（IC 用）；``rets`` =
    ``close.pct_change()`` 引擎对齐口径（组合回测用，第 i 行 = i-1→i 收益，
    引擎守卫会拦 shift 面板——2026-09-24 实测）。两者不可混用。
    """
    pred = pred.reindex(index=test_days)
    fwd_t = fwd.reindex(index=test_days)
    rets_t = (rets if rets is not None else close.pct_change()
              ).reindex(index=test_days)
    ic_raw = calc_ic_series(pred, fwd_t, method="spearman").dropna()
    ic_tr = tradable_ic_series(pred, fwd_t, trad).dropna()
    nw, _se, _lag = nw_tstat(ic_tr.to_numpy(dtype=float)) if len(ic_tr) > 2 \
        else (float("nan"), None, None)

    sig = pred.copy()
    # T+1 open 执行（正名口径）：信号 T → T+1 开盘成交
    s_ = pd.Series(test_days, index=test_days)
    firsts = s_.groupby(test_days.to_period("M")).first()
    pos_ = {d: i for i, d in enumerate(test_days)}
    rb_exec = {test_days[pos_[t] + 1] for t in firsts
               if pos_[t] + 1 < len(test_days)}
    exec_split = build_execution_split(
        open_px.reindex(index=test_days, columns=close.columns), close, rb_exec)
    mask = trad.reindex(index=test_days, columns=close.columns).fillna(False) \
        if trad is not None else None
    strat = TopFracLongOnly(frac=0.10, weight_mode="equal")
    bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                        initial_capital=1_000_000.0, costs=default_costs())
    res = bt.run(sig.shift(1).where(mask) if mask is not None else sig.shift(1),
                 rets_t, horizon=1, rebalance_days=rb_exec,
                 execution_split=exec_split)
    m = RG.res_metrics(res.daily_returns, bench_eqw, res.turnover_series) \
        if bench_eqw is not None else {
            "annual": float("nan"), "sharpe": float("nan"),
            "max_dd": float("nan"), "turnover": float(res.turnover_series.mean())}
    return {
        "ic_raw": float(ic_raw.mean()) if len(ic_raw) else float("nan"),
        "ic_tradable": float(ic_tr.mean()) if len(ic_tr) else float("nan"),
        "ic_ir_tradable": (float(ic_tr.mean() / ic_tr.std() * np.sqrt(244))
                           if len(ic_tr) > 2 and ic_tr.std() > 0 else float("nan")),
        "nw_t": float(nw), "n_days": int(len(ic_tr)),
        "annual": m["annual"], "sharpe": m["sharpe"],
        "max_dd": m["max_dd"], "turnover": m["turnover"],
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="hs300", choices=["hs300", "all_a"])
    ap.add_argument("--begin", type=int, default=20220101)
    ap.add_argument("--end", type=int, default=20260921)
    ap.add_argument("--test-begin", type=int, default=20240101,
                    help="OOS 起点（此前数据只用于首个训练窗）")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--windows", default="2,3", help="滚动窗月数（逗号分隔）")
    ap.add_argument("--arms", default="gbdt,tabpfn,tabicl,tabpfn_gbdt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="tabpfn/tabicl 推理设备（全A 须 cuda）")
    ap.add_argument("--n-estimators", type=int, default=2)
    ap.add_argument("--context", type=int, default=6000)
    ap.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default="reports/tabpfn_rolling_compare")
    ap.add_argument("--save-preds", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== 批次 8 表格基础模型对照：pool=%s device=%s ===", args.pool,
             args.device)
    panel, bwd = load_pool_panel(args.pool, args.begin, args.end, args.offline)
    panel = {k: panel[k] for k in SIX if k in panel}

    from data.tradability import build_tradable_mask

    from factor.classic import compute_classic_features
    from model.labels import build_labels, forward_returns

    close = panel["close"]
    feats = compute_classic_features(panel)
    labels, _ = build_labels(close, horizon=args.horizon, mode="rank")
    fwd = forward_returns(close, horizon=args.horizon)
    trad = build_tradable_mask(close, bwd=bwd)
    all_days = labels.index
    months = month_grid(all_days, str(args.test_begin), str(args.end))
    test_days = all_days[(all_days >= pd.Timestamp(str(args.test_begin)))
                         & (all_days <= pd.Timestamp(str(args.end)))]
    rets = close.pct_change()                       # 引擎对齐口径
    bench_eqw = rets.where(trad).mean(axis=1).reindex(test_days).fillna(0.0)
    log.info("面板 %s × %s；测试 %d 日 / %d 个月窗",
             *(close.shape, len(test_days), len(months)))

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows: list[dict] = []
    preds_store: dict[tuple[str, int], pd.DataFrame] = {}
    for w in [int(x) for x in args.windows.split(",")]:
        for arm in arms:
            t0 = time.perf_counter()
            if arm == "tabpfn_gbdt":
                need = ["gbdt", "tabpfn"]
                missing = [a for a in need if a not in arms]
                if missing:
                    log.warning("臂 %s 依赖 %s（本次未跑，跳过）", arm, missing)
                    continue
                a_pred = preds_store[("gbdt", w)]
                b_pred = preds_store[("tabpfn", w)]
                pred = rank_blend(a_pred, b_pred)
                timing = {"fit_s": 0.0, "pred_s": 0.0, "n_retrain": 0}
                rk = pred.rank(axis=1, pct=True)
                rk_a = a_pred.rank(axis=1, pct=True)
                day_corr = pd.DataFrame({
                    d: rk.loc[d].corr(rk_a.loc[d])
                    for d in rk.index if rk.loc[d].notna().sum() > 5},
                    index=["r"]).T["r"]
                rank_corr = float(day_corr.mean())
            else:
                key, base = ARM_SPECS[arm]
                if arm == "gbdt":
                    kw = base
                else:
                    kw = {**ARM_DEFAULTS.get(arm, {}),
                          "max_context_samples": args.context,
                          "n_estimators": args.n_estimators, **base,
                          "device": args.device}
                pred, timing = run_arm(
                    key, kw, feats, labels, all_days, months, w,
                    embargo=args.horizon)
                preds_store[(arm, w)] = pred
                rank_corr = float("nan")
            metrics = evaluate_pred(pred, fwd, trad, close, panel["open"],
                                    test_days, bench_eqw, rets=rets)
            row = {"arm": arm, "window_m": w, **metrics,
                   "rankcorr_vs_gbdt": rank_corr,
                   "wall_s": round(time.perf_counter() - t0, 1), **timing}
            rows.append(row)
            log.info("[arm=%s W=%dm] tradableIC=%.4f IR=%.2f NWt=%.2f "
                     "年化=%.2f%% Sharpe=%.2f 换手=%.1f%% (%.0fs)",
                     arm, w, metrics["ic_tradable"], metrics["ic_ir_tradable"],
                     metrics["nw_t"], metrics["annual"] * 100, metrics["sharpe"],
                     metrics["turnover"] * 100, row["wall_s"])
            if args.save_preds:
                pred.to_parquet(out_dir / f"pred_{arm}_w{w}.parquet")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    meta = {"pool": args.pool, "device": args.device, "begin": args.begin,
            "end": args.end, "test_begin": args.test_begin,
            "horizon": args.horizon, "windows": args.windows, "arms": args.arms,
            "tabpfn_version": _pkg_version("tabpfn"),
            "tabicl_version": _pkg_version("tabicl")}
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("产物落盘 %s（%d 行）", out_dir / "summary.csv", len(df))


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


if __name__ == "__main__":
    main()
