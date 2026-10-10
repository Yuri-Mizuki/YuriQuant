"""业绩预告 / 快报事件因子探针（P0 新信息源：PEAD / SUE 口径）。

为什么是这条线
--------------
P0 边际增益曲线已证「量价池 ≥36 即平台、组合层仅 +0.63pp」⇒ 存量维度榨干，
增量须来自**新信息源或新归纳偏置**。项目已三次「补数据」负结果（alt 三族 0/36、
news 族杠杆 0、标签三臂），但那三次都是**再买/再抓数据**；本探针走另一条路：
**用本地已有数据造新信息**——`sdk_cache/infodata/` 里躺着 12.6 万条
**业绩预告** + 2.9 万条**业绩快报**，项目从未把它们做成因子。

新在何处（相对于池内 130 个非量价因子）
---------------------------------------
1. **信息早于财报**：预告比定期报告早 1~2 个月，且是**前瞻**数字；池内基本面因子
   全部来自已披露的定期报告。
2. **新归纳偏置 = 事件日条件化**：池内因子都是「状态量」，本族是「事件 +
   事件后漂移窗」——这正是 PEAD（盈余公告后漂移）的机制，也是 §五 列的空白方向。

口径（与主实验对齐）
--------------------
h=1 日频 · M 月频调仓 · equal 等权 · `default_costs` · 掩码 = `_base/tradable_mask`
（T+1 口径）· 训练标签 = `close_adj.pct_change(1).shift(-1)`（IC / 因子库口径）·
喂回测引擎的 fwd 用未 shift 的 `pct_change`（引擎 h=1 约定）。

PIT 纪律
--------
- 事件日 = `FIRST_ANN_DATE`（首次预告日，避开修正污染；缺失回退 `ANN_DATE`）。
- 面板在**公告日之后的第一个交易日**起生效（`side="right"`）——确保 T 日收盘前
  已知，杜绝「当天公告当天成交」的纸面收益。
- 市值用**事件日当天**的 `market_cap`（同日对齐，非未来值）。

判据（项目铁律：只有可交易口径 IC 才判真假）
--------------------------------------------
1. 单因子：全样本 IC vs **可交易 IC**（复用 `news_alpha_probe.masked_ic`）
2. 独立建模：PEAD-only 滚动 LGBM 预测的可交易 IC
3. **组合层**：与主模型预测 rank 融合后，年化 / Sharpe / 换手是否改善
   （这是 alt / news 都倒在的那一关）

用法
----
    python -m scripts.evaluation.pead_alpha_probe                 # 全量
    python -m scripts.evaluation.pead_alpha_probe --oos-begin 20240101   # 快速窗
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("pead_alpha_probe")

SDK_INFO = Path(r"E:/data/sdk_cache/infodata")
DRIFT_W = 60          # PEAD 漂移窗（交易日）
CHG_CLIP = (-100.0, 500.0)   # 预告变动幅度剪裁区间（%）


# ---------------------------------------------------------------------------
# 事件 → 面板
# ---------------------------------------------------------------------------
def _to_dt(s: pd.Series) -> pd.Series:
    """YYYYMMDD（int/str/float）→ datetime；本项目 PIT 铁律：必须 str + format。"""
    return pd.to_datetime(
        s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def _event_trading_pos(events: pd.DataFrame, days: pd.DatetimeIndex,
                       cols: pd.Index, code_idx: dict) -> tuple:
    """事件长表 → (交易日下标数组, 代码下标数组, 值数组)。

    公告日映射到**之后的第一个交易日**（``side="right"``）：确保 T 日收盘前
    已知，杜绝「当天公告当天成交」的纸面收益。
    """
    e = events.dropna(subset=["code", "date", "value"])
    e = e[e["code"].isin(code_idx)].sort_values("date")
    ci = e["code"].map(code_idx).astype(int).to_numpy()
    pi = np.searchsorted(days.to_numpy(), e["date"].to_numpy(), side="right")
    ok = pi < len(days)
    return pi[ok], ci[ok], e["value"].to_numpy(dtype=np.float32)[ok]


def _events_to_panel(events: pd.DataFrame, days: pd.DatetimeIndex,
                     cols: pd.Index, window: int | None,
                     code_idx: dict) -> pd.DataFrame:
    """(code, event_date, value) 长表 → date×code 面板。

    ``window`` 非空 = 事件后 ``window`` 个交易日内有效（PEAD 漂移窗）；
    为空 = 持有到下一次事件（step function）。
    """
    ev = np.full((len(days), len(cols)), np.nan, dtype=np.float32)
    pi, ci, vv = _event_trading_pos(events, days, cols, code_idx)
    ev[pi, ci] = vv
    dense = pd.DataFrame(ev, index=days, columns=cols).ffill()
    if window is None:
        return dense

    # 事件新鲜度 = 距最近一次事件的交易日数（向量化）
    pos = np.where(~np.isnan(ev), np.arange(len(days))[:, None], -1)
    age = np.arange(len(days))[:, None] - np.maximum.accumulate(pos, axis=0)
    out = np.where(age <= window, dense.to_numpy(), np.nan)
    return pd.DataFrame(out.astype(np.float32), index=days, columns=cols)


def build_pead_panels(days: pd.DatetimeIndex, cols: pd.Index,
                      market_cap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """构造 PEAD / SUE 面板族（事件日 + 漂移窗 / 持续两种口径）。"""
    out: dict[str, pd.DataFrame] = {}
    code_idx = {c: i for i, c in enumerate(cols)}
    mc = market_cap.reindex(index=days, columns=cols)

    def emit(tag: str, ev: pd.DataFrame) -> None:
        ev = ev.dropna(subset=["value"])
        for win, sfx in ((DRIFT_W, "drift"), (None, "persist")):
            out[f"{tag}_{sfx}"] = _events_to_panel(ev, days, cols, win, code_idx)
        log.info("  → %s_{drift,persist}（%d 条事件）", tag, len(ev))

    # ---------------- 业绩预告 profit_notice ----------------
    pn = pd.read_hdf(SDK_INFO / "profit_notice" / "profit_notice.h5")
    pn = pn.rename(columns={"MARKET_CODE": "code"})
    pn["ann"] = _to_dt(pn["ANN_DATE"])
    pn["date"] = _to_dt(pn["FIRST_ANN_DATE"]).fillna(pn["ann"])   # 事件日=首次公告
    pn["rp"] = pn["REPORTING_PERIOD"].astype(str)
    pn = pn.dropna(subset=["date"]).sort_values(["code", "rp", "ann"])
    n0 = len(pn)
    pn = pn.groupby(["code", "rp"], sort=False).head(1)   # 首次预告（避开修正）
    log.info("业绩预告 %d 行 → 去重后 %d 条事件（code×报告期）", n0, len(pn))

    # 连续型惊喜：同比变动幅度（下限优先，缺失回退上限）
    chg = pn["P_CHANGE_MIN"].where(pn["P_CHANGE_MIN"].notna(), pn["P_CHANGE_MAX"])
    log.info("变动幅度覆盖率 %.1f%%（缺失样本多为扭亏/不确定，百分比无意义）",
             float(chg.notna().mean()) * 100)
    emit("pead_pn_chg", pn[["code", "date"]].assign(value=chg.clip(*CHG_CLIP)))

    # 预测盈利收益率 = 预告净利润中值 / 事件日市值（万元 → 元）
    np_mid = (pn["NET_PROFIT_MIN"].fillna(pn["NET_PROFIT_MAX"])
              + pn["NET_PROFIT_MAX"].fillna(pn["NET_PROFIT_MIN"])) / 2.0
    np_mid = np_mid.where(pn["NET_PROFIT_MIN"].notna() | pn["NET_PROFIT_MAX"].notna())
    ev = pn[["code", "date"]].assign(value=np_mid)
    dpos = mc.index.get_indexer(ev["date"].to_numpy())
    cpos = mc.columns.get_indexer(ev["code"].to_numpy())
    ok = (dpos >= 0) & (cpos >= 0)
    mcv = np.full(len(ev), np.nan)
    mcv[ok] = mc.to_numpy()[dpos[ok], cpos[ok]]
    emit("pead_pn_ey", ev.assign(value=ev["value"].to_numpy() * 1e4 / mcv))

    # ---------------- 业绩快报 profit_express ----------------
    pe = pd.read_hdf(SDK_INFO / "profit_express" / "profit_express.h5")
    pe = pe.rename(columns={"MARKET_CODE": "code"})
    pe["date"] = _to_dt(pe["ACTUAL_ANN_DATE"]).fillna(_to_dt(pe["ANN_DATE"]))
    pe["rp"] = pe["REPORTING_PERIOD"].astype(str)
    pe = pe.dropna(subset=["date"]).sort_values(["code", "rp", "date"])
    pe = pe.groupby(["code", "rp"], sort=False).tail(1)   # 快报=实际数，取最新
    emit("pead_pe_yoy", pe[["code", "date"]].assign(
        value=pe["YOY_GR_NET_PROFIT_PARENT"].clip(*CHG_CLIP)))

    return out


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred")
    ap.add_argument("--main-signal", default="ens_h1h5__h1.parquet")
    ap.add_argument("--out", default="reports/pead_probe")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--refit-every", type=int, default=20)
    ap.add_argument("--oos-begin", default=None)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--skip-model", action="store_true",
                    help="只跑单因子判据（不跑 PEAD-only 建模与融合）")
    args = ap.parse_args(argv)

    from scripts.evaluation.news_alpha_probe import (  # 复用已固化的判据
        fuse_rank,
        ic_stats,
        masked_ic,
        rolling_predict,
    )
    from scripts.pipelines.rolling_grid_alla import load_base
    from stats.ic import calc_ic_series

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = load_base()
    close, mask = base["close"], base["mask"].astype(bool)
    all_days, cols = close.index, close.columns
    main = pd.read_parquet(Path(args.pred_dir) / args.main_signal)
    oos_days = main.index
    if args.oos_begin:
        oos_days = oos_days[oos_days >= pd.Timestamp(args.oos_begin)]
    log.info("主模型 %s: %d 日 | OOS %d 日（%s ~ %s）", args.main_signal,
             main.shape[0], len(oos_days), oos_days[0].date(), oos_days[-1].date())

    # ---------------- 1) 造面板 ----------------
    panels = build_pead_panels(all_days, cols, base["market_cap"])
    log.info("PEAD 面板 %d 个：%s", len(panels), list(panels))
    pdir = dest / "panels"
    pdir.mkdir(exist_ok=True)
    for n, p in panels.items():
        p.to_parquet(pdir / f"{n}.parquet")

    # ---------------- 2) 单因子判据：全样本 vs 可交易 ----------------
    labels = close.pct_change(args.horizon, fill_method=None).shift(-args.horizon)
    fwd_ic = labels.reindex(index=oos_days, columns=cols)
    mask_oos = mask.reindex(index=oos_days, columns=cols).fillna(True)
    main_oos = main.reindex(index=oos_days, columns=cols)

    rows = [dict(signal="主模型(基准)", cov=1.0,
                 **{f"full_{k}": v for k, v in
                    ic_stats(calc_ic_series(main_oos, fwd_ic)).items()},
                 **{f"trad_{k}": v for k, v in
                    ic_stats(masked_ic(main_oos, fwd_ic, mask_oos)).items()})]
    for n, p in panels.items():
        sig = p.reindex(index=oos_days, columns=cols).replace(
            [np.inf, -np.inf], np.nan)
        cov = float(sig.notna().to_numpy(dtype=float).mean())
        full = ic_stats(calc_ic_series(sig, fwd_ic))
        trad = ic_stats(masked_ic(sig, fwd_ic, mask_oos))
        # 与主模型预测的截面 rank 相关（正交性粗查）
        rc = float(sig.rank(axis=1, pct=True).corrwith(
            main_oos.rank(axis=1, pct=True), axis=1).mean())
        rows.append(dict(signal=n, cov=cov, rank_corr_main=rc,
                         **{f"full_{k}": v for k, v in full.items()},
                         **{f"trad_{k}": v for k, v in trad.items()}))
        log.info("%-26s cov=%.2f 全样本IC=%+.4f | 可交易IC=%+.4f IR=%.2f "
                 "胜率=%.0f%% | rank_corr(main)=%+.3f",
                 n, cov, full["ic"], trad["ic"], trad["ir"], trad["win"] * 100, rc)
    pd.DataFrame(rows).to_csv(dest / "ic_summary.csv", index=False,
                              encoding="utf-8-sig")

    if args.skip_model:
        log.info("完成（--skip-model） %.0fs -> %s", time.time() - t0, dest)
        return

    # ---------------- 3) PEAD-only 滚动建模 ----------------
    feats = {n: p.reindex(index=all_days, columns=cols).replace(
        [np.inf, -np.inf], np.nan).astype(float) for n, p in panels.items()}
    pead_pred = rolling_predict(feats, labels, oos_days, all_days,
                               window=args.window, refit_every=args.refit_every)
    pead_pred.to_parquet(dest / "pead_pred.parquet")
    cov_any = None
    for v in feats.values():
        nv = v.notna()
        cov_any = nv if cov_any is None else (cov_any | nv)
    cov_any = cov_any.reindex(index=oos_days, columns=cols).fillna(False)
    log.info("PEAD 覆盖率（OOS，至少一个特征非 NaN）：%.3f",
             float(cov_any.to_numpy(dtype=float).mean()))

    full = ic_stats(calc_ic_series(pead_pred, fwd_ic))
    trad = ic_stats(masked_ic(pead_pred, fwd_ic, mask_oos))
    log.info("PEAD-only 预测：全样本 IC=%+.4f | 可交易 IC=%+.4f (IR%.2f, 胜率%.0f%%)",
             full["ic"], trad["ic"], trad["ir"], trad["win"] * 100)
    pd.DataFrame([dict(signal="PEAD-only模型",
                       **{f"full_{k}": v for k, v in full.items()},
                       **{f"trad_{k}": v for k, v in trad.items()})]).to_csv(
        dest / "pead_model_ic.csv", index=False, encoding="utf-8-sig")

    # ---------------- 4) 组合层融合 ----------------
    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest
    from scripts.pipelines.rolling_grid_alla import _ann, res_metrics
    from strategy.examples import TopFracLongOnly, TopKLongOnly

    fwd_engine = close.pct_change(fill_method=None).loc[oos_days]
    arms = [("主模型 only", main_oos), ("PEAD only", pead_pred)]
    for w in (0.5, 0.7, 0.8, 0.9):
        arms.append((f"主模型⊕PEAD w={w:.1f}",
                     fuse_rank(main_oos, pead_pred, w, cov_any)))

    costs = default_costs()
    b_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)
    bench_idx = base["bench_index"]
    hold = [("TopFrac0.10", lambda: TopFracLongOnly(frac=0.10, weight_mode="equal")),
            ("TopK45", lambda: TopKLongOnly(k=45))]
    rows2 = []
    for hold_tag, factory in hold:
        for tag, sig in arms:
            sig = sig.reindex(index=oos_days, columns=cols)
            bt = VectorBacktest(strategy=factory(), rebalance_freq="M",
                                initial_capital=1_000_000.0, costs=costs)
            res = bt.run(sig, fwd_engine, executable_mask=mask_oos, horizon=1,
                         rebalance_days=None)
            m = res_metrics(res.daily_returns,
                            bench_idx.reindex(res.daily_returns.index).fillna(0.0),
                            res.turnover_series)
            rows2.append(dict(hold=hold_tag, arm=tag, annual=m["annual"],
                              excess_eqw=m["annual"] - _ann(b_eqw),
                              sharpe=m["sharpe"], max_dd=m["max_dd"],
                              turnover=m["turnover"]))
            log.info("[%s] %-22s 年化=%+6.2f%% Sharpe=%.3f 回撤=%5.1f%% 换手=%5.1f%%",
                     hold_tag, tag, m["annual"] * 100, m["sharpe"],
                     m["max_dd"] * 100, m["turnover"] * 100)
    pd.DataFrame(rows2).to_csv(dest / "fusion_results.csv", index=False,
                               encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, dest)


if __name__ == "__main__":
    main()
