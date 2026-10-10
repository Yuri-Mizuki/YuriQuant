"""news 族独立建模 + 可交易口径验真伪 + 组合层融合（2026-10-10，P0 路径）。

背景
----
``reports/altf_p1`` 称 news 族「含独立信息、有增量」，但依据是**全样本截面 IC**
（h20 |IC| 0.034~0.040），**未过可交易口径筛查**——项目铁律：只有可交易口径 IC
才判真假，全样本 IC 可能是纸面。且实测 news 以旁路特征进 GBDT 时两臂 pred 秩相关
0.995+（模型基本没用它，见 ``reports/altf_p1/model_ablation/``）。

本脚本回答两问
--------------
1. **验真伪**：news 族单独建模的 OOS 预测，在**可交易口径**（T+1 可成交，剔停牌 /
   涨跌停一字板 / ST）下还剩多少 IC？
2. **能否变现**：news-only 预测与主模型预测在**组合层**融合（截面 rank 加权），
   年化 / Sharpe / 换手相对主模型怎么变？

口径（与主实验对齐）
--------------------
h=1 日频 · M 月频调仓 · equal 等权 · ``default_costs`` ·
掩码 = ``_base/tradable_mask``（T+1 口径）· 训练标签 = ``close_adj.pct_change(1)
.shift(-1)``（IC / 因子库口径）。喂给回测引擎的 fwd 用**未 shift** 的 ``pct_change``
（引擎 h=1 约定，见项目记忆）。

用法
----
    python -m scripts.evaluation.news_alpha_probe
    python -m scripts.evaluation.news_alpha_probe --oos-begin 20240101   # 快速窗
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

log = setup_logging("news_alpha_probe")

# news 族 6 个截面面板（factor_library/all_a_2018_2026/panels）
NEWS_PANELS = (
    "alt_news_attention_5d",
    "alt_news_cnt_5d",
    "alt_news_red_cnt_20d",
    "alt_news_cnt_20d_z",
    "alt_news_mean_len_20d",
    "alt_news_lexsent_20d",
)


def load_news_features(idx: pd.DatetimeIndex, cols: pd.Index,
                       names=NEWS_PANELS) -> dict[str, pd.DataFrame]:
    """载入 news 族面板并对齐到 (idx, cols) 网格（NaN 保留，LGBM 原生处理）。"""
    from scripts.pipelines.rolling_grid_alla import ds_root

    pdir = ds_root() / "panels"
    out: dict[str, pd.DataFrame] = {}
    for n in names:
        fp = pdir / f"{n}.parquet"
        if not fp.exists():
            log.warning("缺面板，跳过: %s", fp)
            continue
        p = pd.read_parquet(fp).reindex(index=idx, columns=cols)
        p = p.replace([np.inf, -np.inf], np.nan).astype(float)
        out[n] = p
    if not out:
        raise SystemExit(f"news 面板一个都没找到：{pdir}")
    log.info("news 特征 %d 个：%s", len(out), list(out))
    return out


def rolling_predict(features: dict[str, pd.DataFrame], labels: pd.DataFrame,
                    oos_days: pd.DatetimeIndex, all_days: pd.DatetimeIndex,
                    window: int = 500, refit_every: int = 20, embargo: int = 2,
                    min_train: int = 250) -> pd.DataFrame:
    """walk-forward 滚动预测：每 ``refit_every`` 日重训一次，训练用测试段之前
    ``window`` 日（并留 ``embargo`` 隔离带），与生产前推纪律一致。"""
    from model.predictor import LGBMPredictor

    out = pd.DataFrame(np.nan, index=oos_days, columns=labels.columns)
    n_fold = n_ok = 0
    for i in range(0, len(oos_days), refit_every):
        chunk = oos_days[i:i + refit_every]
        n_fold += 1
        cut = all_days[all_days < chunk[0]]
        if embargo and len(cut) > embargo:
            cut = cut[:-embargo]
        train_days = cut[-window:]
        if len(train_days) < min_train:
            continue
        m = LGBMPredictor()
        m.fit({k: v.loc[train_days] for k, v in features.items()},
              labels.loc[train_days])
        out.loc[chunk] = m.predict({k: v.loc[chunk] for k, v in features.items()})
        n_ok += 1
    log.info("滚动预测：%d 折（有效 %d），OOS 覆盖 %d 日",
             n_fold, n_ok, int(out.notna().any(axis=1).sum()))
    return out


def masked_ic(pred: pd.DataFrame, fwd: pd.DataFrame, mask: pd.DataFrame,
              min_n: int = 20) -> pd.Series:
    """可交易口径逐日 Spearman IC：仅用 mask==True 且 pred/fwd 均非 NaN 的样本。

    mask 为 T+1 成交口径（``build_tradable_mask``），配 fwd[T] = T→T+1 收益，
    故过滤用同日 mask[T]（=信号日 T 判定的、T+1 能否成交）。
    """
    mask = mask.reindex(index=pred.index, columns=pred.columns).fillna(False)
    mask = mask.astype(bool)
    vals: dict[pd.Timestamp, float] = {}
    fwd = fwd.reindex(index=pred.index, columns=pred.columns)
    for dt in pred.index:
        pv = pred.loc[dt].to_numpy(dtype=float)
        fv = fwd.loc[dt].to_numpy(dtype=float)
        mv = mask.loc[dt].to_numpy(dtype=bool)
        ok = ~np.isnan(pv) & ~np.isnan(fv) & mv
        if int(ok.sum()) >= min_n:
            vals[dt] = float(pd.Series(pv[ok]).corr(pd.Series(fv[ok]),
                                                    method="spearman"))
    return pd.Series(vals).sort_index()


def fuse_rank(main: pd.DataFrame, news: pd.DataFrame, w: float,
              cov: pd.DataFrame | None = None) -> pd.DataFrame:
    """截面 rank 加权融合：w·rank(main) + (1-w)·rank(news)。

    只在 ``cov``（news 特征真正有值的样本）上做调整，未覆盖处保持纯
    ``rank(main)`` —— news 族覆盖率仅 ~0.61，未覆盖股票不应被「缺失分支」的
    常数预测牵连相对排序。
    """
    rm = main.rank(axis=1, pct=True)
    rn = news.rank(axis=1, pct=True)
    if cov is None:
        cov = rn.notna() & rm.notna()
    cov = (cov.reindex(index=rm.index, columns=rm.columns).fillna(False)
           & rm.notna() & rn.notna())
    fused = w * rm + (1.0 - w) * rn
    return fused.where(cov, rm)


def ic_stats(s: pd.Series) -> dict:
    s = s.dropna()
    if not len(s):
        return dict(n=0, ic=float("nan"), ir=float("nan"), win=float("nan"))
    return dict(n=int(len(s)), ic=float(s.mean()),
                ir=float(s.mean() / s.std()) if s.std() else float("nan"),
                win=float((s > 0).mean()))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred")
    ap.add_argument("--main-signal", default="ens_h1h5__h1.parquet",
                    help="主模型信号文件名（默认 882 ortho 主实验）")
    ap.add_argument("--out", default="reports/news_alpha_probe")
    ap.add_argument("--window", type=int, default=500, help="训练滚动窗口（交易日）")
    ap.add_argument("--refit-every", type=int, default=20, help="重训间隔（交易日）")
    ap.add_argument("--oos-begin", default=None, help="OOS 起点 YYYYMMDD（快速窗）")
    ap.add_argument("--horizon", type=int, default=1,
                    help="训练/IC 的标签 horizon（news 族宣称在 h20 有效）")
    ap.add_argument("--out-tag", default="",
                    help="产出子目录后缀（多 horizon 结果分开放）")
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest
    from scripts.pipelines.rolling_grid_alla import _ann, load_base, res_metrics
    from stats.ic import calc_ic_series
    from strategy.examples import TopFracLongOnly, TopKLongOnly

    dest = Path(args.out) / args.out_tag if args.out_tag else Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base = load_base()
    close, mask = base["close"], base["mask"].astype(bool)
    bench_idx, bench_eqw = base["bench_index"], base["bench_eqw"]
    all_days = close.index

    main = pd.read_parquet(Path(args.pred_dir) / args.main_signal)
    oos_days = main.index
    if args.oos_begin:
        oos_days = oos_days[oos_days >= pd.Timestamp(args.oos_begin)]
    log.info("主模型 %s: %d 日 · OOS 段 %d 日（%s ~ %s）", args.main_signal,
             main.shape[0], len(oos_days), oos_days[0].date(), oos_days[-1].date())

    # 训练标签 / IC 收益（IC 口径：shift(-h)，h 由 --horizon 给定）
    labels = close.pct_change(args.horizon, fill_method=None).shift(-args.horizon)
    # 引擎 fwd（引擎 h=1 约定：未 shift）
    fwd_engine = close.pct_change(fill_method=None).loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)

    feats = load_news_features(all_days, close.columns)
    news_cov = None
    for v in feats.values():
        nv = v.notna()
        news_cov = nv if news_cov is None else (news_cov | nv)
    news_cov = (news_cov.reindex(index=oos_days, columns=close.columns)
                .fillna(False))
    log.info("news 覆盖率（OOS，至少一个特征非 NaN）：%.3f",
             float(news_cov.to_numpy(dtype=float).mean()))
    news_pred = rolling_predict(feats, labels, oos_days, all_days,
                                window=args.window, refit_every=args.refit_every)
    news_pred.to_parquet(dest / "news_pred.parquet")

    # ---------------- 验真伪：IC（全样本 vs 可交易口径） ----------------
    fwd_ic = labels.reindex(index=oos_days, columns=close.columns)
    ic_rows = []
    for name, sig in [("主模型 " + args.main_signal, main.reindex(oos_days)),
                      ("news-only", news_pred)]:
        full = ic_stats(calc_ic_series(sig, fwd_ic))
        trad = ic_stats(masked_ic(sig, fwd_ic, mask_oos))
        ic_rows.append(dict(signal=name, **{f"full_{k}": v for k, v in full.items()},
                            **{f"trad_{k}": v for k, v in trad.items()}))
        log.info("%-28s 全样本 IC=%+.4f (IR%.2f) | 可交易 IC=%+.4f (IR%.2f, 胜率%.0f%%)",
                 name, full["ic"], full["ir"], trad["ic"], trad["ir"],
                 trad["win"] * 100)
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(dest / "ic_summary.csv", index=False, encoding="utf-8-sig")

    # ---------------- 融合 + 回测 ----------------
    arms = [("主模型 only", main.reindex(oos_days)),
            ("news only", news_pred)]
    for w in (0.5, 0.8, 0.9):
        arms.append((f"主模型⊕news w={w:.1f}",
                     fuse_rank(main.reindex(oos_days), news_pred, w, news_cov)))

    costs = default_costs()
    b_eqw = bench_eqw.reindex(oos_days).fillna(0.0)
    hold_sets = [("TopFrac0.10", lambda: TopFracLongOnly(frac=0.10, weight_mode="equal")),
                 ("TopK45", lambda: TopKLongOnly(k=45))]
    rows = []
    for hold_tag, factory in hold_sets:
        for tag, sig in arms:
            sig = sig.reindex(index=oos_days, columns=close.columns)
            bt = VectorBacktest(strategy=factory(), rebalance_freq="M",
                                initial_capital=1_000_000.0, costs=costs)
            res = bt.run(sig, fwd_engine, executable_mask=mask_oos, horizon=1,
                         rebalance_days=None)
            m = res_metrics(res.daily_returns,
                            bench_idx.reindex(res.daily_returns.index).fillna(0.0),
                            res.turnover_series)
            rows.append(dict(hold=hold_tag, arm=tag, annual=m["annual"],
                             excess_eqw=m["annual"] - _ann(b_eqw),
                             sharpe=m["sharpe"], max_dd=m["max_dd"],
                             turnover=m["turnover"]))
            log.info("[%s] %-24s 年化=%+6.2f%% Sharpe=%.3f 回撤=%5.1f%% 换手=%5.1f%%",
                     hold_tag, tag, m["annual"] * 100, m["sharpe"],
                     m["max_dd"] * 100, m["turnover"] * 100)
    pd.DataFrame(rows).to_csv(dest / "fusion_results.csv", index=False,
                              encoding="utf-8-sig")
    log.info("完成 %.0fs -> %s", time.time() - t0, dest)


if __name__ == "__main__":
    main()
