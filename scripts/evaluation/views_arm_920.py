"""批次 12④：观点注入三臂对照（BL 优化层 vs 预测层 vs 不注入）。

TODO_ARCHIVE 册Ⅲ 批次 12④：同一观点因子（默认 ``bp`` 价值观点，须披露
单因子 IC——观点可能是错的）下三臂对照，``model/views.py`` /
``optimize.solver.bl_views_from_factor`` 已入库。基座 = 920 定版信号 defv
（h1020 秩平均 ⊕ 慢信号 s0.5）。

臂设计（production 口径：T+1 open、buffer(15/40) 月频、固定 10bp）：
  - ``defv_base``：不注入锚（= buffer_arm_920 defv，逐位自检）；
  - ``bl_sX_nY``：BL 优化层注入——top10% 宇宙内 ``solve_portfolio(method="bl")``,
    观点 = ``bl_views_from_factor(bp 截面, n_top, view_scale)``，滚动 LW Σ；
  - 预测层注入（h1 槽位配对对照，protocol 同 retrain_freq_ic：y{sel_year}__h1
    清单 + bp 追加为公共特征，window 500 / refit 60 / rank 标签）：
    ``h1_k1``（清单+bp，无注入对照）→ ``h1_dup_k{K}``（特征复制）→
    ``h1_two_stage``（两段模型）；组合层 = defv 的 h1 槽位替换为各臂 pred。

用法:
    python -m scripts.evaluation.views_arm_920 \
        --sig reports/frontier_audit_920_reval/defv_sig__h1.parquet \
        --out reports/views_arm_920
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402
from strategy.base import Strategy  # noqa: E402

log = setup_logging("views_arm_920")

PRED_DIR = ROOT / "reports" / "alla_rolling_ortho920fundind"
SEL_DIR = PRED_DIR / "selection"


class BLTopFracLongOnly(Strategy):
    """top-frac 宇宙内 Black-Litterman 观点注入加权（滚动 LW Σ，月频）。"""

    def __init__(self, close: pd.DataFrame, view_panel: pd.DataFrame,
                 frac: float, window: int, n_top: int, view_scale: float,
                 max_weight: float, min_periods: int = 60):
        self.close = close
        self.view_panel = view_panel
        self.frac = float(frac)
        self.window = int(window)
        self.n_top = int(n_top)
        self.view_scale = float(view_scale)
        self.max_weight = float(max_weight)
        self.min_periods = int(min_periods)
        self.name = f"bl_s{view_scale:.4f}_n{n_top}"
        self.n_solve = 0
        self.n_fallback = 0

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        from optimize.solver import bl_views_from_factor, estimate_covariance, solve_portfolio

        vals = factor_values.dropna()
        if len(vals) == 0:
            return pd.Series(dtype=float)
        date = factor_values.name
        n_target = max(int(round(self.frac * len(vals))), 1)
        univ = vals.nlargest(n_target).index

        hist = self.close.reindex(columns=univ)
        hist = hist.loc[hist.index < date].tail(self.window + 1)
        rets = hist.pct_change(fill_method=None).iloc[1:]
        # 停牌洞处理同 cvar_arm_920：列覆盖 ≥80% + 残余洞保守零填充
        cov_ok = rets.notna().mean(axis=0) >= 0.8
        rets = rets.loc[:, cov_ok].fillna(0.0)
        usable = rets
        view_row = self.view_panel.reindex(columns=usable.columns) \
            .loc[:date].iloc[-1] if len(self.view_panel.index) else pd.Series(dtype=float)
        if (len(usable) < self.min_periods or usable.shape[1] < 2
                or view_row.notna().sum() < 2 * self.n_top):
            self.n_fallback += 1
            w = pd.Series(1.0 / len(univ), index=univ)
            return w.reindex(factor_values.index).fillna(0.0)
        try:
            Sigma = estimate_covariance(usable)
            views = bl_views_from_factor(view_row, n_top=self.n_top,
                                         view_scale=self.view_scale)
            w = solve_portfolio(alpha=vals.rank(pct=True).reindex(usable.columns),
                                Sigma=Sigma, method="bl", views=views,
                                max_weight=self.max_weight)
            if w.abs().sum() <= 1e-8:
                raise RuntimeError("退化零解")
            self.n_solve += 1
        except Exception as exc:
            self.n_fallback += 1
            log.debug("bl solve 降级 %s: %s", date, exc)
            w = pd.Series(1.0 / len(univ), index=univ)
        return w.reindex(factor_values.index).fillna(0.0)


def rolling_pred_cls(feats: dict[str, pd.DataFrame], labels: pd.DataFrame,
                     oos_days: pd.DatetimeIndex, all_days: pd.DatetimeIndex,
                     window: int, embargo: int, min_train: int, refit: int,
                     params: dict, model_factory) -> pd.DataFrame:
    """retrain_freq_ic.rolling_pred 的模型类参数化版（口径逐行对齐）。"""
    out = pd.DataFrame(np.nan, index=oos_days, columns=labels.columns)
    for i in range(0, len(oos_days), refit):
        chunk = oos_days[i:i + refit]
        cut = all_days[all_days < chunk[0]]
        if embargo and len(cut) > embargo:
            cut = cut[:-embargo]
        tr = cut[-window:]
        if len(tr) < min_train:
            continue
        model = model_factory()
        model.fit({k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
        pred = model.predict({k: v.loc[chunk] for k, v in feats.items()})
        out.loc[chunk[0]:chunk[-1]] = pred.reindex(index=chunk)
        log.info("  折 %s（%d 日）完成", chunk[0].date(), len(chunk))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sig", default="reports/frontier_audit_920_reval/"
                                     "defv_sig__h1.parquet")
    ap.add_argument("--out", default="reports/views_arm_920")
    ap.add_argument("--view-factor", default="bp")
    ap.add_argument("--sel-year", type=int, default=2023)
    ap.add_argument("--dup-k", type=int, default=2)
    ap.add_argument("--two-stage-frac", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--refit", type=int, default=60)
    ap.add_argument("--entry", type=float, default=0.15)
    ap.add_argument("--exit", dest="exit_frac", type=float, default=0.40)
    ap.add_argument("--frac-qp", type=float, default=0.10)
    ap.add_argument("--bl-scales", default="0.001,0.003")
    ap.add_argument("--bl-n-top", type=int, default=20)
    ap.add_argument("--max-weight", type=float, default=0.05)
    ap.add_argument("--skip-pred", action="store_true",
                    help="跳过预测层注入（只跑 BL 臂 + 锚）")
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest, build_execution_split
    from model.labels import build_labels
    from model.params import DEFAULT_MODEL_PARAMS
    from model.predictor import LGBMPredictor
    from model.views import TwoStagePredictor, inject_by_feature_duplication
    from scripts.common import alla_common as AC
    from scripts.common.alla_common import FeatureStore, ds_root, existence_mask
    from scripts.evaluation.fundamental_blend import build_slow_panel
    from scripts.evaluation.horizon_mix import load
    from scripts.evaluation.news_alpha_probe import ic_stats, masked_ic
    from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    params = dict(DEFAULT_MODEL_PARAMS["gbdt"])

    sig = pd.read_parquet(args.sig)
    base = AC.load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    oos_days = sig.index
    fwd1 = close.pct_change(1, fill_method=None).shift(-1)

    # ---- 观点因子单因子 IC 披露（观点可能是错的）----
    store = FeatureStore(ds_root() / "panels_neu_fundind")
    view_panel = store.get(args.view_factor)
    vp = view_panel.reindex(index=oos_days, columns=close.columns)
    view_ic = masked_ic(vp, fwd1, mask)
    view_st = ic_stats(view_ic)
    from stats.significance import mean_inference  # noqa: E402
    view_row = {"view_factor": args.view_factor,
                "raw_ic": view_st["ic"], "raw_ir": view_st["ir"],
                "raw_win": view_st["win"],
                "t": mean_inference(view_ic, robust=False)["t_stat"]}
    log.info("观点因子 %s 单因子可交易 IC=%.4f IR=%.2f t=%.2f（披露：观点可能是错的）",
             args.view_factor, view_st["ic"], view_st["ir"], view_row["t"])

    # ---- production 回测 harness（buffer_arm_920 同款）----
    costs = default_costs()
    fwd = close.pct_change(fill_method=None).reindex(close.index).loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)
    bench_eqw = base["bench_eqw"].reindex(oos_days).fillna(0.0)
    fp = PRED_DIR / "_base" / "open_adj.parquet"
    fill = pd.read_parquet(fp)
    s_ = pd.Series(oos_days, index=oos_days)
    firsts = s_.groupby(s_.index.to_period("M")).first()
    pos_ = {d: i for i, d in enumerate(oos_days)}
    rb_signal = {t for t in firsts if pos_[t] + 1 < len(oos_days)}
    rb_exec = {oos_days[pos_[t] + 1] for t in rb_signal}
    exec_split = build_execution_split(fill, close, rb_signal)

    def run_arm(name, signal, strat=None):
        sig_in = signal.reindex(index=oos_days, columns=close.columns)
        if strat is None:
            strat = BufferedTopFracLongOnly(args.entry, args.exit_frac)
        bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        res = bt.run(sig_in.shift(1).where(mask_oos), fwd, horizon=1,
                     rebalance_days=rb_exec, execution_split=exec_split)
        m = AC.res_metrics(res.daily_returns, bench_idx, res.turnover_series)
        row = {"variant": name, "annual": m["annual"], "excess_idx": m["excess"],
               "excess_eqw": m["annual"] - AC.ann(bench_eqw),
               "sharpe": m["sharpe"], "ir": m["ir"], "max_dd": m["max_dd"],
               "turnover": m["turnover"],
               "n_days": len(res.daily_returns.dropna())}
        log.info("%s | 年化=%.2f%% Sharpe=%.3f 换手=%.1f%%", name,
                 m["annual"] * 100, m["sharpe"], m["turnover"] * 100)
        return row

    rows = [run_arm("defv_base_不注入锚", sig)]

    # ---- BL 优化层注入臂（vs 同宇宙等权对照）----
    rows.append(run_arm("top10_ew_对照", sig,
                        strat=TopFracLongOnly(frac=args.frac_qp,
                                              weight_mode="equal")))
    for vs in [float(x) for x in args.bl_scales.split(",")]:
        strat = BLTopFracLongOnly(close, view_panel, args.frac_qp, 120,
                                  args.bl_n_top, vs, args.max_weight)
        rows.append(run_arm(f"bl_s{vs:.4f}_n{args.bl_n_top}", sig, strat=strat))
        log.info("bl_s%.4f: solve=%d fallback=%d", vs, strat.n_solve,
                 strat.n_fallback)

    # ---- 预测层注入（h1 槽位配对对照）----
    ic_rows = []
    if not args.skip_pred:
        sel_path = SEL_DIR / f"y{args.sel_year}__h1.json"
        names = json.loads(sel_path.read_text(encoding="utf-8"))
        names = list(dict.fromkeys(names + [args.view_factor]))
        i0 = int(close.index.get_indexer([oos_days[0]])[0])
        keep = close.index[max(0, i0 - args.window - 1 - 20):]
        feats = store.get_many(names)
        feats = {k: v.reindex(index=keep, columns=close.columns)
                 for k, v in feats.items()}
        labels, embargo = build_labels(close, horizon=1, mode="rank",
                                       tradable_mask=mask)
        valid = existence_mask(feats, close, oos_days)
        all_days = close.index

        arms = {
            "h1_k1_对照": (lambda: LGBMPredictor(**params), feats),
            f"h1_dup_k{args.dup_k}": (
                lambda: LGBMPredictor(**params),
                inject_by_feature_duplication(feats, [args.view_factor],
                                              k=args.dup_k)),
            "h1_two_stage": (
                lambda: TwoStagePredictor(
                    LGBMPredictor, args.view_factor,
                    top_frac=args.two_stage_frac, **params),
                feats),
        }
        h1_orig = load(PRED_DIR, "gbdt__h1")
        h5, h10, h20 = (load(PRED_DIR, f"gbdt__h{h_}")
                        for h_ in (5, 10, 20))
        slow, _, _ = build_slow_panel(base, sig.index, 24, 2, 0.5, 12,
                                      log=lambda *a, **k: None)

        def defv_with_h1(h1_arm: pd.DataFrame) -> pd.DataFrame:
            fast = AC.rank_average([h1_arm, h5, h10, h20])
            return (0.5 * fast.rank(axis=1, pct=True)
                    + 0.5 * slow.rank(axis=1, pct=True))

        pred_arms = {"h1_orig_生产参考": h1_orig}
        pred_arms.update({k: None for k in arms})
        for name, (factory, feats_arm) in arms.items():
            t1 = time.time()
            raw = rolling_pred_cls(feats_arm, labels, oos_days, all_days,
                                   args.window, embargo, 250, args.refit,
                                   params, factory)
            pred = raw.where(valid.reindex(index=oos_days))
            pred.astype("float32").to_parquet(dest / f"pred_{name}.parquet")
            pred_arms[name] = pred
            ic = masked_ic(pred, fwd1, mask)
            st = ic_stats(ic)
            ic_rows.append({"arm": name, "ic": st["ic"], "ir": st["ir"],
                            "win": st["win"],
                            "t": mean_inference(ic, robust=False)["t_stat"],
                            "sec": round(time.time() - t1, 0)})
            log.info("%s OOS IC=%.4f IR=%.2f（%.0fs）", name, st["ic"],
                     st["ir"], time.time() - t1)

        for name, pred in pred_arms.items():
            if pred is None:
                continue
            rows.append(run_arm(f"defv_{name}", defv_with_h1(pred)))

    pd.DataFrame(rows).to_csv(dest / "views_arm.csv", index=False,
                              encoding="utf-8-sig")
    ic_rows.insert(0, {"arm": "view_" + args.view_factor, **view_row})
    pd.DataFrame(ic_rows).to_csv(dest / "pred_ic.csv", index=False,
                                 encoding="utf-8-sig")
    (dest / "meta.json").write_text(json.dumps(
        {**vars(args), "n_days": len(oos_days),
         "oos": [str(oos_days[0].date()), str(oos_days[-1].date())]},
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"完成 {time.time() - t0:.0f}s → {dest}")


if __name__ == "__main__":
    main()
