"""B1 探针：现有 α 能不能在**中证500 成分约束下**做出可投的指数增强？

这条线在问什么
--------------
A 线（全A 选股）已证 alpha 存在但 ~56% 收益是 β；B1 的问题是「把 β 换成基准」
之后 alpha 还剩多少。本探针只做**纯多头指数增强**（个人场景无股指期货权限，
beta hedge 分支不建）：持仓 = 500 只成分，权重 = 指数权重 × (1 + λ·z(α))，
看 **3–5% TE 档位下净超额还剩多少**。

判据（先验固定，不按结果改）
----------------------------
1. 主看 **净超额年化**（扣 10bp 单边成本）与 **excess_t**；TE 3% 档 t ≥ 2 才算有 α；
2. 若净超额 ≈ 0 且只是「波动下降、收益同步下降」（IR 不升） ⇒ 该线按风险偏好归档，
   不算 alpha（与 `TODO.md` §六 B1 判据一致）；
3. placebo（同 λ 下打乱截面 α）超额应 ≈ 0，用来排除「构造本身带来溢价」。

口径与已知限制
--------------
- α 用**已存档**的 OOS 面板 `reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet`
  （2018-01-02→，逐日截面秩平均）。**不是**定版锚 `ortho_fundind` 的 pred ——
  锚那一次运行没落 pred 面板；差异只在 fundind 正交化，探针结论对构造成立性有效，
  对「锚口径下超额多少」只能当上界/下界的近似，README 明确标注。
- 权重：`get_index_weight`（000905）日频，`TRADE_DATE` 无公告时间戳 ⇒ 按 **T+1**
  使用（调仓日 t 用 t 及以前已知的权重，实际落地晚一天不会改变量级）。
- 月频调仓（与生产一致）；α 取调仓日前 N 日（默认 20）均值，降低 h1 噪声与换手。
- 成本：|Δw| 单边 turnover × 2 × cost_bp（买+卖），默认 10bp；初始建仓不计成本。
- 容量/涨跌停：只做「建仓日可交易」过滤（tradable_mask），不做日内冲击模型 ——
  500 只等权附近的篮子容量远大于现有 Top-30，不是本探针的约束项。

用法
----
    python -m scripts.evaluation.index_enhance_probe
    python -m scripts.evaluation.index_enhance_probe --lam 0.05,0.1,0.2,0.3,0.5 --window 1,20

产物 `reports/index_enhance_probe/`
----------------------------------
    lambda_curve.csv     每个 (window, λ) 一行：净超额 / TE / IR / t / 换手 / 成本
    ic_by_rebalance.csv  每次调仓的截面 rank IC（α 在成分内是否有效）
    yearly.csv           逐年净超额（多窗口一致性）
    equity.csv           各组合净值（date × 组合）
    placebo.csv          打乱 α 的安慰剂超额分布
    run_config.json      本次运行参数（pred 路径 / 指数 / 区间 / 成本）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "index_enhance_probe"
PRED_DEFAULT = ROOT / "reports" / "alla_rolling_ortho" / "pred" / "ens_h1h5__h1.parquet"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_index_weights(index_code: str) -> pd.DataFrame:
    """指数成分日权重（date × code，百分比）——读 SDK 缓存 h5。

    缓存由 `scripts/evaluation/sdk_table_audit.py` 回源落盘；缺失时给出明确指引，
    不在探针里隐式联网（探针要可离线复跑）。
    """
    from config import Config

    safe = str(index_code).upper()
    p = (Path(str(Config.paths()["sdk_local_path"])) / "infodata"
         / "index_weight" / f"{safe}.h5")
    if not p.exists():
        raise FileNotFoundError(
            f"指数权重缓存缺失：{p}\n"
            "先跑 `python -m scripts.evaluation.sdk_table_audit` 回源落盘。")
    df = pd.read_hdf(p, key=f"/{safe}")
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"].astype(str), format="%Y%m%d")
    w = df.pivot_table(index="TRADE_DATE", columns="CON_CODE", values="WEIGHT",
                       aggfunc="last")
    return w.sort_index()


def load_pred(path: Path) -> pd.DataFrame:
    p = pd.read_parquet(path)
    p.index = pd.DatetimeIndex(p.index)
    return p.sort_index()


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """复权收盘 + 可交易掩码（与主实验同一 _base，口径一致）。"""
    from scripts.common.alla_common import load_base

    b = load_base()
    return b["close"], b["mask"]


def month_end_rebalances(dates: pd.DatetimeIndex, weights: pd.DataFrame) -> list[pd.Timestamp]:
    """调仓日 = 权重面板里每个月的最后一个交易日（与数据可用性严格一致）。"""
    common = weights.index.intersection(dates)
    s = pd.Series(common, index=common)
    return [pd.Timestamp(x) for x in s.groupby([common.year, common.month]).max()]


# ---------------------------------------------------------------------------
# 组合构造
# ---------------------------------------------------------------------------
def zscore_row(s: pd.Series, clip: float = 3.0) -> pd.Series:
    s = s.dropna()
    if len(s) < 20 or s.std(ddof=0) == 0:
        return pd.Series(dtype=float)
    z = (s - s.mean()) / s.std(ddof=0)
    return z.clip(-clip, clip)


def target_weights(w_idx: pd.Series, z: pd.Series, lam: float,
                   max_name: float = 0.10) -> pd.Series:
    """w = 指数权重 × (1 + λ·z)，纯多头、归一、单票上限。"""
    w = w_idx.reindex(z.index).astype(float) * (1.0 + lam * z.reindex(z.index))
    w = w.clip(lower=0.0)
    if w.sum() <= 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    # 单票上限（超过则截断并再归一，直到收敛；指数本身最大权重也在此内）
    for _ in range(5):
        over = w > max_name
        if not over.any():
            break
        w = w.clip(upper=max_name) / w.clip(upper=max_name).sum()
    return w


# ---------------------------------------------------------------------------
# 模拟
# ---------------------------------------------------------------------------
def simulate(targets: dict[pd.Timestamp, pd.Series], px: pd.DataFrame,
             cost_bp: float) -> tuple[pd.Series, pd.Series, dict[pd.Timestamp, float]]:
    """按目标权重逐段持有，返回 (净值, 日收益, 每次调仓 turnover)。

    价格用 **ffill 后**的复权收盘：停牌日收益记为 0（不虚构成交）。
    首个建仓日不计成本（8 年区间里影响 <1bp）。
    """
    px_ff = px.ffill()
    dates = px_ff.index
    reb = sorted(targets)
    nav = pd.Series(np.nan, index=dates, dtype=float)
    turnover: dict[pd.Timestamp, float] = {}
    value = 1.0
    prev_drift: pd.Series | None = None

    for i, t in enumerate(reb):
        w = targets[t]
        if not len(w):
            continue
        p0 = px_ff.loc[t]
        # 与上一期期末漂移权重比较（真实换手，而非两个目标之间的差）
        if prev_drift is not None:
            union = w.index.union(prev_drift.index)
            turn = float(0.5 * (w.reindex(union, fill_value=0.0)
                                - prev_drift.reindex(union, fill_value=0.0)).abs().sum())
            value *= (1.0 - 2.0 * turn * cost_bp / 1e4)
        else:
            turn = float("nan")   # 初始建仓，不收费
        turnover[t] = turn
        nav.loc[t] = value

        end = reb[i + 1] if i + 1 < len(reb) else dates[-1]
        window = dates[(dates > t) & (dates <= end)]
        for d in window:
            ratio = float((w * (px_ff.loc[d] / p0)).sum(skipna=True))
            nav.loc[d] = value * ratio
        if len(window):
            d_end = window[-1]
            vals = w * (px_ff.loc[d_end] / p0)
            tot = float(vals.sum(skipna=True))
            prev_drift = vals / tot if tot > 0 else w
            value = float(nav.loc[d_end])
    return nav.dropna(), nav.pct_change(fill_method=None), turnover


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def stats(port: pd.Series, bench: pd.Series, turnover: dict,
          n_months: int, cost_bp: float) -> dict:
    from backtest.metrics import calc_all_metrics
    from research.benchmarks import compare_to_benchmark

    p, b = port.align(bench, join="inner")
    m = calc_all_metrics(p.dropna(), b.reindex(p.index).fillna(0.0))
    cmp = compare_to_benchmark(p.dropna(), b.reindex(p.index).fillna(0.0))
    turns = pd.Series([v for v in turnover.values() if np.isfinite(v)])
    return {
        "annual": m["annual_return"],
        "bench_annual": m.get("benchmark_annual_return", np.nan),
        "excess": m.get("excess_return", np.nan),
        "te": cmp["tracking_error"],
        "ir": m.get("information_ratio", np.nan),
        "excess_t": m.get("excess_t_stat", np.nan),
        "sharpe": m["sharpe"],
        "max_dd": m["max_drawdown"],
        "beta": cmp["beta"],
        "corr": cmp["correlation"],
        "turnover_monthly": float(turns.mean()) if len(turns) else np.nan,
        "cost_drag_annual": float(2.0 * turns.mean() * 12.0 * cost_bp / 1e4)
                            if len(turns) else np.nan,
        "n_months": n_months,
    }


def eligible_components(reb: list[pd.Timestamp], weights: pd.DataFrame,
                        pred: pd.DataFrame, px: pd.DataFrame,
                        mask: pd.DataFrame, window: int
                        ) -> dict[pd.Timestamp, tuple[pd.Series, pd.Series]]:
    """每个调仓日的（指数权重, α 均值）——只留「有权重 + 有价 + 可交易」。

    同一 window 内 λ 与安慰剂共用这份结果，避免重复切 α 面板。
    """
    out: dict[pd.Timestamp, tuple[pd.Series, pd.Series]] = {}
    for t in reb:
        w_idx = weights.loc[t].dropna()
        if len(w_idx) < 100:
            continue
        hist = pred.loc[:t].tail(window)
        if not len(hist):
            continue
        score = hist.mean(axis=0)   # 逐股对窗口内 α 取均值（平滑 h1 噪声）
        ok = w_idx.index.intersection(px.columns)
        ok = ok[px.loc[t].reindex(ok).notna().values
                & mask.loc[t].reindex(ok).fillna(False).astype(bool).values]
        if len(ok) < 100:
            continue
        out[t] = (w_idx.reindex(ok), score.reindex(ok))
    return out


def build_targets(ev: dict[pd.Timestamp, tuple[pd.Series, pd.Series]],
                  lam: float, shuffle_seed: int | None = None
                  ) -> dict[pd.Timestamp, pd.Series]:
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    out: dict[pd.Timestamp, pd.Series] = {}
    for t, (w_idx, score) in ev.items():
        z = zscore_row(score)
        if not len(z):
            continue
        # 无 α 的成分（新股/特征缺失）按中性 0 处理：保留指数权重，不被动偏离
        z = z.reindex(score.index).fillna(0.0)
        if rng is not None:
            z = pd.Series(rng.permutation(z.values), index=z.index)
        w = target_weights(w_idx, z, lam)
        if len(w):
            out[t] = w
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="B1 纯多头指数增强探针（CSI500）")
    ap.add_argument("--index", default="000905.SH", help="基准指数（默认中证500）")
    ap.add_argument("--pred", default=str(PRED_DEFAULT), help="OOS α 面板 parquet")
    ap.add_argument("--begin", default="2018-01-02")
    ap.add_argument("--end", default=None, help="默认取权重与价格都覆盖的最后一天")
    ap.add_argument("--window", default="1,20", help="α 平滑窗口（交易日）")
    ap.add_argument("--lam", default="0.05,0.1,0.2,0.3,0.5", help="主动倾斜强度网格")
    ap.add_argument("--cost-bp", type=float, default=10.0, help="单边成本（bp）")
    ap.add_argument("--max-name", type=float, default=0.10, help="单票权重上限")
    ap.add_argument("--placebo", type=int, default=20, help="安慰剂重复次数（0=关）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    windows = [int(x) for x in str(args.window).split(",") if x.strip()]
    lams = [float(x) for x in str(args.lam).split(",") if x.strip()]

    print(f"[load] 指数权重 {args.index} ...")
    weights = load_index_weights(args.index)
    print(f"[load] α 面板 {args.pred} ...")
    pred = load_pred(Path(args.pred))
    print("[load] 复权价 / 可交易掩码 ...")
    px, mask = load_prices()

    end = pd.Timestamp(args.end) if args.end else min(px.index.max(), weights.index.max())
    reb = month_end_rebalances(px.index, weights)
    reb = [t for t in reb if pd.Timestamp(args.begin) <= t <= end]
    print(f"[plan] 调仓 {len(reb)} 次：{reb[0].date()} → {reb[-1].date()}")

    # 基准：指数日收益（本地缓存；缺失时回源一次）
    from data.cache_helpers import load_index_returns
    bench = load_index_returns(args.index, int(reb[0].strftime("%Y%m%d")),
                               int(end.strftime("%Y%m%d")), real=True)
    if bench is None:
        raise RuntimeError("基准指数日线不可用")
    bench = bench[~bench.index.duplicated()].sort_index()

    # 截面 rank IC（每次调仓：α 与「本段持有期」成分收益）
    px_ff = px.ffill()
    ic_rows = []
    for wk in windows:
        for i, t in enumerate(reb):
            w_idx = weights.loc[t].dropna()
            hist = pred.loc[:t].tail(wk)
            if not len(hist) or len(w_idx) < 100:
                continue
            score = hist.mean(axis=0).reindex(w_idx.index).dropna()
            if len(score) < 100 or score.std(ddof=0) == 0:
                continue
            nxt = reb[i + 1] if i + 1 < len(reb) else min(end, px_ff.index.max())
            p0 = px_ff.loc[t].reindex(score.index)
            p1 = px_ff.loc[nxt].reindex(score.index)
            fwd = (p1 / p0 - 1.0).dropna()
            common = score.index.intersection(fwd.index)
            if len(common) < 100:
                continue
            ic = score.reindex(common).corr(fwd.reindex(common), method="spearman")
            ic_rows.append({"window": wk, "date": t, "ic": ic, "n": len(common)})
    ic_df = pd.DataFrame(ic_rows)
    if len(ic_df):
        ic_df.to_csv(out / "ic_by_rebalance.csv", index=False, encoding="utf-8-sig")
        for wk, sub in ic_df.groupby("window"):
            ic_s = sub["ic"].dropna()
            tstat = ic_s.mean() / ic_s.std(ddof=1) * np.sqrt(len(ic_s)) if len(ic_s) > 2 else np.nan
            print(f"[ic] window={wk}: mean={ic_s.mean():.4f} "
                  f"icir={ic_s.mean() / ic_s.std(ddof=1) if ic_s.std(ddof=1) else np.nan:.3f} "
                  f"t={tstat:.2f} n={len(ic_s)}")

    rows, yearly_rows, eq = [], [], {}
    from backtest.metrics import annual_return
    ev_cache: dict[int, dict] = {}
    for wk in windows:
        ev_cache[wk] = eligible_components(reb, weights, pred, px, mask, wk)
        for lam in lams:
            targets = build_targets(ev_cache[wk], lam)
            nav, ret, turns = simulate(targets, px, args.cost_bp)
            if len(ret) < 100:
                continue
            tag = f"w{wk}_lam{lam:g}"
            eq[tag] = nav
            m = stats(ret, bench, turns, len(targets), args.cost_bp)
            m.update({"window": wk, "lam": lam, "tag": tag})
            rows.append(m)
            print(f"[run] {tag}: net_ann={m['annual']:.2%} excess={m['excess']:+.2%} "
                  f"TE={m['te']:.2%} IR={m['ir']:.2f} t={m['excess_t']:.2f} "
                  f"turn/m={m['turnover_monthly']:.1%}")
            # 逐年
            p, b = ret.align(bench, join="inner")
            for year, sub in p.groupby(p.index.year):
                subb = b.reindex(sub.index).fillna(0.0)
                if len(sub.dropna()) < 20:
                    continue
                pr = annual_return(sub.dropna())
                br = annual_return(subb)
                ex = (sub - subb).dropna()
                yearly_rows.append({
                    "tag": tag, "window": wk, "lam": lam, "year": int(year),
                    "port": pr, "bench": br, "excess": pr - br,
                    "te": float(ex.std() * np.sqrt(252)),
                })

    curve = pd.DataFrame(rows)
    curve.to_csv(out / "lambda_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_rows).to_csv(out / "yearly.csv", index=False,
                                     encoding="utf-8-sig")
    pd.DataFrame(eq).to_csv(out / "equity.csv", encoding="utf-8-sig")

    # 安慰剂：取第一个窗口 × 中间 λ，打乱截面 α（同构造、同成本）
    if args.placebo > 0 and len(curve):
        wk = windows[0]
        lam = lams[len(lams) // 2]
        ev = ev_cache.get(wk) or eligible_components(reb, weights, pred, px, mask, wk)
        nav, ret, turns = simulate(build_targets(ev, lam), px, args.cost_bp)
        actual = stats(ret, bench, turns, 0, args.cost_bp)["excess"]
        pl = []
        for seed in range(args.placebo):
            _, ret, turns = simulate(build_targets(ev, lam, shuffle_seed=seed + 1),
                                     px, args.cost_bp)
            pl.append(stats(ret, bench, turns, 0, args.cost_bp)["excess"])
        pl_df = pd.DataFrame({"window": wk, "lam": lam, "seed": range(1, args.placebo + 1),
                              "excess": pl})
        pl_df.to_csv(out / "placebo.csv", index=False, encoding="utf-8-sig")
        arr = np.array(pl)
        print(f"[placebo] window={wk} lam={lam}: mean={arr.mean():+.2%} "
              f"sd={arr.std(ddof=1):.2%} actual={actual:+.2%} "
              f"P(placebo>=actual)={float((arr >= actual).mean()):.2f}")

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"rebalances": [str(t.date()) for t in reb[:1]] + [str(t.date()) for t in reb[-1:]],
                "n_rebalances": len(reb), "index_daily_source": "cache/sdk"})
    (out / "run_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
