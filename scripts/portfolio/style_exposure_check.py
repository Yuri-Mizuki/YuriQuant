"""定版持仓的 Barra 风格暴露体检（2026-11-25 建）。

做法：跑一档定版拿到最后一次调仓的目标权重 → 用
``research.barra.standardize_exposures`` 把当日 13 个风格原始暴露标准化 →
组合暴露 = Σ w·z（单位：个股截面 1 个标准差）→ 与「同规模随机组合」零分布对照。

随机零分布：从当日有效暴露股票中抽 N 只（N = 实际持仓数）、把实际权重向量随机
指派给它们，重复 B 次 ⇒ 判「这么宽的持仓是不是真押了某个风格」。

⚠️ 这是**体检**不是生产件：没有任何生产链路读 `panels_barra`，本脚本只回答
「定版持仓到底押了哪些风格、押了多少」。结论见 `reports/style_exposure/README.md`。

用法::

    python -m scripts.portfolio.style_exposure_check --tier yqcore-breadth
    python -m scripts.portfolio.style_exposure_check --tier yqcore-focus-k8
    python -m scripts.portfolio.style_exposure_check --summary

一次只跑一档、分进程：本机提交内存常年只剩 ~3G，两档上下文同时驻留会 OOM
（`numpy._core._exceptions._ArrayMemoryError`）。`--no-lot` 可再省一块价格面板。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from research.barra import STYLE_VARS, standardize_exposures  # noqa: E402
from scripts.evaluation import caliber as cal  # noqa: E402
from strategy.registry import get_tier  # noqa: E402

END = 20260907
CAPS = {"yqcore-breadth": (2.0e6, 100), "yqcore-focus-k15": (5.0e5, 100),
        "yqcore-focus-k8": (1.0e5, 100)}
B = 3000
OUT = ROOT / "reports" / "style_exposure"
RNG = np.random.default_rng(20261125)
#: f50 数据态复合信号（集中档；882 ortho pred 本机已不存在）
F50_COMPOSITE = (ROOT / "reports" / "alla_rolling_ab_f50" / "pred"
                 / "gbdt_mid13__h1.parquet")


def _lean_amount() -> pd.DataFrame:
    """只读 ``amount`` 一列（``caliber._load_daily_amount`` 会读全表，本机提交内存吃紧）。"""
    p = Path(str(Config.cache()["root"])) / "daily_all_a.parquet"
    d = pd.read_parquet(p, columns=["amount"])
    out = d["amount"].unstack("code").astype(float)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


cal._load_daily_amount = _lean_amount  # noqa: SLF001  （只减内存，不改口径）


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns).ffill().shift(1)


def panel_for(name: str, ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / f"{name}.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def lean_context(signal: str, end: int = END,
                 pred_path: Path | None = None) -> cal.CaliberContext:
    """只装 ``run_once(open)`` 真正用到的面板（省掉 ADV/vol60/ret_hist，本机省 ~1G）。

    生产成本模型无 impact 项 ⇒ ``amt`` / ``vol60`` 不会被引擎读取；``ret_hist``
    只有 QP 类组合构造用。pred / close / mask / fill / fwd 五块与
    :func:`caliber.load_context` 同源同截断。

    ``pred_path`` 显式指定信号面板（f50 数据态用：集中档的 882 ortho pred
    本机已不存在，改读 ``ab_f50/pred/gbdt_mid13__h1.parquet``）。
    """
    close_full = pd.read_parquet(cal.BASE_DIR / "close_adj.parquet")
    mask_full = pd.read_parquet(cal.BASE_DIR / "tradable_mask.parquet")
    spec = cal.SIGNALS[signal]
    pred_full = (pd.read_parquet(pred_path) if pred_path is not None
                 else cal._composite_panel(signal) if spec.get("components")
                 else pd.read_parquet(cal.ROOT / spec["path"]))
    fill_full = pd.read_parquet(cal.BASE_DIR / "open_adj.parquet")

    idx = pred_full.index
    keep = idx[(idx >= pd.Timestamp(cal.WINDOWS["prod"]))
               & (idx <= pd.Timestamp(str(end)))]
    cols = pred_full.columns
    pred = pred_full.reindex(index=keep, columns=cols)
    close = close_full.reindex(index=keep, columns=cols)
    mask = mask_full.reindex(index=keep, columns=cols).fillna(True)
    fill = fill_full.reindex(index=keep, columns=cols)
    del close_full, mask_full, pred_full, fill_full
    fwd = (close.pct_change(fill_method=None)
           .replace([np.inf, -np.inf], np.nan).fillna(0.0))
    return cal.CaliberContext(
        window="prod", signal=signal, begin=cal.WINDOWS["prod"], pred=pred,
        close=close, fwd=fwd, mask=mask, mask_same_day=mask, amt=None,
        vol60=None, ret_hist=fwd, fingerprint=cal.fingerprint_panel(pred),
        execution="open", fill=fill)


def last_weights(tier: str, lot_on: bool = True,
                 pred_path: Path | None = None) -> tuple[pd.Series, pd.Timestamp]:
    capital, lot = CAPS[tier]
    lot = lot if lot_on else None
    spec = get_tier(tier)
    ctx = lean_context(spec.runnable_signal, pred_path=pred_path)
    overrides = {}
    if spec.panel is not None:
        overrides[spec.panel] = panel_for(spec.panel, ctx)
    strat = spec.build(**overrides)
    bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=float(capital),
                            costs=cal.cost_profile("production"))
    res = cal.run_once(bt, ctx, "open",
                       price_panel=price_panel(ctx) if lot else None,
                       lot_size=lot)
    wh = res.weights_history
    # 最后一次调仓（执行日）；风格暴露取它前一个交易日的截面（信号日 T，无前视）
    reb = [d for d in wh.index if d in set(cal.next_day_rebalance(ctx.pred.index))]
    e = reb[-1]
    t = ctx.pred.index[ctx.pred.index.get_loc(e) - 1]
    w = wh.loc[e]
    w = w[w > 1e-12]
    return w / w.sum(), t


def load_exposures(date: pd.Timestamp) -> pd.DataFrame:
    cfg = Config.get()["factor_library"]
    barra = Path(str(cfg["root"])) / "all_a_2018_2026" / "panels_barra"
    cap = pd.read_parquet(cal.BASE_DIR / "market_cap.parquet")
    cap_row = cap.loc[date] if date in cap.index else pd.Series(dtype=float)
    cols = {}
    for v in STYLE_VARS:
        raw = pd.read_parquet(barra / f"{v}.parquet")
        row = raw.loc[date] if date in raw.index else raw.iloc[-1]
        cols[v] = standardize_exposures(row, cap_row, v)
    return pd.DataFrame(cols)


def probe(tier: str, lot_on: bool = True,
          pred_path: Path | None = None) -> pd.DataFrame:
    w, date = last_weights(tier, lot_on, pred_path)
    ex_all = load_exposures(date)          # 全市场截面（零分布抽样池）
    ex = ex_all.reindex(index=w.index)     # 只用于陈列持仓
    print(f"=== {tier}  调仓日 {date.date()}  持仓 {len(w)} 只")
    recs = []
    for v in STYLE_VARS:
        z_univ = ex_all[v].dropna()
        hold = z_univ.index.intersection(w.index)
        ww = (w[hold] / w[hold].sum()).to_numpy()
        z = z_univ[hold].to_numpy()
        pe = float(np.dot(ww, z))
        z_univ_arr = z_univ.to_numpy()
        nd = np.empty(B)
        for k in range(B):
            pick = RNG.choice(len(z_univ_arr), size=len(hold), replace=False)
            nd[k] = float(np.dot(ww, z_univ_arr[pick]))
        sd = float(nd.std(ddof=1))
        recs.append({
            "tier": tier, "date": str(date.date()), "n_hold": int(len(hold)),
            "var": v, "exposure": pe, "null_mean": float(nd.mean()), "null_sd": sd,
            "z": (pe - nd.mean()) / sd if sd > 0 else np.nan,
            "p": float((np.abs(nd - nd.mean()) >= abs(pe - nd.mean())).mean()),
            "stock_pct": float((z_univ < pe).mean()),
            "z_univ_sd": float(z_univ.std()),
        })
    # 持仓与市值分位（写盘，供报告引用）
    cap = pd.read_parquet(cal.BASE_DIR / "market_cap.parquet").loc[date].dropna()
    w.to_frame("weight").assign(
        cap_pct=cap.rank(pct=True).reindex(w.index),
        mktcap_yi=cap.reindex(w.index) / 1e8,
    ).to_csv(OUT / f"hold_{tier}.csv")
    print(f"市值分位中位 {cap.rank(pct=True).reindex(w.index).median():.2f}"
          f"（全市场中位 {cap.median()/1e8:.1f} 亿）")

    df = pd.DataFrame(recs)
    print(df[["var", "exposure", "null_mean", "null_sd", "z", "p", "stock_pct"]]
          .round(3).to_string(index=False))

    # 持仓明细（风格贡献最大的前几只）
    head = w.sort_values(ascending=False).head(10)
    cols = ["beta", "lncap", "nl_size", "momentum", "resid_vol", "liquidity", "btop"]
    show = pd.concat([head.rename("weight"), ex.loc[head.index, cols]], axis=1)
    print("\n前 10 大权重：")
    print(show.round(4).to_string())
    return df


def summary() -> int:
    files = sorted(OUT.glob("snap_*.csv"))
    if len(files) < 2:
        print(f"只有 {len(files)} 份快照，先跑两个 --tier")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files])
    tiers = list(df["tier"].unique())
    print(f"{'变量':<12}" + "".join(f"{t:>30}" for t in tiers))
    for v in STYLE_VARS:
        line = f"{v:<12}"
        for t in tiers:
            r = df[(df.tier == t) & (df["var"] == v)].iloc[0]
            line += (f"{r.exposure:>+9.3f} null{r.null_mean:>+7.2f} z{r.z:>+6.2f}"
                     f" p{r.p:>5.2f}")
        print(line)
    df.to_csv(OUT / "exposure_snapshot.csv", index=False)
    print(f"\n合并写入 {OUT / 'exposure_snapshot.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--no-lot", action="store_true")
    ap.add_argument("--f50", action="store_true",
                    help="集中档读 ab_f50 的 gbdt_mid13 复合面板（f50 数据态）")
    a = ap.parse_args()
    if a.summary:
        return summary()
    OUT.mkdir(parents=True, exist_ok=True)
    pred_path = F50_COMPOSITE if (a.f50 and a.tier != "yqcore-breadth") else None
    df = probe(a.tier, not a.no_lot, pred_path)
    df.to_csv(OUT / f"snap_{a.tier}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
