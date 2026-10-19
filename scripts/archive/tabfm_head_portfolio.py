"""批次 8 头部集成·组合层验证（立项①的第二步，头部诊断的直接后续）。

头部诊断（tabfm_head_study）发现：ICL 的 IC 优势在截面中腰部、跨方法头部
Jaccard 仅 0.17-0.19（互补）、三臂秩平均头部超额 15.55pp 超最好单臂——
但那是日频理论值。本脚本在**真实组合口径**（月度调仓、T+1 开盘成交、
可交易掩码、双边成本）下验证：

- H1：三臂（gbdt/tabpfn3/tabicl，同 W 窗）秩平均的组合年化 > 最好单臂？
- H2：两两组合是否已足够（最优对）？
- H3：头部内加权（top-decile 内按信号强度线性加权）vs 头部等权？

全部为 pred 面板的事后算术（无重训）。输出 summary CSV + 逐月收益。
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ns(idx) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx).as_unit("ns")


def _month_bounds(days: pd.DatetimeIndex, test_begin: str) -> list[pd.Timestamp]:
    s = pd.Series(days, index=days)
    tb = pd.Timestamp(test_begin)
    return [pd.Timestamp(d) for d in s.groupby([days.year, days.month]).first()
            if pd.Timestamp(d) >= tb]


def _load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = _ns(df.index)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-dir", default="reports/tabfm_rolling/hs300_2022_2025_tb2025-01")
    ap.add_argument("--arms", default="gbdt_w3m,tabpfn3_w3m,tabicl_w3m",
                    help="参与集成的臂（同 W 窗）")
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--test-begin", default="2025-01-01")
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from config import Config
    from data.cache_helpers import build_panel
    from data.tradability import build_tradable_mask

    pred_dir = Path(args.pred_dir)
    out = pred_dir / "head_portfolio"
    out.mkdir(parents=True, exist_ok=True)

    panel, _ = build_panel(Config.get(), int(Config.discipline()["begin"]),
                           20261231, offline=True)
    openp = panel["open"]
    openp.index = _ns(openp.index)
    close = panel["close"]
    close.index = _ns(close.index)
    days = openp.index
    bf_path = Path(str(Config.cache()["root"])) / "backward_factor.parquet"
    bwd = None
    if bf_path.exists():
        bwd = pd.read_parquet(bf_path)
        bwd.index = _ns(bwd.index)
        bwd = bwd.reindex(index=close.index, columns=close.columns).ffill()
    trad = build_tradable_mask(close, bwd=bwd)
    if trad.columns.duplicated().any():
        trad = trad.loc[:, ~trad.columns.duplicated()]

    c = default_costs()
    round_trip = 2 * c.commission_rate + c.stamp_duty + 2 * c.slippage_bp / 1e4
    bounds = _month_bounds(days, args.test_begin)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    preds = {a: _load_panel(pred_dir / f"pred_{a}.parquet") for a in arms}

    # 候选信号：单臂 + 两两 + 三臂秩平均
    variants: dict[str, pd.DataFrame] = dict(preds)
    for a, b in combinations(arms, 2):
        variants[f"{a}+{b}"] = (preds[a].rank(axis=1, pct=True)
                                + preds[b].rank(axis=1, pct=True)) / 2.0
    variants["TRIPLE_rankavg"] = sum(p.rank(axis=1, pct=True)
                                     for p in preds.values()) / len(preds)
    # H3：头部内按信号强度线性加权（单臂最优 + 三臂）
    for a in arms:
        variants[f"{a}_加权"] = preds[a]
    variants["TRIPLE_加权"] = variants["TRIPLE_rankavg"]

    rows, monthly = [], {}
    for name, sig in variants.items():
        nav, rets = 1.0, []
        for i in range(len(bounds) - 1):
            b, nb = bounds[i], bounds[i + 1]
            if b not in sig.index:
                continue
            try:
                entry = openp.loc[days[days > b][0]]
                exit_d = days[days > nb][0]
            except IndexError:
                continue
            s = sig.loc[b].reindex(entry.index)
            s = s.where(trad.loc[b].reindex(entry.index).fillna(True))
            k = max(1, int(s.dropna().size * args.frac))
            pick = s.dropna().nlargest(k)
            if pick.empty:
                continue
            r = (openp.loc[exit_d].reindex(pick.index)
                 / entry.reindex(pick.index) - 1.0).fillna(0.0)
            if "加权" in name:
                # 头部内按信号强度线性加权（H3）
                w = pick.rank(pct=True)
                w = w / w.sum()
                gross = float((r * w.loc[r.index]).sum())
            else:
                gross = float(r.mean())
            net = gross - round_trip
            nav *= 1.0 + net
            rets.append(net)
            monthly.setdefault(name, {})[str(nb.date())] = round(net, 4)
        years = max(len(rets) / 12.0, 1e-9)
        ann = nav ** (1 / years) - 1.0
        vol = float(np.std(rets, ddof=1)) * np.sqrt(12) if len(rets) > 1 else float("nan")
        eq = pd.Series(rets, dtype=float)
        dd = float((1 + eq).cumprod().div((1 + eq).cumprod().cummax()).min() - 1)
        rows.append({"variant": name, "annual": round(ann * 100, 2),
                     "sharpe": round(ann / vol, 3) if vol and vol > 0 else np.nan,
                     "max_dd": round(dd * 100, 2), "n_months": len(rets)})
        print(f"[{name}] 年化={ann * 100:+.2f}% Sharpe={rows[-1]['sharpe']} "
              f"回撤={dd * 100:.1f}%", flush=True)

    table = pd.DataFrame(rows).sort_values("annual", ascending=False)
    table.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(monthly).T.to_csv(out / "monthly_returns.csv",
                                   encoding="utf-8-sig")
    best_single = table[table["variant"].isin(arms)]["annual"].max()
    triple = table[table["variant"] == "TRIPLE_rankavg"]["annual"].iloc[0]
    print(f"\nH1 三臂头部集成 {triple:+.2f}% vs 最好单臂 {best_single:+.2f}% "
          f"→ {'成立' if triple > best_single else '不成立'}")
    print(f"已写出 {out}")


if __name__ == "__main__":
    main()
