"""EV / 研发 四因子的组合层验证：IC 好不等于赚钱。

上方 `ev_rd_validate` 只回答"有没有预测力"，本脚本回答"能不能变成钱"。
口径对齐生产链（月频 · Top10% 等权 · 往返 27bp），但**用收盘价对收盘价**，
不做 T+1 open 撮合——这里判的是因子相对优劣，不是绝对收益，所以统一简化
不影响排序结论。若某臂值得上生产，再单独走 `port_backtest` 的完整撮合。

关键设计：**统一股票池**。`rd_to_cap` 只有 70% 覆盖，若各臂在各自最大池里选股，
组合差异里会混进"池子不同"效应。故主表用全部候选因子同时非空的交集池；
另报"全池 vs 交集池"的池子效应量化。

成本：往返 27bp（佣金 2 + 印花 5 + 滑点 20），``cost = turn × 27bp``，
turn = Σ|Δw|/2（单边换手）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from scripts.builders.common import load_close_adj  # noqa: E402

NEW_DIR = ROOT / "reports" / "factor_ev_rd" / "panels"
LIB = Path(str(Config.get()["factor_library"]["root"])) / "all_a_2018_2026" / "panels"
OUT = ROOT / "reports" / "factor_ev_rd"

BP_ROUNDTRIP = 27.0     # bp
TOP_FRAC = 0.10
START = "2019-01-01"


def load(name: str) -> pd.DataFrame | None:
    for d in (NEW_DIR, LIB):
        p = d / f"{name}.parquet"
        if p.exists():
            d0 = pd.read_parquet(p).astype(np.float64)
            if not isinstance(d0.index, pd.DatetimeIndex):
                d0.index = pd.to_datetime(d0.index)
            return d0
    return None


def month_ends(idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """每月最后一个交易日（信号日），执行用下一交易日。"""
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).last())


def backtest(sig: pd.DataFrame, px: pd.DataFrame, reb: list[pd.Timestamp],
             pos: dict[pd.Timestamp, int], top_frac: float = TOP_FRAC,
             cost_bp: float = BP_ROUNDTRIP) -> dict | None:
    w_old = pd.Series(0.0, index=px.columns)
    rows, turns = [], []
    for i, d in enumerate(reb[:-1]):
        p0, p1 = pos[d], pos[reb[i + 1]]
        s = sig.loc[d].replace([np.inf, -np.inf], np.nan)
        ok = s.notna() & px.iloc[p0].notna() & px.iloc[p1].notna()
        s = s[ok]
        if len(s) < 50:
            continue
        k = max(int(len(s) * top_frac), 1)
        pick = s.nlargest(k).index
        w = pd.Series(0.0, index=px.columns)
        w[pick] = 1.0 / len(pick)
        hold = (px.iloc[p1] / px.iloc[p0] - 1.0)
        gross = float((w * hold.fillna(0.0)).sum())
        turn = float((w - w_old).abs().sum() / 2)
        rows.append(gross - turn * cost_bp / 1e4)
        turns.append(turn)
        w_old = w
    if len(rows) < 24:
        return None
    r = pd.Series(rows)
    ann = float((1 + r).prod() ** (12 / len(r)) - 1)
    return {"ann": ann, "vol": float(r.std(ddof=1) * np.sqrt(12)),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12))
            if r.std(ddof=1) > 0 else np.nan,
            "mdd": float((1 + r).cumprod().div((1 + r).cumprod().cummax()).min() - 1),
            "turnover": float(np.mean(turns) * 12), "n": len(r), "ret": r}


def bench(px: pd.DataFrame, reb: list[pd.Timestamp],
          pos: dict[pd.Timestamp, int]) -> pd.Series:
    rows = []
    for i, d in enumerate(reb[:-1]):
        p0, p1 = pos[d], pos[reb[i + 1]]
        h = (px.iloc[p1] / px.iloc[p0] - 1.0).dropna()
        rows.append(float(h.mean()) if len(h) else np.nan)
    return pd.Series(rows).dropna()


def main() -> int:
    px = load_close_adj()
    idx = px.index[px.index >= pd.Timestamp(START)]
    reb = month_ends(idx)
    pos = {d: px.index.get_loc(d) for d in reb}

    names = ["bp", "bp_ev", "ep_ttm", "ep_ev", "ep_rd", "sp_ev",
             "rd_to_cap", "sp_ttm"]
    panels = {n: load(n) for n in names}
    panels = {k: v.reindex(index=px.index, columns=px.columns)
              for k, v in panels.items() if v is not None}

    # 交集池：主表所有候选因子同时非空
    core = ["ep_ttm", "ep_rd", "bp", "bp_ev"]
    inter = None
    for c in core:
        m = panels[c].loc[reb].notna()
        inter = m if inter is None else (inter & m)
    pool_mask = inter

    def _rank(df: pd.DataFrame) -> pd.DataFrame:
        """截面 pct rank，统一方向（越大越好）。"""
        return df.where(pool_mask).rank(axis=1, pct=True)

    def _combo(*fs: str) -> pd.DataFrame:
        acc = None
        for f in fs:
            r = _rank(panels[f])
            acc = r if acc is None else acc + r
        return acc

    arms: dict[str, pd.DataFrame] = {}
    for f in names:
        if f in panels:
            arms[f] = _rank(panels[f])
            arms[f"{f}(各自池)"] = panels[f].rank(axis=1, pct=True) * -1.0 \
                if f.startswith("bp") else panels[f].rank(axis=1, pct=True)
    arms["bp+bp_ev"] = _combo("bp", "bp_ev")
    arms["ep_ttm+rd_to_cap"] = _combo("ep_ttm", "rd_to_cap")
    arms["ep_ttm+rd_to_cap+bp_ev"] = _combo("ep_ttm", "rd_to_cap", "bp_ev")

    bh = bench(px, reb, pos)
    bann = float((1 + bh).prod() ** (12 / len(bh)) - 1)

    rows = []
    for name, s in arms.items():
        # 各自池的臂不能与交集池混在一起比较，单独标记
        if name.endswith("(各自池)"):
            continue
        r = backtest(s, px, reb, pos)
        if r is None:
            continue
        rows.append({
            "arm": name, "ann": r["ann"], "ex_geo": r["ann"] - bann,
            "sharpe": r["sharpe"], "mdd": r["mdd"],
            "turnover": r["turnover"], "n": r["n"],
        })

    df = pd.DataFrame(rows).sort_values("ex_geo", ascending=False)
    pd.set_option("display.width", 200, "display.unicode.east_asian_width", True)
    print(f"\n=== 主表（交集池，月频 Top{TOP_FRAC:.0%} 等权，往返 {BP_ROUNDTRIP:.0f}bp）===")
    print(f"    交集池规模：中位 {int(pool_mask.sum(axis=1).median())} 只 / 全市场 "
          f"{px.shape[1]} 只；基准等权年化 {bann:.4f}")
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # 池子效应：ep_ttm 在交集池 vs 全池
    full = panels["ep_ttm"].rank(axis=1, pct=True)
    r_full = backtest(full, px, reb, pos)
    r_int = backtest(arms["ep_ttm"], px, reb, pos)
    print("\n=== 池子效应（ep_ttm）===")
    print(f"  全池    年化 {r_full['ann']:+.4f}  换手 {r_full['turnover']:.0%}")
    print(f"  交集池  年化 {r_int['ann']:+.4f}  换手 {r_int['turnover']:.0%}")
    print(f"  Δ = {r_int['ann'] - r_full['ann']:+.4f}（交集池剔除无研发的公司）")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "bt_main.csv", index=False)
    print(f"\n产物 → {OUT / 'bt_main.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
