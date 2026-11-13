"""三档形态的**行业集中度 vs 个股集中度**逐调仓日截面诊断。

背景（2026-11-13）
------------------
`rtl_phase_tiers` 量出三档的 21 相位均值几乎打平（solo 25.13% vs focus 24.75%），
但 solo 的设计里有一条「每申万一级 ≤1 只」。既然期望没差，这条约束还剩什么价值？

去重的作用有两个方向，**互相抵消**：
- 降**行业**集中度（这是它的设计目的）
- 升**个股**集中度（`fill_to_k=False` ⇒ 踢掉不补，8 只变 5.83 只，单票权重 12.5%→17.1%）

本脚本不谈收益（那是 `SECTOR_DEDUP.md` 的活），只回答：
**在同等期望下，哪个形态的「风险预算」用得更聪明——行业真的更散了吗，代价是单票更重吗？**

四个形态
--------
- ``raw8``      : pred 前 8 只，**不做行业约束**（= P0 臂的选股口径）
- ``dedup8``    : 前 8 只里每个行业只留排名最高的一只，**踢掉不补**（= 定版 solo，实持 5.83）
- ``fill8``     : 按排名往下走，凑够 **8 个不同行业**（≈ P2 臂 `min_sectors=8`，实持 7.89）
- ``raw15``     : pred 前 15 只，不做行业约束（≈ focus 的选股口径，其实持 14.9）

指标（等权）
------------
- ``n`` / ``n_sec``：持仓只数 / 不同行业数
- ``max_sec``：最大行业权重（等权下 = 该行业只数 / n）
- ``sec_hhi``：行业权重 HHI（Σw²，越小越散）
- ``stock_hhi``：个股 HHI（= 1/n，**越大越集中**）

读数要点：``max_sec`` 与 ``stock_hhi`` 是**两个方向**，只报其一都是耍流氓。

用法::

    python -m scripts.evaluation.tier_sector_concentration
    python -m scripts.evaluation.tier_sector_concentration --phases 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal

OUT = Path("reports/tier_sector_concentration")
SIGNAL = "gbdt_mid13"


def industry_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / "industry.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def _pick_dedup(col: pd.Series, n_pool: int) -> list[str]:
    """取前 ``n_pool`` 只，每行业只留排名最高的一只，**不补位**（定版 solo 口径）。"""
    top = col.dropna().nlargest(n_pool)
    seen: set = set()
    keep = []
    for code in top.index:
        sec = IND_AT.get(code)
        if sec is None or (isinstance(sec, float) and np.isnan(sec)):
            continue
        if sec in seen:
            continue
        seen.add(sec)
        keep.append(code)
    return keep


def _pick_fill(col: pd.Series, n_sec: int) -> list[str]:
    """按排名往下走，凑够 ``n_sec`` 个不同行业（≈ P2 的 min_sectors 语义）。"""
    seen: set = set()
    keep = []
    for code in col.dropna().sort_values(ascending=False).index:
        sec = IND_AT.get(code)
        if sec is None or (isinstance(sec, float) and np.isnan(sec)):
            continue
        if sec in seen:
            continue
        seen.add(sec)
        keep.append(code)
        if len(seen) >= n_sec:
            break
    return keep


def _pick_raw(col: pd.Series, n: int) -> list[str]:
    return list(col.dropna().nlargest(n).index)


def _stats(codes: list[str], ind_row: pd.Series) -> dict:
    n = len(codes)
    if n == 0:
        return dict(n=0, n_sec=0, max_sec=np.nan, sec_hhi=np.nan, stock_hhi=np.nan)
    secs = pd.Series([ind_row.get(c) for c in codes]).dropna()
    if secs.empty:
        return dict(n=n, n_sec=0, max_sec=np.nan, sec_hhi=np.nan, stock_hhi=1.0 / n)
    w = secs.value_counts() / n
    return dict(n=n, n_sec=int(secs.nunique()), max_sec=float(w.iloc[0]),
                sec_hhi=float((w ** 2).sum()), stock_hhi=1.0 / n)


IND_AT: dict = {}


def main(argv=None) -> int:
    global IND_AT
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="0", help="相位列表，逗号分隔；21 = 全扫")
    ap.add_argument("--signal", default=SIGNAL)
    a = ap.parse_args(argv)
    phases = [int(p) for p in a.phases.split(",") if p.strip() != ""]

    ctx = cal.load_context(signal=a.signal, window="prod", end=20260907,
                           execution="open")
    ind = industry_panel(ctx)
    forms = (("raw8", lambda c, r: _pick_raw(c, 8)),
             ("dedup8", lambda c, r: _pick_dedup(c, 8)),
             ("fill8", lambda c, r: _pick_fill(c, 8)),
             ("raw15", lambda c, r: _pick_raw(c, 15)))

    rows = []
    for p in phases:
        rb = cal.rebalance_dates(ctx.pred.index, freq="M", phase=p)
        for t in rb:
            if t not in ctx.pred.index:
                continue
            col = ctx.pred.loc[t]
            IND_AT = ind.loc[t].to_dict() if t in ind.index else {}
            for name, fn in forms:
                codes = fn(col, t)
                rows.append({"phase": p, "date": t, "form": name,
                             **_stats(codes, ind.loc[t])})
    df = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "daily_forms.csv", index=False, encoding="utf-8-sig")
    sm = df.groupby("form")[["n", "n_sec", "max_sec", "sec_hhi", "stock_hhi"]].mean()
    sm["n_period"] = df.groupby("form").size()
    sm = sm.loc[[f[0] for f in forms]]
    print(f"\n=== 截面集中度（{a.signal} · 相位 {phases} · {len(df) // len(forms)} 期）===")
    print(sm.round(4).to_string())
    sm.to_csv(OUT / "summary.csv", encoding="utf-8-sig")

    # 相位轴稳定性：max_sec 的期分布，避免「均值掩盖尾部」
    q = df.groupby("form")["max_sec"].quantile([0.5, 0.9, 1.0]).unstack()
    q.columns = ["p50", "p90", "max"]
    print("\n=== 最大行业权重的分布（尾部才是风险）===")
    print(q.loc[[f[0] for f in forms]].round(4).to_string())
    q.loc[[f[0] for f in forms]].to_csv(OUT / "max_sec_quantile.csv",
                                        encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
