"""
ETF 池合理性 / 充分性审计（2026-10-27）
======================================

扩池后对候选池做一次体检，回答三个问题：

  1. **合规**：每只是否满足「上市满 3 年 + 60 日均额 ≥ 3000 万」，当前余量多少；
  2. **冗余**：50 只不是 50 个独立机会——算相关矩阵、近重复对、有效独立资产数，
     并统计 Top3 组合里「同涨同跌双胞胎」与「同一类别」的占比；
  3. **充分**：PIT 可投数量随时间的增长、12-1 动量截面离散度是否够选标。

只读缓存，不写任何东西；结论回写 `reports/etf_pool_expand/README.md`。

用法:
    python -m scripts.portfolio.etf_pool_audit
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_rotation_p2 import build_extra_panels
from scripts.portfolio.etf_rotation_probe import (
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

MIN_YEARS = 3
MIN_AMOUNT = 3e7


def main() -> None:
    cache = DataCache(create_datasource())
    uni, close, amount, _ = load_bundle(cache, WARMUP_BEGIN, None)
    panels = build_panels(close, amount)
    last = close.index[-1]
    amt60 = amount.rolling(60, min_periods=40).mean().loc[last]

    # ---- 1. 合规 ----
    rows = []
    for c in close.columns:
        s = close[c].dropna()
        rows.append({
            "code": c, "name": uni.label(c), "cat": uni.category(c),
            "first": s.index[0].date(),
            "years": round((last - s.index[0]).days / 365.25, 1),
            "amt60_yi": round(float(amt60.get(c, np.nan)) / 1e8, 2),
        })
    t = pd.DataFrame(rows).sort_values("years")
    print("[1] 合规：上市年限最短 / 成交额最小")
    print(t.head(3).to_string(index=False))
    print(t.sort_values("amt60_yi").head(3).to_string(index=False))
    print(f"   不足 {MIN_YEARS} 年：{t[t.years < MIN_YEARS].code.tolist() or '无'}；"
          f"60 日均额 < {MIN_AMOUNT/1e8:.1f} 亿："
          f"{t[t.amt60_yi < MIN_AMOUNT/1e8].code.tolist() or '无'}")

    # ---- 2. 冗余 ----
    r = close.pct_change(fill_method=None).loc["2024-01-01":]
    c2 = r.corr()
    codes = list(c2.columns)
    arr = c2.to_numpy()
    iu = np.triu_indices_from(arr, 1)
    vals = arr[iu]
    pairs = sorted([(float(vals[i]), codes[iu[0][i]], codes[iu[1][i]])
                    for i in range(len(vals)) if np.isfinite(vals[i])],
                   reverse=True)[:10]
    ev = np.linalg.eigvalsh(c2.fillna(0).to_numpy())
    print(f"\n[2] 相关结构（2024-01 起）：均值 {np.nanmean(vals):.2f}，"
          f">0.9 共 {int((vals > 0.9).sum())} 对，>0.95 共 {int((vals > 0.95).sum())} 对；"
          f"有效独立资产 {ev.sum()**2 / (ev**2).sum():.1f} / {len(codes)}")
    for v, a, b in pairs[:8]:
        print(f"   {v:.3f}  {a} {uni.label(a)} ~ {b} {uni.label(b)}")

    raw = (cache.read_daily("etf")["close"].unstack("code")
           .reindex(index=close.index, columns=close.columns))
    full = build_extra_panels(close, panels, raw)
    spec = {"name": "audit", "signal": "mom_crowd_f20", "freq": "W", "top_k": 3,
            "absolute": True, "weighting": "equal"}
    w = run_variant((uni, close, amount, _), full, spec, 20200101, None)["weights"]
    cat_of = {c: uni.category(c) for c in w.columns}
    dup = [(codes[i], codes[j]) for i, j in zip(*np.where(arr > 0.95)) if i < j]

    def _dup(row: pd.Series) -> bool:
        live = {c for c in row.index if row[c] > 0}
        return any(a in live and b in live for a, b in dup)

    same, mx = 0, []
    for _, row in w.iterrows():
        live = [c for c in row.index if row[c] > 0]
        if len(live) < 2:
            continue
        cs = pd.Series([cat_of.get(c, "?") for c in live]).value_counts(normalize=True)
        mx.append(float(cs.iloc[0]))
        same += int(cs.iloc[0] == 1.0)
    print(f"   Top3 组合：含 >0.95 双胞胎的天数 {100 * w.apply(_dup, axis=1).mean():.1f}%；"
          f"同类占比均值 {np.mean(mx):.2f}；全同类天数 {100 * same / len(mx):.1f}%")
    print("   注：双胞胎 = 同指数/同风格的两只产品，如 512480 半导体 ~ 512760 芯片、"
          "159915 创业板 ~ 159949 创业板50")

    # ---- 3. 充分 ----
    elig = panels["mask"].sum(axis=1)
    print("\n[3] PIT 可投数量（年末）：",
          {str(d.date()): int(v) for d, v in elig.resample("YE").last().items()})
    disp = panels["mom12_1"].where(panels["mask"]).std(axis=1).resample("QE").mean()
    print(f"   12-1 动量截面标准差：近三年区间 "
          f"{disp.tail(12).min():.2f} ~ {disp.tail(12).max():.2f}（中位 "
          f"{disp.tail(12).median():.2f}）")


if __name__ == "__main__":
    main()
