"""全库因子"可用性"统计：IC 显著性 × 分层单调性 × IC 稳定性。

回答：这些因子真的有用吗？
- IC 均值/ICIR/NW-t 分布（预测力是否显著）
- 5层月频分层单调比例（Q5>Q1 或 Q1>Q5 单调递增/递减）
- 单调强度（Q5-Q1 累计收益差）
- IC 稳定性：月度 IC 的符号一致性（正负交替程度）
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.factor_library import FactorLibrary  # noqa: E402

lib = FactorLibrary(dataset="hs300_2022_2025")
reg = lib.list_all()

rows = []
for _, r in reg.iterrows():
    name = r["name"]
    ic_mean = r.get("ic_mean")
    ic_ir = r.get("ic_ir")
    t_nw = r.get("t_stat_nw")
    if pd.isna(ic_mean):
        continue
    rows.append({
        "name": name,
        "family": str(r.get("source", "")).split(":")[0],
        "ic_mean": float(ic_mean), "ic_ir": float(ic_ir) if pd.notna(ic_ir) else np.nan,
        "t_nw": float(t_nw) if pd.notna(t_nw) else np.nan,
        "sig": bool(r.get("significant", False)),
    })
df = pd.DataFrame(rows)
print(f"总因子: {len(df)}")
print()

# 1. IC 均值分布
print("=== IC 均值分布 ===")
print(f"  |IC| 均值: {df['ic_mean'].abs().mean():.4f} | 中位: {df['ic_mean'].abs().median():.4f}")
print(f"  IC>0 占比: {(df['ic_mean']>0).mean()*100:.0f}% | IC<0 占比: {(df['ic_mean']<0).mean()*100:.0f}%")
print(f"  |IC|>0.03: {(df['ic_mean'].abs()>0.03).mean()*100:.0f}% | |IC|>0.05: {(df['ic_mean'].abs()>0.05).mean()*100:.0f}%")
print()

# 2. NW-t 显著
print("=== NW-t 显著性（|t_nw|>2 为显著）===")
sig_nw = df["t_nw"].abs() > 2
print(f"  |NW-t|>2: {sig_nw.sum()}/{len(df)} ({sig_nw.mean()*100:.0f}%)")
print(f"  |NW-t|>3: {(df['t_nw'].abs()>3).sum()}/{len(df)}")
print()

# 3. 分层单调性（从报告内嵌数据读 5层月频终点）
html = open("reports/factor_explorer_hs300_2022_2025.html", encoding="utf-8").read()
i0 = html.index("const DATA = "); j0 = html.index("];", i0)
DATA = json.loads(html[i0 + 13:j0 + 1])
mono_rows = []
for f in DATA:
    nav = f.get("q5_M")
    if not nav:
        continue
    months = sorted(nav.keys())
    if len(months) < 6:
        continue
    ends = [nav[months[-1]][i] for i in range(5) if nav[months[-1]][i] is not None]
    if len(ends) < 5:
        continue
    # 严格单调递增 Q5>Q4>Q3>Q2>Q1 或递减
    inc = all(ends[i] <= ends[i+1] for i in range(4))
    dec = all(ends[i] >= ends[i+1] for i in range(4))
    # 宽松：Q5 vs Q1 方向
    spread = (ends[-1] / ends[0] - 1) if ends[0] > 0 else 0
    # Spearman 单调相关
    corr = np.corrcoef(np.arange(5), ends)[0, 1]
    mono_rows.append({
        "name": f["name"], "strict_inc": inc, "strict_dec": dec,
        "spread": spread, "corr": corr,
        "ic_mean": f["m_full"]["ic"], "sig": f["m_full"]["sig"],
    })
mdf = pd.DataFrame(mono_rows)
print(f"=== 分层单调性（5层月频, {len(mdf)} 因子有数据）===")
print(f"  严格单调（递增或递减）: {((mdf['strict_inc'])|(mdf['strict_dec'])).sum()}/{len(mdf)} ({(mdf['strict_inc']|mdf['strict_dec']).mean()*100:.0f}%)")
print(f"  宽松单调相关 |corr|>0.7: {(mdf['corr'].abs()>0.7).sum()}/{len(mdf)} ({(mdf['corr'].abs()>0.7).mean()*100:.0f}%)")
print(f"  |corr|>0.5: {(mdf['corr'].abs()>0.5).sum()}/{len(mdf)}")
print(f"  Q5/Q1 收益差中位: {mdf['spread'].median()*100:.1f}%")
print()

# 4. IC 稳定性（月度 IC 符号一致性）
stab_rows = []
for f in DATA:
    icm = f.get("ic_series")
    if not icm or len(icm) < 12:
        continue
    vals = np.array([v for v in icm.values() if v is not None])
    if len(vals) < 12:
        continue
    pos = (vals > 0).mean()
    # 连续同号最长游程 / 符号翻转次数
    flips = (np.diff(np.sign(vals)) != 0).sum()
    stab_rows.append({
        "name": f["name"], "pos_ratio": pos, "flips": flips, "n": len(vals),
        "ic_mean": f["m_full"]["ic"],
    })
sdf = pd.DataFrame(stab_rows)
print(f"=== 月度 IC 稳定性（{len(sdf)} 因子）===")
print(f"  月度 IC 正占比: 中位 {sdf['pos_ratio'].median()*100:.0f}% | 25分位 {sdf['pos_ratio'].quantile(0.25)*100:.0f}% | 75分位 {sdf['pos_ratio'].quantile(0.75)*100:.0f}%")
print(f"  月度 IC 正占比 >60%: {(sdf['pos_ratio']>0.6).sum()}/{len(sdf)} ({(sdf['pos_ratio']>0.6).mean()*100:.0f}%)")
print(f"  IC 正占比与 |IC| 相关: {sdf['pos_ratio'].corr(sdf['ic_mean'].abs()):.3f}")
print()

# 5. 真正"能用"的因子：IC 显著 + 单调 + IC 稳定
print("=== 综合可用性（显著+单调相关>0.5+IC正占比>0.55）===")
merged = mdf.merge(sdf, on="name", how="inner")
merged = merged.merge(df[["name", "sig"]].rename(columns={"sig": "sig_reg"}), on="name", how="inner")
merged["sig"] = merged["sig_reg"]
use = (merged["sig"]) & (merged["corr"].abs() > 0.5) & (merged["pos_ratio"] > 0.55)
print(f"  可用因子: {use.sum()}/{len(merged)} ({use.mean()*100:.0f}%)")
print()
print("=== 可用因子 Top 10（按 |IC|）===")
use_df = merged[use].copy()
use_df["ic"] = use_df["ic_mean_x"].abs() if "ic_mean_x" in use_df.columns else use_df["ic_mean"].abs()
top = use_df.reindex(use_df["ic"].sort_values(ascending=False).index).head(10)
for _, r in top.iterrows():
    print(f"  {r['name']:20s} IC={r['ic']:+.4f} 单调corr={r['corr']:+.2f} IC正占比={r['pos_ratio']*100:.0f}% Q5/Q1差={r['spread']*100:+.0f}%")
