"""分池 IC 诊断 —— 解释「为什么信号在指数池上衰减」。

用途（2026-11-05）：`pool_scope_probe` 测出全A 超额 +11.8%、hs300 −0.2%，
本脚本回答「信号本身的预测力是不是也随池子变弱」。

口径
----
* IC：**逐日 Spearman rank IC**，信号 T 日 vs **T+1 收益**（``close.pct_change().shift(-1)``，
  与因子库/IC 口径一致；注意引擎 h=1 用的是未 shift 版本，见 MEMORY 铁律）。
* 分组：各指数池按 PIT 成分过滤；``idx_union`` = 四池并集；``non_index`` = 全A 减去并集。
* 只统计当日有效股数 ≥ 30 的截面。
* ``ICIR`` = 日均 IC / 日标准差 × √252；``t`` = 日均 IC / 标准差 × √(天数)
  （未做 NW 校正，仅作量级参考）。
* Top15% 构成：每月首个交易日按信号取全A 前 15%，统计其中各池成分占比。

⚠️ IC 是**双向**的：它同时反映多头端与空头端。A 股不可做空，所以 IC 高不等于
多头端能拿到——多头端是否有效要看 `pool_scope_probe` 的超额列，不要用本脚本下结论。

用法
----
    python -m scripts.evaluation.pool_ic_diagnose
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

from scripts.evaluation.pool_scope_probe import POOLS, pool_mask  # noqa: E402

BASE_DIR = ROOT / "reports" / "alla_rolling" / "_base"


def row_ic(a: pd.DataFrame, b: pd.DataFrame, min_n: int = 30) -> pd.Series:
    """逐行 Spearman IC（先各自截面排序，再算 Pearson）。"""
    ar, br = a.rank(axis=1), b.rank(axis=1)
    am = ar.sub(ar.mean(axis=1), axis=0)
    bm = br.sub(br.mean(axis=1), axis=0)
    num = (am * bm).sum(axis=1)
    den = np.sqrt((am ** 2).sum(axis=1) * (bm ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    n = a.notna().sum(axis=1) & b.notna().sum(axis=1)
    return ic.where(n >= min_n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-path", default="reports/alla_rolling_ortho/pred/ens_h1h5__h1.parquet")
    ap.add_argument("--begin", type=int, default=20180101)
    ap.add_argument("--end", type=int, default=20260907)
    ap.add_argument("--top-frac", type=float, default=0.15)
    args = ap.parse_args()

    pred = pd.read_parquet(ROOT / args.signal_path)
    pred = pred.loc[(pred.index >= pd.Timestamp(str(args.begin)))
                    & (pred.index <= pd.Timestamp(str(args.end)))]
    close = pd.read_parquet(BASE_DIR / "close_adj.parquet").reindex(
        index=pred.index, columns=pred.columns)
    fwd = close.pct_change(fill_method=None).shift(-1)
    pred = pred.where(fwd.notna())

    idx, cols = pred.index, pred.columns
    masks = {"all_a": pd.DataFrame(True, index=idx, columns=cols)}
    for k, c in POOLS.items():
        masks[k] = pool_mask(c, idx, cols)
    masks["idx_union"] = (masks["hs300"] | masks["zz500"]
                          | masks["zz1000"] | masks["zz2000"])
    masks["non_index"] = ~masks["idx_union"]

    print("=== 逐日 rank IC（T 信号 vs T+1 收益）===")
    print(f"{'组':10s} {'有效股中位':>8s} {'IC均值':>9s} {'ICIR(年化)':>10s} "
          f"{'t':>7s} {'IC>0占比':>9s}")
    for k, m in masks.items():
        p, r = pred.where(m), fwd.where(m)
        ic = row_ic(p, r).dropna()
        if len(ic) < 50:
            print(f"{k:10s} 样本不足 {len(ic)}")
            continue
        mean_, std_ = ic.mean(), ic.std()
        nm = int((p.notna() & r.notna()).sum(axis=1).median())
        print(f"{k:10s} {nm:8d} {mean_:9.4f} {mean_/std_*np.sqrt(252):10.2f} "
              f"{mean_/std_*np.sqrt(len(ic)):7.1f} {(ic > 0).mean()*100:8.1f}%")

    print("\n=== 月首 Top15% 持仓的成分构成 ===")
    ks = ["hs300", "zz500", "zz1000", "zz2000", "non_index"]
    s = pd.Series(idx, index=idx)
    firsts = list(s.groupby(s.index.to_period("M")).first())
    rows = []
    for d in firsts:
        sc = pred.loc[d].dropna()
        if len(sc) < 100:
            continue
        n = max(1, int(round(args.top_frac * len(sc))))
        top = sc.nlargest(n).index
        n_all = float(masks["all_a"].loc[d].sum()) or 1.0
        rows.append({
            "top": {k: float(masks[k].loc[d].reindex(top).fillna(False).mean()) for k in ks},
            # 基准占比**逐月同期**取，避免跨年分母漂移（全A 只数 2018→2026 由 ~3500 增到 5800）
            "base": {k: float(masks[k].loc[d].sum()) / n_all for k in ks},
        })
    comp_top = pd.DataFrame([r["top"] for r in rows])
    comp_base = pd.DataFrame([r["base"] for r in rows])
    ratio = (comp_top / comp_base.replace(0, np.nan)).median()
    print(f"样本 {len(comp_top)} 个月")
    print(f"{'组':10s} {'Top15%内占比':>12s} {'同期全市场占比':>13s} {'逐月超配倍数':>12s}")
    for k in comp_top.columns:
        print(f"{k:10s} {comp_top[k].median()*100:11.1f}% "
              f"{comp_base[k].median()*100:12.1f}% {ratio[k]:11.2f}x")
    print("注：zz2000 仅 2023-08 起有成分；其 Top15% 占比为全窗中位（含无成分月份）。")
    print("    超配倍数逐月计算再取中位，>1 = 信号偏好该组。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
