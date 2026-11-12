"""因子库有效独立维度（n_eff）度量：判断"挖出很多因子"里有多少是新的。

背景（2026-11-12）：读华泰 AI95《以空间换时间》后要判断「要不要把 GP 从单目标
改成三目标 NSGA-II」。作者的动机是单目标会让 `x+y`、`x+0.99y` 这类同质因子
指数级复制。我们是否也有这个问题，此前没有量过——库内 950 个因子、
`registry.csv` 有 `dup_corr_max`（逐对最大相关），但没有**全库维度**口径。

度量（Krishnamurthy 等的 participation ratio，用于表征学习里衡量表征维度）：
    ``n_eff = (Σλ)² / Σλ²``，λ 为因子间相关矩阵的特征值。
等价解读：n 个因子等价于 n_eff 个正交因子。n 个完全独立 ⇒ n_eff = n；
n 个完全相同 ⇒ n_eff = 1；均匀分成 k 个一维簇 ⇒ n_eff ≈ k。

为什么不用「平均相关系数折算」：``n/(1+(n-1)ρ̄)`` 需要 ρ̄ 小且同质，
因子库里有整簇高相关时会被单簇支配；PR 是谱口径，对簇大小自适应。

⚠️ 口径边界：
- 用 **IC 序列**（`ic_h20.parquet`，936×2511）而非因子值面板。IC 序列相关
  衡量的是"预测力的同质性"，两个因子值不相关但 IC 同步的概率存在，反之亦然。
  对"因子库有没有重复造轮子"这个问题，IC 口径更贴题，但不等价于因子值相关。
- 日频 IC 自相关高 ⇒ 所有 ρ 被系统性抬高 ⇒ n_eff 偏低。这是**保守方向**
  （低估库内多样性），结论若是"已经同质"则更可信。
- 只用 h=20：价值/基本面类因子在 h=1 上无信号（本仓库 2026-11-12 实测
  `rd_to_cap` h=1 IC 0.003 / h=20 0.052），h=1 口径会把这类因子噪声化。

用法:
    python -m scripts.evaluation.factor_library_diversity
    python -m scripts.evaluation.factor_library_diversity --h 1
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

LIB = Path(str(Config.get()["factor_library"]["root"])) / "all_a_2018_2026"
OUT = ROOT / "reports" / "factor_ev_rd"

MIN_COVER = 0.50      # IC 序列非空比例下限
MIN_OBS = 200         # 两两相关的最少重叠样本


def participation_ratio(corr: np.ndarray) -> float:
    """PR = (Σλ)² / Σλ²。corr 须为方阵、对角为 1。"""
    lam = np.linalg.eigvalsh(corr)
    lam = np.clip(lam, 1e-12, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def group_stats(ic: pd.DataFrame, tag: str) -> tuple[dict, np.ndarray]:
    n = ic.shape[1]
    if n < 3:
        return {"group": tag, "n": n, "n_eff": np.nan,
                "ratio": np.nan, "mean_abs_rho": np.nan, "top_eig": np.nan}, np.array([])
    corr = ic.corr(min_periods=MIN_OBS)
    c = np.nan_to_num(corr.to_numpy(), nan=0.0)
    np.fill_diagonal(c, 1.0)
    # 相关矩阵必须半正定；NaN 填 0 可能破坏，投影到最近 PSD 会掩盖问题，
    # 这里直接对最小特征值截断（等价于"缺失即不相关"的保守假设）。
    lam, vec = np.linalg.eigh(c)
    lam = np.clip(lam, 1e-12, None)
    c_psd = (vec * lam) @ vec.T
    d = np.sqrt(np.diag(c_psd))
    c_psd = c_psd / np.outer(d, d)
    n_eff = participation_ratio(c_psd)
    iu = np.triu_indices(n, k=1)
    return {
        "group": tag, "n": n,
        "n_eff": round(n_eff, 2),
        "ratio": round(n_eff / n, 4),
        "mean_abs_rho": round(float(np.nanmean(np.abs(c[iu]))), 4),
        "pairs_gt_090": int((np.abs(c[iu]) >= 0.90).sum()),
        "pairs_gt_075": int((np.abs(c[iu]) >= 0.75).sum()),
        "top_eig_share": round(float(lam[-1] / lam.sum()), 4),
    }, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--min-cover", type=float, default=MIN_COVER)
    args = ap.parse_args()

    p = LIB / f"ic_h{args.h}.parquet"
    if not p.exists():
        print(f"缺 {p}")
        return 2
    ic = pd.read_parquet(p).astype(np.float64)
    reg = pd.read_csv(LIB / "registry.csv")
    fam = reg.set_index("name")["family"].to_dict()

    keep = ic.columns[ic.notna().mean() >= args.min_cover]
    dropped = len(ic.columns) - len(keep)
    ic = ic[keep]
    print(f"ic_h{args.h}: {p.stat().st_size/1e6:.1f}MB  "
          f"因子 {len(keep)}（丢弃覆盖<{args.min_cover:.0%} 的 {dropped} 个）"
          f"  日期 {ic.index.min().date()}~{ic.index.max().date()}")

    rows, corr_full = [], np.array([])
    allst, corr_full = group_stats(ic, "全库")
    rows.append(allst)

    labels = pd.Series({c: fam.get(c, "未登记") for c in ic.columns})
    for g, cols in labels.groupby(labels):
        sub = ic[[c for c in cols.index if c in ic.columns]]
        st, _ = group_stats(sub, g)
        rows.append(st)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.unicode.east_asian_width", True)
    print("\n=== 有效独立维度（h=%d，PR 口径）===" % args.h)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 最冗余的因子对：给出具体名字，方便后续去重
    if corr_full.size:
        iu = np.triu_indices(corr_full.shape[0], k=1)
        vals = corr_full[iu]
        order = np.argsort(-np.abs(vals))[:20]
        pairs = pd.DataFrame({
            "a": [ic.columns[iu[0][i]] for i in order],
            "b": [ic.columns[iu[1][i]] for i in order],
            "rho": vals[order],
        })
        print("\n=== 最冗余的 20 对（|ρ| 降序）===")
        print(pairs.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        fam_a = [fam.get(x, "未登记") for x in pairs["a"]]
        fam_b = [fam.get(x, "未登记") for x in pairs["b"]]
        pairs["fam_a"], pairs["fam_b"] = fam_a, fam_b
    else:
        pairs = pd.DataFrame()

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"diversity_h{args.h}.csv", index=False)
    if not pairs.empty:
        pairs.to_csv(OUT / f"diversity_pairs_h{args.h}.csv", index=False)
    print(f"\n产物 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
