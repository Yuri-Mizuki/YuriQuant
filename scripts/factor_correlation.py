"""
因子相关性矩阵报告
==================

对因子库某数据集的全部 raw 因子计算两两 spearman 相关（逐日截面相关再
时间平均），输出：
1. 平均相关矩阵 CSV（reports/factor_corr_<dataset>.csv）
2. 热力图 PNG（reports/factor_corr_<dataset>.png，层次聚类重排）
3. 聚类分组（基于 1-|corr| 距离的层次聚类，输出每组因子）

用途：识别因子冗余/因子族，判断合成前哪些因子是"真正的增量信息"。

用法
----
    python -m scripts.factor_correlation --dataset hs300_2025
    python -m scripts.factor_correlation --dataset hs300_2025 --kind raw
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cli_common import setup_logging  # noqa: E402


matplotlib.use("Agg")
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from research.factor_library import FactorLibrary  # noqa: E402

log = setup_logging("factor_correlation")

plt.rcParams["axes.unicode_minus"] = False

def mean_corr_matrix(panels: dict[str, pd.DataFrame], method: str = "spearman") -> pd.DataFrame:
    """逐日截面相关矩阵的时间平均。

    panels: {因子名: date×code 面板}。两两因子的逐日相关取平均；
    某天样本不足（<10）跳过。
    """
    names = list(panels)
    dates = panels[names[0]].index
    n = len(names)
    acc = np.zeros((n, n))
    cnt = np.zeros((n, n))
    for d in dates:
        cols = []
        for name in names:
            s = panels[name].loc[d].dropna()
            cols.append(s)
        common = None
        for s in cols:
            common = s.index if common is None else common.intersection(s.index)
        if len(common) < 10:
            continue
        mat = pd.concat([s[common] for s in cols], axis=1)
        mat.columns = names
        c = mat.corr(method=method).values
        valid = ~np.isnan(c)
        acc[valid] += c[valid]
        cnt[valid] += 1
    with np.errstate(invalid="ignore"):
        avg = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    return pd.DataFrame(avg, index=names, columns=names)

def cluster_groups(corr: pd.DataFrame, threshold: float = 0.5) -> dict[int, list[str]]:
    """层次聚类分组：距离 = 1 - |corr|，threshold 为合并阈值（corr >= 1-threshold 一组）。"""
    d = (1.0 - corr.abs().values)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    condensed = squareform(d, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    groups: dict[int, list[str]] = {}
    for name, lab in zip(corr.index, labels):
        groups.setdefault(int(lab), []).append(name)
    return groups

def main():
    parser = argparse.ArgumentParser(description="因子相关性矩阵报告")
    parser.add_argument("--dataset", default="hs300_2025")
    parser.add_argument("--kind", default="raw", help="raw | composite | None(全部)")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--cluster-threshold", type=float, default=0.5,
                        help="聚类阈值：|corr|>=1-threshold 视为一族")
    args = parser.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    panels = lib.load_library_features(kind=args.kind)
    if not panels:
        log.error("数据集 %s 无 %s 因子", args.dataset, args.kind)
        return
    log.info("因子数: %d", len(panels))
    corr = mean_corr_matrix(panels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"factor_corr_{args.dataset}_{args.kind or 'all'}"

    # CSV
    csv_path = out_dir / f"{stem}.csv"
    corr.round(4).to_csv(csv_path)
    log.info("相关矩阵已保存: %s", csv_path)

    # 聚类 + 热力图（层次聚类重排）
    groups = cluster_groups(corr, args.cluster_threshold)
    order = [g for _, g in sorted(groups.items())]
    ordered = [n for g in groups.values() for n in g]
    corr_ordered = corr.loc[ordered, ordered]

    fig, ax = plt.subplots(figsize=(max(10, len(panels) * 0.55), max(8, len(panels) * 0.5)))
    im = ax.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered)))
    ax.set_yticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=90, fontsize=7)
    ax.set_yticklabels(ordered, fontsize=7)
    # 聚类组边界线
    idx = 0
    for g in groups.values():
        idx += len(g)
        if idx < len(ordered):
            ax.axhline(idx - 0.5, color="#2C2C2A", linewidth=0.8)
            ax.axvline(idx - 0.5, color="#2C2C2A", linewidth=0.8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{args.dataset} raw 因子平均 spearman 相关（层次聚类，阈值 {args.cluster_threshold}）")
    fig.tight_layout()
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("热力图已保存: %s", png_path)

    # 控制台分组 + 组内平均 |corr|
    log.info("===== 聚类分组（|corr| >= %.2f 视为一族）=====", 1 - args.cluster_threshold)
    for gi, g in groups.items():
        sub = corr.loc[g, g]
        vals = sub.abs().values[np.triu_indices(len(g), k=1)]
        mean_abs = float(np.nanmean(vals)) if vals.size else float("nan")
        log.info("族%d (组内均值|corr|=%.2f, %d 个): %s",
                 gi, mean_abs, len(g), ", ".join(g))

    # 高相关对
    pairs = []
    names = list(corr.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = corr.iloc[i, j]
            if not np.isnan(v) and abs(v) >= 0.6:
                pairs.append((v, names[i], names[j]))
    pairs.sort(key=lambda x: -abs(x[0]))
    log.info("===== |相关| >= 0.6 的因子对 (%d 对) =====", len(pairs))
    for v, a, b in pairs[:20]:
        log.info("  %.2f  %s  ~  %s", v, a, b)

if __name__ == "__main__":
    main()