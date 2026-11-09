"""31 个申万一级的「有效独立方向数」诊断（P0-1）
=================================================

问题来源
--------
行业轮动线的硬约束是**统计功效**：31 个点的截面、单期 IC 标准差 0.254、
t=2 需 |IC| ≥ 0.050。但"31 个点"不等于"31 个独立赌注"——如果其中很多行业
彼此高度相关，那 Top5 实际押的是更少的几注。本脚本量化这件事。

🚨 适用范围（**别误用**，2026-11-09 自查纠正）
--------------------------------------------
本脚本的 n_eff **不能**用来放大 IC 的 t 统计量。``t = mean(IC) / std(IC) · √T``
里的 ``std(IC)`` 是**跨期经验标准差**，行业间相关已经包含在里面了（相关越高
→ 单期 IC 波动越大 → t 反而越小）。所以"n_eff 低于 31 ⇒ t 被高估"是**错的**。

n_eff 真正回答的是三件事：
  ① **赌注数**：Top5-of-31 到底押了几注（→ Top-K 该不该按簇去重）；
  ② **换池翻号的机制**：两个 ρ=0.95 的行业互换，组合几乎不变，超额却会翻号；
  ③ **这条线的功效上限**：31 个行业只承载 N 个维度的变异，可分出的序就那么多。

为什么必须报「残差」口径（原始口径会误导）
------------------------------------------
31 个行业的原始收益有一个共同的大盘因子，PC1 独占 **58%**。但截面 IC 是
**排序统计**——共同因子对所有行业同向，不影响排序。真正决定"能不能分开名次"
的是**扣掉大盘之后**的离散度。所以：

- ``raw``  口径：直接对行业收益求相关 —— 回答"31 个行业像不像一个东西"；
- ``resid`` 口径：对 31 行业等权做 β 残差后再求相关 —— 回答"扣掉大盘还剩几个
  独立方向"。**后者才是 Top-K 分散度与 IC 功效该看的口径。**
残差化方式与 ``scripts.portfolio.corr_sector`` / ``reports/etf_coverage`` 同源
（β 残差、市场 = 31 行业等权），避免同一个概念三套实现。

用法
----
    python -m scripts.portfolio.industry_neff_probe
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from config import Config

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "industry_rotation"
PQ = Path(Config.paths()["cache_root"])

EVAL_START = "2018-01-02"
EVAL_END = "2026-10-30"
RHO_CUTS = (0.50, 0.60, 0.70, 0.80, 0.90)
#: Top-K 跨度诊断用哪个阈值做主口径（残差口径下 0.80 已能把同族并起来）
SPAN_RHO = 0.80
TOP_K = 5
MC_N = 5000


# --------------------------------------------------------------------------- 数据
def load_ret() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(PQ / "sw_industry_daily.parquet")
    px = d.pivot_table(index="date", columns="code", values="close").sort_index()
    px = px.loc[EVAL_START:EVAL_END]
    ret_d = px.pct_change(fill_method=None)
    ret_m = px.resample("ME").last().pct_change(fill_method=None)
    return ret_d, ret_m


def resid(r: pd.DataFrame, m: pd.Series) -> pd.DataFrame:
    """逐列对市场收益做单变量 OLS 取残差（NaN 行不参与估计）。"""
    A = r.to_numpy(dtype=float)
    mv = m.to_numpy(dtype=float)
    ok = np.isfinite(A) & np.isfinite(mv)[:, None]
    n = ok.sum(axis=0).astype(float)
    Az = np.where(ok, A, 0.0)
    Mz = np.where(ok, mv[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = Az.sum(axis=0) / n
        mm = Mz.sum(axis=0) / n
    Mc = Mz - mm
    varm = (Mc ** 2).sum(axis=0)
    cov = (np.where(ok, A - my, 0.0) * Mc).sum(axis=0)
    beta = np.divide(cov, varm, out=np.zeros_like(cov), where=varm > 0)
    R = np.where(ok, (A - my) - beta * (mv[:, None] - mm), np.nan)
    return pd.DataFrame(R, index=r.index, columns=r.columns)


def pairwise_corr(r: pd.DataFrame) -> pd.DataFrame:
    """成对完整相关：不丢行不丢列，样本量下限取 min(全长, max(12, 50%))。

    🚨 用 ``max(60, ...)`` 这种硬下限会在月频分段（45 期）上把所有格点判成
    NaN —— 2026-11-09 首次实现即踩到，表现为「mean_corr = 0 / n_eff = 31」。
    """
    mp = min(len(r), max(12, int(0.5 * len(r))))
    return r.corr(min_periods=mp)


def cluster_labels(c: pd.DataFrame, rho_cut: float) -> pd.Series:
    """相关矩阵 → 层次聚类（平均连接）→ 在给定 ρ 阈值处切簇。"""
    n = c.shape[0]
    if n == 1:
        return pd.Series([1], index=c.index)
    A = c.to_numpy(dtype=float)
    A = np.where(np.isnan(A), 0.0, A)
    dist = np.sqrt(np.clip(0.5 * (1.0 - A), 0.0, 1.0))
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")
    lab = fcluster(link, t=float(np.sqrt(0.5 * (1.0 - rho_cut))), criterion="distance")
    return pd.Series(lab, index=c.index)


# --------------------------------------------------------------------------- 指标
def metrics_from_corr(c: pd.DataFrame) -> dict:
    n = c.shape[0]
    A = c.to_numpy(dtype=float)
    off = ~np.eye(n, dtype=bool)
    n_missing = int(np.isnan(A[off]).sum() // 2)
    A = np.where(np.isnan(A), 0.0, A)
    np.fill_diagonal(A, 1.0)
    lam = np.clip(np.linalg.eigvalsh(A)[::-1], 1e-12, None)
    tot = lam.sum()
    p = lam / tot
    rho_bar = float(A[off].sum() / (n * (n - 1)))
    cum = np.cumsum(p)
    return {
        "mean_corr": rho_bar,
        "n_eff_avgcorr": n / (1.0 + (n - 1.0) * max(rho_bar, 0.0)),
        "participation": float(tot ** 2 / (lam ** 2).sum()),
        "eff_rank": float(np.exp(-(p * np.log(p)).sum())),
        "pc1_share": float(p[0]),
        "n_pc_80": int(np.searchsorted(cum, 0.80) + 1),
        "n_pc_90": int(np.searchsorted(cum, 0.90) + 1),
        "n_pair_gt90": int((A[off] > 0.90).sum() / 2),
        "n_pair_gt70": int((A[off] > 0.70).sum() / 2),
        "n_pair_missing": n_missing,
        "max_offdiag": float(A[off].max()),
    }


def null_control(n_assets: int, n_obs: int, reps: int = 5, seed: int = 7) -> pd.DataFrame:
    """零假设：``n_assets`` 条**互相独立**的同分布序列，做同样的"对本行均值残差"。

    🚨 为什么必须有这个对照：对「自身参与构成的均值」做残差，会**机械地**把平均
    相关压到 ``−1/(n−1)``（31 条 ⇒ −0.033），并压低参与率。只看观测值会误以为
    "扣掉大盘后行业接近独立"。本函数给出「如果行业之间本来就毫无关系」时这些指标
    该是多少，观测值减去它才是真实结构的贡献。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        X = rng.standard_normal((n_obs, n_assets))
        E = X - X.mean(axis=1, keepdims=True)
        c = pd.DataFrame(np.corrcoef(E.T))
        rows.append(metrics_from_corr(c))
    return pd.DataFrame(rows)


def topk_span(ret: pd.DataFrame, rho_cut: float, mom: pd.DataFrame) -> pd.DataFrame:
    """逐月：用 trailing 250 日相关切簇，看动量 Top-K 跨了几个簇。

    对照组 = 从同一时点同一簇结构里**随机**抽 K 个行业（MC），给出"如果瞎选"
    的期望跨簇数。实测 < 随机期望才说明动量在往同一簇里扎堆。
    """
    month_ends = (ret.index.to_series()
                  .groupby([ret.index.year, ret.index.month]).last().tolist())
    rng = np.random.default_rng(20261109)
    rows = []
    for d in month_ends:
        if d not in ret.index:
            continue
        pos = ret.index.get_loc(d)
        if pos < 250:
            continue
        sub = ret.iloc[pos - 249:pos + 1]
        m = sub.mean(axis=1)
        c = pairwise_corr(resid(sub, m))
        lab = cluster_labels(c, rho_cut)
        mm = mom.loc[d].dropna()
        mm = mm[mm.index.isin(lab.index)]
        if len(mm) < TOP_K + 5:
            continue
        top = list(mm.sort_values(ascending=False).index[:TOP_K])
        vals = lab.reindex(mm.index).to_numpy()
        draws = np.array([np.unique(rng.choice(vals, size=TOP_K, replace=False)).size
                          for _ in range(MC_N)])
        rows.append({
            "date": d,
            "n_clusters": int(lab.nunique()),
            "topk_distinct": int(lab[top].nunique()),
            "rand_mean": float(draws.mean()),
            "rand_p05": float(np.quantile(draws, 0.05)),
            "rand_p95": float(np.quantile(draws, 0.95)),
            "top5": "|".join(top),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 主流程
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ret_d, ret_m = load_ret()

    d_px = pd.read_parquet(PQ / "sw_industry_daily.parquet")
    pxf = d_px.pivot_table(index="date", columns="code", values="close").sort_index()
    mom = (pxf.shift(21) / pxf.shift(261) - 1).loc[EVAL_START:EVAL_END]

    segs = {
        "全窗 2018-2026": (EVAL_START, EVAL_END),
        "前半 2018-2022": ("2018-01-02", "2022-12-31"),
        "后半 2023-2026": ("2023-01-02", EVAL_END),
    }
    rows = []
    for name, (a, b) in segs.items():
        for freq, r in (("日频", ret_d.loc[a:b]), ("月频", ret_m.loc[a:b])):
            m = r.mean(axis=1)
            for kind, x in (("raw", r), ("resid", resid(r, m))):
                rec = {"segment": name, "freq": freq, "kind": kind, "n_obs": int(len(r))}
                rec.update(metrics_from_corr(pairwise_corr(x)))
                rows.append(rec)
    main_df = pd.DataFrame(rows)
    main_df.to_csv(OUT_DIR / "neff_main.csv", index=False)

    # ---- 零假设对照（残差化本身的机械效应）----
    null_d = null_control(31, int((ret_d.index >= pd.Timestamp(EVAL_START)).sum()))
    null_m = null_control(31, 105, seed=8)
    null_d.insert(0, "freq", "日频")
    null_m.insert(0, "freq", "月频")
    null_df = pd.concat([null_d, null_m], ignore_index=True)
    null_df.to_csv(OUT_DIR / "neff_null.csv", index=False)

    show = ["segment", "freq", "kind", "n_obs", "mean_corr", "pc1_share",
            "participation", "eff_rank", "n_pc_90", "n_pair_gt70", "n_pair_gt90"]
    print("== 有效维数（31 个申万一级）==")
    print(main_df[show].round(3).to_string(index=False))

    print("\n== 零假设对照（31 条独立序列，同样对本行均值残差）==")
    for f, nd in (("日频", null_d), ("月频", null_m)):
        obs = main_df[(main_df["segment"] == "全窗 2018-2026")
                      & (main_df["freq"] == f) & (main_df["kind"] == "resid")].iloc[0]
        print(f"  [{f}] mean_corr 观测 {obs['mean_corr']:+.3f} / 零假设 "
              f"{nd['mean_corr'].mean():+.3f}   participation 观测 "
              f"{obs['participation']:.2f} / 零假设 {nd['participation'].mean():.2f}   "
              f"eff_rank 观测 {obs['eff_rank']:.2f} / 零假设 {nd['eff_rank'].mean():.2f}")

    # ---- 簇划分（残差主口径 + 原始参考）----
    m_full = ret_d.mean(axis=1)
    c_res = pairwise_corr(resid(ret_d, m_full))
    c_raw = pairwise_corr(ret_d)
    rows = []
    for kind, cc in (("resid", c_res), ("raw", c_raw)):
        for rc in RHO_CUTS:
            lab = cluster_labels(cc, rc)
            for cid, members in lab.groupby(lab).groups.items():
                rows.append({
                    "kind": kind, "rho_cut": rc, "n_clusters": int(lab.nunique()),
                    "cluster_id": int(cid), "size": len(members),
                    "members": ",".join(sorted(members)),
                })
    clus = pd.DataFrame(rows)
    clus.to_csv(OUT_DIR / "neff_clusters.csv", index=False)

    print("\n== 簇划分 ==")
    for kind in ("resid", "raw"):
        for rc in RHO_CUTS:
            sub = clus[(clus["kind"] == kind) & (clus["rho_cut"] == rc)]
            ncl = int(sub["n_clusters"].iloc[0])
            sizes = sub["size"].value_counts().sort_index(ascending=False)
            print(f"  [{kind}] ρ≥{rc:.2f} -> {ncl:2d} 簇；规模 {dict(sizes)}")
            big = sub[sub["size"] >= 3].sort_values("size", ascending=False).head(5)
            for _, r in big.iterrows():
                print(f"      size {r['size']:2d}: {r['members']}")

    # ---- 截面宽度随时间（"31 个"只在 2021-12 之后成立）----
    n_ind = ret_d.notna().sum(axis=1)
    print("\n== 截面行业数随时间 ==")
    print("  " + n_ind.value_counts().sort_index(ascending=False).to_string()
          .replace("\n", "\n  "))
    full = n_ind[n_ind == 31]
    if len(full):
        print(f"  首次达到 31 个：{full.index.min().date()}；"
              f"此前平均 {n_ind[n_ind.index < full.index.min()].mean():.1f} 个"
              f"（{int((n_ind.index < full.index.min()).sum())} 个交易日）")

    print("\n== 成对相关分布（全窗，465 对）==")
    for kind, cc in (("raw", c_raw), ("resid", c_res)):
        A = cc.to_numpy()
        n = len(cc)
        pairs = [(cc.index[i], cc.columns[j], A[i, j])
                 for i in range(n) for j in range(i + 1, n) if np.isfinite(A[i, j])]
        v = np.array([p[2] for p in pairs])
        print(f"  [{kind}] ρ>0.5 {int((v>0.5).sum()):3d}  >0.6 {int((v>0.6).sum()):3d}"
              f"  >0.7 {int((v>0.7).sum()):3d}  >0.8 {int((v>0.8).sum()):3d}"
              f"  >0.9 {int((v>0.9).sum()):2d}  max {v.max():+.3f}")
        top = sorted(pairs, key=lambda t: -t[2])[:5]
        print("      最强 5 对: " + "  ".join(
            f"{a[:6]}-{b[:6]} {r:+.2f}" for a, b, r in top))

    # ---- Top-K 跨度（残差口径）----
    span = topk_span(ret_d, SPAN_RHO, mom)
    span.to_csv(OUT_DIR / "neff_top5_span.csv", index=False)
    print(f"\n== 动量 Top{TOP_K} 的跨簇数（残差相关，ρ≥{SPAN_RHO:.2f}）==")
    print(f"  月数 {len(span)}  实测 {span['topk_distinct'].mean():.2f}  "
          f"随机期望 {span['rand_mean'].mean():.2f}  "
          f"(随机 P05/P95 {span['rand_p05'].mean():.2f}/{span['rand_p95'].mean():.2f})")
    vc = span["topk_distinct"].value_counts().sort_index()
    print("  分布 " + "  ".join(f"{k}簇:{v}" for k, v in vc.items()))
    print(f"  平均每簇行业数 {span['n_clusters'].mean():.2f} -> "
          f"理论最大跨簇 {span['n_clusters'].mean():.2f}")
    print(f"  扎堆月份占比 {(span['topk_distinct'] < span['rand_mean']).mean():.1%}")
    print("\n产物：neff_main.csv / neff_clusters.csv / neff_top5_span.csv")


if __name__ == "__main__":
    main()
