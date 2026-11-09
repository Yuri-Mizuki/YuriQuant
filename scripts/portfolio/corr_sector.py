"""相关性行业归属面板（"长得像谁"）—— 构建 + 与申万标签的重叠率诊断
=====================================================================

动机（2026-11-09）
------------------
映翡量化 2026-11-09 转述的论文《Synthetic Sector Classification for
Statistical Relative Valuation》用「个股与 11 只行业 SPDR ETF 的历史相关性」
替代官方行业标签，再拿这个分组当回归里的行业哑变量。本模块只复刻它的**分组
算子**，用于两个探针：

1. ``industry_neff_probe``（P0-1）：申万标签与"走势同类"的分配重合率是多少；
2. ``neutralize_group_ab``（P1-1）：换掉 ``neutralize(industry_panel=...)``
   喂的面板，因子 IC 变不变。

与论文的口径差异（**都是我们比它严的地方**，不是偷懒）
------------------------------------------------------
= 论文 ============================== = 本模块 ==========================
原始 Pearson/Spearman 相关            **β 残差相关**（先把个股与行业的市场
                                      β 剥掉再相关）——理由见
                                     ``reports/etf_coverage/README.md`` §〇#5：
                                     宽基/市场因子让几乎所有标的原始 ρ>0.6，
                                     不剥市场会把分配结果虚高
不交代窗口与生效时点                  窗口结束于 d，**d+1 起生效**（PIT），
                                     避免论文自认的 look-ahead 风险
季度重新拟合                          季度重新拟合（同论文）
11 只 SPDR（官方分类的艺术品）         31 个申万一级指数（我们的行业全集）

⚠️ **能做什么 / 不能做什么**：它只回答「分组换掉后因子还剩多少信息」，
不回答「哪套分组更能赚钱」。论文自己证明过 R² 高与收益高可以反号
（GICS 的 R² 在 5/6 个模型上更高、累计收益却低 24pp），所以本模块的产物
**禁止**用 R² / 解释力排序。

用法
----
    python -m scripts.portfolio.corr_sector            # 构建 + 落盘 + 打重叠率
    python -m scripts.portfolio.corr_sector --check    # 只体检缓存，不重建
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "industry_rotation"
BASE = ROOT / "reports" / "alla_rolling" / "_base"
PQ = Path(Config.paths()["cache_root"])

#: 相关性窗口（交易日）——与 ``etf_coverage`` 的 250 日一致
WINDOW = 250
#: 窗口内最少有效交易日；不足则该股当期不指派（留 NaN，不做兜底）
MIN_VALID = 200
#: 面板起点：与近端窗口（2016-07 起）留出满窗暖机
PANEL_START = "2017-09-01"
#: 重叠率诊断的评价窗起点（与行业轮动主表一致）
EVAL_START = "2018-01-02"

PANEL_PATH = OUT_DIR / "corr_sector_panel.parquet"
LOG_PATH = OUT_DIR / "corr_sector_log.csv"
MIGRATE_PATH = OUT_DIR / "corr_sector_migration.csv"


# --------------------------------------------------------------------------- 数据
def load_returns() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """个股收益 / 31 行业指数收益 / 市场代理（= 31 行业等权）。

    口径与 ``etf_coverage`` 一致：市场代理取 31 个申万一级等权收益，
    不用宽基指数（宽基含风格与市值偏差，会把「大票属性」错记成「行业属性」）。
    """
    close = pd.read_parquet(BASE / "close_adj.parquet")
    ret_s = close.pct_change(fill_method=None)

    d = pd.read_parquet(PQ / "sw_industry_daily.parquet")
    px = d.pivot_table(index="date", columns="code", values="close").sort_index()
    ret_i = px.pct_change(fill_method=None)
    ret_i = ret_i.reindex(ret_s.index)

    mkt = ret_i.mean(axis=1)
    return ret_s, ret_i, mkt


# --------------------------------------------------------------------------- 核心
def _resid(Y: np.ndarray, ok: np.ndarray, m: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐列对市场收益 ``m`` 做单变量 OLS，返回 (残差, 残差平方和, 有效数)。

    Args:
        Y: (T, K) 收益矩阵，允许 NaN。
        ok: (T, K) 布尔掩码，False 处既不参与估计也不贡献残差。
        m: (T,) 市场收益。

    残差在 ``ok=False`` 的位置**置 0**，这样后续 ``Rxᵀ @ Ri`` 天然等价于
    「只在两只标的都有效的行上求和」，不用逐对构造掩码。
    """
    n = ok.sum(axis=0).astype(float)
    Yz = np.where(ok, Y, 0.0)
    Mz = np.where(ok, m[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        my = Yz.sum(axis=0) / n
        mm = Mz.sum(axis=0) / n
    Mc = Mz - mm
    Yc = np.where(ok, Y - my, 0.0)
    varm = (Mc ** 2).sum(axis=0)
    cov = (Yc * Mc).sum(axis=0)
    beta = np.divide(cov, varm, out=np.zeros_like(cov), where=varm > 0)
    R = np.where(ok, (Y - my) - beta * (m[:, None] - mm), 0.0)
    ss = (R ** 2).sum(axis=0)
    return R, ss, n


def assign_at(win_s: pd.DataFrame, win_i: pd.DataFrame, m: np.ndarray
              ) -> tuple[pd.Series, pd.Series]:
    """在单个窗口上做 β 残差相关 → argmax 指派。

    Returns:
        (label, max_corr)，index 均为股票代码；有效样本不足者不出现。
    """
    X = win_s.to_numpy(dtype=float)
    IND = win_i.to_numpy(dtype=float)
    # 🚨 市场代理自身可能缺日（申万指数与 A 股日历差 1 天，实测 250 日窗内有 0~3 天）。
    # 若只在 Y 上判有效，那些行会算出 NaN 残差、把整列平方和污染成 NaN，
    # 结果是「整期 0 只被指派」且**不报错**（2022-03/06/09/12 四个快照全中）。
    mok = np.isfinite(m)
    okx = np.isfinite(X) & mok[:, None]
    oki = np.isfinite(IND) & mok[:, None]
    Rx, ssx, nx = _resid(X, okx, m)
    Ri, ssi, _ = _resid(IND, oki, m)

    num = (Rx * okx).T @ Ri                      # (N, 31)
    ssi_on_j = okx.astype(float).T @ (Ri ** 2)   # (N, 31) 行业残差在个股有效行上的平方和
    den = np.sqrt(np.maximum(ssx, 0.0))[:, None] * np.sqrt(np.maximum(ssi_on_j, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    corr[nx < MIN_VALID, :] = np.nan

    codes = win_s.columns
    ok_row = np.isfinite(corr).any(axis=1)
    best = np.nanargmax(corr[ok_row], axis=1)
    label = pd.Series(np.asarray(win_i.columns)[best], index=codes[ok_row])
    mx = pd.Series(corr[ok_row, best], index=codes[ok_row])
    return label, mx


def quarter_ends(idx: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    """每个季度的最后一个交易日。"""
    s = pd.Series(idx, index=idx)
    g = s.groupby([idx.year, idx.quarter]).last()
    return [pd.Timestamp(x) for x in g.tolist() if pd.Timestamp(x) >= pd.Timestamp(start)]


def build(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_s, ret_i, mkt = load_returns()
    idx = ret_s.index
    snaps = quarter_ends(idx, PANEL_START)
    snaps = [d for d in snaps if idx.get_loc(d) >= WINDOW]

    rows: dict[pd.Timestamp, pd.Series] = {}
    logs = []
    for d in snaps:
        pos = idx.get_loc(d)
        lo = pos - WINDOW + 1
        win_s = ret_s.iloc[lo:pos + 1]
        win_i = ret_i.iloc[lo:pos + 1]
        m = mkt.iloc[lo:pos + 1].to_numpy(dtype=float)
        lab, mx = assign_at(win_s, win_i, m)
        if lab.empty:
            raise RuntimeError(
                f"{d.date()} 窗口内 0 只被指派——这不是「数据稀疏」，是算错了。"
                "先查 mkt/收益面板在该窗内是否全 NaN。")
        rows[d] = lab
        logs.append({
            "snapshot": d,
            "n_window_days": len(win_s),
            "n_assigned": int(lab.size),
            "n_codes_total": int(ret_s.shape[1]),
            "max_corr_median": float(mx.median()),
            "max_corr_p10": float(mx.quantile(0.10)),
            "n_industries_used": int(lab.nunique()),
        })
        if verbose:
            print(f"  {d.date()}  assigned {lab.size:5d}  "
                  f"med_max_rho {mx.median():.3f}  groups {lab.nunique():2d}")

    log = pd.DataFrame(logs)
    # PIT：窗口结束于 d，指派从 d 的下一个交易日起生效
    wide = pd.DataFrame(rows).T
    panel = wide.reindex(idx).ffill().shift(1)
    panel = panel.reindex(columns=ret_s.columns)
    return panel, log


# --------------------------------------------------------------------------- 诊断
def industry_names() -> dict[str, str]:
    """申万一级 code → 中文名（迁移表可读性；缺失时退化为 code 前缀）。"""
    p = PQ / "industry_classification_level1.parquet"
    if not p.exists():
        return {}
    d = pd.read_parquet(p, columns=["industry_code", "industry_name"]).drop_duplicates()
    return dict(zip(d["industry_code"], d["industry_name"], strict=False))


def overlap_report(panel: pd.DataFrame) -> pd.DataFrame:
    """与申万标签的重合率 + 迁移对（谁被挪到了哪儿）。"""
    sw = pd.read_parquet(BASE / "cov_industry.parquet")
    common = panel.index.intersection(sw.index)
    cc = panel.loc[common]
    sw = sw.reindex(index=common, columns=panel.columns)

    both = cc.notna() & sw.notna()
    rec = {
        "n_cell": int(both.to_numpy().sum()),
        "overlap_rate": float((cc.where(both) == sw.where(both)).to_numpy().sum()
                              / max(both.to_numpy().sum(), 1)),
        "n_days": int(len(common)),
    }

    pairs = pd.DataFrame({
        "sw": sw.where(both).to_numpy().ravel(),
        "cc": cc.where(both).to_numpy().ravel(),
    }).dropna()
    pairs = pairs[pairs["sw"] != pairs["cc"]]
    mig = (pairs.groupby(["sw", "cc"]).size()
           .sort_values(ascending=False).rename("n").reset_index().head(40))
    mig["share_of_moved"] = mig["n"] / max(len(pairs), 1)
    nm = industry_names()
    mig["sw_name"] = mig["sw"].map(nm).fillna(mig["sw"])
    mig["cc_name"] = mig["cc"].map(nm).fillna(mig["cc"])
    return pd.DataFrame([rec]), mig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只体检缓存，不重建")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not PANEL_PATH.exists():
            raise SystemExit(f"缺少 {PANEL_PATH}，先跑一次不带 --check 的")
        panel = pd.read_parquet(PANEL_PATH)
        log = pd.read_csv(LOG_PATH)
        n_days = int(panel.notna().any(axis=1).sum())
        print(f"面板 {panel.shape}  有效期数 {n_days}  "
              f"快照 {len(log)}  起点 {panel.index.min().date()}  "
              f"终点 {panel.index.max().date()}")
        print(f"PIT 断言：首快照 {log['snapshot'].iloc[0]} 之前必须全空 -> "
              f"{panel.loc[:log['snapshot'].iloc[0]].notna().to_numpy().sum() == 0}")
        rec, mig = overlap_report(panel)
        print(rec.to_string(index=False))
        print(mig.head(10).to_string(index=False))
        return

    print("构建相关性行业归属面板 …")
    panel, log = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH)

    rec, mig = overlap_report(panel)
    log.to_csv(LOG_PATH, index=False)
    mig.to_csv(MIGRATE_PATH, index=False)

    print("\n== 快照日志 ==")
    print(log.to_string(index=False))
    print("\n== 与申万标签的重合率 ==")
    print(rec.to_string(index=False))
    print("\n== 迁移对（Top 10，被挪走的股票去了哪）==")
    print(mig.head(10).to_string(index=False))
    print(f"\n产物：{PANEL_PATH.name} / {LOG_PATH.name} / {MIGRATE_PATH.name}")


if __name__ == "__main__":
    main()
