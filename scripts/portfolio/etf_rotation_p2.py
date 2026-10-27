"""
ETF 轮动优化验证（P2 前置，2026-10-26）
======================================

在 P1 基线 ``dual_ivvt15_m5`` 之上验证三项「低成本、可借自业界」的增量：

  1. **防御腿**：风险关闭时不再留 0 收益现金，而是退到国债 / 黄金 ETF；
  2. **趋势一致性**：以 60 日对数价格回归 R² 缩放 12-1 动量（借自聚宽「斜率×R²」）；
  3. **拥挤度**：份额历史分位（PIT，按 ``ANN_DATE`` 生效）过滤高拥挤标的
     （借自银河金工动量择势 / 国盛景气-趋势-拥挤）。

判据：相对 P1 基线 ΔSharpe > 0.1、MDD 不恶化、成本后为正（方案 §四）。

用法:
    python -m scripts.portfolio.etf_rotation_p2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config
from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_rotation_probe import (
    BENCH,
    MIN_AMOUNT,
    SEGMENTS,
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_rotation_p2"
CACHE_ROOT = Path(str(Config.cache()["root"]))

BOND, GOLD = "511010.SH", "518880.SH"
P1_BASE = {"signal": "mom12_1", "freq": "M", "top_k": 5, "absolute": True,
           "weighting": "inverse_vol", "target_vol": 0.15}

VARIANTS: list[dict] = [
    {**P1_BASE, "name": "p1_base"},
    {**P1_BASE, "name": "def_bond", "defense": "bond"},
    {**P1_BASE, "name": "def_gold", "defense": "gold"},
    {**P1_BASE, "name": "def_mix", "defense": "mix"},
    {**P1_BASE, "name": "mom_r2", "signal": "mom_r2"},
    {**P1_BASE, "name": "crowd_f20", "signal": "mom_crowd_f20"},
    {**P1_BASE, "name": "crowd_f10", "signal": "mom_crowd_f10"},
    {**P1_BASE, "name": "def_mix_crowd_f20", "signal": "mom_crowd_f20", "defense": "mix"},
]

# 调仓频率对照：月频（基准）/ 每 10 个交易日 / 周频。
FREQ: list[dict] = [
    {**P1_BASE, "name": "champ_m", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "champ_2w", "signal": "mom_crowd_f20", "defense": "mix", "every_n": 10},
    {**P1_BASE, "name": "champ_w", "signal": "mom_crowd_f20", "defense": "mix", "freq": "W"},
]

# 成本对照：现行假设（佣金 5bp）vs 用户实际费率（万分之一 = 1bp）× 滑点档位。
COST: list[dict] = [
    {**P1_BASE, "name": "cost_c5_s5", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "cost_c1_s5", "signal": "mom_crowd_f20", "defense": "mix",
     "commission_bp": 1.0},
    {**P1_BASE, "name": "cost_c1_s2", "signal": "mom_crowd_f20", "defense": "mix",
     "commission_bp": 1.0, "slippage_bp": 2.0},
    {**P1_BASE, "name": "cost_c1_s10", "signal": "mom_crowd_f20", "defense": "mix",
     "commission_bp": 1.0, "slippage_bp": 10.0},
]

# S3：折溢价（price / IOPV - 1）。IOPV 覆盖 49/50（511090.SH 无）。
PREMIUM: list[dict] = [
    {**P1_BASE, "name": "p1_base"},
    {**P1_BASE, "name": "mom_prem_f80", "signal": "mom_prem_f80"},
    {**P1_BASE, "name": "mom_prem_f90", "signal": "mom_prem_f90"},
    {**P1_BASE, "name": "mom_prem_f67", "signal": "mom_prem_f67"},
    {**P1_BASE, "name": "champ_mom_prem", "signal": "mom_prem_f80_crowd", "defense": "mix"},
]

# P3 组合层：风险预算（ERC / HRP）与权重上限。控制项 p1_base 的 inverse-vol
# 是「对角协方差」的近似，ERC/HRP 引入完整协方差（含相关结构）。
P3: list[dict] = [
    {**P1_BASE, "name": "p1_base"},
    {**P1_BASE, "name": "erc", "weighting": "erc"},
    {**P1_BASE, "name": "hrp", "weighting": "hrp"},
    {**P1_BASE, "name": "maxw25", "max_weight": 0.25},
    {**P1_BASE, "name": "maxw20", "max_weight": 0.20},
    {**P1_BASE, "name": "champ", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "champ_erc", "signal": "mom_crowd_f20", "defense": "mix",
     "weighting": "erc"},
    {**P1_BASE, "name": "champ_hrp", "signal": "mom_crowd_f20", "defense": "mix",
     "weighting": "hrp"},
    {**P1_BASE, "name": "champ_maxw25", "signal": "mom_crowd_f20", "defense": "mix",
     "max_weight": 0.25},
]

# 持仓只数分布（单变体诊断）。
HOLD: list[dict] = [
    {**P1_BASE, "name": "champ_hold", "signal": "mom_crowd_f20", "defense": "mix"},
]

# S1 残余：Δshare(5/20/60d) 的资金流动量 vs 申赎反转两说。
# 纯 Δshare 排序**不用绝对动量过滤**（否则会变成"要求净申购为正"，把两说混在一起）。
SHARE: list[dict] = [
    {**P1_BASE, "name": "p1_base"},
    {**P1_BASE, "name": "share_d20", "signal": "dshare20", "absolute": False},
    {**P1_BASE, "name": "share_d20_rev", "signal": "dshare20_rev", "absolute": False},
    {**P1_BASE, "name": "share_d5", "signal": "dshare5", "absolute": False},
    {**P1_BASE, "name": "share_d60", "signal": "dshare60", "absolute": False},
    # 叠加用法：动量选标 + 剔除「申购最热」的 Δshare 截面分位（反转方向）
    {**P1_BASE, "name": "mom_dsh_f80", "signal": "mom_dsh_f80"},
    {**P1_BASE, "name": "mom_dsh_f67", "signal": "mom_dsh_f67"},
    {**P1_BASE, "name": "mom_dsh_f90", "signal": "mom_dsh_f90"},
    {**P1_BASE, "name": "champ_mom_dsh", "signal": "mom_dsh_f80_crowd", "defense": "mix"},
    {**P1_BASE, "name": "champ_mom_dsh_f67", "signal": "mom_dsh_f67_crowd", "defense": "mix"},
    {**P1_BASE, "name": "champ_mom_dsh_f90", "signal": "mom_dsh_f90_crowd", "defense": "mix"},
]

# 最优组合项（def_mix_crowd_f20）的邻域网格：防御腿类型 / 成本 / 目标波动 / 集中度。
ROBUST: list[dict] = [
    {**P1_BASE, "name": "dm_mix_c20", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "dm_bond_c20", "signal": "mom_crowd_f20", "defense": "bond"},
    {**P1_BASE, "name": "dm_mix_c20_cost2x", "signal": "mom_crowd_f20", "defense": "mix",
     "cost_x": 2.0},
    {**P1_BASE, "name": "dm_mix_c20_vt12", "signal": "mom_crowd_f20", "defense": "mix",
     "target_vol": 0.12},
    {**P1_BASE, "name": "dm_mix_c20_vt18", "signal": "mom_crowd_f20", "defense": "mix",
     "target_vol": 0.18},
    {**P1_BASE, "name": "dm_mix_c20_m3", "signal": "mom_crowd_f20", "defense": "mix",
     "top_k": 3},
    {**P1_BASE, "name": "dm_mix_c20_m8", "signal": "mom_crowd_f20", "defense": "mix",
     "top_k": 8},
]

# 长线定版（champ + 动态去重 0.90/120）的剩余优化点 A1–A3、B4–B5（10-27）。
# 成本统一为实际费率 1bp + 5bp，与已发布的 champ 13.75% / 1.08 / −13.97% 对齐。
CHAMP = {**P1_BASE, "signal": "mom_crowd_f20", "defense": "mix",
         "commission_bp": 1.0}
OPT: list[dict] = [
    {**CHAMP, "name": "champ_dd90", "dedup_corr": 0.90},
    {**CHAMP, "name": "champ_nodd"},
    # A1 去重阈值 0.85/0.90/0.95 × 相关窗口 60/120/250（看邻域是否成片，不看单点）
    {**CHAMP, "name": "dd85_w120", "dedup_corr": 0.85},
    {**CHAMP, "name": "dd95_w120", "dedup_corr": 0.95},
    {**CHAMP, "name": "dd90_w60", "dedup_corr": 0.90, "dedup_window": 60},
    {**CHAMP, "name": "dd90_w250", "dedup_corr": 0.90, "dedup_window": 250},
    {**CHAMP, "name": "dd85_w60", "dedup_corr": 0.85, "dedup_window": 60},
    {**CHAMP, "name": "dd85_w250", "dedup_corr": 0.85, "dedup_window": 250},
    {**CHAMP, "name": "dd95_w60", "dedup_corr": 0.95, "dedup_window": 60},
    {**CHAMP, "name": "dd95_w250", "dedup_corr": 0.95, "dedup_window": 250},
    # A2 Δshare 过滤与去重是否重复（去重可能已吃掉「拥挤/同质」的信息）
    {**CHAMP, "name": "dd90_dsh67", "signal": "mom_dsh_f67_crowd", "dedup_corr": 0.90},
    {**CHAMP, "name": "dd90_dsh80", "signal": "mom_dsh_f80_crowd", "dedup_corr": 0.90},
    {**CHAMP, "name": "dd90_dsh90", "signal": "mom_dsh_f90_crowd", "dedup_corr": 0.90},
    # A3 缓冲区在去重之后复测（去重缩小候选池，滞后成本可能变便宜）
    {**CHAMP, "name": "dd90_buf1540", "dedup_corr": 0.90, "enter_rank": 15, "exit_rank": 40},
    {**CHAMP, "name": "dd90_buf1030", "dedup_corr": 0.90, "enter_rank": 10, "exit_rank": 30},
    # B4 权重：HRP（60 日协方差）与 30% 单只上限
    {**CHAMP, "name": "dd90_hrp", "dedup_corr": 0.90, "weighting": "hrp"},
    {**CHAMP, "name": "dd90_hrp_mw30", "dedup_corr": 0.90, "weighting": "hrp",
     "max_weight": 0.30},
    # B5 防御腿内部配比（金 70% / 债 70%）与绝对动量阈值（−5% / +5%）
    {**CHAMP, "name": "dd90_def_b30g70", "dedup_corr": 0.90, "defense": "mix30"},
    {**CHAMP, "name": "dd90_def_b70g30", "dedup_corr": 0.90, "defense": "mix70"},
    {**CHAMP, "name": "dd90_abs_m5", "dedup_corr": 0.90, "abs_threshold": -0.05},
    {**CHAMP, "name": "dd90_abs_p5", "dedup_corr": 0.90, "abs_threshold": 0.05},
]


def level_from_returns(ret: pd.Series) -> pd.Series:
    """把日收益串成净值水平线（末尾乘以 100，量纲无关）。"""
    return (1.0 + ret.fillna(0.0)).cumprod() * 100.0


def build_defense(close: pd.DataFrame) -> dict[str, pd.Series]:
    """防御腿净值水平线：债 / 金 / 债金等权 / 偏股性(金 70%) / 偏防守(债 70%)。"""
    ret = close.pct_change(fill_method=None)
    bond, gold = ret[BOND], ret[GOLD]
    return {"bond": bond, "gold": gold, "mix": 0.5 * bond + 0.5 * gold,
            "mix30": 0.3 * bond + 0.7 * gold, "mix70": 0.7 * bond + 0.3 * gold}


def build_share_pit(close: pd.DataFrame) -> pd.DataFrame:
    """份额的 PIT 日频面板：t 日只知 ``ANN_DATE <= t`` 的最新一期。

    份额表字段以「万份」计，量纲不影响分位排序。缺失公告日或份额的行丢弃；
    同一公告日多条取最后一条。
    """
    fs = pd.read_parquet(CACHE_ROOT / "fund_share.parquet").reset_index()
    fs["ann_date"] = pd.to_datetime(fs["ann_date"], errors="coerce")
    fs["value"] = fs["total_share"].fillna(fs["fund_share"])
    fs = fs.dropna(subset=["ann_date", "value"])

    cols = {}
    for code, g in fs.groupby("code"):
        if code not in close.columns:
            continue
        g = g.sort_values("ann_date").drop_duplicates("ann_date", keep="last")
        s = pd.Series(g["value"].to_numpy(), index=pd.DatetimeIndex(g["ann_date"]))
        cols[code] = s.reindex(close.index, method="ffill")
    pit = pd.DataFrame(cols).reindex(columns=close.columns)
    return pit


def build_crowding(close: pd.DataFrame) -> pd.DataFrame:
    """份额的 250 日滚动分位（0=历史最低，1=历史最高）。"""
    pit = build_share_pit(close)
    return pit.rolling(250, min_periods=120).rank(pct=True)


def build_r2(close: pd.DataFrame, win: int = 60) -> pd.DataFrame:
    """对数价格对时间的滚动回归 R²（趋势一致性）。"""
    logp = np.log(close.where(close > 0))
    t = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    return logp.rolling(win, min_periods=45).corr(t).pow(2)


def build_share_signals(close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Δshare(5/20/60d) 面板：份额的 PIT 日频序列做 n 个交易日变化率。

    份额按日公告（实测 510300 的 ``ann_date`` 间隔中位数 1 天），因此交易日窗口
    有意义；``pit`` 已按 ``ANN_DATE <= t`` 前向填充，无前视。
    """
    pit = build_share_pit(close)
    out: dict[str, pd.DataFrame] = {}
    for n in (5, 20, 60):
        chg = pit / pit.shift(n) - 1.0
        out[f"dshare{n}"] = chg
        out[f"dshare{n}_rev"] = -chg
    return out


def month_first_dates(idx: pd.DatetimeIndex, begin: int, end: int) -> pd.DatetimeIndex:
    """[begin, end] 内每月第一个交易日（与引擎 ``freq="M"`` 的信号日对齐）。"""
    sel = idx[(idx >= pd.Timestamp(str(begin))) & (idx <= pd.Timestamp(str(end)))]
    if len(sel) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(sel, index=sel)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).first().to_numpy())


def _spearman(a: pd.Series, b: pd.Series, min_n: int = 8) -> float:
    ok = a.notna() & b.notna()
    if int(ok.sum()) < min_n:
        return float("nan")
    x, y = a[ok], b[ok]
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(x.corr(y, method="spearman"))


def share_ic_table(close: pd.DataFrame, signals: dict[str, pd.DataFrame],
                   begin: int, end: int, placebo: int = 20,
                   seed: int = 20261026) -> pd.DataFrame:
    """月频 rank IC：t 日因子 vs 下一调仓日收益；附整条序列打乱的安慰剂。

    判据（方案 §四）：|IC| ≥ 0.03、t ≥ 2、安慰剂 0/20 超越。
    安慰剂按**整条序列**重置（每期都在截面内独立打乱，重复 ``placebo`` 次），
    比较的是「mean IC」而不是单期 |IC| —— 单期 |IC| 在该截面上天然有
    1/sqrt(N)≈0.14 的噪声（N≈50），用它做门槛会把真实信号误判。
    """
    rb = month_first_dates(close.index, begin, end)
    pairs = [(t, tn) for t, tn in zip(rb[:-1], rb[1:])]
    if len(pairs) < 10:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)

    def _mean_ic(sig: pd.DataFrame, shuffle: bool) -> float:
        ics = []
        for t, tn in pairs:
            f = sig.loc[t].reindex(close.columns)
            r = (close.loc[tn] / close.loc[t] - 1.0).reindex(close.columns)
            ok = f.notna() & r.notna()
            if int(ok.sum()) < 8:
                continue
            fv = f[ok].to_numpy()
            if shuffle:
                fv = rng.permutation(fv)
            ic = _spearman(pd.Series(fv, index=r[ok].index), r[ok])
            if not np.isnan(ic):
                ics.append(ic)
        return float(np.mean(ics)) if ics else float("nan")

    rows = []
    for name, sig in signals.items():
        ics = []
        for t, tn in pairs:
            f = sig.loc[t].reindex(close.columns)
            r = (close.loc[tn] / close.loc[t] - 1.0).reindex(close.columns)
            ic = _spearman(f, r)
            if not np.isnan(ic):
                ics.append(ic)
        s = pd.Series(ics, dtype=float)
        n = int(s.notna().sum())
        mean = float(s.mean()) if n else float("nan")
        std = float(s.std(ddof=1)) if n > 1 else float("nan")
        placebo_means = [abs(_mean_ic(sig, shuffle=True)) for _ in range(placebo)]
        placebo_means = [x for x in placebo_means if not np.isnan(x)]
        rows.append({
            "signal": name, "n_periods": n, "mean_ic": mean,
            "icir": (mean / std) if std and std > 0 else float("nan"),
            "t_stat": (mean / (std / np.sqrt(n))) if std and std > 0 and n > 1 else float("nan"),
            "pos_rate": float((s > 0).mean()) if n else float("nan"),
            "placebo_max_abs_mean_ic": max(placebo_means) if placebo_means else float("nan"),
            "n_placebo": len(placebo_means),
        })
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False)


def _rebalance_days(close: pd.DataFrame, spec: dict,
                    w0: int | None, w1: int | None) -> set | None:
    """``every_n`` 型调仓日：窗口内每 n 个交易日调一次；其余变体返回 None（用引擎频率）。"""
    n = spec.get("every_n")
    if not n:
        return None
    lo = pd.Timestamp(str(w0)) if w0 is not None else close.index.min()
    hi = pd.Timestamp(str(w1)) if w1 is not None else close.index.max()
    idx = close.index[(close.index >= lo) & (close.index <= hi)]
    return set(idx[:: int(n)])


def build_premium(raw_close: pd.DataFrame, columns: pd.Index) -> dict[str, pd.DataFrame]:
    """折溢价面板：原始收盘价 / IOPV - 1，及 20 日均值。

    只用一级市场 IOPV 与二级市场收盘价，天然 PIT（都是当日数据）。
    ``|prem| > 50%`` 视为数据错误置 NaN（IOPV 缺失/错误时会出现）。
    """
    d = pd.read_parquet(CACHE_ROOT / "fund_iopv.parquet")
    iopv = d["iopv_nav"].unstack("code").reindex(
        index=raw_close.index, columns=columns)
    prem = raw_close.reindex(columns=columns) / iopv - 1.0
    prem = prem.where(prem.abs() <= 0.5)
    return {"prem": prem, "prem20": prem.rolling(20, min_periods=10).mean()}


def build_extra_panels(close: pd.DataFrame, panels: dict,
                       raw_close: pd.DataFrame | None = None) -> dict:
    """在 P1 面板上追加 P2 三个信号面板 + S1 Δshare + S3 折溢价。"""
    r2 = build_r2(close)
    crowd = build_crowding(close)
    mom = panels["mom12_1"]
    panels = dict(panels)
    panels["crowd"] = crowd
    panels["mom_r2"] = mom * r2
    panels["mom_crowd_f20"] = mom.where(crowd <= 0.80)
    panels["mom_crowd_f10"] = mom.where(crowd <= 0.90)
    # 软惩罚（2026-10-27）：拥挤度超出 0.80 的部分线性扣最多 10pp 动量，
    # 不硬剔除，避免调仓日候选被砍到不足只数。缺份额数据的标的按 0 惩罚。
    over = ((crowd - 0.80) / 0.20).clip(lower=0.0, upper=1.0).fillna(0.0)
    panels["mom12_1_crowdsoft"] = mom - 0.10 * over
    # 反转双筛（2026-10-27 定版 v2）：剔除最近 20 日涨幅 >10% 的"过热/尖峰动量"
    # 标的。只用短期信息做减法（不拿它选标），专治急涨冲榜后均值回归的形态。
    mom20 = close / close.shift(20) - 1.0
    panels["mom20"] = mom20
    panels["mom12_1_crowdsoft_pre10"] = panels["mom12_1_crowdsoft"].where(mom20 <= 0.10)
    panels.update(build_share_signals(close))
    # Δshare 的反转用法：剔除「申购最热」的截面分位（资金流入 → 后续收益低）。
    dsh_rank = panels["dshare20"].rank(axis=1, pct=True)
    panels["mom_dsh_f80"] = mom.where(dsh_rank <= 0.80)
    panels["mom_dsh_f67"] = mom.where(dsh_rank <= 2.0 / 3.0)
    panels["mom_dsh_f90"] = mom.where(dsh_rank <= 0.90)
    panels["mom_dsh_f80_crowd"] = panels["mom_dsh_f80"].where(crowd <= 0.80)
    panels["mom_dsh_f67_crowd"] = panels["mom_dsh_f67"].where(crowd <= 0.80)
    panels["mom_dsh_f90_crowd"] = panels["mom_dsh_f90"].where(crowd <= 0.80)
    if raw_close is not None:
        prem = build_premium(raw_close, close.columns)
        panels.update(prem)
        pr = prem["prem20"].rank(axis=1, pct=True)
        panels["mom_prem_f80"] = mom.where(pr <= 0.80)
        panels["mom_prem_f67"] = mom.where(pr <= 2.0 / 3.0)
        panels["mom_prem_f90"] = mom.where(pr <= 0.90)
        panels["mom_prem_f80_crowd"] = panels["mom_prem_f80"].where(crowd <= 0.80)
    return panels


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="ETF 轮动 P2 前置优化验证")
    ap.add_argument("--begin", type=int, default=20200102)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variants", choices=("main", "robust", "freq", "cost", "hold",
                                           "share", "premium", "p3", "opt"),
                    default="main")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, args.end)
    close = bundle[1]
    raw_close = (cache.read_daily("etf")["close"].unstack("code")
                 .reindex(index=close.index, columns=close.columns))
    panels = build_extra_panels(close, build_panels(close, bundle[2]), raw_close)
    def_levels = {k: level_from_returns(v) for k, v in build_defense(close).items()}
    print(f"份额 PIT 面板覆盖 {panels['mom_crowd_f20'].notna().sum().sum():,} 个有效格", flush=True)

    rows, seg_rows, eq = [], [], {}
    variants = {"main": VARIANTS, "robust": ROBUST, "freq": FREQ,
                "cost": COST, "hold": HOLD, "share": SHARE,
                "premium": PREMIUM, "p3": P3, "opt": OPT}[args.variants]
    tag = {"main": "", "robust": "_robust", "freq": "_freq",
           "cost": "_cost", "hold": "_hold", "share": "_share",
           "premium": "_premium", "p3": "_p3", "opt": "_opt"}[args.variants]

    if args.variants == "share":
        ic_tab = share_ic_table(
            close, {k: v for k, v in panels.items() if k.startswith("dshare")},
            args.begin, args.end if args.end else int(close.index.max().strftime("%Y%m%d")))
        ic_tab.to_csv(out / "share_ic.csv", index=False, encoding="utf-8-sig")
        print("\n[S1 Δshare 月频 rank IC]\n" +
              ic_tab.to_string(index=False, float_format=lambda x: f"{x:+.3f}"), flush=True)

    if args.variants == "premium":
        ic_tab = share_ic_table(
            close, {k: panels[k] for k in ("prem", "prem20")},
            args.begin, args.end if args.end else int(close.index.max().strftime("%Y%m%d")))
        ic_tab.to_csv(out / "premium_ic.csv", index=False, encoding="utf-8-sig")
        print("\n[S3 折溢价月频 rank IC]\n" +
              ic_tab.to_string(index=False, float_format=lambda x: f"{x:+.3f}"), flush=True)

    for spec in variants:
        extra = None
        if spec.get("defense"):
            extra = pd.DataFrame({"DEF": def_levels[spec["defense"]]})
        full = run_variant(bundle, panels, spec, args.begin, args.end, extra_close=extra,
                           rebalance_days=_rebalance_days(close, spec, args.begin, args.end))
        segs = []
        for y0, y1 in SEGMENTS:
            lo, hi = max(args.begin, y0 * 10000 + 101), y1 * 10000 + 1231
            r = run_variant(bundle, panels, spec, lo, hi, extra_close=extra,
                            rebalance_days=_rebalance_days(close, spec, lo, hi))
            segs.append(r["annual_return"])
        n_pos = int(sum(s > 0 for s in segs))
        rows.append({**spec, **{k: v for k, v in full.items()
                                 if k not in ("equity", "hold_counts", "weights",
                                              "rebalance_dates")},
                     "seg_pos": n_pos,
                     **{f"seg_{i + 1}": s for i, s in enumerate(segs)}})
        seg_rows.append({"name": spec["name"], **{f"seg_{i + 1}": s for i, s in enumerate(segs)}})
        eq[spec["name"]] = full["equity"]
        print(f"{spec['name']:<22} ann={full['annual_return']:+.2%} "
              f"sh={full['sharpe']:.2f} mdd={full['max_drawdown']:.2%} "
              f"to={full['avg_turnover']:.2%} hold={full['avg_holdings']:.1f} "
              f"cash={full['avg_cash']:.1%} cost={full['cost_drag']:.2%}/yr "
              f"seg={n_pos}/3", flush=True)
        if args.variants == "hold":
            cnt = full["hold_counts"]
            dist = cnt.value_counts(normalize=True).sort_index()
            print("  持仓只数分布: " + " / ".join(
                f"{int(k)}只 {v:.1%}" for k, v in dist.items()) +
                f"；平均 {cnt.mean():.2f} 只、最少 {int(cnt.min())} 只", flush=True)

    summary = pd.DataFrame(rows)
    if (summary["name"] == "p1_base").any():
        base = summary.loc[summary["name"] == "p1_base"].iloc[0]
        summary["d_sharpe"] = summary["sharpe"] - base["sharpe"]
        summary["d_mdd"] = summary["max_drawdown"] - base["max_drawdown"]
    summary = summary.sort_values("sharpe", ascending=False)
    summary.to_csv(out / f"summary{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seg_rows).to_csv(out / f"segments{tag}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eq).to_csv(out / f"equity{tag}.csv", encoding="utf-8-sig")
    (out / f"run_config{tag}.json").write_text(json.dumps({
        "begin": args.begin, "end": args.end, "bench": BENCH,
        "pool_size": len(close.columns), "min_amount": MIN_AMOUNT,
        "segments": SEGMENTS,
        "judge": "dSharpe>0.1 vs p1_base, MDD not worse, cost-positive",
        "tag": tag or "main", "variants": variants,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
