"""
ETF 轮动扩展面板（生产组件）
============================

`etf_rotation_probe.build_panels` 只给基础面板（mom12_1 等）；本模块补三块：

  1. **拥挤度**：份额 250 日滚动分位（PIT，按 ``ANN_DATE`` 生效）；
  2. **拥挤度软惩罚 / 反转双筛**：定版信号 ``mom12_1_crowdsoft_pre10`` 的两个分量；
  3. **折溢价**：``prem`` / ``prem20``，执行时点闸门用。

另有防御腿净值（债 / 金 / 混合）与 Δshare 面板，供长线档「磐石」与信号板调用。
所有面板均为 PIT：份额只知 ``ANN_DATE <= t``，折溢价当日可得。

被 `scripts/portfolio/etf_champ_signals.py`（生产信号板）与
`scripts/portfolio/etf_rotation_probe.py`（回测引擎）共用。
历史变体清单与实验主循环已于 2026-10-29 随实验记录一并回收，结论见
``reports/etf_contest5/README.md`` §六。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

CACHE_ROOT = Path(str(Config.cache()["root"]))

BOND, GOLD = "511010.SH", "518880.SH"

#: 长线档「磐石」的参数基底（其它字段由 ``etf_champ_signals.CHAMP`` 覆盖）。
CHAMP_BASE = {"signal": "mom12_1", "freq": "M", "top_k": 5, "absolute": True,
              "weighting": "inverse_vol", "target_vol": 0.15}


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
