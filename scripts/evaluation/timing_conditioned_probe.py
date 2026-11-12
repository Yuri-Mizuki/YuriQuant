"""条件化月频择时（P0）：复现并本地化「newsy 月」结构。

动机（2026-11-11 由莉酱「跑 P0」）
----------------------------------
外部文献检索（``reports/timing_lit_review/README.md`` §二）发现本项目从未测过的一格：

    **Guo, H. (2025), "Earnings Extrapolation and Predictable Stock Market Returns",
    RFS 38(6) 1730-1782.**

美股的「月度自相关≈0」不是没有信号，而是**两个方向相反的条件相关平均掉了**：
季度首月（1/4/7/10，财报季，"newsy month"）收益**负向**预测未来 newsy 月、
**正向**预测未来非 newsy 月。用实时信息构造，月度 **OOS R² 3.6%~4.3%**，
比 Goyal-Welch (2008) 那批高**一个数量级**。

这恰好对上本项目 ``timing_daily_train`` 的诊断：``daily_ic`` +0.0291 但聚合到
月度 ``agg_ic`` **−0.0797 转负**——「把两个反向信号平均掉」的典型特征。

A 股日历与美国不同，**必须重新定 newsy 月**
-------------------------------------------
本脚本实测 ``income.parquet``（statement_type=1，2016 起）的定期报告披露月：

    一季报(report_period=3/31) → **96.3% 在 4 月**
    半年报(report_period=6/30) → **91.8% 在 8 月**
    三季报(report_period=9/30) → **96.1% 在 10 月**
    年报(report_period=12/31)  → 52.7% 在 4 月、38.4% 在 3 月

⇒ 主定义 ``NEWSY = (4, 8, 10)``。备选 ``NEWSY_ALT = (3, 4, 8, 10)``（并入年报季）
作为**预注册的第二口径**同时报出，避免「挑一个好看的」。

⚠️ **``next_newsy`` 不是前视**：它条件于**因变量的日历类型**，而日历是确定的、
事前完全可预知。这与「用未来收益做特征」是两件事。判据「取未来必头满尾空」
针对的是**收益**标签，不针对日历标签——此处不适用，但必须在报告中显式声明。

臂（跑前定，全部报出，不做选择性汇报）
---------------------------------------
* ``M0_mean``          扩展窗历史均值（= Goyal-Welch 的基准，无预测）
* ``M1_ar12``          过去 12 个月累计收益（传统自相关 = 本项目现状）
* ``M2_cond12``        M1 + 「未来月类型」交互（Guo 第一步：条件于因变量时序）
* ``M3_newsy_sum``     过去 12 个月中**newsy 月**的收益和 + 交互（Guo 完整版）
* ``M4_placebo``       M3 但 newsy 月份标签**随机重排**（安慰剂：Δ≈0 才算同源）

判据（跑前定，任一不过即判负）
------------------------------
1. **月频 OOS R² ≥ 盈亏平衡门槛**（``breakeven_r2``：由成本/波动/广度反解）。
2. **逐相位配对 ≥4/6 为正**（21 相位取 6）。
3. **必须带恒定仓位对照**（``G_const_expo``，仓位 = 各臂实际平均仓位）：
   剔除「只是拿得少」这一项，这是本项目踩过三次的坑。
4. **安慰剂 M4 不得优于 M3**（否则是拟合噪声而非机制）。

用法
----
    python -m scripts.evaluation.timing_conditioned_probe
    python -m scripts.evaluation.timing_conditioned_probe --skip-arms   # 只诊断

产物
----
    reports/timing_conditioned/{oos,arms,pairs,yearly,month_profile}.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "timing_conditioned"
BASE = ROOT / "reports" / "alla_rolling" / "_base"
EQUITY = (ROOT / "reports" / "alla_rolling_ortho" / "equity"
          / "eq__ens_h1h5__h1__M__raw__f0.10.csv")
DATA = Path(Config.paths()["cache_root"])

#: newsy 月主定义：定期报告密集披露月（本脚本 ``month_profile()`` 实测）。
NEWSY = (4, 8, 10)
#: 备选定义：并入年报季（3 月）。**预注册第二口径**，与主定义一起报。
NEWSY_ALT = (3, 4, 8, 10)

#: 扩展窗最少训练月数。36 个月 ⇒ 2016-08 起的主样本可在 2019-08 出样本外，
#: 覆盖策略期（2019-01 起）。60 个月作敏感性对照。
MIN_TRAIN = 36
MIN_TRAIN_SENS = 60

#: 成本与换手锚点（与 ``timing_daily_train_probe`` 同源，来自本项目实测）。
COST_BP = 36.0
TURNOVER_MONTHLY = 11.6

#: 仓位切换成本（单边 bp）：ω 变动 1.0 单位 = 满仓↔空仓。
#: 与 ``timing_probe`` / ``timing_daily_train_probe`` 同口径（不额外引入假设）。
POSITION_TURNOVER_BP = 10.0

#: 相位：月频调仓日在月内的第 n 个交易日（定版 = 0）。
PHASES = tuple(range(6))

ARMS = {
    "M0_mean": "扩展窗历史均值（无预测）",
    "M1_ar12": "过去12月累计收益",
    "M2_cond12": "M1 + 未来月类型交互",
    "M3_newsy_sum": "过去12月 newsy 月收益和 + 交互",
    "M4_placebo": "M3 + newsy 标签随机重排（安慰剂）",
}


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_ew_ret() -> pd.Series:
    """等权全A 日频收益（``close_raw`` 列向均值，与策略同源）。"""
    close = pd.read_parquet(BASE / "close_raw.parquet")
    # 🚨 守卫：close_raw 有 2 格为 0.0（301390.SZ / 603159.SH @ 2026-09-16），
    # 不清洗会在 pct_change 出 inf 并污染列向聚合。
    close = close.where(close > 0)
    r1 = close.pct_change(fill_method=None)
    return (r1.mean(axis=1).replace([np.inf, -np.inf], np.nan)
            .dropna().rename("ew"))


def load_index_ret(code: str = "000300.SH") -> pd.Series:
    """指数日频收益（沪深300：2013 起，用于**独立扩样本**的旁证检验）。"""
    f = DATA / f"index_daily_{code.replace('.', '_')}.parquet"
    d = pd.read_parquet(f)
    s = d["close"].droplevel(1) if isinstance(d.index, pd.MultiIndex) else d["close"]
    return s.sort_index().pct_change(fill_method=None).dropna().rename(code)


def load_strategy() -> pd.Series:
    """月频策略日频收益（882 ortho · ens_h1h5 · M · raw · f0.10）。"""
    d = pd.read_csv(EQUITY, index_col=0, parse_dates=True)
    return d["daily_ret"].dropna().rename("strategy")


# ---------------------------------------------------------------------------
# 月频化
# ---------------------------------------------------------------------------
def rebalance_days(dates: pd.DatetimeIndex, phase: int = 0) -> pd.DatetimeIndex:
    """每月第 ``phase`` 个交易日（phase=0 即定版：当月首个交易日）。"""
    s = pd.Series(dates, index=dates)
    out = [blk.iloc[phase] for _, blk in s.groupby(s.index.to_period("M"))
           if len(blk) > phase]
    return pd.DatetimeIndex(out)


def to_monthly(ret: pd.Series, phase: int = 0) -> pd.Series:
    """日频 → 月频：第 i 个月收益 = 调仓日 ``rb[i-1]`` → ``rb[i]`` 的累计。

    ⚠️ 月份归属用**区间右端点**（``rb[i]`` 所在日历月），因为该收益是
    「到下一个调仓日为止」实现的，标签必须与决策时点对齐。
    """
    rb = rebalance_days(ret.index, phase)
    vals, idx = [], []
    for i in range(1, len(rb)):
        seg = ret.loc[rb[i - 1]:rb[i]]
        if len(seg) == 0:
            continue
        vals.append(float((1 + seg).prod() - 1))
        idx.append(rb[i])
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).rename("ret")


def build_features(mret: pd.Series, newsy: tuple[int, ...],
                   shuffle_seed: int | None = None) -> pd.DataFrame:
    """构造预测变量与预测目标。

    🔑 **对齐口径（本脚本唯一允许的）**：行 ``t`` = 在调仓日 ``rb[t]`` 决策，
    预测 ``fwd[t] = ret[t+1]``（下一个月的收益）。因此

    * 特征可用 ``ret[:t+1]``（``ret[t]`` 是刚结束那个月的已实现收益，已知）；
    * 目标 ``fwd[t]`` 严格在未来；
    * ``next_newsy[t]`` = ``t+1`` 的日历月是否 newsy —— 日历属性，事前完全可知。

    🚨 **踩过的坑**：初版设 ``target = ret[t]`` 而 ``past12[t]`` 含 ``ret[t]``
    ⇒ 特征直接吃到标签，纯噪声下 M1 的 OOS R² 达 **0.0944**（`t_mse` 3.5）。
    改成 ``fwd`` 口径后纯噪声 R² 回到 ≈0。

    * ``past12``      ：过去 12 个月累计收益（传统自相关口径）
    * ``newsy_sum12`` ：过去 12 个月中、日历月 ∈ ``newsy`` 的月份收益之和
    * ``next_newsy``  ：**被预测月**（``t+1``）的日历月是否 ∈ ``newsy``

    安慰剂（``shuffle_seed``）：把「日历月 → 是否 newsy」的映射随机重排，
    保持 newsy 月个数不变（12 个月里 3 个），从而破坏「哪个日历月是财报季」这一机制，
    但保留「存在 3 个特殊月」的边际分布。
    """
    m = mret.index.month.to_numpy()
    is_newsy = np.isin(m, newsy)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        # 在「12 个日历月」层面重排：保持 newsy 数量，打乱是哪些月
        perm_months = rng.permutation(np.arange(1, 13))
        n_newsy = len(newsy)
        newsy_perm = tuple(int(x) for x in perm_months[:n_newsy])
        is_newsy = np.isin(m, newsy_perm)

    df = pd.DataFrame({"ret": mret, "newsy": is_newsy.astype(float)}, index=mret.index)
    df["fwd"] = df["ret"].shift(-1)          # 预测目标：下月收益
    df["past12"] = df["ret"].rolling(12).sum()
    df["newsy_sum12"] = (df["ret"] * df["newsy"]).rolling(12).sum()
    df["next_newsy"] = df["newsy"].shift(-1)  # 被预测月的日历属性（事前可知）
    return df


# ---------------------------------------------------------------------------
# 模型：扩展窗 OLS（walk-forward，OOS）
# ---------------------------------------------------------------------------
def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """带截距最小二乘（列标准化以稳定数值，返回原始尺度系数）。"""
    xm = np.nanmean(x, axis=0)
    xs = np.nanstd(x, axis=0)
    xs[xs < 1e-12] = 1.0
    z = np.nan_to_num((x - xm) / xs, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.column_stack([z, np.ones(len(z))])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    w = beta[:-1] / xs
    return np.concatenate([w, [beta[-1] - xm @ w]])


def model_matrix(df: pd.DataFrame, kind: str) -> list[str]:
    if kind == "M0_mean":
        return []
    if kind == "M1_ar12":
        return ["past12"]
    if kind in ("M2_cond12",):
        return ["past12", "next_newsy"]
    return ["newsy_sum12", "next_newsy"]


def walk_forward(df: pd.DataFrame, kind: str, min_train: int = MIN_TRAIN,
                 interact: bool = True) -> pd.Series:
    """扩展窗滚动重估：预测 ``t`` 的**下月**收益 ``fwd[t] = ret[t+1]``。

    交互项 ``x * next_newsy`` 是 Guo 的核心：允许「同一历史收益」在
    未来 newsy 月与非 newsy 月上有**不同符号**的预测系数。

    🔑 PIT：训练集用 ``[0, t]``（含 ``ret[t]``，在 ``rb[t]`` 已实现），
    但**目标**用 ``fwd[:t]``，其中 ``fwd[t]`` 是未来的 —— 故训练集末端为
    ``t+1`` 行之前，即 ``df.iloc[:t]`` 的 ``fwd`` 列全部 ``<= ret[t]``，安全。
    """
    cols = model_matrix(df, kind)
    y = df["fwd"].to_numpy()
    pred = pd.Series(np.nan, index=df.index)

    for t in range(min_train, len(df)):
        tr = df.iloc[:t]
        yy = y[:t]
        if kind == "M0_mean":
            pred.iloc[t] = float(np.nanmean(yy))
            continue
        x = tr[cols].to_numpy(dtype=float)
        parts = [x]
        if interact and "next_newsy" in cols:
            base = [c for c in cols if c != "next_newsy"]
            nn = tr["next_newsy"].to_numpy(dtype=float)[:, None]
            parts = [tr[base].to_numpy(dtype=float),
                     tr[base].to_numpy(dtype=float) * nn,
                     nn]
        xx = np.column_stack(parts)
        ok = np.isfinite(xx).all(axis=1) & np.isfinite(yy)
        if ok.sum() < max(24, min_train // 2):
            continue
        beta = _ols(xx[ok], yy[ok])

        xt = df.iloc[t]
        if interact and "next_newsy" in cols:
            base = [c for c in cols if c != "next_newsy"]
            b = np.array([xt[c] for c in base], dtype=float)
            nn = float(xt["next_newsy"])
            xt_vec = np.concatenate([b, b * nn, [nn]])
        else:
            xt_vec = np.array([xt[c] for c in cols], dtype=float)
        if np.isfinite(xt_vec).all():
            pred.iloc[t] = float(xt_vec @ beta[:-1] + beta[-1])
    return pred


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def expanding_mean(y: pd.Series, min_periods: int = 12) -> pd.Series:
    """扩展窗历史均值（**PIT**：``t`` 时刻只用 ``[0, t)``）。

    🔑 这是 Goyal-Welch (2008) 的基准，也是本脚本唯一允许的基准。
    初版误用 ``rolling(48).mean().shift(1)`` ⇒ 纯噪声下 OOS R² 系统性 +0.031：
    因为模型内部用扩展窗截距，**扩展窗天然比 48 期滚动窗更准**，这是基准不对称
    造成的假阳性，不是前视。修正后 ``M0_mean`` 对同一基准的 R² 必须 ≈ 0（自检 4）。
    """
    return y.shift(1).expanding(min_periods=min_periods).mean()


def oos_r2(pred: pd.Series, y: pd.Series, base: pd.Series) -> dict:
    """样本外 R²（相对扩展窗历史均值 ``base``）；附 MSE 差 t 值。

    R² 定义沿用 Goyal-Welch（与文献可直接对照）：
    ``1 - SSE(model) / SSE(historical mean)``。**负值 = 不如直接用均值**。
    """
    d = pd.concat([pred.rename("p"), y.rename("y"), base.rename("b")], axis=1).dropna()
    if len(d) < 24:
        return {"n": len(d), "oos_r2": float("nan"), "t_mse": float("nan"),
                "hit": float("nan"), "hit_all_long": float("nan"), "ic": float("nan")}
    sse_m = float(((d.y - d.p) ** 2).sum())
    sse_b = float(((d.y - d.b) ** 2).sum())
    e_m = (d.y - d.p) ** 2
    e_b = (d.y - d.b) ** 2
    diff = e_b - e_m                      # >0 表示模型更优
    t_mse = (float(diff.mean() / diff.std() * np.sqrt(len(diff)))
             if diff.std() > 0 else float("nan"))
    hit = float((np.sign(d.p) == np.sign(d.y)).mean())
    return {"n": len(d), "oos_r2": 1 - sse_m / sse_b if sse_b > 0 else float("nan"),
            "t_mse": t_mse, "hit": hit, "hit_all_long": float((d.y > 0).mean()),
            "ic": float(d.p.corr(d.y, method="spearman"))}


def breakeven_r2(vol_ann: float, cost_bp: float = COST_BP,
                 turn: float = TURNOVER_MONTHLY) -> dict:
    """盈亏平衡所需的月度 R²（本项目的门槛，不是文献的）。

    ``年成本 = turn × cost_bp``；所需 IR = 年成本 / 年化波动；
    月频广度 12 ⇒ Grinold ``IR = IC·√12`` ⇒ ``IC = IR/√12`` ⇒ ``R² = IC²``。
    """
    cost = turn * cost_bp / 1e4
    ir = cost / vol_ann
    ic = ir / np.sqrt(12.0)
    return {"cost_ann": cost, "ir_needed": ir, "ic_needed": ic, "r2_needed": ic ** 2}


# ---------------------------------------------------------------------------
# 策略层：预测 → 仓位乘数 → 净值
# ---------------------------------------------------------------------------
def to_expo(pred: pd.Series, dates: pd.DatetimeIndex,
            lo: float = 0.5, hi: float = 1.0) -> pd.Series:
    """月度预测 → ω ∈ {lo, hi}：用**滚动分位**（仅过去 24 期，PIT）判强弱。

    ⚠️ 最朴素的映射，不调参。目的：测「信号有没有方向」，不是找最优仓位规则。
    """
    q = pred.rolling(24, min_periods=12).apply(
        lambda w: (w[-1] > np.nanmedian(w[:-1])).astype(float), raw=True)
    e = pd.Series(np.where(q > 0.5, hi, lo), index=pred.index)
    e = e.where(q.notna(), hi)
    return e.reindex(dates, method="ffill")


def apply_cost(base: pd.Series, expo: pd.Series,
               bp: float = POSITION_TURNOVER_BP) -> pd.Series:
    """``r = r_strategy·ω − |Δω|·bp``（仓位切换成本单边 bp）。"""
    e = expo.reindex(base.index).ffill().fillna(1.0)
    turn = e.diff().abs().fillna(0.0)
    return base * e - turn * bp / 1e4


def annual_return(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (252 / len(r)) - 1) if len(r) else float("nan")


def geo_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    v = float(r.std() * np.sqrt(252))
    return annual_return(r) / v if v > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    c = (1 + r.dropna()).cumprod()
    return float((c / c.cummax() - 1).min())


def _t(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(len(x)))


# ---------------------------------------------------------------------------
# 自查
# ---------------------------------------------------------------------------
def self_check() -> None:
    """跑前守卫：防「实现错了但结论看着对」。"""
    # 1) 月频化长度自洽
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    fake = pd.Series(0.0005, index=idx)
    m = to_monthly(fake, 0)
    assert 22 <= len(m) <= 25, f"月频化长度异常：{len(m)}"

    # 2) newsy 标签在 12 个月里恰好 len(NEWSY) 个
    df = build_features(m, NEWSY)
    frac = float(df["newsy"].mean())
    assert abs(frac - len(NEWSY) / 12) < 0.02, f"newsy 占比异常：{frac:.3f}"

    # 3) 安慰剂必须改变「哪些月是 newsy」，但保持数量
    df2 = build_features(m, NEWSY, shuffle_seed=7)
    assert abs(df2["newsy"].mean() - frac) < 0.02, "安慰剂改变了 newsy 数量"

    # 4) 基准自洽：``M0_mean``（扩展窗均值）对同一扩展窗基准的 R² 必须 ≈ 0。
    #    若不为 0，说明模型内部截距与基准口径不一致（基准不对称 ⇒ 假阳性）。
    rng = np.random.default_rng(11)
    noise = pd.Series(rng.standard_normal(300) * 0.1,
                      index=pd.date_range("2000-01-31", periods=300, freq="ME"))
    dnf = build_features(noise, NEWSY)
    bn = expanding_mean(dnf["fwd"])
    p0 = walk_forward(dnf, "M0_mean", min_train=48)
    r0 = oos_r2(p0, dnf["fwd"], bn)
    assert abs(r0["oos_r2"]) < 1e-6, (
        f"M0_mean 与基准口径不一致（应为 0）：{r0['oos_r2']:.6f}")

    # 5) 纯噪声下 M3 的 OOS R² 不应显著为正（防前视泄漏）。
    #    判据用**中位数**而非单跑：单 seed 的抽样波动可达 ±0.03，
    #    单点断言会产生假警报；12 个 seed 的中位数才稳定。
    seeds = []
    for sd in range(12):
        rr = np.random.default_rng(sd)
        nz = pd.Series(rr.standard_normal(600) * 0.1,
                       index=pd.date_range("1990-01-31", periods=600, freq="ME"))
        dfn = build_features(nz, NEWSY)
        pn_ = walk_forward(dfn, "M3_newsy_sum", min_train=48)
        bnn = expanding_mean(dfn["fwd"], min_periods=48)
        seeds.append(oos_r2(pn_, dfn["fwd"], bnn)["oos_r2"])
    med = float(np.median(seeds))
    assert med < 0.005, f"纯噪声 OOS R2 中位数异常（疑前视）：{med:.4f}"

    # 6) 日历标签不是前视：next_newsy 只依赖 index.month
    assert df["next_newsy"].dropna().isin([0.0, 1.0]).all(), "next_newsy 非 0/1"
    logging.getLogger("cond").info(
        "self_check 通过（噪声 R2 中位数 %.4f）", med)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def month_profile(income_path: Path) -> pd.DataFrame:
    """实测 A 股定期报告披露月分布 ⇒ 为 newsy 定义提供数据依据。"""
    i = pd.read_parquet(income_path, columns=["ann_date", "report_period",
                                              "statement_type"])
    i = i[(i["statement_type"] == "1")].dropna(subset=["ann_date", "report_period"])
    i = i[i["ann_date"].dt.year >= 2016]
    i = i.assign(rp_m=i["report_period"].dt.month,
                 ann_m=i["ann_date"].dt.month)
    ct = pd.crosstab(i["rp_m"], i["ann_m"], normalize="index") * 100
    ct = ct.loc[[3, 6, 9, 12]]
    ct.index = ["Q1(3/31)", "H1(6/30)", "Q3(9/30)", "FY(12/31)"]
    return ct.round(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-arms", action="store_true", help="只跑诊断段")
    ap.add_argument("--phases", type=int, default=len(PHASES))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("cond")
    OUT.mkdir(parents=True, exist_ok=True)
    self_check()

    # ---- 数据 ----
    ew = load_ew_ret()
    hs300 = load_index_ret("000300.SH")
    strat = load_strategy()
    log.info("等权全A %d 天 %s ~ %s", len(ew), ew.index.min().date(),
             ew.index.max().date())
    log.info("沪深300 %d 天 %s ~ %s", len(hs300), hs300.index.min().date(),
             hs300.index.max().date())
    log.info("策略净值 %d 天 %s ~ %s", len(strat), strat.index.min().date(),
             strat.index.max().date())

    mp = month_profile(DATA / "income.parquet")
    mp.to_csv(OUT / "month_profile.csv")
    log.info("\n=== A股定期报告披露月分布（%%）===\n%s", mp.to_string())

    # ---- 诊断段：两样本 × 两口径 × 五臂 ----
    rows = []
    cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    for sname, sret in (("等权全A", ew), ("沪深300", hs300)):
        for cname, newsy in (("主口径{4,8,10}", NEWSY), ("备选{3,4,8,10}", NEWSY_ALT)):
            m = to_monthly(sret, 0)
            df = build_features(m, newsy)
            cache[(sname, cname, 0)] = df
            base = expanding_mean(df["fwd"], min_periods=MIN_TRAIN)
            for kind in ARMS:
                seed = 20261111 if kind == "M4_placebo" else None
                d = (build_features(m, newsy, shuffle_seed=seed)
                     if seed else df)
                p = walk_forward(d, kind)
                r = oos_r2(p, d["fwd"], base)
                rows.append({"sample": sname, "newsy_def": cname, "arm": kind,
                             "label": ARMS[kind], **r})
    oos = pd.DataFrame(rows)
    oos.to_csv(OUT / "oos.csv", index=False)
    log.info("\n=== 样本外 R²（月度，相对扩展窗均值）===\n%s",
             oos[["sample", "newsy_def", "arm", "n", "oos_r2", "t_mse",
                  "hit", "hit_all_long", "ic"]].round(4).to_string(index=False))

    # 盈亏平衡门槛
    be = breakeven_r2(float(ew.std() * np.sqrt(252)))
    log.info("\n=== 盈亏平衡门槛（本项目口径）===\n%s",
             pd.Series({k: round(v, 5) for k, v in be.items()}).to_string())

    if args.skip_arms:
        log.info("产物 → %s", OUT)
        return

    # ---- 臂段 ----
    arms, pairs = [], []
    n_ph = min(args.phases, len(PHASES))

    for ph in range(n_ph):
        m = to_monthly(ew, ph)
        df = build_features(m, NEWSY)
        preds = {}
        for kind in ARMS:
            seed = 20261111 if kind == "M4_placebo" else None
            d = build_features(m, NEWSY, shuffle_seed=seed) if seed else df
            preds[kind] = walk_forward(d, kind)
        rb = rebalance_days(strat.index, ph)
        for kind in ARMS:
            e = to_expo(preds[kind], rb)
            r = apply_cost(strat, e)
            arms.append({"arm": kind, "phase": ph, "ann": annual_return(r),
                         "sharpe": geo_sharpe(r), "mdd": max_drawdown(r),
                         "exposure_mean": float(e.reindex(strat.index)
                                                .ffill().mean())})
        # 对照：恒定仓位 = M3 的实际平均仓位（逐相位算）
        ex3 = float(to_expo(preds["M3_newsy_sum"], rb).reindex(strat.index)
                    .ffill().mean())
        e_const = pd.Series(ex3, index=rb)
        r = apply_cost(strat, e_const)
        arms.append({"arm": "G_const_expo", "phase": ph, "ann": annual_return(r),
                     "sharpe": geo_sharpe(r), "mdd": max_drawdown(r),
                     "exposure_mean": ex3})
        e_full = pd.Series(1.0, index=rb)
        r = apply_cost(strat, e_full)
        arms.append({"arm": "base_buyhold", "phase": ph, "ann": annual_return(r),
                     "sharpe": geo_sharpe(r), "mdd": max_drawdown(r),
                     "exposure_mean": 1.0})

    ad = pd.DataFrame(arms)
    ad.to_csv(OUT / "arms.csv", index=False)

    for ref in ("base_buyhold", "G_const_expo"):
        b = ad[ad.arm == ref].set_index("phase")["ann"]
        bs = ad[ad.arm == ref].set_index("phase")["sharpe"]
        for kind in ARMS:
            d = (ad[ad.arm == kind].set_index("phase")["ann"] - b).dropna()
            ds = (ad[ad.arm == kind].set_index("phase")["sharpe"] - bs).dropna()
            if len(d) < 2:
                continue
            pairs.append({"arm": kind, "vs": ref, "n": len(d),
                          "mean_pp": 100 * d.mean(), "t": _t(d),
                          "pos": int((d > 0).sum()), "d_sharpe": float(ds.mean())})
    pd_ = pd.DataFrame(pairs)
    pd_.to_csv(OUT / "pairs.csv", index=False)

    log.info("\n=== 各臂（%d 相位均值）===\n%s",
             n_ph, ad.groupby("arm")[["ann", "sharpe", "mdd", "exposure_mean"]]
             .mean().round(4).to_string())
    log.info("\n=== 配对差（逐相位）===\n%s",
             pd_.round(3).to_string(index=False))

    # 逐年（phase=0）
    yrs = []
    m = to_monthly(ew, 0)
    df = build_features(m, NEWSY)
    rb = rebalance_days(strat.index, 0)
    for kind in ARMS:
        seed = 20261111 if kind == "M4_placebo" else None
        d = build_features(m, NEWSY, shuffle_seed=seed) if seed else df
        p = walk_forward(d, kind)
        r = apply_cost(strat, to_expo(p, rb))
        for y, sub in r.groupby(r.index.year):
            yrs.append({"arm": kind, "year": y, "ann": annual_return(sub),
                        "sharpe": geo_sharpe(sub)})
    yt = pd.DataFrame(yrs)
    yt.to_csv(OUT / "yearly.csv", index=False)
    log.info("\n=== 逐年（phase=0）===\n%s",
             yt.pivot(index="year", columns="arm", values="ann").round(4).to_string())
    log.info("产物 → %s", OUT)


if __name__ == "__main__":
    main()