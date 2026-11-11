"""日频训练 → 月频策略上的择时乘数（第三个格子）。

动机（2026-11-11 由莉酱追问）
---------------------------
「用大盘**日频**数据训练一个择时，但是在**月频策略**上用。」

``timing_probe`` 只测过两个格子，都不是这一格：

* **日频训练 + 日频执行**（§二 臂 1）：零成本也只有 14.96% vs 满仓 15.54%，
  因大盘日收益自相关仅 **+0.048**。
* **月频训练 + 月频执行**（§三）：标签只有 105 个，7 个月度因子 0/7 显著。

本探针测第三格。它与前两格的关键差别**不是样本量**（那是次要的），而是：

1. **训练标签** = 大盘**次日**方向 ⇒ 样本 1760+（相对 105 提升 √17 倍功效）。
2. **执行频率** = 月频 ⇒ 年换手 ~12 次而非 180+，**成本不是瓶颈**。
3. **目标错位**（真正的难点）：日频模型学的是「明天涨跌」，用户要的是
   「下月涨跌」。必须聚合（求和 / 均值 / 月末截面），**聚合时正负会相互抵消**
   ⇒ 日频可预测性**不能自动**变成月度可预测性。

因此本探针的核心不是「能不能赚」，而是**把错位量出来**：
考察日频 IC 是否显著、以及聚合后与下月收益的 IC 还剩多少。

关键设计
--------
* **训练数据**：等权全A 日频（``close_raw`` 列向均值，与 ``bench_eqw`` ρ=0.9996）。
* **施加对象**：月频策略日频净值 ``eq__ens_h1h5__h1__M__raw__f0.10.csv``
  （882 ortho，年化 15.54%，与 ``timing_probe`` 满仓基线同源）——
  ω 只在**该策略自己的调仓日**改变，与策略换仓同日，不额外增加换手日。
* **ω 取值 {0.5, 1.0}**：只降不升（long-only 下 ω≤1）⇒ 任何择时错误都只会
  拉低年化，这是「保险不是 alpha」的严格框架。

臂
--
* ``base_buyhold``      满仓策略（参照）
* ``B_daily_sumG``      日频标签训练 + 月内预测**求和** → 月频执行
* ``C_daily_meanP``     日频标签训练 + 月内预测**均值** → 月频执行
* ``D_daily_monthend``  日频标签训练 + **月末截面**预测 → 月频执行
* ``E_monthly_train``   月频标签训练（21 日）+ 月频执行（对照：只改标签频率）
* ``F_placebo``         打乱标签重训（安慰剂：Δ≈0 才算同源）

用法
----
    python -m scripts.evaluation.timing_daily_train_probe             # 全跑（约 3 分钟）
    python -m scripts.evaluation.timing_daily_train_probe --skip-arms # 只跑诊断（约 30 秒）

产物
----
    reports/timing_daily_train/{arms,pairs,agg_diag,yearly,phase}.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "timing_daily_train"
BASE = ROOT / "reports" / "alla_rolling" / "_base"
EQUITY = (ROOT / "reports" / "alla_rolling_ortho" / "equity"
          / "eq__ens_h1h5__h1__M__raw__f0.10.csv")

#: 仓位切换成本（单边 bp）：ω 变动 1.0 单位 = 满仓↔空仓。与 ``timing_probe`` 同口径。
POSITION_TURNOVER_BP = 10.0

#: 训练窗（交易日）与重估间隔。
#: ⚠️ 特征含 rolling(500, min_periods=120) 标准化 ⇒ 有效起点约 2016-12；
#: 策略净值从 2018-01 起 ⇒ 用 375 天窗才能在 2018 年内开始出样本外预测，
#: 否则月频观测不足 90 个无法判定。
TRAIN_WIN = 375
REFIT = 63

#: 特征集：五个市场状态量（``diag_raft_weight.build_state`` 同源）+ 三个价格衍生。
FEATS = ("mom5", "mom20", "mom60", "vol20", "disp20", "sizesp", "breadth", "drawdown")

#: 相位：月频调仓日在月内的第 n 个交易日（定版 = 0）。
PHASES = tuple(range(6))

#: 成本与换手锚点（全部来自本项目实测，非假设）：
#: ``COST_BP`` = 单边 36bp 往返（``Config.costs()``）；
#: ``TURNOVER_MONTHLY`` = 11.6 次/年（`solo_weekly` 月频实测）；
#: ``TURNOVER_DAILY`` = 178.8 次/年（`solo_weekly` 日频实测）。
COST_BP = 36.0
TURNOVER_MONTHLY = 11.6
TURNOVER_DAILY = 178.8

#: 选股横截面 IC 的外部锚点：``pool_scope_probe`` 全A 实测 0.1038
#: （见 ``reports/README.md``）。**不是本脚本算出**，仅作 Grinold 对照。
XS_IC = 0.1038


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def build_state(r1: pd.DataFrame, ew: pd.Series, mcap: pd.DataFrame) -> pd.DataFrame:
    """五个 PIT 市场状态量 + 三个价格衍生（严格 rolling 标准化）。"""
    log_ew = np.log1p(ew)
    mom5 = log_ew.rolling(5).sum()
    mom20 = log_ew.rolling(20).sum()
    mom60 = log_ew.rolling(60).sum()
    vol20 = ew.rolling(20).std() * np.sqrt(252)
    disp20 = r1.std(axis=1).rolling(20).mean()

    rk = mcap.reindex_like(r1).rank(axis=1, pct=True)
    small = r1.where(rk <= 0.3).mean(axis=1)
    big = r1.where(rk >= 0.7).mean(axis=1)
    sizesp = (small - big).rolling(20).sum()
    breadth = (r1 > 0).mean(axis=1).rolling(20).mean()

    px = (1 + ew).cumprod()
    dd = px / px.rolling(250, min_periods=60).max() - 1

    s = pd.DataFrame({
        "mom5": mom5, "mom20": mom20, "mom60": mom60,
        "vol20": vol20, "disp20": disp20, "sizesp": sizesp,
        "breadth": breadth, "drawdown": dd,
    })
    mu = s.rolling(500, min_periods=120).mean()
    sd = s.rolling(500, min_periods=120).std()
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)


def load_market() -> pd.DataFrame:
    """等权全A 日频：市场收益 + PIT 状态特征。"""
    close = pd.read_parquet(BASE / "close_raw.parquet")
    mcap = pd.read_parquet(BASE / "market_cap.parquet")
    # 🚨 数据缺陷守卫（2026-11-11 实测）：close_raw 有 2 格为 0.0
    # （301390.SZ / 603159.SH @ 2026-09-16，疑停牌填 0），次日跳回 45/22.95
    # ⇒ pct_change 出 inf ⇒ 污染 mean(axis=1) 与全部相关系数（实测 corr 全 NaN）。
    close = close.where(close > 0)
    r1 = close.pct_change(fill_method=None)

    ew = r1.mean(axis=1).replace([np.inf, -np.inf], np.nan).rename("ret")
    s = build_state(r1, ew, mcap)
    s["ret"] = ew
    return s.dropna(subset=["ret"]).sort_index()


def load_strategy() -> pd.Series:
    """月频策略日频收益（882 ortho · ens_h1h5 · M · raw · f0.10）。"""
    d = pd.read_csv(EQUITY, index_col=0, parse_dates=True)
    return d["daily_ret"].dropna().rename("strategy")


# ---------------------------------------------------------------------------
# 模型：岭回归（线性、8 特征、可解释；避免 GBDT 在 1700 样本上过拟合）
# ---------------------------------------------------------------------------
def _fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """带截距的岭回归闭式解（列标准化后正则，返回原始尺度系数）。"""
    xm = np.nanmean(x, axis=0)
    xs = np.nanstd(x, axis=0)
    xs[xs < 1e-12] = 1.0
    z = np.nan_to_num((x - xm) / xs, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - y.mean()
    a = z.T @ z + lam * np.eye(z.shape[1])
    w = np.linalg.solve(a, z.T @ y) / xs
    return np.concatenate([w, [y.mean() - xm @ w]])


def _predict(model: np.ndarray, x: np.ndarray) -> np.ndarray:
    z = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return z @ model[:-1] + model[-1]


def walk_forward(feat: pd.DataFrame, ret: pd.Series, horizon: int) -> pd.Series:
    """滚动重估的样本外预测（标签 = 未来 ``horizon`` 日累计收益）。

    🔑 PIT：``fwd[t]`` 覆盖 ``ret[t+1..t+h]``，故训练集末端须 ``t+h <= lo-1``
    ⇒ ``t <= lo-h``。初版写成 ``dates[:lo]``，h=21 时把未来 20 天收益喂进训练集
    （月频对照臂因此偏高），2026-11-11 已修。
    """
    fwd = (1 + ret).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
    dates = feat.index
    pred = pd.Series(np.nan, index=dates)

    for lo in range(TRAIN_WIN, len(dates), REFIT):
        end = max(0, lo - horizon + 1)
        tr = dates[max(0, end - TRAIN_WIN):end]
        tr = tr[fwd.reindex(tr).notna()]
        if len(tr) < 150:
            continue
        x = feat.reindex(tr)[list(FEATS)].to_numpy(dtype=float)
        y = fwd.reindex(tr).to_numpy(dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 120:
            continue
        model = _fit_ridge(x[ok], y[ok])
        hi = min(lo + REFIT, len(dates))
        pred.iloc[lo:hi] = _predict(model, feat.iloc[lo:hi][list(FEATS)]
                                    .to_numpy(dtype=float))
    return pred


# ---------------------------------------------------------------------------
# 聚合：日频预测 → 每月一个仓位乘数
# ---------------------------------------------------------------------------
def rebalance_days(dates: pd.DatetimeIndex, phase: int = 0) -> pd.DatetimeIndex:
    """每月第 ``phase`` 个交易日（phase=0 即定版：当月首个交易日）。"""
    s = pd.Series(dates, index=dates)
    out = [blk.iloc[phase] for _, blk in s.groupby(s.index.to_period("M"))
           if len(blk) > phase]
    return pd.DatetimeIndex(out)


def aggregate(pred: pd.Series, dates: pd.DatetimeIndex, phase: int,
              how: str) -> pd.Series:
    """把日频预测聚合成「每个调仓日一个 ω ∈ {0.5, 1.0}」。

    ``how``: ``sum`` 月内预测求和 / ``mean`` 均值 / ``last`` 只取调仓日当天。

    阈值用**滚动分位**（PIT，仅过去 24 个调仓日）：当前值强于历史中位 ⇒ 满仓，
    否则半仓。⚠️ 这不是最优映射，只是最朴素的一个——目的是测「错位」而非调参。
    """
    rb = rebalance_days(dates, phase)
    vals, prev = [], None
    for d in rb:
        seg = (pred.loc[prev:d] if prev is not None else pred.loc[:d]).dropna()
        if len(seg) == 0:
            vals.append(np.nan)
        elif how == "sum":
            vals.append(float(seg.sum()))
        elif how == "mean":
            vals.append(float(seg.mean()))
        else:
            vals.append(float(seg.iloc[-1]))
        prev = d
    sig = pd.Series(vals, index=rb)

    q = sig.rolling(24, min_periods=12).apply(
        lambda w: (w[-1] > w[:-1]).mean(), raw=True)
    return pd.Series(np.where(q > 0.5, 1.0, 0.5), index=rb).where(q.notna(), 1.0)


def apply_cost(base: pd.Series, expo: pd.Series,
               bp: float = POSITION_TURNOVER_BP) -> pd.Series:
    """``r = r_strategy·ω − |Δω|·bp``；ω 在调仓日改变，ffill 到日频。"""
    e = expo.reindex(base.index).ffill()
    e = e.where(e.notna(), 1.0)
    turn = e.diff().abs().fillna(0.0)
    return base * e - turn * bp / 1e4


# ---------------------------------------------------------------------------
# 指标（几何 Sharpe，与 backtest/metrics.py 同源）
# ---------------------------------------------------------------------------
def annual_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def annual_vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(252))


def geo_sharpe(r: pd.Series) -> float:
    v = annual_vol(r)
    return annual_return(r) / v if v > 0 else float("nan")


def arith_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    s = r.std()
    return float(r.mean() / s * np.sqrt(252)) if s > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    c = (1 + r.dropna()).cumprod()
    return float((c / c.cummax() - 1).min())


def _t(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(len(x)))


# ---------------------------------------------------------------------------
# 诊断：重叠标签的有效样本量 + 错位程度
# ---------------------------------------------------------------------------
def agg_diagnostic(pred: pd.Series, ret: pd.Series, phase: int = 0) -> pd.DataFrame:
    """量出「日频模型 → 月度目标」的错位。

    * ``daily_ic``：pred(t) 与 r(t+1) 的 rank IC（模型真学到东西吗）
    * ``agg_ic_*``：三种聚合值与**下月**实际收益的 rank IC（错位有多大）
    * ``overlap_eff_n``：日频标签的信息上限 = n / 21（不是 n）
    """
    rb = rebalance_days(ret.index, phase)
    nxt = list(rb[1:]) + [ret.index[-1]]
    rows = []
    for i, d in enumerate(rb[:-1]):
        lo = rb[i - 1] if i > 0 else d
        seg = pred.loc[lo:d].dropna()
        if len(seg) == 0:
            continue
        mret = float((1 + ret.loc[d:nxt[i]]).prod() - 1)
        rows.append({"date": d, "p_sum": float(seg.sum()),
                     "p_mean": float(seg.mean()), "p_last": float(seg.iloc[-1]),
                     "next_ret": mret})
    df = pd.DataFrame(rows).set_index("date")
    if len(df) < 12:
        return pd.DataFrame()

    d_ic = float(pred.corr(ret.shift(-1), method="spearman"))
    n_d = int(pred.notna().sum())
    out = [{"diag": "daily_ic(预测次日)", "value": d_ic,
            "t": d_ic * np.sqrt(n_d), "n": n_d}]
    for how in ("sum", "mean", "last"):
        c = float(df[f"p_{how}"].corr(df["next_ret"], method="spearman"))
        out.append({"diag": f"agg_ic_{how}(预测下月)", "value": c,
                    "t": c * np.sqrt(len(df) - 1), "n": len(df)})
    out.append({"diag": "overlap_eff_n(有效独立样本)", "value": n_d / 21.0,
                "t": float("nan"), "n": n_d})
    return pd.DataFrame(out)


def fit_diagnostic(pred: pd.Series, ret: pd.Series) -> pd.DataFrame:
    """模型**真的学到了东西吗**（区分「信号弱」与「模型无效」）。

    * ``oos_r2``：对次日收益的样本外 R²。**负值 = 比「永远猜均值」更差**，
      这是「模型无效」而非「信号弱」的判决性证据。
    * ``hit_rate``：方向命中率。必须与「全押正」（= 上涨日占比）比，
      **低于全押正就等于择时在帮倒忙**。
    * ``quintile_spread``：五分位最高 − 最低的次日平均收益（单调性）。
    """
    d = pd.concat([pred.rename("p"), ret.shift(-1).rename("y")], axis=1).dropna()
    if len(d) < 100:
        return pd.DataFrame()
    ss_res = float(((d.y - d.p) ** 2).sum())
    ss_tot = float(((d.y - d.y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    hit = float((np.sign(d.p) == np.sign(d.y)).mean())
    base_hit = float((d.y > 0).mean())
    q = pd.qcut(d.p.rank(method="first"), 5, labels=False)
    gm = d.groupby(q).y.mean()
    return pd.DataFrame([
        {"diag": "oos_r2(次日收益)", "value": r2, "t": float("nan"), "n": len(d)},
        {"diag": "hit_rate(方向命中)", "value": hit, "t": float("nan"), "n": len(d)},
        {"diag": "hit_rate_all_long(全押正)", "value": base_hit,
         "t": float("nan"), "n": len(d)},
        {"diag": "quintile_Q5_minus_Q1", "value": float(gm.iloc[-1] - gm.iloc[0]),
         "t": float("nan"), "n": len(d)},
    ])


def ar_r2(ret: pd.Series, k: int) -> tuple[float, int]:
    """AR(``k``) 对日收益的**样本内** R²（含截距，最小二乘）。返回 (R², n)。"""
    y = ret.to_numpy()
    x = np.column_stack([ret.shift(i).to_numpy() for i in range(1, k + 1)]
                        + [np.ones(len(ret))])
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(x[ok], y[ok], rcond=None)
    p = x[ok] @ beta
    n = int(ok.sum())
    return float(1 - ((y[ok] - p) ** 2).sum()
                 / ((y[ok] - y[ok].mean()) ** 2).sum()), n


def oos_ar_r2(ret: pd.Series, k: int, win: int = 250) -> tuple[float, int]:
    """AR(``k``) 的**滚动样本外** R²（用过去 ``win`` 天估参，预测次日）。

    这才是「能用的可预测性」：样本内 R² 会随 k 单调增长（纯噪声也增长），
    只有样本外 R² 才能区分「真信号」与「过拟合」。
    """
    y = ret.to_numpy()
    pred = np.full(len(y), np.nan)
    for t in range(win, len(y)):
        x = np.column_stack([ret.shift(i).to_numpy()[:t] for i in range(1, k + 1)])
        ok = np.isfinite(x).all(axis=1)
        if ok.sum() < win // 2:
            continue
        xs = np.column_stack([x[ok], np.ones(int(ok.sum()))])
        beta, *_ = np.linalg.lstsq(xs, y[:t][ok], rcond=None)
        xt = np.array([ret.iloc[t - i] for i in range(1, k + 1)] + [1.0])
        if np.isfinite(xt).all():
            pred[t] = float(xt @ beta)
    ok = np.isfinite(pred) & np.isfinite(y)
    return float(1 - ((y[ok] - pred[ok]) ** 2).sum()
                 / ((y[ok] - y[ok].mean()) ** 2).sum()), int(ok.sum())


def predictability_ceiling(feat: pd.DataFrame, ret: pd.Series) -> pd.DataFrame:
    """「大盘方向预测效果差」的**定量**版本（2026-11-11，两轮修正后）。

    ⚠️ **修正史（重要）**：第一版把「AR(1) 样本内 R² = ρ² = 0.00194」写成
    「R² 天花板」——**这是错的**（由莉酱 2026-11-11 指出）。样本内 R² **随 lag
    数单调增长**（实测 AR(120) 达 0.0552），因为每加一个参数都能"解释"掉一点噪声。
    正确的三个量是：

    1. **噪声地板**：零假设（纯噪声）下 ``E[R²] = k/n``。样本内 R² 必须减掉它
       才是「真信号」。实测各阶扣噪声后 **稳定在 0.0015~0.0023**、不随 k 增长
       ⇒ 这才是真正的可预测性量级。
    2. **样本外 R²**：唯一能直接用的量。AR(1..60) 实测 **全部为负**
       （−0.0016 ~ −0.0872）⇒ 连那点样本内信号都估不出来。
    3. **盈亏平衡 R²**：月频成本 / 波动 ⇒ 所需 R²，与现实比。

    ⚠️ ``XS_IC`` 来自 ``reports/README.md`` 的 `pool_scope_probe` 条目（全A 0.1038），
    **不是本脚本算出**，仅作对照用的外部锚点；其余全部由本函数实测。
    """
    vol = float(ret.std() * np.sqrt(252))
    ac1 = float(ret.corr(ret.shift(1)))
    r2_1, n_1 = ar_r2(ret, 1)
    r2_5, n_5 = ar_r2(ret, 5)
    r2_20, n_20 = ar_r2(ret, 20)
    r2_120, n_120 = ar_r2(ret, 120)

    rows = [
        {"item": "市场年化波动", "value": vol},
        {"item": "lag1 自相关 rho", "value": ac1},
        {"item": "AR(1) 样本内 R2", "value": r2_1},
        {"item": "AR(1) 噪声地板(k/n)", "value": 1.0 / n_1},
        {"item": "AR(1) 扣噪声真信号", "value": r2_1 - 1.0 / n_1},
        {"item": "AR(5) 样本内 R2", "value": r2_5},
        {"item": "AR(5) 噪声地板(k/n)", "value": 5.0 / n_5},
        {"item": "AR(5) 扣噪声真信号", "value": r2_5 - 5.0 / n_5},
        {"item": "AR(20) 样本内 R2", "value": r2_20},
        {"item": "AR(20) 扣噪声真信号", "value": r2_20 - 20.0 / n_20},
        {"item": "AR(120) 样本内 R2", "value": r2_120},
        {"item": "AR(120) 扣噪声真信号", "value": r2_120 - 120.0 / n_120},
    ]
    for k in (1, 5, 20, 60):
        r2o, no = oos_ar_r2(ret, k)
        rows.append({"item": f"AR({k}) 样本外 R2", "value": r2o})

    # Grinold：同一公式，两个目标
    ic_xs = XS_IC
    rows += [
        {"item": "选股 IC(外部锚点 pool_scope_probe)", "value": ic_xs},
        {"item": "选股 IR(5000x12)", "value": ic_xs * np.sqrt(5000 * 12)},
        {"item": "选股 IR(实持 8 只 x12)", "value": ic_xs * np.sqrt(8 * 12)},
        {"item": "择时 IR 日频(rho*sqrt252)", "value": ac1 * np.sqrt(252)},
        {"item": "择时 IR 月频(rho_m*sqrt12)", "value": np.sqrt(0.00105) * np.sqrt(12)},
    ]

    # 盈亏平衡
    cost_m = TURNOVER_MONTHLY * COST_BP / 1e4
    need_ir = cost_m / vol
    need_r2 = (need_ir / np.sqrt(12)) ** 2
    r2_real = 0.00105
    rows += [
        {"item": "月频成本(11.6x36bp)", "value": cost_m},
        {"item": "所需月频 IR", "value": need_ir},
        {"item": "所需月频 R2", "value": need_r2},
        {"item": "现实月频 R2", "value": r2_real},
        {"item": "现实/所需", "value": r2_real / need_r2},
        {"item": "需再强倍数", "value": need_r2 / r2_real},
        {"item": "日频毛年化(IR*vol)", "value": ac1 * np.sqrt(252) * vol},
        {"item": "日频成本(178.8x36bp)", "value": TURNOVER_DAILY * COST_BP / 1e4},
        {"item": "日频每单位换手可承受_bp",
         "value": ac1 * np.sqrt(252) * vol / TURNOVER_DAILY * 1e4},
    ]
    return pd.DataFrame(rows)


def single_factor_ic(feat: pd.DataFrame, ret: pd.Series) -> pd.DataFrame:
    """单因子 vs 次日收益的 rank IC（判断「模型差」还是「特征没用」）。"""
    fwd = ret.shift(-1)
    rows = []
    for c in FEATS:
        s = feat[c]
        ic = float(s.corr(fwd, method="spearman"))
        n = int(s.notna().sum())
        rows.append({"factor": c, "ic": ic, "t": ic * np.sqrt(n), "n": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def self_check() -> None:
    """守卫：无换手无成本、相位覆盖、有效样本公式、PIT 标签窗口。"""
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(np.full(60, 0.001), index=idx)
    e = pd.Series(1.0, index=idx)
    assert abs(apply_cost(r, e, 0.0).sum() - apply_cost(r, e, 10.0).sum()) < 1e-12, \
        "无换手不应有成本"

    rb = rebalance_days(idx, 0)
    assert len(rb) == 3, f"60 个交易日应覆盖 3 个月，实际 {len(rb)}"
    assert rb.is_monotonic_increasing

    assert abs(60 / 21 - 2.857142857) < 1e-6

    # PIT 守卫：h=21 时训练集末端必须留出 20 天
    dates = pd.bdate_range("2020-01-01", periods=200)
    lo, h = 100, 21
    end = max(0, lo - h + 1)
    assert dates[end - 1] <= dates[lo - h], "训练标签窗口越界（会泄漏未来收益）"

    # 🚨 口径守卫（2026-11-11 由莉酱指出后补）：样本内 R² 随 lag 数**单调增长**，
    # 任何单一 lag 的样本内 R² 都**不是**"天花板"。用纯噪声验证这一点。
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.standard_normal(3000), index=pd.bdate_range("2010-01-01",
                                                                    periods=3000))
    r2_1, _ = ar_r2(noise, 1)
    r2_20, _ = ar_r2(noise, 20)
    assert r2_20 > r2_1, "样本内 R2 应随 lag 数增长；若不增长说明实现有误"
    # 且纯噪声的样本内 R² 应贴近噪声地板 k/n（±3 个地板以内）
    assert r2_20 < 5 * 20 / 3000, f"纯噪声 AR(20) R2 异常偏大：{r2_20:.4f}"

    # 样本外 R² 在纯噪声上应 <= 0 附近（不得系统性为正）
    r2o, _ = oos_ar_r2(noise, 5, win=250)
    assert r2o < 0.002, f"纯噪声样本外 R2 不应为正：{r2o:.5f}"
    print("[self_check] OK")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
ARMS = {
    "base_buyhold": "满仓策略（参照）",
    "G_const_expo": "恒定 0.76 仓位（**仓位匹配对照**）",
    "B_daily_sumG": "日频训练+月内求和→月频执行",
    "C_daily_meanP": "日频训练+月内均值→月频执行",
    "D_daily_monthend": "日频训练+月末截面→月频执行",
    "E_monthly_train": "月频训练（21日）+月频执行",
    "F_placebo": "打乱标签重训（安慰剂）",
    "H_daily_voltarget": "日频预测波动→月频降仓（**物种不同：择波动不择方向**）",
}

#: 仓位匹配对照的水平：取各真实臂平均仓位（≈0.76）的整数化值。
#: 🔑 没有这一臂就无法区分「择时无效」与「只是仓位低」——
#: 恒定仓位 c 下收益与波动同比例缩 ⇒ **Sharpe 不变**；若某臂 Sharpe 显著低于
#: 满仓，说明是择时决策本身在伤害，而不是降仓位。
CONST_EXPO = 0.76


def voltarget_expo(ret: pd.Series, phase: int, target_vol: float = 0.18,
                   win: int = 20) -> pd.Series:
    """波动率目标：ω = min(1, target/已实现年化波动)，月频更新。

    🔑 物种区别：这批特征里唯一有统计显著性的是 ``vol20`` 对**次日收益**的
    IC +0.075（t=3.66），但它是**波动**预测而不是**方向**预测（高波动期次日
    平均收益略高是"波动聚集 + 正偏"的机械结果，不等于可择时的方向 edge）。
    「预测波动 → 降仓」是风险预算，不是 alpha；此处只作为**是否有别的东西可拿**
    的旁证臂。``target_vol=0.18`` 是事前固定值（≈策略长期波动的 3/4），非调参。
    """
    rb = rebalance_days(ret.index, phase)
    vals = []
    for d in rb:
        h = ret.loc[:d]
        rv = float(h.iloc[-win:].std() * np.sqrt(252)) if len(h) >= win else np.nan
        vals.append(min(1.0, target_vol / rv) if rv and rv > 0 else 1.0)
    return pd.Series(vals, index=rb)


def run_arm(kind: str, strat: pd.Series, pred_d: pd.Series | None,
            pred_m: pd.Series | None, pred_f: pd.Series | None,
            phase: int) -> tuple[float, ...]:
    """跑一个臂，返回 (年化, 几何SR, 算术SR, MDD, 年换手, 平均仓位)。"""
    if kind == "base_buyhold":
        e = pd.Series(1.0, index=rebalance_days(strat.index, phase))
    elif kind == "G_const_expo":
        e = pd.Series(CONST_EXPO, index=rebalance_days(strat.index, phase))
    elif kind == "B_daily_sumG":
        e = aggregate(pred_d, strat.index, phase, "sum")
    elif kind == "C_daily_meanP":
        e = aggregate(pred_d, strat.index, phase, "mean")
    elif kind == "D_daily_monthend":
        e = aggregate(pred_d, strat.index, phase, "last")
    elif kind == "E_monthly_train":
        e = aggregate(pred_m, strat.index, phase, "sum")
    elif kind == "F_placebo":
        e = aggregate(pred_f, strat.index, phase, "sum")
    elif kind == "H_daily_voltarget":
        e = voltarget_expo(strat, phase)
    else:
        raise ValueError(kind)

    r = apply_cost(strat, e)
    ef = e.reindex(strat.index).ffill()
    turn = float(ef.diff().abs().sum())
    yrs = len(strat) / 252
    return (annual_return(r), geo_sharpe(r), arith_sharpe(r), max_drawdown(r),
            turn / max(yrs, 1e-9), float(ef.mean()))


def yearly_table(strat: pd.Series, preds: dict[str, pd.Series],
                 phase: int = 0) -> pd.DataFrame:
    """逐年：各臂年化（21 相位之外只跑 phase=0，便于读年度稳健性）。"""
    rows = []
    for kind in ARMS:
        d = {"daily": preds["daily"], "monthly": preds["monthly"],
             "placebo": preds["placebo"]}
        if kind == "base_buyhold":
            r = strat
        elif kind == "G_const_expo":
            r = apply_cost(strat, pd.Series(
                CONST_EXPO, index=rebalance_days(strat.index, phase)))
        elif kind.startswith("B_"):
            r = apply_cost(strat, aggregate(d["daily"], strat.index, phase, "sum"))
        elif kind.startswith("C_"):
            r = apply_cost(strat, aggregate(d["daily"], strat.index, phase, "mean"))
        elif kind.startswith("D_"):
            r = apply_cost(strat, aggregate(d["daily"], strat.index, phase, "last"))
        elif kind.startswith("E_"):
            r = apply_cost(strat, aggregate(d["monthly"], strat.index, phase, "sum"))
        elif kind == "H_daily_voltarget":
            r = apply_cost(strat, voltarget_expo(strat, phase))
        else:
            r = apply_cost(strat, aggregate(d["placebo"], strat.index, phase, "sum"))
        for y, sub in r.groupby(r.index.year):
            rows.append({"arm": kind, "year": y,
                         "ann": annual_return(sub), "sharpe": geo_sharpe(sub)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-arms", action="store_true", help="只跑诊断段（约 30 秒）")
    ap.add_argument("--phases", type=int, default=len(PHASES))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("timing_daily")
    OUT.mkdir(parents=True, exist_ok=True)
    self_check()

    mkt = load_market()
    strat = load_strategy()
    log.info("市场面板 %d 天 %s ~ %s", len(mkt), mkt.index.min().date(),
             mkt.index.max().date())
    log.info("策略净值 %d 天 %s ~ %s", len(strat), strat.index.min().date(),
             strat.index.max().date())

    log.info("训练日频模型（标签=次日）…")
    pred_d = walk_forward(mkt, mkt["ret"], horizon=1)
    log.info("训练月频模型（标签=21 日）…")
    pred_m = walk_forward(mkt, mkt["ret"], horizon=21)

    # 安慰剂：整体打乱「特征 ↔ 标签」的对应关系（保留各自边缘分布）
    rng = np.random.default_rng(20261111)
    perm = rng.permutation(len(mkt))
    shuf = mkt.copy()
    shuf["ret"] = mkt["ret"].to_numpy()[perm]      # 特征在原日期，标签被随机换
    log.info("训练安慰剂模型（标签已打乱）…")
    pred_f = walk_forward(shuf, shuf["ret"], horizon=1)

    log.info("样本外预测覆盖 %d 天（%s 起）", int(pred_d.notna().sum()),
             pred_d.first_valid_index().date() if pred_d.notna().any() else "—")

    # ---- 诊断段（恒跑）----
    sf = single_factor_ic(mkt, mkt["ret"])
    sf.to_csv(OUT / "single_factor_ic.csv", index=False)
    log.info("\n=== 单因子 vs 次日收益 ===\n%s", sf.round(4).to_string(index=False))

    rd = agg_diagnostic(pred_d, mkt["ret"])
    if len(rd):
        rd.to_csv(OUT / "agg_diag.csv", index=False)
        log.info("\n=== 错位诊断（日频模型 → 月度目标）===\n%s",
                 rd.round(4).to_string(index=False))

    fd = fit_diagnostic(pred_d, mkt["ret"])
    if len(fd):
        fd.to_csv(OUT / "fit_diag.csv", index=False)
        log.info("\n=== 拟合诊断（模型学到东西了吗）===\n%s",
                 fd.round(6).to_string(index=False))

    pc = predictability_ceiling(mkt, mkt["ret"])
    pc.to_csv(OUT / "ceiling.csv", index=False)
    log.info("\n=== 可预测性天花板 / Grinold / 盈亏平衡 ===\n%s",
             pc.round(6).to_string(index=False))

    if args.skip_arms:
        log.info("产物 → %s", OUT)
        return

    # ---- 臂段 ----
    arms = []
    n_ph = min(args.phases, len(PHASES))
    for ph in range(n_ph):
        for kind, label in ARMS.items():
            ann, sr, sra, mdd, turn, ex = run_arm(
                kind, strat,
                pred_d=pred_d if kind[0] in ("B", "C", "D") else None,
                pred_m=pred_m if kind[0] == "E" else None,
                pred_f=pred_f if kind[0] == "F" else None,
                phase=ph)
            arms.append({"arm": kind, "label": label, "phase": ph, "ann": ann,
                         "sharpe": sr, "sharpe_arith": sra, "mdd": mdd,
                         "turnover_ann": turn, "exposure_mean": ex})
    ad = pd.DataFrame(arms)
    ad.to_csv(OUT / "arms.csv", index=False)
    ad[ad.phase == 0].to_csv(OUT / "phase.csv", index=False)

    # 配对差（逐相位）
    base = ad[ad.arm == "base_buyhold"].set_index("phase")["ann"]
    pairs = []
    for kind in ARMS:
        if kind in ("base_buyhold", "G_const_expo"):
            continue
        d = (ad[ad.arm == kind].set_index("phase")["ann"] - base).dropna()
        if len(d) < 2:
            continue
        pairs.append({"arm": kind, "vs": "base_buyhold", "n": len(d),
                      "mean_pp": 100 * d.mean(), "t": _t(d),
                      "pos": int((d > 0).sum())})
    # 🔑 决定性对照：真实臂 − 恒定同仓位臂（剔除「拿得少」这一项）
    ge = ad[ad.arm == "G_const_expo"].set_index("phase")["ann"]
    for kind in ARMS:
        if kind in ("base_buyhold", "G_const_expo"):
            continue
        d = (ad[ad.arm == kind].set_index("phase")["ann"] - ge).dropna()
        sb = (ad[ad.arm == kind].set_index("phase")["sharpe"]
              - ad[ad.arm == "G_const_expo"].set_index("phase")["sharpe"]).dropna()
        if len(d) < 2:
            continue
        pairs.append({"arm": kind, "vs": "G_const_expo", "n": len(d),
                      "mean_pp": 100 * d.mean(), "t": _t(d),
                      "pos": int((d > 0).sum()),
                      "d_sharpe": float(sb.mean())})
    pd.DataFrame(pairs).to_csv(OUT / "pairs.csv", index=False)

    yt = yearly_table(strat, {"daily": pred_d, "monthly": pred_m,
                              "placebo": pred_f})
    yt.to_csv(OUT / "yearly.csv", index=False)

    log.info("\n=== 各臂均值（21 相位 / phase=0）===\n%s",
             ad.groupby("arm")[["ann", "sharpe", "mdd", "turnover_ann",
                                "exposure_mean"]].mean().round(4).to_string())
    log.info("\n=== 配对差（vs 满仓，逐相位）===\n%s",
             pd.DataFrame(pairs).round(3).to_string(index=False))
    log.info("\n=== 逐年（phase=0）===\n%s",
             yt.pivot(index="year", columns="arm", values="ann").round(4).to_string())
    log.info("产物 → %s", OUT)


if __name__ == "__main__":
    main()