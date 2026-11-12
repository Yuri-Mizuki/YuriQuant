"""SOP 收益分解择时（P1）：文献唯一通过多重检验的方法，本地化测试。

动机（2026-11-11 由莉酱「跑 P1」）
----------------------------------
外部文献（``reports/timing_lit_review/README.md`` §二）：

* **Ferreira & Santa-Clara (2011), JFE 100(3) 514-537, "The sum of the parts is
  more than the whole"**：把市场收益拆成 ``D/P + 盈利增速 + P/E 增速`` 分别预测
  ——D/P 用当期值（高度持续）、盈利增速用 20 年均值（低频可预测）、P/E 增速设零。
  月度 **OOS R² 1.32%**、年度 13.43%（传统预测回归 −1.78%~0.69%），Sharpe 增益 0.3。
* **Dichtl, Drobetz, Neuhierl & Wendt (2021), IJF 37(1) 72-94**：140 个策略、
  1966-2018、多重检验校正后**几乎全灭**，**唯一稳健显著存活的就是 SOP**。

🔑 **与本项目的本质区别**：我们此前测的都是**水平分解**（AR 自相关、动量、波动、
newsy 月条件化）——即「从收益序列自身找结构」。SOP 是**结构分解**：把收益按
会计恒等式拆成三块，每块单独预测。这是**换靶子，不是换模型**。

与本项目门槛的对照（关键）
--------------------------
本项目月频盈亏平衡门槛 **R² = 0.00267**（``timing_conditioned_probe.breakeven_r2``，
由 11.6 次换手 × 36bp / 波动 23.34% / 广度 12 反解）。SOP 文献 1.32% = 门槛的
**4.9 倍** ⇒ 它是唯一「理论上能过」的候选。

⚠️ **但必须挡在两处**（否则又是安慰剂）：
1. **样本量**：本项目 ``prod`` 窗口只有 **105 个月**。即便真有 1.3%，105 个观测
   的统计功效也未必测得出 —— 这是「样本量」说法的**正确用法**（限制可验证性，
   不限制可预测性）。
2. **口径移植**：文献用 1927-2007 美股、S&P500、含回购。A 股样本只有 2016-07 起
   的全A，**不能直接套用 1.32% 这个数**，必须本地实测。

实现口径（严格 PIT）
--------------------
恒等式（对数近似）：``r ≈ D/P + ΔE/E + Δ(PE)/PE``

* ``D/P``：**过去 365 天已实施分红 / 当日全A总市值**（用 ``ex_date <= t`` 加总，
  不用公告日 —— 分红实施是可观测事件，比公告更保守）。
* ``ΔE/E``：净利润 TTM 同比增速，用 ``ann_date <= t`` 的单季净利润累加
  （**已做单季化**：累计报表差分）。
* ``Δ(PE)/PE``：**设零**（随机游走假设，SOP 最简版；文献明确这就是它的形式）。

⇒ ``predict = D/P(t) + g_E(t)``，**零自由参数**（斜率约束为 1，这是 SOP 低估计
误差的来源，也是它 OOS 为正的机制）。

臂
--
* ``S0_mean``        扩展窗历史均值（Goyal-Welch 基准）
* ``S1_dp``          只用 D/P（当期股息率）
* ``S2_sop``         D/P + 盈利增速（**SOP 完整版**）
* ``S3_dp_placebo``  D/P 但把「分红序列」循环平移 12 个月（安慰剂）

用法
----
    python -m scripts.evaluation.sop_timing_probe
    python -m scripts.evaluation.sop_timing_probe --skip-arms

产物
----
    reports/sop_timing/{components,oos,arms,pairs,yearly,data_audit}.csv
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

OUT = ROOT / "reports" / "sop_timing"
BASE = ROOT / "reports" / "alla_rolling" / "_base"
EQUITY = (ROOT / "reports" / "alla_rolling_ortho" / "equity"
          / "eq__ens_h1h5__h1__M__raw__f0.10.csv")
DATA = Path(Config.paths()["cache_root"])

#: 盈利增速的长期均值窗（月）。文献用 20 年 = 240；A 股样本只有 ~120 月，
#: 因此同时报 ``42``（3.5 年，可执行的最短窗）与 ``EXPANDING``（扩展均值）。
EARN_WIN = 42
#: 扩展窗最少训练月数（与 ``timing_conditioned_probe`` 一致）。
MIN_TRAIN = 36
#: 仓位切换成本（单边 bp）与月频换手锚点（本项目实测，非假设）。
POSITION_TURNOVER_BP = 10.0
COST_BP = 36.0
TURNOVER_MONTHLY = 11.6
PHASES = tuple(range(6))

ARMS = {
    "S0_mean": "扩展窗历史均值（基准）",
    "S1_dp": "只用当期股息率 D/P",
    "S2_sop": "SOP 完整版：D/P + 盈利增速",
    "S3_dp_placebo": "D/P + 分红序列循环平移12月（安慰剂）",
}


# ---------------------------------------------------------------------------
# 数据构造（全部 PIT）
# ---------------------------------------------------------------------------
def load_mcap() -> pd.Series:
    """全A 总市值（元/日）。"""
    mc = pd.read_parquet(BASE / "market_cap.parquet")
    return mc.sum(axis=1).rename("mcap")


def load_div_amount() -> pd.Series:
    """按 ``ex_date`` 归集的现金分红总额（元）。

    ⚠️ ``base_share`` 单位是**万股** ⇒ 必须 ``×1e4``。初版漏乘，年度分红总额
    算成 0.0 万亿（实为 2.4 万亿），2026-11-11 已修。
    """
    dv = pd.read_parquet(DATA / "dividend.parquet")
    d = dv[dv["cash_per_share_pre_tax"] > 0].dropna(
        subset=["ex_date", "base_share"]).copy()
    d["amt"] = d["cash_per_share_pre_tax"] * d["base_share"] * 1e4
    return d.groupby("ex_date")["amt"].sum().sort_index().rename("div_amt")


def load_earnings_ttm() -> pd.Series:
    """净利润 TTM（元），按 ``ann_date`` 归集、**单季化后**滚动 365 天累加。

    ⚠️ 两个坑：
    1. 累计报表（Q1/H1/Q3/FY）必须差分转单季，否则 TTM 重复计入（实测会把
       净利虚增到 5.5 万亿 vs 年报 4.65 万亿）。
    2. 同一 ``(code, report_period)`` 有修订版 ⇒ 取 ``ann_date`` 最晚的一条。
    """
    i = pd.read_parquet(DATA / "income.parquet",
                        columns=["code", "ann_date", "report_period",
                                 "statement_type", "NET_PRO_EXCL_MIN_INT_INC"])
    i = i[i["statement_type"] == "1"]
    i = i.dropna(subset=["ann_date", "report_period", "NET_PRO_EXCL_MIN_INT_INC"])
    i = i[i["report_period"].dt.month.isin([3, 6, 9, 12])]
    i = (i.sort_values("ann_date")
          .drop_duplicates(["code", "report_period"], keep="last")
          .sort_values(["code", "report_period"]))
    # 单季化：同一 code 的累计值差分；Q1 本身就是单季
    i["single"] = i.groupby("code")["NET_PRO_EXCL_MIN_INT_INC"].diff()
    q1 = i["report_period"].dt.month == 3
    i.loc[q1, "single"] = i.loc[q1, "NET_PRO_EXCL_MIN_INT_INC"]
    g = i.dropna(subset=["single"]).groupby("ann_date")["single"].sum().sort_index()
    return g.rolling("365D").sum().rename("earn_ttm")


def build_components(mcap: pd.Series, div_amt: pd.Series,
                     earn_ttm: pd.Series, dates: pd.DatetimeIndex,
                     shift_div_months: int = 0) -> pd.DataFrame:
    """在给定日历上对齐三个分量（全部 PIT：只用 ``<= t`` 的信息）。

    ``shift_div_months``：安慰剂用 —— 把分红序列整体循环平移，破坏「当期股息率」
    与未来收益的机制联系，但保留分红的边际分布与季节结构。
    """
    if shift_div_months:
        off = pd.DateOffset(months=shift_div_months)
        div_amt = div_amt.copy()
        div_amt.index = div_amt.index + off

    rows = []
    for t in dates:
        den = mcap.asof(t)
        if not np.isfinite(den) or den <= 0:
            continue
        win = div_amt[(div_amt.index <= t)
                      & (div_amt.index > t - pd.Timedelta(days=365))]
        dp = float(win.sum()) / float(den)

        e = earn_ttm.asof(t)
        rows.append({"date": t, "dp": dp, "earn_ttm": e, "mcap": float(den)})

    df = pd.DataFrame(rows).set_index("date")
    df["ep"] = df["earn_ttm"] / df["mcap"]      # 盈利收益率（E/P）
    # SOP 的盈利增速分量：用 E/P 的**同比变化**近似 ΔE/E
    # （在 P/E 不变 ⇒ r ≈ D/P + ΔE/E 的恒等式下等价）
    df["g_earn"] = df["ep"].pct_change(12)
    df["g_earn_ma"] = df["g_earn"].rolling(EARN_WIN, min_periods=24).mean()
    # 🚨 量纲换算（2026-11-11 修）：``dp`` 与 ``g_earn`` 都是**年化**量
    # （D/P = 过去 365 天分红 / 市值；g = 同比增速），而 ``ret`` 是**月度**收益。
    # SOP 恒等式 ``R_annual = D/P + ΔE/E + ΔPE/PE`` 必须 ``/12`` 才是月度预测。
    # 初版漏了这一步 ⇒ 预测值被放大 12 倍（dp≈0.020 当月度 2%），
    # 反而"碰巧"赢了基准（因为 A 股月均收益≈0.9% < 2%），这是量纲 bug 造的假阳性。
    df["dp_m"] = df["dp"] / 12.0
    df["g_earn_ma_m"] = df["g_earn_ma"] / 12.0
    return df


# ---------------------------------------------------------------------------
# 评估（与 timing_conditioned_probe 同口径，便于横向比较）
# ---------------------------------------------------------------------------
def expanding_mean(y: pd.Series, min_periods: int = 12) -> pd.Series:
    """扩展窗历史均值（PIT：``t`` 只用 ``[0, t)``）= Goyal-Welch 基准。"""
    return y.shift(1).expanding(min_periods=min_periods).mean()


def oos_r2(pred: pd.Series, y: pd.Series, base: pd.Series) -> dict:
    """样本外 R²（相对扩展窗均值），附 MSE 差 t 值与方向命中。"""
    d = pd.concat([pred.rename("p"), y.rename("y"), base.rename("b")], axis=1).dropna()
    if len(d) < 24:
        return {"n": len(d), "oos_r2": float("nan"), "t_mse": float("nan"),
                "hit": float("nan"), "hit_all_long": float("nan"), "ic": float("nan")}
    sse_m = float(((d.y - d.p) ** 2).sum())
    sse_b = float(((d.y - d.b) ** 2).sum())
    diff = (d.y - d.b) ** 2 - (d.y - d.p) ** 2
    t_mse = (float(diff.mean() / diff.std() * np.sqrt(len(diff)))
             if diff.std() > 0 else float("nan"))
    return {"n": len(d),
            "oos_r2": 1 - sse_m / sse_b if sse_b > 0 else float("nan"),
            "t_mse": t_mse,
            "hit": float((np.sign(d.p) == np.sign(d.y)).mean()),
            "hit_all_long": float((d.y > 0).mean()),
            "ic": float(d.p.corr(d.y, method="spearman"))}


def breakeven_r2(vol_ann: float) -> dict:
    """本项目月频盈亏平衡门槛（成本/波动/广度反解）。"""
    cost = TURNOVER_MONTHLY * COST_BP / 1e4
    ir = cost / vol_ann
    ic = ir / np.sqrt(12.0)
    return {"cost_ann": cost, "ir_needed": ir, "ic_needed": ic, "r2_needed": ic ** 2}


# ---------------------------------------------------------------------------
# 策略层
# ---------------------------------------------------------------------------
def rebalance_days(dates: pd.DatetimeIndex, phase: int = 0) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    out = [blk.iloc[phase] for _, blk in s.groupby(s.index.to_period("M"))
           if len(blk) > phase]
    return pd.DatetimeIndex(out)


def to_monthly(ret: pd.Series, phase: int = 0) -> pd.Series:
    rb = rebalance_days(ret.index, phase)
    vals, idx = [], []
    for i in range(1, len(rb)):
        seg = ret.loc[rb[i - 1]:rb[i]]
        if len(seg) == 0:
            continue
        vals.append(float((1 + seg).prod() - 1))
        idx.append(rb[i])
    return pd.Series(vals, index=pd.DatetimeIndex(idx)).rename("ret")


def to_expo(pred: pd.Series, dates: pd.DatetimeIndex,
            lo: float = 0.5, hi: float = 1.0) -> pd.Series:
    """月度预测 → ω ∈ {lo, hi}，滚动 24 期中位（PIT）。与 P0 同规则。"""
    q = pred.rolling(24, min_periods=12).apply(
        lambda w: (w[-1] > np.nanmedian(w[:-1])).astype(float), raw=True)
    e = pd.Series(np.where(q > 0.5, hi, lo), index=pred.index)
    return e.where(q.notna(), hi).reindex(dates)


def apply_cost(base: pd.Series, expo: pd.Series,
               bp: float = POSITION_TURNOVER_BP) -> pd.Series:
    e = expo.reindex(base.index).ffill().fillna(1.0)
    return base * e - e.diff().abs().fillna(0.0) * bp / 1e4


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
def dim_check(comp: pd.DataFrame) -> None:
    """量纲守卫（**必须**在建好 ``comp`` 之后跑）。

    🚨 初版把**年化**的 D/P 当**月度**预测用（放大约 12 倍）。在 A 股
    「月均收益≈0.9% < D/P≈2%」的巧合下，它"赢"了扩展窗均值基准，
    OOS R² 显示 +0.0337、看似过门槛 —— 纯属量纲 bug 造的假阳性。
    对照证据：安慰剂臂（分红整体平移 12 月）给出 +0.0348，与真实臂几乎相同
    ⇒ 那个"信号"来自**水平锚定**而非当期分红机制。

    判据：``|predict|`` 的中位数必须 < 3 倍 ``|月收益|`` 的中位数。
    """
    y_m = comp["fwd"].abs().median()
    for col in ("dp_m", "g_earn_ma_m"):
        v = comp[col].abs().median()
        assert np.isfinite(v) and v < 3 * y_m, (
            f"{col} 量纲异常：中位 {v:.4f} vs 月收益中位 {y_m:.4f}（疑年化未转月度）")


def self_check(mcap: pd.Series, div_amt: pd.Series,
               earn_ttm: pd.Series) -> None:
    """跑前守卫：三者都必须通过，否则结论不可信。"""
    log = logging.getLogger("sop")

    # 1) 分红量级（元）：年度应在 0.8~3.5 万亿
    yearly = div_amt.groupby(div_amt.index.year).sum()
    recent = yearly.loc[2018:]
    assert recent.min() > 0.5e12, f"分红总额偏低：{recent.min() / 1e12:.3f} 万亿"
    assert recent.max() < 5.0e12, f"分红总额偏高：{recent.max() / 1e12:.3f} 万亿"

    # 2) 净利润 TTM 不得远超年报合计（虚增 ⇒ 单季化有 bug）
    ann = earn_ttm.resample("YE").last().loc["2019":"2024"] / 1e12
    assert ann.max() < 8.0, f"净利 TTM 虚增：{ann.max():.2f} 万亿（疑未单季化）"

    # 3) 股息率水平：0.5%~5%
    dp = []
    for t in pd.date_range("2019-01-31", "2025-12-31", freq="YE"):
        den = mcap.asof(t)
        w = div_amt[(div_amt.index <= t)
                    & (div_amt.index > t - pd.Timedelta(days=365))].sum()
        if np.isfinite(den) and den > 0:
            dp.append(float(w) / float(den))
    dp = np.array(dp)
    assert (dp > 0.005).all() and (dp < 0.05).all(), (
        f"股息率超出合理区间：{dp.min():.4f}~{dp.max():.4f}")

    # 4) 基准自洽：扩展均值对自身 OOS R² 必须 = 0
    y = pd.Series(np.random.default_rng(5).standard_normal(200) * 0.05,
                  index=pd.date_range("2000-01-31", periods=200, freq="ME"))
    b = expanding_mean(y, min_periods=24)
    d = pd.concat([b.rename("p"), y.rename("y"), b.rename("b")], axis=1).dropna()
    assert abs(float(((d.y - d.b) ** 2).sum() - ((d.y - d.p) ** 2).sum())) < 1e-12

    log.info("self_check 通过（分红 %.2f 万亿 / 净利TTM 峰值 %.2f 万亿 / 股息率 %.2f%%~%.2f%%）",
             recent.iloc[-1] / 1e12, ann.max(), 100 * dp.min(), 100 * dp.max())


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-arms", action="store_true", help="只跑诊断段")
    ap.add_argument("--phases", type=int, default=len(PHASES))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("sop")
    OUT.mkdir(parents=True, exist_ok=True)

    mcap = load_mcap()
    div_amt = load_div_amount()
    earn_ttm = load_earnings_ttm()
    strat = pd.read_csv(EQUITY, index_col=0, parse_dates=True)["daily_ret"].dropna()
    self_check(mcap, div_amt, earn_ttm)

    close = pd.read_parquet(BASE / "close_raw.parquet").where(lambda x: x > 0)
    ew = (close.pct_change(fill_method=None).mean(axis=1)
          .replace([np.inf, -np.inf], np.nan).dropna())

    # ---- 数据审计 ----
    audit = pd.DataFrame([
        {"item": "全A总市值 起/止", "value": f"{mcap.index.min().date()} ~ "
                                            f"{mcap.index.max().date()}"},
        {"item": "最早可算 D/P 的日期", "value": str(div_amt.index.min().date())},
        {"item": "最早可算 E/P 的日期", "value": str(earn_ttm.first_valid_index())},
        {"item": "分红记录数(>0)", "value": len(div_amt)},
        {"item": "净利 TTM 峰值(万亿)",
         "value": round(float(earn_ttm.max()) / 1e12, 3)},
        {"item": "月频样本上限(月)", "value": len(to_monthly(ew))},
        {"item": "策略净值 起/止", "value": f"{strat.index.min().date()} ~ "
                                          f"{strat.index.max().date()}"},
    ])
    audit.to_csv(OUT / "data_audit.csv", index=False)
    log.info("\n=== 数据可得性审计 ===\n%s", audit.to_string(index=False))

    # ---- 分量面板 ----
    mret = to_monthly(ew, 0)
    comp = build_components(mcap, div_amt, earn_ttm, mret.index)
    comp = comp.join(mret.rename("ret"), how="inner")
    comp["fwd"] = comp["ret"].shift(-1)
    dim_check(comp)          # 🚨 量纲守卫必须在 OOS 之前
    comp.to_csv(OUT / "components.csv")
    log.info("\n=== SOP 分量（年末）===\n%s",
             comp[["dp", "ep", "g_earn", "g_earn_ma"]]
             .resample("YE").last().round(4).to_string())

    # ---- 诊断段：OOS R² ----
    rows = []
    base = expanding_mean(comp["fwd"], min_periods=MIN_TRAIN)
    specs = {
        "S0_mean": base,
        "S1_dp": comp["dp_m"],
        "S2_sop": comp["dp_m"] + comp["g_earn_ma_m"],
        "S3_dp_placebo": None,      # 在下面用平移后的分红重算
    }
    # 安慰剂：分红整体后移 12 个月（破坏当期对应关系）
    dshift = div_amt.copy()
    dshift.index = dshift.index + pd.DateOffset(months=12)
    comp_p = build_components(mcap, dshift, earn_ttm, mret.index)
    specs["S3_dp_placebo"] = (comp_p["dp"] / 12.0).reindex(comp.index)

    for kind, p in specs.items():
        r = oos_r2(p.reindex(comp.index), comp["fwd"], base)
        rows.append({"arm": kind, "label": ARMS[kind], **r})
    oo = pd.DataFrame(rows)
    oo.to_csv(OUT / "oos.csv", index=False)
    log.info("\n=== 样本外 R²（月度，相对扩展窗均值）===\n%s",
             oo[["arm", "n", "oos_r2", "t_mse", "hit", "hit_all_long", "ic"]]
             .round(4).to_string(index=False))

    be = breakeven_r2(float(ew.std() * np.sqrt(252)))
    log.info("\n=== 盈亏平衡门槛 ===\n%s",
             pd.Series({k: round(v, 5) for k, v in be.items()}).to_string())
    best = float(oo["oos_r2"].max())
    log.info("最强臂 OOS R² %.5f vs 门槛 %.5f ⇒ %s", best, be["r2_needed"],
             "过门槛" if best >= be["r2_needed"] else "未过门槛")

    if args.skip_arms:
        log.info("产物 → %s", OUT)
        return

    # ---- 臂段 ----
    n_ph = min(args.phases, len(PHASES))
    arms, pairs = [], []
    for ph in range(n_ph):
        m = to_monthly(ew, ph)
        c = build_components(mcap, div_amt, earn_ttm, m.index)
        cp = build_components(mcap, dshift, earn_ttm, m.index)
        sig = {"S0_mean": c["dp"].shift(1).expanding(min_periods=MIN_TRAIN).mean(),
               "S1_dp": c["dp_m"],
               "S2_sop": c["dp_m"] + c["g_earn_ma_m"],
               "S3_dp_placebo": (cp["dp"] / 12.0).reindex(c.index)}
        rb = rebalance_days(strat.index, ph)
        for kind in ARMS:
            e = to_expo(sig[kind].dropna(), rb)
            r = apply_cost(strat, e)
            arms.append({"arm": kind, "phase": ph, "ann": annual_return(r),
                         "sharpe": geo_sharpe(r), "mdd": max_drawdown(r),
                         "exposure_mean": float(e.reindex(strat.index)
                                                .ffill().mean())})
        ex2 = float(to_expo(sig["S2_sop"].dropna(), rb).reindex(strat.index)
                    .ffill().mean())
        for kind, ex in (("base_buyhold", 1.0), ("G_const_expo", ex2)):
            r = apply_cost(strat, pd.Series(ex, index=rb))
            arms.append({"arm": kind, "phase": ph, "ann": annual_return(r),
                         "sharpe": geo_sharpe(r), "mdd": max_drawdown(r),
                         "exposure_mean": ex})

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
    pd.DataFrame(pairs).to_csv(OUT / "pairs.csv", index=False)

    log.info("\n=== 各臂（%d 相位均值）===\n%s", n_ph,
             ad.groupby("arm")[["ann", "sharpe", "mdd", "exposure_mean"]]
             .mean().round(4).to_string())
    log.info("\n=== 配对差（逐相位）===\n%s",
             pd.DataFrame(pairs).round(3).to_string(index=False))

    yrs = []
    m = to_monthly(ew, 0)
    c = build_components(mcap, div_amt, earn_ttm, m.index)
    cp = build_components(mcap, dshift, earn_ttm, m.index)
    sig = {"S0_mean": c["dp"].shift(1).expanding(min_periods=MIN_TRAIN).mean(),
           "S1_dp": c["dp_m"], "S2_sop": c["dp_m"] + c["g_earn_ma_m"],
           "S3_dp_placebo": (cp["dp"] / 12.0).reindex(c.index)}
    rb = rebalance_days(strat.index, 0)
    for kind in ARMS:
        r = apply_cost(strat, to_expo(sig[kind].dropna(), rb))
        for y, sub in r.groupby(r.index.year):
            yrs.append({"arm": kind, "year": y, "ann": annual_return(sub)})
    yt = pd.DataFrame(yrs)
    yt.to_csv(OUT / "yearly.csv", index=False)
    log.info("\n=== 逐年（phase=0）===\n%s",
             yt.pivot(index="year", columns="arm", values="ann").round(4).to_string())
    log.info("产物 → %s", OUT)


if __name__ == "__main__":
    main()