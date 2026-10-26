"""模型自诊断仓位探针（"信号自身状态 → 仓位乘数"）。

为什么需要这个脚本
------------------
项目已把**市场侧**择时全部测穿（`reports/timing_probe/`：7 条因果规则 7/7 Δ年化负、
月度价格因子 0/7、资金面 0/4），也已把 regime 条件化测穿（`reports/regime_probe/`：
两道 kill 点全触发）。剩下唯一**未被证伪**的仓位信息集是 —— **模型自身的状态**：

    「这一期我的信号有多可信？」  而不是  「这一期大盘会涨还是会跌？」

两者的根本差别是**信息集**：市场侧用价格/资金/情绪序列（时序、每月 1 个观测、
实测 ≈ 0）；模型侧用**每日 5000 只横截面就能估出来的量**（滚动 IC、成员分歧、
预测离散度），信息密度高两个数量级，但**能否预测下一期的组合同样收益**仍是
纯时序问题，因此必须用同一把尺子（月度 rank IC + t + 四窗口 + 安慰剂）审。

本脚本一次性回答三件事（顺序不可颠倒，与 timing_probe 的方法论一致）：
1. **上界在哪**：完美预知下月组合收益/下月 IC 的仓位乘数值多少钱。若上界本身
   很薄，真实规则不必跑。
2. **状态量有没有预测力**：4 类模型自诊断状态（ic60 / icir60 / 分歧 / 预测强度）
   对**下月组合收益**的月度 rank IC、t 值、五分位单调性。
3. **因果规则实测**：把通过的状态做成月频仓位乘数 ω_t（含 10bp/单位切换成本），
   报 Δ年化 / ΔSharpe / MDD / 四窗口一致性，并用**循环移位安慰剂**证伪。

口径（🚨 与项目既有教训一致）
------------------------------
- 基线 = **登记表 ``v1w-ortho882``**（882 ortho ens_h1h5 + buffer(15/40)，production
  profile：固定 10bp / T+1 open / 1e6），由 ``caliber.run_once`` 现跑，锚点校验
  ann/Sharpe 必须与登记值一致（16.39% / 0.681，容差 5bp/0.005）。
- **所有状态量一律 ``shift(1)``**：t 日收盘才可知的 IC/分歧，只用于 t+1 及以后。
  月频 ω 取**上月末状态**作用于**当月**（``reindex(method="ffill")``），零前视。
- **仓位切换成本 = |Δω| × 10bp**（与 timing_probe / 生产同量级）。
- **Sharpe 用几何口径**（``backtest.metrics.sharpe_ratio``），与 caliber 同源。
- 判据（项目唯一口径）：**Δ年化为正** 才算改进；四窗口一致性 ≥4/5 为主要抵消证据；
  安慰剂 p ≥ 0.05 一律视为噪声。

用法
----
    python -m scripts.evaluation.model_state_probe --self-check
    python -m scripts.evaluation.model_state_probe

产物
----
    reports/model_state_probe/{state_daily, state_monthly, predictability,
    arm_summary, placebo}.csv
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

OUT = ROOT / "reports" / "model_state_probe"
PRED_DIR = ROOT / "reports" / "alla_rolling_ortho" / "pred"

#: 仓位切换成本（单边，bp）：ω 变动 1.0 单位 = 满仓↔空仓。
POSITION_TURNOVER_BP = 10.0

#: 融资年化利率（只对 ω>1 的借入部分计息）。券商两融常见 4.5%~6%，
#: 取 4.5% 作乐观口径；另报 0% 的上界臂以便看清信息本身值多少。
FUNDING_ANNUAL = 0.045

#: 每日截面有效股票数下限（低于则当日 IC 跳过，与 regime_probe 同值）
MIN_NAMES = 500

#: 四窗口 + 全窗的一致性判据（与 timing_probe / cheap_lever_scan 同格式）。
CONSISTENCY_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2018+", "2018"),
    ("2019+", "2019"),
    ("2021+", "2021"),
    ("2023+", "2023"),
    ("2025+", "2025"),
)

#: 登记锚点（config/baselines.yaml · v1w-ortho882 · production · eqw_buf0.15_0.4）
ANCHOR_ANN = 0.1639
ANCHOR_SHARPE = 0.681


# ---------------------------------------------------------------------------
# 指标（几何口径，与 backtest/metrics / caliber 同源）
# ---------------------------------------------------------------------------
def annual_return(r: pd.Series) -> float:
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def annual_vol(r: pd.Series) -> float:
    return float(r.std() * np.sqrt(252))


def geo_sharpe(r: pd.Series) -> float:
    v = annual_vol(r)
    return annual_return(r) / v if v > 0 else float("nan")


def max_drawdown(r: pd.Series) -> float:
    c = (1 + r).cumprod()
    return float((c / c.cummax() - 1).min())


def _apply_position_cost(base: pd.Series, expo: pd.Series,
                         bp: float = POSITION_TURNOVER_BP,
                         funding: float = 0.0) -> pd.Series:
    """总仓位乘数 + 切换成本 + 融资成本：``r = r_base·ω − |Δω|·bp − (ω−1)⁺·融资``。

    ``funding`` 为年化融资利率（只在 ω>1 的月份计息），默认 0 表示给出
    **无融资成本的上界**；实际产品口径取 ``FUNDING_ANNUAL``。
    """
    expo = expo.fillna(1.0)
    turn = expo.diff().abs().fillna(0.0)
    borrow = expo.clip(lower=1.0) - 1.0
    return base * expo - turn * bp / 1e4 - borrow * funding / 252


# ---------------------------------------------------------------------------
# 状态量（全部 ``shift(1)``：t 日收盘可知 → t+1 生效）
# ---------------------------------------------------------------------------
def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """两个等长向量的 Spearman 相关（秩变换后 Pearson）。"""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def daily_rank_ic(sig: pd.DataFrame, fwd1: pd.DataFrame,
                  mask: pd.DataFrame) -> pd.Series:
    """日度截面 rank IC：``signal[t]`` vs ``t→t+1`` 收益（截面前视仅用于评分）。"""
    sv, rv, mv = (sig.to_numpy(dtype=float), fwd1.to_numpy(dtype=float),
                  mask.to_numpy(dtype=bool))
    out = np.full(len(sig), np.nan)
    for i in range(len(sig) - 1):
        ok = mv[i] & np.isfinite(sv[i]) & np.isfinite(rv[i])
        if ok.sum() > MIN_NAMES:
            out[i] = _rank_corr(sv[i][ok], rv[i][ok])
    return pd.Series(out, index=sig.index)


def daily_panel_corr(a: pd.DataFrame, b: pd.DataFrame,
                     mask: pd.DataFrame) -> pd.Series:
    """两个预测面板**截面秩相关**（成员一致性；低值 = 分歧大）。"""
    av, bv, mv = (a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                  mask.to_numpy(dtype=bool))
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        ok = mv[i] & np.isfinite(av[i]) & np.isfinite(bv[i])
        if ok.sum() > MIN_NAMES:
            out[i] = _rank_corr(av[i][ok], bv[i][ok])
    return pd.Series(out, index=a.index)


def daily_strength(raw: pd.DataFrame, mask: pd.DataFrame,
                   q: float = 0.9) -> pd.Series:
    """预测强度：当日最高分位均值 − 全截面均值（原始分数单位）。

    理论上月频组合超额 ≈ IC × 选股集中度 × 该强度 ⇒ 它是"选股赔率"的直接读数，
    与大盘方向无关。
    """
    rv, mv = raw.to_numpy(dtype=float), mask.to_numpy(dtype=bool)
    out = np.full(len(raw), np.nan)
    for i in range(len(raw)):
        ok = mv[i] & np.isfinite(rv[i])
        if ok.sum() > MIN_NAMES:
            s = rv[i][ok]
            out[i] = float(s[s >= np.quantile(s, q)].mean() - s.mean())
    return pd.Series(out, index=raw.index)


def daily_dispersion(raw: pd.DataFrame, mask: pd.DataFrame) -> pd.Series:
    """预测离散度：当日原始分数截面标准差（预测收益的横截面铺多宽）。"""
    rv, mv = raw.to_numpy(dtype=float), mask.to_numpy(dtype=bool)
    out = np.full(len(raw), np.nan)
    for i in range(len(raw)):
        ok = mv[i] & np.isfinite(rv[i])
        if ok.sum() > MIN_NAMES:
            out[i] = float(np.std(rv[i][ok]))
    return pd.Series(out, index=raw.index)


def build_states(ctx) -> pd.DataFrame:
    """四类模型自诊断状态（daily，已 ``shift(1)``）。"""
    pred = ctx.pred.where(ctx.mask)
    fwd1 = ctx.fwd.shift(-1)
    ic = daily_rank_ic(pred, fwd1, ctx.mask)

    # 成员分歧：集成两臂（gbdt h1 / h5）的截面秩相关越低 = 分歧越大
    h1 = pd.read_parquet(PRED_DIR / "gbdt__h1.parquet").reindex(
        index=ctx.pred.index, columns=ctx.pred.columns).where(ctx.mask)
    h5 = pd.read_parquet(PRED_DIR / "gbdt__h5.parquet").reindex(
        index=ctx.pred.index, columns=ctx.pred.columns).where(ctx.mask)
    corr = daily_panel_corr(h1, h5, ctx.mask)

    states = pd.DataFrame({
        "ic60": ic.rolling(60, min_periods=40).mean(),
        "icir60": (ic.rolling(60, min_periods=40).mean()
                   / ic.rolling(60, min_periods=40).std()),
        "disagree": 1.0 - corr,
        "strength": daily_strength(h1, ctx.mask),
        "disp": daily_dispersion(h1, ctx.mask),
        # 状态在 t 日收盘才可知 ⇒ 次日才可用（🚨 前视守卫）
        "ic_realized": ic,
    }).shift(1)
    return states


# ---------------------------------------------------------------------------
# 月频映射与预测力检验
# ---------------------------------------------------------------------------
def month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).last().values)


def monthly_from_daily(daily: pd.Series) -> pd.Series:
    """日序列 → 月序列（月内均值），索引 = 月末日期。

    用月内均值而非单日值：状态量是慢变量，均值更稳且仍然只在月末收盘可知。
    """
    me = month_end_index(daily.index)
    m = daily.groupby(daily.index.to_period("M")).mean()
    m.index = pd.DatetimeIndex(me)
    return m


def monthly_return(daily: pd.Series) -> pd.Series:
    me = month_end_index(daily.index)
    m = daily.groupby(daily.index.to_period("M")).agg(lambda x: (1 + x).prod() - 1)
    m.index = pd.DatetimeIndex(me)
    return m


def first_trading_days(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).first().values)


def monthly_topk_returns(ctx, ks: tuple[int, ...] = (30, 80)) -> pd.DataFrame:
    """月频硬切 Top-K 等权的**月度收益**（无缓冲、无成本，仅作诊断）。

    用于「不降仓择时」的准入检验：集中组合与分散组合的**收益差**才是要预测的
    目标（而不是绝对收益）。始终满仓 ⇒ 错了也不付 beta 代价。
    """
    idx = ctx.pred.index
    out = {}
    for k in ks:
        rec = {}
        for f in first_trading_days(idx):
            p = ctx.pred.loc[f].where(ctx.mask.loc[f]).dropna()
            if len(p) <= k:
                continue
            top = p.sort_values(ascending=False).iloc[:k].index
            # f 日收盘选股 ⇒ 收益从 f+1 起算（与 h=1 / T+1 开仓口径一致）
            days = idx[(idx > f) & (idx.to_period("M") == f.to_period("M"))]
            r = ctx.fwd.loc[days, top].mean(axis=1).fillna(0.0)
            rec[idx[idx.to_period("M") == f.to_period("M")][-1]] = \
                float((1 + r).prod() - 1)
        out[f"K{k}"] = pd.Series(rec)
    df = pd.DataFrame(out)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def predictability(state_me: pd.Series, ret_me: pd.Series) -> dict:
    """状态（月末可知）→ **下月**组合收益 的 rank IC / t / 五分位单调性。"""
    tgt = ret_me.shift(-1)
    j = pd.concat([state_me.rename("s"), tgt.rename("r")], axis=1).dropna()
    if len(j) < 24:
        return {"n": len(j), "rank_ic": np.nan, "t_stat": np.nan,
                "q1": np.nan, "q5": np.nan, "monotone": False}
    ic = float(j["s"].corr(j["r"], method="spearman"))
    q = pd.qcut(j["s"], 5, labels=False, duplicates="drop")
    gm = j.groupby(q)["r"].mean()
    return {"n": len(j), "rank_ic": ic, "t_stat": ic * np.sqrt(len(j) - 1),
            "q1": float(gm.iloc[0]), "q5": float(gm.iloc[-1]),
            "monotone": bool((gm.diff().dropna() >= 0).all())}


# ---------------------------------------------------------------------------
# 仓位规则（输入月末状态 → 输出下月生效的 ω）
# ---------------------------------------------------------------------------
def gate_expanding_median(state_me: pd.Series, low: float = 0.5,
                          above: bool = False, high: float = 1.0) -> pd.Series:
    """PIT 中位门：过门 → ``high``（缺省满仓），不过 → ``low``。

    ``above=False`` 表示状态越大越差（如分歧/disp），此时低于中位才过门。
    中位未建立（前 12 个月）与状态缺失一律中性满仓，避免噪声段污染对照。
    ``high > 1`` 即**杠杆形态**（不降平均仓位、在好月份加杠杆）。
    """
    med = state_me.expanding(min_periods=12).median()
    ok = (state_me < med) if not above else (state_me > med)
    return pd.Series(np.where(ok.fillna(True), high, low), index=state_me.index)


def omega_daily(omega_me: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """月末 ω → 当月每日 ω（用**上月末**的值，零前视）。

    🚨 不能用 ``shift(1) + reindex(ffill)``：``shift`` 会把首月的值丢成 NaN，
    且挂在月末日期上的值只能从**下月最后一天**才生效（整月错位）。这里直接按
    **月份周期**取「上一个月」的值（被 ``test_omega_maps_prev_month_end`` 锁定）。
    """
    by_period = pd.Series(omega_me.to_numpy(), index=omega_me.index.to_period("M"))
    prev = by_period.reindex(idx.to_period("M") - 1)
    prev.index = idx
    return prev.astype(float).fillna(1.0)


def consistency(base: pd.Series, r_timed: pd.Series) -> tuple[float, ...]:
    out = []
    for _, yr in CONSISTENCY_WINDOWS:
        m = r_timed.index >= yr
        out.append(geo_sharpe(r_timed[m].dropna()) - geo_sharpe(base[m].dropna()))
    return tuple(out)


def placebo_pvalue(base: pd.Series, omega_me: pd.Series,
                   actual_delta: float, n: int = 500, seed: int = 0,
                   funding: float = 0.0) -> float:
    """循环移位安慰剂：打乱 ω 与月份的对应关系，Δ年化分布应覆盖 0。

    只做整月平移（保留 ω 的自相关结构），平移量 ≥1 个月以破坏当期对齐。
    """
    rng = np.random.default_rng(seed)
    vals = omega_me.dropna().to_numpy()
    if len(vals) < 24:
        return float("nan")
    beats = 0
    for _ in range(n):
        k = int(rng.integers(1, len(vals)))
        shifted = pd.Series(np.roll(vals, k), index=omega_me.dropna().index)
        r = _apply_position_cost(base, omega_daily(shifted, base.index),
                                funding=funding)
        if annual_return(r) - annual_return(base) >= actual_delta:
            beats += 1
    return (beats + 1) / (n + 1)


# ---------------------------------------------------------------------------
# 自检：钉死四条最易翻车的口径
# ---------------------------------------------------------------------------
def self_check() -> None:
    """任一断言失败 = 实现有 bug，**不要**相信它的输出数字。"""
    rng = np.random.default_rng(0)
    n_days, n_stk = 300, 800
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B")
    cols = [f"S{i:04d}" for i in range(n_stk)]
    ret = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_stk)), index=idx, columns=cols)
    fwd1 = ret.shift(-1)
    mask = pd.DataFrame(True, index=idx, columns=cols)

    # 1) IC 守卫：信号 = 未来收益 ⇒ 日度 IC 必须显著为正
    ic_perfect = daily_rank_ic(fwd1, fwd1, mask).dropna()
    assert ic_perfect.mean() > 0.95, "IC 守卫失效：完美信号未给出 ≈1 的 IC"
    ic_noise = daily_rank_ic(ret, fwd1, mask).dropna()
    assert abs(ic_noise.mean()) < 0.01, "IC 守卫失效：纯噪声给出了非零 IC"

    # 2) 前视守卫：扰动 t 日之后的预测，不得改变 t 及以前的任何状态量
    sig = ret.copy()
    cutoff_i = 200
    cutoff = idx[cutoff_i]
    pert = sig.copy()
    pert.loc[pert.index > cutoff] *= 5.0
    ic_a = daily_rank_ic(sig, fwd1, mask)
    ic_b = daily_rank_ic(pert, fwd1, mask)
    pd.testing.assert_series_equal(ic_a.loc[:cutoff], ic_b.loc[:cutoff], check_names=False)
    st_a = pd.DataFrame({"ic60": ic_a.rolling(60, min_periods=40).mean()}).shift(1)
    st_b = pd.DataFrame({"ic60": ic_b.rolling(60, min_periods=40).mean()}).shift(1)
    pd.testing.assert_frame_equal(st_a.loc[:cutoff], st_b.loc[:cutoff])

    # 3) 锚点守卫：ω ≡ 1 ⇒ 净值逐位不变
    base = pd.Series(rng.normal(0.0005, 0.01, n_days), index=idx)
    r_anchor = _apply_position_cost(base, pd.Series(1.0, index=idx))
    assert annual_return(r_anchor) == annual_return(base)
    assert geo_sharpe(r_anchor) == geo_sharpe(base)

    # 4) 成本守卫：加大成本必须让净值变差
    expo = pd.Series(rng.uniform(0.3, 1.0, n_days), index=idx)
    assert (annual_return(_apply_position_cost(base, expo, 0))
            >= annual_return(_apply_position_cost(base, expo, POSITION_TURNOVER_BP)))

    print("self-check: 4/4 通过（IC / 前视 / 锚点 / 成本）")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_probe() -> dict:
    from backtest.engine import VectorBacktest
    from scripts.evaluation.caliber import (
        cost_profile,
        load_context,
        run_once,
    )
    from strategy.examples import BufferedTopFracLongOnly

    ctx = load_context(signal="ens_h1h5", window="prod", execution="open")
    cost = cost_profile("production")
    bt = VectorBacktest(strategy=BufferedTopFracLongOnly(0.15, 0.40),
                        rebalance_freq="M", initial_capital=1e6, costs=cost)
    res = run_once(bt, ctx, "open")
    base = res.daily_returns.dropna()

    anc_ann, anc_sh = annual_return(base), geo_sharpe(base)
    print(f"基线 v1w-ortho882（production/open）：年化 {anc_ann:.2%}  "
          f"几何Sharpe {anc_sh:.3f}  MDD {max_drawdown(base):.2%}  "
          f"波动 {annual_vol(base):.2%}  交易日 {len(base)}")
    print(f"  锚点校验（登记 16.39% / 0.681）："
          f"Δann={anc_ann - ANCHOR_ANN:+.2%}  ΔSR={anc_sh - ANCHOR_SHARPE:+.3f}")
    assert abs(anc_ann - ANCHOR_ANN) < 5e-4, "基线与登记值不符，口径漂移"
    assert abs(anc_sh - ANCHOR_SHARPE) < 5e-3, "基线与登记值不符，口径漂移"

    # ---- 一、状态量 ----
    states = build_states(ctx)
    me = month_end_index(base.index)
    st_me = pd.DataFrame({c: monthly_from_daily(states[c]) for c in states.columns})
    st_me = st_me.reindex(me)
    ret_me = monthly_return(base).reindex(me)

    # ---- 二、上界 ----
    up = ret_me > 0
    oracle_ret = pd.Series(np.where(up, ret_me, 0.0), index=ret_me.index)
    # 完美预知下月 IC：月初就知道整月平均 IC 的符号 ⇒ ω=1 / 0.5
    ic_next = st_me["ic_realized"].shift(-1)
    om_ic_oracle = pd.Series(np.where(ic_next > 0, 1.0, 0.5), index=ic_next.index)
    r_ic_oracle = _apply_position_cost(base, omega_daily(om_ic_oracle, base.index))
    orc = pd.DataFrame([
        {"oracle": "完美预知下月组合收益（物理不可能）",
         "annual": annual_return(oracle_ret), "sharpe": geo_sharpe(oracle_ret),
         "max_dd": max_drawdown(oracle_ret), "avg_pos": float(up.mean())},
        {"oracle": "完美预知下月 IC 符号（物理不可能）",
         "annual": annual_return(r_ic_oracle), "sharpe": geo_sharpe(r_ic_oracle),
         "max_dd": max_drawdown(r_ic_oracle), "avg_pos": float(om_ic_oracle.mean())},
        {"oracle": "满仓基线（参照）", "annual": anc_ann, "sharpe": anc_sh,
         "max_dd": max_drawdown(base), "avg_pos": 1.0},
    ])

    # ---- 三、预测力 ----
    rows = []
    for c in ["ic60", "icir60", "disagree", "strength", "disp"]:
        d = predictability(st_me[c], ret_me)
        rows.append({"state": c, **d})
    pred_tbl = pd.DataFrame(rows).set_index("state")

    # ---- 四、规则实测 ----
    # 主臂 4 条，方向按**经济含义**先验固定（不按本期数据挑方向）：
    #   近期 IC 高 / ICIR 高 / 成员分歧小 / 预测强度高 ⇒ 信号更可信 ⇒ 满仓
    arms = {
        "ic60>中位 → 满仓": gate_expanding_median(st_me["ic60"], above=True),
        "icir60>中位 → 满仓": gate_expanding_median(st_me["icir60"], above=True),
        "分歧<中位 → 满仓": gate_expanding_median(st_me["disagree"]),
        "强度>中位 → 满仓": gate_expanding_median(st_me["strength"], above=True),
        # 反向臂：**事后**按本期数据翻方向（= 数据挖掘），仅作诊断，不构成证据
        "〔反转·诊断〕ic60<中位 → 满仓":
            gate_expanding_median(st_me["ic60"]),
        "〔反转·诊断〕icir60<中位 → 满仓":
            gate_expanding_median(st_me["icir60"]),
        "〔反转·诊断〕分歧>中位 → 满仓":
            gate_expanding_median(st_me["disagree"], above=True),
        "〔反转·诊断〕强度<中位 → 满仓":
            gate_expanding_median(st_me["strength"]),
        # 「不降仓择时」：同一份信息用在**加杠杆**上，平均仓位 ≈1，不必先付
        # beta 代价。两档口径：含 4.5% 融资成本 / 0% 融资上界。
        "〔杠杆·4.5%〕强度>中位 → 1.5x/0.5x":
            gate_expanding_median(st_me["strength"], low=0.5, high=1.5, above=True),
        "〔杠杆·0%〕强度>中位 → 1.5x/0.5x":
            gate_expanding_median(st_me["strength"], low=0.5, high=1.5, above=True),
        "〔杠杆·4.5%〕分歧<中位 → 1.5x/0.5x":
            gate_expanding_median(st_me["disagree"], low=0.5, high=1.5),
    }
    rows, placebo = [], []
    for name, om in arms.items():
        fund = FUNDING_ANNUAL if ("4.5%" in name) else 0.0
        r = _apply_position_cost(base, omega_daily(om, base.index), funding=fund)
        ds = consistency(base, r)
        d_ann = annual_return(r) - annual_return(base)
        p = placebo_pvalue(base, om, d_ann, funding=fund)
        # 同平均仓位对照：把"长期半仓"的水平效应与"择时"的信息效应拆开。
        # 🚨 少了这一列就会把「长期降仓换低回撤」误读成择时有 alpha
        #    （timing_probe §五 的 MA20 就是这个坑）。
        ap = float(np.nanmean(om.dropna()))
        r_match = _apply_position_cost(base, pd.Series(ap, index=base.index),
                                       funding=fund)
        rows.append({
            "arm": name, "annual": annual_return(r), "d_annual": d_ann,
            "sharpe": geo_sharpe(r), "d_sharpe": geo_sharpe(r) - geo_sharpe(base),
            "max_dd": max_drawdown(r), "avg_pos": ap,
            "d_ann_vs_matched": annual_return(r) - annual_return(r_match),
            "d_sh_vs_matched": geo_sharpe(r) - geo_sharpe(r_match),
            "cost_drag": float(-om.diff().abs().sum() * POSITION_TURNOVER_BP / 1e4
                               / (len(base) / 252)),
            **{f"d_sh_{lbl}": v for (lbl, _), v in zip(CONSISTENCY_WINDOWS, ds)},
            "consistency": f"{sum(1 for x in ds if x > 0)}/{len(ds)}",
            "placebo_p": p,
        })
        placebo.append({"arm": name, "d_annual": d_ann, "placebo_p": p})
    rows.append({
        "arm": "满仓基线(锚点)", "annual": annual_return(base), "d_annual": 0.0,
        "sharpe": geo_sharpe(base), "d_sharpe": 0.0, "max_dd": max_drawdown(base),
        "avg_pos": 1.0, "d_ann_vs_matched": 0.0, "d_sh_vs_matched": 0.0,
        "cost_drag": 0.0,
        **{f"d_sh_{lbl}": 0.0 for lbl, _ in CONSISTENCY_WINDOWS},
        "consistency": "—", "placebo_p": float("nan"),
    })
    arms_tbl = pd.DataFrame(rows).set_index("arm")
    pcs_tbl = pd.DataFrame(placebo).set_index("arm")

    OUT.mkdir(parents=True, exist_ok=True)
    states.to_csv(OUT / "state_daily.csv", encoding="utf-8-sig")
    st_me.assign(next_ret=ret_me.shift(-1)).to_csv(
        OUT / "state_monthly.csv", encoding="utf-8-sig")
    pred_tbl.to_csv(OUT / "predictability.csv", encoding="utf-8-sig")
    arms_tbl.to_csv(OUT / "arm_summary.csv", encoding="utf-8-sig")
    orc.to_csv(OUT / "oracle.csv", index=False, encoding="utf-8-sig")
    pcs_tbl.to_csv(OUT / "placebo.csv", encoding="utf-8-sig")

    print("\n=== 一、上界 ===")
    print(orc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== 二、状态量 → 下月组合收益（月度 rank IC）===")
    print(pred_tbl.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== 三、因果规则实测（月频 ω + 10bp/单位）===")
    cols = ["annual", "d_annual", "sharpe", "d_sharpe", "max_dd", "avg_pos",
            "d_ann_vs_matched", "d_sh_vs_matched", "cost_drag",
            "consistency", "placebo_p"]
    print(arms_tbl[cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\n  多重检验：本轮共 {len(arms)} 臂（4 先验方向 + 4 事后反转诊断 + "
          f"3 杠杆形态），Bonferroni 门槛 = 0.05/{len(arms)} ≈ {0.05 / len(arms):.4f}。")
    print("  判据：**Δ年化为正** 才算改进；四窗口 ≥4/5；安慰剂 p ≥ 0.05 视为噪声。")
    print(f"\n产物: {OUT}/[state_daily, state_monthly, predictability,"
          " arm_summary, oracle, placebo].csv")
    return {"base": base, "states": states, "st_me": st_me,
            "pred": pred_tbl, "arms": arms_tbl, "orc": orc}


def run_spread_admission() -> pd.DataFrame:
    """「不降仓择时」准入检验：状态 → **下月 (Top30 − Top80) 收益差**。

    预测目标从「下月绝对收益」（已证死，|t|≤1.06）换成「集中与分散的收益差」：
    这是**相对**问题，且始终满仓 ⇒ 判错也不付 beta 代价。过门（|t| ≥ 2）才值得
    实现动态 K 的完整回测。
    """
    from scripts.evaluation.caliber import load_context

    ctx = load_context(signal="ens_h1h5", window="prod", execution="open")
    states = build_states(ctx)
    st_me = pd.DataFrame({c: monthly_from_daily(states[c]) for c in states.columns})
    tk = monthly_topk_returns(ctx, ks=(30, 80))
    spread = tk["K30"] - tk["K80"]

    rows = []
    for c in ["ic60", "icir60", "disagree", "strength", "disp"]:
        rows.append({"state": c, **predictability(st_me[c], spread)})
    tbl = pd.DataFrame(rows).set_index("state")

    OUT.mkdir(parents=True, exist_ok=True)
    tk.assign(spread=spread).to_csv(OUT / "topk_spread.csv", encoding="utf-8-sig")
    tbl.to_csv(OUT / "spread_predictability.csv", encoding="utf-8-sig")

    print(f"Top-K 月频（无缓冲/无成本）：K30 月均 {tk['K30'].mean():.2%}  "
          f"K80 月均 {tk['K80'].mean():.2%}  spread 月均 {spread.mean():.2%}  "
          f"胜率 {(spread > 0).mean():.1%}  n={spread.notna().sum()}")
    print("\n=== 准入检验：状态 → 下月 (Top30 − Top80) 收益差 ===")
    print(tbl.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n判据：|t| ≥ 2 才进入动态 K 回测；否则与仓位探针同构收口。")
    return tbl


def main() -> None:
    ap = argparse.ArgumentParser(description="模型自诊断仓位探针")
    ap.add_argument("--self-check", action="store_true", help="只跑口径守卫")
    ap.add_argument("--spread", action="store_true",
                    help="只跑「不降仓择时」准入检验（状态 → 下月 Top30−Top80 差）")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return
    if args.spread:
        run_spread_admission()
        return
    run_probe()


if __name__ == "__main__":
    main()
