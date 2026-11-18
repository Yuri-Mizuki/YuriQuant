"""focus 内部风险预算：三种降回撤手段（小资金口径优先）
================================================================================

背景
----
§十 已证明「外挂异池日频策略」降不了 focus 的回撤（纯分散化、零增量）。
⇒ 剩下唯一的路是**在 focus 自己的 15/8 只持仓内做风险预算**，不动选股、不换池子。

三种手段（全部**只改权重**，不改选股集合、不改调仓频率）
--------------------------------------------------------
1. **``inv_vol`` 逆波动加权**：``w_i ∝ 1/σ_i``。压制高波个股的权重。
2. **``risk_parity`` 风险平价（ERC）**：``w_i ∝ 1/σ_i`` 的迭代修正，
   使**风险贡献**而非**权重**均等。与 inv_vol 的区别在相关项。
3. **``vol_target`` 波动目标**：整体仓位 ``g_t = min(1, σ_target/σ̂_t)``，
   ``σ̂_t`` 用**过去 60 日**组合已实现波动（无前视）。不改个股相对权重。

⚠️ 必须与「**等权**」基线对照，且必须报**配对检验**——本项目已多次证明
「回撤降了但 SR 掉了」= 降的是收益不是风险（§十一 T4）。

🚨 相位铁律
------------
权重只能用 **T 日收盘前可得**的信息 ⇒ ``σ̂`` 用 ``pct_change().shift(1)`` 滚窗。
若写成当日滚窗，就是「用当天收益估当天仓位」，会白送前视收益。

用法::

    .venv/Scripts/python.exe -m scripts.evaluation.focus_risk_budget --capital 1e5 --k 8
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

from scripts.common.cli_common import setup_logging  # noqa: E402
from scripts.evaluation import caliber as cal  # noqa: E402
from scripts.evaluation.caliber import rebalance_dates  # noqa: E402

log = setup_logging("focus_risk_budget")

OUT = ROOT / "reports" / "focus_riskbudget"


# ----------------------------------------------------------------------
# 权重规则
# ----------------------------------------------------------------------
def _inv_vol(w_sub: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """逆波动加权。``w_sub`` 与 ``vol`` 须**同在持仓子集**上。"""
    ok = np.isfinite(vol) & (vol > 0)
    if not ok.any():
        return w_sub
    inv = np.where(ok, 1.0 / np.where(ok, vol, 1.0), 0.0)
    tot = float(inv.sum())
    if tot <= 0:
        return w_sub
    return w_sub.sum() * inv / tot


def _erc(w_ew: np.ndarray, cov: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """风险平价（循环坐标下降，未加权的 ERC 近似）。

    目标：各持仓的**风险贡献** ``w_i(Σw)_i`` 相等。
    用最朴素的乘性更新 ``w_i ← w_i · (target/rc_i)^0.5`` 迭代，
    再按 ``w_ew`` 的非零位置回填、归一到权重和不变。
    """
    mask = w_ew > 1e-12
    n = int(mask.sum())
    if n < 2:
        return w_ew
    sub = cov[np.ix_(mask, mask)]
    # 数值保护：相关矩阵退化时退回对角
    if not np.isfinite(sub).all() or np.linalg.eigvalsh(sub).min() <= 0:
        sub = np.diag(np.clip(np.diag(sub), 1e-12, None))
    w = np.full(n, 1.0 / n)
    for _ in range(n_iter):
        rc = w * (sub @ w)
        tot = rc.sum()
        if tot <= 0:
            break
        target = tot / n
        step = np.clip((target / np.clip(rc, 1e-18, None)) ** 0.5, 0.5, 2.0)
        w = w * step
        w = w / w.sum()
        if np.abs(rc - target).max() / target < 1e-8:
            break
    out = np.zeros_like(w_ew)
    out[mask] = w * float(w_ew.sum())
    return out


def _apply_rule(w_sub: np.ndarray, rule: str, vol: np.ndarray,
                cov: np.ndarray) -> np.ndarray:
    """``w_sub``/``vol``/``cov`` 三者都必须**在持仓子集**上（长度 = 持仓数）。"""
    if rule == "equal":
        return w_sub
    if rule == "inv_vol":
        return _inv_vol(w_sub, vol)
    if rule == "risk_parity":
        return _erc(w_sub, cov)
    raise ValueError(f"未知规则 {rule}")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="focus 内部风险预算")
    ap.add_argument("--capital", type=float, default=1e5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--vol-win", type=int, default=60)
    ap.add_argument("--vol-target", type=float, default=0.20,
                    help="年化目标波动（默认 20%%，比 focus 实际略低）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cost = cal.cost_profile("production")
    ctx = cal.load_context(signal="gbdt_mid13", window="prod",
                           execution="open", end=20260907)

    # ---- 基准：等权 ----
    from strategy.examples import BufferedTopKLongOnly
    bt = cal.VectorBacktest(strategy=BufferedTopKLongOnly(k=args.k, exit_mult=2.0),
                            rebalance_freq="M", initial_capital=args.capital,
                            costs=cost)
    base = cal.run_once(bt, ctx, "open", freq="M",
                        price_panel=ctx.close, lot_size=100)
    w_hist = base.weights_history.copy()
    log.info("基准持仓中位 %.0f 只", (w_hist > cal.N_ACTIVE_TOL).sum(axis=1).median())

    # ---- 逐调仓日的 σ / 协方差（T 日收盘可得，shift(1) 防前视）----
    px = ctx.close
    dret = px.pct_change()
    vol_panel = dret.rolling(args.vol_win, min_periods=20).std().shift(1)
    log.info("σ 面板 %s，末日 %s", vol_panel.shape, vol_panel.index[-1].date())

    rb_days = sorted(set(rebalance_dates(ctx.pred.index, freq="M")))
    log.info("调仓 %d 个信号日", len(rb_days))

    from strategy.base import Strategy

    # 🚨 为什么不在基准 ``w_hist`` 上改比例再回灌
    # ------------------------------------------------
    # ``w_hist`` 是**整手取整后**的权重。在其上改比例再交给引擎，引擎会**再取整一次**，
    # 而 ``_apply_lot_rounding`` 不幂等（整手税是凹的，二次取整系统性偏低）——
    # 实测 equal 臂因此白丢 2.8pp 年化（27.32% vs run_once 的 30.16%），
    # 且 4 个调仓日因整手后不足 2 只被判「无持仓」直接空仓。
    # ⇒ 必须包一层策略：**在原策略吐出的等权目标上**改比例，只让引擎取整一次。
    class _RiskWeighted(Strategy):
        """在内部策略的等权目标上按 ``rule`` 改比例（选股完全不动）。

        Args:
            inner: 被包装的策略（``BufferedTopKLongOnly``），选股与缓冲带归它。
            rule: ``equal`` / ``inv_vol`` / ``risk_parity``。
            vol: ``shift(1)`` 后的滚动 σ 面板（date × code），T 日行 = T 日收盘可得。
            dret: 日收益面板，供 ``risk_parity`` 估截面相关。
            vol_win: 相关窗口长度。
        """

        def __init__(self, inner, rule: str, vol: pd.DataFrame, dret: pd.DataFrame,
                     vol_win: int = 120):
            self.inner = inner
            self.rule = rule
            self.vol = vol
            self.dret = dret
            self.vol_win = vol_win
            self.name = f"{inner.name}__{rule}"
            self.n_missing = 0          # σ 缺失 ⇒ 退化为等权的次数
            self.n_days = 0

        def get_weights(self, factor_values: pd.Series) -> pd.Series:
            w = self.inner.get_weights(factor_values)
            if self.rule == "equal" or w.empty:
                return w
            name = factor_values.name if hasattr(factor_values, "name") else None
            if name is None or name not in self.vol.index:
                self.n_missing += 1
                return w
            names = list(w.index)
            v = self.vol.loc[name, names].to_numpy(dtype=float)
            if not np.isfinite(v).all() or (v <= 0).any():
                # σ 估不出（次新股 / 停牌过多）⇒ 退化等权，不丢持仓
                self.n_missing += 1
                return w
            w_sub = w.to_numpy(dtype=float)
            if self.rule == "risk_parity":
                corr = self.dret.loc[:name, names].tail(self.vol_win).corr().to_numpy()
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
                cov = np.outer(v, v) * corr
            else:
                cov = np.diag(v ** 2)
            new = _apply_rule(w_sub, self.rule, v, cov)
            self.n_days += 1
            return pd.Series(new, index=names)

        def get_weights_at(self, date, factor_vals):
            # 让内部策略也拿到同一个 date（缓冲带不依赖日期，但保持契约一致）
            return self.get_weights(factor_vals)

    rules = ("equal", "inv_vol", "risk_parity")
    results: dict[str, pd.Series] = {}
    metrics: list[dict] = []

    for rule in rules:
        inner = BufferedTopKLongOnly(k=args.k, exit_mult=2.0)
        strat = _RiskWeighted(inner, rule, vol_panel, dret)
        bt2 = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                 initial_capital=args.capital, costs=cost)
        # 🚨 必须走 run_once：它内含 ``sig_used = panel.shift(1).where(ctx.mask)``
        # 的**预掩码**。手拼 bt.run 漏这一步会让 k15 跑出 37.13%（登记值 29.60%），
        # 虚高 7.5pp。默认信号 = ctx.pred。
        res = cal.run_once(bt2, ctx, "open", freq="M",
                           price_panel=ctx.close, lot_size=100)
        log.info("%s: 改写 %d 次，σ 缺失退化 %d 次", rule, strat.n_days,
                 strat.n_missing)

        dr = res.daily_returns.dropna()
        results[rule] = dr
        m = cal.metrics_row(res, args.capital)
        metrics.append({"rule": rule, "ann": m["ann"], "sharpe": m["sharpe"],
                        "sharpe_arith": m["sharpe_arith"], "max_dd": m["max_dd"],
                        "turnover": m["turnover"], "n_days": m["n_days"]})

    # ---- 同源自检：equal 手拼路径 vs base（run_once）----
    if "equal" in results:
        d_eq = results["equal"].sub(base.daily_returns.dropna()).dropna()
        log.info("同源自检 equal(手拼) − base(run_once)：日均差 %+.6f%%，最大绝对差 %.2e，"
                 "重合 %d 日", 100 * d_eq.mean(), float(d_eq.abs().max()), len(d_eq))

    # ---- 波动目标：扫多个目标位 ----
    # 🚨 只测一个目标位是无效对照：vol_target 是**仓位缩放的单调族**，
    # 单点必然出现「收益低、回撤低」，无法分辨「风险效率改善」与「单纯降杠杆」。
    # 判据：若 SR 随目标位单调不升，则这条规则只是杠杆旋钮（≈ 持有现金），
    # 不是 alpha 或风险管理。
    base_dr = results.get("equal")
    vt_rows: list[dict] = []
    if base_dr is not None and len(base_dr) > 120:
        rv = base_dr.rolling(args.vol_win, min_periods=20).std().shift(1) * np.sqrt(252)
        for tgt in (0.12, 0.16, 0.20, 0.24, 0.30, 0.99):
            g = (tgt / rv).clip(upper=1.0).fillna(1.0)
            vt = base_dr * g
            key = f"vol_target({tgt:.0%})" if tgt < 0.99 else "vol_target(off)"
            results[key] = vt
            sharpe_pair = cal.sharpe_pair(vt)
            vt_rows.append({
                "rule": key,
                "ann": float(cal.annual_return(vt)),
                "sharpe": float(sharpe_pair["sharpe"]),
                "sharpe_arith": float(sharpe_pair["sharpe_arith"]),
                "max_dd": float(cal.max_drawdown(vt)),
                "turnover": float("nan"),
                "avg_exposure": float(g.mean()),
                "n_days": int(len(vt)),
            })
        # off（不缩放）应逐位等于 equal ⇒ 同源自检的第二道锚
        d_off = results["vol_target(off)"].sub(base_dr).abs().max()
        log.info("vol_target(off) vs equal 最大绝对差 %.2e（应严格为 0）", float(d_off))

    # ------------------------------------------------------------------
    # 有没有「空间」？两道上界，任一道为负则本方向不必再挖
    # ------------------------------------------------------------------
    # 🚨 这一节比三条规则的输赢更根本。三条规则都是**估计器**，输了可能是估得差，
    # 不代表没空间。所以要么证明**空间本身不存在**，要么老实承认「可能只是没想到好规则」。
    #
    # 上界一：**完美预知的权重优化能降多少波动**
    #   用**已知**的 120 日协方差求最小方差权重（事后，不构成可交易方案，只做上界），
    #   比等权降多少波动。任何**事前**规则都不可能超过这个数。
    #
    # 上界二：**回撤期内持仓间相关性是否崩掉**
    #   分散化的价值 = 1/rho。若回撤期 rho 跳到 0.8+，有效独立注数 N_eff->1.x，
    #   无论怎么分配权重都只剩「一个注」，权重维度自然无解。
    #
    # ⚠️ beta 归因用**同期**回归（y_t 与 mkt_t 同为收盘价收益），这是**事后分解**，
    #   不是可交易预测。此前误加 ``shift(1)`` 把同期相关切断，beta 被估成 0、
    #   预测序列退化成常数，才跑出「拟合SR=40、R2<0」的假象。
    from scipy import optimize, stats

    def _mdd_window(dr: pd.Series):
        """最大回撤的 (峰, 谷)。

        🚨 必须用 **drawdown 序列**定位，不能对 ``cum`` 取 ``argmin``：
        净值起点恒为 1.0（首个收益为 0），是全局最小值，``argmin`` 会永远返回第 0 行，
        于是"回撤区间"退化成一天（本脚本曾因此报出 2019-01-02 → 2019-01-02）。
        """
        cum = (1 + dr).cumprod()
        dd = cum / cum.cummax() - 1.0
        trough = dd.idxmin()
        peak = cum.loc[:trough].idxmax()
        return peak, trough

    def _minvar_gain(dr: pd.DataFrame, row: pd.Series, t,
                     wcap: float = 1.0, win: int = 120) -> float:
        """完美预知最小方差 vs 等权 的**波动降幅**（正 = 有权重优化空间）。

        ``wcap``：单票权重上限。**无约束（1.0）会退化成「全仓最低波那只」**，
        波动降幅可到 99%（= 零分散的伪优化），不构成有意义的上界；
        故同时报 ``wcap=2/N``（分散度不低于「2 倍等权」）的版本作真正上界。
        """
        names = list(row.index[row > cal.N_ACTIVE_TOL])
        n = len(names)
        if n < 3:
            return float("nan")
        sub = dr.loc[:t].tail(win)[names]
        if sub.shape[0] < 60:
            return float("nan")
        cov = sub.cov().to_numpy()
        if not np.isfinite(cov).all():
            return float("nan")
        cov = cov + np.eye(n) * 1e-10 * float(np.trace(cov)) / n
        w0 = np.full(n, 1.0 / n)
        vol_ew = float(np.sqrt(w0 @ cov @ w0))
        cap = float(np.clip(wcap, 1.0 / n, 1.0))
        try:
            r = optimize.minimize(
                lambda w: float(w @ cov @ w), w0, method="SLSQP",
                bounds=[(0.0, cap)] * n,
                constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                options={"maxiter": 300, "ftol": 1e-12})
            if not r.success:
                return float("nan")
            w = np.clip(r.x, 0, None)
            w = w / w.sum()
        except (ValueError, np.linalg.LinAlgError):
            return float("nan")
        vol_mv = float(np.sqrt(w @ cov @ w))
        return (vol_ew - vol_mv) / vol_ew if vol_ew > 0 else float("nan")

    diag: dict = {}
    if base_dr is not None and len(base_dr) > 400:
        _pos = {d: i for i, d in enumerate(ctx.pred.index)}
        gains, gains_cap = [], []
        for t in rb_days:
            i = _pos.get(t, -1)
            if not (0 <= i < len(ctx.pred.index) - 1):
                continue
            e = ctx.pred.index[i + 1]
            if e not in w_hist.index:
                continue
            row = w_hist.loc[e]
            n_hold = int((row > cal.N_ACTIVE_TOL).sum())
            g = _minvar_gain(dret, row, t)
            gc = _minvar_gain(dret, row, t, wcap=2.0 / max(n_hold, 1))
            if np.isfinite(g):
                gains.append(g)
            if np.isfinite(gc):
                gains_cap.append(gc)
        sr_ew = float(cal.sharpe_pair(base_dr)["sharpe"])
        T = len(base_dr) / 252.0
        diag["sr_ew"] = sr_ew
        diag["T_years"] = float(T)
        # SR 的测量精度（Lo 2002）：SE ≈ sqrt((1 + SR²/2) / T)
        diag["sr_se"] = float(np.sqrt((1.0 + sr_ew ** 2 / 2.0) / T))
        if gains:
            gs = np.array(gains)
            gc = np.array(gains_cap) if gains_cap else np.array([np.nan])
            diag["minvar_gain_med"] = float(np.median(gs))
            diag["minvar_gain_max"] = float(gs.max())
            diag["minvar_gain_cap_med"] = float(np.nanmedian(gc))
            diag["minvar_gain_cap_p90"] = float(np.nanpercentile(gc, 90))
            diag["minvar_n"] = int(len(gs))
            # 换算成 SR：σ 降 g 时，
            #   收益同比例降 ⇒ SR 不变（下沿 0 增益）；
            #   收益一点都不降 ⇒ SR = SR/(1-g)（上沿，不现实但严格是上界）。
            diag["sr_ub_sigma"] = sr_ew / (1.0 - diag["minvar_gain_cap_med"])

        peak_d, trough_d = _mdd_window(base_dr)
        diag["mdd_peak"] = str(peak_d.date())
        diag["mdd_trough"] = str(trough_d.date())
        corr_rows = []
        for e, row in w_hist.iterrows():
            if e not in dret.index:
                continue
            names = list(row.index[row > cal.N_ACTIVE_TOL])
            if len(names) < 3:
                continue
            sub = dret.loc[:e].tail(120)[names].dropna(axis=1, how="any")
            if sub.shape[1] < 3 or sub.shape[0] < 60:
                continue
            cm = sub.corr().to_numpy()
            iu = np.triu_indices_from(cm, k=1)
            rho = float(np.nanmean(cm[iu]))
            n = sub.shape[1]
            neff = n / (1.0 + (n - 1.0) * max(rho, 0.0))
            corr_rows.append({"date": e, "n": n, "rho": rho, "n_eff": neff})
        cr = pd.DataFrame(corr_rows)
        if len(cr):
            in_dd = (cr["date"] >= peak_d) & (cr["date"] <= trough_d)
            diag["rho_all"] = float(cr["rho"].median())
            diag["rho_mdd"] = (float(cr.loc[in_dd, "rho"].median())
                               if in_dd.any() else float("nan"))
            diag["neff_all"] = float(cr["n_eff"].median())
            diag["neff_mdd"] = (float(cr.loc[in_dd, "n_eff"].median())
                                if in_dd.any() else float("nan"))
            diag["rho_p90"] = float(cr["rho"].quantile(0.9))
            cr.to_csv(OUT / "risk_budget_corr.csv", index=False, encoding="utf-8-sig")

    mkt_dr = (px.pct_change().where(ctx.pred.notna()).mean(axis=1).reindex(
        base_dr.index) if base_dr is not None else None)
    decomp_rows = []
    if mkt_dr is not None and mkt_dr.notna().sum() > 400:
        j = pd.concat([base_dr.rename("y"), mkt_dr.rename("m")], axis=1).dropna()
        if len(j) > 400:
            b, a = np.polyfit(j["m"], j["y"], 1)
            pred = a + b * j["m"]
            resid = j["y"] - pred
            decomp_rows.append({
                "item": "beta_vs_alla_ew", "T": int(len(j)), "beta": float(b),
                "alpha_ann": float(a * 252), "corr": float(j["y"].corr(j["m"])),
                "R2": 1.0 - float(resid.var() / j["y"].var()),
                "SR_port": float(cal.sharpe_pair(j["y"])["sharpe"]),
                "SR_beta_part": float(cal.sharpe_pair(pred)["sharpe"]),
                "SR_resid": float(cal.sharpe_pair(resid)["sharpe"]),
                "MDD_port": float(cal.max_drawdown(j["y"])),
                "MDD_resid": float(cal.max_drawdown(resid)),
            })
        pd.DataFrame(decomp_rows).to_csv(
            OUT / "risk_budget_decomp.csv", index=False, encoding="utf-8-sig")

    if diag:
        print("\n  有没有空间？（上界检验，比规则输赢更根本）")
        print("  ※ 判据：上界换算成 SR 增益，与 SR 的测量精度 SE=%.3f 比。"
              % diag.get("sr_se", float("nan")))
        print("    上界 < SE ⇒ **无论规则写得多好都测不出来** ⇒ 权重维度不值得挖；")
        print("    上界 > SE 仍判负 ⇒ 是实现问题，不是没空间（两者要分开报）。")
        if "sr_ub_sigma" in diag:
            gain_max = diag["sr_ub_sigma"] - diag["sr_ew"]
            print("    \u2460 完美预知最小方差（事后上界，非可交易）：")
            print("       σ 可降幅：无约束 中位 %.1f%%（最大 %.1f%%，会退化到全仓"
                  "一只低波股）/ 单票≤2/N 时 中位 %.1f%%" % (
                      100 * diag["minvar_gain_med"], 100 * diag["minvar_gain_max"],
                      100 * diag["minvar_gain_cap_med"]))
            print("       ⇒ SR 增益区间 [0, %+.4f]：σ 降 %.1f%% 时，收益降幅 < %.1f%%"
                  " 才改善 SR；增益上限 = %.2f 个 SE" % (
                      gain_max, 100 * diag["minvar_gain_cap_med"],
                      100 * diag["minvar_gain_cap_med"],
                      gain_max / diag["sr_se"]))
        if "rho_all" in diag:
            print("    \u2461 持仓间相关：常态 rho=%.3f（有效注数 %.1f）"
                  " → 回撤期 rho=%.3f（有效注数 %.1f）" % (
                diag["rho_all"], diag["neff_all"], diag["rho_mdd"], diag["neff_mdd"]))
            print("       回撤区间 %s → %s；rho 的 P90 = %.3f" % (
                diag["mdd_peak"], diag["mdd_trough"], diag["rho_p90"]))
        if decomp_rows:
            r = decomp_rows[0]
            print("    \u2462 beta 归因（同期、事后）："
                  "beta=%.2f, corr=%.3f, R2=%.3f, alpha=%.1f%%/年" % (
                r["beta"], r["corr"], r["R2"], 100 * r["alpha_ann"]))
            print("       组合 SR %.3f vs 残差 SR %.3f；组合 MDD %.1f%% vs 残差 MDD %.1f%%"
                  % (r["SR_port"], r["SR_resid"], 100 * r["MDD_port"],
                     100 * r["MDD_resid"]))
            print("       ※ 回撤 %.1f%% 里有 %.1f%%（占比 %.0f%%）在剔除 beta 后仍在"
                  " ⇒ 这一部分**任何个股配权都动不了**"
                  % (100 * r["MDD_port"], 100 * r["MDD_resid"],
                     100 * r["MDD_resid"] / r["MDD_port"]))
        pd.DataFrame([diag]).to_csv(
            OUT / "risk_budget_room.csv", index=False, encoding="utf-8-sig")

    # ---- 输出 ----
    dfm = pd.DataFrame(metrics)
    dfvt = pd.DataFrame(vt_rows)
    print("=" * 78)
    print(f"focus 内部风险预算 · K={args.k} · 资金 {args.capital:,.0f} · 整手100")
    print("  基准 = 等权（现行生产做法）；σ 用过去 %d 日、shift(1) 防前视" % args.vol_win)
    print("=" * 78)
    print("  %-20s %9s %8s %9s %9s %10s" % (
        "规则", "年化", "SR", "SR(算术)", "MDD", "换手"))
    for _, r in dfm.iterrows():
        print("  %-20s %+8.2f%% %8.4f %9.4f %+8.2f%% %10.3f" % (
            r["rule"], 100 * r["ann"], r["sharpe"], r["sharpe_arith"],
            100 * r["max_dd"], r["turnover"]))

    if len(dfvt):
        print("\n  波动目标族（仓位整体缩放，不看个股）：")
        print("  %-20s %9s %8s %9s %10s %8s" % (
            "目标", "年化", "SR", "MDD", "平均仓位", "SR/仓位"))
        for _, r in dfvt.iterrows():
            eff = r["sharpe"] / r["avg_exposure"] if r["avg_exposure"] > 0 else float("nan")
            print("  %-20s %+8.2f%% %8.4f %+8.2f%% %9.1f%% %8.4f" % (
                r["rule"], 100 * r["ann"], r["sharpe"], 100 * r["max_dd"],
                100 * r["avg_exposure"], eff))
        print("  ※ 「SR/仓位」若随目标位单调不升 ⇒ 纯杠杆旋钮，无风险管理增益")

    # ---- 配对检验 vs 等权 ----
    print("\n  配对检验 vs 等权基准：")
    print("  %-20s %11s %9s %8s  %s" % ("规则", "日均差", "配对t", "p", "判定"))
    prows = []
    for name, dr in results.items():
        if name == "equal":
            continue
        j = pd.concat([dr.rename("a"), base_dr.rename("b")], axis=1).dropna()
        diff = j["a"] - j["b"]
        t, p = stats.ttest_1samp(diff, 0)
        verdict = "✅ 显著" if p < 0.05 else "—"
        print("  %-20s %+10.5f%% %+9.3f %8.4f  %s" % (
            name, 100 * diff.mean(), t, p, verdict))
        prows.append({"rule": name, "mean_diff": float(diff.mean()),
                      "t": float(t), "p": float(p)})

    dfm.to_csv(OUT / "risk_budget_metrics.csv", index=False, encoding="utf-8-sig")
    if len(dfvt):
        dfvt.to_csv(OUT / "risk_budget_voltarget.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prows).to_csv(OUT / "risk_budget_paired.csv", index=False,
                               encoding="utf-8-sig")
    pd.DataFrame(results).to_csv(OUT / "risk_budget_daily.csv", encoding="utf-8-sig")
    print(f"\n已写 {OUT.name}/risk_budget_{{metrics,voltarget,paired,daily}}.csv")


if __name__ == "__main__":
    main()