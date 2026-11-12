"""持仓数与持仓权重探针 —— `yqcore-focus` 能压到几只、权重该不该动。

回答了 2026-11-10 的两个问题：

1. **K 从 15 还能往下压吗？** —— 收益上「压不动也压不坏」，风险上代价明确。
   配对 bootstrap（同月度对齐，vs K=15 的年化差）**全部跨 0**：
   K=5 +2.32pp CI[−4.13,+9.06]、K=8 +1.94pp CI[−3.05,+6.96]、K=3 −1.53pp
   CI[−13.00,+10.42] ⇒ **K∈[5,25] 收益无显著差异**，选 K 只能按风险与可执行性定。
2. **权重为什么是等权？** —— 9 种方案 × 5 个 K：除极端加权显著变差外，全部
   落在 ±2pp。`inv_vol` 随强度单调恶化（−1.09 → −2.11pp），`rank_over` /
   `softmax`（把权重压给信号最强的）亏 5~15pp ⇒ **越相信信号排序越亏**，
   与「h1 在极端处追涨」的机制一致。

口径：production（固定 10bp / T+1 open / 1e6），2019-01-02..2026-09-07，
      信号 `gbdt_mid13`（复合信号，`caliber.SIGNALS`）。

关键实现纪律（踩过三次坑，故写成断言）
---------------------------------------
复刻「引擎实际持仓」时必须逐位对齐 `caliber.run_once(execution="open")`：
信号日 T = 每月首个交易日；执行日 E = 下一交易日；引擎取 `sig_used.iloc[pos[E]]`，
而 `sig_used = pred.shift(1).where(mask)` ⇒ 等价于 `pred[T]` + `mask[T]`，
**不能再手动 shift 一次**；且引擎「先结算后调仓」⇒ 新权重从 `pos[E]+1` 起才计收益。
`_selfcheck()` 用 Jaccard=1.0 锁死这一点（Jaccard 0.05 曾经悄悄毁掉一整轮数字）。

输出（默认 `reports/focus_holding/`）：
  kmin_grid.csv     K × 指标（含 95% bootstrap CI、逐年）
  kmin_loo.csv      毛收益与「去掉每期最好/最差一只」后的年化（收益依赖度）
  kmin_hand.csv     整手约束：实投率、买不起一手的只数、整手导致的内生权重偏离
  kmin_pair.csv     K vs K=15 的配对 bootstrap
  weight_grid.csv   9 种权重方案 × 5 个 K 的净值指标
  weight_shape.csv  各权重方案实际造成的权重分布（证明不是空转）

用法::

    python -m scripts.evaluation.focus_holding_probe            # 全跑
    python -m scripts.evaluation.focus_holding_probe --quick    # 跳过权重网格
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from backtest.engine import VectorBacktest
from strategy.examples import BufferedTopKLongOnly

ROOT = Path(__file__).resolve().parents[2]
END = 20260907
CAP = 1e6
K_GRID = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
W_MODES = ["equal", "inv_vol", "inv_vol_sq", "inv_vol_sqrt", "rank_lin",
           "rank_over", "rank_shrink", "softmax", "tier", "rank_x_invvol"]


# ---------------------------------------------------------------------------
# 持仓复刻（与引擎逐位一致）
# ---------------------------------------------------------------------------
def period_holdings(ctx, k: int, exit_mult: float = 2.0):
    """按月频复刻策略逐期持仓 → ``[(i0, i1, [code...])]``。

    ``i0:i1`` 是该期持仓在 ``ctx.pred.index`` 上的**收益区间**（左闭右开）。
    """
    idx = ctx.pred.index
    pos = {d: i for i, d in enumerate(idx)}
    n = len(idx)
    s = pd.Series(idx, index=idx)
    firsts = s.groupby(s.index.to_period("M")).first().tolist()
    edges = [pos[t] + 1 for t in firsts if pos[t] + 1 < n]      # 执行日行号
    sig = ctx.pred.shift(1).where(ctx.mask)                     # 与 run_once 同源
    strat = BufferedTopKLongOnly(k=k, exit_mult=exit_mult)
    out = []
    for j, e in enumerate(edges):
        vals = sig.iloc[e].dropna()
        if vals.empty:
            continue
        w = strat.get_weights(vals)
        i0 = e + 1                                              # 先结算后调仓
        i1 = (edges[j + 1] + 1) if j + 1 < len(edges) else n
        out.append((i0, i1, list(w.index)))
    return out


def _selfcheck(ctx, k: int = 15) -> float:
    """复刻持仓 vs 引擎实际持仓的 Jaccard（必须 =1.0）。"""

    class _Rec(BufferedTopKLongOnly):
        def __init__(self, kk, em):
            super().__init__(kk, em)
            self.log = []

        def get_weights_at(self, date, vals):
            w = self.get_weights(vals)
            self.log.append(list(w.index))
            return w

    strat = _Rec(k, 2.0)
    bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                        initial_capital=CAP, costs=cal.cost_profile("production"))
    cal.run_once(bt, ctx, "open")
    mine = period_holdings(ctx, k)
    n = min(len(strat.log), len(mine))
    if n == 0:
        raise AssertionError("复刻持仓为空——检查 pred/mask 面板")
    jac = [len(set(strat.log[i]) & set(mine[i][2]))
           / max(len(set(strat.log[i]) | set(mine[i][2])), 1) for i in range(n)]
    lo = float(min(jac))
    if lo < 1.0:
        raise AssertionError(
            f"复刻持仓与引擎不一致（Jaccard min={lo:.3f}, n={len(mine)}/"
            f"{len(strat.log)}）—— 收益区间或 shift 语义已漂，禁止用本轮数字"
        )
    return lo


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def boot_ci(dr: pd.Series, n_boot: int = 2000, seed: int = 20261110):
    m = (1 + dr).resample("ME").prod() - 1
    v = m.to_numpy()
    rng = np.random.default_rng(seed)
    samp = v[rng.integers(0, len(v), size=(n_boot, len(v)))]
    ann = np.prod(1 + samp, axis=1) ** (12.0 / len(v)) - 1
    return (float(np.percentile(ann, 2.5)), float(np.percentile(ann, 97.5)),
            float((ann <= 0).mean()))


def run_k(ctx, cost, k: int, exit_mult: float = 2.0):
    bt = VectorBacktest(strategy=BufferedTopKLongOnly(k=k, exit_mult=exit_mult),
                        rebalance_freq="M", initial_capital=CAP, costs=cost)
    res = cal.run_once(bt, ctx, "open")
    met = cal._metrics(res, CAP)
    lo, hi, pneg = boot_ci(res.daily_returns)
    met.update(ann_lo=lo, ann_hi=hi, p_neg=pneg, ci_width=hi - lo)
    for y, sub in res.daily_returns.groupby(res.daily_returns.index.year):
        met[f"y{y}"] = float((1 + sub).prod() - 1)
    return met, res.daily_returns


# ---------------------------------------------------------------------------
# 收益依赖度（毛收益，close-to-close）
# ---------------------------------------------------------------------------
def loo(ctx, k: int) -> dict:
    """leave-best/worst-out：**毛收益**口径，只做相对依赖度比较。

    与主表（净、T+1 open）不可并列引用；用途是回答「收益多依赖每期冠军」。
    """
    rows = []
    for i0, i1, codes in period_holdings(ctx, k):
        if i1 <= i0:
            continue
        seg = ctx.fwd.iloc[i0:i1]
        cols = [c for c in codes if c in seg.columns]
        if not cols:
            continue
        rows.append((1 + seg[cols].fillna(0)).prod() - 1)
    if not rows:
        return {}
    R = pd.DataFrame(rows)
    eq = R.mean(axis=1)
    best = R.apply(lambda s: s.drop(s.idxmax()).mean(), axis=1)
    worst = R.apply(lambda s: s.drop(s.idxmin()).mean(), axis=1)
    top2 = R.apply(lambda s: s.drop(s.nlargest(2).index).mean(), axis=1)

    def ann(x):
        return float(np.prod(1 + x) ** (12.0 / len(x)) - 1)

    pos_eq = eq > 0
    k_eff = R.notna().sum(axis=1)
    hi = R.max(axis=1) / k_eff                  # 等权：单票对组合的贡献
    lo = R.min(axis=1) / k_eff                  # 单票对组合的拖累
    return dict(
        ann_close=ann(eq), ann_drop_best=ann(best), ann_drop_worst=ann(worst),
        ann_drop_top2=ann(top2), delta_best=ann(best) - ann(eq),
        n_periods=len(R), n_hold=float(R.notna().sum(axis=1).mean()),
        frac_months_best_gt_half=float(((hi / eq.where(pos_eq)) > 0.5)
                                       .where(pos_eq).mean()),
        worst_single_med=float(lo.median()), worst_single_p05=float(lo.quantile(0.05)),
    )


# ---------------------------------------------------------------------------
# 整手约束
# ---------------------------------------------------------------------------
def hand_check(ctx, k: int, capital: float) -> dict:
    holds = period_holdings(ctx, k)
    if not holds:
        return {}
    codes = holds[-1][2]
    px = ctx.close.iloc[-1].reindex(codes).dropna()
    if px.empty:
        return {}
    budget = capital / max(len(codes), 1)
    unit = px * 100
    raw = np.floor(budget / unit)
    lots = np.maximum(raw, 1)
    actual = lots * unit
    w = actual / actual.sum()
    eq = 1.0 / len(px)
    return dict(capital=capital, k=k, n=len(px), budget_per=budget,
                px_med=float(px.median()), lots_med=float(np.median(lots)),
                lots_min=int(lots.min()), unaffordable=int((raw < 1).sum()),
                invest_ratio=float(actual.sum() / capital),
                hand_dev=float(actual.sum() / capital - 1),
                w_dev_max=float((w - eq).abs().max()),
                w_dev_med=float((w - eq).abs().median()))


# ---------------------------------------------------------------------------
# 权重方案
# ---------------------------------------------------------------------------
class WeightedBufferedTopK(BufferedTopKLongOnly):
    """选股同父类（**持仓集合逐位一致**），只改权重；返回 Σ=1。"""

    def __init__(self, k, exit_mult, mode, vol=None, lam=0.5):
        super().__init__(k=k, exit_mult=exit_mult)
        self.mode = mode
        self.vol = vol
        self.lam = float(lam)
        self.name = f"wk_{mode}_{k}"

    def get_weights_at(self, date, factor_values):
        base = self.get_weights(factor_values)
        if base.empty or self.mode == "equal":
            return base
        codes = base.index
        n = len(codes)
        vals = factor_values.reindex(codes)
        ranks = vals.rank(ascending=False, method="first")

        if self.mode.startswith("inv_vol"):
            s = self._sigma(date, codes)
            if s is None:
                return base
            raw = {"inv_vol": 1.0 / s, "inv_vol_sq": 1.0 / s ** 2,
                   "inv_vol_sqrt": 1.0 / np.sqrt(s)}.get(self.mode)
            if raw is None:
                return base
        elif self.mode == "rank_lin":
            raw = (n - ranks + 1.0)
        elif self.mode == "rank_over":
            raw = 1.0 / ranks
        elif self.mode == "rank_shrink":
            eq = pd.Series(1.0 / n, index=codes)
            lin = (n - ranks + 1.0)
            return (self.lam * eq + (1 - self.lam) * (lin / lin.sum())).astype(float)
        elif self.mode == "softmax":
            z = (vals - vals.mean()) / (vals.std(ddof=0) + 1e-12)
            raw = np.exp(z.clip(-3, 3))
        elif self.mode == "tier":
            half = max(1, n // 2)
            raw = pd.Series(np.where(ranks <= half, 1.5, 0.5), index=codes)
        elif self.mode == "rank_x_invvol":
            s = self._sigma(date, codes)
            if s is None:
                return base
            raw = (n - ranks + 1.0) / s
        else:
            return base

        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(raw.median()).clip(lower=0)
        tot = raw.sum()
        if not np.isfinite(tot) or tot <= 0:
            return base
        return (raw / tot).astype(float)

    def _sigma(self, date, codes):
        if self.vol is None:
            return None
        v = self.vol
        if date in v.index:
            row = v.loc[date]
        else:
            hist = v.loc[:date]
            row = hist.iloc[-1] if len(hist) else None
        if row is None:
            return None
        s = row.reindex(codes).replace(0.0, np.nan)
        if s.notna().sum() < max(3, len(codes) // 2):
            return None
        return s.fillna(s.median()).clip(lower=1e-4)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="持仓数 / 权重探针")
    ap.add_argument("--out", default="reports/focus_holding")
    ap.add_argument("--quick", action="store_true", help="跳过权重网格（省 ~1 分钟）")
    ap.add_argument("--end", type=int, default=END)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("加载上下文 ...", flush=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=args.end,
                           execution="open")
    cost = cal.cost_profile("production")
    print(f"  {ctx.n_days} 日 / {ctx.pred.shape[1]} 列 / 指纹 {ctx.fingerprint}",
          flush=True)
    jac = _selfcheck(ctx)
    print(f"  自检：复刻持仓 vs 引擎 Jaccard = {jac:.3f} ✓", flush=True)

    # --- K 网格 ---
    print("\nK 网格（净，T+1 open，10bp）...", flush=True)
    rows = []
    for k in K_GRID:
        m, _ = run_k(ctx, cost, k)
        yr = {c: m.get(c) for c in m if c.startswith("y2")}
        rows.append(dict(k=k, **{c: m.get(c) for c in (
            "ann", "sharpe", "sharpe_arith", "sharpe_t_stat", "vol", "max_dd",
            "turnover", "cost_bp", "n_active", "ann_lo", "ann_hi", "ci_width",
            "p_neg")}, **yr))
        print(f"  K={k:>2} ann={m['ann']*100:6.2f}%  SR={m['sharpe']:.3f}  "
              f"DD={m['max_dd']*100:5.1f}%  CI[{m['ann_lo']*100:5.1f},"
              f"{m['ann_hi']*100:5.1f}]", flush=True)
    pd.DataFrame(rows).to_csv(out / "kmin_grid.csv", index=False)

    # --- 依赖度 ---
    print("\n收益依赖度（毛收益口径，仅作相对比较）...", flush=True)
    lrows = []
    for k in [5, 6, 8, 10, 12, 15, 20, 25]:
        d = loo(ctx, k)
        if d:
            lrows.append(dict(k=k, **d))
            print(f"  K={k:>2} 毛={d['ann_close']*100:6.2f}%  "
                  f"去最好一只={d['delta_best']*100:+6.2f}pp  "
                  f"冠军过半月份={d['frac_months_best_gt_half']*100:4.1f}%  "
                  f"最差单票拖累中位={d['worst_single_med']*100:5.2f}%", flush=True)
    pd.DataFrame(lrows).to_csv(out / "kmin_loo.csv", index=False)

    # --- 整手 ---
    print("\n整手约束 ...", flush=True)
    hrows = [hand_check(ctx, k, cap) for k in (5, 8, 10, 15, 20)
             for cap in (1e5, 2e5, 5e5, 1e6)]
    hrows = [r for r in hrows if r]
    pd.DataFrame(hrows).to_csv(out / "kmin_hand.csv", index=False)
    for r in hrows:
        print(f"  {r['capital']/1e4:5.0f}万 K={r['k']:>2}  实投={r['invest_ratio']*100:5.1f}%"
              f"  买不起={r['unaffordable']}  权重偏离max={r['w_dev_max']*100:4.1f}pp",
              flush=True)

    # --- 配对 bootstrap ---
    print("\n配对 bootstrap（vs K=15，年化算术差）...", flush=True)
    rets = {}
    for k in K_GRID:
        bt = VectorBacktest(strategy=BufferedTopKLongOnly(k=k, exit_mult=2.0),
                            rebalance_freq="M", initial_capital=CAP, costs=cost)
        rets[k] = (1 + cal.run_once(bt, ctx, "open").daily_returns).resample("ME").prod() - 1
    rng = np.random.default_rng(20261110)
    prows = []
    for k, m in rets.items():
        v = (m - rets[15]).dropna().to_numpy()
        samp = v[rng.integers(0, len(v), size=(4000, len(v)))].mean(axis=1) * 12
        lo_, hi_ = np.percentile(samp, [2.5, 97.5])
        prows.append(dict(k=k, ann_diff=float(v.mean() * 12), lo=float(lo_),
                          hi=float(hi_), p_gt0=float((samp > 0).mean()),
                          n_months=len(v)))
        print(f"  K={k:>2} Δ={v.mean()*12*100:+6.2f}pp "
              f"CI[{lo_*100:+6.2f},{hi_*100:+6.2f}]  P(Δ>0)={prows[-1]['p_gt0']:.3f}",
              flush=True)
    pd.DataFrame(prows).to_csv(out / "kmin_pair.csv", index=False)

    if args.quick:
        print("\n--quick：跳过权重网格", flush=True)
        return 0

    # --- 权重方案 ---
    print("\n权重方案（同持仓集合，只改权重）...", flush=True)
    wrows = []
    for k in (5, 8, 10, 12, 15):
        for mode in W_MODES:
            s = WeightedBufferedTopK(k, 2.0, mode, vol=ctx.vol60)
            bt = VectorBacktest(strategy=s, rebalance_freq="M",
                                initial_capital=CAP, costs=cost)
            res = cal.run_once(bt, ctx, "open")
            m = cal._metrics(res, CAP)
            wrows.append(dict(k=k, mode=mode, ann=m["ann"], sharpe=m["sharpe"],
                              sharpe_arith=m["sharpe_arith"],
                              max_dd=m["max_dd"], turnover=m["turnover"]))
            print(f"  K={k:>2} {mode:<14} ann={m['ann']*100:6.2f}%  "
                  f"SR={m['sharpe']:.3f}", flush=True)
    wg = pd.DataFrame(wrows)
    wg.to_csv(out / "weight_grid.csv", index=False)

    p = wg.pivot(index="mode", columns="k", values="ann") * 100
    print("\nΔ vs 等权 (pp):")
    print(p.sub(p.loc["equal"], axis=1).round(2).to_string(), flush=True)

    # --- 权重形态 ---
    sig = ctx.pred.shift(1).where(ctx.mask)
    valid = [d for d in ctx.pred.index[-40:] if sig.loc[d].notna().sum() > 100]
    date = valid[-1]
    vals = sig.loc[date].dropna()
    srows = []
    for k in (8, 15):
        for mode in W_MODES:
            s = WeightedBufferedTopK(k, 2.0, mode, vol=ctx.vol60)
            w = s.get_weights_at(date, vals)
            if w.empty:
                continue
            srows.append(dict(k=k, mode=mode, n=len(w), w_max=float(w.max()),
                              w_min=float(w.min()), hhi=float((w ** 2).sum()),
                              dev_max=float((w - 1.0 / k).abs().max())))
    pd.DataFrame(srows).to_csv(out / "weight_shape.csv", index=False)
    print("\n权重形态（证明权重组确实改了权重）：")
    print(pd.DataFrame(srows).round(4).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
