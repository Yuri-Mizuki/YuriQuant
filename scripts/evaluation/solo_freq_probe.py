"""周频可行性 —— 小资金 solo 方案换调仓频率后，参数量是否需要重估。

回答 2026-11-10 的问题：「现在是月频，周频试过吗？十来只股票周频手动也能操作，
周频上是不是也得重测 horizon 组合 / 融合权重 / K」。

为什么必须重测（而不是把月频参数搬过去）
--------------------------------------
1. **缓冲带的时间尺度变了**：``exit_mult`` 的作用是「已持有的股票排到 TopK×mult
   才卖」。月频一个调仓区间 20 天，周频 5 天 ⇒ 同样的 ``mult`` 每周只给信号
   5 天的容错。把 ``exit_mult=1.0``（= 无缓冲，定版 solo 就是它）直接搬到周频，
   等于**每周重排一次纯 TopK**，换手被最大化。不重扫 ``exit_mult`` 就对周频不公平。
2. **horizon 融合权重的语义变了**：``gbdt_mid13`` 是 h1:h5:h10:h20 = 2:1:1:2，
   加权的是「预测目标的前瞻长度」。持有期从 20 天缩到 5 天，h20 的标签窗口
   跨了 4 个调仓周期 ⇒ 它的信息在周频下会被反复重估。必须重扫。
3. **K 与整手的交互变了**：K 越小，整手欠配越重（定版：10 万 K=6 损 0.57pp、
   K=8 损 0.78pp、K=25 损 2.73pp）；但周频换手高，K 小 ⇒ 单票权重大 ⇒ 每笔
   摩擦占比更高。两个方向相反，只能实测。
4. **相位**：周频固定「周一」是个任意选择。既有记录（``reports/report_replica``）
   显示单相位毛超额可达四相位均值 1.9 倍 ⇒ 必须扫 phase 0~4。

口径
----
production：固定 10bp / T+1 open / 整手 100 / 资金 1e5 / 窗口 2019-01-02..2026-09-07。
调仓日走 ``caliber.rebalance_dates(idx, freq, phase)``（与引擎
``_get_rebalance_days`` 在 ``phase=0`` 时逐位一致，有测试锁）。
**毛收益** = 同参数、费率全 0 的臂（用于把「信号衰减」与「成本吞噬」分开）。

输出 ``reports/solo_weekly/``
-----------------------------
  freq_grid.csv     频率 × 缓冲带 × K 全网格（净 + 毛）
  horizon.csv       周频下的 horizon 组合 / 融合权重
  phase.csv         周频相位 0~4
  phase_month.csv   月频相位 0/1/2/3/4/10/15（``--only month_phase``）
  phase_month_yearly.csv  月频相位逐年收益（同上一段）
  yearly.csv        逐年收益（各臂）
  nav_*.csv         净值（对照组）

用法::

    python -m scripts.evaluation.solo_freq_probe            # 全跑（约 4 分钟）
    python -m scripts.evaluation.solo_freq_probe --quick    # 只跑频率主轴
    python -m scripts.evaluation.solo_freq_probe --only month_phase   # 月频相位
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from backtest.costs import TransactionCosts
from strategy.cross_section import SectorCappedBufferedTopKLongOnly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "solo_weekly"
END = 20260907
LOT = 100
CAPITAL = 1e5
SOLO_K = 8
SOLO_BUF = 1.0

FREQS: list[tuple[str, int]] = [("M", 0), ("2W", 0), ("W", 0),
                                ("W", 1), ("W", 2), ("W", 3), ("W", 4), ("D", 0)]
BUF_GRID = [1.0, 1.5, 2.0, 3.0, 5.0]
K_GRID = [5, 8, 10, 15, 20]

#: horizon 组合（权重 = ``caliber.weighted_rank_average`` 的系数）。
#: ``mid13`` 必须在场 —— 它是定版 ``SIGNALS["gbdt_mid13"]``，用于自检。
HORIZON_COMBOS: dict[str, dict[int, float]] = {
    "h1": {1: 1.0},
    "h5": {5: 1.0},
    "h10": {10: 1.0},
    "h20": {20: 1.0},
    "h1h5": {1: 1.0, 5: 1.0},
    "h10h20": {10: 1.0, 20: 1.0},
    "h1h20": {1: 1.0, 20: 1.0},
    "avg4": {1: 1.0, 5: 1.0, 10: 1.0, 20: 1.0},
    "mid13": {1: 2.0, 5: 1.0, 10: 1.0, 20: 2.0},
    "h1heavy": {1: 4.0, 5: 1.0, 10: 1.0, 20: 1.0},
    "h20heavy": {1: 1.0, 5: 1.0, 10: 1.0, 20: 4.0},
}

ZERO = TransactionCosts(commission_rate=0.0, commission_min=0.0,
                        stamp_duty=0.0, slippage_bp=0.0)


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    """未复权收盘价，ffill 后 **shift(1)**（引擎在执行日 E 取价 ⇒ 拿到 T 日价格）。"""
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return (raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
            .ffill().shift(1))


def industry_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / "industry.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def load_base_panels() -> dict[int, pd.DataFrame]:
    out = {}
    for h in (1, 5, 10, 20):
        key = f"gbdt_h{h}"
        out[h] = pd.read_parquet(ROOT / cal.SIGNALS[key]["path"])
    return out


def blend(base: dict[int, pd.DataFrame], combo: dict[int, float]) -> pd.DataFrame:
    """按 ``caliber.weighted_rank_average`` 构造（复合信号的唯一口径，禁另写平均）。"""
    hs = [h for h, w in combo.items() if w]
    return cal.weighted_rank_average([base[h] for h in hs], [combo[h] for h in hs])


def stats(res, n_reb: int) -> dict:
    m = cal.metrics_row(res, CAPITAL)
    dr = res.daily_returns
    yr = (1 + dr).groupby(dr.index.year).prod() - 1
    mon = (1 + dr).resample("ME").prod() - 1
    wh = res.weights_history
    inv = wh.sum(axis=1)
    nh = (wh > 1e-9).sum(axis=1)
    years = len(dr) / 252.0
    return dict(
        ann=m["ann"], sharpe=m["sharpe"], sharpe_arith=m["sharpe_arith"],
        vol=m["vol"], max_dd=m["max_dd"],
        calmar=(m["ann"] / m["max_dd"]) if m["max_dd"] > 0 else np.nan,
        turnover=m["turnover"], turnover_ann=m["turnover"] * n_reb / years,
        cost_bp=m["cost_bp"], sharpe_t=m["sharpe_t_stat"],
        invest_mean=float(inv.mean()), invest_min=float(inv.min()),
        nhold_mean=float(nh.mean()), nhold_min=int(nh.min()),
        monthly_win=float((mon > 0).mean()), year_worst=float(yr.min()),
        n_year=int(len(yr)), year_pos=int((yr > 0).sum()),
        # 期末净值（元）：< 5e4 说明本金被打到买不起整手，此后的年化不可比
        # （2026-11-10 实测：周频 h1 单用 2019 年末净值 0.4 元 ⇒ 22% 交易日被动空仓）。
        nav_end=float(res.equity_curve.iloc[-1]),
    )


def main(quick: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    dates = ctx.pred.index
    print(f"窗口 {dates[0].date()}..{dates[-1].date()} ({ctx.n_days} 日) | "
          f"资金 {CAPITAL:,.0f} | 整手 {LOT}", flush=True)
    for f, p in FREQS:
        n = len(cal.rebalance_dates(dates, freq=f, phase=p))
        print(f"  调仓频率 {f:3s} phase={p}: {n} 期（≈{n / (ctx.n_days / 252):.1f} 期/年）",
          flush=True)

    base = load_base_panels()
    # 自检：探针构造的 mid13 必须与登记信号面板逐位一致
    mid = blend(base, HORIZON_COMBOS["mid13"])
    chk = (mid - ctx.pred).abs().max().max()
    print(f"  自检：blend(mid13) vs SIGNALS[gbdt_mid13] max|Δ| = {chk:.3e}", flush=True)
    assert chk < 1e-12, "复合信号构造与 caliber 登记不一致"

    rows: list[dict] = []
    yearly: list[dict] = []

    def run(tag: str, sig: pd.DataFrame | None, freq: str, phase: int,
            k: int, buf: float, gross: bool = False, keep_nav: bool = False):
        n_reb = len(cal.rebalance_dates(dates, freq=freq, phase=phase))
        strat = SectorCappedBufferedTopKLongOnly(
            ind, k=k, exit_mult=buf, max_per_sector=1)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq=freq,
                                initial_capital=CAPITAL,
                                costs=(ZERO if gross else cost))
        res = cal.run_once(bt, ctx, "open", signal=sig, freq=freq, phase=phase,
                           price_panel=px, lot_size=LOT)
        r = {**stats(res, n_reb), "arm": tag, "freq": freq, "phase": phase,
             "k": k, "buf": buf, "gross": gross, "n_reb": n_reb,
             "years": len(res.daily_returns) / 252.0}
        rows.append(r)
        if keep_nav:
            (1 + res.daily_returns).cumprod().to_frame("nav").to_csv(
                OUT / f"nav_{tag}.csv")
        dr = res.daily_returns
        for y, v in (1 + dr).groupby(dr.index.year).prod().items():
            yearly.append({"arm": tag, "year": int(y), "ret": float(v - 1)})
        print(f"  {tag:<26s} {freq}{phase} k={k:<2d} b={buf:<4g} "
              f"ann={r['ann'] * 100:7.2f}% SR={r['sharpe_arith']:6.3f} "
              f"DD={r['max_dd'] * 100:5.1f}% 换手/年={r['turnover_ann']:6.2f} "
              f"实投={r['invest_mean']:.3f} 持仓={r['nhold_mean']:.1f}"
              f"{'  [毛]' if gross else ''}", flush=True)
        return r

    # ---- 1) 频率主轴（其余 = 定版 solo 参数），净 + 毛 ----
    print("\n[1] 频率主轴（k=8, buf=1.0 = 定版 solo 参数）", flush=True)
    for f, p in FREQS:
        run(f"freq_{f}p{p}", None, f, p, SOLO_K, SOLO_BUF,
            keep_nav=(f, p) in (("M", 0), ("W", 0), ("W", 2)))
    for f, p in [("M", 0), ("2W", 0), ("W", 0)]:
        run(f"freq_{f}p{p}_gross", None, f, p, SOLO_K, SOLO_BUF, gross=True)
    pd.DataFrame(rows).to_csv(OUT / "freq_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
    if quick:
        return

    # ---- 2) 频率 × 缓冲带（周频自己的缓冲带必须重扫）----
    print("\n[2] 频率 × 缓冲带（k=8）", flush=True)
    for f in ("M", "2W", "W"):
        for b in BUF_GRID:
            if f == "M" and b == SOLO_BUF:
                continue
            if f == "W" and b == SOLO_BUF:
                continue
            run(f"fb_{f}_b{b:g}", None, f, 0, SOLO_K, b)
            if f == "W":
                run(f"fb_{f}_b{b:g}_gross", None, f, 0, SOLO_K, b, gross=True)

    # ---- 3) 周频 K（在周频自己最优缓冲带上）----
    grid = pd.DataFrame(rows)
    w = grid[(grid.freq == "W") & (~grid.gross)]
    best_buf = float(w.loc[w.ann.idxmax(), "buf"]) if len(w) else 3.0
    print(f"\n[3] 周频 K 网格（buf={best_buf:g}）", flush=True)
    for k in K_GRID:
        run(f"wk_W_b{best_buf:g}_k{k}", None, "W", 0, k, best_buf)
    run(f"wk_W_b{best_buf:g}_k{SOLO_K}_gross", None, "W", 0, SOLO_K, best_buf, gross=True)

    # ---- 4) 周频 horizon 组合 / 融合权重 ----
    print(f"\n[4] 周频 horizon 组合（k={SOLO_K}, buf={best_buf:g}）", flush=True)
    hrows = []
    for name, combo in HORIZON_COMBOS.items():
        sig = blend(base, combo)
        r = run(f"hz_W_{name}", sig, "W", 0, SOLO_K, best_buf)
        hrows.append({"combo": name, "weights": str(combo),
                      **{k: r[k] for k in ("ann", "sharpe_arith", "max_dd",
                                           "turnover_ann", "calmar",
                                           "year_worst", "invest_mean")}})
        r0 = run(f"hz_M_{name}", sig, "M", 0, SOLO_K, SOLO_BUF)
        hrows[-1].update({f"M_{k}": r0[k] for k in ("ann", "sharpe_arith", "max_dd")})
    pd.DataFrame(hrows).to_csv(OUT / "horizon.csv", index=False, encoding="utf-8-sig")

    # ---- 5) 周频相位 ----
    print("\n[5] 周频相位（每日历周内第 n 个交易日调仓）", flush=True)
    prows = []
    for p in range(5):
        r = run(f"ph_W_p{p}", None, "W", p, SOLO_K, best_buf)
        prows.append({"phase": p, **{k: r[k] for k in
                                    ("ann", "sharpe_arith", "max_dd",
                                     "turnover_ann", "calmar")}})
    pd.DataFrame(prows).to_csv(OUT / "phase.csv", index=False, encoding="utf-8-sig")

    allr = pd.DataFrame(rows)
    allr.to_csv(OUT / "freq_grid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False, encoding="utf-8-sig")
    print(f"\n产物 → {OUT}", flush=True)


def month_phase() -> None:
    """月频相位扫描 —— 「调仓放在当月第几个交易日」的敏感性。

    为什么单独一段（2026-11-10）：README §5 的月频行（35.57 / 23.32 / 22.98 /
    17.72 / 32.68 / 18.78 / 28.28）此前**只存在于文字里**，``phase.csv`` 只有周频
    0~4 ⇒ 数字无法复核。本段把它落成 ``phase_month.csv`` + ``phase_month_yearly.csv``。

    为什么还要逐年（不只是总年化）：总年化最优只说明「八年合起来最好」，不说明
    「每年都好」。实测 phase=0 八年里只夺魁 1 年（2020），但**逐年收益中位数**
    在七格里最高、且 8/8 年正收益 ⇒ 它的优势是「稳」而不是「尖」。
    只用总年化一句话概括会掩盖这个结构，故两表并出。

    用法：``python -m scripts.evaluation.solo_freq_probe --only month_phase``
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    dates = ctx.pred.index
    print(f"月频相位扫描 | 窗口 {dates[0].date()}..{dates[-1].date()} "
          f"| 资金 {CAPITAL:,.0f} | 整手 {LOT} | k={SOLO_K} b={SOLO_BUF}",
          flush=True)

    prows: list[dict] = []
    yrows: list[dict] = []
    for p in (0, 1, 2, 3, 4, 10, 15):
        n_reb = len(cal.rebalance_dates(dates, freq="M", phase=p))
        strat = SectorCappedBufferedTopKLongOnly(
            ind, k=SOLO_K, exit_mult=SOLO_BUF, max_per_sector=1)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                initial_capital=CAPITAL, costs=cost)
        res = cal.run_once(bt, ctx, "open", signal=None, freq="M", phase=p,
                           price_panel=px, lot_size=LOT)
        r = stats(res, n_reb)
        prows.append({"freq": "M", "phase": p, "n_reb": n_reb,
                      **{k: r[k] for k in ("ann", "sharpe_arith", "max_dd",
                                           "turnover_ann", "calmar",
                                           "invest_mean", "nhold_mean")}})
        dr = res.daily_returns
        for y, v in (1 + dr).groupby(dr.index.year).prod().items():
            yrows.append({"phase": p, "year": int(y), "ret": float(v - 1)})
        print(f"  phase={p:<3} 期数={n_reb:<4} ann={r['ann'] * 100:7.2f}% "
              f"SR_arith={r['sharpe_arith']:6.3f} DD={r['max_dd'] * 100:5.1f}% "
              f"实投={r['invest_mean']:.3f} 持仓={r['nhold_mean']:.1f}", flush=True)

    pm = pd.DataFrame(prows)
    pm.to_csv(OUT / "phase_month.csv", index=False, encoding="utf-8-sig")
    ym = pd.DataFrame(yrows).pivot(index="year", columns="phase", values="ret")
    ym.to_csv(OUT / "phase_month_yearly.csv", encoding="utf-8-sig")

    best, worst = pm.loc[pm.ann.idxmax()], pm.loc[pm.ann.idxmin()]
    print(f"\n  最优 phase={int(best.phase)} {best.ann * 100:.2f}%"
          f" | 最差 phase={int(worst.phase)} {worst.ann * 100:.2f}%"
          f" | 极差 {(best.ann - worst.ann) * 100:.2f}pp", flush=True)
    won = (ym.idxmax(axis=1) == 0).sum()
    print(f"  phase=0 逐年夺魁 {won}/{len(ym)} 年；"
          f"逐年中位数 {ym[0].median() * 100:.2f}%"
          f"（七相位中位数的最大值 {ym.median().max() * 100:.2f}%"
          f" @phase{int(ym.median().idxmax())}）", flush=True)
    print(f"\n产物 → {OUT / 'phase_month.csv'}"
          f" / {OUT / 'phase_month_yearly.csv'}", flush=True)


def bestof() -> None:
    """补扫：双周频全参数 + 「周频各维度都取最优」的组合上界。

    主网格的每一段都改一个维度、其余锁在月频定版参数上 ⇒ 单看某一段会低估周频
    （「周频 + 它自己的缓冲带 + 它自己的 horizon」从没被一起测过）。本段给出
    **周频可达上界**（在已扫过的离散格上取最优），避免「不公平对照」的指控。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    dates = ctx.pred.index
    base = load_base_panels()
    rows: list[dict] = []

    def run(tag, sig, freq, phase, k, buf, gross=False):
        n_reb = len(cal.rebalance_dates(dates, freq=freq, phase=phase))
        strat = SectorCappedBufferedTopKLongOnly(ind, k=k, exit_mult=buf,
                                                 max_per_sector=1)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq=freq,
                                initial_capital=CAPITAL,
                                costs=(ZERO if gross else cost))
        res = cal.run_once(bt, ctx, "open", signal=sig, freq=freq, phase=phase,
                           price_panel=px, lot_size=LOT)
        r = {**stats(res, n_reb), "arm": tag, "freq": freq, "phase": phase,
             "k": k, "buf": buf, "gross": gross, "n_reb": n_reb,
             "years": len(res.daily_returns) / 252.0}
        rows.append(r)
        print(f"  {tag:<24s} {freq}{phase} k={k:<2d} b={buf:<4g} "
              f"ann={r['ann'] * 100:7.2f}% SR={r['sharpe_arith']:6.3f} "
              f"DD={r['max_dd'] * 100:5.1f}% 换手/年={r['turnover_ann']:6.2f} "
              f"实投={r['invest_mean']:.3f} 持仓={r['nhold_mean']:.1f}"
              f"{'  [毛]' if gross else ''}", flush=True)
        return r

    print("[A] 双周频 2W × 缓冲带 × K", flush=True)
    for b in [1.0, 2.0, 3.0, 5.0]:
        for k in [8, 10, 15]:
            run(f"2w_b{b:g}_k{k}", None, "2W", 0, k, b)

    print("\n[B] 周频「各维度取最优」上界", flush=True)
    h1h = blend(base, HORIZON_COMBOS["h1heavy"])
    mid = blend(base, HORIZON_COMBOS["mid13"])
    for p in (0, 3):
        for name, sig in (("mid13", mid), ("h1heavy", h1h)):
            for k in (8, 10):
                run(f"best_W_p{p}_{name}_k{k}", sig, "W", p, k, 5.0)

    pd.DataFrame(rows).to_csv(OUT / "bestof.csv", index=False, encoding="utf-8-sig")
    print(f"\n产物 → {OUT / 'bestof.csv'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑频率主轴")
    ap.add_argument("--only", default=None, choices=["bestof", "month_phase"],
                    help="只跑指定补扫段")
    a = ap.parse_args()
    if a.only == "bestof":
        bestof()
    elif a.only == "month_phase":
        month_phase()
    else:
        main(a.quick)
