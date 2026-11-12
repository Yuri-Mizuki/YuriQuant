"""solo 档的**周期组合**再检验：在 solo 真实执行层下重做 horizon 消融。

为什么需要重做
--------------
``yqcore-solo`` 的信号 ``gbdt_mid13``（h1:h5:h10:h20 = 2:1:1:2）来自
``reports/focus_upgrade/``。但那份证据的口径**不是 solo 的执行层**：

| 项 | focus_upgrade 的裁定 | solo 的实际 |
|---|---|---|
| 策略 | ``BufferedTopKLongOnly(k, exit_mult=2.0)`` | 见下 |
| 策略(续) | — | ``SectorCappedBufferedTopKLongOnly`` ``(k=8, b=1.0, max_per_sector=1)`` |
| 行业约束 | **无** | 每申万一级 ≤1 只 |
| 资金 | 1e6 | **1e5** |
| 整手 | **无** | **100 股** |
| 实际持仓 | = k（6~30） | 平均 **5.88** 只（k=8 经行业去重） |

三项差异各自都能改变周期选择：① 行业去重把有效持仓从 k 压到 ~5.9，等价于把
「有效 K」压小 —— 而 K 越小，短周期（h1）权重越容易过重；② 整手会丢高价股，
不同周期选出的股票价格分布不同；③ 1e5 的实投率 97.8% 把差异等比缩。
⇒ **原结论在 solo 上未必成立，必须单独测。**

本脚本做什么
------------
同一份 ``ctx``（同窗口 / 同成本 / 同成交价 / 同掩码），只换信号面板：

- **主网格（solo 真实层）**：``SectorCappedBufferedTopKLongOnly(k in KS,
  exit_mult=BUF, max_per_sector=1)`` + 整手 100 + 1e5。
- **对照网格（focus_upgrade 原层）**：``BufferedTopKLongOnly(k, exit_mult=2.0)``
  + 无整手 + 1e6 —— 用于验证「本脚本能复现原结论」，否则说明口径接错了。

周期臂（覆盖「两周期」这一族）
------------------------------
``h1h5_prev``   h1+h5 等权（`v1w-ortho882` 工作基线，即「两周期」）
``h1h5``        h1+h5 等权（显式构造，验证与落盘面板一致）
``h1h20``       h1+h20 等权
``h5h20``       h5+h20 等权
``h1020_w``     h1+h20 加权 2:1（h1 主导）
``mid13``       h1:h5:h10:h20 = 2:1:1:2（**当前定版**）
``avg4``        h1:h5:h10:h20 等权
``h1`` / ``h20`` 单周期（边界锚点）
``drop_h5`` / ``drop_h10`` / ``drop_h20`` / ``drop_h1``（去一臂）

判据（写死在这里，跑完照判，不许事后改）
----------------------------------------
1. **A 组（solo 真实层）**：若某个两周期臂在 `k=8, b=1.0` 这一格**同时**年化
   ≥ mid13 且算术 Sharpe ≥ mid13，则该臂构成替代候选；否则 mid13 成立。
2. **稳健性**：结论必须**跨 K 成立**（k∈{6,8,10,12} 里至少 3 格同向），
   单格胜不算 —— 见 ``focus_upgrade`` §B8.5「K 不可按历史最优格挑」。
3. **配对而非均值**：逐相位（``phase=0..5``）配对差 + 年度轴 t 值。相位是共同
   冲击，相位轴 t 不可读（记忆 §四已立）。
4. **B 组必须复现原结论**：若 B 组下 mid13 不再胜过 h1h5_prev，说明本脚本口径
   与原实验不一致，此时 A 组结论也不可信 ⇒ 先查口径，不许拿 A 组交差。

产物（``reports/solo_signal/``）
--------------------------------
  grid_solo.csv    A 组全网格（solo 真实层）
  grid_focus.csv   B 组全网格（focus_upgrade 原层，复现用）
  summary.csv      每个臂在 k=8/b=1.0（A）/ k=15/b=2.0（B）的读数
  yearly.csv       逐年收益（A 组 k=8 b=1.0）

用法::

    python -m scripts.evaluation.solo_signal_probe              # 全跑（约 3 分钟）
    python -m scripts.evaluation.solo_signal_probe --quick      # 只主干（约 50 秒）
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

import scripts.evaluation.caliber as cal  # noqa: E402
from strategy.cross_section import (  # noqa: E402
    BufferedTopKLongOnly,
    SectorCappedBufferedTopKLongOnly,
)

OUT = ROOT / "reports" / "solo_signal"
END = 20260907          # 882 pred 面板末日（与 focus_micro / focus_upgrade 对齐）
LOT = 100
SOLO_K, SOLO_BUF = 8, 1.0        # solo 定版
FOCUS_K, FOCUS_BUF = 15, 2.0     # focus_upgrade 采纳格

# 周期臂：name -> {分量: 权重}
ARMS: dict[str, dict[str, float]] = {
    "h1h5":        {"gbdt_h1": 1, "gbdt_h5": 1},
    "h1h20":       {"gbdt_h1": 1, "gbdt_h20": 1},
    "h5h20":       {"gbdt_h5": 1, "gbdt_h20": 1},
    "h1020_w":     {"gbdt_h1": 2, "gbdt_h20": 1},
    "h5h10":       {"gbdt_h5": 1, "gbdt_h10": 1},
    "mid13":       {"gbdt_h1": 2, "gbdt_h5": 1, "gbdt_h10": 1, "gbdt_h20": 2},
    "avg4":        {"gbdt_h1": 1, "gbdt_h5": 1, "gbdt_h10": 1, "gbdt_h20": 1},
    "drop_h5":     {"gbdt_h1": 1, "gbdt_h10": 1, "gbdt_h20": 1},
    "drop_h10":    {"gbdt_h1": 1, "gbdt_h5": 1, "gbdt_h20": 1},
    "drop_h20":    {"gbdt_h1": 1, "gbdt_h5": 1, "gbdt_h10": 1},
    "drop_h1":     {"gbdt_h5": 1, "gbdt_h10": 1, "gbdt_h20": 1},
    "h1":          {"gbdt_h1": 1},
    "h20":         {"gbdt_h20": 1},
}
# 两周期族（用于「两周期够不够」这个问题的直接回答）
TWO_HORIZON = ("h1h5", "h1h20", "h5h20", "h5h10", "h1020_w")


def _panels() -> dict[str, pd.DataFrame]:
    """加载四个单周期 pred 面板（只在有值的格上取交集由 weighted_rank_average 管）。"""
    out = {}
    for k in ("gbdt_h1", "gbdt_h5", "gbdt_h10", "gbdt_h20"):
        p = ROOT / cal.SIGNALS[k]["path"]
        if not p.exists():
            raise FileNotFoundError(f"分量面板缺失：{p}")
        out[k] = pd.read_parquet(p)
    return out


def build_arms(idx, cols) -> dict[str, pd.DataFrame]:
    raw = _panels()
    arms = {}
    for name, spec in ARMS.items():
        panels = [raw[k].reindex(index=idx, columns=cols) for k in spec]
        arms[name] = cal.weighted_rank_average(panels, list(spec.values()))
    # `h1h5_prev`：落盘实体（v1w-ortho882 工作基线），用来验证显式构造 ≡ 落盘
    prev = ROOT / cal.SIGNALS["ens_h1h5"]["path"]
    if prev.exists():
        arms["h1h5_prev"] = pd.read_parquet(prev).reindex(index=idx, columns=cols)
    return arms


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    """未复权收盘价 ffill 后 shift(1) —— 引擎在执行日 E 取价 ⇒ 拿到 T 日价格。"""
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return (raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
            .ffill().shift(1))


def industry_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / "industry.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


# ---------------------------------------------------------------------------
# IC 口径（2026-11-11 修：此前全项目报的「IC」是 **h=1 日频**，与月频调仓不匹配）
# ---------------------------------------------------------------------------
def monthly_forward(ctx: cal.CaliberContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """**持有期**前瞻收益（非重叠）—— 月频调仓的真正预测目标。

    此前项目里说的「IC = 次日截面排序有多准」是 ``h=1`` 日频口径：用**明天**的
    收益去评今天的排序。但 ``yqcore-solo`` 是**月频**调仓 —— 信号日 T 收盘选股、
    T+1 开盘成交、持有到下一个调仓日。持有期 22~39 天（中位 30），**与 h=1 毫无
    关系**。用 h=1 的 IC 评价月频策略，等于用「当日预测能力」代替「持有期预测能力」。

    本函数给出两套持有期前瞻：

    - ``CC``：T 收盘 → 下一调仓日 T' 收盘。校准/研究口径（与 ``close_adj`` 一致）。
    - ``OO``：T+1 开盘 → T'+1 开盘。**生产口径**（solo 真在 T+1 open 成交）。

    只在调仓日取样（92 期），**非重叠**，所以可以按「月」做配对检验。

    Returns:
        ``(fwd_cc, fwd_oo)``，均为 ``index=调仓信号日``、``columns=股票``。
    """
    idx, cols, close = ctx.pred.index, ctx.pred.columns, ctx.close
    rb = cal.rebalance_dates(idx, freq="M")
    nxt = pd.Series(rb, index=rb).shift(-1).dropna()
    pos = {d: i for i, d in enumerate(idx)}
    ex = {t: idx[pos[t] + 1] for t in rb if pos[t] + 1 < len(idx)}

    f_cc = pd.DataFrame(np.nan, index=rb, columns=cols)
    f_oo = pd.DataFrame(np.nan, index=rb, columns=cols)
    for t, t2 in nxt.items():
        f_cc.loc[t] = close.loc[t2] / close.loc[t] - 1.0
        if t in ex and t2 in ex and ctx.fill is not None:
            f_oo.loc[t] = ctx.fill.loc[ex[t2]] / ctx.fill.loc[ex[t]] - 1.0
    return f_cc, f_oo


def ic_monthly(panel: pd.DataFrame, fwd: pd.DataFrame, rb) -> pd.Series:
    """调仓日截面 Spearman IC（持有期口径，非重叠，n=92）。"""
    return (panel.reindex(rb).corrwith(fwd.reindex(rb), axis=1, method="spearman")
            .dropna())


def ic_daily_h1(panel: pd.DataFrame, ctx: cal.CaliberContext) -> pd.Series:
    """**旧口径**：h=1 日频 IC（``ctx.fwd`` 是当期收益 ⇒ 必须 ``shift(-1)``）。

    ⚠️ 保留只为对照 —— 它评的是「次日排序」，**不是**月频调仓的预测目标。
    2026-11-11 前全项目（含本脚本首版）用的都是这个，导致「IC 与组合反向」的
    错误结论（见 ``reports/solo_signal/README.md`` §2.5）。
    """
    return (panel.corrwith(ctx.fwd.shift(-1), axis=1, method="spearman").dropna())


def nw_t(x, lags: int = 3) -> tuple[float, float]:
    """Newey-West t（月频配对差，滞后 lags 期）。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    m, e = x.mean(), x - x.mean()
    s = e @ e / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        s += 2 * w * (e[lag:] @ e[:-lag]) / n
    return float(m), float(m / np.sqrt(s / n))


def ic_block(arms: dict[str, pd.DataFrame], ctx: cal.CaliberContext,
             fwd_cc, fwd_oo, rb, ann: dict[str, float]) -> pd.DataFrame:
    """三种 IC 口径并排 + 与组合年化的秩相关（本报告的核心表）。"""
    from scipy.stats import spearmanr

    rows = {}
    for name, panel in arms.items():
        ic_cc = ic_monthly(panel, fwd_cc, rb)
        ic_oo = ic_monthly(panel, fwd_oo, rb)
        ic_d = ic_daily_h1(panel, ctx)
        _, t_cc = nw_t(ic_cc)
        rows[name] = dict(
            ic_daily_h1=ic_d.mean(), n_daily=len(ic_d),
            ic_monthly_cc=ic_cc.mean(), ic_monthly_oo=ic_oo.mean(),
            t_monthly_cc=t_cc, pos_monthly_cc=float((ic_cc > 0).mean()),
            n_monthly=len(ic_cc), ann=ann.get(name, np.nan),
        )
    out = pd.DataFrame(rows).T.sort_values("ann", ascending=False)
    for col in ("ic_daily_h1", "ic_monthly_cc", "ic_monthly_oo"):
        sub = out[["ann", col]].dropna()
        rho, pv = spearmanr(sub["ann"], sub[col])
        out.attrs[f"rho_{col}"] = float(rho)
        out.attrs[f"p_{col}"] = float(pv)
    return out


def _row(res, m: dict, reb: set, ind: pd.DataFrame, tag: str, meta: dict) -> dict:
    dr = res.daily_returns
    mon = (1 + dr).resample("ME").prod() - 1
    yr = (1 + dr).groupby(dr.index.year).prod() - 1
    wh = res.weights_history
    rb = [d for d in wh.index if d in reb]
    whr = wh.loc[rb]
    nh = (whr > 1e-9).sum(axis=1)
    nsec = []
    for d, row in whr.iterrows():
        codes = row.index[row.to_numpy() > 1e-9]
        if len(codes) == 0:
            nsec.append(0)
            continue
        if d not in ind.index:
            nsec.append(1)
            continue
        vals = ind.loc[d].reindex(codes).fillna("__NA__").astype(str)
        nsec.append(len(set(vals)))
    nsec = np.asarray(nsec, float)
    out = dict(
        arm=tag, ann=m["ann"], sharpe=m["sharpe"],
        sharpe_arith=m["sharpe_arith"], sharpe_t=m["sharpe_t_stat"],
        vol=m["vol"], max_dd=m["max_dd"],
        calmar=(m["ann"] / m["max_dd"]) if m["max_dd"] > 0 else np.nan,
        turnover=m["turnover"], n_active=m["n_active"],
        invest_mean=float(whr.sum(axis=1).mean()),
        nhold_mean=float(nh.mean()), nhold_min=int(nh.min()),
        nhold_le3=float((nh <= 3).mean()),
        nsector_mean=float(nsec.mean()), nsector_min=int(nsec.min()),
        monthly_win=float((mon > 0).mean()), monthly_worst=float(mon.min()),
        year_worst=float(yr.min()), year_pos=int((yr > 0).sum()),
        n_year=int(len(yr)),
    )
    out.update(meta)
    for y, v in yr.items():
        out[f"y{y}"] = float(v)
    return out


def run_grid(arms, ctx, cost, px, ind, groups: dict, capital: float,
             lot: bool, ks, bufs, reb) -> list[dict]:
    rows = []
    for gname, strat_fn in groups.items():
        for name, panel in arms.items():
            for k in ks:
                for buf in bufs:
                    strat = strat_fn(k, buf)
                    bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                            initial_capital=float(capital),
                                            costs=cost)
                    res = cal.run_once(bt, ctx, "open", signal=panel,
                                       price_panel=(px if lot else None),
                                       lot_size=(LOT if lot else None))
                    rows.append(_row(res, cal._metrics(res, capital), reb, ind,
                                     name, dict(group=gname, k=k, buf=buf,
                                                capital=capital, lot=lot)))
    return rows


def phases(arms, ctx, cost, px, ind, capital, lot, ks, bufs, reb,
           phases_n: int = 6) -> pd.DataFrame:
    """逐相位配对：同一 K/buf 下 mid13 − 每个两周期臂，phase=0..n-1。

    phase 是共同冲击 ⇒ 只能看「同相位内两臂之差」，相位轴 t 值不可读。
    """
    out = []
    for k in ks:
        for buf in bufs:
            for ph in range(phases_n):
                base = None
                for name in ("mid13",) + TWO_HORIZON:
                    if name not in arms:
                        continue
                    strat = SectorCappedBufferedTopKLongOnly(
                        ind, k=k, exit_mult=buf, max_per_sector=1)
                    bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                            initial_capital=float(capital),
                                            costs=cost)
                    res = cal.run_once(bt, ctx, "open", signal=arms[name],
                                       price_panel=(px if lot else None),
                                       lot_size=(LOT if lot else None),
                                       phase=ph)
                    m = cal._metrics(res, capital)
                    rec = dict(k=k, buf=buf, phase=ph, arm=name,
                               ann=m["ann"], sharpe_arith=m["sharpe_arith"])
                    if name == "mid13":
                        base = rec
                    else:
                        out.append({**rec,
                                    "d_ann": (base["ann"] - rec["ann"]
                                              if base else np.nan),
                                    "d_sr": (base["sharpe_arith"]
                                             - rec["sharpe_arith"]
                                             if base else np.nan)})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="只跑 solo k=8 b=1.0 与 focus k=15 b=2.0（约 50 秒）")
    ap.add_argument("--no-phase", action="store_true", help="跳过逐相位配对")
    ap.add_argument("--out", default="reports/solo_signal")
    args = ap.parse_args(argv)

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    ctx = cal.load_context("ens_h1h5", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px, ind = price_panel(ctx), industry_panel(ctx)
    reb = set(cal.next_day_rebalance(ctx.pred.index))
    arms = build_arms(ctx.pred.index, ctx.pred.columns)
    print(f"窗口 {ctx.pred.index[0].date()} .. {ctx.pred.index[-1].date()} "
          f"({ctx.n_days} 日) | {len(arms)} 臂", flush=True)

    # 守卫 0：显式构造的 h1h5 必须与落盘面板高度一致（否则口径接错）
    if "h1h5_prev" in arms:
        a, b = arms["h1h5"], arms["h1h5_prev"]
        both = a.notna() & b.notna()
        rho = float(a.where(both).corrwith(b, axis=1, method="spearman").mean())
        print(f"守卫：h1h5(构造) vs h1h5_prev(落盘) 平均截面 ρ = {rho:.4f}")
        assert rho > 0.99, f"构造与落盘不一致（ρ={rho:.4f}）⇒ 口径接错"

    solo_groups = {
        "solo": lambda k, buf: SectorCappedBufferedTopKLongOnly(
            ind, k=k, exit_mult=buf, max_per_sector=1),
    }
    focus_groups = {
        "focus": lambda k, buf: BufferedTopKLongOnly(k=k, exit_mult=buf),
    }

    if args.quick:
        ks_solo, bufs_solo = [SOLO_K], [SOLO_BUF]
        ks_focus, bufs_focus = [FOCUS_K], [FOCUS_BUF]
    else:
        ks_solo, bufs_solo = [6, 8, 10, 12], [1.0, 1.5]
        ks_focus, bufs_focus = [8, 15, 25], [2.0]

    print("\n[A 组] solo 真实层（行业去重 + 整手100 + 1e5）", flush=True)
    rows_a = run_grid(arms, ctx, cost, px, ind, solo_groups, 1e5, True,
                      ks_solo, bufs_solo, reb)
    print("\n[B 组] focus_upgrade 原层（无行业约束 + 无整手 + 1e6）—— 复现用",
          flush=True)
    rows_b = run_grid(arms, ctx, cost, px, ind, focus_groups, 1e6, False,
                      ks_focus, bufs_focus, reb)

    dfa = pd.DataFrame(rows_a)
    dfb = pd.DataFrame(rows_b)
    dfa.to_csv(outdir / "grid_solo.csv", index=False, encoding="utf-8-sig")
    dfb.to_csv(outdir / "grid_focus.csv", index=False, encoding="utf-8-sig")

    # 汇总：A 组 k=8/b=1.0，B 组 k=15/b=2.0
    key_a = (dfa["k"] == SOLO_K) & (dfa["buf"] == SOLO_BUF)
    key_b = (dfb["k"] == FOCUS_K) & (dfb["buf"] == FOCUS_BUF)
    sa = dfa[key_a].set_index("arm")
    sb = dfb[key_b].set_index("arm")
    summ = pd.DataFrame({
        "ann_solo": sa["ann"], "sr_solo": sa["sharpe_arith"],
        "mdd_solo": sa["max_dd"], "nhold_solo": sa["nhold_mean"],
        "inv_solo": sa["invest_mean"],
        "ann_focus": sb["ann"], "sr_focus": sb["sharpe_arith"],
        "nhold_focus": sb["nhold_mean"],
    }).sort_values("ann_solo", ascending=False)
    summ.to_csv(outdir / "summary.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    print("\n=== 汇总（A=solo 真实层 k8 b1.0 / B=focus 原层 k15 b2.0）===")
    print(summ.round(4).to_string())

    # 跨 K 稳健性：每个臂在 A 组所有 (k,buf) 格上相对 mid13 的胜负
    print("\n=== 跨格稳健性（A 组，相对 mid13 的年化差）===")
    piv = dfa.pivot_table(index=["k", "buf"], columns="arm", values="ann")
    if "mid13" in piv.columns:
        diff = piv.sub(piv["mid13"], axis=0)
        print(diff.round(4).to_string())
        win = (diff > 0).sum()
        print("\n胜格数 / 总格数：")
        print(pd.DataFrame({"win": win, "total": len(diff),
                            "mean_diff": diff.mean().round(4)})
              .sort_values("mean_diff", ascending=False).to_string())

    # ---- IC 三口径对照（本脚本最重要的修正，2026-11-11）----
    print("\n=== IC 三口径对照（日频 h=1 / 月频 CC / 月频 OO）vs 组合年化 ===",
          flush=True)
    rb_m = cal.rebalance_dates(ctx.pred.index, freq="M")
    fwd_cc, fwd_oo = monthly_forward(ctx)
    ann_map = sa["ann"].to_dict()
    icb = ic_block(arms, ctx, fwd_cc, fwd_oo, rb_m, ann_map)
    icb.to_csv(outdir / "ic_by_caliber.csv", encoding="utf-8-sig")
    print(icb.round(5).to_string())
    print("\nIC 口径 vs 组合年化的 Spearman ρ（同一批臂的排序是否一致）：")
    for col in ("ic_daily_h1", "ic_monthly_cc", "ic_monthly_oo"):
        print(f"  {col:<16s} ρ = {icb.attrs[f'rho_{col}']:+.4f}  "
              f"p = {icb.attrs[f'p_{col}']:.4f}")

    # 配对月频 IC 差（mid13 − 每个对照臂），NW-t
    print("\n配对月频 IC 差（mid13 − 对照臂，92 期，NW-t lags=3）：")
    pair_rows = []
    for kind, fwd in (("CC", fwd_cc), ("OO", fwd_oo)):
        mid = ic_monthly(arms["mid13"], fwd, rb_m)
        for name in ("h1h5", "drop_h5", "drop_h10", "avg4", "h1", "h20"):
            if name not in arms:
                continue
            d = (mid - ic_monthly(arms[name], fwd, rb_m)).dropna()
            m, t = nw_t(d)
            pair_rows.append(dict(fwd=kind, pair=f"mid13-{name}", d_ic=m, nw_t=t,
                                  n=len(d)))
            print(f"  {kind}  mid13−{name:<8s} ΔIC = {m:+.5f}  NW-t = {t:+.2f}")
    pd.DataFrame(pair_rows).to_csv(outdir / "ic_pairs.csv", index=False,
                                   encoding="utf-8-sig")

    if not args.no_phase:
        print("\n=== 逐相位配对（solo k8 b1.0，Δ = mid13 − 两周期臂）===",
              flush=True)
        ph = phases(arms, ctx, cost, px, ind, 1e5, True,
                    [SOLO_K], [SOLO_BUF], reb)
        ph.to_csv(outdir / "phases.csv", index=False, encoding="utf-8-sig")
        g = ph.groupby("arm").agg(d_ann_mean=("d_ann", "mean"),
                                  d_ann_pos=("d_ann", lambda s: int((s > 0).sum())),
                                  d_sr_mean=("d_sr", "mean"),
                                  d_sr_pos=("d_sr", lambda s: int((s > 0).sum())),
                                  n=("d_ann", "size"))
        print(g.round(4).to_string())

    print(f"\n产物 → {outdir}")


if __name__ == "__main__":
    main()