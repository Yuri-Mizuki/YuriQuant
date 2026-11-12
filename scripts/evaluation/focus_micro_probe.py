"""小资金（10 万）整手实跑选型 —— K × 缓冲带 × 行业去重 × 仓位。

回答 2026-11-10 的问题：「10 万本金、整手约束、持仓尽量少、要稳、指标还要高，
到底该配哪个方案」。

三件事必须做实，否则数字没意义
------------------------------
1. **整手真的进引擎**：``caliber.run_once(..., price_panel=, lot_size=100)``。
   价格面板 = ``close_raw.ffill().shift(1)``（未复权；引擎在执行日 E 取价，shift 后
   拿到信号日 T 的收盘价 ⇒ 无前视：T 收盘选股、T+1 开盘成交）。整手在引擎里是
   ``trunc(w·capital/price/lot)·lot``，欠配零头留现金 ⇒ **实投率是引擎算的**，
   不是事后折算（``focus_holding_probe`` 是折算口径，会低估损耗）。
2. **稳定性不能只看年化**：并列 Sharpe / MaxDD / Calmar / 最差年 / 月度胜率。
   行业去重臂必须过**随机行业标签安慰剂** —— 否则分不清「真行业信息」与「随机分散」。
3. **「要稳」只有一个诚实答案**：``gross_exposure`` 线性降杠杆（Sharpe 不变、
   收益与回撤等比缩）。波动率目标实测**不如**线性（Calmar 反而降），故不采纳。

口径
----
production（固定 10bp / T+1 open）；资金档默认 1e5；2019-01-02..2026-09-07
（882 pred 面板末日）。信号 ``gbdt_mid13``（复合信号，``caliber.SIGNALS``）。

输出（默认 ``reports/focus_micro/``）
------------------------------------
  grid.csv      全网格（arm × k × buf × 仓位 × 指标，10 万整手口径）
  yearly.csv    逐年收益
  monthly.csv   月度收益
  placebo.csv   随机行业标签 10 次
  roll.csv      2019-23 定参 / 2024-26 验证

用法::

    python -m scripts.evaluation.focus_micro_probe            # 全跑（约 2 分钟）
    python -m scripts.evaluation.focus_micro_probe --quick    # 主干（约 40 秒）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from strategy.cross_section import (
    BufferedTopKLongOnly,
    SectorCappedBufferedTopKLongOnly,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "focus_micro"
END = 20260907
LOT = 100
K_GRID = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25]
BUF_GRID = [1.0, 2.0, 3.0]
SOLO_K = [6, 8, 10, 12]
SOLO_BUF = [1.0, 1.5, 2.0, 3.0]
SCALE = [1.0, 0.85, 0.75, 0.6]
SCALE_K, SCALE_BUF = 6, 1.5
VT_K, VT_TARGETS = 8, [0.25, 0.20, 0.15]


class VolTargetSectorBuffered(SectorCappedBufferedTopKLongOnly):
    """波动率目标版（**判负**，仅留作证据，不进 strategy/）。

    每次调仓用「信号日 T 及之前」的 60 日个股收益估组合波动 ``w'Σw``，
    按 ``target/σ`` 缩放到目标年化波动（clip 到 [0.2, 1]）。
    **只用 <= T 的收益**（``iloc[:-1]`` 去掉执行日 E 当日）⇒ 无前视。
    """

    def __init__(self, sectors, rets, k, exit_mult, target, win=60, min_scale=0.2):
        super().__init__(sectors, k=k, exit_mult=exit_mult, max_per_sector=1)
        self.rets, self.target, self.win, self.min_scale = rets, target, win, min_scale
        self.name = f"voltarget_{k}_{exit_mult}_{target}"

    def get_weights_at(self, date, vals):
        w = super().get_weights_at(date, vals)
        if w.empty:
            return w
        hist = self.rets.loc[:date].iloc[:-1].iloc[-self.win:]
        if len(hist) < 20:
            return w
        R = hist.reindex(columns=w.index).fillna(0.0).to_numpy()
        cov = np.cov(R.T) * 252.0
        sig = float(np.sqrt(max(w.to_numpy() @ cov @ w.to_numpy(), 1e-12)))
        return w * float(np.clip(self.target / sig, self.min_scale, 1.0))


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    """未复权收盘价，ffill 后 **shift(1)**（引擎在执行日 E 取价 ⇒ 拿到 T 日价格）。"""
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return (raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
            .ffill().shift(1))


def industry_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / "industry.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def _stats(res, m: dict, reb: set, ind: pd.DataFrame | None = None) -> dict:
    """回测结果 → 一行指标（含稳定性维度）。

    ``ind`` 传入时额外统计**行业数分布** —— ``max_per_sector=1`` 只管「同质」，
    不管组合会缩到几个行业，故必须把 ``nsector`` 显式量出来。
    """
    dr = res.daily_returns
    mon = (1 + dr).resample("ME").prod() - 1
    yr = (1 + dr).groupby(dr.index.year).prod() - 1
    wh = res.weights_history
    rb = [d for d in wh.index if d in reb]
    whr = wh.loc[rb]
    inv = whr.sum(axis=1)
    nh = (whr > 1e-9).sum(axis=1)
    out = dict(
        ann=m["ann"], sharpe=m["sharpe"], sharpe_arith=m["sharpe_arith"],
        vol=m["vol"], max_dd=m["max_dd"],
        calmar=(m["ann"] / m["max_dd"]) if m["max_dd"] > 0 else np.nan,
        turnover=m["turnover"], cost_bp=m["cost_bp"], n_active=m["n_active"],
        sharpe_t=m["sharpe_t_stat"],
        monthly_win=float((mon > 0).mean()), monthly_worst=float(mon.min()),
        year_worst=float(yr.min()), year_pos=int((yr > 0).sum()), n_year=int(len(yr)),
        invest_mean=float(inv.mean()), invest_min=float(inv.min()),
        nhold_mean=float(nh.mean()), nhold_min=int(nh.min()),
        nhold_le3=float((nh <= 3).mean()),
    )
    if ind is not None:
        nsec = []
        for d, row in whr.iterrows():
            codes = row.index[row.to_numpy() > 1e-9]
            if len(codes) == 0:
                nsec.append(0)
                continue
            if d not in ind.index:
                nsec.append(1)          # 无行业截面 ⇒ 全算 __NA__，只有 1 个「行业」
                continue
            vals = ind.loc[d].reindex(codes).fillna("__NA__").astype(str)
            nsec.append(len(set(vals)))
        nsec = np.asarray(nsec, dtype=float)
        out.update(nsector_mean=float(nsec.mean()), nsector_min=int(nsec.min()),
                   nsector_le2=float((nsec <= 2).mean()))
    return out


def fixed_capital_invest(ctx: cal.CaliberContext, px: pd.DataFrame,
                         ind: pd.DataFrame, k: int, capital: float,
                         lot: int) -> tuple[pd.Series, dict]:
    """固定本金下的逐期实投率 + 持仓集合（整手约束**始终** binding）。

    引擎的 ``invest_mean`` 用**当期 capital** 取整，而 capital 随复利从 10 万涨到近百万
    ⇒ 后期整手几乎不 binding，均值会被摊薄成偏乐观的数字。本函数恒用初始本金，
    给出「利润每月取走、始终 10 万」这一口径下的真实摩擦。

    选股逻辑是**独立另写**的等价实现（max_per_sector=1、不补 k），返回的持仓集合
    在 ``main`` 里与引擎 ``weights_history`` 做 Jaccard 自检（2026-11-10 实测 1.0000），
    防止策略类后续改动造成静默漂移。
    """
    sig = ctx.pred.shift(1).where(ctx.mask)
    rb = sorted(cal.next_day_rebalance(ctx.pred.index))
    prev: set = set()
    out: dict = {}
    holds: dict = {}
    for e in rb:
        if e not in sig.index:
            continue
        vals = sig.loc[e].dropna()
        if vals.empty:
            continue
        ranks = vals.rank(ascending=False, method="first")
        sect = (ind.loc[e] if e in ind.index
                else pd.Series(dtype=object)).reindex(vals.index)

        def _key(code):
            s = sect.get(code)
            return "__NA__" if s is None or pd.isna(s) else str(s)

        counts: dict[str, int] = {}
        keep: list = []
        for code in sorted(prev, key=lambda c: ranks.get(c, np.inf)):
            if code not in ranks.index or ranks[code] > k:
                continue
            key = _key(code)
            if counts.get(key, 0) >= 1:
                continue
            counts[key] = counts.get(key, 0) + 1
            keep.append(code)
            if len(keep) >= k:
                break
        fills: list = []
        for code in ranks.sort_values().index:
            if len(keep) + len(fills) >= k:
                break
            if ranks[code] > k:
                break
            if code in prev and ranks[code] <= k:
                continue
            key = _key(code)
            if counts.get(key, 0) >= 1:
                continue
            counts[key] = counts.get(key, 0) + 1
            fills.append(code)
        port = keep + fills
        prev = set(port)
        holds[e] = set(port)
        prices = px.loc[e]
        tot = 0.0
        for code in port:
            p = prices.get(code, np.nan)
            if not np.isfinite(p) or p <= 0:
                continue
            tot += np.trunc((1.0 / len(port)) * capital / p / lot) * lot * p / capital
        out[e] = tot
    return pd.Series(out).sort_index(), holds


def main(capital: float, quick: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    reb = set(cal.next_day_rebalance(ctx.pred.index))
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} "
          f"({ctx.n_days} 日) | 资金 {capital:,.0f} | 整手 {LOT}", flush=True)

    rows, yearly, monthly, curves = [], [], [], {}

    def run_and_record(tag, strat, with_lot=True, cost_override=None, **meta):
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                initial_capital=float(capital),
                                costs=(cost if cost_override is None else cost_override))
        res = cal.run_once(bt, ctx, "open",
                           price_panel=(px if with_lot else None),
                           lot_size=(LOT if with_lot else None))
        r = {**_stats(res, cal.metrics_row(res, capital), reb, ind), "arm": tag,
             "capital": capital, **meta}
        rows.append(r)
        ms = meta.get("min_sectors") or 0
        mh = meta.get("min_holdings") or 0
        key = (f"{tag}|k{meta.get('k')}|b{meta.get('exit_mult')}"
               f"|g{meta.get('gross_exposure', 1.0):g}"
               + (f"|ms{ms}" if ms else "")
               + (f"|mh{mh}" if mh else ""))
        curves[key] = res.daily_returns
        dr = res.daily_returns
        for y, v in (1 + dr).groupby(dr.index.year).prod().items():
            yearly.append(dict(arm=tag, year=int(y), ret=float(v - 1), **meta))
        for d, v in ((1 + dr).resample("ME").prod() - 1).items():
            monthly.append(dict(arm=tag, month=d.date(), ret=float(v), **meta))
        print(f"  {tag:<10} k={meta.get('k'):>2} buf={meta.get('exit_mult')} "
              f"x{meta.get('gross_exposure', 1.0):.2f} | ann={r['ann']*100:6.2f}% "
              f"SR={r['sharpe_arith']:.3f} DD={r['max_dd']*100:5.1f}% "
              f"Calmar={r['calmar']:.2f} 实投={r['invest_mean']:.3f} "
              f"持仓={r['nhold_mean']:.1f}(min {r['nhold_min']}) "
              f"行业={r['nsector_mean']:.1f}(min {r['nsector_min']}) "
              f"最差年={r['year_worst']*100:5.1f}%", flush=True)
        return r

    # ---- 0) 整手的代价：同参数、无整手（理论上限）----
    for k in [6, 8, 25]:
        run_and_record("nolot", BufferedTopKLongOnly(k=k, exit_mult=1.0),
                       with_lot=False, k=k, exit_mult=1.0,
                       max_per_sector=None, gross_exposure=1.0)

    # ---- 1) 无行业约束基线（K × 缓冲带）----
    for k in ([6, 10, 20] if quick else K_GRID):
        for buf in ([1.5, 2.0] if quick else BUF_GRID):
            run_and_record("plain", BufferedTopKLongOnly(k=k, exit_mult=buf),
                           k=k, exit_mult=buf, max_per_sector=None, gross_exposure=1.0)

    # ---- 2) 行业去重（每申万一级 ≤1 只）----
    for k in ([6, 8] if quick else SOLO_K):
        for buf in ([1.0, 1.5] if quick else SOLO_BUF):
            run_and_record(
                "solo",
                SectorCappedBufferedTopKLongOnly(ind, k=k, exit_mult=buf, max_per_sector=1),
                k=k, exit_mult=buf, max_per_sector=1, gross_exposure=1.0)

    # ---- 2b) 空缺补满 k（对照；实测更差，见 README §2.2b）----
    for k, buf in ((6, 1.5), (8, 2.0)):
        run_and_record(
            "solo_fillk",
            SectorCappedBufferedTopKLongOnly(ind, k=k, exit_mult=buf,
                                             max_per_sector=1, fill_to_k=True),
            k=k, exit_mult=buf, max_per_sector=1, gross_exposure=1.0, fill_to_k=True)

    # ---- 2c) 行业数地板（min_sectors）：管「只剩几个行业」，不管「同质」----
    # max_per_sector=1 只保证同行业不重复；TopK 内可能只有 2~3 个行业 ⇒ 行业 beta 不受控。
    # min_sectors 从 TopK 之外补「新行业」各 1 只（候选排名 ≤ k·mult），补到达标为止。
    for ms in ([3, 4] if quick else [2, 3, 4, 5]):
        run_and_record(
            "solo_mins",
            SectorCappedBufferedTopKLongOnly(
                ind, k=8, exit_mult=1.0, max_per_sector=1, min_sectors=ms),
            k=8, exit_mult=1.0, max_per_sector=1, gross_exposure=1.0, min_sectors=ms)
    # k=6 是唯一会触发地板的一档（实测有 1 期只有 2 个行业），必须单独量代价
    for ms in ([3] if quick else [3, 4]):
        run_and_record(
            "solo_mins",
            SectorCappedBufferedTopKLongOnly(
                ind, k=6, exit_mult=1.5, max_per_sector=1, min_sectors=ms),
            k=6, exit_mult=1.5, max_per_sector=1, gross_exposure=1.0, min_sectors=ms)

    # ---- 2d) 同一「持仓数缺口」的另一种修法：min_holdings（**留现金**，不补股）----
    # 与 min_sectors 触发条件完全相同（max_per_sector=1 下 持仓数 ≡ 行业数），
    # 但处理方式相反：不买低排名股，而是把不足的部分留成现金 ⇒ Sharpe 不动。
    for mh in ([4] if quick else [3, 4, 5]):
        run_and_record(
            "solo_minh",
            SectorCappedBufferedTopKLongOnly(
                ind, k=8, exit_mult=1.0, max_per_sector=1, min_holdings=mh),
            k=8, exit_mult=1.0, max_per_sector=1, gross_exposure=1.0, min_holdings=mh)

    # ---- 3) 仓位系数（线性降杠杆）----
    for sc in ([1.0, 0.75] if quick else SCALE):
        run_and_record(
            "solo_scale",
            SectorCappedBufferedTopKLongOnly(ind, k=SCALE_K, exit_mult=SCALE_BUF,
                                             max_per_sector=1, gross_exposure=sc),
            k=SCALE_K, exit_mult=SCALE_BUF, max_per_sector=1, gross_exposure=sc)

    # ---- 3b) 波动率目标（判负对照）----
    for tgt in ([0.20] if quick else VT_TARGETS):
        run_and_record(
            "voltarget",
            VolTargetSectorBuffered(ind, ctx.fwd, k=VT_K, exit_mult=1.0, target=tgt),
            k=VT_K, exit_mult=1.0, max_per_sector=1, gross_exposure=1.0, vt_target=tgt)

    # ---- 3c) 成本敏感性：逐股平方根冲击（impact 压力口径）----
    # 10 万小资金 + 偏大盘股票的参与率极低 ⇒ impact 通常比 production 的固定 10bp 便宜。
    # 若反直觉（impact 更贵），说明 10bp 假设偏乐观，须以 impact 为准。
    if not quick:
        run_and_record(
            "solo_impact",
            SectorCappedBufferedTopKLongOnly(ind, k=8, exit_mult=1.0, max_per_sector=1),
            k=8, exit_mult=1.0, max_per_sector=1, gross_exposure=1.0,
            cost_override=cal.cost_profile("impact"))

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "grid.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "yearly.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly.csv", index=False)

    # ---- 3d) 实投率口径：引擎按当期 capital（随复利增长）取整 ⇒ 均值偏乐观 ----
    if not quick:
        fx, holds = fixed_capital_invest(ctx, px, ind, k=8,
                                        capital=float(capital), lot=LOT)
        fx.rename("invest").to_csv(OUT / "invest_fixed.csv")
        eng = grid[(grid.arm == "solo") & (grid.k == 8) & (grid.exit_mult == 1.0)]
        e_mean = float(eng.invest_mean.iloc[0]) if len(eng) else float("nan")
        print(f"\n实投率口径：固定 {capital:,.0f} 本金 mean {fx.mean():.4f} / "
              f"min {fx.min():.4f}（引擎浮动资金 mean {e_mean:.4f}）"
              f" ⇒ 差 {(e_mean - fx.mean())*100:+.2f}pp", flush=True)

        # ---- 逐期持仓/行业分布：回答「组合会不会缩到 2~3 只」----
        prof = []
        for d, s in sorted(holds.items()):
            sect = (ind.loc[d].reindex(sorted(s)) if d in ind.index
                    else pd.Series(dtype=object))
            keys = (set(sect.fillna("__NA__").astype(str))
                    if len(sect) else {"__NA__"})
            prof.append(dict(date=d.date(), nhold=len(s), nsector=len(keys)))
        prof = pd.DataFrame(prof)
        prof.to_csv(OUT / "holdings_profile.csv", index=False)
        vc = prof.nhold.value_counts().sort_index()
        print("持仓只数分布：" + "  ".join(f"{int(k)}只×{int(v)}期"
                                       for k, v in vc.items())
              + f"（共 {len(prof)} 期；≤3 只占 {prof.nhold.le(3).mean()*100:.1f}%）",
              flush=True)

        # 审计自检：独立另写的选股必须与引擎持仓逐期一致（防策略类静默漂移）
        chk = SectorCappedBufferedTopKLongOnly(ind, k=8, exit_mult=1.0, max_per_sector=1)
        bt = cal.VectorBacktest(strategy=chk, rebalance_freq="M",
                                initial_capital=float(capital), costs=cost)
        res_chk = cal.run_once(bt, ctx, "open", price_panel=px, lot_size=LOT)
        wh = res_chk.weights_history
        jac = np.array([
            len(s & set(wh.loc[d][wh.loc[d] > 1e-12].index))
            / max(len(s | set(wh.loc[d][wh.loc[d] > 1e-12].index)), 1)
            for d, s in holds.items() if d in wh.index
        ])
        print(f"审计自检：独立复算 vs 引擎持仓 Jaccard min {jac.min():.4f} / "
              f"mean {jac.mean():.4f}（{len(jac)} 期）", flush=True)
        assert jac.min() > 0.99, "独立复算与引擎持仓不一致 ⇒ 策略实现可能已漂移"

    # ---- 4) 安慰剂：随机行业标签 ----
    if not quick:
        print("\n安慰剂：随机打乱行业标签（每行内 shuffle，保留分布）", flush=True)
        rng = np.random.default_rng(2026)
        arr = ind.to_numpy(dtype=object)
        plc = []
        for i in range(10):
            shuf = arr.copy()
            for r_i in range(shuf.shape[0]):
                rng.shuffle(shuf[r_i])
            st = SectorCappedBufferedTopKLongOnly(
                pd.DataFrame(shuf, index=ind.index, columns=ind.columns),
                k=SCALE_K, exit_mult=SCALE_BUF, max_per_sector=1)
            bt = cal.VectorBacktest(strategy=st, rebalance_freq="M",
                                    initial_capital=float(capital), costs=cost)
            res = cal.run_once(bt, ctx, "open", price_panel=px, lot_size=LOT)
            m = cal.metrics_row(res, capital)
            plc.append(dict(draw=i, ann=m["ann"], sharpe=m["sharpe_arith"],
                            max_dd=m["max_dd"]))
            print(f"  随机 {i+1:>2}: ann={m['ann']*100:6.2f}% SR={m['sharpe_arith']:.3f} "
                  f"DD={m['max_dd']*100:5.1f}%", flush=True)
        plc = pd.DataFrame(plc)
        plc.to_csv(OUT / "placebo.csv", index=False)
        base = grid[(grid.arm == "solo")
                    & (grid.k == SCALE_K) & (grid.exit_mult == SCALE_BUF)]
        if len(base):
            real_ann, real_sr = float(base.ann.iloc[0]), float(base.sharpe_arith.iloc[0])
            print(f"\n真实 {real_ann*100:.2f}% / SR {real_sr:.3f} | "
                  f"随机 {plc.ann.mean()*100:.2f}%±{plc.ann.std()*100:.2f}pp "
                  f"(max {plc.ann.max()*100:.2f}%) | 分位 "
                  f"ann {(plc.ann < real_ann).mean()*100:.0f}% "
                  f"SR {(plc.sharpe < real_sr).mean()*100:.0f}%", flush=True)

    # ---- 5) 滚动窗口：2019-23 定参 / 2024-26 验证 ----
    print("\n滚动窗口（同一套参数，时间切分）", flush=True)
    roll = []
    for key, dr in curves.items():
        def f(x):
            return float(np.prod(1 + x) ** (252 / len(x)) - 1)

        is_ = dr[dr.index.year <= 2023]
        oos = dr[dr.index.year >= 2024]
        roll.append(dict(key=key, is_ann=f(is_.values), oos_ann=f(oos.values),
                         diff=f(oos.values) - f(is_.values)))
        print(f"  {key:<36} IS={roll[-1]['is_ann']*100:6.2f}% "
              f"OOS={roll[-1]['oos_ann']*100:6.2f}% Δ={roll[-1]['diff']*100:+6.2f}pp",
              flush=True)
    pd.DataFrame(roll).to_csv(OUT / "roll.csv", index=False)

    # ---- 6) 参数滚动重估：每年只用「截止上年末」的数据挑 (k, buf)，再吃下一年的收益 ----
    # 与 §5 的时间切分不同：这里**每年真的重选一次参数**，拼成一条连续的 OOS 曲线。
    # 搜索空间 = 已有 solo 臂（SOLO_K × SOLO_BUF），不额外回测 ⇒ 无重复劳动。
    # 切换成本未单独计（每年至多一次接近全换手，月频换手本身 ~92%/年，量级 < 1 个月）。
    if not quick:
        print("\n参数滚动重估（每年用过去数据挑 Sharpe 最高的 (k,buf)）", flush=True)
        combos = {(k, b): curves[f"solo|k{k}|b{b}|g1"]
                  for k in SOLO_K for b in SOLO_BUF
                  if f"solo|k{k}|b{b}|g1" in curves}

        def _sr(x: pd.Series) -> float:
            sd = x.std()
            return float(x.mean() / sd * np.sqrt(252)) if sd and sd > 0 else np.nan

        def _spliced(mode: str) -> tuple[pd.Series, pd.DataFrame]:
            parts, picks = [], []
            for y in range(2021, 2027):
                beg = pd.Timestamp(f"{y}-01-01")
                nxt = pd.Timestamp(f"{y + 1}-01-01")
                is_beg = (pd.Timestamp("2019-01-01") if mode == "expanding"
                          else pd.Timestamp(f"{y - 3}-01-01"))
                best, best_sr = None, -np.inf
                for (k, b), dr in combos.items():
                    x = dr[(dr.index >= is_beg) & (dr.index < beg)]
                    if len(x) < 400:                 # 至少 1.5 年才认
                        continue
                    s = _sr(x)
                    if s > best_sr:
                        best, best_sr = (k, b), s
                if best is None:
                    continue
                oos = combos[best]
                oos = oos[(oos.index >= beg) & (oos.index < nxt)]
                parts.append(oos)
                picks.append(dict(mode=mode, year=y, k=best[0], buf=best[1],
                                  is_sr=best_sr, oos_ret=float(np.prod(1 + oos) - 1)))
            return pd.concat(parts).sort_index(), pd.DataFrame(picks)

        def _ser_m(dr: pd.Series) -> dict:
            ann = float(np.prod(1 + dr) ** (252 / len(dr)) - 1)
            eq = (1 + dr).cumprod()
            dd = float((eq / eq.cummax() - 1).min())
            return dict(ann=ann, sharpe_arith=_sr(dr), max_dd=-dd,
                        calmar=ann / (-dd) if dd < 0 else np.nan)

        picks_all = []
        cmp_rows = []
        for mode in ("expanding", "rolling3y"):
            ser, pk = _spliced(mode)
            pk.to_csv(OUT / f"reopt_{mode}_picks.csv", index=False)
            picks_all.append(pk)
            cmp_rows.append({"arm": f"reopt_{mode}", **_ser_m(ser)})
            print(f"  {mode:<12} " + "  ".join(
                f"{int(r.year)}:k{int(r.k)}/b{r.buf:g}" for r in pk.itertuples()),
                flush=True)

        # 同一评估窗口（2021 起）下的固定参数基准 + 事后最优（oracle 上界）
        ev0 = pd.Timestamp("2021-01-01")
        fixed = {"fixed_k8_b1.0": (8, 1.0)}
        best_o, best_o_sr = None, -np.inf
        for (k, b), dr in combos.items():
            s = _ser_m(dr[dr.index >= ev0])
            if s["sharpe_arith"] > best_o_sr:
                best_o, best_o_sr = (k, b), s["sharpe_arith"]
        fixed[f"oracle_k{best_o[0]}_b{best_o[1]:g}"] = best_o
        for name, (k, b) in fixed.items():
            s = _ser_m(combos[(k, b)][combos[(k, b)].index >= ev0])
            cmp_rows.append({"arm": name, **s})

        cmp = pd.DataFrame(cmp_rows)
        cmp.to_csv(OUT / "reopt_compare.csv", index=False)
        print("\n同一评估窗口 2021-01-02.. 下对比：", flush=True)
        for r in cmp.itertuples():
            print(f"  {r.arm:<20} ann={r.ann*100:6.2f}% SR={r.sharpe_arith:.3f} "
                  f"DD={r.max_dd*100:5.1f}% Calmar={r.calmar:.2f}", flush=True)
        pd.concat(picks_all).to_csv(OUT / "reopt_picks.csv", index=False)

    print(f"\n写入 {OUT}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=1e5)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.capital, a.quick)
