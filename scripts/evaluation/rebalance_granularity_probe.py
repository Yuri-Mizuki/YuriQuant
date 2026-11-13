"""调仓粒度探针：solo 档的信息半衰期 + 换手预算（P0 / P1）。

问题来源（2026-11-13）
----------------------
``yqcore-solo`` 的执行层是 ``exit_mult=1.0`` 的硬切 TopK8（**无缓冲带**）：
月单边换手 **92.4%**、8 年累计成本 9749bp（≈12%/年）。它是"每月按预测全部重选"
的字面形态，而定版时**从未测过**这个换手里有多少是信息驱动的。

P0（本文件的 ``ic_decay``）只回答一件事：**信号的信息含量随持有期衰减多快**。

- 半衰期 >> 21 个交易日 ⇒ 月频太频繁，换手里有大量噪声交易。
- 半衰期 << 21 ⇒ 月频已偏长；但提频已被 ``reports/solo_weekly/`` 判负
  （毛口径月频 41.26% 仍高于周频 36.39%），说明瓶颈在成本而非节奏。

P1（本文件的 ``budget_grid``）测**换手预算**：每期最多新增 m 只（行业约束后），
剩余名额从上一期持仓里按当期排名补回。与缓冲带的区别 —— 带宽按**排名**截断，
预算按**只数**截断，后者直接对应"这期最多付多少成本"。

口径
----
signal ``gbdt_mid13``（solo 档指纹 79bdbbcc8fb9）、窗口 ``prod``（2019 起）、
T+1 open、1e5 + 整手 100、往返 27bp 时变。与 ``yqcore-solo`` 逐项对齐。

IC 衰减用 T 日信号对 T→T+h 的**收盘**收益 —— 纯信息含量，不含执行价差与成本。
h 越长样本重叠越重，**不给 t 值**，只看形状与半衰期。

用法::

    python -m scripts.evaluation.rebalance_granularity_probe --only ic
    python -m scripts.evaluation.rebalance_granularity_probe --only anchor
    python -m scripts.evaluation.rebalance_granularity_probe --only budget
    python -m scripts.evaluation.rebalance_granularity_probe            # 全跑
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.evaluation.caliber as cal
from strategy.cross_section import SectorCappedBufferedTopKLongOnly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "rebalance_granularity"
SIGNAL = "gbdt_mid13"
END = 20260907
CAPITAL = 1.0e5
LOT = 100
SOLO_K = 8

HORIZONS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 90)


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    """未复权收盘价，ffill 后 shift(1)（引擎在执行日 E 取价 ⇒ 拿到 T 日价格）。"""
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns).ffill().shift(1)


def industry_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    ind = pd.read_parquet(cal.BASE_DIR / "industry.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 100:
        return np.nan
    return float(a.corr(b, method="spearman"))


def ic_decay(ctx: cal.CaliberContext, topk: int = SOLO_K) -> pd.DataFrame:
    """月频采样下，信号对 T→T+h 收益的 rank IC 与 TopK 超额。

    两列口径不同，别混：
    - ``ic``：全截面秩相关（信号整体排序能力）
    - ``topk_excess``：TopK 等权 h 日收益 − 全截面等权 h 日收益（**经济量**）

    ``retain`` = ic(h) / ic(1)，用于读半衰期。
    """
    close = ctx.close
    sig = ctx.pred
    rb = cal.rebalance_dates(ctx.pred.index, freq="M")
    rows = []
    for h in HORIZONS:
        fwd = close.shift(-h) / close - 1.0
        ics, exc, ns = [], [], []
        for t in rb:
            if t not in fwd.index:
                continue
            s = sig.loc[t].dropna()
            f = fwd.loc[t].reindex(s.index)
            ok = f.notna()
            s, f = s[ok], f[ok]
            if len(s) < 500:
                continue
            ics.append(_spearman(s, f))
            k = min(topk, len(s))
            top = s.nlargest(k).index
            exc.append(float(f.loc[top].mean() - f.mean()))
            ns.append(len(s))
        ex_mean = float(np.nanmean(exc))
        rows.append(dict(h=h, ic=np.nanmean(ics), ic_std=np.nanstd(ics),
                         n_period=len(ics), topk_excess=ex_mean,
                         topk_excess_per_day=ex_mean / h,
                         n_cs=float(np.mean(ns)) if ns else np.nan))
    df = pd.DataFrame(rows)
    base = float(df.loc[df["h"] == 1, "ic"].iloc[0])
    df["retain"] = df["ic"] / base
    return df


def half_life(df: pd.DataFrame) -> float:
    """IC 降到 h=1 一半所需交易日（线性插值；未跌破则返回 nan）。"""
    h = df["h"].to_numpy(dtype=float)
    r = df["retain"].to_numpy(dtype=float)
    for i in range(1, len(r)):
        if r[i] <= 0.5:
            if not np.isfinite(r[i - 1]) or r[i - 1] == r[i]:
                return float(h[i])
            w = (r[i - 1] - 0.5) / (r[i - 1] - r[i])
            return float(h[i - 1] + w * (h[i] - h[i - 1]))
    return float("nan")


class BudgetedSectorCappedBufferedTopKLongOnly(SectorCappedBufferedTopKLongOnly):
    """每期**最多新增 m 只**的行业约束缓冲带（P1）。

    与 ``exit_mult`` 的区别：带宽按**排名**截断（跌出 ``k·exit_mult`` 才卖），
    预算按**只数**截断（这期最多买 m 只新的）。两者叠加时预算优先 ——
    应换未换的名额从上一期持仓里按当期排名补回（即使已跌出保留带）。

    ``max_swaps=None`` 时与父类逐位相同（默认不改行为）。

    ⚠️ 硬留会让持仓"老化"：被留下的票当期排名可能很差。``age_pct`` =
    各期被硬留标的的当期排名分位（0 = 最好，1 = 最差），出报告时必须一并看。
    """

    def __init__(self, *args, max_swaps: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_swaps is not None and max_swaps < 0:
            raise ValueError(f"max_swaps 不能为负: {max_swaps}")
        self.max_swaps = max_swaps
        self.age_pct: list[float] = []
        self.n_swap: list[int] = []
        self.name = f"{self.name}_sw{max_swaps}"

    def get_weights_at(self, date, factor_values: pd.Series) -> pd.Series:
        if self.max_swaps is None:
            return super().get_weights_at(date, factor_values)
        w = self._get_weights_budgeted(date, factor_values)
        return w

    def _get_weights_budgeted(self, date, factor_values: pd.Series) -> pd.Series:
        vals = factor_values.dropna()
        if vals.empty:
            self._prev = set()
            return pd.Series(dtype=float)
        sect = (self.sectors.loc[date] if date in self.sectors.index
                else pd.Series(index=vals.index, dtype=object)).reindex(vals.index)

        def _key(code):
            s = sect.get(code)
            return "__NA__" if s is None or pd.isna(s) else str(s)

        from strategy.examples import rank_desc

        ranks = rank_desc(vals)
        n_exit = max(self.k, int(round(self.k * self.exit_mult)))
        total = len(ranks)

        counts: dict[str, int] = {}
        keep: list = []
        for code in sorted(self._prev, key=lambda c: ranks.get(c, float("inf"))):
            if code not in ranks.index or ranks[code] > n_exit:
                continue
            key = _key(code)
            if counts.get(key, 0) >= self.max_per_sector:
                continue
            counts[key] = counts.get(key, 0) + 1
            keep.append(code)
            if len(keep) >= self.k:
                break

        limit = n_exit if self.fill_to_k else self.k
        fills: list = []
        for code in ranks.sort_values().index:
            if len(keep) + len(fills) >= self.k:
                break
            if ranks[code] > limit:
                break
            if code in self._prev and ranks[code] <= n_exit:
                continue
            key = _key(code)
            if counts.get(key, 0) >= self.max_per_sector:
                continue
            counts[key] = counts.get(key, 0) + 1
            fills.append(code)

        n_swap = len(fills)
        self.n_swap.append(n_swap)
        if n_swap <= self.max_swaps:
            port = keep + fills
        else:
            fills = fills[:self.max_swaps]
            port = keep + fills
            # 剩余名额：从**上一期持仓中已跌出保留带**的票里按当期排名补回
            held_back = [c for c in sorted(self._prev,
                                           key=lambda c: ranks.get(c, float("inf")))
                         if c not in port and c in ranks.index]
            for code in held_back:
                if len(port) >= self.k:
                    break
                key = _key(code)
                if counts.get(key, 0) >= self.max_per_sector:
                    continue
                counts[key] = counts.get(key, 0) + 1
                port.append(code)
                pct = (float(ranks[code]) - 1.0) / max(total - 1, 1)
                self.age_pct.append(pct)

        if not port:
            self._prev = set()
            return pd.Series(dtype=float)
        self._prev = set(port)
        return pd.Series(1.0 / len(port), index=port)


def _run(ctx, strat, px, ind, freq="M", phase=0):
    bt = cal.VectorBacktest(strategy=strat, rebalance_freq=freq,
                            initial_capital=float(CAPITAL),
                            costs=cal.cost_profile("production"))
    res = cal.run_once(bt, ctx, "open", price_panel=px, lot_size=LOT,
                       freq=freq, phase=phase)
    m = cal.metrics_row(res, CAPITAL)
    dr = res.daily_returns
    yr = (1 + dr).groupby(dr.index.year).prod() - 1
    wh = res.weights_history
    return dict(ann=m["ann"], sharpe=m["sharpe"], sharpe_arith=m["sharpe_arith"],
                sharpe_t=m["sharpe_t_stat"], vol=m["vol"], max_dd=m["max_dd"],
                turnover=m["turnover"], cost_bp=m["cost_bp"], n_active=m["n_active"],
                year_worst=float(yr.min()), years_pos=int((yr > 0).sum()),
                n_year=int(len(yr)), nhold_mean=float((wh > 1e-9).sum(axis=1).mean()))


def anchor(ctx, px, ind) -> dict:
    strat = SectorCappedBufferedTopKLongOnly(ind, k=SOLO_K, exit_mult=1.0,
                                             max_per_sector=1)
    return _run(ctx, strat, px, ind)


def budget_grid(ctx, px, ind) -> pd.DataFrame:
    rows = []
    for m in (None, 1, 2, 3, 4):
        strat = BudgetedSectorCappedBufferedTopKLongOnly(
            ind, k=SOLO_K, exit_mult=1.0, max_per_sector=1, max_swaps=m)
        r = _run(ctx, strat, px, ind)
        r.update(arm=f"sw{m if m is not None else 'none'}", max_swaps=m,
                 swap_mean=float(np.mean(strat.n_swap)) if strat.n_swap else np.nan,
                 age_mean=float(np.mean(strat.age_pct)) if strat.age_pct else 0.0,
                 age_p95=float(np.percentile(strat.age_pct, 95)) if strat.age_pct else 0.0)
        rows.append(r)
        print(f"  sw={m}: ann={r['ann']:.4f} SR={r['sharpe']:.4f} "
              f"turn={r['turnover']:.3f} swaps={r['swap_mean']:.2f} "
              f"age={r['age_mean']:.3f}", flush=True)
    return pd.DataFrame(rows)


def cadence_grid(ctx, px, ind) -> pd.DataFrame:
    """P0 落地面：调仓间隔 × 缓冲带 的**净收益**（同 K8 / 同行业约束）。

    ``turnover`` 是**每期**换手，跨频率不可直接比 ⇒ 另给 ``turnover_ann``
    （= 每期 × 每年期数）。成本 ``cost_bp`` 已折成占资金比例，可跨频率比。
    """
    per_year = {"D": 252, "W": 52, "2W": 26, "M": 12, "2M": 6, "Q": 4}
    # Q 只有 3 个不同相位（错开 3 个月即回到同一分组），不是 4 个。
    # M 的相位 = 月内第 n 个交易日（0/5/10/15）；不足时 rebalance_dates 回退月末。
    phases = {"M": (0, 5, 10, 15), "2M": (0, 1), "Q": (0, 1, 2)}
    rows = []
    for freq in ("M", "2M", "Q"):
        for ph in phases[freq]:
            for em in (1.0, 1.5):
                strat = SectorCappedBufferedTopKLongOnly(
                    ind, k=SOLO_K, exit_mult=em, max_per_sector=1)
                r = _run(ctx, strat, px, ind, freq=freq, phase=ph)
                r.update(arm=f"{freq}_p{ph}_b{em}", freq=freq, phase=ph,
                         exit_mult=em,
                         turnover_ann=r["turnover"] * per_year[freq])
                rows.append(r)
                print(f"  {freq:>2} p{ph} b{em}: ann={r['ann']:.4f} "
                      f"SR={r['sharpe']:.4f} turn_ann={r['turnover_ann']:.2f} "
                      f"cost={r['cost_bp']:.0f}bp", flush=True)
    return pd.DataFrame(rows)


def swap_diagnostics(ctx, topk: int = SOLO_K) -> pd.DataFrame:
    """每次换手的**边际收益**：换入票 vs 换出票在下一持有期的收益差。

    这是"换手是信息驱动还是噪声"的直接判据 —— 与"少换就亏"互补：
    后者只说明预算 binding，前者给出**每换一只值多少 bp**。

    口径：纯排名 TopK（**不含行业约束**，故与 solo 的 5.9 只不同，只读方向与量级）；
    close→close 到下一调仓日，不扣成本、不含 T+1 开盘价差。
    """
    sig, close = ctx.pred, ctx.close
    rb = cal.rebalance_dates(ctx.pred.index, freq="M")
    rows, prev = [], None
    for i, t in enumerate(rb[:-1]):
        nxt = rb[i + 1]
        s = sig.loc[t].dropna()
        if len(s) < topk:
            continue
        top = set(s.nlargest(topk).index)
        if prev is not None:
            r = close.loc[nxt] / close.loc[t] - 1.0
            out = prev - top
            into = top - prev
            rows.append(dict(
                date=t, n_out=len(out), n_in=len(into),
                out_ret=float(r.reindex(sorted(out)).mean()) if out else np.nan,
                in_ret=float(r.reindex(sorted(into)).mean()) if into else np.nan,
            ))
        prev = top
    df = pd.DataFrame(rows)
    df["edge"] = df["in_ret"] - df["out_ret"]
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only",
                    choices=("ic", "anchor", "budget", "cadence", "swap"),
                    default=None)
    ap.add_argument("--end", type=int, default=END)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context(SIGNAL, window="prod", end=a.end, execution="open")
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} "
          f"({ctx.n_days} 日) | 指纹 {ctx.fingerprint}", flush=True)

    if a.only in (None, "ic"):
        df = ic_decay(ctx)
        df.to_csv(OUT / "ic_decay.csv", index=False, encoding="utf-8-sig")
        hl = half_life(df)
        print("\n=== P0 IC 衰减（月频采样，T→T+h 收盘收益）===")
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"半衰期 = {hl:.1f} 个交易日（月频周期 ≈ 21）")

    if a.only in (None, "swap"):
        df = swap_diagnostics(ctx)
        df.to_csv(OUT / "swap_diagnostics.csv", index=False, encoding="utf-8-sig")
        print("\n=== 换手归因（换入 vs 换出，下一持有期 close→close）===")
        print(f"  期数 {len(df)} | 每期换入 {df['n_in'].mean():.2f} / "
              f"换出 {df['n_out'].mean():.2f}（纯排名 K={SOLO_K}）")
        print(f"  换入 {df['in_ret'].mean():.4%} vs 换出 {df['out_ret'].mean():.4%} "
              f"⇒ edge {df['edge'].mean():.4%}/期（中位 {df['edge'].median():.4%}，"
              f"{int((df['edge'] > 0).sum())}/{len(df)} 期为正）")

    if a.only in (None, "anchor", "budget", "cadence"):
        px, ind = price_panel(ctx), industry_panel(ctx)
        if a.only in (None, "anchor"):
            r = anchor(ctx, px, ind)
            pd.DataFrame([r]).to_csv(OUT / "anchor.csv", index=False,
                                     encoding="utf-8-sig")
            print("\n=== 锚点（solo K8 b1.0，应与登记 36.58% 逐位一致）===")
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in r.items()})
        if a.only in (None, "cadence"):
            print("\n=== P0 调仓间隔 × 缓冲带（净收益）===")
            df = cadence_grid(ctx, px, ind)
            df.to_csv(OUT / "cadence_grid.csv", index=False, encoding="utf-8-sig")
        if a.only in (None, "budget"):
            print("\n=== P1 换手预算（每期最多新增 m 只）===")
            df = budget_grid(ctx, px, ind)
            df.to_csv(OUT / "budget_grid.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
