"""成交日相位：能不能把成交提前到「当月第 1 个交易日开盘」。

回答 2026-11-10 的问题：「现在是信号日 = 当月第 1 个交易日、成交日 = 第 2 个交易日开盘，
那有没有试过成交日 = 第 1 个交易日开盘？」

为什么不能直接测「成交日改成月首开盘」
--------------------------------------
这句话在代码上有**三种**解读，其中一种是前视陷阱：

  A. 现状（基线）   信号 T=月首收盘   → 成交 E=月第2日开盘   （差 1 天，无前视）
  B. 合法前移       信号 T=上月末收盘 → 成交 E=月首开盘      （差 1 天，无前视）
                     —— 「成交在月首开盘」唯一合法的实现方式。
  C. 前视陷阱       信号 T=月首收盘   → 成交 E=月首开盘
                     —— 用当天收盘才知道的信息，去当天已发生的开盘价成交。
                        实盘无法复制，只能用来**证伪**，不能当备选方案。

只测 B 不测 C，「提前到底虚高多少」无从计量；只测 A/B 又分不清差异来自信号还是成交时点。
故补 D 臂凑成 2×2：

                    成交 = 月首开盘      成交 = 月第2日开盘
    信号 = 上月末      B  (合法)             D  (延迟臂)
    信号 = 月首收盘    C  (前视·不可用)       A  (现状)

  纵向 D−B = 成交日延后一天（固定旧信号）→ 成交时点本身的价值
  横向 A−D = 信号新鲜一天（固定成交日）  → 信号新鲜度的价值
  A−B = 两者之和

引擎语义（踩过的坑）
--------------------
``bt.run(sig_used, ..., rebalance_days=R, execution_split=split)`` 在日 i：
1a) 旧权重结算 ``on[i]`` → 1b) 若 ``dates[i] ∈ R`` 用 ``fp.iloc[i]`` 换仓 →
1b+) 新权重赚 ``id[i]``。
``split`` 由**信号日集合 S** 生成，写入 ``S+1``；``sig_used = panel.shift(1)`` ⇒
执行日 E 拿到 ``panel[E-1]``。**必须让 S+1 == R**，否则前视/前移会被静默抵消
（首版把 S 和 R 都传月首，前视只剩 +0.40pp，几乎看不出）。

故四臂统一令 ``R = 成交日``、``S = 成交日的前一交易日``；C 臂额外把 panel 换成
``pred.shift(-1)``，抵消引擎的 shift(1) ⇒ 执行日拿到**当日**信号。

输出 ``reports/exec_phase_month_open/``
---------------------------------------
  arms.csv            四臂主表（年化/SR/回撤/换手）
  arms_yearly.csv     四臂逐年收益
  decomp.csv          2×2 分解（D−B / A−D / A−B / C−B）

用法::

    python -m scripts.evaluation.exec_phase_probe
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.evaluation.caliber as cal
from scripts.evaluation.solo_freq_probe import (
    CAPITAL,
    END,
    LOT,
    SOLO_BUF,
    SOLO_K,
    industry_panel,
    price_panel,
    stats,
)
from strategy.cross_section import SectorCappedBufferedTopKLongOnly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "exec_phase_month_open"


def _month_anchors(idx: pd.DatetimeIndex):
    """返回（月首, 上月末, 月第2日），三列**逐期对齐**。

    首月没有「上月末」⇒ 剔除，保证四臂期数一致、可比。
    """
    pos = {d: i for i, d in enumerate(idx)}
    s = pd.Series(idx, index=idx)
    first = pd.DatetimeIndex(s.groupby(s.index.to_period("M")).first().values)
    first = first[1:]                       # 去掉首月（无上月末）
    prev_last = pd.DatetimeIndex([idx[pos[t] - 1] for t in first])
    second = pd.DatetimeIndex([idx[pos[t] + 1] for t in first
                               if pos[t] + 1 < len(idx)])
    n = len(second)
    return first[:n], prev_last[:n], second


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = cal.load_context("gbdt_mid13", window="prod", end=END, execution="open")
    cost = cal.cost_profile("production")
    px = price_panel(ctx)
    ind = industry_panel(ctx)
    first, prev_last, second = _month_anchors(ctx.pred.index)
    print(f"窗口 {ctx.pred.index[0].date()}..{ctx.pred.index[-1].date()} "
          f"| 资金 {CAPITAL:,.0f} | 整手 {LOT} | k={SOLO_K} b={SOLO_BUF} "
          f"| 期数 {len(first)}", flush=True)

    arms = [
        # tag, 面板, 信号日集合 S, 成交日集合 R
        ("A_现状_月首信号_次2日开", ctx.pred, first, second),
        ("D_上月末信号_次2日开", ctx.pred, prev_last, second),
        ("B_合法_上月末信号_月首开", ctx.pred, prev_last, first),
        ("C_前视_月首信号_月首开", ctx.pred.shift(-1), prev_last, first),
    ]

    rows, yearly = [], []
    for tag, panel, sig_days, exec_days in arms:
        sig_used = panel.shift(1).where(ctx.mask)
        split = cal.build_execution_split(ctx.fill, ctx.close, set(sig_days))
        strat = SectorCappedBufferedTopKLongOnly(
            ind, k=SOLO_K, exit_mult=SOLO_BUF, max_per_sector=1)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                initial_capital=CAPITAL, costs=cost)
        res = bt.run(sig_used, ctx.fwd, horizon=1,
                     rebalance_days=set(exec_days), execution_split=split,
                     adv_panel=ctx.amt, vol_panel=ctx.vol60,
                     price_panel=px, lot_size=LOT)
        r = stats(res, len(exec_days))
        rows.append({"arm": tag, "ann": r["ann"],
                     "sharpe_arith": r["sharpe_arith"], "max_dd": r["max_dd"],
                     "turnover_ann": r["turnover_ann"], "n_reb": len(exec_days)})
        dr = res.daily_returns
        for y, v in (1 + dr).groupby(dr.index.year).prod().items():
            yearly.append({"arm": tag, "year": int(y), "ret": float(v - 1)})
        print(f"  {tag:<28} ann={r['ann'] * 100:7.2f}% "
              f"SR={r['sharpe_arith']:6.3f} DD={r['max_dd'] * 100:5.1f}% "
              f"换手/年={r['turnover_ann']:6.2f}", flush=True)

    pm = pd.DataFrame(rows)
    pm.to_csv(OUT / "arms.csv", index=False, encoding="utf-8-sig")
    ym = pd.DataFrame(yearly).pivot(index="year", columns="arm", values="ret")
    ym.to_csv(OUT / "arms_yearly.csv", encoding="utf-8-sig")

    a = pm.set_index("arm").ann
    kA, kD, kB, kC = (pm.set_index("arm").index.tolist()[i] for i in range(4))
    dec = pd.DataFrame([
        {"item": "A−B 合计（提前到月首开的总代价）", "pp": (a[kA] - a[kB]) * 100},
        {"item": "D−B 成交日延后一天（成交时点价值）", "pp": (a[kD] - a[kB]) * 100},
        {"item": "A−D 信号从月末变月首（信号新鲜度）", "pp": (a[kA] - a[kD]) * 100},
        {"item": "C−B 前视（不可实现，仅证伪用）", "pp": (a[kC] - a[kB]) * 100},
    ])
    dec.to_csv(OUT / "decomp.csv", index=False, encoding="utf-8-sig")
    print()
    for _, x in dec.iterrows():
        print(f"  {x['item']:<38} {x['pp']:+7.2f}pp", flush=True)
    print(f"\n产物 → {OUT}", flush=True)


if __name__ == "__main__":
    main()