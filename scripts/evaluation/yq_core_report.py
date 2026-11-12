"""`yqcore` 族（YQ 核心）产品线报数脚本 —— 三档的统一口径对照 + 净值曲线。

为什么单独一个脚本
------------------
`yqcore` 的成立依据是「三条线的持仓宽度对收益的敏感方向不同」这件事本身，报数必须
**同窗口、同执行层、同费率、同末日**，否则一列之差就会把结论读反（2026-11-10 实测：
920 侧 pred 晚 36 个交易日会让它年均虚低 1.4~1.6pp）。所以固定流程而不是临时命令。

三种评测视图（`--view`，报数必须写明是哪一种）
------------------------------------------
- ``native``（默认）= **各自推荐资金 + 整手** ⇒ 回答「按建议下单能拿到多少」。
  这是产品线的**可达成值**：solo 10 万 / focus 50 万 / breadth 200 万，整手 100 股。
- ``uniform`` = 统一 1e6、不启用整手 ⇒ 回答「纯粹的策略差异」，用于跨档比较 alpha
  形状（历史 ``report.json`` 就是这个视图，保留以维持可比）。
- ``stress`` = 统一 1e5 整手 ⇒ 回答「都很小的时候谁先垮」（资金约束的极端面）。

口径（两次报数必须逐字相同）
--------------------------
- 窗口 ``prod`` = 2019-01-02 起，**末日强制截到 2026-09-07**（882 的 pred 面板末日；
  920 重建代理到 11-04，不对齐则 2026 年混进 36 个交易日）
- 成本 ``cost_profile("production")`` = **往返 27bp 时变**（万1佣金双边 + 印花万5单边、
  2023-08-28 前千1 + 滑点 10bp 双边）；成交 ``T+1 open``
- 信号：``yqcore-focus`` / ``yqcore-solo`` → ``gbdt_mid13``（**复合信号**：
  gbdt h1:h5:h10:h20 = 2:1:1:2 秩平均，无落盘实体，由 ``caliber.load_context`` 现场构造）；
  ``yqcore-breadth`` → ``dingban_fundind_local``（登记路径本机缺失，走重建代理；
  这条是**换数据态**，与 v2 登记的 21.57% 不可逐位比）
- 整手：``price_panel = close_raw.ffill().shift(1)``（``shift(1)`` 是无前视的关键），
  `lot_size=100`；实投率由引擎算，不是事后折算

用法::

    python -m scripts.evaluation.yq_core_report                    # 三档 native（产品线口径）
    python -m scripts.evaluation.yq_core_report --view uniform     # 跨档可比（历史口径）
    python -m scripts.evaluation.yq_core_report --view stress      # 统一 10 万极限压力
    python -m scripts.evaluation.yq_core_report --tiers yqcore-solo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import caliber as cal  # noqa: E402
from strategy.registry import TIERS, describe, get_tier  # noqa: E402

END_ALIGN = 20260907
PROFILE = "production"
LOT = 100

#: 每档的**推荐资金**（产品线分界依据见 ``reports/yq_core/README.md`` §2.1）
#: 取该档「整手几乎不 binding 且持仓宽度撑得住」的最小资金。
NATIVE_CAPITAL = {
    "yqcore-solo": 1.0e5,
    "yqcore-focus": 5.0e5,
    "yqcore-breadth": 2.0e6,
}

#: 评测视图 → (统一资金或 None=用 NATIVE_CAPITAL, 是否启用整手)
VIEWS = {
    "native": (None, True),
    "uniform": (1.0e6, False),
    "stress": (1.0e5, True),
}


def price_panel(ctx: cal.CaliberContext) -> pd.DataFrame:
    """无前视成交价面板：执行日 E 用**信号日 T** 的收盘价。"""
    raw = pd.read_parquet(cal.BASE_DIR / "close_raw.parquet")
    return (raw.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
            .ffill().shift(1))


def panel_for(name: str, ctx: cal.CaliberContext) -> pd.DataFrame:
    """注册表声明的外部面板（`industry`）→ 对齐到信号面板的截面。"""
    ind = pd.read_parquet(cal.BASE_DIR / f"{name}.parquet")
    return ind.reindex(index=ctx.pred.index, columns=ctx.pred.columns)


def _drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def run_tier(name: str, cost, capital: float, lot_size: int | None,
             want_curve: bool = True) -> tuple[dict, pd.Series | None]:
    """跑一个档位。返回 (指标记录, 日收益序列或 None)。"""
    spec = get_tier(name)
    ctx = cal.load_context(signal=spec.runnable_signal, window="prod",
                           end=END_ALIGN, execution="open")
    overrides = {}
    if spec.panel is not None:
        overrides[spec.panel] = panel_for(spec.panel, ctx)
    strat = spec.build(**overrides)

    px = price_panel(ctx) if lot_size else None
    bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=float(capital), costs=cost)
    res = cal.run_once(bt, ctx, "open", price_panel=px, lot_size=lot_size)
    m = cal.metrics_row(res, capital)

    # 实投率（整手欠配的真实度量；未启用整手时恒 1）
    wh = res.weights_history
    reb = set(cal.next_day_rebalance(ctx.pred.index))
    whr = wh.loc[[d for d in wh.index if d in reb]]
    invest = whr.sum(axis=1)
    nhold = (whr > 1e-9).sum(axis=1)

    rec = {
        "tier": name,
        "label": spec.label,
        "signal_key": spec.signal_key,
        "signal_used": spec.runnable_signal,
        "signal_fingerprint": ctx.fingerprint,
        "strategy": spec.strategy,
        "params": spec.params,
        "window": f"{ctx.pred.index.min().date()}..{ctx.pred.index.max().date()}",
        "n_days": ctx.n_days,
        "capital": float(capital),
        "lot_size": lot_size,
        "ann": m["ann"],
        "sharpe": m["sharpe"],
        "sharpe_arith": m["sharpe_arith"],
        "sharpe_t_stat": m["sharpe_t_stat"],
        "max_dd": m["max_dd"],
        "vol": m["vol"],
        "calmar": m["ann"] / m["max_dd"] if m["max_dd"] > 0 else float("nan"),
        "turnover": m["turnover"],
        "cost_bp": m["cost_bp"],
        "n_active": m["n_active"],
        "invest_mean": float(invest.mean()),
        "invest_min": float(invest.min()),
        "nhold_mean": float(nhold.mean()),
        "nhold_min": int(nhold.min()),
    }
    dr = res.daily_returns.copy() if want_curve else None
    if dr is not None and len(dr):
        # ``run_once`` 不附带 ``yearly_ret_*``（那是 ``evaluate_strategy`` 补的），
        # 直接从日收益算，口径与 ``_metrics`` 的复利一致。
        rec["yearly"] = {int(y): round(float((1 + sub).prod() - 1), 6)
                         for y, sub in dr.groupby(dr.index.year)}
    else:
        rec["yearly"] = {}
    return rec, dr


def run_grid(names: list[str], cost, capitals: list[float]) -> pd.DataFrame:
    """资金 × 档位 交叉网格（**一律开整手**）—— 产品线分界的唯一硬证据。

    分界不能用「感觉够买」定：整手约束是**非线性**的（买不起一手的零头留现金），
    所以每档在哪个资金上从「能建满仓」掉到「结构性欠配」必须实测。
    """
    rows = []
    ctxs: dict[str, cal.CaliberContext] = {}
    px_cache: dict[int, pd.DataFrame] = {}
    for cap in capitals:
        for n in names:
            spec = get_tier(n)
            key = spec.runnable_signal
            if key not in ctxs:
                ctxs[key] = cal.load_context(signal=key, window="prod",
                                             end=END_ALIGN, execution="open")
            ctx = ctxs[key]
            if id(ctx) not in px_cache:
                px_cache[id(ctx)] = price_panel(ctx)
            overrides = {}
            if spec.panel is not None:
                overrides[spec.panel] = panel_for(spec.panel, ctx)
            strat = spec.build(**overrides)
            bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                    initial_capital=float(cap), costs=cost)
            res = cal.run_once(bt, ctx, "open",
                               price_panel=px_cache[id(ctx)], lot_size=LOT)
            m = cal.metrics_row(res, cap)
            wh = res.weights_history
            reb = set(cal.next_day_rebalance(ctx.pred.index))
            whr = wh.loc[[d for d in wh.index if d in reb]]
            inv = whr.sum(axis=1)
            nh = (whr > 1e-9).sum(axis=1)
            row = dict(tier=n, capital=cap, ann=m["ann"], sharpe=m["sharpe"],
                       sharpe_arith=m["sharpe_arith"], max_dd=m["max_dd"],
                       calmar=m["ann"] / m["max_dd"] if m["max_dd"] > 0 else float("nan"),
                       turnover=m["turnover"],
                       invest_mean=float(inv.mean()), invest_min=float(inv.min()),
                       nhold_mean=float(nh.mean()), nhold_min=int(nh.min()))
            rows.append(row)
            print(f"  {n:16s} 资金 {cap:>10,.0f}  年化 {row['ann']*100:6.2f}%  "
                  f"SR {row['sharpe']:.3f}  回撤 {row['max_dd']*100:5.1f}%  "
                  f"实投 {row['invest_mean']*100:5.1f}%  持仓 {row['nhold_mean']:5.1f}",
                  flush=True)
    return pd.DataFrame(rows)


def run_capacity(names: list[str], cost, adv_participation: float = 0.10) -> pd.DataFrame:
    """每档的**容量上限**（元）—— 产品线分界的主依据。

    收益上 ``solo`` 各资金档都最高，所以分界不能按收益定；真正卡住的是
    「单票占组合权重 × 组合规模 ≤ 单票 ADV × 参与率」。单票权重由档位决定
    （solo ≈16% / focus ≈6.6% / breadth ≈0.14%），ADV 由信号选到的股票决定，
    两者相乘才有容量 —— 只报 ADV 会得出「三档容量差不多」的错误结论。
    """
    rows = []
    for name in names:
        spec = get_tier(name)
        cap0 = NATIVE_CAPITAL[name]
        ctx = cal.load_context(signal=spec.runnable_signal, window="prod",
                               end=END_ALIGN, execution="open")
        overrides = {}
        if spec.panel is not None:
            overrides[spec.panel] = panel_for(spec.panel, ctx)
        strat = spec.build(**overrides)
        bt = cal.VectorBacktest(strategy=strat, rebalance_freq="M",
                                initial_capital=float(cap0), costs=cost)
        px = price_panel(ctx)
        res = cal.run_once(bt, ctx, "open", price_panel=px, lot_size=LOT)
        wh = res.weights_history
        reb = set(cal.next_day_rebalance(ctx.pred.index))
        whr = wh.loc[[d for d in wh.index if d in reb]]
        amt = ctx.amt.reindex(index=ctx.pred.index, columns=ctx.pred.columns)
        advs, wts = [], []
        for d, row in whr.iterrows():
            if d not in amt.index:
                continue
            codes = row.index[row.to_numpy() > 1e-9]
            a = amt.loc[d].reindex(codes).dropna()
            advs.extend(a.to_numpy())
            wts.extend(row.reindex(a.index).to_numpy())
        adv = pd.Series(advs, dtype=float).dropna()
        wt = pd.Series(wts, dtype=float)
        cap_est = adv_participation * adv.to_numpy() / wt.to_numpy()
        rows.append(dict(
            tier=name, n_hold=float(len(wt) / max(len(whr), 1)),
            weight_median=float(wt.median()),
            adv_median=float(adv.median()), adv_p10=float(adv.quantile(0.10)),
            capacity_median=float(np.median(cap_est)),
            capacity_p10=float(np.quantile(cap_est, 0.10)),
            participation_assumed=adv_participation,
        ))
        r = rows[-1]
        print(f"  {name:16s} 持仓 {r['n_hold']:5.1f} 只  单票权重中位 "
              f"{r['weight_median']*100:5.2f}%  ADV 中位 {r['adv_median']/1e8:6.2f} 亿"
              f"(p10 {r['adv_p10']/1e8:5.2f})  容量中位 {r['capacity_median']/1e8:7.2f} 亿"
              f"(p10 {r['capacity_p10']/1e8:6.2f})", flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiers", default="", help="逗号分隔的档位名；空=全部")
    ap.add_argument("--view", default="native",
                    choices=sorted(VIEWS) + ["grid", "capacity"],
                    help="native=各自推荐资金+整手（产品线口径）"
                         " | uniform=统一1e6无整手 | stress=统一10万整手"
                         " | grid=资金×档位网格 | capacity=各档容量上限")
    ap.add_argument("--out", default="reports/yq_core")
    ap.add_argument("--tag", default="", help="产物后缀（避免覆盖主报告）")
    args = ap.parse_args(argv)

    names = ([x.strip() for x in args.tiers.split(",") if x.strip()]
             or list(TIERS))
    cost = cal.cost_profile(PROFILE)
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    # 默认后缀按视图分文件（2026-11-12 修正）：此前 bare run 写 `report.json`，
    # 但 README 约定 `report_native.json` 是产品线口径 ⇒ 跑一次 native 就把
    # uniform 的 `report.json` 覆盖掉，两份产物互相踩。现在 native → `_native`，
    # uniform → 无后缀（`report.json` = 历史 uniform 口径），其余视图同理。
    # ⚠️ grid / capacity 的文件名**已自带视图名**（`grid_lot` / `capacity`），
    #    再加后缀会变成 `grid_lot_grid.csv` ⇒ 这两个视图默认不加后缀。
    suffix = f"_{args.tag}" if args.tag else {
        "native": "_native", "uniform": "", "stress": "_stress"}.get(args.view, "")

    if args.view == "capacity":
        print(f"各档容量上限（单票 ≤ ADV 的 10%，整手 {LOT} 股）\n")
        df = run_capacity(names, cost)
        p = outdir / f"capacity{suffix}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n产物: {p}")
        return 0

    if args.view == "grid":
        caps = [5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6]
        print(f"资金 × 档位网格（整手 {LOT} 股）：{', '.join(f'{c:,.0f}' for c in caps)}\n")
        df = run_grid(names, cost, caps)
        p = outdir / f"grid_lot{suffix}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n产物: {p}")
        pivot = df.pivot(index="capital", columns="tier", values="ann") * 100
        print("\n年化（%）")
        print(pivot.map(lambda x: f"{x:7.2f}").to_string())
        piv2 = df.pivot(index="capital", columns="tier", values="invest_mean") * 100
        print("\n整手实投率（%）")
        print(piv2.map(lambda x: f"{x:7.2f}").to_string())
        return 0

    uni_cap, use_lot = VIEWS[args.view]
    recs, curves = [], {}
    for n in names:
        cap = float(uni_cap) if uni_cap is not None else NATIVE_CAPITAL[n]
        lot = LOT if use_lot else None
        print(describe(n))
        r, dr = run_tier(n, cost, cap, lot)
        recs.append(r)
        if dr is not None:
            curves[n] = (1 + dr).cumprod()
        print(f"  {r['window']}  {r['n_days']} 日  指纹 {r['signal_fingerprint']}")
        print(f"  资金 {cap:,.0f}  整手 {'100' if lot else '关'}  "
              f"年化 {r['ann']*100:.2f}%  Sharpe {r['sharpe']:.3f} "
              f"(算术 {r['sharpe_arith']:.3f}, t {r['sharpe_t_stat']:.2f})  "
              f"回撤 {r['max_dd']*100:.1f}%  Calmar {r['calmar']:.2f}")
        print(f"  换手 {r['turnover']*100:.1f}%  成本 {r['cost_bp']:.0f}bp  "
              f"持仓 {r['nhold_mean']:.1f}(min {r['nhold_min']})  "
              f"实投 {r['invest_mean']*100:.1f}%(min {r['invest_min']*100:.1f}%)\n")

    payload = {
        "family": "yqcore",
        "profile": PROFILE,
        "view": args.view,
        "execution": "open",
        "lot_size": LOT if use_lot else None,
        "price_panel": "close_raw.ffill().shift(1)" if use_lot else None,
        "end_align": END_ALIGN,
        "generated_by": "scripts/evaluation/yq_core_report.py",
        "tiers": {r["tier"]: r for r in recs},
    }
    p = outdir / f"report{suffix}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {p}")

    # 逐年对照表（人读）
    yrs = [str(y) for y in range(2019, 2027)]
    table = pd.DataFrame(
        {r["tier"]: [r["yearly"].get(int(y), float("nan")) * 100 for y in yrs]
         for r in recs}, index=yrs)
    table.loc["年化"] = [r["ann"] * 100 for r in recs]
    table.loc["Sharpe"] = [r["sharpe"] for r in recs]
    table.loc["最大回撤"] = [-r["max_dd"] * 100 for r in recs]
    table.index.name = f"年份({args.view})"
    print("\n逐年收益（%，末三行为汇总）")
    print(table.map(lambda x: f"{x:7.2f}").to_string())
    table.to_csv(outdir / f"yearly{suffix}.csv", encoding="utf-8-sig")

    if curves:
        nav = pd.DataFrame(curves).sort_index()
        nav.index.name = "date"
        nav.to_csv(outdir / f"nav{suffix}.csv", encoding="utf-8-sig")
        dd = nav.apply(_drawdown)
        dd.to_csv(outdir / f"drawdown{suffix}.csv", encoding="utf-8-sig")
        print(f"净值/回撤: {outdir}/nav{suffix}.csv, drawdown{suffix}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
