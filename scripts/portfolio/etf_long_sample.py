"""指数代际长样本（2013–2019）：HRP「集中放大器」的区制存活检验（README §4.6）。

背景（TODO §一 ②，先验最高）：2020–2026 的 30.63% 全部来自 ETF 大发展之后的样本，
从未经历 2015 式主题崩盘与 2018 熊市。要判断「押最不相关的那只」是真机制还是
2018 后这批产品的特性，只能在更长的样本上重跑同一个机制。

**数据来源的关键发现（2026-10-30）**：不需要指数代理！
`DataCache.get_index_daily` 能直接拉 **ETF 自身代码**的 2013 年起日线（SDK 不限品种），
比跟踪指数更干净（无跟踪偏离）。45 只池里有 **24 只**在 2013–2019 有数据，
且**避险腿齐全**：黄金 518880(2013-07)、国债 511010(2013-03)、纳指 513100(2013-05)、
标普 513500(2014-01)、恒生 159920(2013-01)。
⇒ HRP 依赖的「低相关腿」在长样本里存在，机制可检验。

口径与生产**逐项相同**：后复权价（×backward_factor，已与生产 `load_close` 逐位核对，
1815 个交易日最大相对差 0.00e+00）、T+1 open 成交、同成本（`FINAL`）、同 mask
（60 日均成交额 ≥ 3000 万）、**同信号 `mom12_1_crowdsoft_pre10`**。

🚨 2026-10-30 自查修正：前一版误判「无份额数据」，把 soft 惩罚置 0、退化成
`mom12_1_pre10`。实测 `fund_share.parquet` 覆盖 **2004-12 起**，2013–2019 有
34 只成分（510050 自 2004、510300 自 2012-05、518880 自 2013-07），因此
`build_extra_panels` 在长样本上可直接用真份额算 crowd。修正后长样本信号与生产同源，
「换池」比较才不再混入信号变更。

⚠️ 不用 `w0/w1` 切片跑（会让 `ret_panel` 只有窗口内长度 ⇒ HRP 协方差与去重冷启动，
见 `run_variant` docstring 的窗口口径陷阱）。做法 = **单次连续回测 + 事后按日期截断
equity 重算指标**。

复现：`python -m scripts.portfolio.etf_long_sample --mode core`
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from backtest.engine import build_execution_split
from scripts.portfolio.etf_contest_spec import FINAL
from scripts.portfolio.etf_rotation_panels import build_extra_panels
from scripts.portfolio.etf_rotation_probe import build_panels, run_variant
from strategy.etf_rotation import EtfRotationAdvanced

# 2013–2019 有数据的 24 只（实测，见 etf_index_proxy_2026-10-30.txt / 起点统计）
LONG_CODES = [
    "510300.SH", "510050.SH", "510500.SH", "159915.SZ", "159901.SZ",
    "510880.SH", "512010.SH", "512070.SH", "512880.SH", "512660.SH",
    "512100.SH", "512800.SH", "512400.SH", "512200.SH", "512980.SH",
    "159928.SZ", "159949.SZ", "518880.SH", "511010.SH", "511260.SH",
    "513100.SH", "513500.SH", "159920.SZ", "513050.SH",
]
BENCH = "000300.SH"
WARMUP = 20120101          # 暖机起点（满足 252 日动量 + 60 日成交额）
LONG_END = 20191231
LONG_START = 20140101      # 长样本计分起点
NEAR_END = 20261231
NEAR_START = 20200101      # 同池对照计分起点（池固定为 24 只）


def load_env(cache, codes, begin: int, end: int, min_amount: float = 3e7,
             pit_listed: int = 0) -> dict:
    """拉 ETF 自身日线 → 后复权 close/open + amount 面板（含全池 mask）。

    ``pit_listed`` > 0 时给 mask 叠一条**时点可得性**约束：该标的须已上市（有价）
    满 N 个交易日。语义 = 「回到 T 日那一天，你当时真的买得到它吗」。前提是
    ``begin`` 早于所有标的上市日（否则 cumsum 从窗口起点起算，会把老标的误判为新）。
    """
    frames, oframes, aframes = {}, {}, {}
    for code in codes:
        d = cache.get_index_daily(code, begin, end)
        if d is None or not len(d):
            continue
        frames[code] = d["close"].unstack("code").iloc[:, 0]
        oframes[code] = d["open"].unstack("code").iloc[:, 0]
        aframes[code] = d["amount"].unstack("code").iloc[:, 0]
    if not frames:
        raise RuntimeError("长样本价格为空")
    close = pd.DataFrame(frames).sort_index()
    open_ = pd.DataFrame(oframes).reindex(close.index)
    amount = pd.DataFrame(aframes).reindex(close.index)
    close = close[[c for c in codes if c in close.columns]]

    bf = cache.get_backward_factor(list(close.columns))
    if bf is not None and not bf.empty:
        bf = bf.reindex(index=close.index, columns=close.columns).ffill()
        close = close * bf
        open_ = open_ * bf
    idx = cache.get_index_daily(BENCH, begin, end)
    bclose = idx["close"].unstack("code").iloc[:, 0].reindex(close.index).ffill()
    bench = bclose.pct_change(fill_method=None)

    panels = build_panels(close, amount)
    # 真份额 → crowd 软惩罚 + pre10（与生产 `mom12_1_crowdsoft_pre10` 同源）。
    # 份额 PIT 按 ANN_DATE<=t ffill，2013-2019 有 34 只覆盖；缺份额的标的
    # over=fillna(0)⇒不扣分（与生产同一降级规则）。
    panels = build_extra_panels(close, panels)
    # 流动性门槛敏感性：长样本早期 ETF 成交额普遍低于 3000 万，硬门槛会把池子
    # 压到 2-3 只、使「选标」失去意义。故门槛可配（0 = 不加流动性约束）。
    panels["mask"] = (amount.rolling(60, min_periods=40).mean() >= min_amount
                      if min_amount > 0 else close.notna())
    if pit_listed > 0:
        # 上市天数代理 = 有价日累计（begin 早于全部上市日时等价于交易日龄）。
        panels["mask"] = panels["mask"] & (close.notna().cumsum() >= pit_listed)
    panels["age"] = close.notna().cumsum()
    mfs = pd.DatetimeIndex(
        pd.Series(close.index, index=close.index)
        .groupby(close.index.to_period("M")).first().to_numpy())
    return dict(bundle=(None, close, amount, bench), panels=panels, close=close,
                open_adj=open_, amount=amount, bench=bench, mfs=mfs)


def _run(env, spec: dict, cls=None, begin: int | None = None) -> dict:
    """单次回测，T+1 open 口径，与 probe.run 同源。

    ``begin`` 非 None 时按生产口径在计分起点切片（`slice_bundle`），代价是
    `ret_panel` 只有窗口内长度 ⇒ **HRP 协方差与去重冷启动**（run_variant docstring
    的窗口口径陷阱）。生产 `run(begin=BEGIN)` 就是这么跑的，故对照时两种都要看。
    """
    b, p = env["bundle"], env["panels"]
    r0 = run_variant(b, p, spec, begin, None, strategy_cls=cls)
    spl = build_execution_split(env["open_adj"], env["close"],
                                set(pd.DatetimeIndex(r0["rebalance_dates"])))
    return run_variant(b, p, spec, begin, None, execution_split=spl,
                       execution_shift=True, strategy_cls=cls)


def _eq_metrics(eq: pd.Series, start) -> dict:
    """按日期截断净值后重算指标；年化用几何、Sharpe = 几何年化 / 年化波动
    （与 `backtest.metrics.sharpe_ratio` 同口径，勿换成算术版）。"""
    eq = eq[eq.index >= pd.Timestamp(str(start))].dropna()
    if len(eq) < 30:
        return {"ann": np.nan, "sharpe": np.nan, "mdd": np.nan, "n": 0}
    ret = eq.pct_change().dropna()
    n = len(ret)
    ann = float((eq.iloc[-1] / eq.iloc[0]) ** (252.0 / n) - 1.0)
    vol = float(ret.std() * np.sqrt(252.0))
    mdd = float((eq / eq.cummax() - 1.0).min())
    return {"ann": ann, "sharpe": ann / vol if vol > 1e-9 else np.nan,
            "mdd": mdd, "n": n}


def _yearly(eq: pd.Series, start) -> pd.Series:
    eq = eq[eq.index >= pd.Timestamp(str(start))].dropna()
    g = eq.groupby(eq.index.year)
    return g.apply(lambda s: s.iloc[-1] / s.iloc[0] - 1.0 if len(s) > 1 else np.nan)


def _rand_sel_cls(seed: int, gate: bool = True):
    """随机选标（与 etf_weight_source 同实现，保留闸门）。"""

    class _RandSel(EtfRotationAdvanced):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._rng = np.random.default_rng(seed)

        def _select(self, vals, date=None):
            v = vals
            if gate and self.absolute_filter:
                v = v[v > self.abs_threshold]
            if v.empty:
                return []
            cand = list(v.index)
            k = min(self.top_k, len(cand))
            idx = self._rng.choice(len(cand), size=k, replace=False)
            return [cand[i] for i in idx]

    return _RandSel


class _PoolEq(EtfRotationAdvanced):
    """可入场池等权（**过同一道 abs 动量闸门 + pre10 筛后**的标的等权）。

    ⚠️ 语义边界：引擎传进来的 `factor_values` 是已剔 NaN 的信号截面，而 `pre10`
    会把 `mom20 > 10%` 的标的置 NaN ⇒ 它不是「全池买入持有」，而是「同一道筛子下
    不按排名选标的」的对照。用它读「选标 vs 不选标」，不要读成「池子 beta」。
    """

    def _weights(self, factor_values, date):
        vals = factor_values.dropna()
        if vals.empty:
            self._held = []
            return pd.Series(dtype=float)
        self._held = list(vals.index)
        return pd.Series(1.0 / len(vals), index=vals.index)


def main(argv=None) -> int:
    from data.cache import DataCache
    from data.datasource import create_datasource

    ap = argparse.ArgumentParser(description="指数代际长样本（2013–2019）区制检验")
    ap.add_argument("--mode", default="core", help="core / rand / phase / all")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--min-amount", type=float, default=3e7,
                    help="流动性门槛（元）；0 = 不加约束（敏感性检验用）")
    ap.add_argument("--start", type=int, default=LONG_START,
                    help="长样本计分起点（池子越早越小，见 docstring）")
    ap.add_argument("--cold-start", action="store_true",
                    help="按生产口径在计分起点切片（HRP 协方差冷启动），用于口径对齐")
    ap.add_argument("--csv-dir", default="reports/etf_contest5")
    args = ap.parse_args(argv)
    modes = args.mode.split(",") if args.mode != "all" else ["core", "rand", "phase"]
    cache = DataCache(create_datasource())

    print(f"长样本池 {len(LONG_CODES)} 只（2013–2019 有数据），暖机 {WARMUP}")
    env_l = load_env(cache, LONG_CODES, WARMUP, LONG_END, args.min_amount)
    print(f"  流动性门槛 {args.min_amount / 1e4:.0f} 万；计分起点 {args.start}")
    print(f"  长样本面板 {env_l['close'].shape} {env_l['close'].index.min().date()}"
          f"~{env_l['close'].index.max().date()}")
    env_n = load_env(cache, LONG_CODES, 20180101, NEAR_END, args.min_amount)
    print(f"  同池近样本面板 {env_n['close'].shape}")

    out = {}
    if "diag" in modes:
        for tag, env in (("2013–2019", env_l), ("2020–2026", env_n)):
            cl = env["close"]
            d = pd.DataFrame({
                "有价": cl.notna().sum(axis=1),
                "可交易": env["panels"]["mask"].sum(axis=1),
                "信号有效": env["panels"][FINAL["signal"]].notna().sum(axis=1),
                "有crowd": env["panels"]["crowd"].notna().sum(axis=1),
            })
            print(f"\n  == 覆盖诊断 {tag}（年日均只数）==")
            print(d.groupby(d.index.year).mean().round(1).to_string())
        print(f"\n  信号 = {FINAL['signal']}；成本 {FINAL['commission_bp']}bp+"
              f"{FINAL['slippage_bp']}bp；dedup {FINAL['dedup_corr']}")
    if "core" in modes:
        rows = []
        for tag, env, start in (("2013–2019", env_l, args.start),
                                ("2020–2026", env_n, NEAR_START)):
            b0 = start if args.cold_start else None
            print(f"  [{tag}] begin={b0}")
            for wt in ("hrp", "equal"):
                sp = dict(FINAL)
                sp["weighting"] = wt
                r = _run(env, sp, begin=b0)
                m = _eq_metrics(r["equity"], start)
                rows.append({"window": tag, "arm": wt, **m,
                             "holdings": float(r["avg_holdings"]),
                             "rebal": int(r["n_rebalance"])})
                print(f"  [{tag}] {wt:<6} 年化 {m['ann']:>7.2%} SR {m['sharpe']:>5.2f} "
                      f"MDD {m['mdd']:>7.2%} 持仓 {r['avg_holdings']:.2f}")
            r = _run(env, dict(FINAL), _PoolEq, begin=b0)
            m = _eq_metrics(r["equity"], start)
            rows.append({"window": tag, "arm": "pool_eq", **m,
                         "holdings": float(r["avg_holdings"]), "rebal": int(r["n_rebalance"])})
            print(f"  [{tag}] pool  年化 {m['ann']:>7.2%} SR {m['sharpe']:>5.2f} "
                  f"MDD {m['mdd']:>7.2%} 持仓 {r['avg_holdings']:.2f}")
        d = pd.DataFrame(rows)
        piv = d.pivot(index="window", columns="arm", values="ann")
        print("\n  == 放大器检验（同池同口径，只换窗口）==")
        for w in piv.index:
            hrp, eq, pool = piv.loc[w, "hrp"], piv.loc[w, "equal"], piv.loc[w, "pool_eq"]
            print(f"  {w}: HRP−等权 = {(hrp - eq) * 100:+.2f}pp；"
                  f"等权−全池 = {(eq - pool) * 100:+.2f}pp；HRP−全池 = {(hrp - pool) * 100:+.2f}pp")
        out["core"] = d
        y = pd.DataFrame({"2013–2019": _yearly(_run(env_l, dict(FINAL))["equity"], LONG_START),
                          "2020–2026": _yearly(_run(env_n, dict(FINAL))["equity"], NEAR_START)})
        print("\n  == HRP 分年 ==")
        print((y * 100).round(1).to_string())
        out["yearly"] = y.reset_index()

    if "pit" in modes:
        from data.etf_universe import ETF_CANDIDATES
        from scripts.portfolio.etf_contest_spec import EXCLUDED_5

        pool_all = [c for c in ETF_CANDIDATES if c not in set(EXCLUDED_5)]
        print(f"\n  == 时点可得池（PIT pool）== 生产池 {len(pool_all)} 只，"
              f"数据从 2012-01 起（早于全部上市日）")
        env_p = load_env(cache, pool_all, 20120101, NEAR_END, args.min_amount)
        cl = env_p["close"]
        ipo = cl.notna().idxmax()
        print("\n  -- 上市年份分布 --")
        print(ipo.dt.year.value_counts().sort_index().to_string())
        # 每年末「当时真的买得到」的只数（上市满 250 交易日 且 60 日均额达标）。
        base = env_p["panels"]["mask"]
        rows = []
        for y in range(2018, 2027):
            d = pd.Timestamp(f"{y}-12-31")
            if d > cl.index.max():
                d = cl.index.max()
            cut = cl.index[cl.index <= d][-1]
            avail = int(base.loc[cut].sum())
            listed = int((ipo <= cut).sum())
            rows.append({"year": y, "listed": listed, "tradable": avail})
        print("\n  -- 每年末可达池（上市且达标）--")
        print(pd.DataFrame(rows).set_index("year").to_string())

        for tag, n_listed in (("后视池（无 PIT 约束）", 0), ("上市满 250 日", 250),
                              ("上市满 500 日", 500)):
            pn = dict(env_p["panels"])
            pn["mask"] = (base if n_listed == 0
                          else (base & (pn["age"] >= n_listed)))
            env_x = dict(env_p, panels=pn)
            sp = dict(FINAL)
            r = _run(env_x, sp, begin=NEAR_START)
            m = _eq_metrics(r["equity"], NEAR_START)
            print(f"  [{tag:<18}] 年化 {m['ann']:>7.2%} SR {m['sharpe']:>5.2f} "
                  f"MDD {m['mdd']:>7.2%} 持仓 {r['avg_holdings']:.2f}")
            out[f"pit_{n_listed}"] = {"tag": tag, **m,
                                      "holdings": float(r["avg_holdings"])}
        pd.DataFrame(out).T.to_csv(os.path.join(args.csv_dir, "pool_pit.csv"),
                                   encoding="utf-8-sig")

    if "rand" in modes:
        rows = []
        for tag, env, start in (("2013–2019", env_l, LONG_START),
                                ("2020–2026", env_n, NEAR_START)):
            for wt in ("equal", "hrp"):
                sp = dict(FINAL)
                sp["weighting"] = wt
                a = []
                for sd in range(args.seeds):
                    a.append(_eq_metrics(_run(env, sp, _rand_sel_cls(sd))["equity"],
                                         start)["ann"])
                a = np.array(a)
                rows.append({"window": tag, "arm": f"rand_{wt}", "ann": float(a.mean()),
                             "sd": float(a.std(ddof=1)), "min": float(a.min()),
                             "max": float(a.max()), "n": args.seeds})
                print(f"  [{tag}] 随机+{wt:<6} {a.mean():>7.2%} (±{a.std(ddof=1):.2%})")
        out["rand"] = pd.DataFrame(rows)

    if "phase" in modes:
        rows = []
        days = list(env_l["close"].index)
        per = pd.Series(days, index=pd.DatetimeIndex(days)).groupby(
            pd.DatetimeIndex(days).to_period("W"))
        for wt in ("hrp", "equal"):
            sp = dict(FINAL)
            sp["weighting"] = wt
            anns = []
            for k in range(5):
                rb = {list(g)[k] for _, g in per if len(list(g)) > k}
                r0 = run_variant(env_l["bundle"], env_l["panels"], sp, None, None,
                                 rebalance_days=rb)
                spl = build_execution_split(env_l["open_adj"], env_l["close"],
                                            set(pd.DatetimeIndex(r0["rebalance_dates"])))
                r1 = run_variant(env_l["bundle"], env_l["panels"], sp, None, None,
                                 rebalance_days=rb, execution_split=spl,
                                 execution_shift=True)
                m = _eq_metrics(r1["equity"], LONG_START)
                anns.append(m["ann"])
                rows.append({"window": "2013–2019", "weighting": wt, "phase": k + 1, **m})
            print(f"  [{wt}] 相位 {['%.2f%%' % (x * 100) for x in anns]} "
                  f"极差 {(max(anns) - min(anns)) * 100:.2f}pp")
        out["phase"] = pd.DataFrame(rows)

    os.makedirs(args.csv_dir, exist_ok=True)
    for k, v in out.items():
        p = f"{args.csv_dir}/long_sample_{k}.csv"
        v.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"落盘 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
