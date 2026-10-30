"""收益来源分解：选标、集中、池子各自贡献多少（README §4.5）。

背景：定版三锋（Top3 + HRP）年化 30.63%，而**同选标三只等权只有 10.55%**
⇒ 表面上 +20pp 全来自「押重一只」。但那 10.55% 相对「全池等权」是高还是低？
集中到底是不是「押中黄金」的运气？

三个问题三组对照：
1. `pool`  —— 全池等权（当期 mask 内所有 ETF 等权）：市场/池子 beta。
2. `rand`  —— 随机选标 × {等权, HRP} × {闸门内, 全池}，N 个 seed：
   把「选标信息」与「集中形态」拆开。若「随机 3 只 + HRP」也能到 25%+，
   则集中是纯形态杠杆；若只有 12%，则选标与集中互补。
3. `attr`  —— 按标的的**加性**收益归因（对数分摊，总和精确等于组合对数收益），
   回答「收益是哪几只给的」。区别于 mode_attr（只看单期前瞻 t）。

口径与 eval 完全一致：T 日收盘出信号 + T+1 open 成交、同成本、同调仓日。
复现：`python -m scripts.portfolio.etf_weight_source --mode pool,rand,attr --seeds 8`
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from scripts.portfolio.etf_contest_spec import FINAL
from scripts.portfolio.etf_weight_probe import (
    BEGIN,
    _run_cls,
    hold_matrix,
    load_env,
    run,
)
from strategy.etf_rotation import EtfRotationAdvanced


class _PoolEq(EtfRotationAdvanced):
    """全池等权：当期因子截面（已 mask）里所有标的等权，不做 top_k 截断。

    这是「选池 + 每月全买」的基准，用来度量选标（Top3）是否真有信息。
    成本与调仓日同 eval ⇒ 换手更高，是它应承担的代价。
    """

    def _weights(self, factor_values, date):
        vals = factor_values.dropna()
        if vals.empty:
            self._held = []
            return pd.Series(dtype=float)
        sel = list(vals.index)
        self._held = sel
        return pd.Series(1.0 / len(sel), index=sel)


def _rand_sel_cls(seed: int, gate: bool):
    """随机选标：从当期候选里均匀抽 top_k 只，权重规则沿用 spec。

    ``gate=True`` 保留绝对动量闸门（只从「动量为正」的候选里抽），
    ``gate=False`` 则全池随机（含下跌标的）⇒ 给出「零信息」下界。
    """

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


def _summ(r1):
    return {"ann": r1["annual_return"], "sharpe": r1["sharpe"],
            "mdd": r1["max_drawdown"], "n": r1["n_rebalance"]}


def mode_pool(env, spec):
    print("\n== A. 全池等权基准（当期 mask 内全买）==")
    _, r1 = _run_cls(env, spec, _PoolEq)
    s = _summ(r1)
    hw = hold_matrix(r1["weights"])
    print(f"  年化 {s['ann']:.2%}  SR {s['sharpe']:.2f}  MDD {s['mdd']:.2%}"
          f"  调仓 {s['n']}  平均持仓只数 {float((hw > 1e-9).sum(axis=1).mean()):.1f}")
    return pd.DataFrame([{"arm": "全池等权", **s}])


def mode_rand(env, spec, n_seed: int):
    print(f"\n== B. 随机选标 × 权重规则（各 {n_seed} seed）==")
    rows = []
    for gate, gname in ((True, "闸门内"), (False, "全池")):
        for wt, wname in (("equal", "等权"), ("hrp", "HRP")):
            sp = dict(spec)
            sp["weighting"] = wt
            anns, srs, mdds, neff = [], [], [], []
            for sd in range(n_seed):
                _, r1 = _run_cls(env, sp, _rand_sel_cls(sd, gate))
                anns.append(r1["annual_return"])
                srs.append(r1["sharpe"])
                mdds.append(r1["max_drawdown"])
                hw = hold_matrix(r1["weights"])
                neff.append(float((1.0 / (hw ** 2).sum(axis=1).replace(0, np.nan)).median()))
            a = np.array(anns)
            rows.append({"arm": f"随机{gname}+{wname}", "ann": float(a.mean()),
                         "ann_sd": float(a.std(ddof=1)), "ann_min": float(a.min()),
                         "ann_max": float(a.max()), "sharpe": float(np.mean(srs)),
                         "mdd": float(np.mean(mdds)), "n_eff": float(np.mean(neff)),
                         "n": n_seed})
            print(f"  {rows[-1]['arm']:<14} 年化 {a.mean():>7.2%} "
                  f"(±{a.std(ddof=1):.2%}, {a.min():.1%}~{a.max():.1%}) "
                  f"SR {np.mean(srs):.2f} MDD {np.mean(mdds):.2%} "
                  f"有效只数 {np.mean(neff):.2f}")
    d = pd.DataFrame(rows)
    # 与实盘档位的差
    print("\n  对照：真选标+等权 10.55% / 真选标+HRP 30.63%（已有档位）")
    return d


def mode_attr(env, spec):
    """按标的的加性（对数分摊）收益归因，覆盖全部调仓期。"""
    _, r1 = run(env, spec)
    reb = list(pd.DatetimeIndex(r1["rebalance_dates"]))
    hw = hold_matrix(r1["weights"])
    close = env["close"]
    rows = []
    for i, E in enumerate(reb[:-1]):
        E2 = reb[i + 1]
        if E not in hw.index or E not in close.index or E2 not in close.index:
            continue
        w = hw.loc[E]
        w = w[w > 1e-9]
        if w.empty:
            continue
        px = close.loc[[E, E2], w.index]
        if px.isna().any().any():
            px = px.dropna(axis=1)
            w = w.reindex(px.columns)
            if w.empty:
                continue
        r = (px.loc[E2] / px.loc[E] - 1.0).astype(float)
        pret = float((w * r).sum())
        if pret <= -1.0:
            continue
        lret = np.log1p(pret)
        for c in w.index:
            rows.append({"date": E, "code": c, "w": float(w[c]), "fwd": float(r[c]),
                         "contrib": lret * float(w[c] * r[c]) / pret if abs(pret) > 1e-12 else 0.0})
    d = pd.DataFrame(rows)
    g = d.groupby("code").agg(days=("date", "nunique"), w_mean=("w", "mean"),
                              contrib=("contrib", "sum"))
    g["share"] = g["contrib"] / g["contrib"].sum()
    g = g.sort_values("contrib", ascending=False)
    total = float(g["contrib"].sum())
    print(f"\n== C. 按标的的加性收益归因（{d['date'].nunique()} 期，"
          f"{len(g)} 只曾持有）==")
    print(f"  全部持仓的对数收益合计 {total:.4f} "
          f"（≈累计 {np.expm1(total):.1%}）")
    print(f"  {'代码':<12}{'被持天数':>9}{'平均权重':>10}{'贡献(对数)':>12}{'占比':>9}")
    for c, row in g.head(12).iterrows():
        print(f"  {c:<12}{int(row['days']):>9}{row['w_mean']:>10.1%}"
              f"{row['contrib']:>+12.4f}{row['share']:>9.1%}")
    print(f"  前 3 只合计占比 {g['share'].head(3).sum():.1%}；"
          f"前 10 只 {g['share'].head(10).sum():.1%}；"
          f"负贡献只数 {(g['contrib'] < 0).sum()}")
    # 集中度：多少比例收益来自「被押(>50%权重)的那些期」
    dw = d[d["w"] > 0.5]
    print(f"  单只权重 >50% 的「标的×期」共 {len(dw)} 个，"
          f"贡献 {dw['contrib'].sum() / total:.1%}；"
          f"其余 {1 - dw['contrib'].sum() / total:.1%}")
    return g.reset_index()


def mode_phase(env, spec, weighting: str):
    """相位运气测试：同样 5 个调仓相位，只换权重规则。

    若等权下的相位极差远小于 HRP 的 20.88pp，则「起点月份运气」很大程度上是
    **杠杆放大出来的**，而非策略本身对起点敏感。
    """
    from scripts.portfolio.etf_weight_probe import mode_phase as _mp

    sp = dict(spec)
    sp["weighting"] = weighting
    print(f"\n== D. 相位（weighting={weighting}）==")
    d = _mp(env, sp)
    d["weighting"] = weighting
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="收益来源分解（选标 / 集中 / 池子）")
    ap.add_argument("--mode", default="pool,rand,attr")
    ap.add_argument("--seeds", type=int, default=8, help="随机选标方的 seed 数")
    ap.add_argument("--csv-dir", default="reports/etf_contest5")
    args = ap.parse_args(argv)
    modes = [s.strip() for s in args.mode.split(",")]
    env = load_env()
    spec = dict(FINAL)
    print(f"窗口 {BEGIN}→末日；spec = FINAL（三锋）")
    out = {}
    if "pool" in modes:
        out["pool"] = mode_pool(env, spec)
    if "rand" in modes:
        out["rand"] = mode_rand(env, spec, args.seeds)
    if "attr" in modes:
        out["attr"] = mode_attr(env, spec)
    if "phase" in modes:
        out["phase"] = pd.concat([mode_phase(env, spec, "hrp"),
                                  mode_phase(env, spec, "equal")],
                                 ignore_index=True)
    for k, v in out.items():
        p = f"{args.csv_dir}/weight_source_{k}.csv"
        v.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"落盘 → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
