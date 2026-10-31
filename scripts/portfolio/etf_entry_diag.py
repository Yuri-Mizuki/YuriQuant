"""ETF 轮动 · 两个「入场前提」诊断：HRP 集中度是否异常、回测起点是否合理。

回答的是两个**不属于参数优化**的问题，因此不进判否清单：

## ``--mode hrp``：80% 单只是 HRP 的固有输出，还是策略额外放大？

``§4.4.1`` 已给出 n=3 的解析式 ``alpha = 1 − 1/(3+2ρ)``。本模式补一个**外部验证**：
从候选池随机抽 (3 只 × 60 日) 组合直接喂 ``_hrp_weights``，看最大权重的分布。
若本策略实测的 80.7% 落在该分布中部 ⇒ 集中度来自「HRP + Top3」，不是来自选标。

## ``--mode begin``：``BEGIN=20200101`` 是暖机边界还是挑出来的窗口？

数据能到 2013（24 只老池），``WARMUP_BEGIN=20180101``（12-1 动量需 ~273 交易日）
⇒ 技术最早可起跑约 2019-02。本模式跑 2019 / 2020 / 2021 三个起点，量化窗口选择
的代价，并给出逐年收益分解（看收益是否集中在个别年份）。

两者都只产数，判断写在 ``reports/etf_contest5/README.md`` §4.4.1 与 §零 硬约束 4。

## ``--mode combo``：两个独立报数偏差叠加后到底是多少？

相位（k=1 是放大档，§4.8.2）与起点（2020 排除 2019 零收益年）是**两个独立偏差**。
本模式跑 5 相位 × 2 起点共 10 格，给出 ``30.63% → 17.17% → 15.64%`` 这条收窄路径，
并拆出「起点偏差在 k=1 上是多少、在相位平均上又是多少」——实测两者都集中在 k=1。

用法::

    python -m scripts.portfolio.etf_entry_diag --mode hrp
    python -m scripts.portfolio.etf_entry_diag --mode begin
    python -m scripts.portfolio.etf_entry_diag --mode combo
    python -m scripts.portfolio.etf_entry_diag --mode hrp,begin,combo --csv out.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.etf_rotation import _hrp_weights

ROOT = Path(__file__).resolve().parents[2]
PHASES_NOTE = "起点敏感性用 T+1 open 口径，与 etf_weight_probe.run 同源"


def _cache_close():
    from config import Config
    from data.cache import DataCache

    cache = DataCache(Config.paths())
    d = cache.read_daily("etf")
    return d["close"].unstack("code").sort_index()


def mode_hrp(n_draw: int = 200, seed: int = 7) -> pd.DataFrame:
    """随机 (3 只 × 60 日) 组合的 HRP 最大权重分布。"""
    from data.etf_universe import ETF_CANDIDATES

    close = _cache_close()
    ret = close.pct_change(fill_method=None).dropna(how="all")
    codes = [c for c in ETF_CANDIDATES if c in close.columns]
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for _ in range(n_draw):
        pick = list(rng.choice(codes, 3, replace=False))
        end = int(rng.integers(120, len(ret) - 1))
        sub = ret[pick].iloc[end - 60:end].dropna()
        if len(sub) < 40:
            continue
        cov = sub.cov().to_numpy() * 252
        w = _hrp_weights(cov)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        i = int(np.argmax(w))
        others = [j for j in range(3) if j != i]
        rows.append({
            "maxw": float(w[i]),
            "rho_top_to_others": float(np.mean([corr[i, j] for j in others])),
            "rho_others_pair": float(corr[others[0], others[1]]),
            "vol_ratio_top_vs_others":
                float(sd[i] / np.mean([sd[j] for j in others])),
        })
    d = pd.DataFrame(rows)

    print(f"\n== HRP 在 n=3 时的最大权重分布（随机 {len(d)} 组，60 日窗）==")
    print("  分位: " + "  ".join(
        f"p{int(q * 100)}={d.maxw.quantile(q):.0%}"
        for q in (0.10, 0.25, 0.50, 0.75, 0.90)))
    print(f"  >60% 占 {(d.maxw > 0.60).mean():.1%}；"
          f">70% 占 {(d.maxw > 0.70).mean():.1%}；"
          f">80% 占 {(d.maxw > 0.80).mean():.1%}")
    print("  对照：本策略实测最大单只权重中位 80.7%（README §4.4.2）")

    hi = d[d.maxw > 0.60]
    if len(hi):
        print(f"\n  最大权重 >60% 的 {len(hi)} 组，其池子结构：")
        print(f"    头部与另两只平均 ρ {hi.rho_top_to_others.mean():+.2f}"
              f"　另两只之间 ρ {hi.rho_others_pair.mean():+.2f}"
              f"　头部波动/另两只 {hi.vol_ratio_top_vs_others.mean():.2f}")
        print("    ⇒ 典型分区 = 1 个离群 + 1 对孪生，`alpha` 逼近 0.79")
    return d


def mode_begin(begins: tuple[int, ...] = (20190101, 20200101, 20210101)) -> pd.DataFrame:
    """起点敏感性 + 逐年收益分解。"""
    from scripts.portfolio.etf_contest_spec import FINAL
    from scripts.portfolio.etf_weight_probe import load_env, run

    env = load_env()
    spec = dict(FINAL)
    print(f"\n  面板范围 {env['close'].index.min().date()} → "
          f"{env['close'].index.max().date()}")

    rows, eqs = [], {}
    for begin in begins:
        _r0, r1 = run(env, spec, begin=begin)
        e = r1["equity"].dropna()
        eqs[begin] = e
        rows.append({
            "begin": begin,
            "first_reb": str(pd.Timestamp(r1["rebalance_dates"][0]).date()),
            "n_reb": len(r1["rebalance_dates"]),
            "years": (e.index[-1] - e.index[0]).days / 365.25,
            "ann": r1["annual_return"],
            "sharpe": r1["sharpe"],
            "mdd": r1["max_drawdown"],
        })
    d = pd.DataFrame(rows)

    print("\n== 起点敏感性（同池 45 只、同信号、T+1 open）==")
    print(f"  {'begin':>9}{'首个调仓':>12}{'调仓次':>7}{'年数':>7}"
          f"{'年化':>9}{'SR':>7}{'MDD':>9}")
    for _, x in d.iterrows():
        print(f"  {x.begin:>9}{x.first_reb:>12}{x.n_reb:>7.0f}{x.years:>7.1f}"
              f"{x.ann:>9.2%}{x.sharpe:>7.2f}{x.mdd:>9.2%}")
    print("  自检：begin=20200101 应复现对外口径 30.63% / 1.16 / 24.97%")

    if 20190101 in eqs and 20200101 in eqs:
        a0 = float(d.loc[d.begin == 20190101, "ann"].iloc[0])
        a1 = float(d.loc[d.begin == 20200101, "ann"].iloc[0])
        print(f"\n  ⇒ 排除 2019 年使年化 {(a1 - a0) * 100:+.2f}pp"
              f"（{a0:.2%} → {a1:.2%}）")

    print("\n== 逐年收益（几何分解）==")
    yr = {}
    for b, e in eqs.items():
        ye = e.resample("YE").last()
        r = ye.pct_change()
        r.iloc[0] = ye.iloc[0] / float(e.iloc[0]) - 1.0
        yr[b] = r
    tab = pd.DataFrame(yr)
    tab.index = tab.index.year
    print((tab * 100).round(1).to_string())

    top = tab[20200101].dropna().sort_values(ascending=False)
    print("\n  收益最集中的三年（2020 起跑）："
          + "、".join(f"{int(y)} {v:.1%}" for y, v in top.head(3).items()))
    print("  ⇒ 披露时须说明：收益主要由少数年份贡献，不能按「七年均值」外推")
    return d


def mode_combo(begins: tuple[int, ...] = (20190101, 20200101)) -> pd.DataFrame:
    """相位平均 × 起点：两个**独立**报数偏差叠加后的诚实读数。"""
    from scripts.portfolio.etf_contest_spec import FINAL
    from scripts.portfolio.etf_phase_rule_matrix import phase_rebalance_days
    from scripts.portfolio.etf_weight_probe import load_env, run

    env = load_env()
    rb = phase_rebalance_days(env["close"].index)
    spec = dict(FINAL)

    rows: list[dict] = []
    for begin in begins:
        for k in range(1, 6):
            _r0, r1 = run(env, spec, rebalance_days=rb[k], begin=begin)
            rows.append({"begin": begin, "k": k,
                         "ann": r1["annual_return"],
                         "sharpe": r1["sharpe"], "mdd": r1["max_drawdown"]})
            print(f"  begin={begin} k={k}  {r1['annual_return']:>7.2%}  "
                  f"SR {r1['sharpe']:>5.2f}  MDD {r1['max_drawdown']:>7.2%}",
                  flush=True)
    d = pd.DataFrame(rows)

    print("\n== 相位平均 × 起点 ==")
    print(f"  {'起点':>9}{'年化均':>9}{'极差':>8}{'SR均':>7}{'MDD均':>9}{'Calmar':>8}")
    for b, g in d.groupby("begin"):
        rng_ = (g.ann.max() - g.ann.min()) * 100
        print(f"  {b:>9}{g.ann.mean():>9.2%}{rng_:>7.1f}pp{g.sharpe.mean():>7.2f}"
              f"{g.mdd.mean():>9.2%}{g.ann.mean() / g.mdd.mean():>8.2f}")

    if 20190101 in begins and 20200101 in begins:
        a19 = float(d[d.begin == 20190101].ann.mean())
        a20 = float(d[d.begin == 20200101].ann.mean())
        k1_19 = float(d[(d.begin == 20190101) & (d.k == 1)].ann.iloc[0])
        k1_20 = float(d[(d.begin == 20200101) & (d.k == 1)].ann.iloc[0])
        print("\n  两个独立偏差的叠加：")
        print(f"    k=1 + 2020 起（原对外口径）  {k1_20:>7.2%}")
        print(f"    相位平均 + 2020 起           {a20:>7.2%}")
        print(f"    相位平均 + 2019 起（最诚实） {a19:>7.2%}")
        print(f"    总落差                        {(a19 - k1_20) * 100:>+6.2f}pp")
        print(f"    起点偏差：k=1 上 {(k1_19 - k1_20) * 100:+.2f}pp，"
              f"相位平均上 {(a19 - a20) * 100:+.2f}pp")
        print("    ⇒ 两个偏差都集中在 k=1（周一）⇒ 30.63% 是双重最幸运的组合")
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ETF 入场前提诊断（HRP 集中度 / 回测起点）")
    ap.add_argument("--mode", default="hrp,begin",
                    help="hrp / begin / combo / 逗号分隔")
    ap.add_argument("--draws", type=int, default=200, help="hrp 模式随机抽样次数")
    ap.add_argument("--begins", default="20190101,20200101,20210101",
                    help="begin/combo 模式的起点列表（YYYYMMDD，逗号分隔）")
    ap.add_argument("--csv", default="", help="明细落盘路径（可选）")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.mode.split(",") if m.strip()]
    out: dict[str, pd.DataFrame] = {}

    if "hrp" in modes:
        out["hrp"] = mode_hrp(args.draws)
    if "begin" in modes:
        begins = tuple(int(b) for b in args.begins.split(",") if b.strip())
        out["begin"] = mode_begin(begins)
    if "combo" in modes:
        cb = tuple(int(b) for b in args.begins.split(",") if b.strip())[:2]
        out["combo"] = mode_combo(cb or (20190101, 20200101))

    if args.csv and out:
        frames = []
        for name, df in out.items():
            x = df.copy()
            x.insert(0, "mode", name)
            frames.append(x)
        pd.concat(frames, ignore_index=True, sort=False).to_csv(
            args.csv, index=False, encoding="utf-8-sig")
        print(f"\n明细已落盘 {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
