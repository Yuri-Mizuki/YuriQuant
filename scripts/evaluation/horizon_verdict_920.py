"""定版 horizon 结构的正式裁决：h20-only 该不该取代 h1020 坐上快信号腿。

为什么要有这个脚本
------------------
归档里 h20-only 只有一句判词——「年化最高、换手最低，但 2020/2022/2025 大胜、
2019/2024/2026 落后，回撤 41.1% 最深；ens_h1h5h10h20 曲线最平滑」
（``TODO_ARCHIVE.md`` 1813-1817），此后每次引用都靠这句 caveat 顶着。它缺三件东西，
导致这条线索既不能投产也不能结案：

1. **壳不对**。归档数是 naive Top10%（``TopFracLongOnly(0.10)``）口径，定版是
   buffer(15/40)。buffer 把换手腰斩、把截止线噪声钝化，**恰好是 h20 相对 h1020
   唯一的结构性优势（换手低）的作用面**——旧壳测出来的优势在生产壳里可能被吃掉
   （集中化裁决就是这么翻的：研究口径 +1.9pp 被 T+1 open + 成本吃光）。
2. **没跟慢腿共存过**。定版是 ``h1020⊕s0.5``，h20 从未与慢信号组合。慢腿本身
   换手极低、信息偏 h20，两者**可能是同一个 beta 的两个测量**，叠加不分散。
3. **没有预注册门槛**。「年化高 1.3~2pp」单独一条不构成换掉定版的理由——本项目
   自测 PBO 0.70，全样本选峰值必然赢。

本脚本一次补齐三件事，并把门槛**写死在 ``GATES`` 里、跑之前就定好**（先定标准
后看数，避免「看着数字找理由」）。

预注册裁决门槛（改定版需**全部**通过）
------------------------------------
G1 年化领先 ≥ +1.00pp（相对定版 ``h1020⊕s0.5``，同壳同窗口）
G2 Sharpe 不低于定版 − 0.010
G3 最大回撤不深于定版 + 1.00pp
G4 逐年收益胜率 ≥ 5/9（vs 定版逐年）
G5 换手 ≤ 定版 × 1.25（否则省下的换手就是它唯一的来源，成本口径一变就翻）

任一不过 ⇒ h20 记为「同族替身」，定版维持 h1020，**不因单点年化改版**。
两窗口（全样本 2018 起 / 2019 起）都算，只在全样本过、2019 窗不过的也算不过。

诊断量（不进门槛，只用于解释）：OOS IC、β、两腿截面秩相关、日收益差 t 值。

用法：
    python -m scripts.evaluation.horizon_verdict_920
    python -m scripts.evaluation.horizon_verdict_920 --only h1020_s0.5,h20_s0.5
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("horizon_verdict_920")

#: 定版锚（``buffer_arm_920 --variant defv``，2026-10-21）：λ=0.5 臂必须复现
ANCHOR = {"annual": 0.1631, "sharpe": 0.738}
TOL = {"annual": 0.0015, "sharpe": 0.006}
ANCHOR_ARM = "h1020_s0.5"

#: 预注册门槛（先定后看；阈值改动必须回写报告）
GATES = {
    "annual_gain_pp": 1.00,   # G1 年化领先
    "sharpe_slack": 0.010,    # G2 允许的 Sharpe 让步
    "max_dd_slack": 0.010,    # G3 允许的额外回撤
    "win_years": 5,           # G4 逐年胜率 ≥ 5/9
    "turnover_mult": 1.25,    # G5 换手上限倍数
}

#: 臂定义。kind: single=单 horizon / avg=秩平均 / wavg=加权秩平均 / slow=快慢融合
ARMS: dict[str, dict] = {
    "h1_only": {"kind": "single", "h": ["h1"]},
    "h5_only": {"kind": "single", "h": ["h5"]},
    "h10_only": {"kind": "single", "h": ["h10"]},
    "h20_only": {"kind": "single", "h": ["h20"]},
    "ens_h1h5": {"kind": "avg", "h": ["h1", "h5"]},
    "h1020": {"kind": "avg", "h": ["h1", "h5", "h10", "h20"]},
    "blend_h20_l25": {"kind": "wavg", "h": ["h1", "h5", "h20"], "w": [3, 3, 2]},
    "blend_h20_l40": {"kind": "wavg", "h": ["h1", "h5", "h20"], "w": [3, 3, 4]},
    "h1020_s0.5": {"kind": "slow", "base": "h1020", "lam": 0.5},
    "h1020_s0.3": {"kind": "slow", "base": "h1020", "lam": 0.3},
    "h20_s0.3": {"kind": "slow", "base": "h20_only", "lam": 0.3},
    "h20_s0.5": {"kind": "slow", "base": "h20_only", "lam": 0.5},
    "h20_s0.7": {"kind": "slow", "base": "h20_only", "lam": 0.7},
    # 归档数对拍：同信号、naive Top10% 壳（复现 19.22% 那个口径）
    "naive_h1020": {"kind": "avg", "h": ["h1", "h5", "h10", "h20"], "naive": True},
    "naive_h20": {"kind": "single", "h": ["h20"], "naive": True},
}


def wrank_average(panels: list[pd.DataFrame], weights: list) -> pd.DataFrame:
    """加权秩平均（与 ``horizon_mix.weighted_rank_average`` 同构）。"""
    out = pd.DataFrame(np.nan, index=panels[0].index, columns=panels[0].columns)
    wv = np.asarray(weights, dtype=float)
    for d in panels[0].index:
        rows = []
        for p in panels:
            if d not in p.index:
                continue
            row = p.loc[d]
            r = row.rank(pct=True)
            rows.append(r[~row.isna()])
        if len(rows) < 2:
            continue
        valid = rows[0].index
        for r in rows[1:]:
            valid = valid.intersection(r.index)
        if len(valid) < 1:
            continue
        sub = [r.reindex(valid) for r in rows]
        w = wv[: len(rows)]
        out.loc[d, valid] = (pd.concat(sub, axis=1) * w).sum(axis=1) / w.sum()
    return out


def build_signal(spec: dict, ranks: dict, slow_r: pd.DataFrame) -> pd.DataFrame:
    """按臂定义合成信号面板（ranks 只存已算好的快信号秩面板）。"""
    kind = spec["kind"]
    if kind == "single":
        return ranks[spec["h"][0]]
    if kind == "avg":
        cols = [ranks[h] for h in spec["h"]]
        return sum(cols) / len(cols)
    if kind == "wavg":
        w = np.asarray(spec["w"], dtype=float)
        w = w / w.sum()
        return sum(r * wi for r, wi in zip([ranks[h] for h in spec["h"]], w))
    if kind == "slow":
        base = ranks[spec["base"]]
        lam = float(spec["lam"])
        return (1.0 - lam) * base.rank(axis=1, pct=True) + lam * slow_r
    raise KeyError(kind)


def rank_one(p: pd.DataFrame) -> pd.DataFrame:
    return p.rank(axis=1, pct=True).astype(np.float32)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="h20-only 正式裁决（生产壳）")
    ap.add_argument("--pred-dir", default=str(ROOT / "reports" / "alla_rolling_ortho920fundind"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "horizon_verdict_920"))
    ap.add_argument("--entry", type=float, default=0.15)
    ap.add_argument("--exit-frac", type=float, default=0.40)
    ap.add_argument("--naive-frac", type=float, default=0.10,
                    help="naive 对照壳的 Top 分位（归档口径 0.10）")
    ap.add_argument("--start", default="2018-01-01", help="主窗口起点")
    ap.add_argument("--sub-start", default="2019-01-01", help="次窗口起点")
    ap.add_argument("--only", default="", help="只跑指定臂（逗号分隔，调试用）")
    args = ap.parse_args(argv)

    from backtest.costs import default_costs
    from backtest.engine import VectorBacktest, build_execution_split
    from scripts.common import alla_common as AC
    from scripts.evaluation.fundamental_blend import build_slow_panel
    from scripts.evaluation.horizon_mix import load
    from stats.ic import calc_ic_series
    from strategy.examples import BufferedTopFracLongOnly, TopFracLongOnly

    pred_dir, dest = Path(args.pred_dir), Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    names = [a for a in (args.only.split(",") if args.only else ARMS)]
    missing = [a for a in names if a not in ARMS]
    if missing:
        raise SystemExit(f"未知臂：{missing}")

    raw = {h: load(pred_dir, f"gbdt__{h}") for h in ("h1", "h5", "h10", "h20")}
    oos_days = raw["h1"].index
    log.info("预测面板 %s 就绪 %.0fs", raw["h1"].shape, time.time() - t0)

    base = AC.load_base()
    close = base["close"]
    mask = base["mask"].astype(bool)
    costs = default_costs()
    fwd = close.pct_change(fill_method=None).reindex(close.index).loc[oos_days]
    mask_oos = mask.reindex(index=oos_days, columns=close.columns).fillna(True)
    bench_idx = base["bench_index"].reindex(oos_days).fillna(0.0)

    need_slow = any(ARMS[a]["kind"] == "slow" for a in names)
    slow_r = None
    if need_slow:
        slow, _sm, _w = build_slow_panel(base, oos_days, 24, 2, 0.5, 12, log=print)
        slow_r = rank_one(slow.reindex(index=oos_days, columns=close.columns))
        del slow
        gc.collect()
        log.info("慢信号就绪 %.0fs", time.time() - t0)

    # 执行口径：T+1 open（生产），月度调仓日 = 每月首日信号的次一交易日
    fp = pred_dir / "_base" / "open_adj.parquet"
    if not fp.exists():
        raise SystemExit(f"{fp} 缺失：先跑 rolling_grid_alla --stage prep")
    fill = pd.read_parquet(fp)
    s_ = pd.Series(oos_days, index=oos_days)
    firsts = s_.groupby(s_.index.to_period("M")).first()
    pos_ = {d: i for i, d in enumerate(oos_days)}
    rb_signal = {t for t in firsts if pos_[t] + 1 < len(oos_days)}
    rb_exec = {oos_days[pos_[t] + 1] for t in rb_signal}
    exec_split = build_execution_split(fill, close, rb_signal)

    def _bt(sig_in: pd.DataFrame, naive: bool):
        strat = TopFracLongOnly(frac=args.naive_frac, weight_mode="equal") if naive \
            else BufferedTopFracLongOnly(args.entry, args.exit_frac)
        bt = VectorBacktest(strategy=strat, rebalance_freq="M",
                            initial_capital=1_000_000.0, costs=costs)
        return bt.run(sig_in.shift(1).where(mask_oos), fwd, horizon=1,
                      rebalance_days=rb_exec, execution_split=exec_split)

    def _window(dr: pd.Series, start: str) -> dict:
        d = dr.dropna()
        d = d[d.index >= pd.Timestamp(start)]
        m = AC.res_metrics(d, bench_idx, None)
        return {"annual": m["annual"], "sharpe": m["sharpe"], "max_dd": m["max_dd"]}

    # 排序：先算快信号臂（缓存秩面板），slow 臂最后（依赖 base 秩面板）
    order = [a for a in names if ARMS[a]["kind"] != "slow"] + \
            [a for a in names if ARMS[a]["kind"] == "slow"]
    cache_keys = {ARMS[a].get("base") for a in names if ARMS[a]["kind"] == "slow"}
    ranks: dict[str, pd.DataFrame] = {}
    for k in sorted(x for x in cache_keys if x):
        if ARMS[k]["kind"] == "single":
            ranks[k] = rank_one(raw[ARMS[k]["h"][0]])
        else:
            ranks[k] = rank_one(sum(rank_one(raw[h]) for h in ARMS[k]["h"])
                                / len(ARMS[k]["h"]))
        log.info("缓存基数面板 %s", k)

    rows, yearly_rows, dall = [], [], {}
    for name in order:
        spec = ARMS[name]
        if spec["kind"] == "slow":
            sig = build_signal(spec, ranks, slow_r)
        elif spec["kind"] == "single":
            sig = rank_one(raw[spec["h"][0]])
        elif spec["kind"] == "avg":
            sig = sum(rank_one(raw[h]) for h in spec["h"]) / len(spec["h"])
        else:
            sig = wrank_average([raw[h] for h in spec["h"]], spec["w"])

        sig_in = sig.reindex(index=oos_days, columns=close.columns)
        ic = float(calc_ic_series(sig_in, fwd.shift(-1)).mean())
        res = _bt(sig_in, bool(spec.get("naive", False)))
        dr = res.daily_returns
        m = AC.res_metrics(dr, bench_idx, res.turnover_series)
        w19 = _window(dr, args.sub_start)
        d = dr.dropna()
        ret_y = {int(y): float((1 + sub).prod() - 1)
                 for y, sub in d.groupby(d.index.year)}
        row = {"arm": name, "wrapper": "naive" if spec.get("naive") else "buffer_15_40",
               "oos_ic": ic, "annual": m["annual"], "sharpe": m["sharpe"],
               "ir": m["ir"], "max_dd": m["max_dd"], "turnover": m["turnover"],
               "beta": m["beta"], "excess_idx": m["excess"],
               "annual_2019": w19["annual"], "sharpe_2019": w19["sharpe"],
               "max_dd_2019": w19["max_dd"], "n_days": int(d.shape[0]),
               **{f"ret_{y}": v for y, v in ret_y.items()}}
        rows.append(row)
        yearly_rows.append({"arm": name, **{f"{y}": round(v * 100, 2)
                                            for y, v in ret_y.items()}})
        dall[name] = d
        log.info("%-14s %s | IC=%.4f 年化=%6.2f%% Sharpe=%.3f 回撤=%5.1f%% "
                 "换手=%5.1f%% | 2019窗 %6.2f%%/%.3f",
                 name, row["wrapper"][:6], ic, m["annual"] * 100, m["sharpe"],
                 m["max_dd"] * 100, m["turnover"] * 100,
                 w19["annual"] * 100, w19["sharpe"])
        del sig, sig_in, res
        gc.collect()

    table = pd.DataFrame(rows)
    table.to_csv(dest / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly_rows).to_csv(dest / "yearly.csv", index=False,
                                     encoding="utf-8-sig")

    # ---- 锚自检：定版臂必须复现 ----
    hit = table[table["arm"] == ANCHOR_ARM]
    if not hit.empty:
        a, s = float(hit["annual"].iloc[0]), float(hit["sharpe"].iloc[0])
        ok = abs(a - ANCHOR["annual"]) <= TOL["annual"] and \
            abs(s - ANCHOR["sharpe"]) <= TOL["sharpe"]
        log.info("锚自检 %s：%.2f%% / %.3f vs 登记 %.2f%% / %.3f → %s",
                 ANCHOR_ARM, a * 100, s, ANCHOR["annual"] * 100, ANCHOR["sharpe"],
                 "通过" if ok else "❌ 不符，先查口径")
        if not ok:
            raise SystemExit("定版臂未复现锚点，裁决不可用")

    # ---- 诊断：两腿截面秩相关 + 日收益差显著性 ----
    diag = {}
    if "h20_only" in dall and "h1020" in dall:
        dr_a, dr_b = dall["h20_only"], dall["h1020"]
        diff = (dr_a - dr_b).dropna()
        t = float(diff.mean() / diff.std() * np.sqrt(252)) if diff.std() > 0 else np.nan
        diag["diff_t_h20_vs_h1020"] = t
        diag["corr_daily_h20_h1020"] = float(dr_a.corr(dr_b))
    if "h20_s0.5" in dall and ANCHOR_ARM in dall:
        diff = (dall["h20_s0.5"] - dall[ANCHOR_ARM]).dropna()
        diag["diff_t_h20s_vs_defv"] = float(
            diff.mean() / diff.std() * np.sqrt(252)) if diff.std() > 0 else np.nan
        diag["corr_daily_h20s_defv"] = float(dall["h20_s0.5"].corr(dall[ANCHOR_ARM]))
    if slow_r is not None:
        def _cs_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
            cs = []
            for d in oos_days[::20]:  # 每 20 日取样，控制内存
                x, y = a.loc[d], b.loc[d]
                ok = x.notna() & y.notna()
                if ok.sum() > 100:
                    cs.append(x[ok].corr(y[ok], method="spearman"))
            return float(np.nanmean(cs)) if cs else np.nan

        for k in ("h20_only", "h1020"):
            if k in ranks:
                diag[f"cs_rank_corr_{k}_slow"] = _cs_corr(ranks[k], slow_r)
        if "h20_only" in ranks and "h1020" in ranks:
            diag["cs_rank_corr_h20_h1020"] = _cs_corr(ranks["h20_only"], ranks["h1020"])
    pd.Series(diag).to_csv(dest / "diagnostics.csv", header=["value"],
                           encoding="utf-8-sig")
    log.info("诊断：%s", {k: round(v, 4) for k, v in diag.items() if v == v})

    # ---- 门槛裁决 ----
    gate_rows = []
    if "h20_s0.5" in set(table["arm"]) and ANCHOR_ARM in set(table["arm"]):
        anc = table.set_index("arm").loc[ANCHOR_ARM]
        cand = table.set_index("arm").loc["h20_s0.5"]
        for win, sfx in (("full", ""), ("2019", "_2019")):
            a_ann, c_ann = float(anc[f"annual{sfx}"]), float(cand[f"annual{sfx}"])
            a_sh, c_sh = float(anc[f"sharpe{sfx}"]), float(cand[f"sharpe{sfx}"])
            a_dd, c_dd = float(anc[f"max_dd{sfx}"]), float(cand[f"max_dd{sfx}"])
            # 逐年胜率：2019 窗口不含 2018（该年样本只覆盖 10 月起）
            floor = "ret_2018" if win == "full" else "ret_2019"
            yrs = [c for c in table.columns
                   if c.startswith("ret_") and c >= floor]
            wins = sum(float(cand[c]) > float(anc[c]) + 1e-9 for c in yrs)
            checks = [
                ("G1_annual", (c_ann - a_ann) * 100 >= GATES["annual_gain_pp"],
                 f"{(c_ann - a_ann) * 100:+.2f}pp (需 ≥+{GATES['annual_gain_pp']:.2f})"),
                ("G2_sharpe", c_sh >= a_sh - GATES["sharpe_slack"],
                 f"{c_sh:.3f} vs {a_sh:.3f} (容差 −{GATES['sharpe_slack']:.3f})"),
                ("G3_maxdd", c_dd <= a_dd + GATES["max_dd_slack"],
                 f"{c_dd * 100:.1f}% vs {a_dd * 100:.1f}% (容差 +{GATES['max_dd_slack'] * 100:.0f}pp)"),
                ("G4_winyears", wins >= GATES["win_years"],
                 f"{wins}/{len(yrs)} (需 ≥{GATES['win_years']})"),
                ("G5_turnover",
                 float(cand["turnover"]) <= float(anc["turnover"]) * GATES["turnover_mult"],
                 f"{cand['turnover'] * 100:.1f}% vs {anc['turnover'] * 100:.1f}% "
                 f"(上限 ×{GATES['turnover_mult']:.2f})"),
            ]
            for gate, ok, detail in checks:
                gate_rows.append({"window": win, "gate": gate,
                                  "passed": bool(ok), "detail": detail})
    if gate_rows:
        gdf = pd.DataFrame(gate_rows)
        gdf.to_csv(dest / "gate_check.csv", index=False, encoding="utf-8-sig")
        for win in gdf["window"].unique():
            sub = gdf[gdf["window"] == win]
            verdict = "全部通过 → 可改定版" if sub["passed"].all() else "未通过 → 维持定版"
            log.info("门槛[%s]：%s", win, verdict)
            for _, r in sub.iterrows():
                log.info("  %-12s %s | %s", r["gate"],
                         "PASS" if r["passed"] else "FAIL", r["detail"])

    cols = ["arm", "wrapper", "oos_ic", "annual", "sharpe", "max_dd", "turnover",
            "annual_2019", "sharpe_2019"]
    print(table[cols].round(4).to_string(index=False))
    print(f"完成 {time.time() - t0:.0f}s → {dest}")


if __name__ == "__main__":
    main()
