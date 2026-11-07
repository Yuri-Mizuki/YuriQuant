"""重建前后影响对比：把「重建会改多少」变成可判定的数（2026-11-07）。

背景
----
`year_offset` 前视 bug 修复后要全量重建 965 个面板。重建会改 `panels_neu`，而慢信号
`slow` 直接读它 ⇒ 定版信号 `0.5·rank(h1020) + 0.5·rank(slow)` 会变。问题是**变多少**：
如果组合基本没动，说明这个 bug 对生产几乎无影响（只影响回测里"看起来的"业绩）；
如果持仓大换血，那 16.31%/0.738 这个定版读数就得重新标定。

三道口径都在这里，别另写：
* 快信号与执行切分走 `blend_weight_920.build_ctx()`（T+1 open、buffer 15/40、月频）；
* 慢信号走同一份 `build_slow_panel`（24 月 ICIR、embargo 2 月、min_cov 0.5）；
* 成本走 `default_costs()`。

**预注册门槛**（跑之前写死，见 :data:`GATES`）：重建后若要直接切换生产，必须
持仓重合率 ≥0.90、日收益相关 ≥0.99、Sharpe 不降超 0.05、年化不降超 2pp。
四门全过 = 切换无争议；任一不过 = 人工看逐月重合率，不看总量。

用法::

    python -m scripts.evaluation.rebuild_impact_920 --snapshot    # 重建前，存基线
    python -m scripts.evaluation.rebuild_impact_920 --compare     # 重建后，出对比
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("rebuild_impact_920")

#: 定版口径（与 `baselines.yaml` 的 `v1-dingban` 一致）。
LAM = 0.5
ENTRY, EXIT_FRAC = 0.15, 0.40

#: 预注册门槛：四项全过才算「重建可无争议切换」。
GATES = {
    "holdings_overlap_med": (">=", 0.90, "持仓重合率中位数"),
    "daily_corr": (">=", 0.99, "日收益相关"),
    "sharpe_delta": (">=", -0.05, "Sharpe 变化"),
    "annual_delta_pp": (">=", -2.00, "年化变化（pp）"),
}


def _snapshot_dir(root: Path) -> Path:
    return root / "before"


def _collect(pred_dir: Path) -> dict:
    """跑一次 λ=0.5 定版臂，返回指标 + 持仓 + 慢信号面板。"""
    from scripts.evaluation.blend_weight_920 import build_ctx, run_arms

    ctx = build_ctx(pred_dir)
    table, daily, held = run_arms(ctx, [LAM], entry=ENTRY, exit_frac=EXIT_FRAC,
                                  keep_weights=True)
    row = table.iloc[0].to_dict()
    w = held[LAM].copy()
    w.index = pd.to_datetime(w.index)
    long = (w.stack().loc[lambda s: s > 0].rename("weight").reset_index()
            .rename(columns={"level_0": "date", "level_1": "code"}))
    return {"ctx": ctx, "metrics": row, "daily": daily[LAM], "holdings": long,
            "slow": ctx["slow"], "slow_weights": ctx["weights"]}


def do_snapshot(args) -> None:
    dest = Path(args.out) / "before"
    dest.mkdir(parents=True, exist_ok=True)
    got = _collect(Path(args.pred_dir))

    pd.DataFrame([got["metrics"]]).to_csv(dest / "metrics.csv", index=False,
                                          encoding="utf-8-sig")
    got["daily"].rename("ret").to_frame().to_parquet(dest / "daily_returns.parquet")
    got["holdings"].to_parquet(dest / "holdings.parquet", index=False)
    got["slow"].astype("float32").to_parquet(dest / "slow.parquet")
    got["slow_weights"].to_parquet(dest / "slow_weights.parquet")
    (dest / "meta.json").write_text(json.dumps(
        {"lam": LAM, "entry": ENTRY, "exit_frac": EXIT_FRAC,
         "pred_dir": str(args.pred_dir), "gates": {k: [v[0], v[1]] for k, v in GATES.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    m = got["metrics"]
    log.info("基线已存：%s · 年化 %.2f%% Sharpe %.3f 回撤 %.1f%% 换手 %.1f%% "
             "持仓 %d 只/%d 期", dest, m["annual"] * 100, m["sharpe"],
             m["max_dd"] * 100, m["turnover"] * 100,
             got["holdings"].groupby("date").size().iloc[0], got["holdings"]["date"].nunique())


def _holdings_overlap(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """逐调仓日的持仓重合率（Jaccard 与 top-N 交集两种口径都给）。"""
    rows = []
    for d, g in old.groupby("date"):
        a = set(g["code"])
        gn = new[new["date"] == d]
        b = set(gn["code"]) if len(gn) else set()
        if not a or not b:
            continue
        rows.append({"date": d, "n_old": len(a), "n_new": len(b),
                     "common": len(a & b),
                     "overlap_min": len(a & b) / min(len(a), len(b)),
                     "jaccard": len(a & b) / len(a | b)})
    return pd.DataFrame(rows)


def _slow_rank_corr(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """逐月看慢信号的截面秩相关：同一批股票被重排了多少。"""
    idx = old.index.intersection(new.index)
    rows = []
    for d in idx[::21]:                       # 每 21 个交易日取一月，控制开销
        a, b = old.loc[d], new.loc[d]
        ok = a.notna() & b.notna()
        if ok.sum() > 100:
            rows.append({"date": d, "n": int(ok.sum()),
                         "rank_corr": float(a[ok].corr(b[ok], method="spearman"))})
    return pd.DataFrame(rows)


def do_compare(args) -> None:
    base = _snapshot_dir(Path(args.out))
    if not (base / "metrics.csv").exists():
        raise SystemExit(f"{base} 下没有基线 ⇒ 先跑 --snapshot（重建前）")
    old = {
        "metrics": pd.read_csv(base / "metrics.csv").iloc[0].to_dict(),
        "daily": pd.read_parquet(base / "daily_returns.parquet")["ret"],
        "holdings": pd.read_parquet(base / "holdings.parquet"),
        "slow": pd.read_parquet(base / "slow.parquet"),
    }
    got = _collect(Path(args.pred_dir))
    new = {"metrics": got["metrics"], "daily": got["daily"],
           "holdings": got["holdings"], "slow": got["slow"]}

    dest = Path(args.out)
    ov = _holdings_overlap(old["holdings"], new["holdings"])
    ov.to_csv(dest / "holdings_overlap.csv", index=False, encoding="utf-8-sig")
    sc = _slow_rank_corr(old["slow"], new["slow"])
    sc.to_csv(dest / "slow_rank_corr.csv", index=False, encoding="utf-8-sig")

    joined = pd.concat([old["daily"].rename("old"), new["daily"].rename("new")],
                       axis=1).dropna()
    daily_corr = float(joined.corr().iloc[0, 1])

    from backtest.metrics import sharpe_ratio
    diff = (joined["new"] - joined["old"]).dropna()
    t_stat = float(diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))) if diff.std() > 0 else 0.0
    metrics = {
        "annual_old": float(old["metrics"]["annual"]),
        "annual_new": float(new["metrics"]["annual"]),
        "annual_delta_pp": (float(new["metrics"]["annual"]) - float(old["metrics"]["annual"])) * 100,
        "sharpe_old": float(old["metrics"]["sharpe"]),
        "sharpe_new": float(new["metrics"]["sharpe"]),
        "sharpe_delta": float(new["metrics"]["sharpe"]) - float(old["metrics"]["sharpe"]),
        "max_dd_old": float(old["metrics"]["max_dd"]),
        "max_dd_new": float(new["metrics"]["max_dd"]),
        "turnover_old": float(old["metrics"]["turnover"]),
        "turnover_new": float(new["metrics"]["turnover"]),
        "daily_corr": daily_corr,
        "daily_diff_t": t_stat,
        "holdings_overlap_med": float(ov["overlap_min"].median()) if len(ov) else float("nan"),
        "holdings_overlap_min": float(ov["overlap_min"].min()) if len(ov) else float("nan"),
        "slow_rank_corr_med": float(sc["rank_corr"].median()) if len(sc) else float("nan"),
        "n_rebalances": int(len(ov)),
    }

    rows = []
    for key, (op, thr, label) in GATES.items():
        val = metrics[key]
        ok = (val >= thr) if op == ">=" else (val <= thr)
        rows.append({"gate": key, "label": label, "op": op, "threshold": thr,
                     "value": round(val, 4), "pass": bool(ok)})
    gate_tab = pd.DataFrame(rows)
    gate_tab.to_csv(dest / "gate_check.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(dest / "summary.csv", index=False, encoding="utf-8-sig")

    print()
    print("== 重建前后（λ=0.5 定版口径）")
    print(f"   年化   {metrics['annual_old']*100:7.2f}% → {metrics['annual_new']*100:7.2f}%"
          f"  ({metrics['annual_delta_pp']:+.2f}pp)")
    print(f"   Sharpe {metrics['sharpe_old']:7.3f}  → {metrics['sharpe_new']:7.3f}"
          f"  ({metrics['sharpe_delta']:+.3f})")
    print(f"   回撤   {metrics['max_dd_old']*100:7.2f}% → {metrics['max_dd_new']*100:7.2f}%")
    print(f"   换手   {metrics['turnover_old']*100:7.2f}% → {metrics['turnover_new']*100:7.2f}%")
    print(f"   日收益相关 {metrics['daily_corr']:.4f}（差收益 t {metrics['daily_diff_t']:.2f}）")
    print(f"   持仓重合率 中位 {metrics['holdings_overlap_med']:.3f} / 最低 "
          f"{metrics['holdings_overlap_min']:.3f}（{metrics['n_rebalances']} 个调仓日）")
    print(f"   慢信号秩相关 中位 {metrics['slow_rank_corr_med']:.3f}")
    print("\n-- 预注册门槛 --")
    print(gate_tab[["label", "threshold", "value", "pass"]].to_string(index=False))
    if gate_tab["pass"].all():
        print("\n✅ 四门全过 ⇒ 重建可直接切换生产")
    else:
        bad = gate_tab[~gate_tab["pass"]]["label"].tolist()
        print(f"\n❌ 未过：{'、'.join(bad)} ⇒ 看 holdings_overlap.csv 逐月，别只看总量")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="重建前后影响对比（λ=0.5）")
    ap.add_argument("--pred-dir", default=str(ROOT / "reports" / "alla_rolling_ortho920fundind"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "rebuild_impact_920"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", action="store_true", help="重建前存基线")
    g.add_argument("--compare", action="store_true", help="重建后出对比")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    if args.snapshot:
        do_snapshot(args)
    else:
        do_compare(args)


if __name__ == "__main__":
    main()
