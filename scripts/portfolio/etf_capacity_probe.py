"""
ETF 轮动容量表（P3 组合层前置）
================================

回答一个问题：**当前 champion 策略在多大规模资金下仍然可执行？**

口径（先验固定，跑完不改）:
  - 参与率 = 单只单边下单金额 / 该标的 60 日均成交额。
    上限 5%（``PARTICIPATION``）—— 超过即视为该调仓日不可执行。
  - 下单金额 = |Δw| × A，Δw 是该调仓日相对上一调仓日的目标权重变化；
    首期按建仓（前值 0）计。
  - 防御腿 DEF 是合成资产（0.5·511010.SH + 0.5·518880.SH），容量计算时
    按 50/50 拆回两只真实 ETF，与 ``build_defense`` 的构造一致。
  - 成交额用 PIT 60 日均值（``amount.rolling(60, min_periods=40).mean()``），
    与回测掩码同口径。
  - 容量上限 = min_i (amt60_i × 5% / |Δw_i|)；报 5% / 25% / 50% 分位，
    分别代表极端月 / 较紧月 / 常态。

用法:
    python -m scripts.portfolio.etf_capacity_probe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data.cache import DataCache
from data.datasource import create_datasource
from scripts.portfolio.etf_rotation_p2 import (
    P1_BASE,
    build_defense,
    build_extra_panels,
    level_from_returns,
)
from scripts.portfolio.etf_rotation_probe import (  # noqa: E402
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_capacity"
PARTICIPATION = 0.05      # 单日单只成交额占比上限
AMOUNTS = [1e6, 5e6, 1e7, 3e7, 5e7, 1e8, 2e8, 5e8]
DEF_PARTS = {"511010.SH": 0.5, "518880.SH": 0.5}

STRATEGIES: list[dict] = [
    {**P1_BASE, "name": "def_mix_crowd_f20", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "champ_mom_dsh_f67", "signal": "mom_dsh_f67_crowd",
     "defense": "mix"},
]


def delta_table(weights: pd.DataFrame, rb_dates: pd.DatetimeIndex,
                codes: list[str]) -> pd.DataFrame:
    """调仓日 → 各真实标的的 |Δw|（DEF 按 50/50 拆到债/金）。"""
    rows = {}
    prev = pd.Series(0.0, index=weights.columns)
    for t in rb_dates:
        cur = weights.loc[t].reindex(weights.columns).fillna(0.0)
        d = (cur - prev).abs()
        if "DEF" in d.index:
            v = float(d["DEF"])
            d = d.drop("DEF")
            for c, share in DEF_PARTS.items():
                d[c] = d.get(c, 0.0) + share * v
        rows[t] = d.reindex(codes).fillna(0.0)
        prev = cur
    return pd.DataFrame(rows).T.sort_index()


def capacity_report(weights: pd.DataFrame, rb_dates: pd.DatetimeIndex,
                    amt60: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = [c for c in weights.columns if c not in ("CASH", "DEF")]
    delta = delta_table(weights, rb_dates, codes)
    amt = amt60.reindex(index=delta.index, columns=codes)

    summ = []
    for a in AMOUNTS:
        part = (delta * a / amt).where(amt > 0)
        mx = part.max(axis=1)
        summ.append({
            "strategy": name, "capital_wan": a / 1e4,
            "days_over_5pct": float((mx > PARTICIPATION).mean()),
            "median_max_participation": float(mx.median()),
            "p95_max_participation": float(mx.quantile(0.95)),
            "worst_asset": part.idxmax(axis=1).mode().iloc[0] if len(part) else "",
        })
    summary = pd.DataFrame(summ)

    by_asset = []
    for c in codes:
        d = delta[c]
        hit = d[d > 1e-9]
        if not len(hit):
            continue
        by_asset.append({
            "strategy": name, "code": c,
            "n_rebalance_held": int(len(hit)),
            "median_delta_w": float(hit.median()),
            "max_delta_w": float(hit.max()),
            "amt60_median_yi": float(amt[c].median() / 1e8),
            "asset_capacity_wan_at_5pct": float(
                amt[c].median() * PARTICIPATION / hit.median() / 1e4),
        })
    return summary, pd.DataFrame(by_asset).sort_values("asset_capacity_wan_at_5pct")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 轮动容量表")
    ap.add_argument("--begin", type=int, default=20200102)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, args.end)
    close, amount = bundle[1], bundle[2]
    panels = build_extra_panels(close, build_panels(close, amount))
    def_levels = {k: level_from_returns(v) for k, v in build_defense(close).items()}
    amt60 = amount.rolling(60, min_periods=40).mean()

    summaries, assets, caps = [], [], {}
    for spec in STRATEGIES:
        extra = pd.DataFrame({"DEF": def_levels[spec["defense"]]})
        r = run_variant(bundle, panels, spec, args.begin, args.end, extra_close=extra)
        w, rb = r["weights"], r["rebalance_dates"]
        summ, by_asset = capacity_report(w, rb, amt60, spec["name"])
        summaries.append(summ)
        assets.append(by_asset)

        codes = [c for c in w.columns if c not in ("CASH", "DEF")]
        delta = delta_table(w, rb, codes)
        amt = amt60.reindex(index=delta.index, columns=codes)
        caps[spec["name"]] = (amt * PARTICIPATION / delta.where(delta > 1e-9)).min(axis=1)
        print(f"\n[{spec['name']}] 调仓 {len(rb)} 次 · 平均持仓 {r['avg_holdings']:.1f} 只")
        print(summ[["capital_wan", "days_over_5pct", "median_max_participation",
                    "p95_max_participation"]].to_string(index=False,
                                                       float_format=lambda x: f"{x:.3f}"))
        print("  容量分位（万元）：p5={:,.0f} p25={:,.0f} p50={:,.0f}".format(
            *(caps[spec["name"]].quantile(q) for q in (0.05, 0.25, 0.50))), flush=True)

    pd.concat(summaries).to_csv(out / "capacity_summary.csv", index=False,
                                encoding="utf-8-sig")
    pd.concat(assets).to_csv(out / "capacity_by_asset.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({k: v for k, v in caps.items()}).to_csv(
        out / "capacity_by_rebalance.csv", encoding="utf-8-sig")
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
