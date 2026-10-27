"""
ETF 轮动样本外验证（P4，2026-10-26）
====================================

对照 `reports/docs/design/ETF轮动_研究与实施计划_2026-10-26.md` §三 P4：
把「信号 + 组合构造」的定版参数在一段**未单独评估过**的时间上重跑，
并检查多起点稳定性。

窗口设计（先验固定，跑完不改）：

  - **IS 2020-01 → 2024-12**：参数与结构的选择区间（P0–P3 都在此区间内看过结果）；
  - **OOS 2025-01 → 数据末日**：时间切片留出段（P1–P3 的报表里从未单独评估）；
  - **多起点**：2020/2021/2022/2023 各自起跑到末日，检验「收益是否只来自 2020–21」。

判据（先验固定）：
  - OOS Sharpe ≥ 0.5、年化收益 > 0、|MDD| ≤ 35%；
  - 四个起点 Sharpe 全部 ≥ 0.5（不能只靠某一年）；
  - OOS Sharpe 相对 IS 的衰减 ≤ 0.3（参数不是拟合噪声）。

注意口径：候选参数确实在 2020–2026 全样本上被挑选过，因此这里的 OOS
是「时间切片留出」，不等同于完全独立的验证；它的作用是把「只在
2020–21 有效」和「各段都成立」区分开。

用法:
    python -m scripts.portfolio.etf_p4_probe
"""
from __future__ import annotations

import argparse
import json
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
from scripts.portfolio.etf_rotation_probe import (
    BENCH,
    MIN_AMOUNT,
    WARMUP_BEGIN,
    build_panels,
    load_bundle,
    run_variant,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "etf_p4"

IS_BEGIN, IS_END = 20200102, 20241231
OOS_BEGIN = 20250101
STARTS = (2020, 2021, 2022, 2023)

# 定版候选：P1 基线 / P2 champion / S1 邻近点 / P3 两个风险平价版本。
CANDIDATES: list[dict] = [
    {**P1_BASE, "name": "p1_base"},
    {**P1_BASE, "name": "champ", "signal": "mom_crowd_f20", "defense": "mix"},
    {**P1_BASE, "name": "champ_dsh_f67", "signal": "mom_dsh_f67_crowd", "defense": "mix"},
    {**P1_BASE, "name": "champ_erc", "signal": "mom_crowd_f20", "defense": "mix",
     "weighting": "erc"},
    {**P1_BASE, "name": "champ_hrp", "signal": "mom_crowd_f20", "defense": "mix",
     "weighting": "hrp"},
]

KEEP = ("annual_return", "sharpe", "max_drawdown", "excess", "avg_turnover",
        "avg_holdings", "cost_drag")


def _run(bundle, panels, spec: dict, extra, w0: int, w1: int | None) -> dict:
    r = run_variant(bundle, panels, spec, w0, w1, extra_close=extra)
    return {k: r[k] for k in KEEP}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="ETF 轮动 P4 样本外验证")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = DataCache(create_datasource())
    bundle = load_bundle(cache, WARMUP_BEGIN, None)
    close = bundle[1]
    panels = build_extra_panels(close, build_panels(close, bundle[2]))
    def_levels = {k: level_from_returns(v) for k, v in build_defense(close).items()}

    rows, verdict, eq = [], [], {}
    for spec in CANDIDATES:
        name = spec["name"]
        extra = pd.DataFrame({"DEF": def_levels[spec["defense"]]}) if spec.get("defense") else None

        is_r = _run(bundle, panels, spec, extra, IS_BEGIN, IS_END)
        oos_r = _run(bundle, panels, spec, extra, OOS_BEGIN, None)
        start_r = {y: _run(bundle, panels, spec, extra, y * 10000 + 101, None)
                   for y in STARTS}

        rows.append({"candidate": name, "window": "IS_2020_2024", **is_r})
        rows.append({"candidate": name, "window": "OOS_2025_2026", **oos_r})
        for y, r in start_r.items():
            rows.append({"candidate": name, "window": f"start_{y}", **r})

        start_sharpes = [start_r[y]["sharpe"] for y in STARTS]
        decay = is_r["sharpe"] - oos_r["sharpe"]
        oos_pass = (oos_r["sharpe"] >= 0.5 and oos_r["annual_return"] > 0
                    and abs(oos_r["max_drawdown"]) <= 0.35)
        starts_pass = min(start_sharpes) >= 0.5
        decay_pass = decay <= 0.3
        verdict.append({
            "candidate": name,
            "is_sharpe": is_r["sharpe"], "oos_sharpe": oos_r["sharpe"],
            "oos_annual_return": oos_r["annual_return"],
            "oos_max_drawdown": oos_r["max_drawdown"],
            "oos_sharpe_decay": decay,
            "starts_min_sharpe": min(start_sharpes),
            "start_sharpes": ";".join(f"{s:.2f}" for s in start_sharpes),
            "oos_pass": bool(oos_pass), "starts_pass": bool(starts_pass),
            "decay_pass": bool(decay_pass),
            "pass": bool(oos_pass and starts_pass and decay_pass),
        })

        eq[name] = run_variant(bundle, panels, spec, IS_BEGIN, None,
                               extra_close=extra)["equity"]
        print(f"{name:<16} IS sh={is_r['sharpe']:.2f} ann={is_r['annual_return']:+.2%} | "
              f"OOS sh={oos_r['sharpe']:.2f} ann={oos_r['annual_return']:+.2%} "
              f"mdd={oos_r['max_drawdown']:.2%} | starts="
              f"{'/'.join(f'{s:.2f}' for s in start_sharpes)} "
              f"{'PASS' if verdict[-1]['pass'] else ''}", flush=True)

    df = pd.DataFrame(rows)
    vd = pd.DataFrame(verdict).sort_values("oos_sharpe", ascending=False)
    df.to_csv(out / "windows.csv", index=False, encoding="utf-8-sig")
    vd.to_csv(out / "verdict.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eq).to_csv(out / "equity.csv", encoding="utf-8-sig")
    (out / "run_config.json").write_text(json.dumps({
        "is_window": [IS_BEGIN, IS_END], "oos_window": [OOS_BEGIN, None],
        "starts": STARTS, "bench": BENCH, "pool_size": len(close.columns),
        "min_amount": MIN_AMOUNT,
        "judge": "OOS Sharpe>=0.5 & ann>0 & |MDD|<=35%; all starts Sharpe>=0.5; "
                 "decay<=0.3",
        "candidates": CANDIDATES,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n[P4 判定]\n" + vd.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
