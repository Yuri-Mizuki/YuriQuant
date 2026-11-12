"""按**定版回测窗口**核对因子库每个成员的面板覆盖（只读）。

存在理由
--------
`registry.coverage` 是「全历史 × 全代码格点非空率」，回答不了「在定版窗口里够
不够用」：分母含未上市代码，一个正常因子也会显示 0.70。真正要查的是两件事——

1. **日期覆盖**：面板索引是否覆盖定版窗口的每一个交易日（缺头/缺尾/只覆盖一小段）；
2. **有效值覆盖**：在**当日有行情的股票**里有多少非空（排除「索引全、值全空」）。

2026-11-12 的 12 个 ``pap_*`` 就是第 1 项查出来的：面板只有 2025 一年，却登记在
全A 库里，``registry`` 上看着完全正常。第 2 项则识别出 ``pap_ann_reversal`` 这类
事件型因子（索引全 2514 天，但只有 1209 天有值、每日中位 16 只）。

用法
----
    python -m scripts.factors.audit_panel_window                       # 全库
    python -m scripts.factors.audit_panel_window --window 20190101 20261109
    python -m scripts.factors.audit_panel_window --names pap_value_bp,pap_rev_5d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.factor_library import FactorLibrary  # noqa: E402
from scripts.builders.common import resolve_panel_file  # noqa: E402

OUT_CSV = ROOT / "reports" / "library_governance" / "panel_window_coverage.csv"


def _market_panels(ds: Path):
    """库统一交易日历 + 原始收盘面板（有效覆盖的分母来自后者）。

    用 builder 同一个入口 ``load_close_adj(with_raw=True)``，保证日历与
    ``KEEP_FROM`` 截断口径和定版链完全一致（各自另读 parquet 迟早会漂）。
    """
    from scripts.builders.common import load_close_adj

    return load_close_adj(with_raw=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="按定版窗口核对面板覆盖")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--window", nargs=2, default=["20190101", "20261109"],
                    metavar=("BEGIN", "END"))
    ap.add_argument("--names", default=None, help="逗号分隔；缺省=全库")
    ap.add_argument("--probe-dates", default="20190628,20230630,20251128",
                    help="有效覆盖抽样日（YYYYMMDD，逗号分隔）")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    ds = Path(lib.root)
    reg = lib._load_registry()

    names = ([n.strip() for n in args.names.split(",") if n.strip()]
             if args.names else list(reg["name"]))
    close_adj, close_raw = _market_panels(ds)
    tdays = pd.DatetimeIndex(close_adj.index)
    win = tdays[(tdays >= pd.Timestamp(args.window[0])) &
                (tdays <= pd.Timestamp(args.window[1]))]
    probes = [pd.Timestamp(x.strip()) for x in args.probe_dates.split(",") if x.strip()]
    listed = {dt: set(close_raw.loc[dt].dropna().index) for dt in probes
              if dt in close_raw.index}

    set_of = reg.drop_duplicates("name").set_index("name")["set"].to_dict()
    rows = []
    for n in names:
        pf = resolve_panel_file(ds, n)
        rec = {"name": n, "set": set_of.get(n, ""), "exists": pf is not None,
               "panel_file": pf.name if pf else ""}
        if pf is None:
            rows.append(rec)
            continue
        idx = pd.DatetimeIndex(pd.read_parquet(pf, columns=[]).index)
        rec.update({
            "first": idx.min().date(), "last": idx.max().date(),
            "n_dates": len(idx),
            "n_in_win": int(((idx >= win[0]) & (idx <= win[-1])).sum()),
            "frac_in_win": round(int(((idx >= win[0]) & (idx <= win[-1])).sum())
                                 / max(1, len(win)), 4),
        })
        # 有效值覆盖在第二轮读实际列（全量读 800+ 面板太慢，且要现算分母）
        rows.append(rec)

    # 第二轮：有效覆盖（需要列，单独一遍以免把全量面板一直留在内存）
    probes_in = [dt for dt in probes if dt in tdays]
    for rec in rows:
        if not rec.get("exists"):
            continue
        pf = resolve_panel_file(ds, rec["name"])
        panel = pd.read_parquet(pf)
        vals = []
        for dt in probes_in:
            if dt not in panel.index:
                vals.append(None)
                continue
            codes = [c for c in panel.columns if c in listed.get(dt, set())]
            vals.append(round(float(panel.loc[dt, codes].notna().mean()), 4)
                        if codes else None)
        rec["eff_cov"] = "/".join("NA" if v is None else f"{v:.2f}" for v in vals)
        nn = panel.notna().sum(axis=1)
        rec["n_days_nonempty"] = int((nn > 0).sum())
        rec["median_per_day"] = int(nn[nn > 0].median()) if (nn > 0).any() else 0
        del panel

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    short = out[out["frac_in_win"] < 0.99]
    print(f"窗口 {win[0].date()}~{win[-1].date()}（{len(win)} 个交易日）")
    print(f"全库 {len(out)} 个；日期覆盖 <99% 的 {len(short)} 个")
    if len(short):
        print(short[["name", "first", "last", "n_dates", "frac_in_win"]].to_string(index=False))
    ev = out[out["n_days_nonempty"].notna() &
             (out["n_days_nonempty"] < len(win) * 0.5)] if "n_days_nonempty" in out else out.iloc[:0]
    if len(ev):
        print(f"\n有效天数 <50% 窗口的 {len(ev)} 个（事件型因子属正常）：")
        print(ev[["name", "n_days_nonempty", "median_per_day"]].to_string(index=False))
    print(f"\n→ {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())