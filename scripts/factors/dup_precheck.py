"""**登记前**冗余预检：这批因子该不该进库，先看一眼它最像谁。

为什么要有这个入口
------------------
`FactorLibrary.ingest(check_dup=True)` 是"边登记边查"——查完行已经写进
registry 了。而新因子的主要来源（builder → `merge_outputs` → `upsert_rows`）
走的是**轻量登记**通道，连面板都不读，天然没有查重。所以"入库前能不能查"
这件事此前没有入口，答案是不能。

本脚本补这个缺口。它只读：不写 registry、不动面板、不改索引。

耗时（2026-11-12，all_a 950 因子库实测）
  - 与全库相关最低 (<0.85) 的因子：**0.15~0.41 s**（只做一次签名矩阵乘）
  - 疑似重复（ρ≥0.85）的因子：**8~13 s**（额外做全样本逐日截面残差回归）
残差只在"疑似重复"时才值得算——ρ 低的因子不存在"增量够不够"的问题。

    # 单个已在库的因子（排除自身后看它最像谁）
    python -m scripts.factors.dup_precheck --name alpha158_SUMP10

    # 新因子：面板还没入库
    python -m scripts.factors.dup_precheck --panel path/to/new.parquet --name my_new

    # 批量（每行一条），有重复时退出码 2，便于卡在构建脚本里
    python -m scripts.factors.dup_precheck --names a,b,c --reject-rho 0.95

判据边界（引用必须带上）
  - ρ 是抽样日历上的**逐日截面秩相关**均值，只说明"因子值像不像"，
    不说明"组合层会不会重复"。要判后者得跑组合，不是这里能回答的。
  - 判重看 `|ρ|`：反向完全重复（ρ=−1）同样是重复。`corr` 列保留符号。
  - `resid_t ≥ 2` 表示对最近邻仍有独立增量——只看 ρ 会把这类因子误杀
    （实测 `VSUMN30` vs `VSUMD30` ρ=1.000 但残差 t=25.4）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.factor_library import FactorLibrary  # noqa: E402

OUT_DEFAULT = Path("reports/library_governance")


def main() -> None:
    ap = argparse.ArgumentParser(description="登记前冗余预检（只读）")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--name", default="", help="已在库的因子名")
    ap.add_argument("--panel", default="", help="未入库因子的面板 parquet")
    ap.add_argument("--names", default="", help="逗号分隔的多个因子名")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--reject-rho", type=float, default=0.95,
                    help="|ρ|≥此值且残差无增量 ⇒ 退出码 2")
    ap.add_argument("--csv", default="", help="写出明细 CSV")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    if lib.dup_index() is None:
        raise SystemExit("冗余索引不存在：先跑 scripts.factors.dup_audit "
                         "--dataset <ds>（索引随库重建会丢，需重建）")

    jobs: list[tuple[str, pd.DataFrame]] = []
    if args.panel:
        p = Path(args.panel)
        if not p.exists():
            raise SystemExit(f"面板不存在：{p}")
        jobs.append((args.name or p.stem, pd.read_parquet(p)))
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if args.name and not args.panel:
        names.append(args.name)
    for n in names:
        panel = lib.get_panel(n)
        if panel is None:
            print(f"!! 找不到 {n} 的面板（未登记？用 --panel 指定路径）")
            continue
        jobs.append((n, panel))
    if not jobs:
        raise SystemExit("没有可检查的目标：给 --name / --names / --panel")

    rows = []
    t_all = time.time()
    for i, (nm, panel) in enumerate(jobs, 1):
        t0 = time.time()
        res = lib.precheck_dup(panel, name=nm, topk=args.topk)
        dt = time.time() - t0
        row = {
            "name": nm, "secs": round(dt, 2),
            "verdict": res.get("verdict"), "top_corr": res.get("top_corr"),
            "top_name": res.get("top_name"), "resid_ic": res.get("resid_ic"),
            "resid_t": res.get("resid_t"),
            "top": " | ".join(f"{t['name']}({t['corr']:+.3f})" for t in res.get("top", [])),
        }
        rows.append(row)
        print(f"[{i}/{len(jobs)}] {nm:26s} {dt:5.2f}s  {row['verdict']:14s} "
              f"|ρ|={row['top_corr']:.3f} ← {row['top_name']}"
              + (f"  resid_t={row['resid_t']:+.2f}" if pd.notna(row["resid_t"]) else ""))
        print(f"      top{args.topk}: {row['top']}")

    df = pd.DataFrame(rows)
    # 判「独立」的行不跑残差检验 ⇒ resid_t 为 None ⇒ 整列被推断成 object，
    # `.abs()` 直接 TypeError（2026-11-13 实测：单面板独立判定必崩）。
    # 这里强制转数值再比，None ⇒ NaN ⇒ fillna(0) ⇒ 视为「无增量」只在 ρ 达标时才生效。
    resid_t_num = pd.to_numeric(df["resid_t"], errors="coerce")
    bad = df[(df["top_corr"] >= args.reject_rho)
             & (resid_t_num.abs().fillna(0) < 2.0)]
    print(f"\n合计 {len(df)} 个 | 用时 {time.time() - t_all:.1f}s | "
          f"|ρ|≥{args.reject_rho} 且残差无增量：{len(bad)} 个")
    if len(bad):
        print("判为重复（建议不进库，或先进 factor_pool 评审）：")
        print(bad[["name", "top_corr", "top_name", "resid_t"]]
              .to_string(index=False))

    if args.csv:
        out = Path(args.csv)
    elif len(df) > 1:
        OUT_DEFAULT.mkdir(parents=True, exist_ok=True)
        out = OUT_DEFAULT / f"dup_precheck_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    else:
        out = None
    if out is not None:
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"明细 → {out}")

    raise SystemExit(2 if len(bad) else 0)


if __name__ == "__main__":
    main()
