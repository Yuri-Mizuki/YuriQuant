"""入库后补齐「三件套」：中性化面板 + ic_h 列 + registry 统计。

存在理由
--------
库里有两条入库通道，落盘的产物**不是同一套**：

======================  ==================  ==========  =========  =========
通道                    面板                evals       中性化面板  ic_h 列
======================  ==================  ==========  =========  =========
builder（merge_outputs） panels/<name>      无          ✅          ✅
FactorLibrary.register   panels/<slug>      ✅          无列       无列
======================  ==================  ==========  =========  =========

于是走库入口登记的因子看着"已入库"（registry 有行、evals 有值），却**永远
进不了定版链**——定版读的是 ``panels_neu_fundind`` 与 ``ic_h{h}.parquet``。
2026-11-12 的 12 个 ``pap_*`` 就是这么卡住的（也是那次 2025 单年事故暴露得
比较晚的原因：没人按定版口径去查它们）。

本脚本把缺口补齐，顺序不能换：

1. **registry.coverage** —— 中性化与 ic 融合都拿它当门槛（MIN_*_COVERAGE），
   必须先有值，否则因子被静默跳过；
2. **中性化** —— ``panels_neu_fundind/<name>.parquet``（行业+市值，E5 口径）；
3. **ic 融合** —— 逐 horizon 重算并写入 ``ic_h{h}.parquet``；
4. **ic_mean_h*** —— 与 821 个 builder 因子同口径：直接取 ic 列均值
   （已核对：registry 现值与 ic 列均值逐位一致）。

用法
----
    python -m scripts.factors.sync_factor_artifacts --names pap_value_bp,pap_rev_5d
    python -m scripts.factors.sync_factor_artifacts --since 2026-11-12   # 当天新登记
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
from scripts.builders.common import (  # noqa: E402
    HORIZONS,
    fuse_horizon_ic,
    resolve_panel_file,
    run_neutralization,
    setup_logging,
)
from scripts.common.alla_common import layer_panel_file  # noqa: E402

log = setup_logging("sync_factor_artifacts")

NEU_SUBDIR = "panels_neu_fundind"   # E5 覆盖层（只存与基准层不同的因子）
NEU_BASE_SUBDIR = "panels_neu"      # 基准层（定版 focus + 慢信号源）


def _coverage(panel: pd.DataFrame) -> float:
    """与 builder 完全一致的口径：全历史 × 全代码格点非空率。"""
    return float(panel.notna().mean().mean())


def main() -> int:
    ap = argparse.ArgumentParser(description="补齐中性化面板 + ic 列 + registry 统计")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--names", default=None, help="逗号分隔因子名")
    ap.add_argument("--since", default=None,
                    help="按 registry.created_at >= 该日期挑因子（YYYY-MM-DD）")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fundind-names", default=None,
                    help="这批里按 E5「只剥行业」口径中性化的因子（基本面/价值族："
                         "市值中性化会把 bp/规模风格本身剥掉），逗号分隔")
    ap.add_argument("--skip-neu", action="store_true")
    ap.add_argument("--skip-ic", action="store_true")
    args = ap.parse_args()

    lib = FactorLibrary(dataset=args.dataset)
    ds = Path(lib.root)
    reg = lib._load_registry()

    if args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
    elif args.since:
        ts = pd.to_datetime(args.since)
        names = reg.loc[pd.to_datetime(reg["created_at"], errors="coerce") >= ts, "name"].tolist()
    else:
        ap.error("需要 --names 或 --since")
    unknown = [n for n in names if n not in set(reg["name"])]
    if unknown:
        log.error("registry 里没有这些因子：%s", unknown)
        return 2
    log.info("待补齐 %d 个：%s", len(names), ", ".join(names))

    # ---- 1. coverage（中性化/ic 融合的门槛输入）--------------------------
    patch = []
    for n in names:
        pf = resolve_panel_file(ds, n)
        if pf is None:
            log.error("面板缺失，跳过 %s", n)
            continue
        cov = _coverage(pd.read_parquet(pf))
        patch.append({"name": n, "coverage": cov})
        log.info("  coverage %-22s %.4f (%s)", n, cov, pf.name)
    if patch:
        lib.upsert_rows(patch, fill_missing_only=True)
        log.info("registry.coverage 已刷新 %d 行", len(patch))

    # ---- 2. 中性化 ------------------------------------------------------
    if not args.skip_neu:
        fin = {n.strip() for n in (args.fundind_names or "").split(",") if n.strip()}
        if fin:
            log.info("只剥行业（E5 口径）：%s", ", ".join(sorted(fin)))
        # 基准层必须先有：2026-11-18 起覆盖层只存「与基准层内容不同」的因子，
        # 只写覆盖层会让基准层缺该因子，其它读 panels_neu 的路径全部落空。
        n_ok = run_neutralization(ds, NEU_BASE_SUBDIR, workers=args.workers,
                                  only=names, force=True, set_name="paper",
                                  label="中性化(paper 补齐·基准层)")
        log.info("中性化完成 %d 个 -> %s", n_ok, NEU_BASE_SUBDIR)
        if fin:
            n_ok = run_neutralization(ds, NEU_SUBDIR, workers=args.workers,
                                      only=sorted(fin), force=True,
                                      set_name="paper",
                                      label="中性化(paper 补齐·E5 覆盖层)",
                                      industry_only=frozenset(fin))
            log.info("E5 覆盖层完成 %d 个 -> %s", n_ok, NEU_SUBDIR)
        neu_patch = []
        for n in names:
            f = layer_panel_file(n, NEU_SUBDIR, ds)
            if f.exists():
                neu_patch.append({"name": n,
                                  "neutralized_coverage": _coverage(pd.read_parquet(f))})
        if neu_patch:
            lib.upsert_rows(neu_patch, fill_missing_only=True)
            log.info("registry.neutralized_coverage 已刷新 %d 行", len(neu_patch))

    # ---- 3. ic 融合 -----------------------------------------------------
    if not args.skip_ic:
        # close_adj 只建一次：4 个 horizon 各建一遍等于白读 4 次日线
        from scripts.builders.common import load_close_adj
        _ca = load_close_adj()
        for h in HORIZONS:
            fuse_horizon_ic(ds, h, "paper", names=names,
                            close_adj_loader=lambda: _ca)

    # ---- 4. ic_mean_h* = ic 列均值（与 builder 因子同口径）--------------
    patch = []
    for h in HORIZONS:
        ic = pd.read_parquet(ds / f"ic_h{h}.parquet", columns=None)
        miss = [n for n in names if n not in ic.columns]
        if miss:
            log.warning("ic_h%d 仍缺 %d 列：%s", h, len(miss), ", ".join(miss))
        for n in names:
            if n in ic.columns:
                patch.append({"name": n, f"ic_mean_h{h}": float(ic[n].mean())})
    if patch:
        merged = pd.DataFrame(patch).groupby("name", as_index=False).first()
        lib.upsert_rows(merged.to_dict("records"), fill_missing_only=True)
        log.info("registry.ic_mean_h* 已刷新 %d 行", len(merged))

    # ---- 5. 核验 --------------------------------------------------------
    reg = lib._load_registry().set_index("name")
    ic20 = pd.read_parquet(ds / "ic_h20.parquet")
    print(f"\n{'name':24s} {'cov':>6s} {'neucov':>7s} {'ic20':>8s} {'ic_h20':>8s} {'neu':>4s}")
    for n in names:
        cov = reg.at[n, "coverage"] if "coverage" in reg.columns else float("nan")
        nc = reg.at[n, "neutralized_coverage"] if "neutralized_coverage" in reg else float("nan")
        icm = reg.at[n, "ic_mean_h20"] if "ic_mean_h20" in reg.columns else float("nan")
        has_neu = (ds / NEU_SUBDIR / f"{n}.parquet").exists()
        has_ic = n in ic20.columns
        print(f"{n:24s} {float(cov):6.3f} {float(nc):7.3f} "
              f"{float(icm):8.4f} {'✓' if has_ic else '✗':>8s} {'✓' if has_neu else '✗':>4s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
