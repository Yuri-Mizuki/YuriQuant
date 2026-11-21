"""因子库指标批量回填：canonical 分层回测 + 显著性 + 路径/公式/族标签（维护工具）。

背景（2026-11-05 体检）：all_a_2018_2026 库 950 因子由 build_alla_alpha_panels 等
批量通道经 ``upsert_rows`` 入库，只写了 name/set/label/coverage/ic_mean_h*；
``register()`` 正常入库会算的 canonical 回测（ls_M/lo_M/ls_W）、IC 显著性
（NW-t / |t_nw|>2）、panel_path/eval_path/formula 全部缺位（0% 填充）。
本脚本对**整库**补齐这些列。2026-11-19 起回测不再自造实现，直接调用
``FactorLibrary.evaluate(..., persist=False)``——与单因子入库通道共用同一份
代码（可交易掩码、三套 canonical 回测、eval 文件、eval_status/口径标记），
本脚本只负责批量调度与落盘节奏，两条通道的评估口径从此不会分家：

- 收益面板：``lib.default_returns_panel()``（复权收盘 1 日远期），与
  register/ingest 同源；
- IC 显著性：**不重算 IC**——直接把既有 ``ic_h1.parquet`` 注入
  ``evaluate(ic_series=...)``，省 3 s/因子（register 同判据：NW-t、|t|>2）；
- 超额年化 / 换手序列：由 ``evaluate`` 的 ``_canonical_eval`` 统一产出
  （仅 lo_* 算超额，基准 = 全A等权；换手序列照旧落 evals 供看板切片）；
- ic_decay5 / autocorr：证据列、不参与任何门槛，默认关（``--with-evidence``
  开，每因子约 30 s）；
- formula := label（hs300 库先例：alpha 系 formula 即引用式标签，label 已 100% 填充）；
- family 旧码迁移：rnd/lev/liq/pro/qual/cfq 共 17 行 → family=基本面，
  细分进 subfamily（受控词表无这些风格细分，不新造值；builder 源码
  build_alla_style_factors.FACTOR_DEFS 确认语义，非猜测）；
- panel_path / eval_path / n_dates / n_codes / best_sharpe / best_config 回填。

写盘纪律：
- registry.csv 先 cp 快照（.bak-<ts>），再经 ``lib._save_registry``（列对齐真源）写回；
- 每 --flush-every 因子落一次盘，中断可 --resume 续跑（跳过已填行）；
- evals/<slug>.parquet 落 dret_/equity_/turnover_ × {ls_M, lo_M, ls_W}
  （换手序列供看板按时间区间重算区间换手）。

用法:
    python -m scripts.factors.backfill_library_metrics --dataset all_a_2018_2026
    python -m scripts.factors.backfill_library_metrics --limit 5          # 冒烟
    python -m scripts.factors.backfill_library_metrics --resume           # 断点续跑
    python -m scripts.factors.backfill_library_metrics --skip-backtest    # 只回填 IC 显著性/路径/公式
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.common.cli_common import setup_logging  # noqa: E402

log = setup_logging("backfill_library_metrics")

#: family 旧码 → (受控词表值, subfamily 细分)。语义源 =
#: scripts/builders/build_alla_style_factors.py FACTOR_DEFS（逐因子确认过）。
FAMILY_MIGRATE: dict[str, tuple[str, str]] = {
    "rnd": ("基本面", "研发费用"),
    "lev": ("基本面", "杠杆"),
    "liq": ("基本面", "偿债能力"),
    "pro": ("基本面", "盈利能力"),
    "qual": ("基本面", "资产质量"),
    "cfq": ("基本面", "现金流质量"),
}


def slug(name: str) -> str:
    """与 research.factor_library._slug 同规则（md5[:12]），避免私有依赖。"""
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def resolve_panel(panels_dir: Path, name: str) -> Path | None:
    """alpha 批量按因子名存盘；走 register() 通道的按 slug 存盘——两种都试。"""
    for cand in (panels_dir / f"{name}.parquet", panels_dir / f"{slug(name)}.parquet"):
        if cand.exists():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="因子库指标批量回填")
    ap.add_argument("--dataset", default="all_a_2018_2026")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个（冒烟）")
    ap.add_argument("--force", action="store_true", help="已填行也重算")
    ap.add_argument("--resume", action="store_true", help="跳过已填行（默认行为）")
    ap.add_argument("--skip-backtest", action="store_true",
                    help="只回填 IC 显著性/panel_path/formula/family，不跑回测")
    ap.add_argument("--with-evidence", action="store_true",
                    help="同时补 ic_decay5 / autocorr（每因子约 30 s，批量回填默认关）")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--stale-only", action="store_true",
                    help="已评估行按水位重评：仅面板 mtime 晚于 evaluated_at 的"
                         "（面板延长/重建后的重评口径；随 run_factor_pipeline 接线）")
    args = ap.parse_args()

    from research.factor_analysis import calc_ir, ic_stats
    from research.factor_library import FactorLibrary, name_slug
    from stats.significance import mean_inference

    lib = FactorLibrary(dataset=args.dataset)
    ds_dir = Path(lib.root)          # 已含 <dataset>/ 段（2026-11-19 修双重拼接）
    reg = pd.read_csv(ds_dir / "registry.csv")
    reg = reg.drop_duplicates(subset="name", keep="last").reset_index(drop=True)

    # 备份（写盘纪律）
    bak = ds_dir / f"registry.csv.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(ds_dir / "registry.csv", bak)
    log.info("registry 快照: %s", bak.name)

    # 收益面板：统一用库内默认口径（复权收盘 1 日远期），与 register/ingest 同源
    returns_panel = None
    if not args.skip_backtest:
        t0 = time.time()
        returns_panel = lib.default_returns_panel()
        log.info("收益面板: %d 日 × %d 股（%.0fs）", *returns_panel.shape, time.time() - t0)

    ic1 = pd.read_parquet(ds_dir / "ic_h1.parquet") if (ds_dir / "ic_h1.parquet").exists() else None

    # 超额基准：全A等权日收益（截面均值，NaN=停牌自动剔除）。数据集级一份。
    bench = None
    if returns_panel is not None:
        bench = returns_panel.mean(axis=1, skipna=True)
        bp = ds_dir / "bench_returns.parquet"
        bench.rename("bench_ew").to_frame().to_parquet(bp)
        log.info("基准(全A等权): %d 日 → %s", len(bench), bp.name)

    # 超额列预建（避免逐行 .at 增列的版本差异）
    EXCESS_COLS = ("excess_return_lo_M", "information_ratio_lo_M",
                   "excess_t_stat_lo_M")
    for c in EXCESS_COLS:
        if c not in reg.columns:
            reg[c] = np.nan

    def _panel_mtime(r) -> float:
        """面板文件 mtime（panel_path 优先，退化按 name_slug）；缺失返回 0。"""
        pp = r.get("panel_path")
        fp = (Path(str(pp)) if pd.notna(pp) and str(pp).strip()
              else ds_dir / "panels" / f"{name_slug(str(r['name']))}.parquet")
        try:
            return fp.stat().st_mtime
        except OSError:
            return 0.0

    def _evaluated_at_ts(r) -> float:
        v = r.get("evaluated_at")
        if pd.isna(v) or not str(v).strip():
            return 0.0
        try:
            return pd.Timestamp(str(v)).timestamp()
        except ValueError:
            return 0.0

    todo_idx: list[int] = []
    for i, r in reg.iterrows():
        already = (not args.force and pd.notna(r.get("annual_return_ls_M"))
                   and (ic1 is None or pd.notna(r.get("t_stat_nw"))))
        if already and args.stale_only:
            # 水位比较：面板在评估之后没变 ⇒ 跳过；变过 ⇒ 重评
            if _panel_mtime(r) <= _evaluated_at_ts(r):
                continue
        elif already:
            continue
        if args.skip_backtest and pd.notna(r.get("t_stat_nw")) and \
                pd.notna(r.get("panel_path")):
            continue
        todo_idx.append(i)
    if args.limit:
        todo_idx = todo_idx[: args.limit]
    log.info("待回填 %d / %d 个因子", len(todo_idx), len(reg))

    n_ok = n_missing = 0
    t0 = time.time()
    for k, i in enumerate(todo_idx, 1):
        name = reg.at[i, "name"]
        ppath = resolve_panel(ds_dir / "panels", str(name))
        reg.at[i, "panel_path"] = str(ppath) if ppath else ""
        # ---- formula := label（hs300 先例）----
        # 🚨 这里只是**兜底**：真正能反解析出表达式的走
        # ``scripts/factors/formula_extract.py``（alpha 四族 + builder 标签，覆盖率 100%）。
        # 本处填的 ``formula_src`` 必须标 ``label``——否则 ``formula_extract --apply``
        # 会把「回填前的存量」当手写保护起来，源码改了公式永不刷新
        # （2026-11-13 实测：第二次回填 0 行就是这么来的）。
        if pd.isna(reg.at[i, "formula"]) or not str(reg.at[i, "formula"]).strip():
            lab = reg.at[i, "label"]
            reg.at[i, "formula"] = str(lab) if pd.notna(lab) else str(name)
            if "formula_src" in reg.columns:
                reg.at[i, "formula_src"] = "label"
        # ---- family 旧码迁移 ----
        fam = str(reg.at[i, "family"] or "")
        if fam in FAMILY_MIGRATE:
            new_fam, subfam = FAMILY_MIGRATE[fam]
            reg.at[i, "family"] = new_fam
            if pd.isna(reg.at[i, "subfamily"]) or not str(reg.at[i, "subfamily"]).strip():
                reg.at[i, "subfamily"] = subfam
        # ---- IC 显著性（ic_h1 既有口径）----
        if ic1 is not None and name in ic1.columns:
            ic = ic1[name].dropna()
            if len(ic) >= 2:
                inf = mean_inference(ic)
                reg.at[i, "ic_mean"] = float(ic.mean())
                reg.at[i, "ic_std"] = float(ic.std())
                reg.at[i, "ic_ir"] = float(calc_ir(ic))
                reg.at[i, "t_stat"] = inf["t_stat"]
                reg.at[i, "t_stat_nw"] = inf["t_stat_nw"]
                reg.at[i, "p_value_nw"] = inf["p_value_nw"]
                reg.at[i, "ic_win_rate"] = float(ic_stats(ic)["win"])
                reg.at[i, "significant"] = bool(abs(inf["t_stat_nw"]) > 2.0)
        # ---- canonical 回测 ----
        if not args.skip_backtest:
            if ppath is None:
                n_missing += 1
                log.warning("[%d/%d] %s: 无面板文件，跳过回测", k, len(todo_idx), name)
                continue
            try:
                panel = pd.read_parquet(ppath)
                # 统一走库的 evaluate：可交易掩码、三套 canonical 回测、eval
                # 文件（ic + dret/equity/turnover）、eval_status / 口径标记
                # 全由同一实现产出——本脚本只负责批量调度与落盘节奏。
                # ic 直接用库内预计算的 ic_h1（与 builder IC 同一份），省 3 s/因子。
                ic_s = (ic1[name] if (ic1 is not None and name in ic1.columns)
                        else None)
                metrics = lib.evaluate(
                    name, returns_panel, panel=panel, persist=False,
                    use_mask=True, ic_series=ic_s)
                for key, val in metrics.items():
                    if key not in reg.columns:
                        reg[key] = np.nan
                    reg.at[i, key] = val
                # 证据列不参与任何门槛，且每因子约 30 s，批量默认关
                # （2026-11-19 起；原先无条件计算，876 个因子要 8 小时）。
                if args.with_evidence:
                    from research.factor_analysis import calc_ic_decay, factor_autocorr
                    with np.errstate(all="ignore"):
                        reg.at[i, "ic_decay5"] = float(
                            calc_ic_decay(panel, returns_panel, max_lag=10).get(5, np.nan))
                        reg.at[i, "autocorr"] = float(factor_autocorr(panel))
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("[%d/%d] %s: 回测失败 %s", k, len(todo_idx), name, str(exc)[:150])
        if k % 10 == 0:
            log.info("[%d/%d] 最近: %s | %.0fs", k, len(todo_idx), name, time.time() - t0)
        if k % args.flush_every == 0:
            lib._save_registry(reg)
            log.info("— 中途落盘 %d/%d —", k, len(todo_idx))

    lib._save_registry(reg)
    log.info("完成: 回测 %d 成功 / %d 缺面板 | 总耗时 %.0fs | registry: %s",
             n_ok, n_missing, time.time() - t0, ds_dir / "registry.csv")

    # 汇总报告（实验目录）
    out = ROOT / "reports" / "library_governance" / "backfill_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    filled = reg["annual_return_ls_M"].notna().mean() * 100 if "annual_return_ls_M" in reg else 0
    sig = reg["significant"].fillna(False).astype(bool).sum() if "significant" in reg else 0
    out.write_text(json.dumps({
        "dataset": args.dataset, "n_total": len(reg), "n_processed": len(todo_idx),
        "n_backtest_ok": n_ok, "n_panel_missing": n_missing,
        "annual_return_ls_M_fill_pct": round(filled, 1),
        "significant_raw": int(sig),
        "backup": str(bak),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("汇总: %s", out)


if __name__ == "__main__":
    main()
