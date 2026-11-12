"""因子库指标批量回填：canonical 分层回测 + 显著性 + 路径/公式/族标签（维护工具）。

背景（2026-11-05 体检）：all_a_2018_2026 库 950 因子由 build_alla_alpha_panels 等
批量通道经 ``upsert_rows`` 入库，只写了 name/set/label/coverage/ic_mean_h*；
``register()`` 正常入库会算的 canonical 回测（ls_M/lo_M/ls_W）、IC 显著性
（NW-t / |t_nw|>2）、panel_path/eval_path/formula 全部缺位（0% 填充）。
本脚本对**整库**补齐这些列，口径逐项对齐 ``register()``，不自造：

- 分层回测：``VectorBacktest(cfg.strategy(k=cfg.k), rebalance_freq=cfg.freq)``，
  cfg ∈ CANONICAL_CONFIGS（k=30，月/周），成本走引擎默认（``Config.costs()`` + 借券费），
  与 hs300 库既有登记口径同源；
- 收益面板：后复权 close（与 build_alla_alpha_panels.load_alla_panels 的 close 分支
  逐行同构）→ ``pct_change(1, fill_method=None).shift(-1)``（库内 shift(-1) 口径）；
- IC 显著性：**不重算 IC**——直接消费既有 ic_h1.parquet（批量链的 canonical IC，
  stride-3 抽稀截面口径，与 registry 已填 99% 的 ic_mean_h* 同源），
  逐因子过 ``stats.significance.mean_inference``（NW），significant = |t_nw| > 2
  （register 同判据）；ic_ir 用 ``research.factor_analysis.calc_ir``（register 同源）；
- 超额年化（看板对齐聚宽「超额收益」列）：仅对 lo_M（TopK 纯多）配置传
  ``metrics(benchmark_returns=全A等权日收益)``（returns_panel 截面均值，含当日
  有交易个股），经 ``calc_all_metrics`` 现成口径产出 excess_return_lo_M /
  information_ratio_lo_M / excess_t_stat_lo_M（新增列，无历史口径冲突）；
  LS 多空为自融资组合，对基准求超额无经济含义，不算。基准序列落
  ``bench_returns.parquet``（数据集级一份）；
- ic_decay5 / autocorr：全截面面板现算（register 同函数，新增列无历史口径冲突）；
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


def load_close_adj(cache_root: Path) -> pd.DataFrame:
    """后复权 close 面板——与 build_alla_alpha_panels.load_alla_panels 的
    close 分支逐行同构（raw × backward_factor，ffill 复权因子）。"""
    daily = pd.read_parquet(cache_root / "daily_all_a.parquet")
    daily.index = daily.index.set_levels(daily.index.levels[0].normalize(), level=0)
    d = daily.reset_index()
    d["date"] = d["date"].dt.normalize()
    c = d.pivot(index="date", columns="code", values="close").sort_index()
    bf = pd.read_parquet(cache_root / "backward_factor.parquet")
    bf = bf[[x for x in c.columns if x in bf.columns]]
    f = bf.reindex(index=c.index, columns=c.columns).ffill()
    out = (c.values * f.values)
    return pd.DataFrame(out, index=c.index, columns=c.columns)


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
    ap.add_argument("--flush-every", type=int, default=50)
    args = ap.parse_args()

    from research.factor_analysis import calc_ir
    from research.factor_library import FactorLibrary
    from research.factor_library import METRIC_COLS, CANONICAL_CONFIGS
    from stats.significance import mean_inference

    lib = FactorLibrary(dataset=args.dataset)
    ds_dir = Path(lib.root) / args.dataset
    reg = pd.read_csv(ds_dir / "registry.csv")
    reg = reg.drop_duplicates(subset="name", keep="last").reset_index(drop=True)

    # 备份（写盘纪律）
    bak = ds_dir / f"registry.csv.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(ds_dir / "registry.csv", bak)
    log.info("registry 快照: %s", bak.name)

    # 收益面板（全量一次进内存，串行用）
    returns_panel = None
    if not args.skip_backtest:
        from config import Config
        t0 = time.time()
        close_adj = load_close_adj(Path(str(Config.cache()["root"])))
        k0 = pd.Timestamp("20160701")   # 与批量链 keep_from 一致
        close_kept = close_adj.loc[close_adj.index >= k0]
        returns_panel = close_kept.pct_change(1, fill_method=None).shift(-1)
        del close_adj, close_kept
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

    todo_idx: list[int] = []
    for i, r in reg.iterrows():
        if not args.force and pd.notna(r.get("annual_return_ls_M")) and \
                (ic1 is None or pd.notna(r.get("t_stat_nw"))):
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
        if pd.isna(reg.at[i, "formula"]) or not str(reg.at[i, "formula"]).strip():
            lab = reg.at[i, "label"]
            reg.at[i, "formula"] = str(lab) if pd.notna(lab) else str(name)
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
                reg.at[i, "ic_win_rate"] = float((ic > 0).mean())
                reg.at[i, "significant"] = bool(abs(inf["t_stat_nw"]) > 2.0)
        # ---- canonical 回测 ----
        if not args.skip_backtest:
            if ppath is None:
                n_missing += 1
                log.warning("[%d/%d] %s: 无面板文件，跳过回测", k, len(todo_idx), name)
                continue
            try:
                panel = pd.read_parquet(ppath)
                reg.at[i, "n_dates"] = panel.shape[0]
                reg.at[i, "n_codes"] = panel.shape[1]
                eval_cols: dict = {}
                if ic1 is not None and name in ic1.columns:
                    eval_cols["ic"] = ic1[name]
                best_sharpe, best_config = -np.inf, CANONICAL_CONFIGS[0].key
                for cfg in CANONICAL_CONFIGS:
                    from backtest.engine import VectorBacktest
                    bt = VectorBacktest(cfg.strategy(k=cfg.k), rebalance_freq=cfg.freq)
                    res = bt.run(panel, returns_panel, check_convention=False)
                    eval_cols[f"dret_{cfg.key}"] = res.daily_returns
                    eval_cols[f"equity_{cfg.key}"] = res.equity_curve
                    # 换手序列（调仓日有值）：看板按时间区间重算区间换手要用
                    eval_cols[f"turnover_{cfg.key}"] = res.turnover_series
                    # 超额只对纯多配置算（LS 对基准求超额无经济含义，见 docstring）
                    is_lo = cfg.key.startswith("lo")
                    m = res.metrics(benchmark_returns=bench if is_lo else None)
                    for mc in METRIC_COLS:
                        if mc in m:
                            reg.at[i, f"{mc}_{cfg.key}"] = m[mc]
                    if is_lo:
                        for ek in ("excess_return", "information_ratio", "excess_t_stat"):
                            if ek in m:
                                reg.at[i, f"{ek}_{cfg.key}"] = m[ek]
                    if m["sharpe"] > best_sharpe:
                        best_sharpe, best_config = m["sharpe"], cfg.key
                reg.at[i, "best_sharpe"] = best_sharpe
                reg.at[i, "best_config"] = best_config
                ep = ds_dir / "evals" / f"{slug(str(name))}.parquet"
                ep.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(eval_cols).sort_index().astype(np.float32).to_parquet(ep)
                reg.at[i, "eval_path"] = str(ep)
                # ic_decay5 / autocorr（register 同函数，全截面口径）
                try:
                    from research.factor_analysis import calc_ic_decay, factor_autocorr
                    with np.errstate(all="ignore"):
                        reg.at[i, "ic_decay5"] = float(
                            calc_ic_decay(panel, returns_panel, max_lag=10).get(5, np.nan))
                        reg.at[i, "autocorr"] = float(factor_autocorr(panel))
                except Exception as exc:  # noqa: BLE001
                    log.debug("decay/autocorr 失败 %s: %s", name, exc)
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
    out = ROOT / "reports" / "factor_library_backfill" / "backfill_summary.json"
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
