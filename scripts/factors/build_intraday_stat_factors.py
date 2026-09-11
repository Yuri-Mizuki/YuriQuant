"""
分钟统计特征 → 因子库体检入库（降维路线第一层）
================================================

对 ``factor/intraday_features.py`` 的 42 个分钟统计特征做单因子体检：

1. **口径对齐**：除权除息日剔除（分钟线不复权，除权日 bar 收益是污染源，
   掩码逻辑与 ``build_intraday_factors`` 一致）→ 截面 zscore → 入库
   ``FactorLibrary``（IC / NW-t / IC 衰减 / canonical 回测全套自动计算）。
2. **冗余体检**：与库内全部已有因子做逐日截面秩相关（批量向量化，每个库
   面板只读一次），报告每个新因子最相似的 top-K 旧因子；对 top1 做正交
   残差 IC 检验（残差 |NW-t|>2 = 含库外增量信息）。

产出：
- 入库 42 个 ``im_*`` 因子（hs300_2022_2025 数据集，覆盖同名重注册）
- ``reports/intraday_stat_checkup.csv``：体检明细表
- 终端摘要（按 |NW-t| 排序）

用法
----
    python -m scripts.factors.build_intraday_stat_factors                 # 全量体检
    python -m scripts.factors.build_intraday_stat_factors --no-dup        # 只入库不做冗余体检
    python -m scripts.factors.build_intraday_stat_factors --features vol_rv,pv_vwap_dev
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.cache import DataCache  # noqa: E402
from data.intraday import MinutePanelStore  # noqa: E402
from data.offline import OfflineDataSource  # noqa: E402
from data.universe import Universe  # noqa: E402
from factor.intraday_features import FEATURE_DOCS, extract_all  # noqa: E402
from scripts.factors.build_intraday_factors import ex_div_keys  # noqa: E402
from scripts.common.cli_common import (  # noqa: E402
    record_experiment_safe,
    register_panels,
    returns_from_daily,
    setup_logging,
)

log = setup_logging("build_intraday_stat_factors")

#: 特征组 → 六维标签 family（入库元数据）
FAMILY_MAP = {
    "mom_": "动量", "ret_": "其他", "vol_": "波动率",
    "shape_": "技术", "pv_": "流动性", "candle_": "技术",
}
PREFIX = "im_"


def load_context(period: int, pool: str):
    """离线读分钟面板 + 日线 + 状态表（不连 SDK）。"""
    store = MinutePanelStore.open(period, pool)
    days = store.all_days()
    begin, end = days[0], days[-1]

    cache = DataCache(OfflineDataSource())
    uni = Universe(cache)
    from data.cache_helpers import load_daily
    codes, _cal, daily = load_daily(cache, uni, "000300.SH", begin, end)
    status = cache.get_history_stock_status(codes, begin, end)
    log.info("面板 %d 日（%d→%d）/ 日线 %d 行 / 状态 %d 行",
             len(days), begin, end, len(daily), len(status))
    return store, codes, daily, status


def mask_ex_div(features: dict[str, pd.DataFrame],
                status: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """除权除息日置 NaN（该日分钟 bar 收益被跳变污染）。"""
    bad = ex_div_keys(status)
    if not bad:
        return features
    dates = next(iter(features.values())).index
    codes = next(iter(features.values())).columns
    mask = pd.DataFrame(False, index=dates, columns=codes)
    for d, c in bad:
        d = pd.Timestamp(d)
        if d in mask.index and c in mask.columns:
            mask.loc[d, c] = True
    n = int(mask.sum().sum())
    log.info("除权除息掩码: %d 个 (date, code) 格子", n)
    return {k: p.where(~mask) for k, p in features.items()}


# ----------------------------------------------------------------------
# 批量冗余体检
# ----------------------------------------------------------------------
def daily_rank_corr_mean(new_stack: dict[str, np.ndarray], valid_union: np.ndarray,
                         old: pd.DataFrame, dates: pd.DatetimeIndex, codes: list[str],
                         min_codes: int = 10) -> np.ndarray:
    """单个库面板 vs 42 个新面板的逐日截面秩相关均值（向量化）。

    new_stack: {name: [T, C] 已按 (dates, codes) 对齐的排名矩阵}。
    返回 [K] 每个新因子的平均相关（NaN=重叠不足）。
    """
    old_a = old.reindex(index=dates, columns=codes)
    old_v = np.isfinite(old_a.to_numpy())
    if old_v.sum() < 30:
        return np.full(len(new_stack), np.nan)
    old_r = np.where(old_v, old_a.rank(axis=1).to_numpy(), np.nan)

    import warnings
    out = []
    for name, new_r in new_stack.items():
        both = old_v & valid_union[name]
        n_codes = both.sum(axis=1)  # [T] 每日有效代码数
        if (n_codes >= min_codes).sum() < 30:
            out.append(np.nan)
            continue
        a = np.where(both, old_r, np.nan)
        b = np.where(both, new_r, np.nan)
        # 无重叠日（库面板日期错开时）nanmean 全 NaN 切片告警——结果 NaN 本身
        # 会被后续 isfinite 过滤，压制告警防刷屏
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ma = np.nanmean(a, axis=1, keepdims=True)  # [T,1] 日内截面均值
            mb = np.nanmean(b, axis=1, keepdims=True)
        da = np.where(both, a - ma, 0.0)
        db = np.where(both, b - mb, 0.0)
        cov = (da * db).sum(axis=1)  # [T]
        va = (da * da).sum(axis=1)
        vb = (db * db).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.sqrt(va * vb)
        corr = corr[n_codes >= min_codes]
        corr = corr[np.isfinite(corr)]
        out.append(float(corr.mean()) if len(corr) else np.nan)
    return np.array(out)


def batch_dup_report(lib, new_panels: dict[str, pd.DataFrame],
                     returns_panel: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """42 个新因子 vs 全库的相似度体检表。

    每个库面板只读一次（register 的逐因子 check_dup 是 42×N 次读盘，
    这里降为 N 次），对 42 个新因子一次向量化算完。
    """
    names = list(new_panels)
    dates = next(iter(new_panels.values())).index
    codes = list(next(iter(new_panels.values())).columns)
    # 预排名 + 有效掩码（rank 与库内 _run_dup_check 同口径：截面 rank）
    new_stack, valid_union = {}, {}
    for name, p in new_panels.items():
        r = p.reindex(index=dates, columns=codes).rank(axis=1).to_numpy()
        new_stack[name] = r
        valid_union[name] = np.isfinite(r)

    reg = lib.list_all()
    reg = reg[~reg["name"].isin(names)]  # 排除本次新增（覆盖重跑场景）
    best = {name: [] for name in names}  # name -> [(corr, 旧因子名)]
    log.info("冗余体检: 与库内 %d 个因子比对（逐面板流式）...", len(reg))
    for i, (_, r) in enumerate(reg.iterrows(), 1):
        p = Path(str(r.get("panel_path", "")))
        if not p.exists():
            continue
        try:
            old = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        corrs = daily_rank_corr_mean(new_stack, valid_union, old, dates, codes)
        for name, c in zip(names, corrs):
            if np.isfinite(c):
                best[name].append((float(c), r["name"]))
        if i % 200 == 0:
            log.info("  已比对 %d/%d", i, len(reg))

    rows = []
    for name in names:
        cand = sorted(best[name], reverse=True)[:top_k]
        top1 = cand[0] if cand else (np.nan, "")
        resid_t = np.nan
        if cand and abs(top1[0]) > 0.5:
            try:
                _ric, resid_t = lib._residual_ic(new_panels[name], returns_panel, top1[1])
            except Exception:  # noqa: BLE001
                pass
        rows.append({
            "name": name,
            "dup_top1": top1[1], "dup_corr1": top1[0],
            "dup_top2": cand[1][1] if len(cand) > 1 else "",
            "dup_corr2": cand[1][0] if len(cand) > 1 else np.nan,
            "dup_top3": cand[2][1] if len(cand) > 2 else "",
            "dup_corr3": cand[2][0] if len(cand) > 2 else np.nan,
            "resid_t_nw": resid_t,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="分钟统计特征体检入库（第一层）")
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--universe", default="hs300")
    ap.add_argument("--dataset", default="hs300_2022_2025")
    ap.add_argument("--no-dup", action="store_true", help="跳过冗余体检（只入库）")
    ap.add_argument("--dup-only", action="store_true",
                    help="只重跑冗余比对+报告（面板沿用库内已入库版本，跳过重注册）")
    ap.add_argument("--no-save", action="store_true", help="只算不入库")
    ap.add_argument("--features", default=None, help="只体检指定特征（逗号分隔，不带 im_ 前缀）")
    args = ap.parse_args()

    store, codes, daily, status = load_context(args.period, args.universe)
    feats = extract_all(store)
    if args.features:
        want = [f.strip() for f in args.features.split(",") if f.strip()]
        bad = [w for w in want if w not in FEATURE_DOCS]
        if bad:
            raise SystemExit(f"未知特征 {bad}，可选 {sorted(FEATURE_DOCS)}")
        feats = {k: v for k, v in feats.items() if k in want}
    feats = mask_ex_div(feats, status)
    returns_panel = returns_from_daily(daily)

    prefixed = {f"{PREFIX}{k}": v for k, v in feats.items()}
    defs = {f"{PREFIX}{k}": f"[min{args.period}统计特征] {d}"
            for k, d in FEATURE_DOCS.items() if k in feats}

    if args.no_save:
        log.info("--no-save：面板 %d 个，形状 %s", len(prefixed),
                 next(iter(prefixed.values())).shape)
        return

    from research.factor_library import FactorLibrary
    lib = FactorLibrary(dataset=args.dataset)

    if args.dup_only:
        # 面板已入库（含掩码+标准化），只重跑冗余比对/报告——比对一次 25 分钟，
        # 入库只要 5 分钟，报告迭代不应重复入库计算
        reg_rows = [{"name": n} for n in prefixed]
        prefixed = {n: lib.get_panel(n) for n in prefixed}
    else:
        log.info("入库 %d 个统计特征 → 数据集 %s ...", len(prefixed), args.dataset)
        reg_rows = register_panels(
            lib, prefixed, defs, returns_panel,
            source=f"intraday:stat_features:min{args.period}",
        )
        # 六维标签（register_panels 不带 family 入参，入库后补 tag）
        for row in reg_rows:
            base = row["name"][len(PREFIX):]
            family = next((v for k, v in FAMILY_MAP.items() if base.startswith(k)), "其他")
            lib.set_tag(row["name"], family=family, frequency="日内")

    # ---- 体检报告 ----
    reg = lib.list_all()
    new_reg = reg[reg["name"].isin([r["name"] for r in reg_rows])].copy()
    # registry 自带 register(check_dup=False) 的占位 dup 列（空串），先丢弃
    # 再 merge 批量体检结果，否则产生 _x/_y 后缀撞名
    new_reg = new_reg.drop(columns=[c for c in
                                    ("dup_checked", "dup_corr_max", "dup_top",
                                     "resid_ic", "resid_t_nw") if c in new_reg.columns])
    if not args.no_dup:
        dup = batch_dup_report(lib, prefixed, returns_panel)
        new_reg = new_reg.merge(dup, on="name", how="left")
    else:
        for col in ("dup_top1", "dup_corr1", "dup_top2", "dup_corr2",
                    "dup_top3", "dup_corr3", "resid_t_nw"):
            new_reg[col] = np.nan

    show_cols = ["name", "ic_mean", "t_stat_nw", "ic_ir", "best_sharpe",
                 "dup_corr1", "dup_top1", "resid_t_nw"]
    new_reg["_abs_t"] = new_reg["t_stat_nw"].abs()
    report = new_reg.sort_values("_abs_t", ascending=False)[show_cols]
    out = ROOT / "reports" / "intraday_stat_checkup.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_colwidth", 28,
                           "display.float_format", "{:.3f}".format):
        print("\n==== 分钟统计特征体检（按 |NW-t| 排序，未中性化口径） ====")
        print(report.to_string(index=False))
    log.info("报告落盘: %s", out)

    n_sig = int((new_reg["t_stat_nw"].abs() > 2).sum())
    n_dup = int((new_reg["dup_corr1"] > 0.7).sum()) if not args.no_dup else -1
    log.info("摘要: %d/%d 显著(|t_nw|>2)，%d 个与库内因子相关>0.7",
             n_sig, len(new_reg), n_dup)

    record_experiment_safe(
        kind="intraday_stat_checkup",
        command=" ".join(sys.argv),
        params={"period": args.period, "universe": args.universe,
                "dataset": args.dataset, "n_features": len(reg_rows)},
        fingerprint=None, result_path=str(out),
        metrics={"n_significant": n_sig, "n_dup_gt_07": n_dup,
                 "best_t": float(new_reg["_abs_t"].max()) if len(new_reg) else 0.0},
        note="分钟统计特征第一层体检入库",
    )


if __name__ == "__main__":
    main()
