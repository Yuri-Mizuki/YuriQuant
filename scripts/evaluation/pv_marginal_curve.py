"""量价因子边际增益曲线 —— 阶段 B（DPP 组合口径）。

问题
----
在定版流程下，**多给模型量价因子，组合层面还有没有增益**？

与 `reports/ic_headroom/` 前作的关系（重要）
------------------------------------------
前作结论「候选池 50 → 全 150，IC 0.0785 → 0.0598」**对照不公平**：④ 臂在扩池的
同时**换掉了选择机制**（全 150 直接最优合成，无 DPP 精选）。降幅可能来自
「等权/最优合成高度相关的 150 个因子 → 噪声互相抵消」，而非「信息耗尽」。

本实验修正：**唯一自变量 = 量价候选池大小 K，选择机制恒为 DPP 选 50**。

设计
----
- `mixed_K{K}`  ：池 = 量价 top-K（按 Y-1 |IC|）+ 非量价全量
- `pure_pv_K{K}` ：池 = 量价 top-K（不含非量价）
- `random_K200_s{seed}`：量价随机 200（不看 IC）→ 排序是否带来选择偏差
- 选择：`dpp_select(k=50, sigma=0.2)`，质量项 = Y-1 |IC|；合成：等权 rank 平均
- **PIT**：quality / 排序只用 Y-1 前 500 日；评估在 Y 年

性能（关键）
------------
每个面板 parquet = 2471 行 × 5808 列、**单 row group、几乎不压缩**（57MB），
行过滤无效 ⇒ 读文件必读全量（0.37s/文件）。故本脚本**一遍读取抽齐所有年份采样**
（旧实现每年重读一遍，7 倍 IO）。

用法
----
    python -m scripts.evaluation.pv_marginal_curve           # 全量（~20 min）
    python -m scripts.evaluation.pv_marginal_curve --quick   # 冒烟（1 年）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
sys.path.insert(0, r"E:/YuriQuant")

from research.dpp_selection import corr_matrix, dpp_select  # noqa: E402
from stats.ic import calc_ic_series  # noqa: E402

LIB = r"E:\data\factor_library\all_a_2018_2026"
P_NEU = os.path.join(LIB, "panels_neu_fundind")
IC_H1 = os.path.join(LIB, "ic_h1.parquet")
REG = os.path.join(LIB, "registry.csv")
CLOSE_ADJ = os.path.join(
    r"E:\YuriQuant\reports\alla_rolling_ortho920fundind\_base", "close_adj.parquet")

QUALITY_WINDOW = 500
MIN_COVERAGE = 0.5
MAX_FEATURES = 50
SIGMA = 0.2
SAMPLE_STEP = 12          # corr 采样日步长（500 日 → ~42 日/年）
PV_GRID = [0, 36, 50, 100, 200, 350, 500, 791]
PURE_PV_GRID = [36, 50, 100, 200, 350, 791]
RANDOM_K = 200
RANDOM_SEEDS = [0, 1, 2]

_COL_CACHE: dict[str, set] = {}


def _cols_of(name: str) -> set:
    if name not in _COL_CACHE:
        _COL_CACHE[name] = set(pq.read_schema(
            os.path.join(P_NEU, f"{name}.parquet")).names)
    return _COL_CACHE[name]


def read_cols(name: str, cols) -> pd.DataFrame | None:
    """读一个面板（全历史行），只取指定列。"""
    use = [c for c in cols if c in _cols_of(name)]
    if not use:
        return None
    try:
        return pd.read_parquet(os.path.join(P_NEU, f"{name}.parquet"), columns=use)
    except Exception:
        return None


def combo_equal_weight(sel, panels, sign_all, dY, cols):
    acc = None
    idx = None
    for nm in sel:
        p = panels.get(nm)
        if p is None:
            continue
        r = p.reindex(index=dY, columns=cols)
        r = (r.rank(axis=1, pct=True) - 0.5) * sign_all.get(nm, 1.0)
        v = r.fillna(0.0).values.astype(np.float64)
        if acc is None:
            acc = np.zeros(v.shape)
            idx = r.index
        acc += v
    if acc is None:
        return None
    return pd.DataFrame(acc.astype(np.float32), index=idx, columns=cols)


def main() -> int:
    ap = argparse.ArgumentParser(description="量价因子边际增益曲线（阶段 B：DPP 组合口径）")
    ap.add_argument("--years", default="2019,2020,2021,2022,2023,2024,2025")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=r"E:\YuriQuant\reports\pv_marginal")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    years = [int(x) for x in args.years.split(",")]
    if args.quick:
        years = years[:1]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(r"E:\YuriQuant\logs", exist_ok=True)
    log_path = os.path.join(
        r"E:\YuriQuant\logs",
        f"pv_marginal_dpp{('_' + args.tag) if args.tag else ''}"
        f"{'_quick' if args.quick else ''}_{time.strftime('%Y%m%d_%H%M')}.log")
    log_fh = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    t_all = time.time()
    log(f"[0] 阶段 B 启动 | years={years} | log={log_path}")

    close_adj = pd.read_parquet(CLOSE_ADJ)
    fwd = close_adj.pct_change(1, fill_method=None).shift(-1)
    all_days = close_adj.index
    cols = list(close_adj.columns)
    cols_s = cols[::2]
    log(f"    收益面板 {close_adj.shape}")

    reg = pd.read_csv(REG)
    fam = dict(zip(reg["name"], reg["family"]))
    covd = dict(zip(reg["name"], reg["coverage"]))
    ic = pd.read_parquet(IC_H1)

    have = {f[:-8] for f in os.listdir(P_NEU) if f.endswith(".parquet")}
    pv_all = sorted(n for n in fam if fam[n] == "量价" and n in have
                    and covd.get(n, 0) >= MIN_COVERAGE)
    nonpv_all = sorted(n for n in fam if fam[n] != "量价" and n in have
                       and covd.get(n, 0) >= MIN_COVERAGE and n in set(ic.columns))
    all_factors = pv_all + nonpv_all
    log(f"    因子池：量价 {len(pv_all)} | 非量价 {len(nonpv_all)} | 合计 {len(all_factors)}")

    # ---- 0) 各年窗口计划 ----
    plan: dict[int, dict] = {}
    for Y in years:
        cut = pd.Timestamp(f"{Y - 1}-12-31")
        q_days = ic.index[ic.index <= cut][-QUALITY_WINDOW:]
        dY = all_days[(all_days >= f"{Y}-01-01") & (all_days <= f"{Y}-12-31")]
        if len(q_days) < 30 or len(dY) < 50:
            log(f"    [{Y}] 窗口不足 → 跳过")
            continue
        plan[Y] = dict(q_days=q_days, sample_days=q_days[::SAMPLE_STEP], dY=dY)
    years = sorted(plan)
    log(f"    参与年份 {years}")

    # ---- 1) 单遍读取：抽齐所有年份的采样数据 ----
    t1 = time.time()
    sample_by_year: dict[int, dict] = {Y: {} for Y in years}
    for i, n in enumerate(all_factors):
        sub = read_cols(n, cols_s)
        if sub is None:
            continue
        idx = sub.index
        for Y in years:
            sd = plan[Y]["sample_days"]
            keep = sd.intersection(idx)
            if len(keep) >= 20:
                sample_by_year[Y][n] = sub.reindex(index=keep)
        del sub
        if (i + 1) % 150 == 0:
            log(f"    采样读取 {i + 1}/{len(all_factors)}（{time.time() - t1:.0f}s）")
    log(f"[1] 单遍采样读取完成 {time.time() - t1:.0f}s")

    # ---- 2) 逐年 corr（各臂取子矩阵）----
    corr_by_year = {}
    for Y in years:
        corr_by_year[Y] = corr_matrix(sample_by_year[Y], method="cross")
        log(f"    [{Y}] corr {corr_by_year[Y].shape[0]}×{corr_by_year[Y].shape[1]}")
    del sample_by_year

    # ---- 3) 逐年 DPP 选 50 ----
    arms_by_year: dict[int, list[tuple[str, list[str]]]] = {}
    for Y in years:
        corr = corr_by_year[Y]
        qi = ic.loc[plan[Y]["q_days"]]
        q_mean = qi.mean()
        quality = q_mean.abs()
        sign_all = {n: (float(np.sign(q_mean.get(n, 0.0))) or 1.0) for n in ic.columns}
        plan[Y]["sign_all"] = sign_all

        pv_ranked = [n for n in quality.reindex(pv_all).sort_values(
            ascending=False).index if n in corr.index]
        nonpv_in = [n for n in nonpv_all if n in corr.index]
        arms: list[tuple[str, list[str]]] = []

        def build(tag: str, pool: list[str], _corr=corr, _q=quality) -> None:
            pool = [n for n in pool if n in _corr.index]
            if len(pool) < MAX_FEATURES:
                return
            sub = _corr.loc[pool, pool]
            qv = _q.reindex(pool).fillna(0).values
            sel = [n for n in dpp_select(sub, k=MAX_FEATURES, quality=qv,
                                         sigma=SIGMA)["selected"]]
            n_pv = sum(1 for n in sel if fam.get(n) == "量价")
            arms.append((tag, sel))
            log(f"    [{Y}] {tag}: 池 {len(pool)} → 选 {len(sel)}（量价 {n_pv}）")

        for K in PV_GRID:
            build(f"mixed_K{K}", (pv_ranked[:K] if K > 0 else []) + nonpv_in)
        for K in PURE_PV_GRID:
            build(f"pure_pv_K{K}", pv_ranked[:K])
        for sd in RANDOM_SEEDS:
            rng = np.random.default_rng(sd)
            idxs = rng.choice(len(pv_ranked), size=min(RANDOM_K, len(pv_ranked)),
                              replace=False)
            build(f"random_K{RANDOM_K}_s{sd}", [pv_ranked[i] for i in idxs] + nonpv_in)
        arms_by_year[Y] = arms

    # ---- 4) 第二遍读取：只取被选因子的 dY 面板 → 合成 → IC ----
    rows = []
    for Y in years:
        dY = plan[Y]["dY"]
        fwdY = fwd.reindex(index=dY)
        sels = arms_by_year[Y]
        need = sorted({n for _, sel in sels for n in sel})
        panels = {}
        for n in need:
            s = read_cols(n, cols_s)
            if s is None:
                continue
            s = s.reindex(index=s.index.intersection(dY))
            if len(s) >= 50:
                panels[n] = s
        log(f"  [{Y}] 载入被选因子 {len(panels)}/{len(need)}（{len(dY)} 日）")
        for tag, sel in sels:
            sub = [n for n in sel if n in corr_by_year[Y].index]
            if len(sub) > 1:
                c = corr_by_year[Y].loc[sub, sub].values
                nn = len(sub)
                mac = float(np.abs(c[np.triu_indices(nn, 1)]).mean())
            else:
                mac = np.nan
            combo = combo_equal_weight(sel, panels, plan[Y]["sign_all"], dY, cols_s)
            v = np.nan if combo is None else float(
                calc_ic_series(combo, fwdY).dropna().mean())
            rows.append(dict(year=Y, arm=tag, n_sel=len(sel),
                             mean_abs_corr=round(mac, 3), combo_ic=round(v, 4)))
            log(f"      {tag:22s} IC={v:+.4f}  mean|corr|={mac:.3f}")
        del panels

    df = pd.DataFrame(rows)
    stem = "dpp_curve" + (f"_{args.tag}" if args.tag else "")
    csv = os.path.join(args.out_dir, f"{stem}.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    if not df.empty:
        log("\n===== 跨年平均（combo_ic）=====")
        agg = df.groupby("arm").agg(
            combo_ic=("combo_ic", "mean"),
            mean_abs_corr=("mean_abs_corr", "mean")).round(4)
        log(agg.to_string())
    log(f"\n[✓] 完成 {time.time() - t_all:.0f}s → {csv}")
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
