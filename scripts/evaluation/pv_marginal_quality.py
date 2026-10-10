"""量价因子边际增益曲线 —— 阶段 A（池质量口径，零额外 IO）。

问题
----
多给模型量价因子，**候选池边际质量**还有没有提升？

为什么先做这一版
----------------
- `ic_h1.parquet` 已含 2471 日 × 948 因子的逐日 IC，**不需要读 915 个 57MB 面板**
  （DPP 版需要相关矩阵，见同目录 `pv_marginal_curve.py`，IO 成本分钟~小时级）。
- 选择器用「池内 |IC| top-50」而非 DPP：**去掉多样性约束这个混淆变量**，直接测
  「池子变大后，最强的 50 个是否变强」。这是边际问题的最大公约数。
- PIT：quality 只用 Y-1 前 500 日；评估读 Y 年 IC。无 look-ahead。

臂
--
- `mixed_K{K}`  ：池 = 量价 top-K（按 Y-1 |IC|）+ 非量价全量
- `pure_pv_K{K}` ：池 = 量价 top-K（不含非量价）
- `pool_avg_K{K}`：不做选择，**池内全部因子的平均 OOS |IC|** → 测「扩容稀释」

输出
----
`reports/pv_marginal/quality_curve.csv` + 控制台汇总。

用法
----
    python -m scripts.evaluation.pv_marginal_quality
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"E:/YuriQuant")

LIB = r"E:\data\factor_library\all_a_2018_2026"
IC_H1 = os.path.join(LIB, "ic_h1.parquet")
REG = os.path.join(LIB, "registry.csv")

QUALITY_WINDOW = 500
MIN_COVERAGE = 0.5
MAX_FEATURES = 50
MIN_POOL = 50
PV_GRID = [0, 36, 50, 100, 200, 350, 500, 791]
PURE_PV_GRID = [36, 50, 100, 200, 350, 791]


def main() -> int:
    ap = argparse.ArgumentParser(description="量价池边际增益（阶段 A：池质量口径）")
    ap.add_argument("--years", default="2019,2020,2021,2022,2023,2024,2025")
    ap.add_argument("--out-dir", default=r"E:\YuriQuant\reports\pv_marginal")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    args = ap.parse_args()
    years = [int(x) for x in args.years.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    ic = pd.read_parquet(IC_H1)
    reg = pd.read_csv(REG)
    fam = dict(zip(reg["name"], reg["family"]))
    covd = dict(zip(reg["name"], reg["coverage"]))

    usable = [n for n in ic.columns
              if n in fam and covd.get(n, 0) >= args.min_coverage]
    pv_all = sorted(n for n in usable if fam[n] == "量价")
    nonpv_all = sorted(n for n in usable if fam[n] != "量价")
    print(f"因子池：量价 {len(pv_all)} | 非量价 {len(nonpv_all)} | 有 IC 列 {len(ic.columns)}")
    print(f"IC 面板 {ic.shape} | {ic.index[0].date()}~{ic.index[-1].date()}\n")

    rows = []
    for Y in years:
        cut = pd.Timestamp(f"{Y - 1}-12-31")
        q_days = ic.index[ic.index <= cut][-QUALITY_WINDOW:]
        if len(q_days) < 30:
            print(f"[{Y}] 质量窗口不足 → 跳过")
            continue
        qi = ic.loc[q_days]
        q_mean = qi.mean()
        quality = q_mean.abs()                       # Y-1 |IC|（选择依据）
        sign = np.sign(q_mean).replace(0.0, 1.0)     # Y-1 方向（对齐符号）

        dY = ic.index[(ic.index >= f"{Y}-01-01") & (ic.index <= f"{Y}-12-31")]
        icY = ic.loc[dY]
        # 样本外：先按训练期符号对齐，再取逐因子的 Y 年均值
        oos = (icY * sign.reindex(icY.columns)).mean()
        oos_abs = icY.abs().mean()

        pv_ranked = [n for n in quality.reindex(pv_all).sort_values(
            ascending=False).index if pd.notna(quality.get(n))]

        def record(tag: str, pool: list[str]) -> None:
            pool = [n for n in pool if n in icY.columns]
            if len(pool) < MIN_POOL:
                return
            sel = sorted(pool, key=lambda n: -float(quality.get(n, 0.0)))[:MAX_FEATURES]
            n_pv = sum(1 for n in sel if fam.get(n) == "量价")
            rows.append(dict(
                year=Y, arm=tag, n_pool=len(pool), n_sel=len(sel),
                pv_share_sel=round(n_pv / max(len(sel), 1), 3),
                sel_oos_ic=round(float(oos.reindex(sel).mean()), 4),
                sel_oos_absic=round(float(oos_abs.reindex(sel).mean()), 4),
                sel_quality=round(float(quality.reindex(sel).mean()), 4),
                pool_oos_absic=round(float(oos_abs.reindex(pool).mean()), 4),
            ))

        for K in PV_GRID:
            record(f"mixed_K{K}", (pv_ranked[:K] if K > 0 else []) + nonpv_all)
        for K in PURE_PV_GRID:
            record(f"pure_pv_K{K}", pv_ranked[:K])
        print(f"[{Y}] 质量窗 {q_days[0].date()}~{q_days[-1].date()} | "
              f"OOS {dY[0].date()}~{dY[-1].date()} | {len(dY)} 日")

    df = pd.DataFrame(rows)
    csv = os.path.join(args.out_dir, "quality_curve.csv")
    df.to_csv(csv, index=False, encoding="utf-8-sig")

    piv = df.pivot_table(index="arm", columns="year", values="sel_oos_ic", aggfunc="mean")
    piv["mean"] = piv.mean(axis=1)
    print("\n===== 选中 50 个的 OOS 平均 IC（按臂 × 年）=====")
    print(piv.round(4).to_string())

    print("\n===== 跨年平均（臂排序）=====")
    agg = df.groupby("arm").agg(
        n_pool=("n_pool", "mean"),
        pv_share_sel=("pv_share_sel", "mean"),
        sel_oos_ic=("sel_oos_ic", "mean"),
        sel_oos_absic=("sel_oos_absic", "mean"),
        sel_quality=("sel_quality", "mean"),
        pool_oos_absic=("pool_oos_absic", "mean"),
    ).round(4)

    def order(tag: str) -> tuple[int, int]:
        if tag.startswith("mixed_K"):
            return (0, int(tag.split("K")[1]))
        return (1, int(tag.split("K")[1]))

    agg = agg.loc[sorted(agg.index, key=order)]
    print(agg.to_string())
    # ---- 补充：因子质量随排名的衰减曲线（回答「第 N 名还有多少 IC」）----
    buckets = [(0, 25), (25, 50), (50, 100), (100, 200),
               (200, 350), (350, 500), (500, len(pv_all))]
    decay: dict[tuple, list] = {b: [] for b in buckets}
    for Y in years:
        cut = pd.Timestamp(f"{Y - 1}-12-31")
        qd = ic.index[ic.index <= cut][-QUALITY_WINDOW:]
        if len(qd) < 30:
            continue
        qm = ic.loc[qd].mean()
        sign = np.sign(qm).replace(0.0, 1.0)
        dY = ic.index[(ic.index >= f"{Y}-01-01") & (ic.index <= f"{Y}-12-31")]
        icY = ic.loc[dY]
        oos = (icY * sign.reindex(icY.columns)).mean()
        ranked = qm.reindex(pv_all).abs().sort_values(ascending=False).index.tolist()
        for b in buckets:
            sel = ranked[b[0]:b[1]]
            if sel:
                decay[b].append(float(oos.reindex(sel).mean()))
    dec_rows = []
    base = float(np.mean(decay[buckets[0]])) if decay[buckets[0]] else np.nan
    for b in buckets:
        m = float(np.mean(decay[b])) if decay[b] else np.nan
        dec_rows.append(dict(rank_range=f"{b[0] + 1}-{b[1]}", oos_ic=round(m, 4),
                             rel_to_top25=round(m / base, 3) if base else np.nan))
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(os.path.join(args.out_dir, "quality_decay.csv"),
               index=False, encoding="utf-8-sig")
    print("\n===== 量价因子质量随排名衰减（OOS IC，符号对齐）=====")
    print(dec.to_string(index=False))

    print(f"\n[✓] {time.time() - t0:.1f}s → {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
