"""集中档信号选型探针 —— horizon 集 × 稠密 K 网格 × IS/OOS 切分。

为什么要有这个脚本（而不是一次性探针）
------------------------------------
`yqcore-focus` 的信号不是拍出来的：2026-11-10 它从 `ens_h1h5`（只用 h1+h5 两个短
周期）换到 `gbdt_mid13`（h1:h5:h10:h20 = 2:1:1:2 秩平均）。判据有两层，缺一层就会
把噪声当改进：

1. **整条 K 曲线**是否系统性上移 —— 单格读数不构成证据（12 格极差 3.5pp，单格差异
   全在噪声里）。
2. **去一臂（leave-one-horizon-out）** —— 若去掉某周期就塌，说明它是结构性的；
   若去掉谁都差不多，说明「多周期融合」只是平均掉噪声。

另外本脚本报 **IS（2019-2023）/ OOS（2024-2026）跨格秩相关**：若为负，说明「历史
最优的 K」对未来无预测力 ⇒ K 不能按回测最优格挑。这是唯一能挡住「挑格子」的检验。

口径：`caliber.load_context` + `evaluate_strategy`（prod / T+1 open / 固定 10bp /
1e6 / 末日截 2026-09-07），与登记档位逐字同源。**所有臂共用同一份 context**，
只换信号面板 ⇒ 差异只来自信号。

用法::

    python -m scripts.evaluation.focus_upgrade_probe                # 全臂 × 全 K
    python -m scripts.evaluation.focus_upgrade_probe --out reports/focus_upgrade
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import caliber as cal  # noqa: E402
from strategy.cross_section import BufferedTopKLongOnly  # noqa: E402

END_ALIGN = 20260907
PRED = ROOT / "reports/alla_rolling_ortho/pred"
KS = (6, 8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 30)
CAP = 1e6
PROFILE = "production"

#: 臂 = 信号构造式（`gbdt_h*` 是落盘分量；`rank_average` 走 caliber 的唯一构造口径）
HORIZONS = (1, 5, 10, 20)


def _load_pred() -> dict[int, pd.DataFrame]:
    return {h: pd.read_parquet(PRED / f"gbdt__h{h}.parquet") for h in HORIZONS}


def _arms(raw: dict[int, pd.DataFrame], idx, cols) -> dict[str, pd.DataFrame]:
    r = {h: v.reindex(index=idx, columns=cols) for h, v in raw.items()}
    w = cal.weighted_rank_average

    def sub(keys, weights=None):
        return w([r[k] for k in keys], weights)

    return {
        # 现登记档（rev2）
        "gbdt_mid13": sub([1, 5, 10, 20], [2.0, 1.0, 1.0, 2.0]),
        # 四周期等权（机制最朴素的对照）
        "avg4": sub([1, 5, 10, 20]),
        # 「去掉一个 horizon」全因子臂：判 3 个周期是否真的比 4 个好
        "drop_h1_5_10_20": sub([5, 10, 20]),
        "drop_h5_1_10_20": sub([1, 10, 20]),
        "drop_h10_1_5_20": sub([1, 5, 20]),
        "drop_h20_1_5_10": sub([1, 5, 10]),
        # 无中期 / 单长周期
        "1+20": sub([1, 20]),
        "h20only": r[20],
        "h1h5_prev": None,            # rev1 登记信号，下面用落盘面板替换
    }


def _cagr(yearly: dict[int, float], n_days: int) -> float:
    g = 1.0
    for v in yearly.values():
        g *= 1 + v
    return g ** (252.0 / max(n_days, 1)) - 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="reports/focus_upgrade",
                    help="产物目录（相对仓库根）")
    ap.add_argument("--ks", default="", help="逗号分隔的 K；空=内置网格")
    ap.add_argument("--arms", default="", help="逗号分隔的臂名；空=全臂")
    args = ap.parse_args(argv)

    ks = ([int(x) for x in args.ks.split(",") if x.strip()] or list(KS))
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)

    ctx = cal.load_context(signal="ens_h1h5", window="prod", end=END_ALIGN,
                           execution="open")
    cost = cal.cost_profile(PROFILE)
    idx, cols = ctx.pred.index, ctx.pred.columns
    print(f"窗口 {idx.min().date()} .. {idx.max().date()}  {len(idx)} 日")

    arms = _arms(_load_pred(), idx, cols)
    arms["h1h5_prev"] = ctx.pred            # rev1 登记信号（落盘面板，不重算）

    if args.arms:
        keep = [x.strip() for x in args.arms.split(",") if x.strip()]
        arms = {k: v for k, v in arms.items() if k in keep}

    cover = {k: float(v.notna().sum().sum()) / float(ctx.pred.notna().sum().sum())
             for k, v in arms.items()}
    print("\n信号覆盖率（相对 ens_h1h5 非空格）")
    for k, c in cover.items():
        print(f"  {k:<18s} {c * 100:6.2f}%")

    rows = []
    for name, panel in arms.items():
        for k in ks:
            m = cal.evaluate_strategy(BufferedTopKLongOnly(k=k, exit_mult=2.0),
                                      ctx, cost, CAP, execution="open",
                                      signal=panel, yearly=True)
            yr = {int(y): m[f"yearly_ret_{y}"] for y in range(2019, 2027)
                  if f"yearly_ret_{y}" in m}
            i5 = {y: v for y, v in yr.items() if y <= 2023}
            o3 = {y: v for y, v in yr.items() if y >= 2024}
            n_is = int((idx.year <= 2023).sum())
            n_oos = int((idx.year >= 2024).sum())
            rows.append({
                "arm": name, "k": k, "ann": m["ann"], "sharpe": m["sharpe"],
                "sharpe_arith": m["sharpe_arith"],
                "max_dd": m["max_dd"], "turnover": m["turnover"],
                "n_active": m["n_active"], "cost_bp": m["cost_bp"],
                "is_ann": _cagr(i5, n_is), "oos_ann": _cagr(o3, n_oos),
                **{f"y{y}": v for y, v in yr.items()},
            })
            r = rows[-1]
            print(f"  {name:<18s} K={k:<3d} 年化 {r['ann']*100:6.2f}%  "
                  f"SR {r['sharpe']:.3f}  回撤 {r['max_dd']*100:5.1f}%  "
                  f"IS {r['is_ann']*100:6.2f}%  OOS {r['oos_ann']*100:6.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "grid.csv", index=False, encoding="utf-8-sig")

    print("\n=== 各臂 K 曲线（年化 %） ===")
    piv = df.pivot(index="k", columns="arm", values="ann") * 100
    print(piv.round(2).to_string())
    print("\n各臂 K 曲线统计（跨格；单格差异 <1.5pp 一律视为噪声）:")
    print(piv.agg(["mean", "std", "min", "max"]).T.round(2).to_string())

    print("\n=== IS/OOS 可迁移性（各臂跨格秩相关） ===")
    from scipy.stats import spearmanr
    for name in arms:
        s = df[df["arm"] == name]
        r, p = spearmanr(s["is_ann"], s["oos_ann"])
        flag = "  ← 负相关：K 不可按历史最优挑" if r < 0 else ""
        print(f"  {name:<18s} ρ={r:+.3f}  p={p:.3f}  n={len(s)}{flag}")
    r, p = spearmanr(df["is_ann"], df["oos_ann"])
    print(f"  {'（全部格子）':<14s} ρ={r:+.3f}  p={p:.3f}  n={len(df)}"
          "  ← 含跨臂水平差，不是 K 的可迁移性，别单独引用")

    print("\n=== OOS（2024-2026）年化 % ===")
    print((df.pivot(index="k", columns="arm", values="oos_ann") * 100)
          .round(2).to_string())

    print("\n=== 判定用的三条读数 ===")
    stat = piv.agg(["mean", "min", "max"])
    for name in arms:
        print(f"  {name:<18s} 均值 {stat.loc['mean', name]:5.2f}%  "
              f"最差格 {stat.loc['min', name]:5.2f}%  最好格 {stat.loc['max', name]:5.2f}%")
    print(f"\n产物: {outdir / 'grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
