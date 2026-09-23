"""实验产物接 PBO 出报告（TODO §二；华泰 AI19/22 方法论落地）。

把「N 个候选配置的期间收益矩阵」接入 :mod:`stats.pbo`：CSCV PBO（组合对称
切分）+ DSR（缩水夏普），作为实验出报告的**固定环节**——任何"在候选集里挑
样本内最优"的实验（参数网格 / 模型臂 / 多标签臂）都应附一份。

两种入口：
1. rolling_grid 网格模式（默认）：扫 ``--eq-dir`` 下
   ``eq__<model>__<h>__<freq>__<sig>__<frac>.csv`` 的 ``daily_ret``，
   按 (freq, sig, frac) 聚成网格，网格内 model×h 臂组成 T×N 矩阵。
   **PBO 只在同一网格内臂间有意义**——它回答"在这组候选里挑 IS 最优、
   OOS 掉队的概率"，跨网格比较无定义。
2. 通用矩阵模式：``--returns-csv`` 指向 date×N 收益 CSV（列 = 候选配置），
   单独出一份报告（AI97 三臂 / stage2 各配置等产物落地后同一入口接入）。

判读（引用时须带口径）：PBO > 0.5 → 选择过程大概率过拟合；``omega_mean``
< 0.5 同向佐证；DSR ≥ 0.95 且 PBO 低 → 挑出的最优在计入选择偏差后仍显著。
PBO/DSR **不**回答策略本身是否有效（见 stats/pbo.py 模块注释）。

跑法（系统 Python 3.12）：
  python -m scripts.evaluation.pbo_report                  # 主实验三网格
  python -m scripts.evaluation.pbo_report --grids all      # 发掘全部网格
  python -m scripts.evaluation.pbo_report --returns-csv x.csv --label myexp

产物：``--out`` 目录下 {pbo_summary.csv, pbo_detail_<grid>.json}。
2026-09-23 从 scripts/oneoff/_pbo_rolling_ortho.py 固化（算法未改，产物
与 09-22 首跑逐位一致）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats.pbo import cscv_pbo, deflate_best  # noqa: E402

log = logging.getLogger("pbo_report")

#: 主实验口径三网格（与 09-22 首跑一致；--grids all 可发掘全部）
DEFAULT_GRIDS = [
    ("M__raw__f0.10", "M", "raw", "f0.10"),
    ("M__neut__f0.10", "M", "neut", "f0.10"),
    ("W__raw__f0.10", "W", "raw", "f0.10"),
]


def discover_grids(eq_dir: Path) -> list[tuple[str, str, str]]:
    """从 eq 文件名发掘全部 (freq, sig, frac) 网格（臂数 ≥2 才算）。"""
    groups: dict[tuple[str, str, str], set[str]] = {}
    for f in sorted(Path(eq_dir).glob("eq__*.csv")):
        parts = f.stem[3:].split("__")
        if len(parts) != 5:
            continue
        key = (parts[2], parts[3], parts[4])
        groups.setdefault(key, set()).add(f"{parts[0]}__{parts[1]}")
    grids = []
    for (freq, sig, frac), arms in sorted(groups.items()):
        if len(arms) < 2:
            continue
        grids.append((f"{freq}__{sig}__{frac}", freq, sig, frac))
    return grids


def load_returns_matrix(eq_dir: Path, freq: str, sig: str, frac: str,
                        min_periods: int = 100) -> pd.DataFrame | None:
    """该网格所有臂的 daily_ret 对齐成 T×N（内连接——臂间日期须一致）。"""
    cols: dict[str, pd.Series] = {}
    for f in sorted(Path(eq_dir).glob("eq__*.csv")):
        parts = f.stem[3:].split("__")
        if len(parts) != 5 or parts[2] != freq or parts[3] != sig or parts[4] != frac:
            continue
        arm = f"{parts[0]}__{parts[1]}"
        df = pd.read_csv(f, index_col="date", parse_dates=True)
        r = df["daily_ret"].dropna()
        if len(r) < min_periods:
            log.warning("%s 收益不足 %d 期，跳过", arm, min_periods)
            continue
        cols[arm] = r
    if len(cols) < 2:
        return None
    return pd.DataFrame(cols).dropna()


def load_returns_csv(path: Path) -> pd.DataFrame:
    """通用入口：date×N 收益 CSV（首列日期，其余列 = 候选配置）。"""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] < 2:
        raise ValueError(f"{path}: 须为 date×N（N ≥ 2 个候选配置）收益矩阵")
    return df


def evaluate_matrix(mat: pd.DataFrame, grid: str, n_partitions: int = 16) -> dict:
    """T×N 收益矩阵 → PBO + DSR 报告 dict（与 09-22 oneoff 首跑同构）。"""
    res = cscv_pbo(mat, n_partitions=n_partitions, metric="sharpe")
    best = deflate_best(mat)
    return {
        "grid": grid,
        "n_trials": res["n_trials"],
        "n_periods": int(mat.shape[0]),
        "pbo": res["pbo"],
        "omega_mean": res["omega_mean"],
        "logit_median": res["logit_median"],
        "logit_iqr": res["logit_iqr"],
        "best_arm": str(mat.columns[int(np.argmax(
            [(mat[c] / mat[c].std()).mean() for c in mat.columns]))]),
        "dsr": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in best.items()},
    }


def _write_detail(out_dir: Path, detail: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"pbo_detail_{detail['grid'].replace('__', '_')}.json"
    (out_dir / fname).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")


def run_grids(eq_dir: Path, grids: list[tuple], out_dir: Path,
              n_partitions: int = 16) -> pd.DataFrame:
    rows = []
    for grid, freq, sig, frac in grids:
        mat = load_returns_matrix(eq_dir, freq, sig, frac)
        if mat is None:
            log.warning("网格 %s 臂数不足，跳过", grid)
            continue
        log.info("网格 %s: %s 期 × %s 臂", grid, *mat.shape)
        detail = evaluate_matrix(mat, grid, n_partitions)
        _write_detail(out_dir, detail)
        rows.append({k: detail[k] for k in
                     ("grid", "n_trials", "n_periods", "pbo", "omega_mean",
                      "logit_median", "best_arm")})
        log.info("  PBO=%.3f omega_mean=%.3f best=%s",
                 detail["pbo"], detail["omega_mean"], detail["best_arm"])
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eq-dir", default=None,
                    help="rolling_grid equity 目录（缺省 reports/alla_rolling_ortho/equity）")
    ap.add_argument("--grids", default=None,
                    help="逗号分隔网格名；'all' = 发掘全部；缺省 = 主实验三网格")
    ap.add_argument("--returns-csv", default=None,
                    help="通用模式：date×N 收益 CSV（与 --eq-dir 互斥）")
    ap.add_argument("--label", default=None, help="通用模式的报告名（缺省取文件名主干）")
    ap.add_argument("--n-partitions", type=int, default=16,
                    help="CSCV 块数 S（偶数，默认 16；T 小时降到 8 或 6）")
    ap.add_argument("--out", default=None, help="输出目录（缺省 reports/pbo_<来源>/）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.returns_csv is None and (args.eq_dir is None or args.eq_dir == "default"):
        pass  # 网格模式（缺省走 rolling_grid 主实验）
    elif args.returns_csv is not None and args.eq_dir is not None:
        ap.error("--eq-dir 与 --returns-csv 二选一")

    if args.returns_csv is not None:
        path = Path(args.returns_csv)
        label = args.label or path.stem.strip("_")
        mat = load_returns_csv(path)
        out_dir = Path(args.out) if args.out else ROOT / "reports" / f"pbo_{label}"
        log.info("矩阵 %s: %s 期 × %s 臂", label, *mat.shape)
        detail = evaluate_matrix(mat, label, args.n_partitions)
        _write_detail(out_dir, detail)
        summary = pd.DataFrame([{k: detail[k] for k in
                                 ("grid", "n_trials", "n_periods", "pbo",
                                  "omega_mean", "logit_median", "best_arm")}])
    else:
        eq_dir = (Path(args.eq_dir) if args.eq_dir
                  else ROOT / "reports" / "alla_rolling_ortho" / "equity")
        out_dir = Path(args.out) if args.out else ROOT / "reports" / "pbo_rolling_ortho"
        if args.grids is None:
            grids = DEFAULT_GRIDS
        elif args.grids.strip().lower() == "all":
            grids = discover_grids(eq_dir)
            log.info("发掘到 %d 个网格", len(grids))
        else:
            want = [g.strip() for g in args.grids.split(",") if g.strip()]
            all_grids = discover_grids(eq_dir)
            grids = [g for g in all_grids if g[0] in want]
            missing = set(want) - {g[0] for g in grids}
            if missing:
                log.warning("未找到网格: %s", sorted(missing))
        summary = run_grids(eq_dir, grids, out_dir, args.n_partitions)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "pbo_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        log.info("\n%s", summary.to_string(index=False))
    log.info("产物: %s", out_dir)


if __name__ == "__main__":
    main()
