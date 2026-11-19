"""因子检验统一入口 —— 一条命令跑完 L1~L5 并输出标准表（2026-11-18 建立）。

## 用法

    # 跑一个数据集下的指定因子（逗号或空格分隔）
    python -m scripts.evaluation.factor_gate \\
        --ds-root reports/cyq_probe_2026-11-18/all_a_2018_2026 \\
        --panels-dir panels_neu --only cyq_benefit cyq_conc90 cyq_trap \\
        --out reports/cyq_probe_2026-11-18/gate

    # 整个目录（只跑 registry 里登记过的）
    python -m scripts.evaluation.factor_gate --ds-root <ds> --panels-dir panels_neu

## 输出

``<out>/gate_h{h}.csv`` 每行一条因子，列分为五组：

| 组 | 列 | 含义 |
|---|---|---|
| L1 | ``cov`` ``cov_eff`` ``n_valid`` ``n_days`` | 覆盖 |
| L2 | ``raw_ic`` ``raw_ir`` ``raw_win`` ``raw_nw_t`` | 原始预测力 |
| L3 | ``resfac_ic`` ``resfac_nw_t`` ``restgt_*`` ``both_*`` | 正交主信号后（主判据 = ``resfac``） |
| L4 | ``rank_corr_main`` | 与主信号共线度 |
| L5 | ``plc_ic`` ``plc_nw_t`` ``eff_ratio`` ``eff_gap`` | 循环平移安慰剂与效应量 |
| 判定 | ``verdict`` ``why`` | 筛选标记（**不是结论**） |

## 🚨 读这张表的顺序

1. 先看 ``verdict``。``DATA`` / ``NOIC`` / ``ABSORBED`` / ``NOISE`` 直接否。
2. ``PASS`` 只代表"值得进组合层"。**IC ≠ 经济价值** —— 必须再跑 Top-N
   （``scripts/evaluation/cyq_panel.py topn`` 是这条的范例）。
3. ``resfac_nw_t`` **不要单独引用**：该样本上安慰剂也能拿到 t≈4（见
   :func:`scripts.evaluation.gate_core.verdict` 的说明）。要引就引
   ``eff_ratio`` + ``eff_gap``。
4. 跨 horizon 不可并列 —— IC 的期限必须等于持有期，报数必须带 ``h``。
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluation import gate_core as gc

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("factor_gate")

SHOW = ["name", "cov", "raw_ic", "raw_nw_t", "resfac_ic", "resfac_nw_t",
        "rank_corr_main", "plc_ic", "plc_nw_t", "eff_ratio", "eff_gap",
        "years_pos", "verdict"]


def _resolve_ds_root(explicit: str | None) -> Path:
    """定位数据集根（含 ``panels*`` 与 ``registry.csv``）。"""
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"数据集根不存在：{p}")
        return p
    cand = sorted((ROOT / "reports").glob("*/all_a_2018_2026"))
    if not cand:
        raise FileNotFoundError("未找到数据集根，请显式给 --ds-root")
    return cand[0]


def _select_files(ds: Path, pdir: Path, only: list[str] | None,
                  sets: str | None, max_n: int | None) -> list[Path]:
    files = sorted(pdir.glob("*.parquet"))
    if only:
        wanted = set(only)
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            log.warning("指定因子不存在：%s", sorted(missing))
        return files[: max_n] if max_n else files

    reg_p = ds / "registry.csv"
    if reg_p.exists():
        reg = pd.read_csv(reg_p)
        registered = set(reg["name"].astype(str))
        files = [p for p in files if p.stem in registered]
        if sets:
            allowed = {s.strip() for s in sets.split(",") if s.strip()}
            set_by = dict(zip(reg["name"].astype(str), reg["set"].astype(str)))
            files = [p for p in files if set_by.get(p.stem) in allowed]
    return files[: max_n] if max_n else files


def evaluate_one(name: str, x: pd.DataFrame, ctx: dict, seed: int,
                 th: dict | None = None) -> dict | None:
    """单因子 L1~L5，返回一行记录。"""
    fwd, mask, main = ctx["fwd"], ctx["mask"], ctx["main"]
    row: dict = {"name": name, "horizon": ctx["horizon"],
                 "main_signal": ctx["main_signal"]}
    row.update(gc.panel_coverage(x, fwd, mask))

    ic = gc.ic_calibers(x, fwd, mask, main)
    if ic is None:
        row.update({"verdict": "DATA", "why": "有效格不足 1000，无法算 IC"})
        return row
    row.update(ic)

    # L5 安慰剂：同一套 IC 口径，只把因子的时点对应打掉
    plc = gc.cyclic_placebo(x, seed=seed, name=name)
    ic_plc = gc.ic_calibers(plc, fwd, mask, main)
    if ic_plc is None:
        row.update({"plc_ic": np.nan, "plc_nw_t": np.nan})
    else:
        for k in ("ic", "nw_t", "win"):
            row[f"plc_{k}"] = ic_plc.get(f"resfac_{k}", np.nan)
        row["plc_years_pos"] = ic_plc.get("years_pos", np.nan)
    row.update(gc.effect_size(row.get("resfac_ic", np.nan),
                              row.get("plc_ic", np.nan)))
    tag, why = gc.verdict(row, th)
    row.update({"verdict": tag, "why": why})
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="因子检验统一入口（L1~L5，见模块 docstring）")
    ap.add_argument("--ds-root", default=None, help="数据集根（含 panels*/registry.csv）")
    ap.add_argument("--panels-dir", default="panels_neu",
                    help="面板子目录：panels_neu（中性化，默认）/ panels（raw）")
    ap.add_argument("--pred-dir", default="reports/alla_rolling_ortho/pred")
    ap.add_argument("--main-signal", default="ens_h1h5__h1.parquet",
                    help="主信号文件（残差口径的正交对象）")
    ap.add_argument("--horizon", type=int, default=1,
                    help="IC 期限（天）；🚨 必须等于持有期，跨期限不可并列")
    ap.add_argument("--window", default="prod", help="窗口（默认 prod = 2019-01 起）")
    ap.add_argument("--end", type=int, default=None, help="末日 YYYYMMDD（对齐用）")
    ap.add_argument("--only", nargs="+", default=None, help="只跑这些因子名")
    ap.add_argument("--sets", default=None, help="registry set 过滤（逗号分隔）")
    ap.add_argument("--max-factors", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261118, help="安慰剂平移种子")
    ap.add_argument("--out", default=None, help="输出目录（缺省不落盘，只打印）")
    ap.add_argument("--out-tag", default="", help="输出文件名后缀")
    ap.add_argument("--min-cov", type=float, default=gc.DEFAULTS["min_cov"])
    ap.add_argument("--ic-floor", type=float, default=gc.DEFAULTS["ic_floor"])
    ap.add_argument("--gap-floor", type=float, default=gc.DEFAULTS["gap_floor"])
    ap.add_argument("--eff-ratio", type=float, default=gc.DEFAULTS["eff_ratio"])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ds = _resolve_ds_root(args.ds_root)
    pdir = ds / args.panels_dir
    if not pdir.exists():
        raise FileNotFoundError(f"面板目录不存在：{pdir}")

    # 窗口从 caliber 取（唯一真源），避免本脚本自带第二份起点定义
    from scripts.evaluation import caliber as cal
    begin = cal.WINDOWS[args.window]

    ctx = gc.load_ic_context(horizon=args.horizon, window=args.window,
                             pred_dir=args.pred_dir,
                             main_signal=args.main_signal, end=args.end,
                             begin=begin)
    log.info("数据集 %s | 面板 %s", ds.name, pdir.name)
    log.info("评估轴 %s ~ %s（%d 日）| h=%d | 主信号 %s | 窗口 %s%s",
             ctx["idx"][0].date(), ctx["idx"][-1].date(), len(ctx["idx"]),
             args.horizon, ctx["main_signal"], args.window,
             f"（begin={begin}）" if begin else "")

    files = _select_files(ds, pdir, args.only, args.sets, args.max_factors)
    if not files:
        log.error("没有可评估的面板")
        return 1
    log.info("待评估 %d 个因子", len(files))

    th = {"min_cov": args.min_cov, "ic_floor": args.ic_floor,
          "gap_floor": args.gap_floor, "eff_ratio": args.eff_ratio}

    rows, t0 = [], time.time()
    for i, fp in enumerate(files, 1):
        x = gc.load_panel(fp, ctx["idx"], ctx["cols"])
        if x is None:
            log.warning("%s: 面板不可用，跳过", fp.stem)
            continue
        row = evaluate_one(fp.stem, x, ctx, args.seed, th)
        if row is not None:
            rows.append(row)
        if i % 20 == 0:
            log.info("  %d/%d | %.0fs", i, len(files), time.time() - t0)

    res = pd.DataFrame(rows)
    if res.empty:
        log.error("没有产出任何结果")
        return 1

    if args.out:
        dest = Path(args.out)
        if not dest.is_absolute():
            dest = ROOT / dest
        dest.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.out_tag}" if args.out_tag else ""
        out_csv = dest / f"gate_h{args.horizon}{suffix}.csv"
        res.to_csv(out_csv, index=False, encoding="utf-8-sig")
        log.info("落盘 %s", out_csv)

    show = [c for c in SHOW if c in res.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", lambda v: f"{v: .4f}"):
        print(res[show].to_string(index=False))
    log.info("完成 %d 个 | %.0fs", len(res), time.time() - t0)
    for tag, grp in res.groupby("verdict"):
        log.info("  %-9s %d 个：%s", tag, len(grp),
                 ", ".join(grp["name"].astype(str).head(6)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
