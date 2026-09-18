"""多因子10 T 扫描：合成权重回看窗 T 的参数敏感性实验（研报 §2.1 口径）。

华泰《多因子合成方法实证分析》（多因子系列 10）的参数敏感性扫描：
``T ∈ {3, 6, 9, 12, 24, 36} 个月``——T 是**估计历史 IC（及其协方差）的回看窗长**。
2026-09-16 补齐最大化 IC_IR / 最大化 IC / 半衰加权时把"切窗口"留给调用方
（见 ``factor.synthesis.synthesize_ic_ir_max`` docstring 复现边界第 3 条），
本脚本就是那个调用方：对每个 T 切训练窗 → 合成 → **测试段**评估 OOS RankIC。

用法（全量跑，成本与 e2e 同量级，预算见 RESEARCH_TODO 待办）::

    python -m scripts.evaluation.mf10_t_scan --dataset hs300_2022_2025 --top 8
    python -m scripts.evaluation.mf10_t_scan --dataset hs300_2022_2025 \\
        --features gp_im_1,im_mom_20 --method ic_max --horizon 10

核心函数 :func:`t_scan_composite` 与数据装载解耦，可直接喂合成面板（单测用）。

防未来函数的两处边界（本项目口径，研报未讨论）：
1. IC / Σ 只用 ``train_dates`` 段估计（synthesis 层已保证）；
2. **标签实现期不跨决策点**：date 日的标签是 d+1→d+h 的未来收益，其信息在
   d+h 才落地。故训练窗再截掉末端 ``horizon`` 个交易日——否则训练窗尾部
   标签的实现落在测试段内（h=10 时是 10 天的未来函数，量级不可忽略）。
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

log = logging.getLogger("mf10_t_scan")

#: 研报 §2.1 的 T 网格（月）
REPORT_T_GRID: tuple[int, ...] = (3, 6, 9, 12, 24, 36)

#: 月 → 交易日折算（研报未指明口径，取项目惯例 21）
TRADING_DAYS_PER_MONTH = 21


def _usable_train_dates(all_dates: pd.DatetimeIndex, test_begin,
                        horizon: int) -> pd.DatetimeIndex:
    """决策点前、且截掉标签实现期的可用训练日期（防未来函数边界 2）。"""
    usable = all_dates[all_dates < pd.Timestamp(test_begin)]
    if horizon > 0:
        usable = usable[:-int(horizon)] if len(usable) > horizon else usable[:0]
    return usable


def t_scan_composite(
    components: list,
    returns_panel: pd.DataFrame,
    *,
    test_begin,
    t_grid: tuple[int, ...] = REPORT_T_GRID,
    method: str = "ic_ir_max",
    horizon: int = 10,
    half_life: float | None = None,
    shrinkage: float = 0.5,
    long_only: bool | None = None,
    trading_days_per_month: int = TRADING_DAYS_PER_MONTH,
) -> pd.DataFrame:
    """对每个 T（月）切训练窗合成复合因子，返回逐 T 的 OOS 评估表。

    Args:
        components: 参与合成的因子（``factor.synthesis.CompositeInput`` 列表）。
            ``ic`` 字段仅用于符号对齐（见 synthesize_ic_ir_max 的符号口径说明）。
        returns_panel: 未来收益面板（date × code；date 日的值 = d+1→d+h 收益）。
        test_begin: 测试段起点（含）。训练窗只能取此前的日期。
        t_grid: 回看窗长网格（月）。
        method: ``ic_ir_max``（研报主力，默认 ``long_only=True``）或
            ``ic_max``（默认 ``long_only=False``，研报只对 IC_IR 要求 w≥0）。
        horizon: 标签实现期（交易日）。训练窗末端截掉这么多天，
            防止训练标签的实现落在测试段（见模块 docstring 边界 2）。
        half_life / shrinkage: 透传给合成函数（None = 不做半衰加权）。
        long_only: 缺省按 method 惯例；显式给出则覆盖。

    Returns:
        DataFrame(index=T_months)，列：
        ``valid / n_train_days / n_ic_periods / cond / train_begin / train_end /
        oos_ic_mean / oos_ic_ir / oos_t_nw / oos_n_days / weights``（JSON 串）。
        ``valid=False`` 表示该 T 超出可用历史——研报 T=36 个月 × 21 天 ≈ 756
        个交易日，2022 年起的数据面跑不动大 T，须如实呈现而非报错。
    """
    from factor.synthesis import synthesize_ic_ir_max, synthesize_ic_max
    from stats.ic import calc_ic_series
    from stats.robust_stats import nw_tstat

    if not components:
        raise ValueError("components 为空")
    if method not in ("ic_ir_max", "ic_max"):
        raise ValueError(f"未知 method {method!r}（ic_ir_max / ic_max）")

    # 统一网格：因子面板与收益面板的公共 (date, code)
    idx = components[0].panel.index
    cols = components[0].panel.columns
    for c in components[1:]:
        idx = idx.intersection(c.panel.index)
        cols = cols.intersection(c.panel.columns)
    idx = idx.intersection(returns_panel.index).sort_values()
    cols = cols.intersection(returns_panel.columns)

    usable = _usable_train_dates(idx, test_begin, horizon)
    test_dates = idx[idx >= pd.Timestamp(test_begin)]
    if len(test_dates) == 0:
        raise ValueError("测试段为空：test_begin 晚于面板末端")

    if long_only is None:
        long_only = (method == "ic_ir_max")

    synth = synthesize_ic_ir_max if method == "ic_ir_max" else synthesize_ic_max
    rows: list[dict] = []
    for t_months in t_grid:
        window = int(round(t_months * trading_days_per_month))
        row: dict = {"T_months": int(t_months), "valid": False,
                     "n_train_days": 0, "n_ic_periods": 0,
                     "cond": np.nan, "train_begin": pd.NaT, "train_end": pd.NaT,
                     "oos_ic_mean": np.nan, "oos_ic_ir": np.nan,
                     "oos_t_nw": np.nan, "oos_n_days": 0, "weights": "{}"}
        if len(usable) < window:
            rows.append(row)                    # 历史不足：如实记 invalid
            continue
        train_dates = usable[-window:]
        comps_view = [
            type(c)(name=c.name,
                    panel=c.panel.reindex(index=idx, columns=cols),
                    ic=c.ic, ir=c.ir)
            for c in components
        ]
        rts = returns_panel.reindex(index=idx, columns=cols)
        if method == "ic_ir_max":
            _composite, diag = synth(comps_view, rts, train_dates,
                                     long_only=long_only, half_life=half_life,
                                     shrinkage=shrinkage,
                                     returns_diagnostics=True)
        else:
            _composite, diag = synth(comps_view, rts, train_dates,
                                     long_only=long_only,
                                     returns_diagnostics=True)
        composite = _composite  # 面板为全样本加权（权重只由训练段决定）
        ic_oos = calc_ic_series(composite.loc[test_dates],
                                rts.loc[test_dates]).dropna()
        t_nw, _se, _lag = nw_tstat(ic_oos.to_numpy(dtype=float))
        std = float(ic_oos.std(ddof=1)) if len(ic_oos) > 1 else np.nan
        row.update({
            "valid": True,
            "n_train_days": int(len(train_dates)),
            "n_ic_periods": int(diag.get("n_periods", 0)),
            "cond": float(diag.get("cond", np.nan)),
            "train_begin": pd.Timestamp(train_dates[0]),
            "train_end": pd.Timestamp(train_dates[-1]),
            "oos_ic_mean": float(ic_oos.mean()) if len(ic_oos) else np.nan,
            "oos_ic_ir": float(ic_oos.mean() / std)
                if std is not None and np.isfinite(std) and std > 0 else np.nan,
            "oos_t_nw": float(t_nw),
            "oos_n_days": int(len(ic_oos)),
            "weights": json.dumps(diag.get("weights", {}), ensure_ascii=False),
        })
        rows.append(row)
    return pd.DataFrame(rows).set_index("T_months")


def _load_real_data(dataset: str, top: int, features: str | None, horizon: int,
                    test_begin):
    """因子库面板 + 前瞻收益面板 + 各因子训练段 IC（CLI 用；单测喂合成面板）。

    Returns:
        ``(names, panels, returns, ics)``——``ics`` 只用于 CompositeInput 的
        符号对齐（合成函数内部会按各自窗口重算 IC 矩阵）。
    """
    from research.factor_library import FactorLibrary
    from scripts.common.e2e_common import load_library_grid_panels
    from stats.ic import calc_ic_series

    lib = FactorLibrary(dataset=dataset)
    feats = lib.load_library_features()
    if not feats:
        raise SystemExit(f"因子库 {dataset!r} 无可加载面板")
    reg = lib.list_all()
    if features:
        names = [s.strip() for s in features.split(",") if s.strip()]
        missing = [n for n in names if n not in feats]
        if missing:
            raise SystemExit(f"库内不存在: {missing}")
    else:
        inlib = list(feats.keys())
        tcol = "t_nw" if "t_nw" in reg.columns else None
        if tcol is not None:
            sub = reg.loc[reg.index.intersection(inlib)]
            names = list(
                sub.reindex(sub[tcol].abs().sort_values(ascending=False).index)
                .index[:top])
        else:
            names = inlib[:top]

    px = load_library_grid_panels(dataset)
    close = px["close"]
    returns = close.shift(-horizon) / close - 1

    usable = _usable_train_dates(returns.index, test_begin, horizon)
    ics: dict[str, float] = {}
    for n in names:
        p = feats[n].reindex(index=usable, columns=returns.columns)
        ics[n] = float(calc_ic_series(p, returns.loc[usable]).mean())
    return names, {n: feats[n] for n in names}, returns, ics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="hs300_2022_2025")
    ap.add_argument("--top", type=int, default=8,
                    help="未显式给 --features 时按注册表 |t_nw| 取前 N 个因子")
    ap.add_argument("--features", default=None,
                    help="逗号分隔的因子名；缺省按 |t_nw| 自动选 --top 个")
    ap.add_argument("--method", default="ic_ir_max", choices=["ic_ir_max", "ic_max"])
    ap.add_argument("--horizon", type=int, default=10,
                    help="标签前瞻期（交易日），同时用作防未来函数截断")
    ap.add_argument("--half-life", type=float, default=None)
    ap.add_argument("--test-begin", default="2025-01-01")
    ap.add_argument("--out", default=None, help="CSV 输出路径（缺省 reports/mf10_t_scan/）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from factor.synthesis import CompositeInput, standardize_zscore

    names, panels, returns, ics = _load_real_data(
        args.dataset, args.top, args.features, args.horizon, args.test_begin)
    log.info("因子 %d 个: %s | horizon=%d | 训练段 IC=%s",
             len(names), names, args.horizon,
             {n: round(v, 4) for n, v in ics.items()})
    comps = [CompositeInput(name=n, panel=standardize_zscore(panels[n]), ic=ics[n])
             for n in names]
    res = t_scan_composite(
        comps, returns, test_begin=args.test_begin, method=args.method,
        horizon=args.horizon, half_life=args.half_life,
    )
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(res.drop(columns=["weights"]).to_string())
    out = Path(args.out) if args.out else \
        ROOT / "reports" / "mf10_t_scan" / f"{args.dataset}_{args.method}_h{args.horizon}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out)
    log.info("已写出 %s", out)


if __name__ == "__main__":
    main()
