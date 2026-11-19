"""因子检验的**数据装载层** + 判据兼容转出口（2026-11-19 重构）。

## 本模块现在负责什么

1. **装载**：把面板 / 主信号 / 前瞻收益 / 掩码对齐成统一轴 ——
   :func:`load_ic_context` / :func:`load_panel`。这是唯一离不开 ``scripts``
   的原因（要读 ``scripts.common.alla_common`` 的基础数据）。
2. **兼容转出口**：判据函数全部 re-export 自 :mod:`research.factor_criteria`，
   本文件**不再持任何判据逻辑或阈值**。

## 为什么判据搬走了

此前判据写在 scripts 层，导致 ``research.factor_library`` 要反向 import scripts
（已在两处函数级这么做了，属分层倒挂）；同时 ``MATURITY_GATES`` / ``DEFAULTS`` /
各探针 argparse 三处各有一套阈值。2026-11-19 把判据收口到 research 层后，
三段流水线（准入 / 晋升 / 组合）共用一份实现。

🚨 **禁止在本文件新增判据函数或阈值** —— 那是第二份实现。
守卫断言见 ``tests/test_factor_criteria.py``：
``gate_core.ic_calibers is factor_criteria.ic_calibers``。

## 口径

``prod`` 窗口（2019-01 起）、日频 h 期前瞻收益、生产掩码。
🚨 报数必须带 horizon —— IC 的期限必须等于持有期，跨期限 IC 不可并列。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.factor_criteria import (  # noqa: F401  (re-export)
    ADMISSION_GATES,
    CORR_TRIGGER,
    MIN_SPAN,
    REL_STD_DEGENERATE,
    REL_STD_TINY,
    RESID_IC_NEGLIGIBLE,
    Verdict,
    admission_verdict,
    all_thresholds,
    classify_independent,
    cyclic_placebo,
    dup_classify,
    dup_explain,
    effect_size,
    ic_calibers,
    ic_stats,
    panel_coverage,
    promotion_gates,
    qualifier,
    rank_corr,
    resid_rank,
    seg_stability,
    should_check,
    tier_of,
    verdict,
    verdict_admission,
    yearly_ic,
)

#: 兼容别名：旧调用方写 ``gc.DEFAULTS["min_cov"]``。真源是
#: :data:`research.factor_criteria.ADMISSION_GATES`（同一对象，不是副本）。
DEFAULTS = ADMISSION_GATES


# ---------------------------------------------------------------------------
# L0 数据装载（本模块唯一的自有逻辑）
# ---------------------------------------------------------------------------
def load_ic_context(horizon: int = 1, window: str = "prod",
                    pred_dir: str = "reports/alla_rolling_ortho/pred",
                    main_signal: str = "ens_h1h5__h1.parquet",
                    end: int | None = None,
                    begin: int | str | None = None,
                    stride: int = 1) -> dict:
    """装载 IC 检验所需的全部面板（与 ``residual_ic_probe`` 同一口径）。

    返回 ``{idx, cols, fwd, mask, main, main_rank}``。

    - ``fwd``：h 期前瞻收益 ``close.pct_change(h).shift(-h)``，即 t 收盘 → t+h 收盘。
    - ``mask``：生产可交易掩码（涨跌停 / 停牌 / ST），NaN 按不可交易。
    - ``idx``：**主信号**的日期轴（OOS 区间），不是数据轴 —— IC 只在有预测的日子算。
      ``begin`` / ``end``（``YYYYMMDD``）从这里再截一刀，起点缺省取自
      ``caliber.WINDOWS``，保证与组合层同窗口。
    """
    from scripts.common.alla_common import load_base

    base = load_base()
    close = base["close"]
    mask_all = base["mask"].astype(bool)

    sig_path = Path(main_signal)
    main = pd.read_parquet(sig_path if sig_path.is_absolute()
                           else Path(pred_dir) / main_signal)
    main = main.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    idx = main.index
    if begin is not None:
        lo = pd.to_datetime(str(int(begin)), format="%Y%m%d")
        idx = idx[idx >= lo]
    if end is not None:
        cut = pd.to_datetime(str(int(end)), format="%Y%m%d")
        idx = idx[idx <= cut]
    if stride > 1:
        idx = idx[:: stride]
    cols = main.columns.intersection(close.columns)

    fwd = (close.pct_change(horizon, fill_method=None)
           .shift(-horizon)
           .reindex(index=idx, columns=cols)
           .replace([np.inf, -np.inf], np.nan)
           .astype(np.float32))
    main_c = main.reindex(index=idx, columns=cols)
    return {
        "idx": idx,
        "cols": cols,
        "fwd": fwd,
        "mask": mask_all.reindex(index=idx, columns=cols).fillna(False).astype(bool),
        "main": main_c,
        "main_rank": main_c.rank(axis=1, pct=True),
        "main_signal": main_signal,
        "horizon": int(horizon),
        "window": window,
    }


def load_panel(path, index: pd.DatetimeIndex,
               columns: pd.Index) -> pd.DataFrame | None:
    """读一张因子面板并对齐；非数值 / 空面板返回 None。"""
    raw = pd.read_parquet(path)
    if not isinstance(raw.index, pd.DatetimeIndex):
        try:
            raw.index = pd.to_datetime(raw.index)
        except Exception:
            return None
    if raw.columns.has_duplicates:
        raw = raw.loc[:, ~raw.columns.duplicated()]
    raw = raw.select_dtypes(include=[np.number])
    if raw.shape[1] == 0:
        return None
    x = raw.reindex(index=index, columns=columns)
    return x.replace([np.inf, -np.inf], np.nan).astype(np.float32)


__all__ = [
    "ADMISSION_GATES", "CORR_TRIGGER", "DEFAULTS", "MIN_SPAN",
    "REL_STD_DEGENERATE", "REL_STD_TINY", "RESID_IC_NEGLIGIBLE", "Verdict",
    "admission_verdict", "all_thresholds", "classify_independent",
    "cyclic_placebo", "dup_classify", "dup_explain", "effect_size",
    "ic_calibers", "ic_stats", "load_ic_context", "load_panel",
    "panel_coverage", "promotion_gates", "qualifier", "rank_corr",
    "resid_rank", "seg_stability", "should_check", "tier_of", "verdict",
    "verdict_admission", "yearly_ic",
]
