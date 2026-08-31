"""国君研报口径因子构造 —— 次日 VWAP 执行链收益 / 可交易性掩码。

2026-08-31 自 ``scripts/mine_factors`` 下沉（跨 scripts 复用：gtja_repro_eval /
gtja_discipline_eval / 挖掘主流程），沉淀为纯因子层构造：

- ``build_vwap_exec_returns``：次日 VWAP 执行链收益率（研报表1「调仓价格=次日
  VWAP」口径）。
- ``build_gtja_tradable``：可交易性掩码（T+1 停牌/封板 + 当日 ST/停牌 剔除）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_vwap_exec_returns(panel: dict[str, pd.DataFrame],
                            bwd: pd.DataFrame | None = None) -> pd.DataFrame:
    """次日 VWAP 执行链收益率（国君研报表1「调仓价格=次日VWAP」口径）。

    信号 T 日收盘后计算 → T+1 日 VWAP 成交建仓 → 持有至 T+2 日 VWAP 换仓，
    故 T 日因子配对的持仓收益 = vwap_adj[T+2] / vwap_adj[T+1] - 1。

    VWAP = amount / volume 为未复权每股均价，乘以后复权因子
    （``load_backward_factor``）与 returns/因子库口径对齐，消除除权跳空；
    ``bwd`` 可传入预取的复权因子（多次构建复用同一份，保证口径一致）。
    复权因子不可得时（离线/mock）退化为不调整的 VWAP 链——mock 无分红，
    真实数据 SDK 可用时会自动带复权。
    """
    close = panel["close"]
    vwap_raw = panel["amount"] / panel["volume"]
    if bwd is None:
        try:
            from data.cache import DataCache
            from data.cache_helpers import load_backward_factor
            from data.datasource import create_datasource
            bwd = load_backward_factor(DataCache(create_datasource()), list(close.columns))
        except Exception as exc:
            bwd = None
            log.warning("后复权 VWAP 构建失败，退化用未复权 VWAP 链（mock 场景正常）: %s", exc)
    if bwd is not None and len(bwd):
        bwd_al = bwd.reindex(index=close.index, columns=close.columns).ffill()
        vwap_adj = vwap_raw * bwd_al
        log.info("VWAP 执行链收益率：已按后复权因子对齐")
    else:
        vwap_adj = vwap_raw
    rets = vwap_adj.shift(-2) / vwap_adj.shift(-1) - 1.0
    return rets.replace([np.inf, -np.inf], np.nan)


def build_gtja_tradable(panel_full: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """gtja 真实模式构建可交易性掩码（T+1 停牌/封板 + 当日 ST/停牌 剔除）。

    复用 VWAP 链的后复权因子（封板判定需要未复权收盘价）。
    状态表缺失 → 返回 None（适应度退化为不过滤，日志可见）。
    """
    try:
        from data.cache import DataCache
        from data.cache_helpers import build_tradable_mask, load_backward_factor
        from data.datasource import create_datasource
        close = panel_full["close"]
        bwd = load_backward_factor(DataCache(create_datasource()), list(close.columns))
        bwd_al = bwd.reindex(index=close.index, columns=close.columns).ffill() if len(bwd) else None
        mask = build_tradable_mask(close, bwd=bwd_al)
        frac = float(mask.values.mean())
        log.info("可交易性掩码：平均可交易比例 %.2f（False=剔除 T+1 停牌/封板、当日 ST/停牌）", frac)
        if frac > 0.999:
            log.warning("掩码几乎全 True——状态表可能缺失，请先跑 scripts/fetch_status_batched.py")
        return mask
    except Exception as exc:
        log.warning("可交易性掩码构建失败（适应度将不过滤）: %s", exc)
        return None
