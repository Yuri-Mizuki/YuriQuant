"""
每日交易信号生成（P3，2026-08-20）
==================================

把多期组合执行产出的**目标权重**（optimize.multi_period）转成带约束校验的
**每日可执行交易信号**：目标持仓 → 目标股数（整手化）→ 方向 → 交易约束校验
（涨停不可买 / 跌停不可卖 / 停牌冻结 / 不足一手）。

职责边界（不接实盘）：
- 本模块只把"理想权重"转成"可下单信号"，不维护成交账本、不滑点撮合。
- 多头腿用【未复权】收盘价换算股数（后复权价不可用于下单）。
- 目标权重可含空头（负值），股数与方向均带符号处理。

链路：
    target_weights(调仓日×code) ──价格/可交易掩码──▶ signal frame(每日×code)
                                  │
                                  └─ 每行: 目标股数 / 整手 / 方向 / 受阻标记

用法：
    from optimize.signals import build_signal_frame, signals_from_rebalances
    frame = signals_from_rebalances(target_weights, close_raw, buyable=b, sellable=s,
                                    capital=1e8, lot=100)
export 后即得到下一要执行的信号清单。
"""
from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["build_signal_frame", "signals_from_rebalances"]


class StockNotTradeable(RuntimeError):
    """股票无可交易记录或价格缺失（不该进信号）。"""


def build_signal_frame(
    signal_date,
    target_weights: pd.Series,
    price: pd.Series,
    capital: float,
    tradable: pd.Series | None = None,
    buyable: pd.Series | None = None,
    sellable: pd.Series | None = None,
    current_positions: pd.Series | None = None,
    lot: int = 100,
) -> pd.DataFrame:
    """把某个调仓日的目标权重转成当日交易信号。

    Args:
        signal_date: 信号生效日（调仓日），写入每行 signal_date。
        target_weights: code → 目标权重（和≈1；空头为负）。
        price: code → 参考价（**未复权**收盘价）。
        capital: 组合总资金，用于股数换算（target_weight × capital / price）。
        tradable: code → bool，可选。False 表示该股当日不可交易（禁买也禁卖）。
        buyable / sellable: code → bool，可选。精细方向掩码：涨停禁买(buyable=False)、
            跌停禁卖(sellable=False)。优先级高于 tradable。
        current_positions: code → 当前持仓股数，可选。提供时输出净变化单
            （目标持仓 − 当前持仓）；缺省视为从 0 建仓/全换。
        lot: 最小交易单位（A 股 100 股/手），成交股数向零取整。

    Returns:
        DataFrame，每行一只股票的信号，按 |目标股数| 降序：
        signal_date, code, target_weight, price, current_position, qty_target,
        qty_delta, qty_order, direction, status, note
        - direction: BUY / SELL / HOLD / FLAT
        - status:  OK / BLOCKED_BUY / BLOCKED_SELL / NO_PRICE
        - qty_order 已整手化；受阻或不足一手时可能为 0。
    """
    records: list[dict[str, Any]] = []
    p = price.dropna()

    for code in p.index:
        w = float(target_weights.get(code, 0.0))
        px = float(p[code])
        qty_target = round(w * capital / px)
        pos = 0.0
        if current_positions is not None:
            pos = float(current_positions.get(code, 0.0))
            qty_delta = qty_target - pos
        else:
            qty_delta = qty_target

        note_parts: list[str] = []
        if qty_delta > 0:
            direction = "BUY"
        elif qty_delta < 0:
            direction = "SELL"
        elif pos != 0:
            direction = "HOLD"
        else:
            direction = "FLAT"

        status = "OK"
        can_buy = True
        can_sell = True
        if tradable is not None and not bool(tradable.get(code, True)):
            can_buy = can_sell = False
        if buyable is not None and not bool(buyable.get(code, True)):
            can_buy = False
        if sellable is not None and not bool(sellable.get(code, True)):
            can_sell = False

        qty_order = _to_lot(qty_delta, lot)
        if direction in ("BUY", "SELL"):
            blocked = (direction == "BUY" and not can_buy) or (direction == "SELL" and not can_sell)
            if blocked:
                if direction == "BUY":
                    status = "BLOCKED_BUY"
                    note_parts.append(_buy_block_reason(can_sell, can_buy))
                else:
                    status = "BLOCKED_SELL"
                    note_parts.append(_sell_block_reason(can_sell, can_buy))
                qty_order = 0
            elif qty_order == 0 and qty_delta != 0:
                note_parts.append("不足一手")

        records.append({
            "signal_date": pd.Timestamp(signal_date),
            "code": code,
            "target_weight": w,
            "price": px,
            "current_position": pos,
            "qty_target": qty_target,
            "qty_delta": qty_delta,
            "qty_order": qty_order,
            "direction": direction,
            "status": status,
            "note": "；".join(note_parts),
        })

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).sort_values("qty_target", key=abs, ascending=False)
    return frame.reset_index(drop=True)


def signals_from_rebalances(
    target_weights: pd.DataFrame,
    price: pd.DataFrame,
    capital: float,
    tradable: pd.DataFrame | None = None,
    buyable: pd.DataFrame | None = None,
    sellable: pd.DataFrame | None = None,
    current_positions: pd.DataFrame | None = None,
    lot: int = 100,
) -> pd.DataFrame:
    """逐调仓日生成信号（历史/全集回放），拼接成长表（signal_date 列）。

    各掩码/持仓面板与 target_weights 同索引（date×code）。target_weights 为空的行
    （该调仓日协方差不足）自动跳过。
    """
    frames: list[pd.DataFrame] = []
    for t in target_weights.index:
        row = target_weights.loc[t]
        if isinstance(row, pd.DataFrame):  # 单一列时 loc 可能升维
            row = row.squeeze()
        if float(row.abs().sum()) == 0.0:
            continue
        frames.append(build_signal_frame(
            t, row,
            price.loc[t] if t in price.index else price.iloc[-1],
            capital,
            tradable=tradable.loc[t] if tradable is not None else None,
            buyable=buyable.loc[t] if buyable is not None else None,
            sellable=sellable.loc[t] if sellable is not None else None,
            current_positions=current_positions.loc[t] if current_positions is not None else None,
            lot=lot,
        ))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _to_lot(qty: float, lot: int = 100) -> int:
    """向零整手化（空头也向零，保持不超买/不超卖）。"""
    import math
    return int(math.copysign(abs(qty) // lot * lot, qty))


def _buy_block_reason(can_sell: bool, can_buy: bool) -> str:
    if not can_buy and not can_sell:
        return "停牌/不可交易，买入冻结"
    return "涨停封板，买不进"


def _sell_block_reason(can_sell: bool, can_buy: bool) -> str:
    if not can_sell and not can_buy:
        return "停牌/不可交易，卖出冻结"
    return "跌停封板，卖不出"