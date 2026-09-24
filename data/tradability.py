"""
可交易性 / 可执行性口径 —— 单一模块
====================================

把"这只股票这天能不能被买卖"这件事集中建模，避免向量化回测悄悄假设
涨停能买、跌停能卖、停牌能成交、ST 能持仓。

本项目存在**两种口径**，差别只在"状态取哪一天"，混用会让结论不可比。
选哪个由「**成交发生在哪一天**」决定，不由调用方喜好决定：

+---------------------------+------------------+------------------------------+
| 函数                       | 状态取样日        | 适用成交日 / 现有调用方        |
+===========================+==================+==============================+
| ``build_tradable_mask``    | **T+1 日状态**    | 成交在 T+1：信号 T 日收盘产生 |
| （T+1 成交口径）            | （前瞻一天）      | → T+1 开盘/VWAP 执行。        |
|                           |                  | ``factor/gtja``、GTJA 两个     |
|                           |                  | 复现脚本、``genetic_mining``   |
|                           |                  | 的 GP 适应度                   |
+---------------------------+------------------+------------------------------+
| ``build_directional_masks``| **T 日状态**      | 信号发布时 T+1 未知，只能用   |
| （信号日诊断口径）          | （当日已知）      | T 日状态保守标注"想买但涨停 / |
|                           |                  | 想卖但跌停"；``generate_signals`` |
+---------------------------+------------------+------------------------------+

**为什么必须有 T 日那一套**：每日选股排名要在盘后立刻发布，此时 T+1 的停牌/封板
还不知道，把 T+1 口径套到发布环节等于用未来信息。

🚨 **掩码取样日必须与回测成交日一致**（2026-09-21 实测修订）。本表格原先只写
"回测用 T+1 口径"，**未绑定成交日假设**，是隐患来源：

  - 成交在 **T+1**（次日开盘 / VWAP）→ T+1 掩码**正确**，剔除的正是"次日买不进"。
  - 成交在 **T 日收盘**（信号与成交同日）→ T+1 掩码**不自洽**：它判的是次日能不能
    买，而成交已经发生在当日。此时它同时产生两种方向相反的偏差 ——
      · **过剔**：剔掉「T 日可买、T+1 涨停」的票（真实收益，OOS 占 0.92% 格子）；
      · **漏剔**：放行「T 日封板买不进、T+1 打开」的票（纸面，同样 0.92% 格子）。
    净效应实测（`reports/docs/consistency_checks/口径核对_掩码取样日错配.md`）：
    月频在噪声量级（4 信号 2 正 2 负，均值 +0.29pp）；**周频 4/4 一致为正（+3.76pp）**
    ⇒ 换手越高代价越大。

  - ⚠️ 现行回测（``rolling_grid_alla.py``）正是「**T 日收盘成交 + T+1 掩码**」这一
    混合口径：成交取 ``close[T]``（h=1 收益面板 ``close.pct_change()`` 未 shift），
    掩码取 T+1 状态。修正方向**二选一、不可只改一半**：
      ① 掩码改 T 日（自洽于 T 日成交，但仍不可执行）；
      ② 成交改 T+1（自洽且可执行，2026-09-15 实测代价 −0.94pp 量级）。

**历史沿革（2026-09-10 收敛）**：此前 ``build_tradable_mask`` 住在
``data/cache_helpers.py``，与 ``data/tradability.py`` 里的另一套并存——同一件事
两个模块、两种口径，跨模块阅读时极易错认（GP 评判因子的口径与最终回测不一致这类
问题就出在这里）。现已全部收敛到本模块：**新增可交易性逻辑一律加在这里**，
不要再在 ``cache_helpers`` 或脚本内联一份。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Config


# ---------------------------------------------------------------------------
# 口径一：T+1 成交口径（回测 / 适应度标准）
# ---------------------------------------------------------------------------

def build_tradable_mask(close_adj: pd.DataFrame,
                        bwd: pd.DataFrame | None = None,
                        cache_root: str | None = None,
                        execution_lag: int = 1) -> pd.DataFrame:
    """构建**可交易性掩码**（bool 宽表，True=该日信号可进入组合）。

    口径：信号 T 日收盘产生、**T+1 日** VWAP 成交（国君研报执行链口径）。
    ``execution_lag`` 控制状态取哪一天（2026-09-24 补，供"信号与成交同日"
    的调用方使用，如 stage2 的 :mod:`scripts.factors.run_portfolio_phase2`
    在决策日收盘调仓）：

    - ``1``（默认）：T+1 状态——现行回测/适应度口径，零回归；
    - ``0``：T 日状态——成交发生在 T 日收盘时，封板/停牌/ST 看 T 日本身。

    ⚠️ **调用方前置条件**：本掩码只在「成交日 = T+1」时成立。若调用方的回测在
    **T 日收盘**成交（信号与成交同日），掩码必须改用 T 日状态 —— 否则口径不自洽
    （实测影响见模块 docstring 的 🚨 段）。

    剔除规则（对齐华泰单因子测试口径 + 涨跌停一字板可成交性）：

    - T+1 日**停牌**（is_suspended）：无法成交 → 剔除；
    - T+1 日**收盘封涨停/封跌停**（raw_close ≥ high_limited / ≤ low_limited）：
      VWAP 处买不进/平不掉 → 剔除（用收盘封板而非盘中触板，保守适中）;
    - is_st / is_suspended 在 T 日或 T+1 日任一为真：ST 与停牌是持续状态，
      成交日（T+1）处于该状态即不可交易。

    封板判定需要**未复权收盘价**与状态表中的原始涨跌停价比较：
    ``raw_close = close_adj / bwd``（bwd=累积后复权因子，None 时退化为
    close_adj 本身——仅当面板未复权或无除权差异时可用，真实复权数据必须传）。

    状态表缺失的 (date, code) 视为可交易（保守补 True）；返回前 reindex 到
    ``close_adj`` 的行列。状态表不存在时返回全 True 掩码（调用方日志可见）。

    Args:
        close_adj: (date, code) 复权收盘价面板，行列即输出掩码的行列。
        bwd: 累积后复权因子面板，用于还原未复权收盘价比较涨跌停价。
        cache_root: 缓存根目录，None 时取 ``Config.cache()["root"]``。

    Returns:
        DataFrame(index=close_adj.index, columns=close_adj.columns)，dtype=bool。

    注意：**不要**把本函数用于"盘后立刻发布选股结果"的场景——那里 T+1 状态未知。
    该场景用 ``build_directional_masks``。
    """
    root = cache_root or str(Config.cache()["root"])
    p = Path(root) / "history_stock_status.parquet"
    idx, cols = close_adj.index, close_adj.columns
    if not p.exists():
        return pd.DataFrame(True, index=idx, columns=cols)

    st = pd.read_parquet(p)
    st = st[st.index.get_level_values("code").isin(cols)]
    if st.empty:
        return pd.DataFrame(True, index=idx, columns=cols)

    def _wide(col: str) -> pd.DataFrame:
        if col not in st.columns:
            return pd.DataFrame(False, index=idx, columns=cols)
        return st[col].unstack("code").reindex(index=idx, columns=cols)

    susp = _wide("is_suspended").fillna(False)
    is_st = _wide("is_st").fillna(False)
    hi_lim = _wide("high_limited")
    lo_lim = _wide("low_limited")

    # 执行日状态（默认 T+1：T 日信号 → T+1 成交；execution_lag=0 则取 T 日）
    lag = int(execution_lag)
    susp_next = susp.shift(-lag).fillna(False) if lag > 0 else susp.fillna(False)
    hi_next = hi_lim.shift(-lag)
    lo_next = lo_lim.shift(-lag)

    if bwd is not None and len(bwd):
        bwd_al = bwd.reindex(index=idx, columns=cols).ffill()
        raw_next = (close_adj / bwd_al).shift(-lag)
    else:
        raw_next = close_adj.shift(-lag)

    tol = 1e-6
    sealed_up = (raw_next >= hi_next * (1 - tol)) & hi_next.notna()
    sealed_dn = (raw_next <= lo_next * (1 + tol)) & lo_next.notna()

    st_next = is_st.shift(-lag).fillna(False) if lag > 0 else is_st.fillna(False)
    bad = susp_next | sealed_up | sealed_dn | st_next | susp | is_st
    mask = ~bad
    return mask.fillna(True).astype(bool)


# ---------------------------------------------------------------------------
# 口径二：T 日信号诊断（盘后发布场景）
# ---------------------------------------------------------------------------

def build_directional_masks(
    status_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    codes: pd.Index,
    close_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构建**有方向的**可交易掩码 (buyable, sellable)，用于信号级约束校验。

    口径：只使用 **T 日**（信号日）已知状态——盘后发布选股结果时 T+1 尚未发生。

    比 ``build_tradable_mask`` 更细：涨停封板只是"买不进"（buyable=False），
    跌停封板只是"卖不出"（sellable=False），停牌则两者都禁。这样信号层能区分
    "想买但涨停"与"想卖但跌停"，分别标记 BLOCKED_BUY / BLOCKED_SELL。

    Args:
        status_df: get_history_stock_status 长表，列含 date, code, high_limited,
            low_limited, is_suspended。index 可以是 (date, code) 多索引或 RangeIndex。
        dates / codes: 交易日 / 代码索引。
        close_panel: (date, code) 收盘价面板，判涨停/跌停封板；None 时只按停牌过滤。
    Returns:
        (buyable, sellable)：均为 (date, code) 布尔面板，True=允许该方向。默认全 True。
    """
    buyable = pd.DataFrame(True, index=dates, columns=codes)
    sellable = pd.DataFrame(True, index=dates, columns=codes)

    if status_df is None or status_df.empty:
        return buyable, sellable

    df = status_df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])

    if "is_suspended" in df.columns:
        susp = df.pivot_table(index="date", columns="code", values="is_suspended", aggfunc="max")
        susp = susp.reindex(index=dates, columns=codes)
        susp = susp.astype(float).fillna(0.0).astype(bool)
        buyable &= ~susp
        sellable &= ~susp

    if close_panel is not None and {"high_limited", "low_limited"}.issubset(df.columns):
        hi = df.pivot_table(index="date", columns="code", values="high_limited")
        lo = df.pivot_table(index="date", columns="code", values="low_limited")
        high_wide = hi.reindex(index=dates, columns=codes)
        low_wide = lo.reindex(index=dates, columns=codes)
        close_aligned = close_panel.reindex(index=dates, columns=codes)

        hit_up = ((close_aligned >= high_wide) & high_wide.notna()).fillna(False)
        hit_down = ((close_aligned <= low_wide) & low_wide.notna()).fillna(False)
        buyable &= ~hit_up      # 涨停封板：买不进
        sellable &= ~hit_down   # 跌停封板：卖不出

    return buyable, sellable
