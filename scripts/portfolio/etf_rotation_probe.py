"""
ETF 轮动回测引擎（数据装载 / 面板 / 回测 / 出场调度）
=====================================================

生产信号板与回测共用：

  - ``load_bundle``：一次装载全样本价格 / 成交额 / 基准（含暖机），窗口切片复用；
  - ``build_panels``：基础因子面板（12-1 动量等），扩展面板见 ``etf_rotation_panels``；
  - ``run_variant``：单变体回测。**次日执行口径必须三件套**（见函数 docstring 禁忌）；
  - ``build_exit_schedule`` / ``ExitOverlay`` / ``ExposureOverlay``：出场与仓位调度的
    语义实现——出场比赛本身已全部判否，函数保留供测试与未来复用。

方案与结论见 ``reports/etf_contest5/README.md``；历史变体清单与实验主循环已于
2026-10-29 随实验记录一并回收。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import EtfCosts
from backtest.engine import VectorBacktest
from config import Config
from data.etf_universe import EtfUniverse
from strategy.base import Strategy
from strategy.etf_rotation import EtfRotationAdvanced
from strategy.multi_signal import compose_signals

BENCH = Config.benchmarks()["etf"]          # 沪深300
ROOT = Path(__file__).resolve().parents[2]
MIN_AMOUNT = 3e7        # 近 60 日均成交额下限（元）
SEGMENTS = [(2020, 2021), (2022, 2023), (2024, 2026)]
WARMUP_BEGIN = 20180101  # 12-1 动量需要 ~273 个交易日暖机


def _pit_shift(df: pd.DataFrame) -> pd.DataFrame:
    """把宽表整体下移一行（第 i 行 = 第 i−1 行的值，首行 NaN）。

    用于修正「面板按调用日切片」造成的信息渗漏：策略内部一律写
    ``ret_panel.loc[:date]`` / ``vol_panel.loc[date]``，而 ``execution_shift``
    模式下 ``date`` 是**执行日 E**（开盘下单那天），该日的收益与波动在撮合时
    尚未发生。下移一行后 ``loc[:E]`` 的最后一根变成 E−1 = 信号日 T，与真实
    可得信息一致。见 ``scripts/portfolio/etf_leak_probe.py``。
    """
    return df.shift(1)


def load_bundle(cache, begin: int = WARMUP_BEGIN, end: int | None = None):
    """构建全样本面板束（含暖机），后续按窗口切片复用。"""
    uni = EtfUniverse(cache)
    close = uni.load_close(adjust=True)
    hi = pd.Timestamp(str(end)) if end else close.index.max()
    close = close.loc[(close.index >= pd.Timestamp(str(begin))) & (close.index <= hi)]

    raw = cache.read_daily("etf")
    amount = raw["amount"].unstack("code").reindex(index=close.index, columns=close.columns)

    idx = cache.get_index_daily(BENCH, begin, int(hi.strftime("%Y%m%d")))
    bench_close = idx["close"].unstack("code")
    bcol = [c for c in bench_close.columns if c == BENCH or c == BENCH.replace(".SH", "")]
    bench = (bench_close[bcol[0]].reindex(close.index).ffill()
             .pct_change(fill_method=None) if bcol else pd.Series(dtype=float))
    return uni, close, amount, bench


def build_panels(close: pd.DataFrame, amount: pd.DataFrame) -> dict:
    """由收盘价/成交额构建因子、波动率、流动性掩码面板。"""
    ret = close.pct_change(fill_method=None)
    composite, _, _ = compose_signals(close)
    return {
        "composite": composite,
        "mom12_1": close.shift(21) / close.shift(252) - 1.0,
        "vol": ret.rolling(20).std() * np.sqrt(252.0),
        "mask": amount.rolling(60, min_periods=40).mean() >= MIN_AMOUNT,
    }


def slice_bundle(bundle, panels: dict, w0: int | None, w1: int | None):
    """按 [w0, w1] 切片所有面板（因子已在暖机后计算，切片不损失暖机）。"""
    uni, close, amount, bench = bundle
    if w0 is None:
        return bundle, panels
    hi = pd.Timestamp(str(w1)) if w1 is not None else close.index.max()
    lo = pd.Timestamp(str(w0))
    m = (close.index >= lo) & (close.index <= hi)
    sub = (uni, close.loc[m],
           amount.loc[m], bench.loc[bench.index.isin(close.index[m])] if len(bench) else bench)
    sub_panels = {k: v.loc[m] for k, v in panels.items()}
    return sub, sub_panels


class ExitOverlay(Strategy):
    """日级出场覆盖层（2026-10-27，赛后探针用）。

    周频调仓日照常选标；非调仓的**触发日**只做一件事——把触发离场的标的权重
    清零，腾出的资金落到 ``CASH``（0 收益列），一直留到下个周频调仓日（不设
    再入场规则）。引擎会把多头腿归一化到 1，所以必须显式写 CASH 才能真留现金。
    """

    def __init__(self, base: EtfRotationAdvanced, weekly: set,
                 exits: dict[pd.Timestamp, set[str]]):
        self.base = base
        self.weekly = set(weekly)
        self.exits = exits
        self.cash_code = base.cash_code or "CASH"
        self._last = pd.Series(dtype=float)
        self.name = base.name + "_exit"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        return self.get_weights_at(None, factor_values)

    def get_weights_at(self, date: pd.Timestamp | None,
                       factor_values: pd.Series) -> pd.Series:
        if date in self.weekly:
            w = self.base.get_weights_at(date, factor_values)
        else:
            w = self._last.copy()          # 触发日：沿用上次调仓权重
        drop = [c for c in self.exits.get(date, set()) if c in w.index]
        if drop:
            w = w.drop(labels=drop)
        w = w[w.abs() > 1e-12]
        total = float(w.sum())
        if total <= 1e-12:
            w = pd.Series({self.cash_code: 1.0})
        elif total < 1.0:
            w[self.cash_code] = 1.0 - total
        self._last = w.copy()
        return w


class ExposureOverlay(Strategy):
    """组合层仓位缩放（2026-10-27，探索用）。

    每个调仓日把基础权重整体乘以 ``factors``（0–1），余量落到 ``CASH`` 列。
    用途：波动率缩放 / regime 降仓——**不是**逐只止损，不改变选标。
    """

    def __init__(self, base: EtfRotationAdvanced, factors: pd.Series,
                 cash_code: str = "CASH"):
        self.base = base
        self.factors = factors
        self.cash_code = cash_code
        self.name = base.name + "_expo"

    def get_weights(self, factor_values: pd.Series) -> pd.Series:
        return self.get_weights_at(None, factor_values)

    def get_weights_at(self, date: pd.Timestamp | None,
                       factor_values: pd.Series) -> pd.Series:
        w = self.base.get_weights_at(date, factor_values)
        f = float(self.factors.get(date, 1.0)) if date is not None else 1.0
        f = min(max(f, 0.0), 1.0)
        if f >= 1.0:
            return w
        w = w * f
        w[self.cash_code] = w.get(self.cash_code, 0.0) + (1.0 - f)
        return w


def build_exit_schedule(close: pd.DataFrame, weekly: list[pd.Timestamp],
                        selections: dict, rule: dict,
                        mom_rank: pd.DataFrame | None = None,
                        gate: pd.Series | None = None) -> dict:
    """按出场规则预生成触发日（无前视：只用触发日及之前的收盘价）。

    Args:
        close: date × code 收盘价（后复权）。
        weekly: 周频调仓日（升序）。
        selections: {调仓日: [建仓代码]}（基线周频选标，出场不改变选标）。
        rule: ``{"rule": "anchor"|"mom12_0_rank"|"ma60"|"trail", ...参数}``。
        mom_rank: B2 用的 ``mom12_0`` 截面排名（0 = 最强），与 close 同索引。
        gate: 可选的日级开关（如「组合波动高位才启用」）；False 的日子不触发。

    Returns:
        ``{"weekly": set, "exits": {date: set(codes)}, "events": DataFrame}``。
    """
    px = close
    ma = close.rolling(60, min_periods=40).mean()
    pos = {d: i for i, d in enumerate(px.index)}
    weekly = [d for d in weekly if d in selections and d in pos]
    exits: dict[pd.Timestamp, set[str]] = {}
    events: list[dict] = []
    kind = rule["rule"]
    # 建仓日与入场后最高价跨周保留：连续两次周频都持有同一只时，锚定价/峰值
    # 必须仍指向**真实建仓日**，不能随周频重选一起前滚（否则 B1 退化成滚动
    # 21 日收益 <0 的日频闸门）。
    entry_of: dict[str, pd.Timestamp] = {}
    peaks: dict[str, float] = {}

    for k, d in enumerate(weekly):
        sel = [c for c in selections[d] if c in px.columns]
        entry_of = {c: entry_of.get(c, d) for c in sel}
        peaks = {c: (peaks[c] if c in peaks else px[c].loc[d]) for c in sel}
        i0 = pos[d]
        i1 = pos[weekly[k + 1]] if k + 1 < len(weekly) else len(px.index)
        held = list(sel)
        for i in range(i0 + 1, i1):
            t = px.index[i]
            for c in list(held):
                p = px[c].iloc[i]
                if not np.isfinite(p):
                    continue
                peaks[c] = max(peaks[c], p)
                ref = np.nan
                if kind == "anchor":
                    ia = pos[entry_of[c]] - 21
                    a = px[c].iloc[ia] if ia >= 0 else np.nan
                    if np.isfinite(a):
                        ref = float(rule.get("mult", 1.0)) * a
                        trig = p < ref
                    else:
                        trig = False
                elif kind == "trail":
                    ref = peaks[c] * (1.0 - float(rule["dd"]))
                    trig = p <= ref
                elif kind == "ma60":
                    ref = ma[c].iloc[i]
                    trig = bool(np.isfinite(ref) and p < ref)
                elif kind == "mom12_0_rank":
                    r = mom_rank[c].iloc[i] if c in mom_rank.columns else np.nan
                    trig = bool(np.isfinite(r) and r >= int(rule.get("max_rank", 5)))
                else:
                    raise ValueError(f"未知 exit rule: {kind}")
                if trig and gate is not None and not bool(gate.get(t, False)):
                    trig = False
                if trig:
                    exits.setdefault(t, set()).add(c)
                    p0 = float(px[c].loc[entry_of[c]])
                    events.append({
                        "entry": entry_of[c], "date": t, "code": c,
                        "entry_px": p0, "exit_px": float(p),
                        "since_entry": float(p / p0 - 1.0),
                        "peak": float(peaks[c]),
                        "threshold": float(ref) if np.isfinite(ref) else np.nan,
                    })
                    held.remove(c)
                    entry_of.pop(c, None)
                    peaks.pop(c, None)
    return {"weekly": set(weekly), "exits": exits,
            "events": pd.DataFrame(events), "rule": kind}


def whip_audit(events: pd.DataFrame, close: pd.DataFrame, rule: dict,
               window: int = 20) -> dict:
    """鞭打审计：离场后 ``window`` 日内「收复锚定价」的比例与代价。

    参照价：``trail`` 用**入场后最高价**（真正的鞭打 = 砍在坑里又涨回前高）、
    ``anchor`` 用锚定价阈值、其余用离场价。``med_fwd`` 是离场后窗口末的
    中位收益——为正说明这笔卖出事后是亏的（提前离场代价）。
    """
    if events.empty:
        return {"n": 0, "recover": np.nan, "med_fwd": np.nan, "med_cost": np.nan}
    pos = {d: i for i, d in enumerate(close.index)}
    ref_col = {"trail": "peak", "anchor": "threshold"}.get(rule["rule"], "exit_px")
    rec, fwd = [], []
    for _, e in events.iterrows():
        i = pos[e["date"]]
        fut = close[e["code"]].iloc[i + 1: i + 1 + window].dropna()
        if fut.empty:
            continue
        ref = float(e[ref_col]) if np.isfinite(e[ref_col]) else float(e["exit_px"])
        rec.append(bool(fut.max() >= ref))
        fwd.append(float(fut.iloc[-1] / e["exit_px"] - 1.0))
    rec_s, fwd_s = pd.Series(rec, dtype=float), pd.Series(fwd)
    return {
        "n": len(rec_s), "recover": float(rec_s.mean()) if len(rec_s) else np.nan,
        "med_fwd": float(fwd_s.median()) if len(fwd_s) else np.nan,
        "med_cost": float(fwd_s[fwd_s < 0].median()) if (fwd_s < 0).any() else np.nan,
    }


def run_variant(bundle, panels: dict, spec: dict, w0: int | None = None,
                w1: int | None = None, extra_close: pd.DataFrame | None = None,
                rebalance_days: set | None = None,
                exit_plan: dict | None = None,
                exposure: pd.Series | None = None,
                execution_split: tuple[pd.DataFrame, pd.DataFrame] | None = None,
                execution_shift: bool = False,
                ) -> dict:
    """在（可选）窗口上跑一个变体，返回指标与净值。

    ``extra_close`` 用于注入合成资产列（如防御腿 DEF：债+金的净值水平线）。
    合成资产的因子/波动率为 NaN（不会被选中），掩码恒为 True。

    ``execution_split`` 为引擎的**次日执行价模式**（``build_execution_split`` 产物）：
    传入后调仓日 T 的权重在 T+1 的 fill 价成交，默认为 ``None`` = 现行收盘价成交
    （略乐观口径，见 ``etf_contest_spec.EXEC_NOTE``）。

    ⚠️ ``execution_split`` 必须与 ``execution_shift=True`` 一起用，二者是同一件事的两半
    （口径同 ``scripts/evaluation/caliber.py::run_once``）：
      - 信号面板 ``shift(1)`` → 执行日 E 取信号日 T 的因子；
      - 调仓日传 **E = T+1**（而非 T）；
      - split 面板由**信号日 T** 构造（``build_execution_split(fill, close, rb_signal)``）。
    缺任一项都会 silent 失效：只传 split、调仓日仍在 T 时，引擎在 E 日结算用的
    ``current_weights`` 已是 T 日设定的**新权重**，隔夜段被错付给新仓位，
    结果退化回「T 收盘成交」而只差二阶项（实测 equity 末值仅差 +0.27%，
    会让人误判"次日开盘几乎无代价"）。

    ``execution_shift=True`` 与 ``rebalance_days`` 同给时，后者按**信号日**
    （``rb_signal``）解释，引擎收到的调仓日是其**下一个交易日**。非 ``freq``
    可表达的相位（如「周内第 k 个交易日」）必须走这条路径，否则会被 ``freq``
    分支覆盖、静默退回周一调仓。

    🚨 **PIT 修正（2026-10-29）**：``execution_shift=True`` 时**策略侧**的
    ``ret_panel`` / ``vol_panel`` 会额外整体 ``shift(1)``。原因是引擎在执行日 E 调
    ``get_weights_at(E)``，而策略内部按 ``loc[:date]`` 取窗口 —— 不修正就会读到
    E 日当天收益（开盘下单时尚未发生）。**引擎结算的 ``returns`` 不 shift**
    （它按「第 i 行是 i−1→i 收益」对齐）。影响量级：三锋 +4.5pp、五衡 −4.0pp
    （方向都不同 ⇒ 不是"保守偏差"），详见 ``reports/etf_contest5/README.md`` §4.3。

    ⚠️ **窗口口径陷阱（2026-10-29 实测）**：`ret_panel` 由**切片后的** close 重算，
    逐窗独立调用时它只有窗口内 ~40 个交易日 ⇒ `_dedup` 的生效门槛
    ``max(40, dedup_window // 2) = 60`` 与 `_risk_parity` 的 ``len(sub) >= 20``
    会在窗口里冷启动失效。实测后果：定版 contest3 的 2 月窗口成绩从
    ``均值 +5.59% / p10 −3.64% / 最差 −12.34%`` 假性恶化到
    ``+3.24% / −7.78% / −19.66%``（Sharpe 1.28 → 1.86 的假性"改善"）。
    **要窗口成绩请用「单次连续回测（20200101→末日）+ 事后按月首切 2 月窗口」**
    （旧 `etf_contest_probe.two_month_table` 口径，已逐位复现）。
    """
    (uni, close, _amount, bench), p = slice_bundle(bundle, panels, w0, w1)
    extra_cols: list[str] = []
    if extra_close is not None:
        extra_close = extra_close.reindex(close.index)
        close = close.join(extra_close)
        extra_cols = list(extra_close.columns)
    # 现金资产：常数 1（0 收益）。引擎在施加掩码后会把多头腿归一化到 1，
    # 波动率目标的「留现金」只能显式落在一个 0 收益列上。
    returns = close.assign(CASH=1.0).pct_change(fill_method=None)
    factor = p[spec["signal"]].copy()
    factor["CASH"] = np.nan
    vol = p["vol"].copy()
    vol["CASH"] = np.nan
    mask = p["mask"].copy()
    mask["CASH"] = True
    for c in extra_cols:
        factor[c] = np.nan
        vol[c] = np.nan
        mask[c] = True
    # 占位腿（国债/黄金）必须可执行：引擎施加掩码后会把多头腿重新归一化，
    # 若占位腿被掩掉，只数仍会不足。它们在真实账户中是流动的债券/黄金 ETF。
    for c in spec.get("fill_codes") or []:
        if c in mask.columns:
            mask[c] = True
    cash = spec.get("cash_code")
    if cash is None and spec.get("target_vol"):
        cash = "DEF" if spec.get("defense") else "CASH"
    if cash is None:
        # 出场覆盖层把腾出的资金落到显式 CASH 列；无波动目标时也要能统计现金占比
        cash = "CASH"
    # 选标只在可交易集合内进行：不可执行标的的因子置 NaN。否则策略会先选中
    # 它们，引擎再把权重清零并重新归一化多头腿 ⇒ 持仓只数被动缩水、
    # 收益被"第 6 名"顶替，与「不许持有不达标标的」的真实约束不符。
    if spec.get("mask", True):
        factor = factor.where(mask)
    # 🚨 PIT 修正（2026-10-29，由莉酱发现）：`execution_shift` 下引擎在**执行日 E**
    # 调 `get_weights_at(E)`，而策略内部按 `ret_panel.loc[:date]` / `vol_panel.loc[date]`
    # 取窗口 ⇒ 会读到 E 日当天的收益与波动，而 E 日开盘下单时它们尚未发生。
    # 三条实际路径：`_risk_parity` 的 60 日协方差、`_dedup` 的 120 日相关、
    # `_weights` 的 `vol_panel.loc[date]`（`inverse_vol` / `target_vol`，磐石走这条）。
    # 修正 = 策略侧面板整体下移一行，使 E 日只看到 ≤E−1 = 信号日 T。
    # **引擎结算用的 `returns` 绝不能 shift**：它按行对齐「第 i 行是 i−1→i 收益」，
    # 移动会直接改写收益标签。探针与量级见 `scripts/portfolio/etf_leak_probe.py`。
    ret_for_strategy = _pit_shift(returns) if execution_shift else returns
    vol_for_strategy = _pit_shift(vol) if execution_shift else vol
    strategy = EtfRotationAdvanced(
        top_k=spec.get("top_k", 5),
        absolute_filter=spec.get("absolute", False),
        abs_threshold=spec.get("abs_threshold", 0.0),
        weighting=spec.get("weighting", "equal"),
        vol_panel=vol_for_strategy,
        target_vol=spec.get("target_vol"),
        enter_rank=spec.get("enter_rank"),
        exit_rank=spec.get("exit_rank"),
        cash_code=cash,
        ret_panel=ret_for_strategy,
        cov_window=spec.get("cov_window", 60),
        max_weight=spec.get("max_weight"),
        dedup_corr=spec.get("dedup_corr"),
        dedup_window=spec.get("dedup_window", 120),
        dedup_fill=spec.get("dedup_fill", False),
        fill_to=spec.get("fill_to"),
        fill_codes=spec.get("fill_codes"),
    )
    cx = float(spec.get("cost_x", 1.0))
    commission_bp = float(spec.get("commission_bp", 5.0))
    slippage_bp = float(spec.get("slippage_bp", 5.0))
    if exit_plan is not None:
        strategy = ExitOverlay(strategy, exit_plan["weekly"], exit_plan["exits"])
        rebalance_days = set(rebalance_days or ()) | exit_plan["weekly"] \
            | set(exit_plan["exits"].keys())
    if exposure is not None:
        strategy = ExposureOverlay(strategy, exposure, cash_code=cash)
    if execution_shift:
        # 次日执行价三件套之一：信号 shift(1) + 调仓日搬到 E=T+1。
        # 与 build_execution_split(fill, close, rb_signal) 配套（见 docstring）。
        freq = spec.get("freq", "M")
        _s = pd.Series(close.index, index=close.index)
        if rebalance_days:
            # 显式传入的自定义调仓日按**信号日 rb_signal** 解释（再把 E=T+1 算出来）。
            # 非 freq 可表达的相位（如「周内第 k 个交易日」）必须走这条，否则会被
            # 下面的 freq 分支覆盖掉、静默变成周一调仓。
            _in_index = set(close.index)
            _sig = sorted(d for d in rebalance_days if d in _in_index)
        elif freq == "W":
            _sig = list(_s.groupby(_s.index.to_period("W")).first())
        elif freq == "M":
            _sig = list(_s.groupby(_s.index.to_period("M")).first())
        elif freq == "2W":
            _sig = list(_s.groupby(_s.index.to_period("W")).first())[::2]
        else:
            _sig = list(close.index)
        _days = list(close.index)
        _pos = {d: i for i, d in enumerate(_days)}
        rebalance_days = {_days[_pos[t] + 1] for t in _sig
                          if t in _pos and _pos[t] + 1 < len(_days)}
        factor = factor.shift(1)
    bt = VectorBacktest(strategy=strategy, rebalance_freq=spec.get("freq", "M"),
                        initial_capital=1_000_000.0,
                        costs=EtfCosts(commission_rate=commission_bp * 1e-4 * cx,
                                       commission_min=0.1 * cx,
                                       slippage_bp=slippage_bp * cx))
    res = bt.run(factor_panel=factor, returns_panel=returns,
                 executable_mask=mask if spec.get("mask", True) else None,
                 horizon=1, rebalance_days=rebalance_days,
                 execution_split=execution_split)
    m = res.metrics(benchmark_returns=bench if len(bench) else None)
    w = res.weights_history
    live = w[w.abs().sum(axis=1) > 1e-9]
    held = [c for c in w.columns if c not in ("CASH", "DEF")]
    cash_w = live[cash] if cash in live.columns else pd.Series(0.0, index=live.index)
    years = max(len(res.daily_returns) / 252.0, 1e-9)
    cost_drag = float(res.cost_series.sum()) / 1_000_000.0 / years
    rb_dates = res.turnover_series.dropna().index
    wd = w.loc[rb_dates] if len(rb_dates) else w
    hold_counts = (wd[held].abs() > 1e-9).sum(axis=1)
    return {
        "annual_return": m["annual_return"],
        "sharpe": m["sharpe"],
        "max_drawdown": m["max_drawdown"],
        "excess": m["annual_return"] - (m.get("benchmark_annual_return") or 0.0),
        "avg_turnover": m.get("avg_turnover", 0.0),
        "n_rebalance": int(res.turnover_series.notna().sum()),
        "avg_holdings": float((live[held].abs() > 1e-9).sum(axis=1).mean()) if len(live) else 0.0,
        "avg_cash": float(cash_w.mean()) if len(cash_w) else 0.0,
        "cost_drag": cost_drag,
        "hold_counts": hold_counts,
        "rebalance_dates": rb_dates,
        "weights": w,
        "equity": res.equity_curve,
        # 供信号板在「最新交易日 == 调仓日」时直接取当日权重：引擎不允许在最后
        # 一根 bar 建仓（i < n-1），否则实盘指令会落后一周。
        "strategy": strategy,
    }
