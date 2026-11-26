"""「过热降仓」overlay 的落地实现（2026-11-26 建）。

规则（R2，参数全部从 ``config/baselines.yaml`` 的 ``overlays.topping-guard`` 读）
------------------------------------------------------------------------------
t 月末，若 ``liquidity / nl_size / momentum / beta`` 中**任一**风格的
**过去 12 个月复利收益**在该风格**自身扩张窗口**（只用 t 及之前的数据）里的分位
超过 ``quantile``（默认 90%），则 t+1 月组合权重乘 ``weight``（默认 0.8）。
其余月份权重 1.0。降仓部分留现金（rf=0），月度切换成本由回测引擎按实际换手计。

周频决策（2026-11-26 加，仅研究 / 择时对比，**不进生产**）
--------------------------------------------------------
同一公式改成**周五收盘**算、**下周一**生效：12 月 lookback → 52 周、min_periods
24 月 → 104 周。目的：把「过热 → 降仓」的滞后从 ≤21 天压到 ≤5 天。代价是切换
更频繁（周频成本按每周 |Δw|×13.5bp 计）。结论见 ``reports/topping_risk/README.md``。

为什么是这套触发
----------------
三档超额的来源是**慢速风格动量**（暴露与过去 12 月风格收益正相关 0.32~0.51），
regime 反转时正好追在顶部。风格自身 12 月收益的历史高位，是「这套暴露已经拥挤」
的可观测代理。实测 21 相位：年化 −0.33pp / Sharpe +0.056 / MDD −7.0pp / 2024-01
微盘崩 +6.2pp，且相对同平均敞口的常数仓仍 +1.15pp ⇒ 是择时，不是单纯降杠杆。
证据 ``reports/topping_risk/README.md``。

边界
----
- 只表达**事前已知**规则；断路器（依赖自身已实现超额）不能用本模块，须事后路径。
- 风格收益来自 ``style_attribution.style_monthly()``（Barra WLS 月频，已行业中性；
  周频走 ``style_weekly()``，日频因子收益→周内复利）。

用法::

    python -m scripts.portfolio.topping_overlay --weights      # 打印月度倍率
    python -m scripts.portfolio.topping_overlay --weekly       # 打印周频倍率（研究用）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_baselines  # noqa: E402
from research.barra import (  # noqa: E402
    STYLE_VARS, load_barra_inputs, run_factor_returns)
from scripts.portfolio.style_attribution import style_monthly  # noqa: E402
from strategy.overlay import WeightScaleOverlay  # noqa: E402

DEFAULT_ID = "topping-guard"
FR_WEEKLY = ROOT / "reports" / "topping_risk" / "factor_returns_weekly.csv"


def overlay_spec(overlay_id: str = DEFAULT_ID) -> dict:
    """从登记表取 overlay 参数（唯一真源；缺条目直接报错，不落回硬编码缺省）。"""
    reg = load_baselines()
    spec = ((reg.get("overlays") or {}).get(overlay_id))
    if not spec:
        raise KeyError(
            f"config/baselines.yaml 缺 overlays.{overlay_id}（参数不得写在代码里）")
    return spec


def expanding_pct(s: pd.Series, min_periods: int = 24) -> pd.Series:
    """扩张窗口分位（只用 t 及之前的数据，无前视；分母剔除窗口内 NaN）。"""
    def _one(x: np.ndarray) -> float:
        v = x[np.isfinite(x)]
        return float((v <= v[-1]).mean()) if len(v) else float("nan")

    return s.expanding(min_periods=int(min_periods)).apply(_one, raw=True)


def hot_flags(styles: pd.DataFrame, trigger_styles: tuple[str, ...],
              quantile: float, lookback: int = 12,
              min_periods: int = 24) -> pd.Series:
    """任一触发风格的 ``lookback`` 月收益扩张分位 > ``quantile`` ⇒ 过热。"""
    flags = {}
    for name in trigger_styles:
        f = styles[name]
        r = (1 + f).rolling(int(lookback)).apply(np.prod, raw=True) - 1
        flags[name] = expanding_pct(r, min_periods) > float(quantile)
    return pd.concat(flags, axis=1).any(axis=1)


def monthly_weights(overlay_id: str = DEFAULT_ID) -> pd.Series:
    """t 月末过热 ⇒ t+1 月倍率。返回 ``PeriodIndex(freq="M")`` 的倍率序列。"""
    spec = overlay_spec(overlay_id)
    styles = style_monthly()
    styles.index = styles.index.to_period("M")
    hot = hot_flags(styles,
                    tuple(spec["trigger_styles"]),
                    float(spec["quantile"]),
                    int(spec.get("lookback_months", 12)),
                    int(spec.get("min_periods", 24)))
    w = pd.Series(1.0, index=hot.index)
    w[hot.shift(1).fillna(False).astype(bool)] = float(spec["weight"])
    return w


def week_last(idx: pd.DatetimeIndex) -> pd.Series:
    """每个自然周（``W-FRI``）的**实际最后交易日**（节假日短周也对齐到真实末日）。"""
    per = idx.to_period("W-FRI")
    return pd.Series(idx, index=per).groupby(level=0).last()


def to_weekly(daily: pd.Series) -> pd.Series:
    """日收益 → 周收益（周内复利），索引换成该周实际最后交易日。"""
    per = daily.index.to_period("W-FRI")
    w = (1 + daily.fillna(0.0)).groupby(per).prod() - 1
    w.index = pd.DatetimeIndex(week_last(daily.index).reindex(w.index).to_numpy())
    return w


def style_weekly(rebuild: bool = False) -> pd.DataFrame:
    """13 个风格的**周频**组合收益（Barra WLS 日频因子收益 → 周内复利）。

    与 :func:`style_attribution.style_monthly` 同源（逐日 WLS、已行业中性），只换
    聚合频率。缓存 ``reports/topping_risk/factor_returns_weekly.csv``（重算约 3~5 分钟）。
    """
    if FR_WEEKLY.exists() and not rebuild:
        return pd.read_csv(FR_WEEKLY, index_col=0, parse_dates=True)

    inputs = load_barra_inputs()
    fr = run_factor_returns(inputs, begin="2018-12-01")
    sty = [c for c in fr.columns if c.startswith("sty:")]
    if len(sty) != len(STYLE_VARS):
        raise RuntimeError(
            f"风格因子列缺失：{set(STYLE_VARS) - {c[4:] for c in sty}}")
    out = pd.DataFrame({c[4:]: to_weekly(fr[c]) for c in sty})
    FR_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FR_WEEKLY)
    return out


def weekly_weights(overlay_id: str = DEFAULT_ID) -> pd.Series:
    """t 周（周五收盘）过热 ⇒ t+1 周倍率。返回 ``PeriodIndex(freq="W-FRI")``。

    参数从登记表读，月 → 周按 ``×52/12`` 换算（12 月 → 52 周、24 月 → 104 周）。
    """
    spec = overlay_spec(overlay_id)
    styles = style_weekly()
    look_w = int(round(float(spec.get("lookback_months", 12)) * 52 / 12))
    min_w = int(round(float(spec.get("min_periods", 24)) * 52 / 12))
    hot = hot_flags(styles,
                    tuple(spec["trigger_styles"]),
                    float(spec["quantile"]),
                    look_w, min_w)
    w = pd.Series(1.0, index=hot.index)
    w[hot.shift(1).fillna(False).astype(bool)] = float(spec["weight"])
    w.index = w.index.to_period("W-FRI")
    return w


def overlay_plan(index: pd.DatetimeIndex, freq: str = "M",
                 overlay_id: str = DEFAULT_ID, execution: str = "open"
                 ) -> tuple[pd.DatetimeIndex, set]:
    """引擎调仓计划：``(signal_days, inner_exec_days)``。

    - ``signal_days``：调仓信号日 T —— 月频选股日 ∪ 周频倍率变化日（周五）。
    - ``inner_exec_days``：需要**重新选股**的日期（引擎实际调用策略的日子）：
      ``execution="close"`` 时 = 选股日 T，T+1 成交时 = 执行日 E=T+1。
      其余周频调仓日复用上一次选股权重，只换倍率。

    月频 overlay 时 ``signal_days`` 就等于选股日，行为与旧版一致。
    """
    from scripts.evaluation.caliber import rebalance_dates  # 延迟导入避免循环

    idx = pd.DatetimeIndex(index)
    msig = pd.DatetimeIndex(rebalance_dates(idx, freq=freq))
    pos = {d: i for i, d in enumerate(idx)}
    if execution == "close":
        inner = set(msig)
    else:
        inner = {idx[pos[t] + 1] for t in msig if pos[t] + 1 < len(idx)}
    if not str(overlay_spec(overlay_id).get("freq", "M")).upper().startswith("W"):
        return msig, inner

    w = weekly_weights(overlay_id)
    chg = w[w.ne(w.shift(1))].index
    last = week_last(idx).reindex(chg).dropna()
    extra = pd.DatetimeIndex(last.to_numpy())
    extra = extra[(extra >= idx[0]) & (extra <= idx[-1])]
    return pd.DatetimeIndex(sorted(set(msig) | set(extra))), inner


def wrap(strategy, overlay_id: str = DEFAULT_ID, inner_dates=None):
    """把 overlay 套到已构造好的策略上（供 caliber --overlay 调用）。"""
    spec = overlay_spec(overlay_id)
    if str(spec.get("freq", "M")).upper().startswith("W"):
        return WeightScaleOverlay(strategy, weekly_weights(overlay_id),
                                  name=overlay_id, inner_dates=inner_dates)
    return WeightScaleOverlay(strategy, monthly_weights(overlay_id),
                              name=overlay_id)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--overlay", default=DEFAULT_ID)
    ap.add_argument("--weights", action="store_true", help="打印月度倍率")
    ap.add_argument("--weekly", action="store_true", help="用周频决策（周五收盘）")
    a = ap.parse_args(argv)
    spec = overlay_spec(a.overlay)
    print(f"{a.overlay}: 触发 {spec['trigger_styles']}  "
          f"q={spec['quantile']}  w={spec['weight']}")
    w = weekly_weights(a.overlay) if a.weekly else monthly_weights(a.overlay)
    freq = "周" if a.weekly else "月"
    cut = w[w < 1.0]
    print(f"降仓 {len(cut)} / {len(w)} {freq}；平均倍率 {w.mean():.4f}")
    if a.weights:
        print(cut.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
