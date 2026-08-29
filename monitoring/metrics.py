"""监控指标计算 —— 因子与模型预测统一口径。

三类输入、一套输出：
1. evals ic 序列（库内预存，注册时口径）→ 全期 / 近期 IC 统计（NW-t 同因子库）
2. 因子面板 → 近期覆盖率、数据新鲜度
3. 收益面板（h=1 次日收益，与因子库 IC 口径一致）→ 近期分位单调性 / 多空日均

模型因子（``model:`` 前缀）额外携带期望基线：注册时 registry 行的 ic_mean
（即上线时的 OOS 表现），供告警规则做「保留率」判断。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from research.factor_analysis import calc_neutral_ic_series
from stats.ic import quantile_backtest
from stats.monitor import monitor_ic_series

MODEL_PREFIX = "model:"
_MODEL_ID_RE = re.compile(r"model_id=(\d+)")

# 冻结基线最低显著度：只有注册时基线 |IC| ≥ 该值才把它当作"冻结基线"做保留率判断。
# 注册时 IC 本就近 0 的因子（无显著基线可衰），回退全期基线，避免小基线放大数值/误判。
_FROZEN_BASELINE_MIN_ABS = 0.01


@dataclass
class MonitorMetrics:
    """单个因子 / 模型预测的监控快照（一行）。"""

    name: str
    category: str = "factor"  # factor | model
    kind: str = ""
    maturity: str = ""
    model_id: str = ""
    source: str = ""
    n_days: int = 0
    recent_n_days: int = 0
    ic_mean_full: float = float("nan")
    ic_mean_recent: float = float("nan")
    ic_drift: float = float("nan")
    ic_ir_recent: float = float("nan")
    ic_t_nw_recent: float = float("nan")
    ic_p_nw_recent: float = float("nan")
    ic_win_rate_recent: float = float("nan")
    # 双窗口长窗（默认 252 交易日 ≈ 12 个月）：60d 提前预警、252d 稳健确认
    ic_mean_recent_252: float = float("nan")
    ic_drift_252: float = float("nan")
    ic_ir_recent_252: float = float("nan")
    ic_t_nw_recent_252: float = float("nan")
    ic_p_nw_recent_252: float = float("nan")
    recent_n_days_252: int = 0
    expected_ic: float = float("nan")
    frozen_baseline: float = float("nan")
    ic_retention: float = float("nan")
    ic_retention_252: float = float("nan")
    coverage_recent: float = float("nan")
    monotonicity_recent: float = float("nan")
    ls_daily_recent: float = float("nan")
    # 中性化 IC（风格剥离后的纯 Alpha IC）：区分风格轮动 vs alpha 衰减
    ic_neutral_mean_full: float = float("nan")
    ic_neutral_mean_recent: float = float("nan")
    ic_neutral_drift: float = float("nan")
    ic_neutral_retention: float = float("nan")
    style_exposure_ratio: float = float("nan")  # raw IC 中风格可解释的比例
    stale_days: int = 0
    # 组合/信号层监控字段（category="signal" 时使用）
    signal_date: str = ""
    n_signal_stocks: int = 0
    hhi_recent: float = float("nan")
    net_turnover: float = float("nan")
    blocked_ratio: float = float("nan")
    signal_freshness_days: int = 0
    # 库级拥挤度监控字段（category="crowd" 时使用）
    corr_n: int = 0
    corr_mean_full: float = float("nan")
    pc1_share_full: float = float("nan")
    status: str = "normal"
    alerts: list = field(default_factory=list)

    @property
    def suffix_short(self) -> int:
        return self.recent_n_days or 60

    @property
    def source_group(self) -> str:
        """来源组：source 前缀（如 "alpha101:..." → "alpha101"），空 → "(未标注)"。"""
        return self.source.split(":")[0] if self.source else "(未标注)"

    @property
    def suffix_long(self) -> int:
        return self.recent_n_days_252 or self.recent_n_days or 252

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "alerts"}
        d["n_alerts"] = len(self.alerts)
        return d


def load_close_panel(cache_root: str | Path) -> pd.DataFrame:
    """从日线缓存读 close 宽表（date×code），尽量乘后复权因子对齐库口径。"""
    root = Path(cache_root)
    daily = pd.read_parquet(root / "daily_hs300.parquet")
    close = daily["close"].unstack("code").sort_index()
    bf_path = root / "backward_factor.parquet"
    if bf_path.exists():
        bf = pd.read_parquet(bf_path)
        bf = bf.reindex(index=close.index, columns=close.columns).ffill()
        close = close * bf
    return close


def load_returns_panel(
    close_panel: pd.DataFrame, as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """h=1 次日收益面板（因子库 IC 口径：pct_change().shift(-1)，不虚构缺口收益）。"""
    close = close_panel.loc[:as_of] if as_of is not None else close_panel
    return close.pct_change(fill_method=None).shift(-1)


def _recent_window(
    panel_index: pd.DatetimeIndex, as_of: pd.Timestamp, window: int
) -> pd.DatetimeIndex:
    days = panel_index[panel_index <= as_of]
    return days[-window:] if len(days) > window else days


def _parse_model_id(note: str) -> str:
    m = _MODEL_ID_RE.search(str(note))
    return m.group(1) if m else ""


def quantile_monotonicity(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    n_quantiles: int = 5,
) -> tuple[float, float]:
    """分位结构：返回 (单调性, Q5-Q1 日均)。

    单调性 = Q1..Qn 日均收益与组序号的 Spearman 相关（+1 完美正向，
    -1 完美反向 —— GP 因子负 IC 属正常反向，模型因子应为正向）。
    """
    qb = quantile_backtest(factor_panel, returns_panel, n_quantiles=n_quantiles)
    ret = qb.diff().dropna(how="all")
    if ret.empty or len(ret.columns) < 2:
        return float("nan"), float("nan")
    mean_ret = ret.mean(axis=0).values
    mono = float(sps.spearmanr(np.arange(1, len(mean_ret) + 1), mean_ret).statistic)
    hi, lo = ret.columns[-1], ret.columns[0]
    ls = float((ret[hi] - ret[lo]).mean())
    return mono, ls


def _pick_baseline(m: MonitorMetrics) -> float:
    """选择保留率基线：注册时的**冻结基线**（|IC| 足够显著时）→ 否则全期基线。

    冻结基线 = 注册时入库的 ic_mean（上线时的 OOS 表现，天然不含近期，杜绝
    "全期含近期"稀释）。但基线若本就近 0（因子无显著 IC 可衰），相对基线的
    语义不成立，回退全期，避免近零基线放大数值。
    """
    if m.expected_ic == m.expected_ic and abs(m.expected_ic) >= _FROZEN_BASELINE_MIN_ABS:
        return m.expected_ic
    return m.ic_mean_full


def compute_factor_metrics(
    name: str,
    registry_row: pd.Series,
    factor_panel: pd.DataFrame,
    ic_series: pd.Series,
    returns_panel: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = 60,
    window_long: int = 252,
    style_covariates: dict[str, pd.DataFrame] | None = None,
) -> MonitorMetrics:
    """计算单个因子的监控指标（registry_row 提供 kind/maturity/note/ic_mean）。

    双窗口口径：``window``（默认 60d）做近期提前预警（灵敏），
    ``window_long``（默认 252d）做长窗稳健确认（抗噪音）。

    中性化 IC（style_covariates 传入时计算）：对因子做 5 因子风格中性化取残差
    后再算 IC，产出 ic_neutral_mean_* / ic_neutral_drift / style_exposure_ratio。
    raw IC 衰减但 neutral IC 稳定 → 风格轮动（告警语义"重新中性化"而非"因子失效"）；
    两者同时衰减 → 真 alpha 衰减（因子应退出）。
    """
    is_model = name.startswith(MODEL_PREFIX)
    m = MonitorMetrics(
        name=name,
        category="model" if is_model else "factor",
        kind=str(registry_row.get("kind") or ""),
        maturity=str(registry_row.get("maturity") or ""),
        model_id=_parse_model_id(str(registry_row.get("note") or "")),
        source=str(registry_row.get("source") or ""),
    )

    ic = ic_series.dropna()
    ic = ic[ic.index <= as_of]
    if len(ic) >= 2:
        base = monitor_ic_series(ic, window=window)
        m.n_days = base["n_days"]
        m.recent_n_days = base["recent_n_days"]
        m.ic_mean_full = base["ic_mean_full"]
        m.ic_mean_recent = base["ic_mean_recent"]
        m.ic_drift = base["ic_drift"]
        m.ic_ir_recent = base["ic_ir_recent"]
        m.ic_t_nw_recent = base["ic_t_nw_recent"]
        m.ic_p_nw_recent = base["ic_p_nw_recent"]

        base252 = monitor_ic_series(ic, window=window_long)
        m.recent_n_days_252 = base252["recent_n_days"]
        m.ic_mean_recent_252 = base252["ic_mean_recent"]
        m.ic_drift_252 = base252["ic_drift"]
        m.ic_ir_recent_252 = base252["ic_ir_recent"]
        m.ic_t_nw_recent_252 = base252["ic_t_nw_recent"]
        m.ic_p_nw_recent_252 = base252["ic_p_nw_recent"]

    reg_ic = registry_row.get("ic_mean")
    m.expected_ic = float(reg_ic) if reg_ic is not None and reg_ic == reg_ic else float("nan")
    if m.expected_ic == m.expected_ic and abs(m.expected_ic) >= _FROZEN_BASELINE_MIN_ABS:
        m.frozen_baseline = m.expected_ic
    baseline = _pick_baseline(m)
    if baseline == baseline and abs(baseline) > 1e-12:
        if m.ic_mean_recent == m.ic_mean_recent:
            m.ic_retention = m.ic_mean_recent / baseline
        if m.ic_mean_recent_252 == m.ic_mean_recent_252:
            m.ic_retention_252 = m.ic_mean_recent_252 / baseline

    recent_days = _recent_window(factor_panel.index, as_of, window)
    if len(recent_days):
        sub = factor_panel.loc[recent_days]
        ret_recent = returns_panel.reindex(index=recent_days)
        # 覆盖率按日计算：每天只在当天有收益的股票（当期可交易成分）上算
        # notna 比例，再取均值。避免 PIT 并集池中历史成分股拉低分母。
        mask_ret = ret_recent.notna()
        mask_fac = sub.notna()
        daily_denom = (mask_ret & mask_fac).sum(axis=1)
        daily_valid = mask_fac.sum(axis=1)
        # 有效天数 = 当天因子有值且收益也有值的股票数 / 当天因子有值的股票数
        ratio = (daily_denom / daily_valid.replace(0, np.nan)).dropna()
        m.coverage_recent = float(ratio.mean()) if len(ratio) else float(
            sub.notna().to_numpy().mean())
        mono, ls = quantile_monotonicity(sub, ret_recent)
        m.monotonicity_recent = mono
        m.ls_daily_recent = ls

        last_valid = factor_panel.index[factor_panel.notna().any(axis=1)]
        if len(last_valid):
            m.stale_days = _stale_days(last_valid[-1], as_of, returns_panel.index)

    # 中性化 IC：风格剥离后的纯 Alpha IC（style_covariates 传入时计算）
    if style_covariates:
        try:
            neutral_ic = calc_neutral_ic_series(
                factor_panel, returns_panel,
                style_covariates=style_covariates,
            )
            neutral_ic = neutral_ic.dropna()
            neutral_ic = neutral_ic[neutral_ic.index <= as_of]
            if len(neutral_ic) >= 2:
                from factor.preprocessing import neutralize as _neutralize_fn
                # 全期中性化 IC 均值
                m.ic_neutral_mean_full = float(neutral_ic.mean())
                # 近期窗口
                recent_neutral = neutral_ic[neutral_ic.index.isin(
                    _recent_window(neutral_ic.index, as_of, window)
                )]
                if len(recent_neutral):
                    m.ic_neutral_mean_recent = float(recent_neutral.mean())
                # 中性化 IC 漂移（同 raw IC 口径）
                if m.ic_neutral_mean_full == m.ic_neutral_mean_full:
                    m.ic_neutral_drift = (
                        m.ic_neutral_mean_recent - m.ic_neutral_mean_full
                    )
                # 风格暴露比例 = 1 - neutral_ic / raw_ic（raw IC 中多少比例来自风格）
                if (m.ic_mean_full == m.ic_mean_full
                        and abs(m.ic_mean_full) > 1e-6
                        and m.ic_neutral_mean_full == m.ic_neutral_mean_full):
                    m.style_exposure_ratio = 1.0 - (
                        m.ic_neutral_mean_full / m.ic_mean_full
                    )
                # 中性化 IC 保留率
                baseline = _pick_baseline(m)
                if (baseline == baseline and abs(baseline) > 1e-12
                        and m.ic_neutral_mean_recent == m.ic_neutral_mean_recent):
                    m.ic_neutral_retention = (
                        m.ic_neutral_mean_recent / baseline
                    )
        except Exception:
            pass  # 中性化失败不影响基础监控

    return m


def _stale_days(
    panel_last: pd.Timestamp,
    as_of: pd.Timestamp,
    calendar: pd.DatetimeIndex,
) -> int:
    """面板最后有效日落后 as_of 的交易日数（以收益面板日历为交易日真源）。"""
    ref = min(as_of, calendar[-1]) if len(calendar) else as_of
    if ref <= panel_last:
        return 0
    return int(((calendar > panel_last) & (calendar <= ref)).sum())
