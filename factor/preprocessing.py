"""
因子预处理
==========

多因子研究的标准前处理流程：去极值 -> 行业/市值中性化 -> 标准化。
所有函数接收/返回 DataFrame(index=date, columns=code)，按截面（axis=1，
即每一行/每个交易日）独立处理，不引入任何跨日信息（不会用到未来数据）。

用法
----
    # 完整流程（需要市值 + 行业面板）
    processed = preprocess_factor(factor_panel, market_cap_panel, industry_panel)

    # Mock 模式（没有市值/行业数据时自动跳过中性化）
    processed = preprocess_factor(factor_panel)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_CONSISTENCY_CONST = 1.4826  # 使 MAD 在正态分布下与标准差同尺度


# ===========================================================================
# 去极值
# ===========================================================================
def winsorize_mad(panel: pd.DataFrame, n_mad: float = 3.0,
                  consistency_scale: bool = True) -> pd.DataFrame:
    """按 MAD（中位数绝对偏差）去极值，逐日（截面）独立处理。

    Args:
        n_mad: 截断倍数。
        consistency_scale: 是否把 MAD 乘以 1.4826（正态一致性常数，使 MAD 与
            std 同尺度）。**华泰研报口径**：去极值用 ``中位数 ± 5×median(|X-中位数|)``
            **不乘** 1.4826 —— 复现研报时传 ``n_mad=5, consistency_scale=False``。
    """
    median = panel.median(axis=1)
    mad = panel.sub(median, axis=0).abs().median(axis=1)
    scaled_mad = mad * (_MAD_CONSISTENCY_CONST if consistency_scale else 1.0)
    lower = median - n_mad * scaled_mad
    upper = median + n_mad * scaled_mad
    return panel.clip(lower=lower, upper=upper, axis=0)


def winsorize_quantile(
    panel: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
) -> pd.DataFrame:
    """按分位数去极值，逐日（截面）独立处理。"""
    lower_bound = panel.quantile(lower, axis=1)
    upper_bound = panel.quantile(upper, axis=1)
    return panel.clip(lower=lower_bound, upper=upper_bound, axis=0)


# ===========================================================================
# 标准化
# ===========================================================================
def standardize_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """按日 z-score 标准化：(x - mean) / std。

    截面 std=0（恒定值因子，如 close/close=1.0）的行会被整行置 NaN，
    避免零信息量因子入库后产生全 NaN 面板。
    """
    mean = panel.mean(axis=1)
    std = panel.std(axis=1)
    return panel.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)


def standardize_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """按日排名标准化，映射到 [0, 1]（不做 rank-to-normal，见模块说明）。"""
    return panel.rank(axis=1, pct=True)


# ===========================================================================
# 中性化
# ===========================================================================
def neutralize(
    panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    log_market_cap: bool = True,
    min_samples_size_only: int = 2,
    rank_margin: int = 3,
    extra_covariates: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """逐日截面最小二乘回归，取残差（行业哑变量 + 对数市值 + 额外连续协变量）。

    没有传入 market_cap_panel 或 industry_panel 时，对应那一项自动跳过；
    两者都未传入且无 extra_covariates 时原样返回 panel（不做任何回归）。

    Args:
        extra_covariates: 额外的连续协变量面板 dict（如 20 日动量/换手率/波动率，
            对应华泰报告五因子中性化中的后三个风格因子），与原市值/行业一起回归取残差。

    回归设计：
    - 行业哑变量用全量哑变量（不 drop_first），不额外加截距列——全量哑变量
      的列和本身就是全1向量，已经span了截距的位置，不会漏掉任何一个
      行业的组均值。
    - 用 numpy.linalg.lstsq 而不是求逆/normal equation，遇到秩不足（比如
      当天截面里某个行业只有极少样本）时会自动退化到最小范数解而不报错。
    - 每天根据有效样本数 n_valid 相对参数数 n_params 的余量，分级降级：
      样本充足 -> 市值+行业+协变量完整模型；样本不足但 >= min_samples_size_only ->
      退化成只用市值回归（丢弃行业哑变量，避免小样本下几乎精确拟合、
      残差趋近于0把因子信号也一起抹掉）；样本过少（<2）-> 当天残差全部
      为 NaN，不编造数值。
    """
    if (market_cap_panel is None and industry_panel is None
            and not extra_covariates):
        return panel

    result = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)

    for d in panel.index:
        y = panel.loc[d]

        size_x = None
        if market_cap_panel is not None and d in market_cap_panel.index:
            mc = market_cap_panel.loc[d]
            size_x = np.log(mc.where(mc > 0)) if log_market_cap else mc

        ind_x = None
        if industry_panel is not None and d in industry_panel.index:
            ind_x = industry_panel.loc[d]

        extra_x: dict[str, pd.Series] = {}
        if extra_covariates:
            for nm, ep in extra_covariates.items():
                if d in ep.index:
                    extra_x[nm] = ep.loc[d]

        valid = y.notna()
        if size_x is not None:
            valid &= size_x.notna()
        if ind_x is not None:
            valid &= ind_x.notna()
        for s in extra_x.values():
            valid &= s.notna()

        n_valid = int(valid.sum())
        if n_valid < min_samples_size_only:
            continue  # 该天全部保持 NaN

        y_valid = y[valid].astype(float)
        codes_valid = y_valid.index

        cols = []
        if size_x is not None:
            size_series = pd.Series(
                size_x[valid].astype(float).values, index=codes_valid, name="size"
            )
            cols.append(size_series)

        if ind_x is not None:
            dummies = pd.get_dummies(ind_x[valid], drop_first=False)
            n_params = len(cols) + dummies.shape[1] + len(extra_x)
            if n_valid >= n_params + rank_margin:
                for col in dummies.columns:
                    dummy_series = pd.Series(
                        dummies[col].astype(float).values, index=codes_valid, name=col
                    )
                    cols.append(dummy_series)

        for nm, s in extra_x.items():
            cov_series = pd.Series(
                s[valid].astype(float).values, index=codes_valid, name=nm
            )
            cols.append(cov_series)

        if not cols:
            # 没有任何回归变量（比如只传了行业但样本太少被丢弃），退化为原值
            result.loc[d, codes_valid] = y_valid.values
            continue

        x_matrix = np.column_stack([c.values for c in cols])
        beta, *_ = np.linalg.lstsq(x_matrix, y_valid.values, rcond=None)
        resid = y_valid.values - x_matrix @ beta
        result.loc[d, codes_valid] = resid

    return result


# ===========================================================================
# 组合入口
# ===========================================================================
def preprocess_factor(
    factor_panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    winsorize: str | None = "mad",
    winsorize_kwargs: dict | None = None,
    neutralize_industry: bool = True,
    neutralize_size: bool = True,
    standardize: str | None = "zscore",
) -> pd.DataFrame:
    """标准预处理流程：去极值 -> 中性化 -> 标准化，每一步都可单独跳过。

    Mock 模式（market_cap_panel=None, industry_panel=None）下，即使
    neutralize_industry/neutralize_size 保持默认 True，中性化也会被自动
    跳过——不需要调用者记得手动关闭这两个开关。
    """
    result = factor_panel

    if winsorize == "mad":
        result = winsorize_mad(result, **(winsorize_kwargs or {}))
    elif winsorize == "quantile":
        result = winsorize_quantile(result, **(winsorize_kwargs or {}))
    elif winsorize is not None:
        raise ValueError(f"未知的 winsorize 方式: {winsorize}")

    mc_panel = market_cap_panel if neutralize_size else None
    ind_panel = industry_panel if neutralize_industry else None
    if mc_panel is not None or ind_panel is not None:
        result = neutralize(result, market_cap_panel=mc_panel, industry_panel=ind_panel)

    if standardize == "zscore":
        result = standardize_zscore(result)
    elif standardize == "rank":
        result = standardize_rank(result)
    elif standardize is not None:
        raise ValueError(f"未知的 standardize 方式: {standardize}")

    return result


# ===========================================================================
# 华泰五因子风格协变量面板
# ===========================================================================
def build_style_covariates(
    panel: dict[str, pd.DataFrame],
    market_cap_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    *,
    mom_window: int = 20,
    vol_window: int = 20,
    turn_window: int = 20,
) -> dict[str, pd.DataFrame]:
    """构建华泰五因子中性化协变量面板（行业 + 市值 + 动量 + 波动 + 流动性）。

    对应研报的五个中性化因子::

        size:     市值 = TOT_SHARE × 后复权 close（传入 market_cap_panel）
        industry: 申万一级行业映射（传入 industry_panel）
        mom:      过去 N 日收益率 = close.pct_change(N)
        vol:      过去 N 日波动率 = 日收益的 N 日滚动 std
        turn:     过去 N 日平均换手率 = (volume / TOT_SHARE) 的 N 日滚动均值

    与 scripts/mine_factors.py:_build_htai_neutral_panels 功能等价但
    提升为 factor 层公共函数，供 IC 计算、监控、入库评估复用。

    Args:
        panel: 原始面板 dict，至少包含 ``close`` 和 ``volume``；
               若有 ``tot_share``（总股本 PIT 面板）则用于算换手率，
               否则需通过 market_cap_panel 传入市值。
        market_cap_panel: 市值面板（date×code）。若为 None 则 size 不产出。
        industry_panel: 行业面板（date×code，值=行业名）。若为 None 则 industry 不产出。
        mom_window/vol_window/turn_window: 各自滚动窗口长度，默认 20 日。

    Returns:
        dict[str, pd.DataFrame]——key 为 ``size``/``industry``/``mom``/``vol``/``turn``，
        构建失败的因子自动从 dict 剔除（neutralize 只回归可用的部分）。
    """
    out: dict[str, pd.DataFrame] = {}
    close = panel.get("close")
    if close is None:
        return out

    # mom: 过去 N 日收益率
    try:
        out["mom"] = close.pct_change(mom_window, fill_method=None)
    except Exception:
        pass

    # vol: 日收益的 N 日滚动 std
    try:
        ret1 = close.pct_change(fill_method=None)
        out["vol"] = ret1.rolling(vol_window).std()
    except Exception:
        pass

    # turn: 换手率 = volume / TOT_SHARE 的 N 日滚动均值
    vol_panel = panel.get("volume")
    tot_share = panel.get("tot_share")
    if vol_panel is not None and tot_share is not None:
        try:
            turn = vol_panel.div(tot_share.where(tot_share > 0))
            out["turn"] = turn.rolling(turn_window).mean()
        except Exception:
            pass

    # size: 直接传入市值面板
    if market_cap_panel is not None:
        out["size"] = market_cap_panel

    # industry: 直接传入行业面板
    if industry_panel is not None:
        out["industry"] = industry_panel

    return out
