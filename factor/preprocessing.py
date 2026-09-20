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
# neutralize 的实现选择（2026-09-20）：
#   "batch"（默认）= :func:`neutralize_batch`：一次性 numpy 预计算 + 逐日 lstsq，
#                    与历史逐日 pandas 版**逐位一致**（真实全A面板 max|Δ| = 0），
#                    实测 2.6x（86 因子 23.3 → 6.4 分钟）。
#   "loop"        = 历史实现，保留为回退（复现旧结果 / 排查异常时用）。
# 想要更激进的加速（8.3x，代价见 :func:`neutralize_grouped` 的 docstring）
# 请显式调用 neutralize_grouped，**不改默认**。
_NEUTRALIZE_IMPL = "batch"


def set_neutralize_impl(name: str) -> None:
    """切换 :func:`neutralize` 的实现：``"batch"``（默认）或 ``"loop"``。"""
    global _NEUTRALIZE_IMPL
    if name not in ("batch", "loop"):
        raise ValueError(f"未知的 neutralize 实现: {name}")
    _NEUTRALIZE_IMPL = name


def get_neutralize_impl() -> str:
    """:func:`neutralize` 当前使用的实现名。"""
    return _NEUTRALIZE_IMPL


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
    - **无行业哑变量时补一列常数列作截距**（2026-09-11 第二批 c 项修复）：
      未传行业、或当天样本不足以容纳行业哑变量而将其丢弃时，都会落到"只有
      市值"的设计矩阵。此前该矩阵仅 ``[log(mc)]`` 一列，等价于**过原点回归**，
      与含截距的 :func:`neutralize_single` 差一个截距项；现在补 ``_intercept``
      列，两者口径一致。含行业哑变量时不补（列和已 span 截距，重复加列会共线）。
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

    if _NEUTRALIZE_IMPL != "loop":
        return neutralize_batch(
            panel, market_cap_panel=market_cap_panel,
            industry_panel=industry_panel, log_market_cap=log_market_cap,
            min_samples_size_only=min_samples_size_only,
            rank_margin=rank_margin, extra_covariates=extra_covariates)

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
        has_industry = False
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
                has_industry = True

        for nm, s in extra_x.items():
            cov_series = pd.Series(
                s[valid].astype(float).values, index=codes_valid, name=nm
            )
            cols.append(cov_series)

        if not cols:
            # 没有任何回归变量（比如只传了行业但样本太少被丢弃），退化为原值
            result.loc[d, codes_valid] = y_valid.values
            continue

        # 无行业哑变量时补一列常数列作**截距**（2026-09-11 第二批 c 项修复）：
        # 此前只传市值时设计矩阵仅 [log(mc)]，等价于过原点回归，与含截距的
        # neutralize_single 差一个截距项。含行业哑变量时其列和本身即全 1 向量、
        # 已 span 截距，不再重复加列（避免共线）。
        if not has_industry:
            cols.append(pd.Series(np.ones(len(codes_valid)), index=codes_valid,
                                  name="_intercept"))

        x_matrix = np.column_stack([c.values for c in cols])
        beta, *_ = np.linalg.lstsq(x_matrix, y_valid.values, rcond=None)
        resid = y_valid.values - x_matrix @ beta
        result.loc[d, codes_valid] = resid

    return result


def neutralize_batch(
    panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    log_market_cap: bool = True,
    min_samples_size_only: int = 2,
    rank_margin: int = 3,
    extra_covariates: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """:func:`neutralize` 的向量化等价实现（**逐日 lstsq 保留**，只砍 pandas 外壳）。

    与 :func:`neutralize` 的区别仅在实现路径，口径逐条对齐：

    - **样本掩码**逐日一致：``y``/市值/行业/协变量的缺失位置同样被剔除；
      某天不在某个协变量面板的 index 里时，该协变量**当天整体不参与**
      （老实现是 ``d not in panel.index`` 就不加这一项，不是置 NaN）。
    - **设计矩阵逐元素相同**：``[size] + [当天出现行业的哑变量(升序)] + [协变量]
      + [截距(仅当无行业时)]``，列顺序与 ``pd.get_dummies(drop_first=False)`` 一致。
    - **降级判据相同**：``n_valid >= (n_size + n_行业 + n_协变量) + rank_margin``，
      其中 ``n_行业`` 用**当天实际出现的行业数**（不是全局行业数），
      因此"样本不足 → 丢行业"的触发天与老实现完全一致。
    - **求解器相同**：仍逐日 ``np.linalg.lstsq(..., rcond=None)``，输入 dtype 也
      保持在 float64（老实现 ``y[valid].astype(float)`` / ``size[valid].astype(float)``；
      市值先按 float32 取 log 再升位，与生产 float32 面板同精度）。

    因此残差应与 :func:`neutralize` **逐位一致**（``max|Δ| = 0``），由
    ``scripts/oneoff/probe_neutralize_batch.py`` 在真实面板上锁定。

    性能：真实全A形态（801 日 × 5801 只 × 31 行业）下，逐日 ``lstsq`` 只占
    总耗时 ~41%，其余 ~59% 是 pandas 的逐日取行 / ``get_dummies`` /
    ``column_stack`` / ``result.loc[d, codes] = resid`` 写回。本函数把这四块
    换成一次性的 numpy 预计算 + 逐日切片，实测单因子 16.5s → 见探针输出。

    Args / Returns 同 :func:`neutralize`。
    """
    if (market_cap_panel is None and industry_panel is None
            and not extra_covariates):
        return panel

    index = panel.index
    columns = panel.columns
    n_day, n_code = panel.shape

    # ---- 因变量：保留原 dtype（生产为 float32），逐日再升 float64 ----
    y_arr = panel.to_numpy()
    y_na = panel.isna().to_numpy()

    # ---- 市值：int 提升为 float；log 在**原浮点精度**上算（与老实现同）----
    size = None
    mc_day = None
    if market_cap_panel is not None:
        raw = market_cap_panel.reindex(index=index, columns=columns).to_numpy()
        if not np.issubdtype(raw.dtype, np.floating):
            raw = raw.astype(np.float64)
        if log_market_cap:
            nonpos = ~(raw > 0)          # NaN / 0 / 负值 → 置 NaN（同 mc.where(mc>0)）
            if nonpos.any():
                raw = raw.copy()
                raw[nonpos] = np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                size = np.log(raw)
        else:
            size = raw
        mc_day = index.isin(market_cap_panel.index)

    # ---- 行业：整数编码；编码按**全局排序**，保证逐日列序 = get_dummies 的子序列 ----
    IND = None
    ind_day = None
    if industry_panel is not None:
        flat = industry_panel.reindex(index=index, columns=columns).to_numpy().ravel()
        codes, _ = pd.factorize(flat, sort=True)   # NaN → -1，且不计入类别
        IND = codes.reshape(n_day, n_code)
        ind_day = index.isin(industry_panel.index)

    # ---- 额外协变量：逐面板 reindex，保留原浮点精度 ----
    ex_arrays: dict[str, np.ndarray] = {}
    ex_day: dict[str, np.ndarray] = {}
    if extra_covariates:
        for nm, ep in extra_covariates.items():
            a = ep.reindex(index=index, columns=columns).to_numpy()
            if not np.issubdtype(a.dtype, np.floating):
                a = a.astype(np.float64)
            ex_arrays[nm] = a
            ex_day[nm] = index.isin(ep.index)

    out = np.full((n_day, n_code), np.nan, dtype=np.float64)
    for i in range(n_day):
        has_mc = size is not None and mc_day[i]
        has_ind = IND is not None and ind_day[i]
        ex_here = [nm for nm in ex_arrays if ex_day[nm][i]]

        valid = ~y_na[i]
        if has_mc:
            valid &= ~np.isnan(size[i])
        if has_ind:
            valid &= IND[i] >= 0
        for nm in ex_here:
            valid &= ~np.isnan(ex_arrays[nm][i])

        n_valid = int(valid.sum())
        if n_valid < min_samples_size_only:
            continue  # 该天全部保持 NaN

        y_valid = y_arr[i][valid].astype(np.float64)

        cols_list: list[np.ndarray] = []
        if has_mc:
            cols_list.append(size[i][valid].astype(np.float64))

        has_industry = False
        if has_ind:
            ids = IND[i][valid]
            present = np.unique(ids)
            n_params = len(cols_list) + present.size + len(ex_here)
            if n_valid >= n_params + rank_margin:
                for k in present:
                    cols_list.append((ids == k).astype(np.float64))
                has_industry = True

        for nm in ex_here:
            cols_list.append(ex_arrays[nm][i][valid].astype(np.float64))

        if not cols_list:
            out[i, valid] = y_valid
            continue

        if not has_industry:
            cols_list.append(np.ones(n_valid, dtype=np.float64))

        x_matrix = np.column_stack(cols_list)
        beta, *_ = np.linalg.lstsq(x_matrix, y_valid, rcond=None)
        out[i, valid] = y_valid - x_matrix @ beta

    return pd.DataFrame(out, index=index, columns=columns)


def neutralize_grouped(
    panel: pd.DataFrame,
    market_cap_panel: pd.DataFrame | None = None,
    industry_panel: pd.DataFrame | None = None,
    log_market_cap: bool = True,
    min_samples_size_only: int = 2,
    rank_margin: int = 3,
    extra_covariates: dict[str, pd.DataFrame] | None = None,
    rcond: float = 1e-12,
) -> pd.DataFrame:
    """完全向量化的 neutralize：``(日, 行业)`` 分组聚合 + 批量伪逆，**零 python 循环**。

    与前两个实现的关系（三者语义对齐，差别在解法与数值行为）::

        neutralize          → 默认分发到 neutralize_batch（逐位一致）
        neutralize_batch    纯 numpy 外壳 + 逐日 lstsq        max|Δ| = 0（严格逐位）
        neutralize_grouped  分组聚合 + 批量 pinv(A) @ b        max|Δ| ≈ 1e-14~1e-8

    为什么能去掉循环
    ----------------
    行业哑变量是 one-hot，每行恰有一个 1，于是设计矩阵 ``X = [size, D, extras]``
    的 ``X'X`` / ``X'y`` 全部退化成**分组求和**：

        D'D = diag(n_k)          D'size = Σ_{i∈k} size_i      D'y = Σ_{i∈k} y_i

    这些用 ``np.bincount`` 按 ``(day, industry)`` 一次算完（O(n)，与行业数无关），
    当天没出现的行业列填 0。零列不改变 ``col(X)``，而最小二乘残差只取决于列空间，
    故残差等价；"样本不足丢行业"的降级判据仍按**当天出现的行业数**计算，
    降级天行业列整块清零、截距列激活（正是老实现补 ones 的语义）；
    "一个回归变量都没有"的天把 A/b 整块清零（β = 0 → 残差 = y，对齐老实现的
    ``if not cols: 返回原值``）。

    ⚠ 代价（**必须显式接受**）
    -------------------------
    解法从 "SVD on X" 变成 "pinv on X'X"，条件数被**平方**，因此残差与
    :func:`neutralize` **不是逐位相同**：

    - 合成边界用例（16 组，含降级 / 缺日 / 极小面板）：``max|Δ| ≤ 2.2e-13``；
    - 真实全A面板（801×5801）：``max|Δ|`` ≈ 1e-12 ~ 1.1e-8，
      其中 1e-8 那档来自 ``κ(X)² ≈ 2.8e5`` 的放大（实测 κ(X) 中位 527、max 561）；
    - **截面名次位移实测为 0**（对 5801 只全池做 rank 比对，全 801 天 max 位移 0 名）。

    因此它适合"面板大、调用次数多、只关心名次/信号形态"的场景；
    需要与历史结果严格对齐（复现实验、审计口径）时用默认的 :func:`neutralize`。

    ``rcond`` 是 ``A = X'X`` 上伪逆的相对阈值。注意它作用在 X'X 上等价于在 X 上
    放大成 ``√rcond``，与 ``lstsq(rcond=None)``（``eps·n``）语义不同；
    实测在 κ(X) ≤ 561 的真实面板上该参数从 1e-8 到 0 结果完全一致，
    仅在人为构造的病态用例里才有影响。

    性能（真实 801 日 × 5801 只 × 31 行业，单因子）：

    ==================  ========  =========
    实现                耗时      相对加速
    ==================  ========  =========
    neutralize(loop)    16.5s     1.0x
    neutralize_batch     5.4s     3.0x
    neutralize_grouped   1.4s     8.3x（86 因子外推 16.6 → 2.0 分钟）
    ==================  ========  =========

    等价性与计时由 ``scripts/oneoff/probe_neutralize_batch.py`` 与
    ``scripts/oneoff/probe_neutralize_phase2.py`` 在真实面板上锁定。

    Args / Returns 同 :func:`neutralize`。
    """
    if (market_cap_panel is None and industry_panel is None
            and not extra_covariates):
        return panel

    index, columns = panel.index, panel.columns
    n_day, n_code = panel.shape
    Y64 = panel.to_numpy().astype(np.float64)
    y_na = panel.isna().to_numpy()

    # ---- 市值（log 在原始浮点精度上算，与逐日实现同）----
    size = None
    S64 = None
    mc_day = np.ones(n_day, dtype=bool)
    if market_cap_panel is not None:
        raw = market_cap_panel.reindex(index=index, columns=columns).to_numpy()
        if not np.issubdtype(raw.dtype, np.floating):
            raw = raw.astype(np.float64)
        if log_market_cap:
            nonpos = ~(raw > 0)
            if nonpos.any():
                raw = raw.copy()
                raw[nonpos] = np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                size = np.log(raw)
        else:
            size = raw
        mc_day = index.isin(market_cap_panel.index)
        S64 = size.astype(np.float64)

    # ---- 行业：全局整数编码（按值排序，保证列序与 get_dummies 的子序列一致）----
    IND = None
    n_ind = 0
    ind_day = np.ones(n_day, dtype=bool)
    if industry_panel is not None:
        flat_ind = industry_panel.reindex(
            index=index, columns=columns).to_numpy().ravel()
        codes, uniq = pd.factorize(flat_ind, sort=True)
        n_ind = len(uniq)
        IND = codes.reshape(n_day, n_code)
        ind_day = index.isin(industry_panel.index)

    # ---- 额外协变量 ----
    ex_names: list[str] = []
    E64: list[np.ndarray] = []
    ex_day: dict[str, np.ndarray] = {}
    if extra_covariates:
        for nm, ep in extra_covariates.items():
            a = ep.reindex(index=index, columns=columns).to_numpy()
            if not np.issubdtype(a.dtype, np.floating):
                a = a.astype(np.float64)
            ex_names.append(nm)
            E64.append(a.astype(np.float64))
            ex_day[nm] = index.isin(ep.index)
    n_ex = len(ex_names)

    # ---- 逐样本有效掩码（某天不在协变量面板 index 里 → 该协变量当天整体不参与）----
    valid = ~y_na
    if size is not None:
        valid &= (~np.isnan(size)) | (~mc_day)[:, None]
    if IND is not None:
        valid &= (IND >= 0) | (~ind_day)[:, None]
    for nm, arr in zip(ex_names, E64):
        valid &= (~np.isnan(arr)) | (~ex_day[nm])[:, None]

    n_valid = valid.sum(axis=1)
    keep = n_valid >= min_samples_size_only
    # 聚合前先把无效值清成 0：该天不参与时 valid 不要求非 NaN，直接参与
    # bincount / 求和会把 NaN 灌进 X'X（求解期表现为 SVD 不收敛）。
    S64z = None if S64 is None else np.where(np.isnan(S64), 0.0, S64)
    E64z = [np.where(np.isnan(a), 0.0, a) for a in E64]

    # ---- (日, 行业) 分组聚合；pair 末位留一个垃圾桶 bin ----
    day_of = np.repeat(np.arange(n_day), n_code)
    n_bin = n_ind + 1
    cnt = None
    if IND is not None:
        ind_safe = np.where(IND >= 0, IND, 0).ravel()
        pair = day_of * n_bin + ind_safe
        pair = np.where(valid.ravel(), pair, day_of * n_bin + n_ind)
        flat_len = n_day * n_bin

        def _gsum(w: np.ndarray) -> np.ndarray:
            return np.bincount(pair, weights=w.ravel(),
                               minlength=flat_len).reshape(n_day, n_bin)[:, :n_ind]

        cnt = np.bincount(pair, minlength=flat_len).reshape(
            n_day, n_bin)[:, :n_ind].astype(np.float64)
    else:
        def _gsum(w: np.ndarray) -> np.ndarray:  # type: ignore[misc]
            raise AssertionError("无行业面板时不应调用 _gsum")

    def _dsum(a: np.ndarray) -> np.ndarray:
        return np.where(valid, a, 0.0).sum(axis=1)

    # ---- 列布局：[size?] + [31 个全局行业] + [extras...] + [intercept] ----
    col_size = 0 if size is not None else None
    base = 1 if size is not None else 0
    ind_lo = base
    ex_lo = base + n_ind
    col_int = ex_lo + n_ex
    n_par = col_int + 1

    A = np.zeros((n_day, n_par, n_par), dtype=np.float64)
    b = np.zeros((n_day, n_par), dtype=np.float64)

    if size is not None:
        s1 = _dsum(S64z)
        A[:, col_size, col_size] = _dsum(S64z * S64z)
        b[:, col_size] = _dsum(S64z * Y64)
        A[:, col_size, col_int] = s1
        A[:, col_int, col_size] = s1

    if IND is not None:
        di = np.arange(n_ind)
        A[:, ind_lo + di, ind_lo + di] = cnt
        b[:, ind_lo:ind_lo + n_ind] = _gsum(np.where(valid, Y64, 0.0))
        A[:, ind_lo:ind_lo + n_ind, col_int] = cnt
        A[:, col_int, ind_lo:ind_lo + n_ind] = cnt
        if size is not None:
            ind_s = _gsum(np.where(valid, S64z, 0.0))
            A[:, col_size, ind_lo:ind_lo + n_ind] = ind_s
            A[:, ind_lo:ind_lo + n_ind, col_size] = ind_s

    for m in range(n_ex):
        j = ex_lo + m
        e1 = _dsum(E64z[m])
        A[:, j, j] = _dsum(E64z[m] * E64z[m])
        b[:, j] = _dsum(E64z[m] * Y64)
        A[:, j, col_int] = e1
        A[:, col_int, j] = e1
        if size is not None:
            se = _dsum(S64z * E64z[m])
            A[:, col_size, j] = se
            A[:, j, col_size] = se
        if IND is not None:
            ind_e = _gsum(np.where(valid, E64z[m], 0.0))
            A[:, ind_lo:ind_lo + n_ind, j] = ind_e
            A[:, j, ind_lo:ind_lo + n_ind] = ind_e
        for m2 in range(m + 1, n_ex):
            cr = _dsum(E64z[m] * E64z[m2])
            A[:, j, ex_lo + m2] = cr
            A[:, ex_lo + m2, j] = cr

    A[:, col_int, col_int] = n_valid.astype(np.float64)
    b[:, col_int] = _dsum(Y64)

    # ---- 激活掩码：与逐日实现的"当天列集"一字对齐 ----
    if IND is not None:
        size_cnt = mc_day.astype(np.float64) if size is not None else np.zeros(n_day)
        ex_cnt = np.zeros(n_day)
        for nm in ex_names:
            ex_cnt += ex_day[nm].astype(np.float64)
        k_d = (cnt > 0).sum(axis=1).astype(np.float64)
        use_ind = ind_day & (n_valid >= size_cnt + k_d + ex_cnt + rank_margin)

        m_ind = use_ind[:, None, None].astype(np.float64)
        A[:, ind_lo:ind_lo + n_ind, :] *= m_ind
        A[:, :, ind_lo:ind_lo + n_ind] *= m_ind
        b[:, ind_lo:ind_lo + n_ind] *= m_ind[:, :, 0]
        m_int = (~use_ind)[:, None].astype(np.float64)
    else:
        use_ind = np.zeros(n_day, dtype=bool)
        m_int = np.ones((n_day, 1), dtype=np.float64)

    A[:, col_int, :] *= m_int
    A[:, :, col_int] *= m_int
    b[:, col_int] *= m_int[:, 0]
    if size is not None and not mc_day.all():
        m = mc_day[:, None].astype(np.float64)
        A[:, col_size, :] *= m
        A[:, :, col_size] *= m
        b[:, col_size] *= m[:, 0]
    for m_i, nm in enumerate(ex_names):
        if not ex_day[nm].all():
            m = ex_day[nm][:, None].astype(np.float64)
            A[:, ex_lo + m_i, :] *= m
            A[:, :, ex_lo + m_i] *= m
            b[:, ex_lo + m_i] *= m[:, 0]

    # "一个回归变量都没有"的天：逐日实现在补截距**之前**就返回原值，此处把 A/b
    # 整块清零使 β = 0、残差 = y（否则会退化成"减均值"，与老实现分叉）。
    any_col = np.zeros(n_day, dtype=bool)
    if size is not None:
        any_col |= mc_day
    if IND is not None:
        any_col |= use_ind
    for nm in ex_names:
        any_col |= ex_day[nm]
    m_any = any_col[:, None].astype(np.float64)
    A *= m_any[:, :, None]
    b *= m_any

    # ---- 批量伪逆：pinv(A) @ b，零 python 循环 ----
    U, sv, Vt = np.linalg.svd(A)
    cut = rcond * sv[:, :1]
    sinv = np.where(sv > cut, 1.0 / np.where(sv > 0.0, sv, 1.0), 0.0)
    tmp = np.einsum("dji,dj->di", U, b)      # U^T b
    tmp *= sinv
    beta = np.einsum("dji,dj->di", Vt, tmp)  # V (S^+ U^T b)

    # ---- 残差 ----
    pred = np.zeros_like(Y64)
    if size is not None:
        pred += beta[:, col_size][:, None] * S64z
    if IND is not None:
        bi = beta[:, ind_lo:ind_lo + n_ind]
        pred += bi[np.arange(n_day)[:, None], np.where(IND >= 0, IND, 0)]
    for m_i in range(n_ex):
        pred += beta[:, ex_lo + m_i][:, None] * E64z[m_i]
    pred += beta[:, col_int][:, None]

    out = np.where(valid & keep[:, None], Y64 - pred, np.nan)
    return pd.DataFrame(out, index=index, columns=columns)


def neutralize_single(
    panel: pd.DataFrame,
    covariate: pd.DataFrame,
    log_covariate: bool = True,
) -> pd.DataFrame:
    """**逐行向量化**的单协变量截面中性化（「panel ~ 截距 + covariate」取残差）。

    这是 :func:`neutralize` 在**单一连续协变量**情形下的快速专化版本：行中心化后
    单变量回归无截距项，``beta = Σ(xa·ya)/Σ(xa²)``，残差 = ``ya − beta·xa``，
    一次矩阵运算算完全部日期——毫秒级，而 :func:`neutralize` 的逐日 lstsq 约
    ~1s/因子。GFlowNet 训练要在同一面板上反复评估成千上万个候选因子，用哪个是
    可行性问题而不是风格问题。

    Args:
        panel: date×code 因子面板。
        covariate: 同形状协变量面板（如市值）。
        log_covariate: 是否先取对数（市值默认取 log）。

    Returns:
        残差面板；``panel`` 或 ``covariate`` 任一为 NaN 的位置为 NaN。

    与 :func:`neutralize` 的关系（2026-09-10 收口，此前
    ``factor.gflownet.reward.neutralize_market_cap`` 自实现一份、本模块另有
    逐日 lstsq 版，同一件事两处维护）：

    - 本函数 = **含截距**的单协变量回归残差（行中心化后单变量回归等效于带截距），
      也是市值中性化的标准做法（截面回归含常数项）。回归热路径用它。
    - :func:`neutralize` 是**多协变量**版（行业哑变量 + 市值 + 风格协变量），
      逐日 lstsq，约 ~1s/因子。⚠️ **单独只传市值时它只有 ``[log(mc)]`` 一列、
      不含截距**——截距靠"全量行业哑变量的列和 = 全 1 向量"来 span，不传行业
      就没有。因此**纯市值情形两者不等价，差一个截距项**（见 2026-09-10 审计
      待办：是否需要给 size-only 路径补 ones 列）。

    选型：需要行业 / 多协变量 / 小样本降级 → ``neutralize``；
    只有一个连续协变量且要跑成千上万次 → 本函数。
    本函数与"含截距 lstsq"的等价性由 ``tests/test_gflownet_phase1`` 锁定。
    """
    x = covariate.reindex_like(panel)
    if log_covariate:
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(x)
    m = panel.notna() & x.notna()
    fp = panel.where(m)
    x = x.where(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        fa = fp.sub(fp.mean(axis=1), axis=0)      # 逐行中心化（axis=0 行向广播）
        xa = x.sub(x.mean(axis=1), axis=0)
        beta = (fa * xa).sum(axis=1) / (xa * xa).sum(axis=1)
        resid = fa.sub(xa.mul(beta, axis=0), axis=0)
    return resid


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

    与 scripts/factors/mine_factors.py:_build_htai_neutral_panels 功能等价但
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
