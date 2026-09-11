"""ML Stacking 合成 —— 模型层 ④ Trainer 的「合成器」实现。

把 N 个因子面板（特征）通过**带时序交叉验证的监督学习**压成 1 个复合因子
（date×code 预测面板）。这是模型层的活而非因子层的活：它拟合有监督模型，
而 ``factor/synthesis.py`` 只做确定性的因子组合（IC 加权 / PCA / Gram-Schmidt）。

提供四种 stacking 合成器（输入都为「已截面标准化」的单因子面板）：

- ``synthesize_stacking``                : ridge 闭式解，无三方依赖
- ``synthesize_stacking_gbdt``           : LightGBM 回归 + purged 时序 CV
- ``synthesize_stacking_gbdt_tuned``     : 上者 + optuna 嵌套 CV 调参
- ``synthesize_stacking_lambdarank``     : LightGBM LambdaRank（按日 query group）

共同纪律：expanding-window 时序 CV、训练段统计量做标准化、任意一天的预测只
用其之前的数据 → 严格无未来函数。

归属（2026-09-11 修复「归属倒挂」）：本模块原为 ``factor/synthesis.py`` 的
第 4 节，因属模型层能力而迁入 ``model/``；依赖方向固定为 model → factor
（复用 ``factor.synthesis.CompositeInput`` / ``long_matrix`` 与
``factor.cv.forward_folds``），``factor/`` 不得反向依赖本模块
（``tests/test_layering.py`` 锁死）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from factor.cv import forward_folds
from factor.preprocessing import standardize_zscore
from factor.synthesis import CompositeInput, long_matrix

__all__ = [
    "synthesize_stacking",
    "synthesize_stacking_gbdt",
    "synthesize_stacking_gbdt_tuned",
    "synthesize_stacking_lambdarank",
]


def _make_target(returns_panel: pd.DataFrame, idx, cols, target_mode: str = "raw") -> np.ndarray:
    """构造回归目标 y（对齐到 (idx, cols) 网格后 ravel）。

    target_mode:
        "raw"  : 原始次日收益（MSE 直接拟合收益值）
        "rank" : 当日截面收益的百分比秩 - 0.5（按日 rank(axis=1, pct=True)，
                 值域约 (-0.5, 0.5]，中心化）。秩目标与评价口径（rank IC）一致，
                 且对收益厚尾/异常值鲁棒（2026-08-05 方案 A）。
    """
    sub = returns_panel.reindex(index=idx, columns=cols)
    if target_mode == "rank":
        return (sub.rank(axis=1, pct=True) - 0.5).values.ravel()
    return sub.values.ravel()


def _time_fold_masks(date_arr: np.ndarray, n_splits: int, embargo_days: int = 0):
    """行级 mask 版 expanding 前推切分（统一走 ``factor.cv.forward_folds``）。

    供 stacking 系列合成函数使用：输入每行观测对应的日期数组（升序，
    ``date_arr`` 与长矩阵行一一对应），返回 list[(bool array, bool array)]，
    语义与旧的本地 ``_time_folds`` 完全一致（按交易日边界切折 + embargo purge），
    但切分实现已收敛到统一 CV 调度器 ``forward_folds``，避免两套切分并存。

    Args:
        date_arr: 每个有效观测对应的日期（升序，对应 obs 第一级并已按 valid 过滤）。
        n_splits: 折数（>1）。
        embargo_days: 训练段末尾剔除与测试段相邻的天数。
    Returns:
        list[(bool array, bool array)]，与 date_arr 等长；每折 train 的日期
        严格早于 test 的日期（无未来函数）。
    """
    folds = forward_folds(pd.DatetimeIndex(date_arr), n_splits, embargo_days)
    out = []
    day_idx = pd.DatetimeIndex(date_arr)  # 统一 dtype，避免 object 数组 isin DatetimeIndex 失配
    for f in folds:
        train_mask = np.isin(day_idx, f.train_days)
        test_mask = np.isin(day_idx, f.test_days)
        out.append((train_mask, test_mask))
    return out


def _inner_split_by_day(
    date_arr: np.ndarray, train_mask: np.ndarray, frac: float = 0.8, min_va_days: int = 1,
):
    """在训练段内按日期边界再切出验证段（嵌套 CV 的折内 split）。

    取 train 段前 frac 比例的交易日作内层训练，其余（>= min_va_days 天）作验证段。
    同样按【日期边界】切分，避免把某一天劈开。
    Returns:
        (tr_mask, va_mask)：与 date_arr 等长的 bool 数组。
    """
    tr_days = pd.unique(date_arr[train_mask])
    n = len(tr_days)
    split = int(n * frac)
    split = min(max(split, 0), max(0, n - min_va_days))
    tr_days_s = tr_days[:split]
    va_days = tr_days[split:]
    return (
        train_mask & np.isin(date_arr, tr_days_s),
        train_mask & np.isin(date_arr, va_days),
    )


def synthesize_stacking(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame,
    n_splits: int = 5,
    alpha: float = 1.0,
    target_mode: str = "raw",
) -> pd.DataFrame:
    """ML stacking：以各因子为特征、未来一期收益为目标，用**时间序列交叉验证**
    的 ridge 回归预测，预测分数即为复合因子。

    - 不一次性全量拟合，而是按时间顺序做 expanding-window 预测，
      任意一天的预测只用到该日之前的数据 → 严格无未来函数。
    - ridge 闭式解，无第三方依赖。
    - target_mode="raw"（默认）拟合收益值；"rank" 拟合当日截面收益的
      百分比秩（与 rank IC 评价口径一致，方案 A）。
    """
    if not components:
        raise ValueError("components 为空")
    X, obs, grid = long_matrix(components, returns_panel)
    idx, cols = grid
    y = _make_target(returns_panel, idx, cols, target_mode)

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xv = X[valid]
    yv = y[valid]
    # obs 由 MultiIndex.from_product([idx, cols]) 生成，行序已是「日期优先」递增，
    # 因此可取出每行对应日期后按【交易日边界】做 expanding-window 时序 CV。
    date_arr = obs.get_level_values(0)[valid].to_numpy()

    n = len(yv)
    pred = np.full(n, np.nan)
    for train_mask, test_mask in _time_fold_masks(date_arr, n_splits, embargo_days=0):
        if train_mask.sum() < max(10, Xv.shape[1] + 5) or test_mask.sum() == 0:
            continue
        Xtr, ytr = Xv[train_mask], yv[train_mask]
        Xte = Xv[test_mask]
        # 列标准化（基于训练集统计量，避免用测试集信息）
        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        # ridge 闭式解
        XtX = Xtr_s.T @ Xtr_s + alpha * np.eye(Xtr_s.shape[1])
        Xty = Xtr_s.T @ ytr
        beta, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
        pred[test_mask] = Xte_s @ beta

    composite_long = np.full(len(y), np.nan)   # 总长度 = 总观测数
    composite_long[valid] = pred
    composite = pd.DataFrame(composite_long.reshape(len(idx), len(cols)), index=idx, columns=cols)
    return standardize_zscore(composite)


def synthesize_stacking_gbdt(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 5,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 6,
    min_child_samples: int = 20,
    n_jobs: int = -1,
    seed: int = 42,
    target_mode: str = "raw",
) -> pd.DataFrame:
    """ML stacking（LightGBM + purged 时序 CV）：以各因子为特征、未来一期
    收益为目标，GBDT 预测分数即为复合因子。

    与 ``synthesize_stacking``（ridge）的差异：
    - **非线性模型**：LightGBM 梯度提升树，可捕获特征交互/阈值分裂的 alpha
    - **purged CV**：训练段尾部剔除与测试段相邻的 ``embargo_days`` 个交易日的
      样本（标签时间重叠/相邻 → 泄漏），防过拟合评估
    - 其余约定一致：expanding-window 时序切分、训练段统计量做标准化、预测
      只用历史数据 → 严格无未来函数

    依赖 lightgbm（可选；未安装时抛出可读 ImportError）。
    """
    try:
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise ImportError(
            "synthesize_stacking_gbdt 需要 lightgbm：pip install lightgbm"
        ) from e

    if not components:
        raise ValueError("components 为空")
    X, obs, grid = long_matrix(components, returns_panel)
    idx, cols = grid
    y = _make_target(returns_panel, idx, cols, target_mode)

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    date_arr = obs.get_level_values(0)[valid].to_numpy()
    n = len(yv)

    pred = np.full(n, np.nan)
    for train_mask, test_mask in _time_fold_masks(date_arr, n_splits, embargo_days=embargo_days):
        if train_mask.sum() < max(50, Xv.shape[1] * 5) or test_mask.sum() == 0:
            continue
        Xtr, ytr = Xv[train_mask], yv[train_mask]
        Xte = Xv[test_mask]
        # 列标准化（训练段统计量）
        mu = np.nanmean(Xtr, axis=0)
        sd = np.nanstd(Xtr, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        model = LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, max_depth=max_depth,
            min_child_samples=min_child_samples,
            n_jobs=n_jobs, random_state=seed, verbose=-1,
        )
        model.fit(Xtr_s, ytr)
        pred[test_mask] = model.predict(Xte_s)

    composite_long = np.full(len(y), np.nan)
    composite_long[valid] = pred
    composite = pd.DataFrame(composite_long.reshape(len(idx), len(cols)), index=idx, columns=cols)
    return standardize_zscore(composite)


def _rank_ic_by_day(pred: np.ndarray, y: np.ndarray, n_codes: int) -> float:
    """按日期分组算预测分数与真实收益的截面 Spearman IC 均值。

    obs 行序 = 日期优先（from_product([idx, cols])），每日期连续 n_codes 行。
    """
    n_days = len(pred) // n_codes
    ics = []
    for d in range(n_days):
        a = pred[d * n_codes:(d + 1) * n_codes]
        b = y[d * n_codes:(d + 1) * n_codes]
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() > 5:
            r = stats.spearmanr(a[m], b[m])[0]
            if not np.isnan(r):
                ics.append(r)
    return float(np.nanmean(ics)) if ics else 0.0


def synthesize_stacking_gbdt_tuned(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 5,
    n_trials: int = 25,
    seed: int = 42,
    n_jobs: int = -1,
    target_mode: str = "raw",
) -> pd.DataFrame:
    """ML stacking（LightGBM + optuna 自动调参 + purged 时序 CV）。

    与 ``synthesize_stacking_gbdt`` 的差异：每个外折内用 optuna 在
    【训练段再切出的验证段】上搜索超参（目标=验证段截面 rank IC），再用最优
    超参在完整训练段重训、预测测试段 —— **嵌套时序 CV**：超参选择只依赖
    折内历史，测试段全程未参与调参，无 look-ahead。

    调参空间：learning_rate / n_estimators / num_leaves / max_depth /
    min_child_samples / feature_fraction / lambda_l1 / lambda_l2。
    成本 ≈ n_splits × n_trials 次小树训练（默认 5×25=125 次，秒级/次）。
    依赖 optuna + lightgbm（可选，缺库可读报错）。
    """
    try:
        import optuna
        from lightgbm import LGBMRegressor
    except ImportError as e:
        raise ImportError(
            "synthesize_stacking_gbdt_tuned 需要 optuna + lightgbm："
            "pip install optuna lightgbm"
        ) from e
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not components:
        raise ValueError("components 为空")
    X, obs, grid = long_matrix(components, returns_panel)
    idx, cols = grid
    y = _make_target(returns_panel, idx, cols, target_mode)

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    date_arr = obs.get_level_values(0)[valid].to_numpy()
    n = len(yv)
    n_codes = len(cols)

    pred = np.full(n, np.nan)
    for train_mask, test_mask in _time_fold_masks(date_arr, n_splits, embargo_days=embargo_days):
        train_idx = np.where(train_mask)[0]
        if len(train_idx) < max(200, Xv.shape[1] * 10) or test_mask.sum() == 0:
            continue
        # ---- 折内切分：训练段末尾 20% 作验证段（最新历史，最接近测试分布）----
        inner_tr_mask, inner_va_mask = _inner_split_by_day(date_arr, train_mask, frac=0.8)
        inner_tr = np.where(inner_tr_mask)[0]
        inner_va = np.where(inner_va_mask)[0]
        Xtr_all, ytr_all = Xv[train_idx], yv[train_idx]
        Xtr, ytr = Xv[inner_tr], yv[inner_tr]
        Xva, yva = Xv[inner_va], yv[inner_va]
        if len(inner_tr) < 10 or len(inner_va) < 1:
            continue

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100, step=10),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            }
            model = LGBMRegressor(**params, n_jobs=n_jobs, random_state=seed, verbose=-1)
            model.fit(Xtr, ytr)
            return _rank_ic_by_day(model.predict(Xva), yva, n_codes)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        model = LGBMRegressor(**best, n_jobs=n_jobs, random_state=seed, verbose=-1)
        model.fit(Xtr_all, ytr_all)
        pred[test_mask] = model.predict(Xv[test_mask])

    composite_long = np.full(len(y), np.nan)
    composite_long[valid] = pred
    composite = pd.DataFrame(composite_long.reshape(len(idx), len(cols)), index=idx, columns=cols)
    return standardize_zscore(composite)


def synthesize_stacking_lambdarank(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 5,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 6,
    min_child_samples: int = 20,
    label_gain: list | None = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """ML stacking（LightGBM **LambdaRank** + purged 时序 CV）：以各因子为特征、
    当日截面收益为 relevance 标签，**每个交易日一个 query group**，直接优化
    NDCG（横截面排序一致性）——拟合目标与评价（rank IC）完全对齐。

    方案 B（2026-08-05，对标西南证券 2025 排序学习选股实证）：
    - 排序学习直接优化相对序位（pairwise），对收益厚尾/极端值鲁棒
    - 分组关键：样本行序=日期优先（from_product([idx, cols])），剔除 NaN 后
      必须**按日重算 group 大小**（valid 后每天行数不同）；purge 后训练段
      group 同样按实际行重算（purge 可能切在一天中间，LGBM 接受不完整组）
    - embargo / 无未来函数约定同 synthesize_stacking_gbdt
    依赖 lightgbm（可选；缺库可读报错）。
    """
    try:
        from lightgbm import LGBMRanker
    except ImportError as e:
        raise ImportError(
            "synthesize_stacking_lambdarank 需要 lightgbm：pip install lightgbm"
        ) from e

    if not components:
        raise ValueError("components 为空")
    X, obs, grid = long_matrix(components, returns_panel)
    idx, cols = grid
    # LambdaRank 要求 label 为整数 relevance 等级：取当日截面收益的百分比秩
    # 分桶为 0-4 五级（适配 lightgbm 默认 label_gain；NaN 保留由 valid 掩码剔除）
    rank_pct = returns_panel.reindex(index=idx, columns=cols).rank(axis=1, pct=True)
    y = (rank_pct * 4).round().values.ravel()

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xv, yv = X[valid], y[valid]
    date_arr = obs.get_level_values(0)[valid].to_numpy()   # 每行对应的日期（valid 后）
    n = len(yv)

    pred = np.full(n, np.nan)
    for train_mask, test_mask in _time_fold_masks(date_arr, n_splits, embargo_days=embargo_days):
        tr_rows = np.where(train_mask)[0]
        if len(tr_rows) < max(200, Xv.shape[1] * 10) or test_mask.sum() == 0:
            continue
        # 训练段 query group：按日重算（行序=日期升序，value_counts().sort_index() 对齐）
        tr_dates = date_arr[train_mask]
        groups = pd.Series(tr_dates).value_counts().sort_index().values
        model = LGBMRanker(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, max_depth=max_depth,
            min_child_samples=min_child_samples,
            label_gain=label_gain,
            n_jobs=n_jobs, random_state=seed, verbose=-1,
        )
        model.fit(Xv[tr_rows], yv[tr_rows], group=groups)
        pred[test_mask] = model.predict(Xv[test_mask])

    composite_long = np.full(len(y), np.nan)
    composite_long[valid] = pred
    composite = pd.DataFrame(composite_long.reshape(len(idx), len(cols)), index=idx, columns=cols)
    return standardize_zscore(composite)
