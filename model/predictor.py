"""截面预测器 —— 模型层 ③ Predictor。

把 N 个因子面板（特征）压成 1 个 date×code 预测面板的可学习函数：

    fit(features: {name: 面板}, labels: 面板) -> self
    predict(features: {name: 面板}) -> date×code 预测面板（已截面标准化）

- ``RidgePredictor``：线性基线（闭式解，无三方依赖；any-NaN 行剔除，
  训练段统计量标准化——纪律与 ``synthesize_stacking`` 一致）
- ``LGBMPredictor``：LightGBM 回归（原生容忍 NaN 特征；排序目标由
  LabelBuilder 的 rank/zscore 标签实现，与 rank IC 评价口径对齐）
- ``TabICLPredictor``：TabICL 表格基础模型（in-context learning，零显式
  训练；context 截最近 ``max_context_samples`` 样本，预测时分块 forward）

``fit_predict_oos``：一次性 OOS 面板（按交易日边界切折 + embargo purge，
复用统一切分调度器 ``factor.cv.make_folds``——stacking 防泄漏纪律的直接继承，支持切换
切分方法（``cv_method``），产出全区间 OOS 预测面板（首段训练区无预测，为 NaN）。

滚动再训练（walk-forward 上线）不需要本函数：调用方按窗口切片后直接
循环 fit/predict 即可（见 scripts/walk_forward_model.py）。
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from factor.preprocessing import standardize_zscore
from factor.cv import make_folds

__all__ = [
    "BasePredictor", "RidgePredictor", "LGBMPredictor", "LGBRankerPredictor",
    "TabICLPredictor", "PREDICTORS", "fit_predict_oos",
]


# ---------------------------------------------------------------------------
# 网格与长矩阵（口径对齐 factor.synthesis._long_matrix：日期优先行序）
# ---------------------------------------------------------------------------
def _grid(features: Mapping[str, pd.DataFrame],
          target: pd.DataFrame | None = None) -> tuple[pd.Index, pd.Index]:
    """所有特征面板（+可选目标面板）的 (date, code) 交集网格。"""
    if not features:
        raise ValueError("features 为空")
    idx = None
    cols = None
    for p in features.values():
        idx = p.index if idx is None else idx.intersection(p.index)
        cols = p.columns if cols is None else cols.intersection(p.columns)
    if target is not None:
        idx = idx.intersection(target.index)
        cols = cols.intersection(target.columns)
    if len(idx) == 0 or len(cols) == 0:
        raise ValueError("特征与目标无公共 (date, code) 网格")
    return idx, cols


def _long_matrix(features: Mapping[str, pd.DataFrame], names: list[str],
                 idx: pd.Index, cols: pd.Index) -> np.ndarray:
    """(n_obs, n_feat) 长矩阵，行序 = 日期优先（from_product 约定）。"""
    mats = [features[n].reindex(index=idx, columns=cols).values.ravel() for n in names]
    return np.column_stack(mats)


def _to_panel(values: np.ndarray, idx: pd.Index, cols: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(values.reshape(len(idx), len(cols)), index=idx, columns=cols)


# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------
class BasePredictor:
    """截面预测器抽象基类：fit / predict 两段式。"""

    name = "base"

    def fit(self, features: Mapping[str, pd.DataFrame], labels: pd.DataFrame) -> "BasePredictor":
        raise NotImplementedError

    def predict(self, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError

    def _check_features(self, features: Mapping[str, pd.DataFrame]) -> None:
        missing = [n for n in self.feature_names_ if n not in features]
        if missing:
            raise KeyError(f"predict 缺少训练时的特征: {missing}")


class RidgePredictor(BasePredictor):
    """线性 ridge 基线：闭式解 + 训练段统计量标准化 + any-NaN 行剔除。"""

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, features: Mapping[str, pd.DataFrame], labels: pd.DataFrame) -> "RidgePredictor":
        idx, cols = _grid(features, labels)
        self.feature_names_ = sorted(features.keys())
        X = _long_matrix(features, self.feature_names_, idx, cols)
        y = labels.reindex(index=idx, columns=cols).values.ravel()

        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        Xv, yv = X[valid], y[valid]
        if len(yv) < max(10, X.shape[1] + 5):
            raise ValueError(
                f"ridge 有效样本不足: {len(yv)} < {max(10, X.shape[1] + 5)}")

        mu = np.nanmean(Xv, axis=0)
        sd = np.nanstd(Xv, axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        Xs = (Xv - mu) / sd
        XtX = Xs.T @ Xs + self.alpha * np.eye(Xs.shape[1])
        beta, *_ = np.linalg.lstsq(XtX, Xs.T @ yv, rcond=None)

        self._mu, self._sd, self._beta = mu, sd, beta
        self.n_samples_ = int(len(yv))
        return self

    def predict(self, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        self._check_features(features)
        idx, cols = _grid(features)
        X = _long_matrix(features, self.feature_names_, idx, cols)
        pred = ((X - self._mu) / self._sd) @ self._beta
        return standardize_zscore(_to_panel(pred, idx, cols))


class LGBMPredictor(BasePredictor):
    """LightGBM 截面回归（内核超参与 ``synthesize_stacking_gbdt`` 一致）。"""

    name = "gbdt"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, max_depth: int = 6,
                 min_child_samples: int = 20, n_jobs: int = -1, seed: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, features: Mapping[str, pd.DataFrame], labels: pd.DataFrame) -> "LGBMPredictor":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as e:
            raise ImportError("LGBMPredictor 需要 lightgbm：pip install lightgbm") from e

        idx, cols = _grid(features, labels)
        self.feature_names_ = sorted(features.keys())
        X = _long_matrix(features, self.feature_names_, idx, cols)
        y = labels.reindex(index=idx, columns=cols).values.ravel()

        # 标签 NaN 必须剔除；特征 NaN 保留（LightGBM 原生处理）
        valid = ~np.isnan(y)
        Xv, yv = X[valid], y[valid]
        if len(yv) < max(50, X.shape[1] * 5):
            raise ValueError(
                f"gbdt 有效样本不足: {len(yv)} < {max(50, X.shape[1] * 5)}")

        self._model = LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs, random_state=self.seed, verbose=-1,
        )
        self._model.fit(Xv, yv)
        self.n_samples_ = int(len(yv))
        return self

    def predict(self, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        self._check_features(features)
        idx, cols = _grid(features)
        X = _long_matrix(features, self.feature_names_, idx, cols)
        pred = self._model.predict(X)
        return standardize_zscore(_to_panel(pred, idx, cols))


class TabICLPredictor(BasePredictor):
    """TabICL 表格基础模型（in-context learning，无显式训练）。

    ICL 范式：``fit`` 只存 context 长表（X, y），``predict`` 把 query 样本
    与 context 一起 forward。context 取**最近** ``max_context_samples``
    个有效样本（TabICL 预训练上限 ~10K；走 ``ignore_pretraining_limits``
    会显著变慢，故直接截断）——与 ridge/gbdt「用全部折前历史」不同，
    但这是 ICL 的设计前提（小样本强泛化），对滚动窗口天然友好。

    NaN 约定：fit 丢 any-NaN 行；predict 的 NaN 特征填 0（特征已截面
    标准化，0 = 截面均值，等价于保守中性化）。
    """

    name = "tabicl"

    def __init__(self, max_context_samples: int = 10000, device: str = "cpu",
                 chunk_size: int = 5000, n_estimators: int = 2, seed: int = 42):
        self.max_context_samples = int(max_context_samples)
        self.device = device
        self.chunk_size = int(chunk_size)
        self.n_estimators = int(n_estimators)
        self.seed = seed

    def fit(self, features: Mapping[str, pd.DataFrame],
            labels: pd.DataFrame) -> "TabICLPredictor":
        try:
            from tabicl import TabICLRegressor
        except ImportError as e:
            raise ImportError("TabICLPredictor 需要 tabicl：pip install tabicl") from e

        idx, cols = _grid(features, labels)
        self.feature_names_ = sorted(features.keys())
        X = _long_matrix(features, self.feature_names_, idx, cols)
        y = labels.reindex(index=idx, columns=cols).values.ravel()

        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        Xv, yv = X[valid], y[valid]
        if len(yv) < 200:
            raise ValueError(f"tabicl 有效样本不足: {len(yv)} < 200")

        Xv = Xv[-self.max_context_samples:]
        yv = yv[-self.max_context_samples:]
        # n_estimators 默认 8（ensemble 次数）在 CPU 上过慢；2 为速度/方差折中
        self._model = TabICLRegressor(
            device=self.device, n_estimators=self.n_estimators,
            random_state=self.seed)
        self._model.fit(Xv, yv)
        self.n_samples_ = int(len(yv))
        return self

    def predict(self, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        self._check_features(features)
        idx, cols = _grid(features)
        X = _long_matrix(features, self.feature_names_, idx, cols)
        X = np.nan_to_num(X, nan=0.0)

        out = np.full(X.shape[0], np.nan)
        for s in range(0, X.shape[0], self.chunk_size):
            e = min(s + self.chunk_size, X.shape[0])
            out[s:e] = self._model.predict(X[s:e])
        return standardize_zscore(_to_panel(out, idx, cols))


class LGBRankerPredictor(BasePredictor):
    """LightGBM 排序学习（pairwise loss, lambdarank）。

    与 ``LGBMPredictor``（回归）的区别：
    - 标签：rank 归一化值（0~1）→ 离散化为 gain labels（0~N-1 整数，按截面 rank）
    - 损失函数：pairwise logistic / lambdarank（直接优化排序而非点值 MSE）
    - group：每日截面为一组，模型学同一截面内的相对排序
    - 预测值：截面 z-score 标准化（与 IC 评价口径对齐）
    """

    name = "ranker"

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, max_depth: int = 6,
                 min_child_samples: int = 20, n_jobs: int = -1, seed: int = 42,
                 labels_bins: int = 2, objective: str = "rank_xendcg"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.seed = seed
        self.labels_bins = labels_bins
        self.objective = objective

    def fit(self, features: Mapping[str, pd.DataFrame],
           labels: pd.DataFrame) -> "LGBRankerPredictor":
        try:
            from lightgbm import LGBMRanker
        except ImportError as e:
            raise ImportError("LGBRankerPredictor 需要 lightgbm：pip install lightgbm") from e

        idx, cols = _grid(features, labels)
        self.feature_names_ = sorted(features.keys())
        X = _long_matrix(features, self.feature_names_, idx, cols)
        y = labels.reindex(index=idx, columns=cols).values.ravel()

        # 剔除标签 NaN 行
        valid = ~np.isnan(y)
        Xv, yv = X[valid], y[valid]
        if len(yv) < max(50, X.shape[1] * 5):
            raise ValueError(
                f"ranker 有效样本不足: {len(yv)} < {max(50, X.shape[1] * 5)}")

        # 离散化标签：每日截面内映射到 0~N_BINS-1 整数（桶数由 labels_bins 控制）
        # LGBMRanker 要求所有 group 的标签在 [0, N_BINS-1] 范围内。
        # 经验：分桶过多（如 30）使 lambdarank/xendcg 无法有效学习排序；
        # 默认二分（labels_bins=2）= 按截面中位数切上下半，IC 显著为正走势稳定。
        N_BINS = int(self.labels_bins)
        n_days = len(idx)
        n_codes = len(cols)
        valid_2d = ~np.isnan(labels.reindex(index=idx, columns=cols).values)
        group_sizes = valid_2d.sum(axis=1)
        day_mask = group_sizes > 0
        group_sizes = group_sizes[day_mask]

        # 按日组离散化标签到 N_BINS 个桶，收益越高桶号越大
        y_int = np.zeros(len(yv), dtype=int)
        pos = 0
        labels_2d = labels.reindex(index=idx, columns=cols).values
        for i in range(n_days):
            if not day_mask[i]:
                continue
            row = labels_2d[i]
            vals = row[~np.isnan(row)]
            n = len(vals)
            if N_BINS >= n:
                ranks = pd.Series(vals).rank(method="dense").astype(int) - 1
            else:
                # 截面内等频分桶；duplicates="drop" 去重，NaN 兜底为 0
                ranks = pd.Series(
                    pd.qcut(vals, N_BINS, labels=False, duplicates="drop"))
                ranks = ranks.fillna(0).astype(int)
            y_int[pos:pos + len(ranks)] = ranks.values
            pos += len(ranks)

        self._model = LGBMRanker(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs, random_state=self.seed, verbose=-1,
            objective=self.objective,
        )
        self._model.fit(Xv, y_int, group=group_sizes)
        self.n_samples_ = int(len(yv))
        return self

    def predict(self, features: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        self._check_features(features)
        idx, cols = _grid(features)
        X = _long_matrix(features, self.feature_names_, idx, cols)
        pred = self._model.predict(X)
        return standardize_zscore(_to_panel(pred, idx, cols))


PREDICTORS: dict[str, type[BasePredictor]] = {
    "ridge": RidgePredictor,
    "gbdt": LGBMPredictor,
    "ranker": LGBRankerPredictor,
    "tabicl": TabICLPredictor,
}


# ---------------------------------------------------------------------------
# 一次性 OOS 面板（统一 CV 调度，纪律继承自 stacking）
# ---------------------------------------------------------------------------
def fit_predict_oos(
    predictor_cls: type[BasePredictor],
    features: Mapping[str, pd.DataFrame],
    labels: pd.DataFrame,
    n_splits: int = 5,
    embargo_days: int = 5,
    min_train_days: int = 120,
    cv_method: str = "forward",
    **predictor_params: Any,
) -> pd.DataFrame:
    """一次性 OOS 预测面板，切分方式由 ``cv_method`` 统一切换。

    折切分复用 ``factor.cv.make_folds``（按交易日边界 + embargo purge，
    与 stacking 同一防泄漏纪律）。``embargo_days`` 应 >= 标签 horizon
    （LabelBuilder 返回值），否则训练段尾部标签前视进测试段。

    Args:
        cv_method: 切分方法，见 ``make_folds``。默认 ``forward``（expanding
            前推，生产唯一合法方法）；研究评估可切 ``purged``/``blocked``
            做横向对比。

    Returns:
        date×code 面板：各折测试段为 OOS 预测（已截面标准化），
        首个训练段为 NaN（该段无预测）。
    """
    idx, cols = _grid(features, labels)
    folds = make_folds(idx, method=cv_method, n_splits=n_splits,
                       embargo_days=embargo_days)
    if not folds:
        raise ValueError(
            f"日期数不足以切 {n_splits} 折（共 {len(idx)} 日）")

    out = pd.DataFrame(np.nan, index=idx, columns=cols)
    n_done = 0
    for fold in folds:
        tr_days = fold.train_days
        te_days = fold.test_days
        if len(tr_days) < min_train_days or len(te_days) == 0:
            continue
        p = predictor_cls(**predictor_params)
        p.fit({k: v.loc[tr_days] for k, v in features.items()},
              labels.loc[tr_days])
        pred = p.predict({k: v.loc[te_days] for k, v in features.items()})
        out.loc[te_days] = pred.reindex(index=te_days, columns=cols)
        n_done += 1
    if n_done == 0:
        raise ValueError(f"无有效折（min_train_days={min_train_days} 太大或数据太短）")
    return out
