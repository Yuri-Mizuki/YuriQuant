"""
多因子合成与正交化
==================

挖掘出大量候选因子后，把其中显著 / 有效的若干因子**合成**为一个复合因子，
是挖掘闭环的最后一环：

    算子空间 → 候选生成 → 批量 IC → 显著性筛选 → 合成/正交化 → 因子库（迭代）

提供四种合成方式（输入都为「已截面标准化」的单因子面板）：

- ``ic_weighted`` : 按 |IC|（或 IR）加权线性组合，最常用、可解释
- ``pca``         : 主成分提取，取前 k 个成分（消除共线性冗余）
- ``orthogonal``  : 逐层回归正交化（Gram-Schmidt），再按 IC 加权组合
- ``stacking``    : 机器学习 stacking（时间序列交叉验证 ridge，严格无未来函数）

所有合成函数输出复合因子面板（date × code），可直接送入回测引擎
（VectorBacktest）。复合因子本身在返回前会再做一次截面标准化，便于横向比较。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from factor.preprocessing import standardize_zscore
from factor.cv import forward_folds
from stats.ic import calc_ic_series, calc_ir


@dataclass
class CompositeInput:
    """一个参与合成的（已标准化）单因子。"""
    name: str
    panel: pd.DataFrame          # date × code，建议已截面标准化
    ic: float = 0.0             # 该因子自身的 IC（用于加权 / 符号对齐）
    ir: float = 0.0


# ===========================================================================
# 因子面板重建
# ===========================================================================
def build_components(
    top_df: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    features: list[str] | None = None,
    windows: tuple[int, ...] = (5, 10, 20, 60),
    depth: int = 2,
) -> list[CompositeInput]:
    """根据挖掘结果 ``top_df``（须含 ``name`` 列）重建对应因子面板。

    通过名称匹配重新生成候选、调用其 ``build`` 闭包还原面板，再截面标准化。
    **GP 公式还原（2026-08-03 新增）**：name 不在 exhaustive 候选空间时（如
    GP HallOfFame 的 ``mul(ts_mean_5(close), cs_rank(ts_delta_20(volume)))``），
    回退到统一公式解析器 ``factor.formula.formula_builder`` 重建（支持 GP 的
    窗口编名语法），不再依赖 deap pset / 模块级 prim_map。
    """
    from factor.mining import dedup_by_formula, generate_candidates
    from factor.formula import formula_builder

    feats = features if features is not None else list(panel.keys())
    cands = dedup_by_formula(generate_candidates(features=feats, windows=windows, depth=depth))
    by_name = {c.name: c for c in cands}

    out: list[CompositeInput] = []
    for _, row in top_df.iterrows():
        name = row["name"]
        c = by_name.get(name)
        if c is not None:
            try:
                fp = c.build(panel)
            except Exception:
                continue
        else:
            # GP / 未覆盖公式：统一公式解析器重建（窗口由公式自身给出）
            try:
                fp = formula_builder(name, features=feats)(panel)
            except Exception:
                continue
        if fp is None or fp.empty:
            continue
        fp = standardize_zscore(fp)
        out.append(CompositeInput(
            name=name,
            panel=fp,
            ic=float(row.get("ic_mean", 0.0)),
            ir=float(row.get("ir", 0.0)),
        ))
    return out


# ===========================================================================
# 符号对齐（让「因子值高 ⇒ 未来收益高」）
# ===========================================================================
def _align_sign(panel: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """把 panel 的全局符号翻转到与 ref 同向（基于两者逐日截面相关均值）。

    ref 通常是未来一期收益面板：翻转后复合因子指向「高值=高收益」方向。
    """
    common = panel.index.intersection(ref.index)
    codes = panel.columns.intersection(ref.columns)
    corr_sum = 0.0
    n = 0
    for d in common:
        a = panel.loc[d, codes].dropna()
        b = ref.loc[d, a.index].dropna()
        if len(b) < 5:
            continue
        c = np.corrcoef(a.values, b.values)[0, 1] if len(b) >= 2 else 0.0
        if not np.isnan(c):
            corr_sum += c
            n += 1
    if n == 0 or corr_sum < 0:
        return -panel
    return panel


def _align_sign_by_ic(comp: CompositeInput) -> CompositeInput:
    """按因子自身 IC 符号翻转，使其指向「高值=高收益」。

    返回新对象，不修改传入的 ``comp``：早期实现原地改写 panel/ic，导致同一份
    输入被先后喂给 ic_weighted / orthogonal 时被反复翻转，第二次合成符号错乱。
    """
    if comp.ic < 0:
        return CompositeInput(name=comp.name, panel=-comp.panel, ic=-comp.ic, ir=comp.ir)
    return comp


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


def rebuild_train_weights(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame,
    train_dates,
) -> list[CompositeInput]:
    """用【训练段】重算各因子 IC/IR，作为 ``ic_weighted`` / ``orthogonal`` 的权重与符号。

    返回的 components 面板保持全样本（用于对全区间合成），但 ``ic``/``ir`` 只用
    ``train_dates`` 内的收益重算 → 训练段信息定权重，测试段复合值无未来函数。

    关键修复（2026-08-17）：旧实现把挖掘/因子库存储的**全样本 IC** 直接当作
    合成权重，等价于用未来收益决定权重（look-ahead）。语义与 ``walk_forward.py``
    用 valid 段 IC 定权重一致。本函数由 ``synthesize_library.py`` / ``synthesize_factors.py``
    共享，避免两处重复实现。
    """
    rts_tr = returns_panel.loc[train_dates]
    out: list[CompositeInput] = []
    for c in components:
        panel_tr = c.panel.loc[train_dates].reindex(columns=rts_tr.columns)
        ic = calc_ic_series(panel_tr, rts_tr).dropna()
        if len(ic) < 5:
            out.append(CompositeInput(name=c.name, panel=c.panel, ic=0.0, ir=0.0))
            continue
        out.append(CompositeInput(name=c.name, panel=c.panel,
                                  ic=float(ic.mean()), ir=calc_ir(ic)))
    return out


# ===========================================================================
# 1) IC 加权组合
# ===========================================================================
def synthesize_ic_weighted(
    components: list[CompositeInput],
    weight_by: str = "ic_abs",
    returns_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按 |IC|（或 IR / 原始 IC 符号）加权线性组合。

    每个因子先按自身 IC 符号对齐（高值=高收益），再加权求和后标准化。
    """
    if not components:
        raise ValueError("components 为空")
    comps = [_align_sign_by_ic(c) for c in components]

    if weight_by == "ic_abs":
        w = np.array([abs(c.ic) for c in comps])
    elif weight_by == "ir":
        w = np.array([abs(c.ir) for c in comps])
    elif weight_by == "ic_signed":
        w = np.array([c.ic for c in comps])  # 已对齐符号，均为正
    else:
        raise ValueError(f"未知 weight_by: {weight_by}")

    if w.sum() <= 0:
        w = np.ones(len(comps)) / len(comps)
    else:
        w = w / w.sum()

    composite = sum(wi * c.panel.fillna(0.0) for wi, c in zip(w, comps))
    return standardize_zscore(composite)


# ===========================================================================
# 2) PCA 主成分合成
# ===========================================================================
def _long_matrix(
    components: list[CompositeInput], returns_panel: pd.DataFrame | None
) -> tuple[np.ndarray, pd.MultiIndex, list[CompositeInput]]:
    """把多因子面板对齐到同一 (date, code) 网格，铺成 (n_obs, n_factors) 长矩阵。"""
    # 取所有因子面板的交集网格
    idx = components[0].panel.index
    cols = components[0].panel.columns
    for c in components[1:]:
        idx = idx.intersection(c.panel.index)
        cols = cols.intersection(c.panel.columns)
    if returns_panel is not None:
        idx = idx.intersection(returns_panel.index)
        cols = cols.intersection(returns_panel.columns)

    obs = pd.MultiIndex.from_product([idx, cols], names=["date", "code"])
    X = np.column_stack([c.panel.reindex(index=idx, columns=cols).values.ravel() for c in components])
    return X, obs, (idx, cols)


def synthesize_pca(
    components: list[CompositeInput],
    n_components: int = 1,
    returns_panel: pd.DataFrame | None = None,
    sign_calib_frac: float = 0.6,
) -> pd.DataFrame:
    """PCA 主成分合成：在 (date×code) 观测上对因子矩阵做 PCA，取前 n 个成分。

    每个成分按与未来收益的相关性翻转符号，最后等权（或按成分方差）合成。

    **无未来函数（2026-08-17 修复）**：主成分方向（特征向量）与符号方向都只在
    时间序列前 ``sign_calib_frac`` 段（默认 60%，训练段）内估计，再投影到全样本。
    旧实现的主成分方向是在**全样本**上 SVD 求得的（隐式使用了全区间协方差结构，
    属 look-ahead），仅符号用了前段校准；现改为方向与符号均只用前段。校准段建议
    至少 20 个交易日（成分方向对训练段长度敏感，段越短方差越大）。
    """
    if not components:
        raise ValueError("components 为空")
    X, obs, grid = _long_matrix(components, returns_panel)
    idx, cols = grid
    n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
    if n_comp < 1:
        n_comp = 1

    # 缺失值填 0（已标准化，截面均值为 0，偏离很小）；列中心化
    Xc = np.nan_to_num(X, nan=0.0)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)

    # 训练段边界（前 sign_calib_frac 段）：完整网格行序 = 日期优先，每天 len(cols) 行
    n_cal = max(10, int(len(idx) * sign_calib_frac))
    n_cal = min(n_cal, len(idx))
    cal_rows = n_cal * len(cols)

    # 主成分方向：只用训练段 SVD（避免用全样本协方差结构 = 未来信息）
    Xc_cal = Xc[:cal_rows]
    U_cal, S_cal, Vt_cal = np.linalg.svd(Xc_cal, full_matrices=False)
    comp_dirs = Vt_cal[:n_comp].T            # (n_factors, n_comp)
    scores = Xc @ comp_dirs                  # (n_obs, n_comp)

    # 每个成分按与收益相关性翻转符号 —— 只用前 sign_calib_frac 段（训练段）定方向
    if returns_panel is not None:
        y = returns_panel.reindex(index=idx, columns=cols).values.ravel()
        y = np.nan_to_num(y, nan=0.0)
        for k in range(n_comp):
            a = scores[:cal_rows, k]
            b = y[:cal_rows]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() > 5:
                r = np.corrcoef(a[valid], b[valid])[0, 1]
                if not np.isnan(r) and r < 0:
                    scores[:, k] = -scores[:, k]

    # 多成分：按解释方差（训练段 S 的平方占比）加权合成
    var_w = (S_cal[:n_comp] ** 2)
    var_w = var_w / var_w.sum() if var_w.sum() > 0 else np.ones(n_comp) / n_comp
    composite_long = scores @ var_w

    composite = pd.DataFrame(composite_long.reshape(len(idx), len(cols)), index=idx, columns=cols)
    return standardize_zscore(composite)


# ===========================================================================
# 3) 正交化（Gram-Schmidt 逐层回归残差）
# ===========================================================================
def orthogonalize(components: list[CompositeInput]) -> list[CompositeInput]:
    """逐层回归正交化：每个因子对其之前所有因子做截面回归，取残差。

    返回正交化后的子因子列表（每个均已与前面的因子线性无关）。

    实现：按 |IC| 降序排列后做**列向 Gram-Schmidt**。关键优化：
    一旦得到归一化正交基 Q，第 k 个因子对前 k-1 个基的投影系数就是
    逐日点积（正交基下无需解 lstsq），可对全部日期向量化：
        v_k = x_k - Σ_j (q_j · x_k) q_j
    原实现"逐日 × 逐因子 lstsq"是 O(K² × T × lstsq)，K 大时（如 65 因子）
    慢到不可用；此处为 O(K² × T × N) 的纯矩阵运算，K=65 时秒级完成。
    """
    if not components:
        return []
    ordered = sorted(components, key=lambda c: abs(c.ic), reverse=True)

    idx = ordered[0].panel.index
    cols = ordered[0].panel.columns
    # 统一索引/列，缺失填 0（zscore 后均值 0，污染极小），NaN 位置后续恢复
    F = np.stack([
        c.panel.reindex(index=idx, columns=cols).fillna(0.0).astype(float).values
        for c in ordered
    ], axis=2)                     # (T, N, K)
    mask = np.stack([
        c.panel.reindex(index=idx, columns=cols).notna().values
        for c in ordered
    ], axis=2)                     # 原始非空掩码

    K = F.shape[2]
    Q = np.zeros_like(F)           # 归一化正交基 (T, N, K)
    resid = np.zeros_like(F)       # 残差因子（未归一化）
    for k in range(K):
        v = F[:, :, k].copy()
        for j in range(k):
            # 逐日点积系数 (T,1)，投影到已正交基 q_j
            coef = np.sum(Q[:, :, j] * v, axis=1, keepdims=True)
            v -= coef * Q[:, :, j]
        resid[:, :, k] = v
        norm = np.sqrt(np.sum(v * v, axis=1, keepdims=True))
        norm = np.where(norm > 1e-12, norm, 1.0)
        Q[:, :, k] = v / norm

    # 恢复 NaN：原始缺失位置保持 NaN（残差不适用）
    resid = np.where(mask, resid, np.nan)

    out: list[CompositeInput] = []
    for k, c in enumerate(ordered):
        panel = pd.DataFrame(resid[:, :, k], index=idx, columns=cols)
        out.append(CompositeInput(name=c.name, panel=standardize_zscore(panel),
                                  ic=c.ic, ir=c.ir))
    return out


def synthesize_orthogonal(
    components: list[CompositeInput],
    weight_by: str = "ic_abs",
) -> pd.DataFrame:
    """先正交化，再按 IC 加权组合正交子因子。"""
    ortho = orthogonalize(components)
    return synthesize_ic_weighted(ortho, weight_by=weight_by)


# ===========================================================================
# 4) ML Stacking（时间序列交叉验证 ridge，无未来函数）
# ===========================================================================
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
    X, obs, grid = _long_matrix(components, returns_panel)
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
    X, obs, grid = _long_matrix(components, returns_panel)
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
    X, obs, grid = _long_matrix(components, returns_panel)
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
    X, obs, grid = _long_matrix(components, returns_panel)
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


# ===========================================================================
# 评估辅助
# ===========================================================================
def composite_stats(
    composite: pd.DataFrame,
    returns_panel: pd.DataFrame,
    method: str = "spearman",
    min_obs: int = 10,
) -> dict:
    """计算复合因子的 IC / IR / t 等统计量。"""
    ic = calc_ic_series(composite, returns_panel, method=method).dropna()
    n = len(ic)
    if n < min_obs:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ir": 0.0,
                "ic_win_rate": float("nan"), "t_stat": 0.0, "n": n}
    m, s = float(ic.mean()), float(ic.std())
    ir = calc_ir(ic)
    t = m / (s / np.sqrt(n)) if s > 0 else 0.0
    return {
        "ic_mean": m, "ic_std": s, "ir": ir,
        "ic_win_rate": float((ic > 0).mean()), "t_stat": t, "n": n,
    }
