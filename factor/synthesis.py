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
from research.factor_analysis import calc_ic_series, calc_ir


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

    **无未来函数（2026-08-03 修复）**：符号方向只在时间序列前 ``sign_calib_frac``
    段（默认 60%）内与收益的相关性决定，再用到全样本。旧实现用全样本收益定符号，
    等价于把未来信息回灌到合成因子，属轻微 look-ahead。校准段建议至少 20 个交易日。
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

    # SVD 取主成分方向
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp_dirs = Vt[:n_comp].T            # (n_factors, n_comp)
    scores = Xc @ comp_dirs              # (n_obs, n_comp)

    # 每个成分按与收益相关性翻转符号 —— 只用前 sign_calib_frac 段（训练段）定方向
    if returns_panel is not None:
        y = returns_panel.reindex(index=idx, columns=cols).values.ravel()
        y = np.nan_to_num(y, nan=0.0)
        n_cal = max(10, int(len(idx) * sign_calib_frac))
        n_cal = min(n_cal, len(idx))
        cal_rows = n_cal * len(cols)
        for k in range(n_comp):
            a = scores[:cal_rows, k]
            b = y[:cal_rows]
            valid = ~np.isnan(a) & ~np.isnan(b)
            if valid.sum() > 5:
                r = np.corrcoef(a[valid], b[valid])[0, 1]
                if not np.isnan(r) and r < 0:
                    scores[:, k] = -scores[:, k]

    # 多成分：按解释方差（S 的平方占比）加权合成
    var_w = (S[:n_comp] ** 2)
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
) -> pd.DataFrame:
    """ML stacking：以各因子为特征、未来一期收益为目标，用**时间序列交叉验证**
    的 ridge 回归预测，预测分数即为复合因子。

    - 不一次性全量拟合，而是按时间顺序做 expanding-window 预测，
      任意一天的预测只用到该日之前的数据 → 严格无未来函数。
    - ridge 闭式解，无第三方依赖。
    """
    if not components:
        raise ValueError("components 为空")
    X, obs, grid = _long_matrix(components, returns_panel)
    idx, cols = grid
    y = returns_panel.reindex(index=idx, columns=cols).values.ravel()

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    Xv = X[valid]
    yv = y[valid]
    # obs 由 MultiIndex.from_product([idx, cols]) 生成，行序已是「日期优先」递增，
    # 因此可直接按行序做 expanding-window 时间序列交叉验证（无未来函数）。
    n = len(yv)
    pred = np.full(n, np.nan)
    edges = np.linspace(0, n, n_splits + 1).astype(int)
    for s in range(1, n_splits):
        train_mask = np.arange(n) < edges[s]
        test_mask = (np.arange(n) >= edges[s]) & (np.arange(n) < edges[s + 1])
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
