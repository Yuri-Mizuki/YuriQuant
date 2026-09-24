"""
多因子合成与正交化（因子层）
============================

挖掘出大量候选因子后，把其中显著 / 有效的若干因子**合成**为一个复合因子，
是挖掘闭环的最后一环：

    算子空间 → 候选生成 → 批量 IC → 显著性筛选 → 合成/正交化 → 因子库（迭代）

提供三种**确定性**合成方式（输入都为「已截面标准化」的单因子面板）：

- ``ic_weighted`` : 按 |IC|（或 IR）加权线性组合，最常用、可解释
- ``pca``         : 主成分提取，取前 k 个成分（消除共线性冗余）
- ``orthogonal``  : 逐层回归正交化（Gram-Schmidt），再按 IC 加权组合

**华泰多因子系列10 口径补齐（2026-09-16）**：研报《因子合成方法实证分析》
的六种方法中，下面四种此前缺失，现已补齐（口径核对报告
``reports/口径核对_AI39_多因子10.md`` 第二节）：

- ``synthesize_ic_ir_max`` : **最大化 IC_IR** —— ``w = Σ⁻¹·ĪC``，
  Σ 为历史 IC 协方差矩阵，可带 ``w ≥ 0`` 约束（研报主力方法）
- ``synthesize_ic_max``    : **最大化 IC** —— ``w = V⁻¹·ĪC``，
  V 为**当前截面因子值**相关系数阵
- ``half_life_weights``    : 半衰加权 ``w_t = 2^((t−T−1)/H)``（可作用于上两者）
- ``ic_weighted(weight_by="equal")`` : 等权法

与研报的差异（**对比时必须披露**）：项目所有历史量（IC 均值 / IC 协方差）
均**只用训练段**估计，研报未讨论该防未来函数处理；研报的「历史因子收益率」
是 WLS 回归产物，本项目无该链路（详见核对报告 2.4）。

所有合成函数输出复合因子面板（date × code），可直接送入回测引擎
（VectorBacktest）。复合因子本身在返回前会再做一次截面标准化，便于横向比较。

**归属边界（2026-09-11）**：第四种「ML stacking」（ridge / LightGBM /
LambdaRank + 时序 CV）拟合的是**有监督模型**，属模型层能力，已迁往
:mod:`model.stacking`；本模块只保留不拟合模型的确定性因子组合。依赖方向固定为
model → factor（``model.stacking`` 反向复用本模块的 ``CompositeInput`` /
``long_matrix`` 与 ``factor.cv.forward_folds``），``factor/`` 不得依赖 model
（``tests/test_layering.py`` 锁死）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factor.preprocessing import standardize_zscore
from stats.ic import calc_ic_series, calc_ir
from stats.significance import mean_inference


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
def _align_sign_by_ic(comp: CompositeInput) -> CompositeInput:
    """按因子自身 IC 符号翻转，使其指向「高值=高收益」。

    返回新对象，不修改传入的 ``comp``：早期实现原地改写 panel/ic，导致同一份
    输入被先后喂给 ic_weighted / orthogonal 时被反复翻转，第二次合成符号错乱。
    """
    if comp.ic < 0:
        return CompositeInput(name=comp.name, panel=-comp.panel, ic=-comp.ic, ir=comp.ir)
    return comp


# 注：原第 4 节「ML Stacking」及其私有辅助（_make_target / _time_fold_masks /
# _inner_split_by_day / _rank_ic_by_day）已于 2026-09-11 迁往 model/stacking.py
# —— 它们拟合有监督模型，属模型层能力。本模块只保留确定性的因子组合。
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
    """按 |IC|（或 IR / 等权）加权线性组合。

    每个因子先按自身 IC 符号对齐（高值=高收益），再加权求和后标准化。

    ``weight_by``：
    - ``"ic_abs"``（默认）：|IC| 加权
    - ``"ir"``：|IR| 加权
    - ``"ic_signed"``：带符号 IC（已对齐，均为正）
    - ``"equal"``：**等权 1/N**（研报六方法之一的「等权法」，2026-09-16 补）
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
    elif weight_by == "equal":
        w = np.ones(len(comps))                       # 等权：不依赖任何历史量
    else:
        raise ValueError(f"未知 weight_by: {weight_by}")

    if w.sum() <= 0:
        w = np.ones(len(comps)) / len(comps)
    else:
        w = w / w.sum()

    composite = sum(wi * c.panel.fillna(0.0) for wi, c in zip(w, comps))
    return standardize_zscore(composite)


# ===========================================================================
# 1b) 半衰加权（研报 w_t = 2^((t−T−1)/H)）
# ===========================================================================
def half_life_weights(
    dates,
    half_life: float,
    *,
    as_of=None,
) -> np.ndarray:
    """半衰权重序列（研报口径）。

    研报定义 ``w_t = 2^((t−T−1)/H)``：``t`` 为距今第 t 期（``t`` 越大越近），
    ``H`` 为半衰期。令最近一期（``t = T``）权重为 1，则距今 k 期的权重为
    ``2^(−k/H)`` —— ``k = H`` 时恰为 0.5，这就是"半衰"的含义。

    Args:
        dates: 升序（或任意序，内部按序对齐）的日期序列。只有**顺序**重要，
            值的绝对大小不重要。
        half_life: 半衰期 H，单位为「期数」（月频合成时 H 就是月数）。
        as_of: 权重锚点（"距今"的参照期）。``None`` 表示最末期。

    Returns:
        ``np.ndarray``，与 ``dates`` **同序**的权重，最近期=1 并向后衰减。
        返回未归一化权重（调用方按需归一化；归一化不改变线性组合方向）。

    Notes:
        研报的 T 是回看窗口长度，即只取最近 T 期参与加权。本项目把"取窗口"
        交给调用方（先切 ``dates`` 再调本函数），避免两个职责混在一个函数里。
    """
    idx = pd.Index(dates)
    if len(idx) == 0:
        return np.zeros(0, dtype=float)
    pos = pd.Series(np.arange(len(idx)), index=idx)
    ref = pos.iloc[-1] if as_of is None else pos.get(pd.Timestamp(as_of), pos.iloc[-1])
    # k = 距锚点的期数（>=0 表示在锚点之前）
    k = (ref - pos).to_numpy(dtype=float)
    return np.power(2.0, -k / float(half_life))


# ===========================================================================
# 共享工具：因子面板 → 长矩阵
# ===========================================================================
def long_matrix(
    components: list[CompositeInput], returns_panel: pd.DataFrame | None
) -> tuple[np.ndarray, pd.MultiIndex, tuple[pd.Index, pd.Index]]:
    """把多因子面板对齐到同一 (date, code) 网格，铺成 (n_obs, n_factors) 长矩阵。

    行序 = **日期优先**（``MultiIndex.from_product([idx, cols])``），即同一交易日的
    全部股票连续排列——时序 CV 按「交易日边界」切折依赖此约定。

    公开接口：模型层 ``model.stacking`` 复用本函数（与 ``model.predictor._long_matrix``
    同口径），故不再以 ``_`` 前缀私有化。
    """
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


# ===========================================================================
# 2) PCA 主成分合成
# ===========================================================================
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
    X, obs, grid = long_matrix(components, returns_panel)
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
# 2b) 最大化 IC_IR / 最大化 IC（华泰多因子系列10 的两个主力方法）
# ===========================================================================
def solve_synthesis_weights(
    objective_w: np.ndarray,
    cov_like: np.ndarray,
    *,
    long_only: bool,
    ridge: float = 1e-8,
) -> np.ndarray:
    """解 ``max_w w'·objective_w  s.t.  w'·cov_like·w = 1``（或 w ≥ 0 时同式）。

    解析解为 ``w = cov_like⁻¹ · objective_w``（归一化不改变线性组合方向）。
    协方差阵可能奇异（因子数 ≥ 期数，或因子高度共线），故：
    1. 先试 **Cholesky**（最快、要求正定）；
    2. 失败退 **伪逆**（`pinv`，自动处理奇异 + 最小范数解）；
    3. 再加 ``ridge`` 到对角线兜底数值稳定性。

    ``long_only=True`` 时把负权重截为 0 并重归一 —— 研报对最大化 IC_IR 明确
    要求 ``w ≥ 0``；本项目用"截断 + 重归一"近似（非严格 QP 解，见函数 docstring
    的边界说明）。
    """
    C = np.asarray(cov_like, dtype=float)
    m = np.asarray(objective_w, dtype=float)
    if C.size == 0 or m.size == 0:
        return np.zeros_like(m)
    C = (C + C.T) / 2.0                              # 对称化（数值误差可能破坏对称）
    if ridge > 0:
        C = C + float(ridge) * np.eye(C.shape[0])

    w: np.ndarray | None = None
    try:
        L = np.linalg.cholesky(C)
        # C w = m  →  L L' w = m  →  两次三角求解
        y = np.linalg.solve(L, m)
        w = np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        w = None
    if w is None or not np.all(np.isfinite(w)) or np.allclose(w, 0.0):
        w = np.linalg.pinv(C) @ m                    # 奇异 → 伪逆（最小范数解）

    if long_only:
        w = np.maximum(w, 0.0)                       # 截断负权重（研报 w ≥ 0）
    s = float(np.abs(w).sum())
    if s <= 0 or not np.all(np.isfinite(w)):
        # 全被截没了（如 objective_w 与 Σ⁻¹ 方向完全相反）→ 退化为等权
        return np.ones_like(m) / max(len(m), 1)
    return w / s


def ic_matrix_from_components(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame | None,
    train_dates,
    *,
    half_life: float | None = None,
) -> pd.DataFrame:
    """各因子的**逐期 IC 序列矩阵**（index=date, columns=因子名）。

    这是最大化 IC_IR 的 ``Σ`` 的来源：研报 ``Σ`` = 历史 IC 协方差阵，
    即本函数输出按行求协方差。

    只用 ``train_dates`` 内的收益 —— 与 ``rebuild_train_weights`` 同一防未来函数
    口径（项目惯例，研报未讨论）。``half_life`` 不为 None 时对**日期方向**做加权
    （权重见 ``half_life_weights``），用于加权协方差 / 加权 IC 均值。
    """
    if returns_panel is None:
        raise ValueError("最大化 IC_IR 需要 returns_panel（Σ 来自历史 IC 序列）")
    if train_dates is None:
        raise ValueError("最大化 IC_IR 需要 train_dates（ĪC 与 Σ 只允许用训练段估计）")
    rts_tr = returns_panel.loc[train_dates]
    cols: dict[str, pd.Series] = {}
    for c in components:
        panel_tr = c.panel.loc[train_dates]
        panel_tr = panel_tr.reindex(columns=rts_tr.columns)
        cols[c.name] = calc_ic_series(panel_tr, rts_tr)
    ic = pd.DataFrame(cols)
    if half_life is not None and len(ic) > 0:
        wts = half_life_weights(ic.index, half_life)
        ic = ic.mul(wts, axis=0)
    return ic


def synthesize_ic_ir_max(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame,
    train_dates,
    *,
    long_only: bool = True,
    half_life: float | None = None,
    shrinkage: float = 0.5,
    returns_diagnostics: bool = False,
):
    """**最大化 IC_IR** 合成（研报口径：``w = Σ⁻¹·ĪC``）。

    研报《多因子系列10》的主力方法之一。记各因子历史 IC 序列的均值为 ``ĪC``、
    协方差阵为 ``Σ``，则最大化组合 IC_IR（≈ ``ĪC'w / √(w'Σw)``）的权重为
    ``w = Σ⁻¹·ĪC``。研报额外要求 ``w ≥ 0``（不允许做空因子）。

    Args:
        components: 参与合成的因子（已标准化）。
        returns_panel: 未来收益面板（date × code）。
        train_dates: **训练段日期**——``ĪC`` 与 ``Σ`` 只在此段估计
            （防未来函数，项目口径；研报未讨论）。
        long_only: 是否施加 ``w ≥ 0``。研报对最大化 IC_IR 要求该约束。
        half_life: 半衰期 H（期数）。给定时对 IC 序列做半衰加权
            （``w_t = 2^((t−T−1)/H)``），即"历史 IC 半衰加权 + 最大化 IC_IR"。
        shrinkage: Σ 的收缩强度（0~1）。因子数 ≥ 期数时 Σ 必然奇异，
            收缩到对角可改善条件数。默认 0.5（**自拟参数，研报只提到用压缩估计**）。
        returns_diagnostics: 为 True 时额外返回诊断字典。

    Returns:
        ``composite`` 面板；``returns_diagnostics=True`` 时返回
        ``(composite, diag)``，diag 含 ``weights`` / ``ic_mean`` / ``n_periods`` /
        ``cond``（Σ 条件数）/ ``long_only`` / ``shrinkage`` / ``sign_flipped``。

    **符号口径（2026-09-16 修正）**：``ĪC`` 是按**符号对齐后**的因子面板重算的，
    因此 ``mu ≥ 0`` 恒成立，``w`` 是"对齐基"下的权重。原始 IC 为负的因子，
    其面板在内部被取负，故 ``weights[name] > 0`` 表示**看多该因子的对齐方向**
    （= 看空其原始方向）。诊断里的 ``sign_flipped[name]`` 显式标出哪些因子被
    翻转，避免调用方按原始方向误读权重符号。

    **复现边界（引用时必须披露）**：
    1. 研报用"**压缩估计协方差**"，未指明具体估计器；本项目用
       ``shrinkage · diag(Σ) + (1−shrinkage) · Σ`` 的对角收缩（自拟）。
    2. ``w ≥ 0`` 用"**截断负权重后重归一**"近似，**不是带约束 QP 的精确解**。
       截断会破坏 ``Σ⁻¹`` 的最优性（截断后目标值 ≤ 原解）—— 这是刻意的工程取舍：
       因子数通常为个位数，重归一后的解与 QP 解在同一量级，且免去 cvxpy 依赖。
       需要精确解时用 :func:`optimize.solver.solve_portfolio` 的带约束 QP。
    3. 研报的 T 扫描（T ∈ {3,6,9,12,24,36} 个月）由调用方切 ``train_dates`` 实现。
    """
    if not components:
        raise ValueError("components 为空")
    comps = [_align_sign_by_ic(c) for c in components]
    sign_flipped = {c0.name: bool(c0.ic < 0) for c0 in components}
    ic = ic_matrix_from_components(comps, returns_panel, train_dates,
                                    half_life=half_life)
    ic = ic.dropna(how="any")                       # 只保留全部因子都有 IC 的期
    if len(ic) < 3:
        raise ValueError(f"训练段有效 IC 期数不足（{len(ic)} < 3），无法估计 Σ")

    mu = ic.mean(axis=0).to_numpy(dtype=float)      # ĪC
    S = np.cov(ic.to_numpy(dtype=float), rowvar=False)
    S = np.atleast_2d(S)
    if half_life is not None:
        # 半衰加权已在 _ic_matrix 里乘到 IC 上；加权均值/协方差需按权重归一
        wts = half_life_weights(ic.index, half_life)
        wts = wts / wts.sum()
        X = ic.to_numpy(dtype=float)
        mu = (X * wts[:, None]).sum(axis=0)
        Xc = X - mu
        S = (Xc * wts[:, None]).T @ Xc / max(1.0 - (wts ** 2).sum(), 1e-12)

    # 收缩到对角（改善条件数；因子数 ≥ 期数时 Σ 必奇异）
    d = np.diag(np.diag(S))
    S_shrunk = float(shrinkage) * d + (1.0 - float(shrinkage)) * S

    w = solve_synthesis_weights(mu, S_shrunk, long_only=long_only)
    idx = comps[0].panel.index.intersection(np.asarray(returns_panel.index))
    cols = comps[0].panel.columns
    for c in comps[1:]:
        idx = idx.intersection(c.panel.index)
        cols = cols.intersection(c.panel.columns)
    composite = sum(wi * c.panel.reindex(index=idx, columns=cols).fillna(0.0)
                    for wi, c in zip(w, comps))
    composite = standardize_zscore(composite)

    if not returns_diagnostics:
        return composite
    cond = float(np.linalg.cond(S_shrunk)) if S_shrunk.size else float("nan")
    return composite, {
        "method": "ic_ir_max", "weights": dict(zip([c.name for c in comps], w)),
        "ic_mean": dict(zip([c.name for c in comps], mu)),
        "n_periods": int(len(ic)), "cond": cond,
        "long_only": bool(long_only), "half_life": half_life,
        "shrinkage": float(shrinkage),
        # weights 为"对齐基"权重：sign_flipped[name]=True 表示该因子面板内部被取负，
        # 故 weights>0 等价于看空其原始方向（2026-09-16 补，防误读符号）。
        "sign_flipped": sign_flipped,
    }


def synthesize_ic_max(
    components: list[CompositeInput],
    returns_panel: pd.DataFrame | None = None,
    train_dates=None,
    *,
    long_only: bool = False,
    returns_diagnostics: bool = False,
):
    """**最大化 IC** 合成（研报口径：``w = V⁻¹·ĪC``）。

    与最大化 IC_IR 的关键区别在 ``V`` 的来源：研报的 ``V`` 是
    **当前截面因子值相关系数阵**（不是历史 IC 协方差阵）。直观上，
    IC_IR 法关心"因子 IC 序列的稳定性"，IC 法关心"因子之间的当期共线性"——
    共线性越强，``V⁻¹`` 越倾向于压低冗余因子的权重。

    Args:
        components: 参与合成的因子（已标准化）。
        returns_panel: 未来收益面板；**仅用于 ĪC 的估计**。
        train_dates: **训练段日期**（ĪC 只在此段估计）。``None`` 表示用
            ``returns_panel`` 全部日期 —— **仅在明确只传训练段面板时使用**；
            直接传含测试段的完整收益面板而不给 train_dates 会引入未来函数。
        long_only: 是否施加 ``w ≥ 0``（研报**只对 IC_IR 法**要求该约束，
            IC 法默认 False）。
        returns_diagnostics: 同 :func:`synthesize_ic_ir_max`。

    **复现边界**：研报未指明 ``V`` 的估计期次与是否压缩，本实现取**决策时点**
    （``panels.index[-1]``，通常为训练段末）截面 + 与 IC_IR 相同的对角收缩
    （0.5，自拟）。

    **两种防未来函数口径（2026-09-16 补齐，09-24 修正②的决策时点）**：
    1. ``ĪC`` 只由 ``train_dates`` 段估计（与 :func:`synthesize_ic_ir_max` 一致）；
    2. ``V`` 只取**决策时点单截面**：给了 ``train_dates`` 取其末位（训练段末），
       否则取 ``panel.index[-1]``——此时调用方必须只传训练段面板；传全时段
       面板而不给 ``train_dates`` 会引入未来函数。
    """
    if not components:
        raise ValueError("components 为空")
    comps = [_align_sign_by_ic(c) for c in components]
    sign_flipped = {c0.name: bool(c0.ic < 0) for c0 in components}

    # ĪC：只由训练段估计
    if returns_panel is not None:
        seg = returns_panel.index if train_dates is None else train_dates
        ic = ic_matrix_from_components(comps, returns_panel,
                                        seg).dropna(how="any")
        mu = ic.mean(axis=0).to_numpy(dtype=float) if len(ic) else \
            np.array([c.ic for c in comps], dtype=float)
    else:
        mu = np.array([c.ic for c in comps], dtype=float)

    # V：**决策时点**的截面因子值相关阵。决策时点 = ``train_dates[-1]``
    # （给了 train_dates 时）；未给时退 ``panel.index[-1]``（调用方须保证
    # 面板只含训练段，见 docstring）。
    # 2026-09-16 修复：原实现用 long_matrix(comps, None) 铺全样本 → V 隐式
    # 含测试段截面结构。2026-09-24 再修：恒取 ``panel.index[-1]`` 在
    # 「全时段面板 + train_dates」调用形态（mf10_t_scan）下仍取到测试段末
    # ——决策时点必须跟随 train_dates。
    if train_dates is not None and len(list(train_dates)):
        last_date = pd.Timestamp(pd.DatetimeIndex(train_dates)[-1])
    else:
        last_date = comps[0].panel.index[-1]

    def _row_at(panel: pd.DataFrame) -> pd.DataFrame:
        pos = panel.index.searchsorted(last_date, side="right") - 1
        if pos < 0:
            raise ValueError(f"面板早于决策时点 {last_date}，无法取 V 截面")
        return panel.iloc[pos:pos + 1]

    snap = [CompositeInput(name=c.name, panel=_row_at(c.panel), ic=c.ic, ir=c.ir)
            for c in comps]
    X, _obs, (idx, cols) = long_matrix(snap, None)
    V = np.corrcoef(np.nan_to_num(X, nan=0.0), rowvar=False)
    V = np.atleast_2d(V)
    V = np.nan_to_num(V, nan=0.0)
    np.fill_diagonal(V, 1.0)
    V_shrunk = 0.5 * np.eye(V.shape[0]) + 0.5 * V      # 对角收缩（自拟，同 IC_IR）

    w = solve_synthesis_weights(mu, V_shrunk, long_only=long_only)
    # 组合面板仍然用**全样本**因子值（权重只由训练段/决策时点定，无未来函数）
    Xf, _obs2, (idx_f, cols_f) = long_matrix(comps, None)
    idx, cols = idx_f, cols_f
    composite = sum(wi * c.panel.reindex(index=idx, columns=cols).fillna(0.0)
                    for wi, c in zip(w, comps))
    composite = standardize_zscore(composite)

    if not returns_diagnostics:
        return composite
    return composite, {
        "method": "ic_max", "weights": dict(zip([c.name for c in comps], w)),
        "ic_mean": dict(zip([c.name for c in comps], mu)),
        "cond": float(np.linalg.cond(V_shrunk)) if V_shrunk.size else float("nan"),
        "long_only": bool(long_only),
        "as_of": last_date,                 # V 的估计时点（决策时点）
        "sign_flipped": sign_flipped,
    }


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
    # OLS t 统一走 stats.significance（std 固定 ddof=1，与旧内联写法逐位一致；
    # n < 2 保持旧的 0.0 边界）。2026-09-11 第二批口径统一。
    _inf = mean_inference(ic, robust=False)
    t = _inf["t_stat"] if _inf["n"] >= 2 else 0.0
    return {
        "ic_mean": m, "ic_std": s, "ir": ir,
        "ic_win_rate": float((ic > 0).mean()), "t_stat": t, "n": n,
    }
