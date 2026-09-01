"""
DPP（Determinantal Point Process）集合级因子筛选
================================================

研报对齐（国金 Alpha 掘金系列之二十四 §3.1）：在因子进入合成/入库之前，先做
**集合级**多样性筛选 —— 最大化所选子集相关矩阵的 log-determinant（log-det）。
直观上 log-det 越大，入选因子之间越不共线、张成的信息空间越大、集合冗余越低。

与现有"两两贪心去重"（factor/gflownet/selection.py: ``select_low_corr``）的区别：
- 两两去重：局部判据、顺序依赖。遇到 A~B=0.8, B~C=0.8, A~C=0.1 的三角结构，
  若先选 A 则 B 被剔、C 又因与 B 相关被连锁误杀。
- DPP：集合级全局判据，直接找使 log det(L_S) 最大的子集（几何意义=子集向量
  张成的平行体体积最大），天然排斥相似元素，结果与顺序无关。

实现：k-DPP 的贪心 log-det 最大化（子模函数最大化，理论近似保证 1-1/e）。
核矩阵 L = diag(q) · S · diag(q)，其中
    S_ij = exp(-(1 - |corr_ij|) / sigma)   # 高斯相似度核（恒半正定）
    q_i  = 质量项（如 |IC| 归一化），默认全 1（纯多样性，对齐研报 DPP 口径）
数值稳定：Cholesky 增量更新，每次加入使 log det 增量最大的因子。

用法（纯函数，不依赖因子库）::

    corr = corr_matrix(panels, method="cross")   # date×code 面板 dict -> 因子×因子相关
    idx, trace = greedy_logdet_dpp(kernel(corr), k=50, quality=ic_abs_norm)
    selected = corr.columns[idx]

因子库集成见 :meth:`FactorLibrary.select_diverse`。
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("dpp_selection")

__all__ = [
    "corr_matrix",
    "similarity_kernel",
    "quality_from_ic",
    "greedy_logdet_dpp",
    "dpp_select",
    "pairwise_dedup",
]


# ===========================================================================
# 相关矩阵（向量化批量实现）
# ===========================================================================
def _rank_rows(X: np.ndarray) -> np.ndarray:
    """逐行 rank（沿 axis=1），NaN 保留；行内有效元素 rank 1..n。

    X: (N, M) float，可含 NaN。返回同形状 rank 矩阵（无效位 NaN）。
    """
    m = ~np.isnan(X)
    Xr = np.where(m, X, -np.inf)
    order = np.argsort(Xr, axis=1, kind="stable")
    ranks = np.full_like(Xr, np.nan, dtype=float)
    rows = np.arange(X.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, X.shape[1] + 1)
    return np.where(m, ranks, np.nan)


def _zscore_rows(R: np.ndarray) -> np.ndarray:
    """逐行 z-score（按行内有效元素），无效位置 0（不贡献相关）。"""
    m = ~np.isnan(R)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(R, axis=1, keepdims=True)
        std = np.nanstd(R, axis=1, keepdims=True)
    Z = np.where(std > 1e-12, (R - mean) / np.maximum(std, 1e-12), 0.0)
    return np.where(m, Z, 0.0)


def _coerce_datetime_index(df: pd.DataFrame) -> pd.DataFrame | None:
    """把面板索引统一为 DatetimeIndex；无法对齐（RangeIndex 等）返回 None。"""
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if pd.api.types.is_integer_dtype(df.index):
        s = df.index.astype(str)
        if np.all(s.map(len) == 8):
            df = df.copy()
            df.index = pd.to_datetime(s, format="%Y%m%d")
            return df
    return None


def _panel_chunks(panels: dict[str, pd.DataFrame], window=None, chunk_days: int = 90):
    """统一 (日期×代码) 网格后按日期分块，逐块产出 (chunk_dates, X)。

    X: N×(nd×M) float32，每因子一行，NaN 填充缺失（日期/代码不重叠部分）。
    非日期索引的面板（RangeIndex 等）自动跳过。
    """
    cleaned: dict[str, pd.DataFrame] = {}
    for n, p in panels.items():
        c = _coerce_datetime_index(p)
        if c is not None and len(c) > 0:
            cleaned[n] = c
    panels = cleaned
    all_dates = sorted(set().union(*[set(p.index) for p in panels.values()]))
    all_codes = sorted(set().union(*[set(p.columns) for p in panels.values()]))
    if window is not None:
        s, e = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        all_dates = [d for d in all_dates if s <= d <= e]
    names = list(panels.keys())
    M = len(all_codes)
    for i in range(0, len(all_dates), chunk_days):
        chunk_dates = all_dates[i:i + chunk_days]
        X = np.full((len(names), len(chunk_dates), M), np.nan, dtype=np.float32)
        for j, n in enumerate(names):
            df = panels[n]
            X[j] = df.reindex(index=chunk_dates, columns=all_codes).to_numpy(
                dtype=np.float32)
        yield chunk_dates, X.reshape(len(names), -1)


def corr_matrix(panels: dict[str, pd.DataFrame] | pd.DataFrame,
                method: str = "cross", min_overlap_dates: int = 30,
                min_overlap_codes: int = 10, window=None) -> pd.DataFrame:
    """从面板集合构造 因子×因子 相关矩阵（向量化，适用数百因子）。

    Args:
        panels: {name: date×code 面板}；传 DataFrame 时按列视为因子（列名即因子名）。
        method:
            - "cross": 逐日截面 spearman（rank→z-score→pearson）取均值，对齐
              factor_library 冗余预检口径，更稳健（默认）。
            - "flat": 整面板 flatten 向量的 spearman（对齐 GFlowNet selection 口径）。
        min_overlap_dates: 参与平均的公共日期数下限（cross；不足记 0）。
        min_overlap_codes: 单日有效股票数下限（cross；不足的日期不计）。
        window: (start, end) 限制比较窗口（可选，None=全部日期）。

    Returns:
        DataFrame（index=columns=因子名），对角线为 1。
    """
    if isinstance(panels, pd.DataFrame):
        panels = {c: panels[[c]] for c in panels.columns}
    # 统一清洗：跳过非日期索引面板，保证 N 与 _panel_chunks 一致
    cleaned: dict[str, pd.DataFrame] = {}
    for n, p in panels.items():
        c = _coerce_datetime_index(p)
        if c is not None and len(c) > 0:
            cleaned[n] = c
    panels = cleaned
    if not panels:
        raise ValueError("无可用面板（全部为非日期索引/空面板）")
    names = list(panels.keys())
    N = len(names)
    corr = np.zeros((N, N))

    if method == "flat":
        P = np.zeros((N, N))
        Q = np.zeros((N, N))          # 共同有效样本计数
        total = 0
        for _dates, X in _panel_chunks(panels, window):
            Z = _zscore_rows(_rank_rows(X))
            m = ~np.isnan(X)
            P += Z @ Z.T
            Q += m.astype(np.float64) @ m.T.astype(np.float64)  # bool matmul 按逻辑或，须转数值
            total += X.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.sqrt(np.clip(np.diag(P), 0, None))
            corr = P / np.outer(dd, dd)
        # 共同有效样本 < max(20, 0.5×全样本) → 视为无重叠，记 0
        need = max(20, int(0.5 * total))
        corr = np.where(Q < need, 0.0, corr)
    else:  # cross：逐日截面相关（rank→z-score→pearson）对日期平均
        S = np.zeros((N, N))
        cnt = np.zeros((N, N))
        for _dates, X in _panel_chunks(panels, window, chunk_days=90):
            nd = len(_dates)
            M = X.shape[1] // nd
            X3 = X.reshape(N, nd, M)
            # 一次性对 (N×nd) 行做截面 rank + z-score，再拆回
            flat = X3.reshape(N * nd, M)
            valid_n = np.sum(~np.isnan(flat), axis=1).reshape(N, nd)  # N×nd
            Z3 = _zscore_rows(_rank_rows(flat)).reshape(N, nd, M)
            P = np.einsum("idc,jdc->dij", Z3, Z3, optimize=True)      # nd×N×N
            dd = np.sqrt(np.clip(np.diagonal(P, axis1=1, axis2=2), 0, None))
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_d = P / np.einsum("di,dj->dij", dd, dd)
            vT = valid_n.T  # (nd, N)
            ok = (vT[:, None, :] >= min_overlap_codes) & \
                 (vT[:, :, None] >= min_overlap_codes)   # ok[d,i,j] = 该日两因子均有效
            good = ok & np.isfinite(corr_d)
            S += np.where(good, corr_d, 0.0).sum(axis=0)
            cnt += ok.sum(axis=0)
        corr = np.where(cnt >= min_overlap_dates, S / np.maximum(cnt, 1), 0.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=names, columns=names)


# ===========================================================================
# 核矩阵
# ===========================================================================
def similarity_kernel(corr: pd.DataFrame | np.ndarray,
                      sigma: float = 0.2, abs_corr: bool = True) -> np.ndarray:
    """把相关矩阵映射为恒半正定的相似度核 S_ij = exp(-(1 - |corr|) / sigma)。

    sigma 越小对高相关惩罚越强（sigma=0.2 时：|corr|=0 → 0.007，0.7 → 0.22，
    0.9 → 0.61，0.99 → 0.95）。高斯核保证 L 半正定，log det 有定义。
    """
    c = np.abs(corr.to_numpy()) if abs_corr else corr.to_numpy().astype(float)
    c = np.clip(c, 0.0, 1.0)
    return np.exp(-(1.0 - c) / sigma)


def quality_from_ic(ic: pd.Series, floor: float = 0.5) -> np.ndarray:
    """质量项：|IC| 归一化到 [floor, 1]（保底为正，避免质量 0 因子概率归零）。

    floor=0.5 → 最强因子 q=1、最弱 q=0.5；纯多样性可传 None（全 1）。
    """
    v = ic.reindex(ic.index).abs().to_numpy(dtype=float)
    mx = np.nanmax(v) if np.isfinite(v).any() else 0.0
    if mx <= 0:
        return np.ones(len(v))
    return floor + (1.0 - floor) * np.nan_to_num(v) / mx


# ===========================================================================
# 贪心 log-det 最大化（k-DPP 确定性筛选）
# ===========================================================================
def greedy_logdet_dpp(L: np.ndarray, k: int, quality: Sequence[float] | None = None,
                      ) -> tuple[list[int], list[float]]:
    """贪心最大化 log det(L_S)，返回 (选中下标列表, log-det 轨迹)。

    L: n×n 半正定核矩阵（需已含质量项，或 quality 传质后内部加权）。
    k: 目标子集大小（>= 1；k > n 时取 n）。
    数值：Cholesky 增量更新 —— 对候选 j，log det 增量
        delta_j = log(L_jj - ||chol^{-1} L_Sj||²)，取最大者加入。
    """
    L = np.asarray(L, dtype=float)
    n = L.shape[0]
    if quality is not None:
        q = np.asarray(quality, dtype=float)
        if q.shape != (n,):
            raise ValueError(f"quality 长度 {q.shape} 与 L 维度 {n} 不符")
        L = (q[:, None] * L) * q[None, :]  # diag(q)·L·diag(q)
    k = max(1, min(int(k), n))
    diag = np.diag(L)
    if np.any(diag <= 0):
        # 数值兜底：半正定核对角线应 >0；<=0 加小 epsilon 保持 log 有定义
        L = L + np.eye(n) * 1e-10
        diag = np.diag(L)

    selected: list[int] = []
    chol: np.ndarray | None = None
    logdet_trace: list[float] = []
    logdet = 0.0
    remaining = list(range(n))

    for _ in range(k):
        if not remaining:
            break
        rem = np.asarray(remaining)
        if len(selected) == 0:
            delta = np.log(np.maximum(diag[rem], 1e-12))
        else:
            S = np.asarray(selected)
            L_Sj = L[np.ix_(S, rem)]            # |S| × m
            B = np.linalg.solve(chol, L_Sj)      # chol·B = L_Sj
            delta = np.log(np.maximum(
                diag[rem] - np.einsum("ij,ij->j", B, B), 1e-12))
        j = int(rem[int(np.argmax(delta))])
        logdet += float(np.max(delta))
        logdet_trace.append(logdet)
        selected.append(j)
        remaining.remove(j)
        # Cholesky 增量更新
        if len(selected) == 1:
            chol = np.array([[np.sqrt(max(diag[j], 1e-12))]])
        else:
            v = np.linalg.solve(chol, L[np.asarray(selected[:-1]), j])
            d = diag[j] - float(v @ v)
            new_chol = np.zeros((len(selected), len(selected)))
            new_chol[:-1, :-1] = chol
            new_chol[-1, :-1] = v
            new_chol[-1, -1] = np.sqrt(max(d, 1e-12))
            chol = new_chol
    return selected, logdet_trace


def dpp_select(corr: pd.DataFrame, k: int, quality: pd.Series | Sequence[float] | None = None,
               sigma: float = 0.2, abs_corr: bool = True,
               ) -> dict:
    """一站式 DPP 筛选：相关矩阵 -> 核 -> 贪心 log-det -> 汇总。

    Args:
        corr: 因子×因子相关矩阵（index/columns 为因子名）。
        k: 目标入选数量。
        quality: 质量项（与 corr.index 对齐的 Series 或 ndarray）；None=纯多样性。
        sigma / abs_corr: 核参数，见 :func:`similarity_kernel`。

    Returns:
        dict: selected(list[str]) / logdet_trace / k / n_pool /
              max_abs_corr_pool / mean_abs_corr_pool /
              max_abs_corr_selected / mean_abs_corr_selected /
              logdet_selected / logdet_pool。
    """
    names = list(corr.index)
    L = similarity_kernel(corr, sigma=sigma, abs_corr=abs_corr)
    if quality is not None:
        if isinstance(quality, pd.Series):
            quality = quality.reindex(names).to_numpy()
        quality = np.nan_to_num(np.asarray(quality, dtype=float))
    idx, trace = greedy_logdet_dpp(L, k=k, quality=quality)
    sel = [names[i] for i in idx]
    pool = np.abs(corr.to_numpy())
    sub = np.abs(corr.to_numpy()[np.ix_(idx, idx)])
    off = lambda M: M[~np.eye(M.shape[0], dtype=bool)]  # noqa: E731
    return {
        "selected": sel,
        "logdet_trace": trace,
        "k": len(sel),
        "n_pool": len(names),
        "max_abs_corr_pool": float(off(pool).max()),
        "mean_abs_corr_pool": float(off(pool).mean()),
        "max_abs_corr_selected": float(off(sub).max()) if len(sel) > 1 else 0.0,
        "mean_abs_corr_selected": float(off(sub).mean()) if len(sel) > 1 else 0.0,
        "logdet_selected": float(trace[-1]) if trace else float("nan"),
        "logdet_pool": float(np.linalg.slogdet(L)[1]),
    }


# ===========================================================================
# 对比基线：两两贪心去重（模拟现有 select_low_corr / check_dup 逻辑）
# ===========================================================================
def pairwise_dedup(corr: pd.DataFrame, order: Sequence[str] | None = None,
                   threshold: float = 0.7, mode: str = "abs") -> list[str]:
    """按给定顺序贪心入选，与已入选因子 |corr| > threshold 则剔除（局部判据）。

    用于与 DPP 做同口径对比；order=None 时按池内 mean |corr| 升序（最独立优先）。
    """
    names = list(corr.columns)
    C = corr.to_numpy()
    order = [n for n in order if n in names] if order else names
    selected: list[str] = []
    for name in order:
        i = names.index(name)
        if any(abs(C[i, names.index(s)]) > threshold for s in selected):
            continue
        selected.append(name)
    return selected
