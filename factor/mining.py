"""
因子挖掘 —— 候选生成 + 批量 IC 检验 + 显著性筛选
================================================

用算子空间在原始特征（OHLCV + 财务字段）上**批量生成候选因子**，
对每个候选用 rank IC 序列做检验，再用 **Benjamini-Hochberg FDR**
做多重检验校正，筛掉"挖出来的纯噪声"。

闭环：算子空间 → 候选生成 → 批量 IC/IR → t 检验 + FDR → 排名表。

输入约定
--------
panel: dict[str, DataFrame]，key 为特征名（open/high/low/close/volume/amount
       + 财务字段如 OPERA_REV/NET_PRO_INCL_MIN_INT_INC ...），
       value 为 date×code 面板。
returns_panel: DataFrame(date, code)，**已对齐为当日因子对应的未来一期收益**
               （即 returns[d] = close[d+1]/close[d]-1）。
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from factor.operators import (
    CS_OPS, DEFAULT_FEATURES, DEFAULT_WINDOWS, TS_OPS, cs_rank, op_registry,
    ts_corr, ts_mean,
)
from stats.ic import calc_ic_series, calc_ir, calc_ic_decay, factor_autocorr


@contextmanager
def _suppress_const_warning():
    """屏蔽 spearmanr 在常数输入时的 ConstantInputWarning（已按 NaN 处理）。"""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="An input array is constant")
        yield


# ===========================================================================
# 并行评估 worker（进程池；模块级函数以便 pickle）
# ===========================================================================
_EVAL_CTX: dict = {}


def _init_eval_worker(panel, returns_panel, features):
    """每个 worker 进程初始化一次，持有面板/收益/特征（避免每次任务重复 pickle）。"""
    _EVAL_CTX["panel"] = panel
    _EVAL_CTX["returns"] = returns_panel
    _EVAL_CTX["features"] = features


def _eval_candidate_worker(task: tuple) -> dict | None:
    """worker：按公式字符串重建求值器，计算 IC 摘要。返回 row dict 或 None（无效/异常）。"""
    from stats.robust_stats import nw_tstat
    from factor.formula import formula_builder

    name, method, min_obs, robust = task
    ctx = _EVAL_CTX
    try:
        builder = formula_builder(name, features=ctx["features"])
        fp = builder(ctx["panel"])
        if fp is None or fp.empty:
            return None
        with np.errstate(all="ignore"), _suppress_const_warning():
            ic = calc_ic_series(fp, ctx["returns"], method=method).dropna()
    except Exception:
        return None
    n = len(ic)
    if n < min_obs:
        return None
    m = float(ic.mean())
    s = float(ic.std())
    ir = calc_ir(ic)
    t = m / (s / np.sqrt(n)) if s > 0 else 0.0
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))
    t_nw, _se_nw, _lag = nw_tstat(ic) if robust else (np.nan, np.nan, 0)
    p_nw = 2.0 * (1.0 - stats.t.cdf(abs(t_nw), df=n - 1)) if robust else np.nan
    return {
        "name": name,
        "ic_mean": m,
        "ic_std": s,
        "ir": ir,
        "ic_win_rate": float((ic > 0).mean()),
        "t_stat": t,
        "p_value": p,
        "t_stat_nw": t_nw,
        "p_value_nw": p_nw,
        "n": n,
    }


# ===========================================================================
# 候选因子
# ===========================================================================
@dataclass
class Candidate:
    """一个候选因子：公式名 + 构造函数。"""
    name: str
    build: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


# ---- 构造函数工厂（用默认参数闭包，避免循环晚绑定）----
def _unary_ts(func, feat, w):
    def build(panel):
        return func(panel[feat], w)
    return build


def _unary_cs(func, feat):
    def build(panel):
        return func(panel[feat])
    return build


def _cs_ts(cs_func, ts_func, feat, w):
    def build(panel):
        return cs_func(ts_func(panel[feat], w))
    return build


def _op_tsmean(op_func, feat, w):
    def build(panel):
        return op_func(panel[feat], ts_mean(panel[feat], w))
    return build


def _pair(op_func, a, b):
    def build(panel):
        return op_func(panel[a], panel[b])
    return build


def _ts_corr_pair(a, b, w):
    def build(panel):
        return ts_corr(panel[a], panel[b], w)
    return build


def generate_candidates(
    features: Sequence[str] = DEFAULT_FEATURES,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    depth: int = 2,
) -> list[Candidate]:
    """枚举算子空间生成候选因子。

    depth=1: 单目时序 + 单目截面算子作用于各特征。
    depth=2: 额外加入 cs_rank(ts_op) 包装、价量二元算子、ts_corr 特征对。
    """
    reg = op_registry()
    cands: list[Candidate] = []

    # 1) 单目时序: ts_op(feature, window)
    for f in features:
        for op in TS_OPS:
            if op.arity != 1:
                continue
            for w in windows:
                cands.append(Candidate(f"{op.name}({f},{w})", _unary_ts(op.func, f, w)))

    # 2) 单目截面: cs_op(feature)
    for f in features:
        for op in CS_OPS:
            if op.arity != 1:
                continue
            cands.append(Candidate(f"{op.name}({f})", _unary_cs(op.func, f)))

    if depth >= 2:
        # 3) cs_rank(ts_op(feature, window)) —— 把时序统计做截面排名，最常用 alpha 形态
        for f in features:
            for op in TS_OPS:
                if op.arity != 1:
                    continue
                for w in windows:
                    cands.append(Candidate(
                        f"cs_rank({op.name}({f},{w}))", _cs_ts(cs_rank, op.func, f, w)
                    ))

        # 4) 价量二元: op(feature, ts_mean(feature, window)) —— 如 close/ts_mean(close,20)
        for f in features:
            for w in windows:
                for name in ("div", "sub", "add", "mul"):
                    op = reg[name]
                    cands.append(Candidate(
                        f"{name}({f},ts_mean({f},{w}))", _op_tsmean(op.func, f, w)
                    ))

        # 5) 跨特征对: op(high,low) / op(close,open) / op(amount,volume) ...
        for a, b in [("high", "low"), ("close", "open"),
                     ("amount", "volume"), ("high", "close"), ("low", "close")]:
            if a not in features or b not in features:
                continue
            for name in ("div", "sub"):
                cands.append(Candidate(f"{name}({a},{b})", _pair(reg[name].func, a, b)))

        # 6) ts_corr 特征对
        for a, b in [("close", "volume"), ("close", "amount"),
                     ("high", "low"), ("open", "close")]:
            if a not in features or b not in features:
                continue
            for w in windows:
                cands.append(Candidate(f"ts_corr({a},{b},{w})", _ts_corr_pair(a, b, w)))

    return cands


# ===========================================================================
# 批量 IC 检验 + 显著性筛选
# ===========================================================================
@dataclass
class EvalResult:
    name: str
    ic_mean: float
    ic_std: float
    ir: float
    ic_win_rate: float
    ic_decay5: float
    ic_decay10: float
    autocorr: float
    t_stat: float
    p_value: float
    n: int
    significant: bool


def _benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    """BH 多重检验校正，返回每个假设是否拒绝（即因子是否显著）。"""
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    # k* = max{k : p_(k) <= k*q/n}
    k_max = 0
    for i in range(n):
        if sorted_p[i] <= (i + 1) * q / n:
            k_max = i + 1
    passed = np.zeros(n, dtype=bool)
    if k_max > 0:
        passed[order[:k_max]] = True
    return passed


def evaluate_candidates(
    candidates: list[Candidate],
    panel: dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    method: str = "spearman",
    fdr_q: float = 0.05,
    min_obs: int = 10,
    sort_by: str = "ir",
    detail_n: int = 50,
    decay_lags: tuple[int, ...] = (5, 10),
    robust: bool = True,
    verbose: bool = False,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """批量评估候选因子，返回 Alphalens 式标准摘要排名表。

    **并行化（2026-08-03 新增）**：``n_jobs > 1`` 时用进程池并行评估候选
    （候选 ``build`` 闭包不可 pickle，worker 内用 ``factor.formula.formula_builder``
    从公式字符串重建求值器；panel/returns 通过 initializer 每进程共享一份）。
    候选按 name 去重（并行路径），结果与串行等价。

    Args:
        candidates: generate_candidates 输出。
        panel: 特征面板字典。
        returns_panel: 未来一期收益面板（已对齐到当日）。
        method: 'spearman'(rank IC, 默认) | 'pearson'。
        fdr_q: BH-FDR 显著性水平（默认 0.05）。
        min_obs: IC 序列有效观测数下限。
        sort_by: 排序主轴，'ir'(默认, 业界标准) | 't'。均按绝对值降序。
        detail_n: 仅对前 N 个因子计算 IC 衰减/自相关（避免全量昂贵计算）。
        decay_lags: 计算 IC 衰减的持有期。
        robust: True 时额外输出 t_stat_nw / p_value_nw（**Newey-West 自相关
            稳健显著性**，业内对强自相关 IC 序列的标配）。**significant 列
            基于 p_value_nw 做 BH-FDR**（2026-08-03 修复）：IC 序列强自相关时
            OLS p 系统性偏小、会标出伪显著；robust=False 或 p_nw 缺失（NaN）
            时回退 OLS p。
        n_jobs: 并行进程数。1（默认）串行；>1 且候选数 > 16 时走进程池。
    Returns:
        DataFrame，按 |ir| 降序，列含 ic_mean/ic_std/ir/ic_win_rate/
        ic_decay5/ic_decay10/autocorr/t_stat/p_value/t_stat_nw/p_value_nw/
        significant。
    """
    from stats.robust_stats import nw_tstat

    cand_map = {c.name: c for c in candidates}
    rows: list[dict] = []

    if n_jobs > 1 and len(candidates) > 16:
        from concurrent.futures import ProcessPoolExecutor

        features = list(panel.keys())
        # 公式字符串即 name；并行路径按名去重（保持顺序），结果与串行等价
        names = list(dict.fromkeys(c.name for c in candidates))
        tasks = [(nm, method, min_obs, robust) for nm in names]
        chunksize = max(1, len(tasks) // (n_jobs * 4))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_eval_worker,
            initargs=(panel, returns_panel, features),
        ) as ex:
            for r in ex.map(_eval_candidate_worker, tasks, chunksize=chunksize):
                if r is not None:
                    rows.append(r)
        if verbose:
            print(f"  ...并行评估完成: {len(rows)}/{len(tasks)} 有效", flush=True)
    else:
        for i, c in enumerate(candidates):
            try:
                fp = c.build(panel)
                if fp is None or fp.empty:
                    continue
                with np.errstate(all="ignore"), _suppress_const_warning():
                    ic = calc_ic_series(fp, returns_panel, method=method).dropna()
            except Exception:
                # 候选构造或 IC 计算异常（形状不一致/全 NaN/常数列等）→ 跳过
                continue
            n = len(ic)
            if n < min_obs:
                continue
            m = float(ic.mean())
            s = float(ic.std())
            ir = calc_ir(ic)
            t = m / (s / np.sqrt(n)) if s > 0 else 0.0
            p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))
            t_nw, _se_nw, _lag = nw_tstat(ic) if robust else (np.nan, np.nan, 0)
            p_nw = 2.0 * (1.0 - stats.t.cdf(abs(t_nw), df=n - 1)) if robust else np.nan
            rows.append({
                "name": c.name,
                "ic_mean": m,
                "ic_std": s,
                "ir": ir,
                "ic_win_rate": float((ic > 0).mean()),
                "t_stat": t,
                "p_value": p,
                "t_stat_nw": t_nw,
                "p_value_nw": p_nw,
                "n": n,
            })
            if verbose and (i + 1) % 50 == 0:
                print(f"  ...已评估 {i + 1}/{len(candidates)}", flush=True)

    if not rows:
        return pd.DataFrame(columns=[
            "name", "ic_mean", "ic_std", "ir", "ic_win_rate",
            "ic_decay5", "ic_decay10", "autocorr",
            "t_stat", "p_value", "t_stat_nw", "p_value_nw", "n", "significant",
        ])

    df = pd.DataFrame(rows)
    # BH-FDR 决策：默认基于 NW 稳健 p（自相关 IC 下 OLS p 会虚高标伪显著）；
    # p_nw 缺失（robust=False / 样本过少）时回退 OLS p。
    if robust and df["p_value_nw"].notna().any():
        p_for_fdr = df["p_value_nw"].fillna(df["p_value"]).values
    else:
        p_for_fdr = df["p_value"].values
    df["significant"] = _benjamini_hochberg(p_for_fdr, fdr_q)

    # 默认按 |IR| 降序（业界统一主轴）；|t| 与 |IR| 对固定样本单调等价，可切换。
    sort_col = "ir" if sort_by == "ir" else "t_stat"
    df = df.reindex(df[sort_col].abs().sort_values(ascending=False).index).reset_index(drop=True)

    # 细节指标（IC 衰减 / 换手代理）仅对 top-N 计算，控制耗时
    df["ic_decay5"] = np.nan
    df["ic_decay10"] = np.nan
    df["autocorr"] = np.nan
    max_lag = max(decay_lags) if decay_lags else 10
    for idx in df.head(detail_n).index:
        name = df.at[idx, "name"]
        c = cand_map.get(name)
        if c is None:
            continue
        try:
            fp = c.build(panel)
            if fp is None or fp.empty:
                continue
            with np.errstate(all="ignore"), _suppress_const_warning():
                decay = calc_ic_decay(fp, returns_panel, max_lag=max_lag)
            df.at[idx, "ic_decay5"] = float(decay.get(5, np.nan))
            df.at[idx, "ic_decay10"] = float(decay.get(10, np.nan))
            df.at[idx, "autocorr"] = factor_autocorr(fp)
        except Exception:
            continue
    return df


def dedup_by_formula(candidates: list[Candidate]) -> list[Candidate]:
    """按公式名去重（生成器可能产生重复表达式）。"""
    seen = set()
    out = []
    for c in candidates:
        if c.name in seen:
            continue
        seen.add(c.name)
        out.append(c)
    return out
