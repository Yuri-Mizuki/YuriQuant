"""
遗传规划因子挖掘
================

用 DEAP 的遗传规划在算子空间上**自动搜索因子表达式**。

设计要点
--------
- **单类型 GP**：所有原语都是 ``panel -> panel``（或 ``panel,panel -> panel``），
  时间窗口直接编进算子名（如 ``ts_mean_5``），避免多类型树的复杂度。
- **适应度** = |mean rank IC| / (std IC + eps)，即 |t|-like 统计量（与 |IR| 单调），
  同时奖励"方向稳定"的预测力；含轻量 bloat 惩罚（按树高）。
- **手动遍历表达式树求值**：DEAP 的 gp.compile 会把字符串终端喂给算子，与面板
  语义不符；这里用栈式解释器直接在 DataFrame 上应用算子。
- 原语集来自 factor.operators，与「候选生成 + 批量 IC」流程共用同一算子空间。

与 exhaustive 挖掘的关系：exhaustive 覆盖浅层组合，GP 探索更深、更稀疏的公式结构。
"""
from __future__ import annotations

import operator
import random
from typing import Sequence

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, gp, tools

from factor.operators import (
    CS_OPS, TS_OPS, abs_, add, cs_demean, cs_rank, cs_rank_normalize, cs_zscore,
    cs_normalize, div, exp_, log_, max_, min_, mul, reverse, sign, sqrt_, sub,
    ts_arg_max, ts_arg_min, ts_avedev, ts_corr, ts_decay_linear, ts_delta,
    ts_diff, ts_ema, ts_kurt, ts_max, ts_mean, ts_median, ts_min, ts_rank,
    ts_skew, ts_std, ts_sum, ts_wma,
)
from research.factor_analysis import calc_ic_series, calc_ir


# 窗口已编入算子名时使用的时序单目算子（不含需要布尔输入的 ts_count）
_GP_TS_UNARY = [
    ("ts_mean", ts_mean), ("ts_sum", ts_sum), ("ts_std", ts_std),
    ("ts_max", ts_max), ("ts_min", ts_min), ("ts_arg_max", ts_arg_max),
    ("ts_arg_min", ts_arg_min), ("ts_rank", ts_rank), ("ts_delta", ts_delta),
    ("ts_diff", ts_diff), ("ts_avedev", ts_avedev), ("ts_skew", ts_skew),
    ("ts_kurt", ts_kurt), ("ts_ema", ts_ema), ("ts_wma", ts_wma),
    ("ts_decay_linear", ts_decay_linear), ("ts_median", ts_median),
]
_GP_CS_UNARY = [
    ("cs_rank", cs_rank), ("cs_zscore", cs_zscore), ("cs_demean", cs_demean),
    ("cs_normalize", cs_normalize), ("cs_rank_normalize", cs_rank_normalize),
]
_GP_ELEM_UNARY = [
    ("abs", abs_), ("sign", sign), ("log", log_), ("sqrt", sqrt_),
    ("reverse", reverse), ("exp", exp_),
]
_GP_ELEM_BINARY = [
    ("add", add), ("sub", sub), ("mul", mul), ("div", div),
    ("max", max_), ("min", min_),
]

# 模块级原语映射（name -> (func, arity)），由 build_primitive_set 填充
_PRIM_MAP: dict[str, tuple] = {}


def build_primitive_set(
    features: Sequence[str], windows: Sequence[int]
) -> tuple[gp.PrimitiveSet, dict[str, tuple]]:
    """构建单类型 GP 原语集：特征为终端，算子为原语（时序窗口编入名称）。

    Returns:
        (pset, prim_map)：prim_map = {name: (func, arity)}，供 ``eval_tree`` 使用。
        显式返回映射而非依赖模块级 ``_PRIM_MAP``：早期实现用全局 dict，第二次
        调用 ``build_primitive_set``（如换一组 features）会覆盖它，导致对旧 run
        的 HallOfFame 调用 ``eval_tree`` 时原语名全部查不到、静默返回 None。
    """
    prim_map: dict[str, tuple] = {}
    pset = gp.PrimitiveSet("MAIN", arity=0)

    # 终端：每个特征（值为特征名字符串，求值时映射到 panel[feat]）
    for feat in features:
        pset.addTerminal(feat, name=feat)

    # 元素单目
    for name, fn in _GP_ELEM_UNARY:
        pset.addPrimitive(fn, 1, name=name)
        prim_map[name] = (fn, 1)
    # 元素双目
    for name, fn in _GP_ELEM_BINARY:
        pset.addPrimitive(fn, 2, name=name)
        prim_map[name] = (fn, 2)
    # 截面单目
    for name, fn in _GP_CS_UNARY:
        pset.addPrimitive(fn, 1, name=name)
        prim_map[name] = (fn, 1)
    # 时序单目（窗口编入名称）
    for name, fn in _GP_TS_UNARY:
        for w in windows:
            full = f"{name}_{w}"
            # 默认参数闭包避免晚绑定
            curried = lambda p, _f=fn, _w=w: _f(p, _w)
            pset.addPrimitive(curried, 1, name=full)
            prim_map[full] = (curried, 1)
    # 时序相关：固定窗口的双目算子
    for w in windows:
        full = f"ts_corr_{w}"
        curried = lambda a, b, _w=w: ts_corr(a, b, _w)
        pset.addPrimitive(curried, 2, name=full)
        prim_map[full] = (curried, 2)

    # 同步模块级映射，仅供未传 prim_map 的旧调用方回退使用
    global _PRIM_MAP
    _PRIM_MAP = prim_map
    return pset, prim_map


def eval_tree(expr, panel: dict[str, pd.DataFrame], prim_map: dict | None = None,
              memo: dict | None = None):
    """递归解释器：在面板上求值 DEAP 前缀表达式树。

    prim_map 为 ``build_primitive_set`` 的返回值；不传时回退到模块级 ``_PRIM_MAP``
    （仅为向后兼容，跨 run 复用 hof 时应显式传入，否则全局表可能已被覆盖）。

    **子树缓存（P2，2026-08-03）**：``memo`` 为 run 级共享 dict —— 相同子树的
    面板求值结果只算一次（同一 run 内 panel 固定，求值确定性）。典型 GP 种群
    中 ``ts_mean_5(close)`` 等共享子树被大量个体重复计算，缓存可省 2-5 倍。
    注意：memo 只在同一 panel 下有效，换 panel 须换 memo（由调用方保证）。
    """
    if prim_map is None:
        prim_map = _PRIM_MAP
    # 未知原语（跨 run 复用 hof 但映射被覆盖）→ 与旧栈式行为一致：整体返回 None
    if any(isinstance(n, gp.Primitive) and n.name not in prim_map for n in expr):
        return None
    memo = memo if memo is not None else {}

    def _eval(i: int):
        node = expr[i]
        if not isinstance(node, gp.Primitive):  # Terminal
            val = panel.get(node.value) if isinstance(node.value, str) else node.value
            return val, i + 1
        args = []
        pos = i + 1
        for _ in range(node.arity):
            a, pos = _eval(pos)
            args.append(a)
        if any(a is None for a in args):
            return None, pos
        key = tuple((n.name, n.arity) if isinstance(n, gp.Primitive)
                    else ("t", str(n.value)) for n in expr[i:pos])
        if key in memo:
            return memo[key], pos
        try:
            val = prim_map[node.name][0](*args)
        except Exception:
            val = None
        memo[key] = val
        return val, pos

    val, _ = _eval(0)
    return val


def _monthly_forward_returns(returns_panel: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """未来 ``window`` 个交易日的累计收益面板（多 horizon 适应度用）。

    ``returns_panel`` 约定为已前移一期的未来收益（returns[t]=r_{t→t+1}），
    则未来 20 日累计 ≈ Σ_{s=t}^{t+19} returns[s]（小收益线性近似），
    位置 t 与日频 IC 同口径对齐，无未来函数。
    """
    return returns_panel.rolling(window, min_periods=1).sum().shift(-(window - 1))


def _mut_window_jitter_or_uniform(individual, expr, pset, windows, prim_map,
                                  jitter_p: float = 0.7):
    """窗口 jitter 变异（P1，轻量窗口搜索，无需双类型树）。

    以概率 ``jitter_p`` 走窗口扰动：随机选树中一个窗口原语（ts_X_w），把窗口
    替换为邻近窗口（±delta，从预置窗口集或 ±delta 扰动生成）；否则回退
    DEAP ``gp.mutUniform`` 子树替换。窗口扰动使窗口参数参与搜索，又不破坏
    单类型树结构（窗口仍编在算子名里）。
    """
    # 从预置窗口扩展出扰动候选（±delta，确保与 pset 内已有原语名一致）
    all_w = sorted(set(windows))
    candidates: dict[str, list[str]] = {}
    for w in all_w:
        neigh = set()
        for dw in (1, 2, 3, 5, -1, -2, -3, -5, 10, -10):
            nw = w + dw
            if nw in all_w:
                neigh.add(nw)
        if not neigh:
            continue
        candidates[w] = [f"{name}_{nw}" for name, _ in _GP_TS_UNARY + [("ts_corr", None)] for nw in neigh]

    if jitter_p <= 0 or not candidates:
        return gp.mutUniform(individual, expr=expr, pset=pset)

    idxs = [i for i, node in enumerate(individual)
            if isinstance(node, gp.Primitive) and any(node.name.startswith(f"{n}_") for n, _ in _GP_TS_UNARY)]
    if random.random() > jitter_p or not idxs:
        return gp.mutUniform(individual, expr=expr, pset=pset)

    i = random.choice(idxs)
    node = individual[i]
    base = node.name.rsplit("_", 1)[0]
    w = int(node.name.rsplit("_", 1)[1])
    alt_names = candidates.get(w, [])
    alt_names = [n for n in alt_names if n in prim_map and n.startswith(f"{base}_")]
    if not alt_names:
        return gp.mutUniform(individual, expr=expr, pset=pset)
    new_name = random.choice(alt_names)
    individual[i] = pset.mapping[new_name]   # 直接取 pset 里注册的原语对象
    # 树结构变化需重置 fitness（DEAP 约定：变异函数须返回 tuple）
    del individual.fitness.values
    return (individual,)


def _ic_stats(ic_series) -> tuple[float, float, int]:
    """(mean, std, n)；空序列返回 (nan, nan, 0)。"""
    ic = ic_series.dropna()
    n = len(ic)
    if n == 0:
        return float("nan"), float("nan"), 0
    return float(ic.mean()), float(ic.std()), n


def _fitness(individual, panel, returns_fit, min_obs, parsimony, prim_map,
             returns_month_fit=None, monthly_weight: float = 0.0,
             lib_ranked: list | None = None, library_penalty: float = 0.0,
             memo: dict | None = None, sample_step: int = 1):
    """适应度（P0 增强，2026-08-03）：

    = |mean rank IC| / (std IC + eps) × (日频与月频加权) - library_penalty×库相关
      - parsimony × height

    - **样本外验证**：``returns_fit`` 为 train 段（``train_frac`` 切分），进化只
      看 train 段 IC，防止全样本选择压力直接对着搜索过的数据过拟合。
    - **多 horizon**：``monthly_weight>0`` 时叠加月频 IC（未来 20 日累计收益），
      缓解"日频 IC 与月频回测符号反转"的 horizon 错配。
    - **与因子库去相关**：``library_penalty>0`` 时按与库内因子 rank 面板的最大
      截面相关均值扣分，防止重复发明库内已有因子。
    - **子树缓存**：``memo`` 跨个体共享（run 级），eval_tree 跳过重复子树。
    - **粗筛**：``sample_step>1`` 时对 IC 计算做时间子采样（进化期粗筛，hof
      汇总仍全样本精算）。
    """
    try:
        fp = eval_tree(individual, panel, prim_map, memo=memo)
    except Exception:
        return (0.0,)
    if fp is None or not hasattr(fp, "shape"):
        return (0.0,)
    if sample_step > 1:
        fp = fp.iloc[::sample_step]
        rfit = returns_fit.iloc[::sample_step]
    else:
        rfit = returns_fit
    try:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="An input array is constant")
                ic_d = calc_ic_series(fp, rfit, method="spearman")
    except Exception:
        return (0.0,)
    m_d, s_d, n_d = _ic_stats(ic_d)
    if n_d < min_obs or not np.isfinite(m_d) or not np.isfinite(s_d) or s_d < 1e-8:
        return (0.0,)
    score_d = abs(m_d) / (s_d + 1e-9)

    score = score_d
    if monthly_weight > 0 and returns_month_fit is not None:
        try:
            with np.errstate(all="ignore"):
                ic_m = calc_ic_series(fp, returns_month_fit.iloc[::sample_step] if sample_step > 1 else returns_month_fit,
                                      method="spearman")
            m_m, s_m, n_m = _ic_stats(ic_m)
            if n_m >= min_obs and np.isfinite(m_m) and np.isfinite(s_m) and s_m > 1e-8:
                score_m = abs(m_m) / (s_m + 1e-9)
                # P0-①（2026-08-04）：月频与日频**符号必须一致**才融合，否则扣分。
                # 旧实现各自取绝对值再加权：日频正 IC + 月频负 IC 的"horizon 符号反转"
                # 坏公式反而被奖励。现在符号反转时按差异扣分，倒逼进化找两个 horizon
                # 都同向的公式。
                if m_d * m_m > 0:
                    score = (score_d + monthly_weight * score_m) / (1.0 + monthly_weight)
                else:
                    score = score_d - monthly_weight * score_m
        except Exception:
            pass

    if library_penalty > 0 and lib_ranked:
        try:
            fpr = fp.rank(axis=1)
            corrs = [float(fpr.corrwith(lib, axis=1).mean()) for lib in lib_ranked if lib is not None]
            if corrs:
                score -= library_penalty * max(corrs)
        except Exception:
            pass

    score = min(score, 50.0)  # 适应度上界，防爆炸
    score -= parsimony * individual.height
    return (float(score),)


# ===========================================================================
# P2-2：GP 种群并行评估 worker（进程池；模块级函数以便 pickle）
# ===========================================================================
_GP_CTX: dict = {}


def _init_gp_worker(panel, returns_fit, returns_month_fit, lib_ranked,
                    features, windows, min_obs, parsimony, monthly_weight,
                    library_penalty, memo, sample_step):
    """每个 worker 进程初始化：重建 prim_map（模块级纯函数，可 pickle）+
    创建 DEAP creator 类（individual 反序列化依赖）+ 持有面板。"""
    _ensure_creator()
    _GP_CTX["panel"] = panel
    _GP_CTX["returns_fit"] = returns_fit
    _GP_CTX["returns_month_fit"] = returns_month_fit
    _GP_CTX["lib_ranked"] = lib_ranked
    _GP_CTX["prim_map"] = build_primitive_set(features, windows)[1]
    _GP_CTX["min_obs"] = min_obs
    _GP_CTX["parsimony"] = parsimony
    _GP_CTX["monthly_weight"] = monthly_weight
    _GP_CTX["library_penalty"] = library_penalty
    _GP_CTX["memo"] = memo
    _GP_CTX["sample_step"] = sample_step


def _gp_eval_worker(ind):
    return _fitness(ind, _GP_CTX["panel"], _GP_CTX["returns_fit"], _GP_CTX["min_obs"],
                    _GP_CTX["parsimony"], _GP_CTX["prim_map"],
                    returns_month_fit=_GP_CTX["returns_month_fit"],
                    monthly_weight=_GP_CTX["monthly_weight"],
                    lib_ranked=_GP_CTX["lib_ranked"],
                    library_penalty=_GP_CTX["library_penalty"],
                    memo=_GP_CTX["memo"], sample_step=_GP_CTX["sample_step"])


def _seg_ic_stats(fp, seg) -> tuple[float, float]:
    """fp 在 seg 段（收益面板）上的 (IC mean, t)。seg 为空返回 (nan, nan)。"""
    if seg is None or len(seg) == 0:
        return float("nan"), float("nan")
    ic = calc_ic_series(fp, seg, method="spearman").dropna()
    if len(ic) == 0:
        return float("nan"), float("nan")
    m, s = float(ic.mean()), float(ic.std())
    t = m / (s / np.sqrt(len(ic))) if s > 0 else 0.0
    return m, t


def _dedup_hof_by_correlation(df: pd.DataFrame, panel, feats, threshold: float = 0.9,
                              returns_fit: pd.DataFrame | None = None) -> pd.DataFrame:
    """按公式面板相关性贪心去重（P1，2026-08-03；P0-④ 2026-08-04）。

    按当前排序（|t| 降序）遍历，与已保留代表的最大截面相关均值 > 阈值即丢弃，
    保证 hof 不输出一堆同一因子的变体。threshold<=0 关闭。

    **P0-④（2026-08-04）**：相关性只在 train 段（``returns_fit`` 对应时间段）
    计算——旧实现用全样本面板，train 与 OOS 相关结构不同时可能误删 OOS 上
    独立的因子。
    """
    if threshold <= 0 or len(df) <= 1:
        return df
    from factor.formula import formula_builder
    reps: list = []
    keep: list[str] = []
    for _, row in df.iterrows():
        try:
            fp = formula_builder(row["formula"], features=feats)(panel)
            if returns_fit is not None and len(returns_fit) < len(fp):
                fp = fp.iloc[:len(returns_fit)]
            fpr = fp.rank(axis=1)
        except Exception:
            keep.append(row["formula"])   # 无法重建（异常公式）宽容保留
            continue
        if reps:
            corrs = [float(fpr.corrwith(r, axis=1).mean()) for r in reps]
            if corrs and max(c for c in corrs if c == c) > threshold:
                continue
        reps.append(fpr)
        keep.append(row["formula"])
    return df[df["formula"].isin(keep)].reset_index(drop=True)


# creator 全局对象只创建一次（避免重复 import 报错）
def _ensure_creator():
    if not hasattr(creator, "FitnessMaxGP"):
        creator.create("FitnessMaxGP", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualGP"):
        creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessMaxGP)


def run_gp_mining(
    panel: dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    features: Sequence[str] | None = None,
    windows: Sequence[int] = (5, 10, 20),
    population: int = 200,
    generations: int = 20,
    cx_prob: float = 0.5,
    mut_prob: float = 0.2,
    min_depth: int = 2,
    max_depth: int = 5,
    tournament: int = 5,
    min_obs: int = 20,
    parsimony: float = 0.002,
    hall_size: int = 20,
    patience: int = 6,
    improve_eps: float = 1e-6,
    train_frac: float = 0.7,
    monthly_weight: float = 0.5,
    library_panels: dict | None = None,
    library_penalty: float = 0.0,
    window_jitter: bool = True,
    dedup_corr: float = 0.9,
    n_jobs: int = 1,
    sample_step: int = 1,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, object]:
    """运行遗传规划因子挖掘。

    **统计严谨性增强（2026-08-03，P0）**：
    - ``train_frac``：进化只用前 ``train_frac`` 时间段算 IC（样本外验证），
      hof 汇总同时报告 train / OOS 段 IC，直观暴露过拟合程度。
    - ``monthly_weight``：多 horizon 融合（日频 IC + 未来 20 日累计收益 IC），
      缓解日频/月频 horizon 错配。
    - ``library_panels`` + ``library_penalty``：与因子库面板去相关惩罚，防止
      重复发明库内已有因子（库面板 dict{name: date×code}，主进程自动 rank）。
    - ``window_jitter``：窗口 jitter 变异（P1），把树中 ts_X_w 原语随机替换为
      邻近窗口，让窗口参与搜索而无需双类型树。

    **早停（2026-08-03）**：连续 ``patience`` 代 hof best 无提升即提前终止，
    ``hof.generations_run`` 记录实际代数。

    Returns:
        (results_df, hall_of_fame)：results_df 列：
        formula/ic_mean(全样本)/ic_train/ic_oos/ic_std/ir/t_stat/t_oos/n/height。
    """
    feats = list(features) if features else list(panel.keys())
    pset, prim_map = build_primitive_set(feats, windows)
    _ensure_creator()

    # ---- P0：样本外切分 + 多 horizon + 库去相关预处理 ----
    n_total = len(returns_panel)
    n_train = max(min_obs + 5, int(n_total * train_frac)) if train_frac < 1.0 else n_total
    n_train = min(n_train, n_total)
    returns_fit = returns_panel.iloc[:n_train]
    returns_oos = returns_panel.iloc[n_train:]
    returns_month_fit = _monthly_forward_returns(returns_fit) if monthly_weight > 0 else None
    lib_ranked: list | None = None
    if library_penalty > 0 and library_panels:
        lib_ranked = []
        for nm, lp in library_panels.items():
            try:
                lr = lp.reindex(index=panel[feats[0]].index, columns=panel[feats[0]].columns)
                lib_ranked.append(lr.rank(axis=1))
            except Exception:
                lib_ranked.append(None)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.IndividualGP, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    memo: dict = {}   # run 级子树缓存（P2-1）
    _ex = None
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        _ex = ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_gp_worker,
            initargs=(panel, returns_fit, returns_month_fit, lib_ranked, feats,
                      list(windows), min_obs, parsimony, monthly_weight,
                      library_penalty, memo, sample_step))
        toolbox.register("map", _ex.map)
        toolbox.register("evaluate", _gp_eval_worker)
    else:
        toolbox.register("evaluate", _fitness, panel=panel, returns_fit=returns_fit,
                         min_obs=min_obs, parsimony=parsimony, prim_map=prim_map,
                         returns_month_fit=returns_month_fit, monthly_weight=monthly_weight,
                         lib_ranked=lib_ranked, library_penalty=library_penalty,
                         memo=memo, sample_step=sample_step)
    toolbox.register("select", tools.selTournament, tournsize=tournament)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    if window_jitter:
        toolbox.register("mutate", _mut_window_jitter_or_uniform, expr=toolbox.expr_mut,
                         pset=pset, windows=windows, prim_map=prim_map)
    else:
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth + 2))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth + 2))

    random.seed(seed)
    np.random.seed(seed)
    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(maxsize=hall_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda v: float(np.mean(v)) if v else 0.0)
    stats.register("max", lambda v: float(np.max(v)) if v else 0.0)

    # ---- 手动进化循环（支持早停；eaSimple 无此能力）----
    try:
        # 初始种群评估
        fits = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit
        hof.update(pop)
        record = stats.compile(pop)
        best_ever = float(hof[0].fitness.values[0])
        stall = 0
        gen_done = 0
        if verbose:
            print(f"gen {0:3d}: avg={record['avg']:.4f} max={record['max']:.4f} best={best_ever:.4f}")

        for gen in range(1, generations + 1):
            offspring = algorithms.varOr(pop, toolbox, len(pop), cx_prob, mut_prob)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit
            hof.update(offspring)
            pop[:] = toolbox.select(offspring, len(pop))
            record = stats.compile(pop)
            gen_done = gen
            cur_best = float(hof[0].fitness.values[0])
            if verbose:
                print(f"gen {gen:3d}: avg={record['avg']:.4f} max={record['max']:.4f} best={cur_best:.4f}")
            if cur_best - best_ever > improve_eps:
                best_ever = cur_best
                stall = 0
            else:
                stall += 1
                if patience and stall >= patience:
                    if verbose:
                        print(f"  [early stop] gen {gen}: best 连续 {patience} 代无提升（> {improve_eps}）")
                    break
        setattr(hof, "generations_run", gen_done)
    finally:
        if _ex is not None:
            _ex.shutdown()

    # 汇总 HallOfFame（全样本 IC + train/OOS 分段报告）
    rows = []
    for ind in hof:
        try:
            fp = eval_tree(ind, panel, prim_map)
            ic = calc_ic_series(fp, returns_panel, method="spearman").dropna()
            n = len(ic)
            m, s = float(ic.mean()), float(ic.std())
            ir = calc_ir(ic)
            t = m / (s / np.sqrt(n)) if s > 0 else 0.0
            ic_train, t_train = _seg_ic_stats(fp, returns_fit)
            ic_oos, t_oos = _seg_ic_stats(fp, returns_oos)
            rows.append({
                "formula": str(ind),
                "ic_mean": m,
                "ic_train": ic_train,
                "ic_oos": ic_oos,
                "ic_std": s,
                "ir": ir,
                "t_stat": t,
                "t_train": t_train,
                "t_oos": t_oos,
                "n": n,
                "height": ind.height,
            })
        except Exception:
            rows.append({"formula": str(ind), "ic_mean": 0.0, "ic_train": float("nan"),
                         "ic_oos": float("nan"), "ic_std": 0.0, "ir": 0.0,
                         "t_stat": 0.0, "t_train": float("nan"), "t_oos": float("nan"),
                         "n": 0, "height": ind.height})
    df = pd.DataFrame(rows)
    if not df.empty:
        # P0-④（2026-08-04）：排序与去重都基于 **train 段** t（t_train），
        # 全样本 t_stat 仅作报告——旧实现按全样本 |t| 排序，把 OOS 信息混入选择。
        sort_col = "t_train" if train_frac is not None and 0.0 < train_frac < 1.0 else "t_stat"
        df = df.reindex(df[sort_col].abs().sort_values(ascending=False).index).reset_index(drop=True)
        df = _dedup_hof_by_correlation(df, panel, feats, threshold=dedup_corr,
                                       returns_fit=returns_fit if train_frac < 1.0 else None)
    return df, hof


# ===========================================================================
# P1-7：GP 播种局部搜索（memetic）—— hof 公式近邻批量检验
# ===========================================================================
def _node_to_gp_string(node) -> str:
    """把 formula.parse_formula 的 AST 节点序列化为 GP 风格公式字符串。"""
    kind = node[0]
    if kind == "feat":
        return node[1]
    if kind == "const":
        return str(node[1])
    spec, children, win_name = node[1], node[2], node[3]
    name = spec.name
    if win_name:
        name = f"{name}_{win_name[0]}"
    return f"{name}({','.join(_node_to_gp_string(c) for c in children)})"


_NEIGHBOR_OP_NAMES = {
    "ts": [n for n, _ in _GP_TS_UNARY],
    "element1": [n for n, _ in _GP_ELEM_UNARY],
    "element2": [n for n, _ in _GP_ELEM_BINARY],
    "cs": [n for n, _ in _GP_CS_UNARY],
}


def _neighbor_ops(spec, reg):
    """同类别同 arity 的算子候选（供算子替换扰动）。"""
    if spec.kind == "ts" and spec.n_window == 1 and spec.arity == 1:
        return [reg[n] for n in _NEIGHBOR_OP_NAMES["ts"] if n != spec.name]
    if spec.kind == "element":
        if spec.arity == 1:
            return [reg[n] for n in _NEIGHBOR_OP_NAMES["element1"] if n != spec.name]
        if spec.arity == 2:
            return [reg[n] for n in _NEIGHBOR_OP_NAMES["element2"] if n != spec.name]
    if spec.kind == "cs" and spec.arity == 1:
        return [reg[n] for n in _NEIGHBOR_OP_NAMES["cs"] if n != spec.name]
    return []


def generate_neighbor_formulas(formula: str, n_per: int = 10, seed: int = 0,
                               max_depth: int = 2) -> list[str]:
    """生成公式的**近邻**（窗口扰动 + 同 arity 算子替换），用于 memetic 局部搜索。

    Args:
        formula: GP 或 exhaustive 风格的公式字符串。
        n_per: 随机采样的近邻数量上限。
        max_depth: 递归扰动深度（只扰动前几层，控制数量）。
    Returns:
        近邻公式列表（不含原公式）。
    """
    from factor.formula import parse_formula
    from factor.operators import op_registry
    reg = op_registry()
    node = parse_formula(formula, features=None, registry=reg)  # 宽容 features
    out: set[str] = set()

    def variants(n, depth):
        if depth > max_depth or n[0] != "call":
            return
        spec, children, win_name = n[1], n[2], n[3]
        # 窗口扰动
        if spec.n_window >= 1:
            for dw in (1, 2, 3, 5, -1, -2, -3, -5):
                if win_name:
                    w = win_name[0] + dw
                    if w >= 2:
                        out.add(_node_to_gp_string(("call", spec, children, (w,))))
                else:
                    for ci in range(len(children)):
                        if children[ci][0] == "const":
                            w = children[ci][1] + dw
                            if w >= 2:
                                ch = list(children)
                                ch[ci] = ("const", w)
                                out.add(_node_to_gp_string(("call", spec, ch, win_name)))
        # 同 arity 算子替换
        for alt in _neighbor_ops(spec, reg):
            out.add(_node_to_gp_string(("call", alt, children, win_name)))
        # 递归子节点
        for c in children:
            variants(c, depth + 1)

    variants(node, 0)
    out.discard(formula)
    lst = sorted(out)
    rng = random.Random(seed)
    rng.shuffle(lst)
    return lst[:n_per]


def refine_gp_neighbors(
    df: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    n_per: int = 10,
    n_jobs: int = 1,
    min_obs: int = 20,
    train_frac: float = 0.7,
    verbose: bool = True,
) -> pd.DataFrame:
    """GP 播种局部搜索：对 hof 公式生成近邻，走 evaluate_candidates 批量 IC 检验。

    **memetic 思路**：GP 负责探索粗结构，近邻批量检验负责在粗结构附近精修
    （窗口微调 / 算子替换），并把近邻并入候选空间统一排名。近邻公式天然是
    字符串，直接进 ``evaluate_candidates`` 并行池（formula_builder 重建）。

    **P0-②（2026-08-04）**：近邻择优**只用 train 段**（``train_frac`` 前段）做
    IC 检验，禁止把 OOS 段信息混入局部搜索选择——旧实现用全样本 returns 评估
    近邻，等价于在进化结束后又对 OOS 过拟合一轮。

    Args:
        df: run_gp_mining 的结果（须含 formula 列）。
        n_per: 每个公式生成的近邻数。
        train_frac: 近邻评估只取前 ``train_frac`` 时间段的收益（<1 时启用；
            >=1 视为回退到全样本，兼容旧调用）。
    Returns:
        合并表（原始 GP + 近邻），列 name/ic_mean/ic_std/ir/t_stat/n/source，
        按 |t| 降序。
    """
    from factor.mining import Candidate, evaluate_candidates
    from factor.formula import formula_builder

    feats = list(panel.keys())
    seen: set[str] = set()
    formulas: list[str] = []
    for _, row in df.iterrows():
        for f in generate_neighbor_formulas(row["formula"], n_per=n_per):
            if f not in seen:
                seen.add(f)
                formulas.append(f)
    if not formulas:
        return df[["formula", "ic_mean", "ir", "t_stat", "n"]].rename(
            columns={"formula": "name"}).copy()

    # 近邻择优只用 train 段（P0-②）
    if train_frac is not None and 0.0 < train_frac < 1.0:
        n_train = max(min_obs + 5, int(len(returns_panel) * train_frac))
        n_train = min(n_train, len(returns_panel))
        returns_fit = returns_panel.iloc[:n_train]
    else:
        returns_fit = returns_panel

    cands = [Candidate(name=f, build=formula_builder(f, features=feats)) for f in formulas]
    nb = evaluate_candidates(cands, panel, returns_fit, n_jobs=n_jobs, min_obs=min_obs,
                             detail_n=0)
    nb["source"] = "neighbor"
    base = df[["formula", "ic_mean", "ir", "t_stat", "n"]].rename(
        columns={"formula": "name"})
    base["source"] = "gp"
    merged = pd.concat(
        [base, nb[["name", "ic_mean", "ic_std", "ir", "t_stat", "n", "source"]]],
        ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.reindex(
        merged["t_stat"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    if verbose:
        print(f"  ...memetic 局部搜索: {len(formulas)} 个近邻（train 段择优），合并后 {len(merged)} 条")
    return merged


# ===========================================================================
# P1-4：NSGA-II 多目标进化（目标1 IC 强度，目标2 换手/稳定性）
# ===========================================================================
def _ensure_creator_multi():
    if not hasattr(creator, "FitnessMultiGP"):
        creator.create("FitnessMultiGP", base.Fitness, weights=(1.0, 1.0))
    if not hasattr(creator, "IndividualGPNSGA"):
        creator.create("IndividualGPNSGA", gp.PrimitiveTree, fitness=creator.FitnessMultiGP)


def _fitness_multi(individual, panel, returns_fit, min_obs, parsimony, prim_map,
                   returns_month_fit=None, monthly_weight=0.0):
    """双目标适应度（NSGA-II）：(f1, f2)。

    f1 = 日频 rank IC 强度（|t|-like，train 段，可选叠加月频权重）；
    f2 = 截面排名自相关（换手率代理，越高 = 排序越稳、交易成本越低）。
    """
    from research.factor_analysis import factor_autocorr
    try:
        fp = eval_tree(individual, panel, prim_map)
    except Exception:
        return (0.0, 0.0)
    if fp is None or not hasattr(fp, "shape"):
        return (0.0, 0.0)
    try:
        with np.errstate(all="ignore"):
            ic_d = calc_ic_series(fp, returns_fit, method="spearman")
        m_d, s_d, n_d = _ic_stats(ic_d)
        if n_d < min_obs or not np.isfinite(m_d) or not np.isfinite(s_d) or s_d < 1e-8:
            return (0.0, 0.0)
        f1 = abs(m_d) / (s_d + 1e-9)
        if monthly_weight > 0 and returns_month_fit is not None:
            ic_m = calc_ic_series(fp, returns_month_fit, method="spearman")
            m_m, s_m, n_m = _ic_stats(ic_m)
            if n_m >= min_obs and np.isfinite(m_m) and np.isfinite(s_m) and s_m > 1e-8:
                f1 = (f1 + monthly_weight * (abs(m_m) / (s_m + 1e-9))) / (1.0 + monthly_weight)
    except Exception:
        return (0.0, 0.0)
    try:
        f2 = float(factor_autocorr(fp, max_lag=1))
    except Exception:
        f2 = 0.0
    f1 = min(f1, 50.0) - parsimony * individual.height
    return (float(f1), float(f2))


def run_gp_nsga2(
    panel: dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    features: Sequence[str] | None = None,
    windows: Sequence[int] = (5, 10, 20),
    population: int = 200,
    generations: int = 20,
    cx_prob: float = 0.5,
    mut_prob: float = 0.2,
    min_depth: int = 2,
    max_depth: int = 5,
    min_obs: int = 20,
    parsimony: float = 0.002,
    train_frac: float = 0.7,
    monthly_weight: float = 0.5,
    patience: int = 6,
    improve_eps: float = 1e-6,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, object]:
    """**NSGA-II 多目标**遗传规划（P1，2026-08-03）。

    双目标 Pareto 前沿：目标1 = 日频 IC 强度（可叠月频），目标2 = 截面排名
    自相关（换手代理）。输出前沿个体而非单目标 hof，让用户按"IC vs 换手"
    取舍 —— 单目标 |t| 会把整个种群推向"高 IC 高换手"角落。

    Returns:
        (results_df, pareto_front)：results_df 列 formula/ic_mean/ic_train/ic_oos/
        t_stat/f1/f2/front，按 f1 降序。
    """
    feats = list(features) if features else list(panel.keys())
    pset, prim_map = build_primitive_set(feats, windows)
    _ensure_creator_multi()

    n_total = len(returns_panel)
    n_train = max(min_obs + 5, int(n_total * train_frac)) if train_frac < 1.0 else n_total
    n_train = min(n_train, n_total)
    returns_fit = returns_panel.iloc[:n_train]
    returns_oos = returns_panel.iloc[n_train:]
    returns_month_fit = _monthly_forward_returns(returns_fit) if monthly_weight > 0 else None

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
    toolbox.register("individual", tools.initIterate, creator.IndividualGPNSGA, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", _fitness_multi, panel=panel, returns_fit=returns_fit,
                     min_obs=min_obs, parsimony=parsimony, prim_map=prim_map,
                     returns_month_fit=returns_month_fit, monthly_weight=monthly_weight)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", _mut_window_jitter_or_uniform, expr=toolbox.expr_mut,
                     pset=pset, windows=windows, prim_map=prim_map)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth + 2))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth + 2))

    random.seed(seed)
    np.random.seed(seed)
    pop = toolbox.population(n=population)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", lambda v: float(np.mean(v)) if v else 0.0)
    stats.register("max", lambda v: float(np.max(v)) if v else 0.0)

    fits = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hof.update(pop)
    best_ever = max(ind.fitness.values[0] for ind in pop)
    stall = 0
    gen_done = 0
    if verbose:
        record = stats.compile(pop)
        print(f"gen {0:3d}: avg={record['avg']:.4f} max_f1={record['max']:.4f} front={len(hof)}")

    for gen in range(1, generations + 1):
        offspring = algorithms.varOr(pop, toolbox, len(pop), cx_prob, mut_prob)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop[:] = toolbox.select(pop + offspring, len(pop))
        hof.update(pop)
        gen_done = gen
        record = stats.compile(pop)
        cur_best = max(ind.fitness.values[0] for ind in pop)
        if verbose:
            print(f"gen {gen:3d}: avg={record['avg']:.4f} max_f1={record['max']:.4f} front={len(hof)}")
        if cur_best - best_ever > improve_eps:
            best_ever = cur_best
            stall = 0
        else:
            stall += 1
            if patience and stall >= patience:
                if verbose:
                    print(f"  [early stop] gen {gen}: f1 连续 {patience} 代无提升")
                break
    setattr(hof, "generations_run", gen_done)

    rows = []
    for fi, ind in enumerate(hof):
        try:
            fp = eval_tree(ind, panel, prim_map)
            ic = calc_ic_series(fp, returns_panel, method="spearman").dropna()
            n = len(ic)
            m, s = float(ic.mean()), float(ic.std())
            t = m / (s / np.sqrt(n)) if s > 0 else 0.0
            ic_train, _ = _seg_ic_stats(fp, returns_fit)
            ic_oos, _ = _seg_ic_stats(fp, returns_oos)
            rows.append({
                "formula": str(ind), "ic_mean": m, "ic_train": ic_train, "ic_oos": ic_oos,
                "t_stat": t, "f1": float(ind.fitness.values[0]),
                "f2": float(ind.fitness.values[1]), "front": fi, "n": n,
            })
        except Exception:
            rows.append({"formula": str(ind), "ic_mean": 0.0, "ic_train": float("nan"),
                         "ic_oos": float("nan"), "t_stat": 0.0,
                         "f1": float(ind.fitness.values[0]), "f2": float(ind.fitness.values[1]),
                         "front": fi, "n": 0})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.reindex(df["f1"].sort_values(ascending=False).index).reset_index(drop=True)
    return df, hof
