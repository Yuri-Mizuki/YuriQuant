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
    CS_OPS, TECH_OPS, TS_OPS, abs_, add, adx, aroonosc, boll_pctb, cs_demean,
    cs_rank, cs_rank_normalize, cs_zscore, cs_normalize, cs_scale_abs, div, exp_,
    ht_dcphase, inv, kama, log_, max_, min_, mul, obv, rank_div, rank_sub,
    reverse, rsi, sign, sigmoid, sqrt_, sub, ts_arg_max, ts_arg_min,
    ts_avedev, ts_corr, ts_cov, ts_decay_linear, ts_delay, ts_delta, ts_diff,
    ts_ema, ts_kurt, ts_max, ts_mean, ts_median, ts_min, ts_product, ts_rank,
    ts_skew, ts_std, ts_sum, ts_wma, ts_zscore,
)
from research.factor_analysis import calc_ic_series, calc_ir


# 窗口已编入算子名时使用的时序单目算子（不含需要布尔输入的 ts_count）
# 对照华泰 gplearn 函数集：delay→ts_delay / delta→ts_delta / stddev→ts_std /
# product→ts_product / zscore→ts_zscore / decay_linear→ts_decay_linear 均已含。
_GP_TS_UNARY = [
    ("ts_mean", ts_mean), ("ts_sum", ts_sum), ("ts_std", ts_std),
    ("ts_max", ts_max), ("ts_min", ts_min), ("ts_arg_max", ts_arg_max),
    ("ts_arg_min", ts_arg_min), ("ts_rank", ts_rank), ("ts_delta", ts_delta),
    ("ts_diff", ts_diff), ("ts_avedev", ts_avedev), ("ts_skew", ts_skew),
    ("ts_kurt", ts_kurt), ("ts_ema", ts_ema), ("ts_wma", ts_wma),
    ("ts_decay_linear", ts_decay_linear), ("ts_median", ts_median),
    ("ts_delay", ts_delay), ("ts_product", ts_product), ("ts_zscore", ts_zscore),
    # 技术指标（2026-08-12，参考华泰报告26）：单目+窗口，窗口编名如 kama_10
    ("kama", kama), ("rsi", rsi), ("boll_pctb", boll_pctb),
]
_GP_CS_UNARY = [
    ("cs_rank", cs_rank), ("cs_zscore", cs_zscore), ("cs_demean", cs_demean),
    ("cs_normalize", cs_normalize), ("cs_rank_normalize", cs_rank_normalize),
    ("scale", cs_scale_abs),  # 华泰 gplearn 函数集 scale(X, a=1)
]
_GP_ELEM_UNARY = [
    ("abs", abs_), ("sign", sign), ("log", log_), ("sqrt", sqrt_),
    ("reverse", reverse), ("exp", exp_), ("inv", inv), ("sigmoid", sigmoid),
]
_GP_ELEM_BINARY = [
    ("add", add), ("sub", sub), ("mul", mul), ("div", div),
    ("max", max_), ("min", min_), ("rank_sub", rank_sub), ("rank_div", rank_div),
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
        full = f"ts_cov_{w}"   # 华泰 gplearn 函数集 covariance(X, Y, d)
        curried = lambda a, b, _w=w: ts_cov(a, b, _w)
        pset.addPrimitive(curried, 2, name=full)
        prim_map[full] = (curried, 2)

    # 技术指标多输入算子（2026-08-12，参考华泰报告26）：窗口编名 + 无窗口
    for w in windows:
        full = f"aroonosc_{w}"
        curried = lambda a, b, _w=w: aroonosc(a, b, _w)
        pset.addPrimitive(curried, 2, name=full)
        prim_map[full] = (curried, 2)
        full = f"adx_{w}"
        curried = lambda a, b, c, _w=w: adx(a, b, c, _w)
        pset.addPrimitive(curried, 3, name=full)
        prim_map[full] = (curried, 3)
    # 无窗口：ht_dcphase（单目）、obv（close, volume 双目）
    pset.addPrimitive(ht_dcphase, 1, name="ht_dcphase")
    prim_map["ht_dcphase"] = (ht_dcphase, 1)
    pset.addPrimitive(obv, 2, name="obv")
    prim_map["obv"] = (obv, 2)

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


def _mutual_info_series(fp: pd.DataFrame, rets: pd.DataFrame,
                        n_bins: int = 10) -> pd.Series:
    """逐截面离散化互信息序列（华泰报告23 适应度指标）。

    I(X;Y) = ΣΣ p(x,y)·log(p(x,y) / (p(x)p(y)))，可捕捉**任意统计依赖（非线性）**，
    对比 RankIC/F 检验只能捕捉线性关系。

    实现：每截面把因子与收益按排名等分为 ``n_bins`` 桶（rank 后分桶，
    等分位离散化，与华泰直方图口径一致），统计联合分布算 MI。
    因子经 zscore 等**单调变换不改变分桶**，因此 htai 环内预处理可直接复用。

    **n_bins 自适应**：华泰全 A（~3000 股）用 20 桶；本项目 300 股小截面下
    20 桶联合分布过稀（每格 <1 样本），MI 被小样本偏差严重高估并趋于饱和，
    故默认 10 桶（每格 ~3 样本，与华泰的偏差水平接近）。

    Returns:
        MI 序列（date 索引），取全期均值作适应度。
    """
    valid = fp.notna() & rets.notna()
    # 小样本保护：联合分布每格至少 ~3 个样本 → n_bins ≤ sqrt(n/3)。
    # 300 股 → 10 桶（与华泰 3000 股 20 桶的偏差水平接近）；小面板自动降桶。
    n_med = int(valid.sum(axis=1).median())
    n_bins = max(2, min(int(n_bins), int(np.sqrt(n_med / 3.0))))
    fr = fp.where(valid).rank(axis=1, pct=True)
    rr = rets.where(valid).rank(axis=1, pct=True)
    f_bin = (fr * n_bins).clip(0, n_bins - 1)
    r_bin = (rr * n_bins).clip(0, n_bins - 1)
    out = []
    for d in fp.index:
        m = valid.loc[d]
        n = int(m.sum())
        if n < 2:
            out.append(float("nan"))
            continue
        fb = f_bin.loc[d, m].astype(int).values
        rb = r_bin.loc[d, m].astype(int).values
        H = np.zeros((n_bins, n_bins))
        np.add.at(H, (fb, rb), 1)
        p = H / n
        px = p.sum(axis=1, keepdims=True)
        py = p.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = float(np.nansum(p * np.log(p / (px * py) + 1e-12)))
        out.append(mi)
    return pd.Series(out, index=fp.index)


def _top_excess_series(fp: pd.DataFrame, rets: pd.DataFrame,
                       top_frac: float = 0.1) -> tuple[float, float, int]:
    """多头超额收益适应度（华泰报告23）。

    参照分层回测：同时考虑正向/负向因子，取因子 Top 与 Bottom 层组合相对
    全池等权的超额收益，**取较大者**作为适应度（允许负向因子）。

    实现：每截面按因子排序取 Top/Bottom ``top_frac`` 股票池，计算其平均
    未来收益（``rets`` 已前移一期）相对全池等权的超额。

    Returns:
        (top_excess, bottom_excess, n_days)：前两者取 max 作适应度；
        ``n_days`` 为有效截面天数（fitness 调用方须要求 >= min_obs，
        防止深树/财务平滑因子在极少截面上的偶然超额被选为高分）。
    """
    tops: list[float] = []
    bots: list[float] = []
    for d in fp.index:
        f = fp.loc[d]
        r = rets.loc[d]
        m = f.notna() & r.notna()
        if int(m.sum()) < 10:
            continue
        fv = f[m]
        rv = r[m]
        n_top = max(1, int(len(fv) * top_frac))
        idx_top = fv.sort_values(ascending=False).index[:n_top]
        idx_bot = fv.sort_values(ascending=True).index[:n_top]
        base = float(rv.mean())
        tops.append(float(rv.loc[idx_top].mean()) - base)
        bots.append(float(rv.loc[idx_bot].mean()) - base)
    if not tops:
        return float("nan"), float("nan"), 0
    return float(np.mean(tops)), float(np.mean(bots)), len(tops)


def _htai_preprocess(fp: pd.DataFrame, neutral_panels: dict | None = None,
                     mad_n: float = 5.0) -> pd.DataFrame:
    """华泰研报口径的**环内因子预处理**（报告21 适应度计算流程 a/b/c 三步）：

    1. 中位数去极值：median ± 5×median(|X - median|)（不乘 1.4826，与研报公式一致）；
    2. 中性化：行业 + 市值 + 20日动量/20日换手/20日波动（``neutral_panels`` 提供，
       由 neutralize 对全部连续协变量回归取残差）；
    3. 标准化：截面 z-score。

    无 neutral_panels（如 mock）时自动跳过中性化，只做去极值 + 标准化。

    Args:
        fp: 待预处理的因子面板（date×code）。
        neutral_panels: dict，键取 ``size``/``industry``/``mom20``/``turn20``/``vol20``，
            由调用方（mine_factors）预先构建。
        mad_n: 去极值倍数（华泰为 5）。
    """
    from factor.preprocessing import neutralize, standardize_zscore, winsorize_mad
    out = winsorize_mad(fp, n_mad=mad_n, consistency_scale=False)
    if neutral_panels:
        out = neutralize(
            out,
            market_cap_panel=neutral_panels.get("size"),
            industry_panel=neutral_panels.get("industry"),
            extra_covariates={
                k: v for k, v in neutral_panels.items()
                if k not in ("size", "industry") and v is not None
            },
        )
        out = standardize_zscore(out)
    return out


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
        candidates[w] = [f"{name}_{nw}" for name, _ in _GP_TS_UNARY + [("ts_corr", None), ("ts_cov", None)] for nw in neigh]

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
             memo: dict | None = None, sample_step: int = 1,
             htai: bool = False, neutral_panels: dict | None = None,
             fitness_mode: str = "tstat"):
    """适应度（P0 增强，2026-08-03；华泰复现模式 2026-08-10）：

    = |mean rank IC| / (std IC + eps) × (日频与月频加权) - library_penalty×库相关
      - parsimony × height

    - **样本外验证**：``returns_fit`` 为 train 段（``train_frac`` 切分），进化只
      看 train 段 IC，防止全样本选择压力直接对着搜索过的数据过拟合。
    - **多 horizon**：``monthly_weight>0`` 时叠加月频 IC（未来 20 日累计收益），
      缓解"日频 IC 与月频回测符号反转"的 horizon 错配。
    - **华泰复现（htai=True）**：对齐《基于遗传规划的选股因子挖掘》(2019.6)——
      ① 环内预处理：MAD 去极值(±5×MAD)→行业/市值/20日动量/换手/波动中性化→z-score；
      ② 预测目标只用 20 交易日收益（月频 IC）；③ ``fitness_mode="rankic_mean"``
      时适应度 = 全期平均 RankIC（研报图表12 的"适应度"即 RankIC 均值，不除 std）；
      ``fitness_mode="tstat"`` 时仍为 |mean|/std（更强的统计口径，可选）。
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
    if htai:
        try:
            fp = _htai_preprocess(fp, neutral_panels=neutral_panels)
        except Exception:
            return (0.0,)
    if sample_step > 1:
        fp = fp.iloc[::sample_step]
        rfit = returns_fit.iloc[::sample_step]
        rmonth = returns_month_fit.iloc[::sample_step] if returns_month_fit is not None else None
    else:
        rfit = returns_fit
        rmonth = returns_month_fit
    try:
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="An input array is constant")
                ic_d = calc_ic_series(fp, rfit, method="spearman")
    except Exception:
        return (0.0,)

    # ---- 华泰复现：预测目标 = 20 交易日收益（只按月频 IC）----
    if htai:
        if rmonth is None:
            return (0.0,)
        # ---- 报告23 改进1：互信息适应度（挖非线性因子）----
        if fitness_mode == "mutual_info":
            try:
                mi = _mutual_info_series(fp, rmonth).dropna()
            except Exception:
                return (0.0,)
            if len(mi) < min_obs or not np.isfinite(mi.mean()):
                return (0.0,)
            score = float(mi.mean())          # 华泰：全期平均互信息（无符号，非线性）
            score = min(score, 50.0) - parsimony * individual.height
            return (float(score),)
        # ---- 报告23 改进1：多头超额收益适应度 ----
        if fitness_mode == "top_excess":
            try:
                t_ex, b_ex, n_days = _top_excess_series(fp, rmonth, top_frac=0.1)
            except Exception:
                return (0.0,)
            # min_obs 门槛：防止深树/财务平滑因子在极少截面上的偶然超额被当高分
            if n_days < min_obs or not np.isfinite(t_ex) or not np.isfinite(b_ex):
                return (0.0,)
            score = max(t_ex, b_ex)           # 华泰：Top/Bottom 层年化超额取较大者
            score = min(score, 50.0) - parsimony * individual.height
            return (float(score),)
        # ---- 报告21 口径：平均 RankIC / |t| ----
        try:
            with np.errstate(all="ignore"):
                ic_m = calc_ic_series(fp, rmonth, method="spearman")
        except Exception:
            return (0.0,)
        m_m, s_m, n_m = _ic_stats(ic_m)
        if n_m < min_obs or not np.isfinite(m_m) or not np.isfinite(s_m) or s_m < 1e-8:
            return (0.0,)
        if fitness_mode == "rankic_mean":
            score = float(m_m)          # 华泰：全期平均 RankIC（允许为负，符号由树自身进化）
        else:
            score = abs(m_m) / (s_m + 1e-9)
        if library_penalty > 0 and lib_ranked:
            try:
                fpr = fp.rank(axis=1)
                corrs = [float(fpr.corrwith(lib, axis=1).mean()) for lib in lib_ranked if lib is not None]
                if corrs:
                    score -= library_penalty * max(corrs)
            except Exception:
                pass
        score = min(score, 50.0)
        score -= parsimony * individual.height
        return (float(score),)

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
                    library_penalty, memo, sample_step, htai=False,
                    neutral_panels=None, fitness_mode="tstat"):
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
    _GP_CTX["htai"] = htai
    _GP_CTX["neutral_panels"] = neutral_panels
    _GP_CTX["fitness_mode"] = fitness_mode


def _gp_eval_worker(ind):
    return _fitness(ind, _GP_CTX["panel"], _GP_CTX["returns_fit"], _GP_CTX["min_obs"],
                    _GP_CTX["parsimony"], _GP_CTX["prim_map"],
                    returns_month_fit=_GP_CTX["returns_month_fit"],
                    monthly_weight=_GP_CTX["monthly_weight"],
                    lib_ranked=_GP_CTX["lib_ranked"],
                    library_penalty=_GP_CTX["library_penalty"],
                    memo=_GP_CTX["memo"], sample_step=_GP_CTX["sample_step"],
                    htai=_GP_CTX["htai"], neutral_panels=_GP_CTX["neutral_panels"],
                    fitness_mode=_GP_CTX["fitness_mode"])


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
        # 全 NaN 面板（如 sqrt(reverse(close))）：无法比较相关性，宽容保留
        if fpr.notna().sum().sum() == 0:
            keep.append(row["formula"])
            continue
        if reps:
            corrs = [float(fpr.corrwith(r, axis=1).mean()) for r in reps]
            corrs = [c for c in corrs if c == c]
            if corrs and max(corrs) > threshold:
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
    htai: bool = False,
    neutral_panels: dict | None = None,
    fitness_mode: str = "tstat",
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

    **华泰复现模式（htai=True，2026-08-10）**：对齐《基于遗传规划的选股因子挖掘》
    (2019.6) —— 环内 MAD 去极值(±5×MAD)→行业/市值/20日动量/换手/波动中性化→
    z-score（``neutral_panels`` 提供协变量面板）；预测目标 = 20 交易日收益
    （月频 IC，忽略日频）；``fitness_mode="rankic_mean"`` 时适应度 = 全期平均
    RankIC（研报口径），``"tstat"`` 时保持 |mean|/std。函数集已按研报扩充
    （inv/delay/ts_cov/ts_product/ts_zscore/scale/sigmoid/rank_sub/rank_div）。
    建议参数（研报）：population=1000, generations=3, min_depth=1, max_depth=4,
    tournament=20, train_frac=1.0（研报21 全样本；要报告23 的 CV 口径则传 0.8）。

    **早停（2026-08-03）**：连续 ``patience`` 代 hof best 无提升即提前终止，
    ``hof.generations_run`` 记录实际代数。

    Returns:
        (results_df, hall_of_fame)：results_df 列：
        formula/ic_mean(htai 时为月频预处理后 IC，否则全样本日频)/ic_train/ic_oos/
        ic_std/ir/t_stat/t_oos/n/height。
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
    returns_month_fit = _monthly_forward_returns(returns_fit) if (monthly_weight > 0 or htai) else None
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
                      library_penalty, memo, sample_step, htai, neutral_panels,
                      fitness_mode))
        toolbox.register("map", _ex.map)
        toolbox.register("evaluate", _gp_eval_worker)
    else:
        toolbox.register("evaluate", _fitness, panel=panel, returns_fit=returns_fit,
                         min_obs=min_obs, parsimony=parsimony, prim_map=prim_map,
                         returns_month_fit=returns_month_fit, monthly_weight=monthly_weight,
                         lib_ranked=lib_ranked, library_penalty=library_penalty,
                         memo=memo, sample_step=sample_step, htai=htai,
                         neutral_panels=neutral_panels, fitness_mode=fitness_mode)
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
    # htai 口径：IC 目标 = 未来 20 日收益，且因子先做环内预处理（与适应度同口径）
    if htai:
        returns_summary = _monthly_forward_returns(returns_panel)
        returns_oos_seg = (_monthly_forward_returns(returns_oos)
                           if train_frac is not None and 0.0 < train_frac < 1.0 else None)
    else:
        returns_summary = returns_panel
        returns_oos_seg = returns_oos
    rows = []
    for ind in hof:
        try:
            fp = eval_tree(ind, panel, prim_map)
            fp_s = _htai_preprocess(fp, neutral_panels=neutral_panels) if htai else fp
            ic = calc_ic_series(fp_s, returns_summary, method="spearman").dropna()
            n = len(ic)
            m, s = float(ic.mean()), float(ic.std())
            ir = calc_ir(ic)
            t = m / (s / np.sqrt(n)) if s > 0 else 0.0
            ic_train, t_train = _seg_ic_stats(fp_s, returns_month_fit if htai else returns_fit)
            ic_oos, t_oos = _seg_ic_stats(fp_s, returns_oos_seg)
            row = {
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
            }
            # 报告23 指标（htai 口径）：互信息 / 多头超额（Top、Bottom 层）
            if htai:
                try:
                    mi = _mutual_info_series(fp_s, returns_summary).dropna()
                    row["mi_mean"] = float(mi.mean()) if len(mi) else float("nan")
                except Exception:
                    row["mi_mean"] = float("nan")
                try:
                    t_ex, b_ex, _ = _top_excess_series(fp_s, returns_summary, top_frac=0.1)
                    row["top_excess"] = t_ex
                    row["bot_excess"] = b_ex
                except Exception:
                    row["top_excess"] = float("nan")
                    row["bot_excess"] = float("nan")
            rows.append(row)
        except Exception:
            rows.append({"formula": str(ind), "ic_mean": 0.0, "ic_train": float("nan"),
                         "ic_oos": float("nan"), "ic_std": 0.0, "ir": 0.0,
                         "t_stat": 0.0, "t_train": float("nan"), "t_oos": float("nan"),
                         "n": 0, "height": ind.height})
    df = pd.DataFrame(rows)
    if not df.empty:
        # P0-④（2026-08-04）：排序与去重都基于 **train 段** t（t_train），
        # 全样本 t_stat 仅作报告——旧实现按全样本 |t| 排序，把 OOS 信息混入选择。
        if train_frac is not None and 0.0 < train_frac < 1.0:
            sort_col = "t_train"
        elif htai and fitness_mode == "mutual_info" and "mi_mean" in df:
            sort_col = "mi_mean"        # MI 模式的优化目标是互信息
        elif htai and fitness_mode == "top_excess" and "top_excess" in df:
            sort_col = "top_excess"     # 多头超额模式的优化目标是 Top/Bottom 超额
        else:
            sort_col = "t_stat"
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


# ===========================================================================
# 报告23 改进方向2：非线性因子的使用方法（线性化）
# ===========================================================================
def cubic_residual_transform(fp: pd.DataFrame) -> pd.DataFrame:
    """三次方回归残差法（华泰报告23，参考 BARRA NLS，Menchero 2010）。

    用 F³ 对 F **过原点回归取残差**：残差峰值位于 F 的中间部分、两端较低
    （中间凸），与互信息高分因子的"中间层收益高、两端低"模式契合。
    只利用因子自身信息（不含收益），实现简单但转换效果较差。

    逐截面独立处理，无未来函数。
    """
    out = pd.DataFrame(np.nan, index=fp.index, columns=fp.columns)
    for d in fp.index:
        f = fp.loc[d]
        m = f.notna()
        n = int(m.sum())
        if n < 3:
            continue
        x = f[m].astype(float).values
        beta, *_ = np.linalg.lstsq(x.reshape(-1, 1), x ** 3, rcond=None)
        resid = x ** 3 - x.reshape(-1, 1) @ beta
        out.loc[d, m] = resid
    return out


def polynomial_transform(fp: pd.DataFrame, returns_panel: pd.DataFrame,
                         fit_window: int = 250, refit: int = 20,
                         degree: int = 3, window: int = 20) -> pd.DataFrame:
    """多项式拟合法（华泰报告23）：r = a·F³ + b·F² + c·F + d，把非线性因子线性化。

    华泰原口径：用 T+1 期收益对 T 期因子回归（滚动回归，历史 500 天样本、
    每 20 个交易日重拟合一次），拟合参数用于转换未来 20 个交易日的因子。
    本项目：目标改用未来 ``window`` 日累计收益（月频 horizon，与挖掘口径一致）；
    ``fit_window`` 因样本量可调（华泰 500 天，单年数据需缩小）。

    Args:
        fp: 待转换因子面板（date×code）。
        returns_panel: 日频次日收益面板（已前移一期）。
        fit_window: 每次拟合使用的历史交易日数。
        refit: 每多少个交易日重拟合一次。
        degree: 多项式最高阶（华泰取 3）。
        window: 预测目标 horizon（华泰 20 交易日）。
    Returns:
        线性化后的因子面板（拟合值），无未来函数（只用历史样本拟合）。
    """
    r_forward = _monthly_forward_returns(returns_panel, window=window)
    idx = fp.index
    n = len(idx)
    out = pd.DataFrame(np.nan, index=idx, columns=fp.columns)
    fit_pts = list(range(fit_window, n, refit))
    if not fit_pts:
        return out
    for t in fit_pts:
        lo = t - fit_window
        F = fp.iloc[lo:t].values.ravel()
        R = r_forward.iloc[lo:t].values.ravel()
        valid = np.isfinite(F) & np.isfinite(R)
        if int(valid.sum()) < degree + 2:
            continue
        Fv, Rv = F[valid], R[valid]
        X = np.column_stack([Fv ** 3, Fv ** 2, Fv, np.ones_like(Fv)])
        beta, *_ = np.linalg.lstsq(X, Rv, rcond=None)
        t_end = min(t + refit, n)
        Fnv = fp.iloc[t:t_end].values.ravel()
        Xn = np.column_stack([Fnv ** 3, Fnv ** 2, Fnv, np.ones_like(Fnv)])
        pred = (Xn @ beta).reshape(t_end - t, fp.shape[1])
        out.iloc[t:t_end] = pred
    return out
