"""
因子 IC 统计
============

面板级因子检验的核心统计量（date×code 面板 → 标量/序列指标）：

- IC (Information Coefficient): 因子值与未来收益的截面相关系数
- IR (Information Ratio): IC 均值 / IC 标准差 × √PERIODS_PER_YEAR
- IC 衰减: 不同持有期的 IC 变化
- 分层回测: 按因子值分 N 组，看各组收益单调性
- 截面排名自相关: 换手率代理（factor_autocorr）

所有函数接收:
- factor_panel: DataFrame(date, code), 因子值
- returns_panel: DataFrame(date, code), 日收益率（未来一期口径）

真源历史：2026-08-29 自 research/factor_analysis.py 下沉（该模块保留
standard_factor_summary / factor_summary / calc_neutral_ic_series 等研究级
API 并对本模块 re-export）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401  （保留原模块 import 面，无直接使用）

from stats import PERIODS_PER_YEAR

__all__ = [
    "calc_ic_series", "calc_ir", "calc_ic_decay",
    "quantile_backtest", "factor_autocorr",
    "monotonicity_ratio", "perturbation_fidelity",
]


def calc_ic_series(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    method: str = "spearman",
    returns_rank: pd.DataFrame | None = None,
) -> pd.Series:
    """计算每日 IC（因子值与未来收益的截面相关系数）。

    Args:
        factor_panel: DataFrame(date, code), 因子值。
        returns_panel: DataFrame(date, code), 未来一期收益。
        method: 'pearson' 或 'spearman'（默认，Rank IC）。
        returns_rank: **可选性能捷径**——预计算好的收益 rank 面板（训练中收益
            面板固定，省掉每次重复 rank，GFlowNet 实测约省 30% IC 耗时）。
            ⚠️ **只在"因子不比收益多出任何 NaN"时才与默认路径完全等价**：
            IC 的有效掩码是 ``factor.notna() & returns.notna()``，依赖因子自身的
            NaN 模式，rank 值随掩码变化，因此预计算的 rank 无法对任意因子都成立。
            实测边界（2026-09-10）：
            - 因子**整行缺失**（如窗口预热）→ 该行掩码全 False，两侧结果一致；
            - 因子在收益有效处还有**行内散点缺失** → 出现差异，30% 散点缺失时
              日均 IC 相差约 1.1e-2（IC 量级 3e-2~5e-2）。
            **判显著 / 入库 / 出报告一律走默认路径**（见 2026-09-10 审计记录）。
    Returns:
        Series(index=date), 每日 IC。

    **向量化实现（2026-08-03 优化）**：spearman IC 等价于对因子与收益做
    截面排名后再算 pearson 相关，因此用 ``rank(axis=1) + corrwith(axis=1)``
    一次向量化算出全部日期的 IC，替代原逐日 ``stats.spearmanr`` 循环
    （evaluate_candidates 全量评估 350 候选耗时 117s → 优化后大幅下降）。

    **真源（2026-09-10）**：本函数是全库唯一的 IC 口径实现。
    ``factor.gflownet.reward.rank_ic_series`` 已改为薄封装（此前自实现一份）。
    """
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]

    if method == "spearman":
        # 只在因子与收益均有效的股票子集上做截面排名（与旧实现"先剔缺失再排名"
        # 严格一致，避免全截面排名带来的 rank 基准漂移），再 pearson = spearman。
        valid = fp.notna() & rp.notna()
        fr = fp.where(valid).rank(axis=1)
        # 预计算路径同样要按本次掩码 where —— 否则 rank 值来自另一个掩码
        if returns_rank is not None:
            rr = returns_rank.reindex(index=common_dates,
                                      columns=common_codes).where(valid)
        else:
            rr = rp.where(valid).rank(axis=1)
        ic = fr.corrwith(rr, axis=1, method="pearson")
    else:
        ic = fp.corrwith(rp, axis=1, method="pearson")
    ic = ic.astype(float)

    # 与旧实现一致：有效观测 <5 的日期视为缺失
    valid_cnt = (fp.notna() & rp.notna()).sum(axis=1)
    ic[valid_cnt < 5] = np.nan
    ic.name = "ic"
    return ic


def calc_ir(ic_series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """信息比率 = IC均值 / IC标准差 × √PERIODS_PER_YEAR。"""
    ic = ic_series.dropna()
    if len(ic) < 2 or ic.std() == 0:
        return 0.0
    return ic.mean() / ic.std() * np.sqrt(periods_per_year)


def calc_ic_decay(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    max_lag: int = 10,
) -> pd.Series:
    """IC 衰减: 持有 lag 期收益的 IC（lag=1 即次日，与主 IC 口径完全一致）。

    ``returns_panel`` 约定与 ``calc_ic_series`` 相同：**未来一期收益面板**
    （即 returns[d] = close[d+1]/close[d]-1，已前移一期）。因此:

        decay[lag] = corr( factor[t], 未来第 lag 期收益 ) = corr(factor[t], aligned[t+lag-1])

    早期实现内部再 shift(-lag)，把"未来一期"又前移一期，导致 decay[1] 实际是
    "未来第 2 天"（off-by-one）—— 已修复为 shift(-(lag-1))。
    """
    decay = {}
    for lag in range(1, max_lag + 1):
        shifted_returns = returns_panel.shift(-(lag - 1))
        ic = calc_ic_series(factor_panel, shifted_returns)
        decay[lag] = ic.mean()
    return pd.Series(decay, name="ic_decay")


def quantile_backtest(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """分层回测: 按因子值分 N 组，计算各组累计收益。

    **口径（2026-08-03 修复）**：与主回测引擎 / IC 完全一致 —— 当日因子
    ``factor[t]`` 赚当日（未来一期）收益 ``returns_panel[t]``。早期实现用
    ``factor[t-1]`` 赚 ``returns_panel[t]``，因子整体晚一天生效，分层单调性
    检验系统性丢失一天信号（与 IC/回测结论错位）。

    Returns:
        DataFrame(index=date, columns=quantile_1~N), 各组累计净值。
    """
    common_dates = factor_panel.index.intersection(returns_panel.index)
    common_codes = factor_panel.columns.intersection(returns_panel.columns)
    fp = factor_panel.loc[common_dates, common_codes]
    rp = returns_panel.loc[common_dates, common_codes]

    group_returns = pd.DataFrame(0.0, index=common_dates, columns=[f"Q{i+1}" for i in range(n_quantiles)])

    for i, date in enumerate(common_dates):
        f = fp.iloc[i].dropna()  # 当日因子
        r = rp.iloc[i]           # 当日（未来一期）收益，与 IC 同口径

        common = f.index.intersection(r.index)
        if len(common) < n_quantiles:
            continue
        f_aligned = f.loc[common]
        r_aligned = r.loc[common]

        try:
            groups = pd.qcut(f_aligned, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        n_actual = groups.nunique()
        for g in range(n_actual):
            mask = groups == g
            if mask.sum() > 0:
                group_returns.iloc[i, g] = r_aligned[mask].mean()

    # 累计净值
    cum = (1 + group_returns).cumprod()
    return cum


def factor_autocorr(factor_panel: pd.DataFrame, max_lag: int = 1) -> float:
    """因子截面排名自相关（换手率代理，Alphalens 风格）。

    对相邻期计算因子排名的 spearman 相关并取均值。越接近 1 表示因子排序越稳定
    （换手越低、交易成本越小）；越接近 0 表示每日大换血（换手高、成本高）。
    无需跑回测即可估算因子本身的"粘性"，是 IC/IR 之外判断因子能否落地的关键维度。

    实现：先对每期做截面排名，再用向量化 corrwith(method='spearman') 算相邻期
    排名相关，避免逐日调 scipy 的性能问题。
    """
    ranked = factor_panel.rank(axis=1)
    vals = []
    for lag in range(1, max_lag + 1):
        prev = ranked.shift(lag)
        c = prev.corrwith(ranked, axis=1, method="spearman").dropna()
        if len(c):
            vals.append(float(c.mean()))
    return float(np.mean(vals)) if vals else 0.0


def monotonicity_ratio(
    factor_panel: pd.DataFrame,
    returns_panel: pd.DataFrame,
    n_quantiles: int = 5,
    direction: str = "auto",
    min_valid_days: int = 60,
) -> float:
    """分层收益单调性占比（"因子动物园"文献的 monotonicity 指标）。

    每个截面日按因子值分 ``n_quantiles`` 组（与 :func:`quantile_backtest` 同口径：
    当日因子赚当日未来一期收益、qcut duplicates=drop），若各组**平均收益从低
    因子组到高因子组依次递增**（严格单调上升）记该日"单调"；单调性占比 =
    单调日数 / 有效日数。

    与 IC 的关系（t+单调性论文的动机）：IC/多空价差只看两端，单调性约束
    **中间分组也须有序**。注意本项目实测（09-23，连续因子子集 n=138）mono
    与 |IC| 的 Spearman ≈ +0.34——弱相关而非论文所称的近零正交；"独立预测
    样本外"是否成立须以本项目 920 后 OOS 验证为准。

    Args:
        direction: "auto"（默认）= 每日取"递增或递减"较优方向计入（因子
            方向未知时用，与入库检验的 |IC| 语义对齐）；"asc" = 只认严格
            递增；"desc" = 只认严格递减。
            注意：auto 会把"始终稳定反号"的因子也算满单调——这正确，因为
            因子可取反；真正被它抓的是"中间组乱序"的结构性缺陷。
        min_valid_days: 有效日下限，低于此值返回 NaN（一个占比只有在足够
            多的日子里才有意义）。

    Returns:
        float ∈ [0, 1]；有效日不足或无有效日返回 NaN。

    .. warning:: 2026-09-23 两处口径修正（低基数因子上旧实现产出误导值）：
       1. **只认完整分组**：qcut duplicates=drop 后组数 < n_quantiles 的日
          跳过。旧实现只要 ≥2 组就计入——而 auto 语义下 2 组日**必然**记
          "单调"（单差分非正即负），3 组日随机基线 50%，二值/哑变量因子
          mono 虚高，甚至由单日拼出假满分（alpha191_004 有效日仅 1 天 →
          mono=1.0，实为 1/1）。合成对照：与收益无关的三值因子旧口径
          mono=0.96（随机基线应 ≈0.1）。简并面板上的正确答案 = NaN
          （5 分位单调性无定义），不是假 1.0。
       2. **min_valid_days 下限**（新增参数）。
    """
    common_dates = factor_panel.index.intersection(returns_panel.index)
    fp = factor_panel.loc[common_dates]
    rp = returns_panel.loc[common_dates]

    n_ok = 0
    n_mono = 0
    for i, _date in enumerate(common_dates):
        f = fp.iloc[i].dropna()
        r = rp.iloc[i]
        common = f.index.intersection(r.index)
        if len(common) < n_quantiles:
            continue
        f_aligned = f.loc[common]
        r_aligned = r.loc[common]
        try:
            groups = pd.qcut(f_aligned, n_quantiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        n_actual = groups.nunique()
        # 只认完整分组（09-23 修正）：组数不足 = 分位数塌缩，该日无定义；
        # 旧口径 ≥2 组即计入，让 auto 在低基数下几乎必真
        if n_actual < n_quantiles:
            continue
        means = np.array([
            r_aligned[groups == g].mean() if (groups == g).sum() else np.nan
            for g in range(n_actual)
        ])
        if not np.isfinite(means).all():
            continue
        diffs = np.diff(means)
        asc = bool((diffs > 0).all())
        desc = bool((diffs < 0).all())
        if direction == "asc":
            mono = asc
        elif direction == "desc":
            mono = desc
        else:
            mono = asc or desc
        n_ok += 1
        n_mono += int(mono)

    if n_ok < max(min_valid_days, 1):
        return float("nan")
    return n_mono / n_ok


def perturbation_fidelity(
    factor_panel: pd.DataFrame,
    noise_scale: float = 0.01,
    n_trials: int = 20,
    seed: int = 42,
    distribution: str = "gauss",
    df_t: float = 5.0,
    tie_aware: bool = True,
) -> float:
    """扰动保真度 PFS（AlphaEval 框架的 robustness 维度）。

    对因子面板加乘性噪声 ``panel * (1 + noise_scale * eps)``，计算扰动前后
    **截面排名**的 Spearman 相关（对每个截面日算再取均值，多 trial 再取均值）。
    越接近 1 = 排名对数据噪声越不敏感（鲁棒）；偏低 = 排名被少数极端值
    主导（扰动后名次大幅洗牌，实盘中对应"对数据瑕疵/口径噪声敏感"）。

    与 :func:`factor_autocorr` 的区别：autocorr 衡量**时间上**因子自身
    延续性（换手代理）；PFS 衡量**横截面上**排名对噪声的敏感度（数据
    鲁棒性）——两者正交，前者高不保证后者高。

    Args:
        noise_scale: 相对噪声幅度（1% 默认，约等于数据瑕疵/复权舍入量级）。
        n_trials: 重复次数（每次独立抽噪声，均值降噪）。
        seed: 随机种子（固定以保证同一因子重复入库结果一致）。
        distribution: "gauss"（默认）或 "t"（重尾扰动，模拟极端数据错误，
            论文用 t 分布作为更强扰动）。
        df_t: t 分布自由度（仅 distribution="t" 时生效）。
        tie_aware: True（默认，09-23 修正）= **同一截面内相同值共享同一扰动**
            ——数据瑕疵语义：同一输入值受同样的系统性误差，并列名次集体
            移动、相对秩序不变。False = 旧口径，逐元素独立噪声，会把本应
            不可分的并列值人为打散（pap_breakout_atr 旧口径 PFS=0.145，
            修正后 1.0——99.4% 并列面板的排名"脆弱"纯为噪声模型 artifact）。
            与 IC 显著性独立：低 PFS 的真实来源应是**连续重尾分布**（少数
            极端值之间的微小间隙被噪声放大），不是并列。

    Returns:
        float ∈ [-1, 1]（实际应接近 1）；无法计算返回 NaN。
    """
    rng = np.random.default_rng(seed)
    fp = factor_panel
    if not isinstance(fp, pd.DataFrame) or fp.empty:
        return float("nan")

    ranked = fp.rank(axis=1)
    # 两个 rank 面板之间的 Spearman ≡ Pearson（单调变换不变性）——直接在
    # 已 rank 的数据上用默认 Pearson 省掉 corrwith 内部的重复整表排名。
    # 2026-09-22 对照验证：单因子 5 trials 结果逐位一致（max|Δ|=0），
    # 32.4s → 21.9s。
    vals: list[float] = []
    for _ in range(n_trials):
        if tie_aware:
            # 并列共享扰动：按截面日的 unique 值抽噪声。pd.factorize 保序
            # 与 pd.unique 保序一致 → draw 分配与逐行 map 版逐位一致
            # （09-23 对照 max|Δ|=0）；向量化 factorize 比 Series.map 快 ~3x
            # （5807×2471 面板 16.7s → 5.5s/trial）。
            fp_vals = fp.to_numpy()
            noise_vals = np.zeros_like(fp_vals)
            for i in range(len(fp)):
                row = fp_vals[i]
                valid = ~np.isnan(row)
                codes, uv = pd.factorize(row[valid])
                if len(uv) == 0:
                    continue
                draws = 1.0 + noise_scale * (
                    rng.standard_t(df_t, len(uv)) if distribution == "t"
                    else rng.normal(0.0, 1.0, len(uv))
                )
                noise_vals[i][valid] = draws[codes]
            noise = pd.DataFrame(noise_vals, index=fp.index, columns=fp.columns)
            perturbed = (fp * noise).rank(axis=1)
        else:
            if distribution == "t":
                eps = rng.standard_t(df_t, size=fp.shape)
            else:
                eps = rng.normal(0.0, 1.0, size=fp.shape)
            noise = pd.DataFrame(eps, index=fp.index, columns=fp.columns)
            perturbed = (fp * (1.0 + noise_scale * noise)).rank(axis=1)
        c = ranked.corrwith(perturbed, axis=1)
        v = c.dropna()
        if len(v):
            vals.append(float(v.mean()))
    if not vals:
        return float("nan")
    return float(np.mean(vals))
