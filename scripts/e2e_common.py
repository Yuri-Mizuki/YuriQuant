"""
端到端工作流共享模块
====================

e2e_stock_picks.py（今日选股）、e2e_backtest.py（walk-forward 回测）与
investment_report / optimize_e2e 等 e2e 家族脚本共用的编排逻辑：
数据加载（``load_daily_data``）、特征选择漏斗（``select_features``）、
面板新鲜度守卫（``drop_stale_factors``）、风格中性化
（``build_neutral_covariates`` / ``neutralize_predictions``）。

约定（e2e 家族必须一致）：
- 股票池：因子库 significant 面板的列并集（HS300 PIT 历史成员，~420 股）
- 因子：经典量价 12（``factor.classic.compute_classic_features``）+ 因子库
  significant（``FactorLibrary.load_significant_features``，默认排除 model:*）
- 特征选择：build_feature_set 三级漏斗（覆盖率>=0.5 → |corr|<0.7 去冗余
  → valid 段 |IC| 质量分降序截断），只在调用方指定的选择窗口上做（防前视）

已下沉（2026-08-31，见 packages 单一实现）：
- 经典特征 / mock 数据 / 标签构建 / 因子库加载 → factor.classic / data.mock /
  model.labels / research.factor_library，scripts 只保留编排入口。
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("e2e_common")

HORIZON = 5
GBDT_PARAMS = {
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_child_samples": 100,
    "n_estimators": 200,
    "seed": 42,
}
RIDGE_ALPHA = 1.0
DATASET = "hs300_2022_2025"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_daily_data(begin: int = 20220101) -> tuple[dict, dict]:
    """读日线缓存并对齐到因子库股票池。

    Returns:
        (px, lib_feats): px = {open/high/low/close/volume/amount: date×code}，
        lib_feats = {name: date×code} 因子库 significant 面板（默认排除 model:*）。
    """
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from research.factor_library import FactorLibrary

    cache = DataCache(OfflineDataSource())
    d = pd.read_parquet(cache.root / "daily_hs300.parquet")

    # 股票池 = significant 因子面板列并集（排除 model:*）
    lib_feats = FactorLibrary(dataset=DATASET).load_significant_features(exclude_model=True)
    pool = None
    for p in lib_feats.values():
        pool = set(p.columns) if pool is None else pool | set(p.columns)
    codes = sorted(pool)
    dates_all = sorted(d.index.get_level_values("date").unique())
    log.info("日线缓存: %s ~ %s | 股票池 %d 股", dates_all[0].date(), dates_all[-1].date(), len(codes))

    px = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        w = (d.reset_index()
             .pivot(index="date", columns="code", values=col).sort_index())
        w = w.reindex(columns=codes)
        w = w[w.index >= pd.Timestamp(str(begin))]
        px[col] = w
    return px, lib_feats


def load_library_grid_panels(dataset: str = DATASET) -> dict[str, pd.DataFrame]:
    """读日线缓存，对齐到因子库全面板网格的 OHLCV 宽表（date×code）。

    2026-08-31 自 ml_synthesis_experiment._px_panels 下沉（cpcv_h1_eval 等共用）。
    """
    from data.cache import DataCache
    from data.offline import OfflineDataSource
    from research.factor_library import FactorLibrary

    cache = DataCache(OfflineDataSource())
    d = pd.read_parquet(cache.root / "daily_hs300.parquet")

    grid = next(iter(FactorLibrary(dataset=dataset).load_library_features().values()))
    codes, dates = grid.columns, grid.index

    out = {}
    for col in ("open", "high", "low", "close", "volume", "amount"):
        w = (d.reset_index()
             .pivot(index="date", columns="code", values=col).sort_index()
             .reindex(index=dates, columns=codes))
        out[col] = w
    return out


# ---------------------------------------------------------------------------
# 特征选择（防前视：质量分与漏斗都只在调用方指定的定型窗口上做）
# ---------------------------------------------------------------------------
def select_features(
    all_feats: dict,
    fwd: pd.DataFrame,
    quality_days: pd.DatetimeIndex,
    panel_days: pd.DatetimeIndex | None = None,
    max_features: int = 30,
    min_coverage: float = 0.5,
    dedup_corr: float = 0.7,
) -> tuple[dict, pd.Series | None]:
    """build_feature_set 三级漏斗（**单一实现**，2026-08-29 收敛 7 份脚本内联副本）。

    质量分在 ``quality_days``（通常 valid 段）上计算 |IC|，漏斗在
    ``panel_days``（通常 dev 段 = train+valid）上做覆盖率 -> 去冗余 -> 截断，
    两窗口分离以匹配"定型期选择、test 不参与"的防前视纪律。

    Args:
        all_feats: {name: 全时段面板}（建议已截面标准化）。
        fwd: horizon 前瞻收益面板（与质量分窗口对齐用）。
        quality_days: 质量分计算窗口。
        panel_days: 漏斗窗口；None = 与 quality_days 相同（旧调用行为）。
        max_features / min_coverage / dedup_corr: 漏斗三参数。
    Returns:
        (入选因子的【全时段】面板 {name: panel}, 质量分 Series | None)。
        质量分为空时返回 None（build_feature_set 回退按独立性去冗余）。
    """
    from model.features import build_feature_set
    from stats.ic import calc_ic_series

    q = {}
    for nm, p in all_feats.items():
        try:
            ic = calc_ic_series(p.reindex(index=quality_days),
                                fwd.reindex(index=quality_days)).dropna()
            if len(ic) >= 10:
                q[nm] = abs(float(ic.mean()))
        except Exception:
            pass
    quality = pd.Series(q).sort_values(ascending=False) if q else None
    days = quality_days if panel_days is None else panel_days

    # reindex 而非 loc：经典因子与因子库面板的日期网格可能不同（reindex 是超集）
    feats_sel = build_feature_set(
        {k: v.reindex(index=days) for k, v in all_feats.items()},
        min_coverage=min_coverage, dedup_corr=dedup_corr,
        max_features=max_features, quality=quality)
    selected = sorted(feats_sel)
    log.info("特征漏斗: %d -> %d（覆盖率>=%.2f, |corr|<%.2f, 上限 %d）",
             len(all_feats), len(selected), min_coverage, dedup_corr, max_features)
    return {k: all_feats[k] for k in selected}, quality


def drop_stale_factors(
    feats: dict,
    as_of_date: pd.Timestamp,
    buffer_days: int = 5,
) -> dict:
    """面板新鲜度守卫：剔除末端早于 as_of_date - buffer 的失效面板。

    单个滞后面板（如数据源中断未回补）会把整个公共网格的末端拖回，
    导致预测日/回测区间人为缩短——剔除并告警，而不是拖垮全流程。
    """
    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(days=buffer_days)
    stale = []
    for k, v in feats.items():
        last = v.dropna(how="all").index
        if len(last) == 0 or last[-1] < cutoff:
            stale.append(k)
    if stale:
        log.warning("剔除 %d 个滞后面板（末端 < %s）: %s ...",
                    len(stale), cutoff.date(), sorted(stale)[:5])
        feats = {k: v for k, v in feats.items() if k not in set(stale)}
    return feats


# ---------------------------------------------------------------------------
# 风格中性化（华泰五因子：市值/行业/动量/波动/换手）
# ---------------------------------------------------------------------------
def build_neutral_covariates(px: dict, close: pd.DataFrame, real: bool = True):
    """构建市值 + 行业 + mom/vol/turn 协变量面板（华泰五因子口径）。

    Returns:
        (market_cap_panel | None, industry_panel | None, extra_covariates dict)
    """
    from factor.preprocessing import build_style_covariates

    mc_panel, ind_panel = None, None
    if real:
        try:
            from data.cache import DataCache
            from data.industry import IndustryClassification
            from data.market_cap import build_market_cap_panel
            from data.offline import OfflineQuietDataSource

            # 注意：必须用 OfflineQuietDataSource（数据源方法返回空 -> 走缓存
            # fallback）。普通 OfflineDataSource 直接抛异常，get_equity_structure
            # 的缓存分支在 try 之外，永远到不了 -> 市值/行业恒为 None。
            cache = DataCache(OfflineQuietDataSource())
            codes = list(close.columns)
            es = cache.get_equity_structure(codes)
            mc_panel = build_market_cap_panel(es, close)
            ind_panel = IndustryClassification(cache).get_industry_panel(codes, close.index)
            log.info("市值面板: %d 日 × %d 股 | 行业: %d 类",
                     mc_panel.shape[0], mc_panel.shape[1],
                     ind_panel.dropna(how="all").shape[1] if len(ind_panel) else 0)
        except Exception as e:
            log.warning("市值/行业加载失败，仅用价量风格: %s", e)
    extra = build_style_covariates(px, mc_panel, ind_panel)
    return mc_panel, ind_panel, extra


def neutralize_predictions(pred: pd.DataFrame, mc, ind, extra) -> pd.DataFrame:
    """预测分数五因子中性化（逐日截面回归取残差）。

    协变量面板列集（全股票池）需对齐到预测面板列集（模型覆盖子集），
    否则 neutralize 内布尔索引 index union 后 Unalignable。
    """
    from factor.preprocessing import neutralize

    cols = pred.columns
    mc_a = mc.reindex(columns=cols) if mc is not None else None
    ind_a = ind.reindex(columns=cols) if ind is not None else None
    # extra 里 industry/size 与 industry_panel/market_cap_panel 重复（且 industry
    # 是字符串，直接当连续协变量 astype(float) 会崩）。剔除后只留 mom/vol/turn。
    extra_a = {k: v.reindex(columns=cols) for k, v in extra.items()
               if k not in ("industry", "size")}
    neu = neutralize(pred, market_cap_panel=mc_a, industry_panel=ind_a,
                     extra_covariates=extra_a)
    log.info("中性化: 有效值 %d (原始 %d)", neu.notna().sum().sum(),
             pred.notna().sum().sum())
    return neu
