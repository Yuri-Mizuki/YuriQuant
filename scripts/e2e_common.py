"""
端到端工作流共享模块
====================

e2e_stock_picks.py（今日选股）与 e2e_backtest.py（walk-forward 回测）共用的
数据加载 / 因子构建 / 特征选择 / 标签逻辑，避免两处重复实现导致口径漂移。

约定（两个脚本必须一致）：
- 股票池：因子库 significant 面板的列并集（HS300 PIT 历史成员，~420 股）
- 因子：经典量价 12 + 因子库 significant（默认排除 model:* 防循环引用）
- 特征选择：build_feature_set 三级漏斗（覆盖率>=0.5 → |corr|<0.7 去冗余
  → valid 段 |IC| 质量分降序截断），只在调用方指定的选择窗口上做（防前视）
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
    d = pd.read_parquet(cache.root / "daily.parquet")

    # 股票池 = significant 因子面板列并集（排除 model:*）
    lib_feats = load_library_factors(exclude_model=True)
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


def load_mock_data(n_days: int = 500, n_codes: int = 50, seed: int = 0) -> dict:
    """生成 mock OHLCV 面板（无外部依赖，供测试/演示）。"""
    import numpy as np

    rng = np.random.RandomState(seed)
    codes = [f"{i:06d}.SZ" for i in range(n_codes)]
    dates = pd.bdate_range(end="2026-08-21", periods=n_days)
    close = 100 * np.exp(np.cumsum(rng.randn(n_days, n_codes) * 0.02, axis=0))
    px = {}
    px["close"] = pd.DataFrame(close, index=dates, columns=codes)
    px["open"] = px["close"] * (1 + rng.randn(n_days, n_codes) * 0.005)
    px["high"] = pd.DataFrame(
        np.maximum(px["open"].values, px["close"].values) * (1 + rng.rand(n_days, n_codes) * 0.01),
        index=dates, columns=codes)
    px["low"] = pd.DataFrame(
        np.minimum(px["open"].values, px["close"].values) * (1 - rng.rand(n_days, n_codes) * 0.01),
        index=dates, columns=codes)
    px["volume"] = pd.DataFrame(rng.randint(1e6, 1e8, (n_days, n_codes)),
                                index=dates, columns=codes)
    px["amount"] = px["volume"] * px["close"]
    return px


# ---------------------------------------------------------------------------
# 因子
# ---------------------------------------------------------------------------
def compute_classic_features(px: dict) -> dict:
    """经典量价因子（动量/反转/波动/流动性/换手结构），截面标准化。"""
    from factor.preprocessing import standardize_zscore

    close, open_ = px["close"], px["open"]
    amount = px["amount"]
    ret1 = close.pct_change(fill_method=None)
    feats = {
        "mom5": close.pct_change(5, fill_method=None),
        "mom10": close.pct_change(10, fill_method=None),
        "mom20": close.pct_change(20, fill_method=None),
        "mom60": close.pct_change(60, fill_method=None),
        "rev1": -ret1,
        "rev5": -close.pct_change(5, fill_method=None),
        "vol20": ret1.rolling(20).std(),
        "vol60": ret1.rolling(60).std(),
        "amihud20": (ret1.abs() / (amount + 1e-12)).rolling(20).mean(),
        "turn_trend": (px["volume"].rolling(5).mean()
                       / (px["volume"].rolling(60).mean() + 1e-12)),
        "gap": open_ / close.shift(1) - 1,
        "range20": (px["high"] - px["low"]).rolling(20).mean() / (close + 1e-12),
    }
    return {k: standardize_zscore(v) for k, v in feats.items()}


def load_library_factors(exclude_model: bool = True) -> dict:
    """加载因子库 significant 因子面板。

    Args:
        exclude_model: 排除 model:* 来源（模型预测回写因子，面板通常滞后且
            与本工作流自身预测循环引用）。默认 True——这是 e2e 预测日能到
            2026-08-21 的关键（model:* 面板截至 2025-12-31）。
    """
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(dataset=DATASET)
    reg = lib.list_all()

    mask = reg["significant"] == True
    if exclude_model:
        mask &= ~reg["source"].fillna("").str.startswith("model:")
    sig_names = set(reg[mask]["name"])
    log.info("因子库: %d 个因子, significant %d 个（排除 model:* 后 %d）",
             len(reg), int((reg["significant"] == True).sum()), len(sig_names))

    all_feats = lib.load_library_features()
    feats = {k: v for k, v in all_feats.items() if k in sig_names}
    if feats:
        sample = next(iter(feats.values()))
        log.info("加载面板 %d 个, 日期范围 %s ~ %s",
                 len(feats), sample.index[0].date(), sample.index[-1].date())
    return feats


# ---------------------------------------------------------------------------
# 特征选择（防前视：只在调用方指定的选择窗口上做）
# ---------------------------------------------------------------------------
def select_features(
    all_feats: dict,
    fwd: pd.DataFrame,
    sel_days: pd.DatetimeIndex,
    max_features: int = 30,
) -> tuple[dict, pd.Series]:
    """build_feature_set 三级漏斗：覆盖率 -> 去冗余 -> 质量排序截断。

    Returns:
        (入选因子面板 {name: 全量日期面板}, valid 段 |IC| 质量分 Series)
    """
    from model.features import build_feature_set
    from research.factor_analysis import calc_ic_series

    q = {}
    for nm, p in all_feats.items():
        try:
            ic = calc_ic_series(p.reindex(index=sel_days), fwd.reindex(index=sel_days)).dropna()
            if len(ic) >= 10:
                q[nm] = abs(float(ic.mean()))
        except Exception:
            pass
    quality = pd.Series(q).sort_values(ascending=False)

    # reindex 而非 loc：经典因子从 2019 起、因子库从 2022 起，日期网格不同
    feats_sel = build_feature_set(
        {k: v.reindex(index=sel_days) for k, v in all_feats.items()},
        min_coverage=0.5, dedup_corr=0.7, max_features=max_features, quality=quality)
    selected = sorted(feats_sel)
    log.info("特征漏斗: %d -> %d（覆盖率>=0.5, |corr|<0.7, 上限 %d）",
             len(all_feats), len(selected), max_features)
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
# 标签
# ---------------------------------------------------------------------------
def build_labels(close: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(labels: rank 变换, fwd: 原始 horizon 收益)。"""
    from model.labels import build_labels as _build_labels, forward_returns

    fwd = forward_returns(close, horizon=horizon)
    labels, _ = _build_labels(close_panel=close, horizon=horizon, mode="rank")
    return labels, fwd
