"""跨实验脚本共享的组合 / 模型编排组件（无 CLI 入口）

2026-09-11 由 ``scripts/pipelines/run_model_portfolio.py`` **下沉**：该模块是主实验入口
（全A正交化管线），却被 ``buffer_tune`` / ``freq_tune`` / ``multiyear_oos`` /
``rolling_grid_alla`` 等脚本 **反向 import 其内部函数**——入口脚本成了依赖库。

这里只放"被多方复用的组件"，不含任何 ``main()``。

⚠️ 口径标注：``load_index_benchmark`` / ``build_style_covariates_panel`` /
``build_model_panel`` 属 **legacy HS300 口径**（2026-08-25 固化版），仅供
历史实验脚本复用；主口径是全A正交化管线，**勿用于新实验**。
``neutralize_panel`` 是通用的信号层中性化适配壳（主口径 ``neutralize=false``
时不走这条路径）。
"""
from __future__ import annotations

import pandas as pd

from model.params import DEFAULT_MODEL_PARAMS

__all__ = [
    "LEGACY_DATASET",
    "neutralize_panel",
    "load_index_benchmark",
    "build_style_covariates_panel",
    "build_model_panel",
]

#: legacy HS300 口径数据集（``build_model_panel`` 用）
LEGACY_DATASET = "hs300_2022_2025"


def neutralize_panel(signal, cov):
    """信号层风格中性化（保留给对照实验；主口径 neutralize=false 不走此路径）。

    ``cov`` 是 ``{"size": 市值面板, "industry": 行业面板, **额外连续协变量}``
    形态的协变量集合（与 ``e2e_common.build_neutral_covariates`` 同约定），
    这里解包后交给 :func:`factor.preprocessing.neutralize` 这个唯一真源。
    """
    from factor.preprocessing import neutralize
    size = cov.get("size")
    ind = cov.get("industry")
    extra = {k: v for k, v in cov.items() if k not in ("size", "industry")}
    return neutralize(signal, market_cap_panel=size, industry_panel=ind,
                      extra_covariates=extra)


def load_index_benchmark(test_days: pd.DatetimeIndex) -> pd.Series:
    """指数基准日收益（默认 config.backtest.benchmark，HS300 口径遗留）。"""
    from config import Config
    from data.cache_helpers import load_index_returns
    code = str(Config.get()["backtest"]["benchmark"])
    begin = int(test_days[0].strftime("%Y%m%d"))
    ret = load_index_returns(code, begin=begin, end=None, reindex_to=test_days)
    if ret is None:
        raise FileNotFoundError(
            f"指数 {code} 无缓存：请先运行 update_data 拉取指数日线")
    return ret


def build_style_covariates_panel(panel):
    """由 panel dict 构建风格协变量面板（市值 / 换手 / 波动等）。"""
    from data.cache import DataCache
    from data.industry import IndustryClassification
    from data.offline import OfflineQuietDataSource
    from factor.preprocessing import build_style_covariates
    cache = DataCache(OfflineQuietDataSource())
    payload = {
        "close": panel["close"], "volume": panel["volume"],
        "tot_share": panel["market_cap"] / panel["close"].where(panel["close"] > 0),
    }
    ind = IndustryClassification(cache, level=1).get_industry_panel(
        panel["close"].columns, panel["close"].index)
    return build_style_covariates(payload, market_cap_panel=panel["market_cap"],
                                  industry_panel=ind)


def build_model_panel(model: str, horizon: int, test_days: pd.DatetimeIndex):
    """HS300 口径 walk-forward（legacy）。返回 (test 段 OOS 预测, panel, fwd)。"""
    from config import Config
    from data.cache_helpers import build_panel
    from factor.preprocessing import standardize_zscore
    from model.labels import build_labels
    from model.predictor import PREDICTORS, rolling_oos
    from research.factor_library import FactorLibrary
    from scripts.common.e2e_common import select_features

    disc = Config.discipline()
    panel, _ = build_panel(Config.get(), disc["begin"], 20261231, offline=True,
                           include_market_cap=True)
    close = panel["close"]
    all_days = close.index
    valid_days = all_days[(all_days > pd.Timestamp(str(disc["train_end"]))) &
                          (all_days <= pd.Timestamp(str(disc["valid_end"])))]
    dev_days = all_days[: len(all_days) - len(test_days)]

    feats = FactorLibrary(dataset=LEGACY_DATASET).load_library_features()
    feats = {k: v for k, v in feats.items() if v.index[0].year <= 2022}
    feats = {k: standardize_zscore(v.reindex(close.index)) for k, v in feats.items()}

    labels, embargo = build_labels(close, horizon=horizon, mode="rank")
    fwd_v = close.pct_change(horizon, fill_method=None).shift(-horizon).loc[valid_days]
    sel, _quality = select_features(feats, fwd_v, quality_days=valid_days,
                                    panel_days=dev_days, max_features=50)

    params = DEFAULT_MODEL_PARAMS.get(model, {})
    pred = rolling_oos(PREDICTORS[model], sel, labels, test_days, all_days,
                       n_folds=12, embargo_days=embargo, min_train_days=120, **params)
    fwd = close.pct_change(fill_method=None) if horizon == 1 \
        else close.pct_change(horizon, fill_method=None).shift(-horizon)
    return pred, panel, fwd
