"""宏观日历**市场状态特征**（altf P1 落地，2026-10-10）。

背景（``reports/altf_p1/README.md``）：``alt_macro_*`` 7 因子已入库
（registry 955→975），但它们是**市场级广播面板**（同日全市场同值，实测行
``nunique ∈ {0, 1}``）⇒ 逐股截面 IC 无定义（5/7 因子截面 IC = NaN）。
它们正确的用法不是横截面因子，而是**市场状态特征** —— 把当前宏观环境告诉
GBDT，让它条件化选股（"普涨/普跌 vs 结构分化"下该选什么）。

与 :mod:`model.market_features`（指数点位派生）**并列**：
  - 指数点位 = 价格/交易维度的市场状态；
  - 宏观 surprise = 经济数据维度的市场状态。
两者注入同一旁路（``--market-features`` / ``--macro-features`` /
``rolling_grid_alla.stage_predict`` 的 ``mkt_feats``），互不覆盖。

⚠️ 与 market_features 同型的三条纪律（勿省）：
  1. **不得进截面 zscore 通道**（同日同值 → std=0 → 整列 NaN）；
  2. **防前视**：只依赖 t 日及以前（expanding 天然回看；宏观 surprise
     本身已按公布时刻 + 15:00 收盘闸门对齐，见 ``build_alla_altfactors``）；
  3. **零信息置 NaN**：expanding std=0 置 NaN，不写防除零常数。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model.market_features import Z_MIN_PERIODS, _expanding_z

log = logging.getLogger("model.macro_features")

__all__ = [
    "build_macro_state_features",
    "MACRO_STATE_PANELS",
    "MACRO_FEAT_VERSION",
]

#: 特征集版本：改特征定义/增删因子时 +1（进入 pred 指纹，防静默混口径）。
MACRO_FEAT_VERSION = 1

#: 参与市场状态化的 alt_macro 广播面板（factor_library 面板名）。
#: 不含 ``alt_macro_surprise_str5_x_vol20`` —— 那个是**个股交互**（×个股 20 日
#: 波动），本身就是 stock-level，截面有意义，不走本通道。
MACRO_STATE_PANELS: tuple[str, ...] = (
    "alt_macro_surprise_str_5d",
    "alt_macro_surprise_net_20d",
    "alt_macro_chn_surprise_str_5d",
    "alt_macro_imp3_cnt_20d",
    "alt_macro_surprise_pos_cnt_1d",
    "alt_macro_surprise_neg_cnt_1d",
)


def _market_series(panel: pd.DataFrame) -> pd.Series:
    """广播面板 → 市场级日序列。

    行内同值（忽略 NaN），故 ``mean(axis=1)`` 即该值、且对全 NaN 行返回 NaN；
    比"取第一列"稳（不同股票的列覆盖日期不同），比 ``apply`` 快两个数量级。
    """
    return panel.mean(axis=1, skipna=True)


def build_macro_state_features(
    macro_panels: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    codes: pd.Index | list[str],
) -> dict[str, pd.DataFrame]:
    """alt_macro 广播面板 → date×code 市场状态特征面板。

    Args:
        macro_panels: {面板名: date×code 广播面板}（如 ``MACRO_STATE_PANELS``
            对应的 factor_library parquet）。
        dates: 目标日期网格（个股面板 index，如 ``close.index``）。
        codes: 目标股票网格（个股面板 columns）。

    Returns:
        {``f"mac_{面板名去前缀}"``: date×code 面板}；同日全市场同值，
        序列缺失/预热不足的日期整行 NaN。float32（全A网格省内存）。
    """
    if not macro_panels:
        raise ValueError("macro_panels 为空")
    cols = pd.Index(codes)
    idx = pd.DatetimeIndex(dates)
    out: dict[str, pd.DataFrame] = {}
    for name, panel in macro_panels.items():
        s = _market_series(panel).reindex(idx)
        if s.notna().sum() == 0:
            log.warning("macro 面板 %s 全 NaN，跳过", name)
            continue
        z = _expanding_z(s)  # expanding z（min_periods=Z_MIN_PERIODS；std=0 → NaN）
        v = z.to_numpy(dtype="float64")
        feat = f"mac_{name.removeprefix('alt_macro_')}"
        out[feat] = pd.DataFrame(
            np.tile(v[:, None], (1, len(cols))).astype("float32"),
            index=idx, columns=cols,
        )
    log.info("宏观市场状态特征: %d 面板（广播 %d 日 × %d 股；min_periods=%d）",
             len(out), len(idx), len(cols), Z_MIN_PERIODS)
    return out
