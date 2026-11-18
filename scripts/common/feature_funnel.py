"""特征选择漏斗（覆盖率 → 去冗余 → 质量分截断）—— 口径中性的单一实现。

2026-11-18 从 ``scripts/archive/e2e_common.py`` 提取。原模块 447 行里只有本函数
是**口径中性**的（不读 HS300 日线、不绑数据集），被 cpcv / tabfm / walk-forward
三条评估链共用；其余部分（``load_daily_data`` / ``load_library_grid_panels`` /
``build_neutral_covariates`` 等）绑死 ``daily_hs300.parquet`` 与 ``hs300_2022_2025``
库，数据已归档、首调用即抛异常，随原模块一并删除。

为什么单独成模块：把中性工具留在 ``archive/`` 里会形成「活代码藏在坟场」——
后来人要么不敢删、要么误删。判据只看 import 可达性与口径中性，不看目录名。
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("feature_funnel")


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
