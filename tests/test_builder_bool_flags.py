"""布尔/水平量因子构建口径的回归测试（2026-11-05 假面板事故）。

三个用例对应三类「原始层就没有截面差异」的成因，详见
``reports/structural_fake_panels/``：

* ``is_st``：``.replace(0.0, np.nan)`` 把「非 ST」当缺失 ⇒ 面板只剩全 1
* ``express_reporting``：只 ffill 出「已披露 = 1」，其余留 NaN ⇒ 值域恒 {1}
* ``float_mktcap``：全族 ``clip(-1e4, 1e4)`` 把单位元的市值整列削成常数
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAL = pd.date_range("2024-01-08", periods=5, freq="B")
CODES = ["000001.SZ", "000002.SZ"]


# ---------------------------------------------------------------------------
# is_st：布尔量的 0 必须保留
# ---------------------------------------------------------------------------
def test_is_st_keeps_zero():
    from scripts.builders.build_alla_status_factors import build_panels

    status = pd.DataFrame({
        "date": [CAL[0], CAL[0]],
        "code": CODES,
        "is_suspended": [False, False],
        "is_st": [True, False],          # 一只 ST、一只正常
        "pre_close": [10.0, 10.0],
        "high_limited": [11.0, 11.0],
    })
    out = build_panels(CAL, pd.Index(CODES), status=status)
    flag = out["is_st"]
    assert flag.notna().all().all(), "is_st 不应有 NaN：0（非 ST）是真值"
    assert set(np.unique(flag.to_numpy(dtype="float32"))) == {0.0, 1.0}
    assert flag.loc[CAL[0], "000001.SZ"] == 1.0
    assert flag.loc[CAL[0], "000002.SZ"] == 0.0


def test_is_st_is_limited_to_listed_domain():
    """上市前的样本不该被算成「已上市且非 ST」——域外必须是 NaN。"""
    from scripts.builders.build_alla_status_factors import build_panels

    status = pd.DataFrame({
        "date": [CAL[0], CAL[0]],
        "code": CODES,
        "is_suspended": [False, False],
        "is_st": [True, False],
        "pre_close": [10.0, 10.0],
        "high_limited": [11.0, 11.0],
    })
    close_raw = pd.DataFrame(10.0, index=CAL, columns=CODES)
    close_raw.loc[CAL[:2], "000002.SZ"] = np.nan     # 后 3 天才上市
    flag = build_panels(CAL, pd.Index(CODES), status=status,
                        close_raw=close_raw)["is_st"]
    assert pd.isna(flag.loc[CAL[0], "000002.SZ"])
    assert flag.loc[CAL[-1], "000002.SZ"] == 0.0
    assert flag.loc[CAL[0], "000001.SZ"] == 1.0


# ---------------------------------------------------------------------------
# express_reporting：域内未披露补 0，域外（上市前）保持 NaN
# ---------------------------------------------------------------------------
def test_express_reporting_fills_zero_within_listed_domain():
    from scripts.builders.build_alla_event_factors import _pit_express

    express = pd.DataFrame({
        "code": ["000001.SZ"],
        "ann_date": [pd.Timestamp(CAL[2])],
        "yoy_gr_net_profit_parent": [12.0],
        "yoy_gr_gross_rev": [8.0],
    })
    listed = pd.DataFrame(True, index=CAL, columns=CODES)
    listed.loc[CAL[:2], "000002.SZ"] = False          # 000002 后 3 天才上市

    flag = _pit_express(express, CAL, pd.Index(CODES),
                        listed=listed)["express_reporting"]

    # 000001：披露前 0、披露日及之后 1
    assert flag.loc[CAL[0], "000001.SZ"] == 0.0
    assert flag.loc[CAL[2], "000001.SZ"] == 1.0
    assert flag.loc[CAL[-1], "000001.SZ"] == 1.0
    # 000002：上市前 NaN，上市后未披露 = 0
    assert pd.isna(flag.loc[CAL[0], "000002.SZ"])
    assert flag.loc[CAL[-1], "000002.SZ"] == 0.0


def test_express_reporting_without_listed_mask_degrades_to_fill_zero():
    """兼容旧调用：不给 listed 时全域补 0（形状不变，只是上市前也变 0）。"""
    from scripts.builders.build_alla_event_factors import _pit_express

    express = pd.DataFrame({
        "code": ["000001.SZ"],
        "ann_date": [pd.Timestamp(CAL[2])],
        "yoy_gr_net_profit_parent": [12.0],
        "yoy_gr_gross_rev": [8.0],
    })
    flag = _pit_express(express, CAL, pd.Index(CODES))["express_reporting"]
    assert flag.notna().all().all()
    assert flag.loc[CAL[0], "000002.SZ"] == 0.0


# ---------------------------------------------------------------------------
# float_mktcap：水平量不能被比率裁剪界削平
# ---------------------------------------------------------------------------
def test_level_factor_clip_is_not_ratio_clip():
    from scripts.builders.build_alla_fundamental_factors import (
        CLIP_LEVEL,
        CLIP_RATIO,
        LEVEL_FACTORS,
    )

    assert "float_mktcap" in LEVEL_FACTORS
    assert CLIP_LEVEL[0] >= 0.0, "市值不可能为负"
    assert CLIP_LEVEL[1] > 1e12, "上界须高于 A 股单只流通市值量级（~1e12 元）"
    assert CLIP_RATIO == 1e4
    # 修复前的 bug 形态：水平量套用比率界 ⇒ 整列被钉在上限
    assert CLIP_LEVEL[1] != CLIP_RATIO


def test_float_mktcap_survives_level_clip():
    """量级 1e10 的市值面板经物理边界裁剪后仍有上千个截面唯一值。"""
    from scripts.builders.build_alla_fundamental_factors import CLIP_LEVEL

    rng = np.random.default_rng(0)
    cap = pd.DataFrame(
        10 ** rng.uniform(9, 13, size=(3, 500)),
        index=CAL[:3], columns=[f"{i:06d}.SZ" for i in range(500)],
    )
    clipped = cap.clip(*CLIP_LEVEL)
    assert (clipped.to_numpy() == cap.to_numpy()).all(), "正常量级不应被裁剪"
    assert np.median([np.unique(r).size for r in clipped.to_numpy()]) > 400
