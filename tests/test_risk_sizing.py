"""风险档位旋钮：``gross_exposure`` 必须真的降敞口，且不能被 ``capital`` 冒充。

2026-11-13 P0：起因是要把「目标回撤 → 仓位」做成可执行流程，先验证两件事：
① ``solo_weights`` 的 ``gross_exposure`` 确实把权重和压到目标值；
② 减 ``capital`` **不等于**降仓（策略权重恒归一到满仓，减资金只改整手颗粒度）。
第 ② 条在回测里实测过：10 万→5 万，年化 24.67%/回撤 41.91%，不是真降仓的
12.58%/23.09%（t=−21）。这条测试钉住「权重和只由 gross 决定」这一半。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation.solo_live import lot_sizing, solo_weights


def _ranking(n: int = 30, n_sector: int = 12) -> pd.DataFrame:
    codes = [f"{i:06d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "score": np.linspace(1.0, 0.0, n),
            "industry_l1": [f"I{i % n_sector}" for i in range(n)],
            "tradable": True,
        },
        index=pd.Index(codes, name="code"),
    )


DATE = pd.Timestamp("2026-09-01")


@pytest.mark.parametrize("gross", [1.0, 0.85, 0.75, 0.5])
def test_gross_scales_weight_sum(gross):
    w = solo_weights(_ranking(), DATE, prev=None, gross_exposure=gross)
    assert w.sum() == pytest.approx(gross, abs=1e-12)
    # 只数不受影响：降仓不是「少买几只」
    assert (w > 0).sum() == 8


def test_default_is_full_position():
    """缺省必须等于登记表（1.0），否则实盘口径会被静默改掉。"""
    assert solo_weights(_ranking(), DATE).sum() == pytest.approx(1.0, abs=1e-12)


def test_capital_does_not_change_exposure():
    """减资金只改整手颗粒度，不降敞口 —— 与回测实测（t=−21）同一断言。"""
    w = solo_weights(_ranking(), DATE)
    px = pd.Series(20.0, index=w.index)
    big = lot_sizing(w, px, 100_000, lot=100)
    small = lot_sizing(w, px, 50_000, lot=100)
    assert big.attrs["gross_target"] == pytest.approx(1.0)
    assert small.attrs["gross_target"] == pytest.approx(1.0)
    assert small.attrs["invest"] > 0.9       # 5 万仍是「满仓」，只是每票整手更粗


def test_gross_survives_lot_rounding():
    """g=0.5 时实投应接近 0.5（整手只往下扣，不会超）。"""
    w = solo_weights(_ranking(), DATE, gross_exposure=0.5)
    px = pd.Series(20.0, index=w.index)
    s = lot_sizing(w, px, 100_000, lot=100)
    assert 0.4 <= s.attrs["invest"] <= 0.5
