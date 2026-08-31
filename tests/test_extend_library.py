"""extend_factor_library 单测：重叠区校验 + GP 公式因子延长回路。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.factor_extension import (
    extend_gp_factors, verify_overlap, warmup_begin,
)


def _gp_reg_row(name: str, formula: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "name": name, "kind": "raw", "formula": formula,
        "source": "gp:mine_factors", "family": "", "frequency": "",
        "maturity": "experimental", "note": "",
    }])


def test_warmup_begin():
    assert warmup_begin(20220101) == 20200101
    assert warmup_begin(20230601) == 20210601
    assert warmup_begin(20200101) == 20190102  # 不早于缓存起点


def test_verify_overlap():
    idx = pd.bdate_range("2024-01-01", periods=50)
    cols = ["a", "b"]
    base = pd.DataFrame(np.arange(100, dtype=float).reshape(50, 2),
                        index=idx, columns=cols)
    v = verify_overlap(base, base.copy())
    assert v["n"] == 100 and v["norm"] == 0.0

    pert = base.copy()
    pert.iloc[10, 0] += 5.0
    v2 = verify_overlap(pert, base)
    assert v2["n"] == 100 and v2["norm"] > 0.1

    shifted = base.copy()
    shifted.index = base.index + pd.Timedelta(days=400)
    v3 = verify_overlap(shifted, base)
    assert v3["n"] == 0


def test_extend_gp_factors_roundtrip(tmp_path):
    from factor.formula import formula_builder
    from research.factor_library import FactorLibrary

    rng = np.random.default_rng(0)
    days = 260
    idx = pd.bdate_range("2024-01-01", periods=days)
    cols = [f"c{i}" for i in range(8)]
    amount = pd.DataFrame(rng.lognormal(18, 0.4, (days, 8)), index=idx, columns=cols)
    close = pd.DataFrame(
        50 * np.exp(np.cumsum(rng.normal(0, 0.02, (days, 8)), axis=0)),
        index=idx, columns=cols)
    panel = {"close": close, "amount": amount}
    rets = close.pct_change().shift(-1)
    full = formula_builder("cs_rank(amount)", features=list(panel))(panel)

    lib = FactorLibrary(root=tmp_path / "fl", dataset="ext")
    old_panel = full.iloc[:200]
    lib.register("cs_rank(amount)", old_panel, rets.iloc[:200],
                 kind="raw", formula="cs_rank(amount)", source="gp:mine_factors")
    row = _gp_reg_row("cs_rank(amount)", "cs_rank(amount)")

    # 延长：重叠区一致 → 覆盖入库到 end
    ok, skipped, failed = extend_gp_factors(
        lib, row, panel, rets, 20240101, 20241231)
    assert ok == ["cs_rank(amount)"] and skipped == [] and failed == []
    p = lib.get_panel("cs_rank(amount)")
    expect_len = len(full.loc["2024-01-01":"2024-12-31"])
    assert len(p) == expect_len and expect_len > 240
    v = verify_overlap(full.loc[p.index], p)
    assert v["norm"] < 1e-12  # 重叠区历史保持不变

    # verify-only 不动库
    ok2, _, _ = extend_gp_factors(
        lib, row, panel, rets, 20240101, 20241231, verify_only=True)
    assert ok2 == ["cs_rank(amount)"]
    assert len(lib.get_panel("cs_rank(amount)")) == expect_len

    # 口径不一致（改变截面排序）→ 跳过，面板保持不动
    bad_amount = amount.copy()
    bad_amount.iloc[:, 0] = bad_amount.iloc[:, 0] * 0.001
    bad = {"close": close, "amount": bad_amount}
    ok3, skipped3, _ = extend_gp_factors(
        lib, row, bad, rets, 20240101, 20241231)
    assert ok3 == [] and skipped3 == ["cs_rank(amount)"]
    p3 = lib.get_panel("cs_rank(amount)")
    assert len(p3) == expect_len
    assert verify_overlap(p3, full.loc[p3.index])["norm"] < 1e-12
