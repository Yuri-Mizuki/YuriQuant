"""论文复现因子族（factor/paper_factors.py）单测。

覆盖四层：
1. 语义锚点：构造已知路径手算参考值（动量/反转/FSCORE/应计/低波/突破）；
2. 全量可计算性：21 个因子形状对齐、无 ±inf（NaN 允许）；
3. 无未来函数：跳变日只影响当日及之后的值；
4. 入库 roundtrip：FactorLibrary.register 消费 paper 面板产出 IC。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.paper_factors import (
    PAPER_FACTORS,
    PaperData,
    compute_paper_factors,
    pap_accruals_bs,
    pap_breakout_atr,
    pap_fscore,
    pap_lowvol_1y,
    pap_mom_12_1,
    pap_rev_5d,
)

DAYS, N = 1000, 40
DATES = pd.bdate_range("2022-01-03", periods=DAYS)
COLS = [f"c{i:03d}" for i in range(N)]


def _flat(value: float) -> pd.DataFrame:
    return pd.DataFrame(value, index=DATES, columns=COLS)


@pytest.fixture(scope="module")
def paper_env() -> dict:
    rng = np.random.default_rng(11)
    rets = pd.DataFrame(rng.normal(0.0003, 0.015, (DAYS, N)), DATES, COLS)
    close = (1 + rets.fillna(0)).cumprod() * 20.0
    high = close * (1 + np.abs(rng.normal(0, 0.005, (DAYS, N))))
    low = close * (1 - np.abs(rng.normal(0, 0.005, (DAYS, N))))
    volume = pd.DataFrame(rng.lognormal(10, 0.5, (DAYS, N)), DATES, COLS)
    mktcap = pd.DataFrame(rng.uniform(5e9, 5e10, (1, N)), DATES[:1], COLS)
    mktcap = pd.DataFrame(np.repeat(mktcap.values, DAYS, axis=0), DATES, COLS)

    # 财务 PIT 面板：常数水平（同比/水平条件可解析预期）
    pit = {
        "NET_PRO_TTM": _flat(100.0),
        "CFO_TTM": _flat(150.0),
        "OPERA_REV_TTM": _flat(1000.0),
        "LESS_OPERA_COST_TTM": _flat(600.0),
        "EBIT_TTM": _flat(120.0),
        "EBITDA_TTM": _flat(180.0),
        "TOTAL_ASSETS": _flat(2000.0),
        "TOTAL_LIAB": _flat(900.0),
        "TOTAL_CUR_ASSETS": _flat(800.0),
        "TOTAL_CUR_LIAB": _flat(400.0),
        "CURRENCY_CAP": _flat(300.0),
        "ST_BORROWING": _flat(100.0),
        "LT_LOAN": _flat(200.0),
        "EQUITY": _flat(1100.0),
        "TOT_SHARE": _flat(1e9),
        "RD_EXP": _flat(50.0),
    }
    ann_flag = pd.DataFrame(False, DATES, COLS)
    ann_flag.iloc[::63] = True   # 约每季度一个公告日

    industry = pd.Series({c: (f"ind{i % 5}") for i, c in enumerate(COLS)})
    data = PaperData(close=close, high=high, low=low, volume=volume,
                     market_cap=mktcap, industry=industry,
                     pit=pit, ann_flag=ann_flag)
    return {"data": data, "close": close, "rets": rets}


# ---------------------------------------------------------------------------
# 1. 语义锚点（小面板手算）
# ---------------------------------------------------------------------------
def _tiny(values: list[float]) -> pd.DataFrame:
    """单股票小面板：index=0..len-1。"""
    idx = pd.bdate_range("2024-01-02", periods=len(values))
    return pd.DataFrame({"s0": values}, index=idx)


def test_pap_rev_5d_hand_computed():
    c = _tiny([100, 100, 100, 100, 100, 110, 121])
    d = PaperData(close=c)
    f = pap_rev_5d(d)
    expect = -(121 / 100 - 1)
    assert f["s0"].iloc[-1] == pytest.approx(expect)


def test_pap_mom_12_1_uses_skip_month_window():
    # 12-1 动量 = c[t-21]/c[t-252]-1：最近 1 个月不进窗口。
    # 构造 100(×260) → 150(×40) → 200(×1)：跳变 150 在窗口内，末跳 200 不在。
    vals = [100.0] * 260 + [150.0] * 40 + [200.0]
    c = _tiny(vals)
    d = PaperData(close=c)
    f = pap_mom_12_1(d)
    assert f["s0"].iloc[:252].isna().all()            # 预热期全 NaN
    assert f["s0"].iloc[252] == pytest.approx(0.0)    # 尚未进入 150 段
    # 末日：c[t-21]=150（在 150 段），c[t-252]=100 → 0.5
    assert f["s0"].iloc[-1] == pytest.approx(0.5)
    # 150 段进入 c[t-21] 窗口的首日：t=281（t-21=260）
    assert f["s0"].iloc[281] == pytest.approx(0.5)
    assert f["s0"].iloc[280] == pytest.approx(0.0)   # 尚差一天


def test_pap_lowvol_constant_price_is_zero():
    c = _flat(50.0)
    f = pap_lowvol_1y(PaperData(close=c))
    assert f.fillna(0.0).values.max() == pytest.approx(0.0)
    assert f.fillna(0.0).values.min() == pytest.approx(0.0)


def test_pap_fscore_anchor_all_constant_pit():
    """常数 PIT 下：ROA>0、CFO>0、CFO>净利、股本未增发 4 项成立，其余同比为 0。

    score = 1+1+0+1+0+0+1+0+0 = 4（有同比数据的区间）。
    """
    pit = {k: _flat(v) for k, v in {
        "NET_PRO_TTM": 100.0, "CFO_TTM": 150.0, "TOTAL_ASSETS": 2000.0,
        "LT_LOAN": 200.0, "TOTAL_CUR_ASSETS": 800.0, "TOTAL_CUR_LIAB": 400.0,
        "TOT_SHARE": 1e9, "OPERA_REV_TTM": 1000.0, "LESS_OPERA_COST_TTM": 600.0,
    }.items()}
    data = PaperData(close=_flat(50.0), pit=pit)
    f = pap_fscore(data)
    valid = f.dropna(how="all")
    assert not valid.empty
    assert (valid == 4.0).all().all()


def test_pap_accruals_bs_anchor():
    """Δ 全为 0（常数面板）→ ACC = -Dep/mean(TA)，factor = -ACC = Dep/TA。"""
    pit = {k: _flat(v) for k, v in {
        "TOTAL_CUR_ASSETS": 800.0, "CURRENCY_CAP": 300.0,
        "TOTAL_CUR_LIAB": 400.0, "ST_BORROWING": 100.0,
        "TOTAL_ASSETS": 2000.0, "EBIT_TTM": 120.0, "EBITDA_TTM": 180.0,
    }.items()}
    f = pap_accruals_bs(PaperData(close=_flat(50.0), pit=pit))
    expect = (180.0 - 120.0) / 2000.0
    valid = f.dropna(how="all")
    assert not valid.empty
    assert np.allclose(valid.values, expect)


def test_pap_breakout_atr_enter_on_highs_exit_on_crash():
    # 持续新高 → 入场持有；末端暴跌 60% → 跌破吊灯止损离场
    up = list(np.linspace(10, 50, 200))
    dn = up + [up[-1] * 0.4]
    c = _tiny(dn)
    d = PaperData(close=c)
    f = pap_breakout_atr(d)
    assert f["s0"].iloc[195] == 1.0       # 上升段在场
    assert f["s0"].iloc[-1] == 0.0        # 暴跌后离场


# ---------------------------------------------------------------------------
# 2. 全量可计算性
# ---------------------------------------------------------------------------
def test_all_paper_factors_computable(paper_env):
    panels = compute_paper_factors(paper_env["data"])
    missing = set(PAPER_FACTORS) - set(panels)
    assert not missing, f"fixture 下应有全部因子可算，缺: {missing}"
    shape = paper_env["close"].shape
    for name, pnl in panels.items():
        assert pnl.shape == shape, name
        vals = pnl.values
        assert np.isfinite(vals[~np.isnan(vals)]).all(), f"{name} 含 ±inf"


# ---------------------------------------------------------------------------
# 3. 无未来函数（跳变日）
# ---------------------------------------------------------------------------
def test_no_lookahead_on_jump_day(paper_env):
    data = paper_env["data"]
    close = data.close.copy()
    f_before = pap_rev_5d(PaperData(close=close))
    close.iloc[-1] = close.iloc[-1] * 1.5   # 末日跳变
    f_after = pap_rev_5d(PaperData(close=close))
    # 倒数第二日及之前完全不变
    pd.testing.assert_frame_equal(f_before.iloc[:-1], f_after.iloc[:-1])


# ---------------------------------------------------------------------------
# 4. 入库 roundtrip（tmp_path 隔离）
# ---------------------------------------------------------------------------
def test_register_roundtrip(paper_env, tmp_path):
    from research.factor_library import FactorLibrary

    panels = compute_paper_factors(paper_env["data"])
    rets = paper_env["rets"]
    lib = FactorLibrary(root=tmp_path / "flib")
    row = lib.register("pap_rev_5d", panels["pap_rev_5d"], rets,
                       kind="raw", formula="-(close/close.shift(5)-1)")
    assert lib.has("pap_rev_5d")
    assert "ic_mean" in row
    got = lib.get_panel("pap_rev_5d")
    assert got.shape == panels["pap_rev_5d"].shape
