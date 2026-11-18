"""因子检验统一入口的等价性与判据测试（2026-11-18）。

锁四件事：

1. ``gate_core.resid_rank`` 与 ``residual_ic_probe.resid_rank`` **逐位一致** ——
   新入口不能是"第二套口径"；
2. ``gate_core.ic_calibers`` 的四口径 IC 与 ``residual_ic_probe.one_factor`` 逐位一致；
3. ``cyclic_placebo`` 可复现（同 seed+name 同结果）且**只打乱时序位置、不改分布**；
4. 判定分支：符号翻转 / 被吸收 / 效应量不足 三条否证路径都能触发。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluation import gate_core as gc
from scripts.evaluation.residual_ic_probe import one_factor, resid_rank


def _panel(rng, n_days=1200, n_codes=40):
    """1200 日足够让循环平移 ``[120,900)`` 不绕回原位（见 cyclic_placebo 的守卫）。"""
    idx = pd.date_range("2019-01-02", periods=n_days, freq="B")
    cols = [f"S{i:03d}" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(size=(n_days, n_codes)), index=idx,
                        columns=cols)


@pytest.fixture(scope="module")
def synthetic():
    """因子 / 收益 / 主信号三件套：因子 = 0.5·主信号 + 噪声，保证有共同成分。"""
    rng = np.random.default_rng(20261118)
    main = _panel(rng)
    noise = _panel(rng)
    factor = 0.5 * main.rank(axis=1, pct=True) + noise
    fwd = 0.6 * main + _panel(rng)
    mask = pd.DataFrame(True, index=fwd.index, columns=fwd.columns)
    mask.iloc[:5, :3] = False
    return factor, fwd, mask, main


def test_resid_rank_matches_residual_ic_probe(synthetic):
    """两个实现必须逐位相同（否则同一因子会给出两套残差 IC）。"""
    factor, _fwd, _mask, main = synthetic
    a = gc.resid_rank(factor.rank(axis=1), main.rank(axis=1))
    b = resid_rank(factor.rank(axis=1), main.rank(axis=1))
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=0.0, rtol=0.0,
                       equal_nan=True)


def test_ic_calibers_matches_residual_ic_probe(synthetic):
    """四口径 IC + NW t 与旧探针逐位一致。"""
    factor, fwd, mask, main = synthetic
    main_rank = main.rank(axis=1, pct=True)
    ref = one_factor(factor, fwd, mask, main_rank, main)
    got = gc.ic_calibers(factor, fwd, mask, main)
    assert ref is not None and got is not None
    for tag in ("raw", "resfac", "restgt", "both"):
        assert got[f"{tag}_ic"] == pytest.approx(ref[f"{tag}_ic"], abs=1e-12)
        assert got[f"{tag}_nw_t"] == pytest.approx(ref[f"{tag}_nw_t"], abs=1e-9)
        assert got[f"{tag}_n"] == ref[f"{tag}_n"]
    assert got["rank_corr_main"] == pytest.approx(ref["rank_corr_main"], abs=1e-12)


def test_cyclic_placebo_is_reproducible_and_order_preserving(synthetic):
    """同 seed+name ⇒ 同结果；跨进程稳定（用 crc32 而非 hash）；分布不变。"""
    factor, *_ = synthetic
    a = gc.cyclic_placebo(factor, seed=7, name="f1")
    b = gc.cyclic_placebo(factor, seed=7, name="f1")
    c = gc.cyclic_placebo(factor, seed=7, name="f2")
    assert a.equals(b), "同一 (seed, name) 必须复现"
    assert not a.equals(c), "不同因子名应得到不同平移量"

    # 逐列排序后与原值相同（float32 往返，容差 1e-6）⇒ 只换了位置，没改分布
    for col in factor.columns:
        x, y = factor[col].to_numpy(), a[col].to_numpy()
        ok = np.isfinite(x)
        np.testing.assert_allclose(np.sort(x[ok].astype(np.float32)),
                                   np.sort(y[ok]), rtol=0, atol=1e-6)
    # 且确实动了（不是恒等）
    assert not a.equals(factor)


def test_cyclic_placebo_rejects_too_short_series():
    """序列短于 2×最小平移量时必须报错 —— 否则 roll 绕回原位，安慰剂等于原因子。"""
    rng = np.random.default_rng(1)
    short = pd.DataFrame(rng.normal(size=(100, 5)),
                         index=pd.date_range("2022-01-03", periods=100, freq="B"),
                         columns=list("ABCDE"))
    with pytest.raises(ValueError, match="太短"):
        gc.cyclic_placebo(short, seed=7, name="f")


def test_verdict_sign_flip_is_rejected():
    """raw 与残差反号 ⇒ SIGNFLIP（cyq_trap 的真实形态）。"""
    tag, why = gc.verdict({"cov": 0.7, "n_valid": 1e7, "raw_ic": 0.009,
                           "resfac_ic": -0.0075, "plc_ic": -0.0028,
                           "eff_ratio": 2.7, "eff_gap": -0.0048})
    assert tag == "SIGNFLIP", why


def test_verdict_absorbs_and_noise():
    """被主信号吸收 / 与安慰剂不可分 都要能触发。"""
    absorbed, _ = gc.verdict({"cov": 0.7, "n_valid": 1e7, "raw_ic": 0.02,
                              "resfac_ic": 0.0004})
    assert absorbed == "ABSORBED"

    noise, _ = gc.verdict({"cov": 0.7, "n_valid": 1e7, "raw_ic": 0.02,
                           "resfac_ic": 0.006, "plc_ic": 0.005,
                           "eff_ratio": 1.2, "eff_gap": 0.001})
    assert noise in ("NOISE", "WEAK")


def test_verdict_pass_requires_placebo_gap():
    """过筛需效应量比与绝对差**同时**达标（此时 raw 与残差同向）。"""
    tag, why = gc.verdict({"cov": 0.7, "n_valid": 1e7, "raw_ic": 0.017,
                           "resfac_ic": 0.007, "plc_ic": 0.0015,
                           "eff_ratio": 4.6, "eff_gap": 0.0055})
    assert tag == "PASS", why

    # 同向但差不够 ⇒ 仍然否
    weak, _ = gc.verdict({"cov": 0.7, "n_valid": 1e7, "raw_ic": 0.017,
                          "resfac_ic": 0.006, "plc_ic": 0.0055,
                          "eff_ratio": 1.1, "eff_gap": 0.0005})
    assert weak in ("NOISE", "WEAK")
