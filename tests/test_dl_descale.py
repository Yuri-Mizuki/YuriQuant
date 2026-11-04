"""去量纲口径（`replica_deep.descale_features`）单元测试
=========================================================

只测变换的**数学恒等式**，不碰行情文件：给一组已知的 open/high/low/close/avg，
按「÷当日收盘」拼出 raw 的 6 列，再验证每种口径变换后等于直接按该基准价算出来的比值。

这一步是 2026-11-01 新增 `--descale {prev,avg,zscore}` 的安全网：如果恒等式写错，
特征会静默变成「另一种东西」，而训练照样收敛 —— 只能靠测试挡住。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.model_train.replica_deep import DESCALES, descale_features, out_name


def _raw_bundle(n_days: int = 30, n_codes: int = 7, seed: int = 0):
    """由价格面板直接造 raw（÷close 口径），返回 (raw, 价格字典)。"""
    rng = np.random.default_rng(seed)
    close = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_codes)), axis=0))
    prev = np.vstack([close[:1], close[:-1]])
    opn = close * (1 + rng.normal(0, 0.01, close.shape))
    high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.01, close.shape)))
    low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.01, close.shape)))
    avg = close * (1 + rng.normal(0, 0.005, close.shape))
    turn = rng.uniform(0.001, 0.05, close.shape)
    raw = np.stack([opn / close, high / close, low / close,
                    close / prev - 1.0, avg / close, turn], axis=-1).astype(np.float32)
    px = {"open": opn, "high": high, "low": low, "close": close,
          "prev": prev, "avg": avg}
    return raw, px


def test_close_is_identity():
    raw, _ = _raw_bundle()
    assert np.array_equal(descale_features(raw, "close"), raw)


def test_prev_basis_equals_direct_ratio():
    raw, px = _raw_bundle()
    out = descale_features(raw, "prev")
    for k, kk in enumerate(("open", "high", "low")):
        np.testing.assert_allclose(out[..., k], px[kk] / px["prev"], rtol=1e-5)
    # 4 号列：vwap/prev；3 号列（收益）与 5 号列（换手）按设计不动
    np.testing.assert_allclose(out[..., 4], px["avg"] / px["prev"], rtol=1e-5)
    np.testing.assert_allclose(out[..., 3], raw[..., 3], rtol=1e-6)
    np.testing.assert_allclose(out[..., 5], raw[..., 5], rtol=1e-6)


def test_avg_basis_equals_direct_ratio():
    raw, px = _raw_bundle()
    out = descale_features(raw, "avg")
    for k, kk in enumerate(("open", "high", "low")):
        np.testing.assert_allclose(out[..., k], px[kk] / px["avg"], rtol=1e-4)
    # 第 5 个特征在 ÷均价 口径下改为 close/avg（原 vwap/close 的倒数）
    np.testing.assert_allclose(out[..., 4], px["close"] / px["avg"], rtol=1e-4)


def test_zscore_uses_only_past_window():
    """时序 z-score 不能看到未来：改最后一天的价格，不应改变更早日的输出。"""
    raw, _ = _raw_bundle(n_days=200)
    out = descale_features(raw, "zscore")
    raw2 = raw.copy()
    raw2[-1, :, 0] *= 3.0                       # 只改最后一天的 open/close
    out2 = descale_features(raw2, "zscore")
    np.testing.assert_allclose(out[:-1, :, 0], out2[:-1, :, 0], rtol=1e-6)
    assert not np.allclose(out[-1, :, 0], out2[-1, :, 0])


def test_zscore_has_no_level_leak_after_nan():
    raw, _ = _raw_bundle(n_days=200)
    raw[50:60, :, 4] = np.nan                   # 中间一段均价缺失
    out = descale_features(raw, "zscore")
    assert np.isfinite(out[150:160, :, 4]).any()   # 缺失段之后能恢复
    assert np.isnan(out[50:60, :, 4]).all()


def test_unknown_descala_raises():
    raw, _ = _raw_bundle(n_days=5, n_codes=2)
    for bad in ("close2", "PREV", ""):
        try:
            descale_features(raw, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} 应抛 ValueError")


def test_descales_registry():
    assert DESCALES == ("close", "prev", "avg", "zscore")


def test_out_name_suffixes():
    assert out_name("TCN_4_day60_10d") == "TCN_4_day60_10d"
    assert out_name("TCN_4_day60_10d", "prev") == "TCN_4_day60_10d@prev"
    assert out_name("TCN_4_day60_10d", tag="step1") == "TCN_4_day60_10d__step1"
    assert out_name("TCN_4_day60_10d", "prev", "step1") == "TCN_4_day60_10d@prev__step1"


def test_zscore_output_shape_dtype():
    raw, _ = _raw_bundle(n_days=120, n_codes=5)
    out = descale_features(raw, "zscore")
    assert out.shape == raw.shape and out.dtype == np.float32
    assert pd.isna(out).sum() > 0            # 前 TS_MIN_PERIODS 天必然是 NaN
