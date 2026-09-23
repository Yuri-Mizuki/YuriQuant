"""monitoring.production_ic 的 characterization 测试（Task #8 漂移监控升格）。

重点钉死 ``signal_mktcap_spearman`` 列：
- 数学性质：daily_rank_ic 对两个已 rank 面板即逐日截面 Spearman；
- 与手写逐日 Series.corr(method="spearman") 实现逐位等价（oneoff 版同型）；
- compute_metrics 端到端含新列、NaN 日期、summarize 摘要键。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring.production_ic import (
    MIN_SAMPLES,
    compute_metrics,
    daily_rank_ic,
    summarize,
)


def _panel(days: int = 40, codes: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, (days, codes)),
        index=pd.bdate_range("2025-01-01", periods=days),
        columns=[f"c{i:03d}" for i in range(codes)],
    )


def test_daily_rank_ic_equals_manual_spearman():
    """与手写逐日 Series.rank().corr() 实现逐位对照（oneoff 版同型算法）。"""
    score = _panel()
    mkt = _panel(days=40, codes=60, seed=99)
    got = daily_rank_ic(score, mkt)

    manual = {}
    for d in score.index:
        s, m = score.loc[d], mkt.loc[d]
        ok = s.notna() & m.notna()
        if ok.sum() < MIN_SAMPLES:
            continue
        manual[d] = s[ok].rank().corr(m[ok].rank())
    manual = pd.Series(manual)

    common = got.dropna().index.intersection(manual.dropna().index)
    assert len(common) == len(manual)
    assert np.allclose(got.reindex(common), manual.reindex(common), atol=1e-12)


def test_daily_rank_ic_is_rank_invariant():
    """Spearman 对单调变换不变：raw 市值与 log 市值给出的相关逐位相等。"""
    score = _panel()
    mkt = _panel(days=40, codes=60, seed=5).abs() + 1.0
    a = daily_rank_ic(score, mkt)
    b = daily_rank_ic(score, np.log(mkt))
    assert np.allclose(a.dropna(), b.dropna(), atol=1e-12)


def test_compute_metrics_has_drift_column_and_summarize_keys():
    """端到端：compute_metrics 产出含 signal_mktcap_spearman 列，summarize 带漂移键。"""
    score = _panel(days=60, codes=80)
    close = (1 + score * 0).add(100) + _panel(days=60, codes=80, seed=3).cumsum() * 0.01
    cov = {"size": _panel(days=60, codes=80, seed=11).abs() + 1.0}

    out = compute_metrics(score, close, cov, out_path=None)
    assert "signal_mktcap_spearman" in out.columns
    col = out["signal_mktcap_spearman"].dropna()
    assert len(col) > 0
    assert (col.abs() <= 1).all()

    s = summarize(out, window=20)
    assert "mktcap_drift_full" in s and "mktcap_drift_recent" in s
    assert isinstance(s["mktcap_drift_yearly"], dict) and s["mktcap_drift_yearly"]


def test_compute_metrics_missing_size_gives_nan_column():
    """缺 cov['size'] 时新列全 NaN 且不炸（向后兼容旧 _base 目录）。"""
    score = _panel(days=30, codes=80)
    close = (1 + score * 0).add(100) + _panel(days=30, codes=80, seed=3).cumsum() * 0.01

    out = compute_metrics(score, close, {}, out_path=None)
    assert "signal_mktcap_spearman" in out.columns
    assert out["signal_mktcap_spearman"].isna().all()
