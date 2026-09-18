"""多因子10 T 扫描冒烟（合成面板，不触真实数据/因子库）。

覆盖：
- 逐 T 一行；历史不足的大 T 如实记 ``valid=False``（不报错、不编造数字）；
- 训练窗切分正确（窗口长度 / horizon 截断 / 不越过 test_begin）；
- ic_ir_max 缺省 w≥0（long_only 截断口径）；
- **防未来函数**：测试段收益整体缩放 100 倍不改变任何一档 T 的权重
  （权重只允许由训练段决定）；
- ic_max 分支可跑、测试段评估数字有限。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from factor.synthesis import CompositeInput
from scripts.evaluation.mf10_t_scan import _usable_train_dates, t_scan_composite


def _synthetic(n_days: int = 900, n_codes: int = 30, seed: int = 11):
    """三个因子：f1 有预测力、f2 = f1+噪声（高相关）、f3 纯噪声。

    returns_panel[d] = d→d+1 的"未来收益"，由 f1/f2 同日值驱动（h=1 语义，
    测试里 horizon 只用于训练窗截断，不影响这个构造的自洽性）。
    """
    from factor.synthesis import standardize_zscore

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    f1 = pd.DataFrame(rng.normal(0, 1.0, (n_days, n_codes)), idx, codes)
    f2 = f1 + pd.DataFrame(rng.normal(0, 0.6, (n_days, n_codes)), idx, codes)
    f3 = pd.DataFrame(rng.normal(0, 1.0, (n_days, n_codes)), idx, codes)
    rets = (0.02 * f1 + 0.01 * f2
            + pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes))
    panels = {n: standardize_zscore(p)
              for n, p in (("f1", f1), ("f2", f2), ("f3", f3))}
    comps = [CompositeInput(name=n, panel=p, ic=1.0) for n, p in panels.items()]
    return comps, rets


def _run(comps, rets, **kw):
    kw.setdefault("test_begin", "2023-11-01")
    kw.setdefault("t_grid", (3, 6, 12, 36))
    kw.setdefault("horizon", 5)
    return t_scan_composite(comps, rets, **kw)


def test_grid_rows_and_insufficient_history():
    comps, rets = _synthetic()
    res = _run(comps, rets)
    assert list(res.index) == [3, 6, 12, 36]
    # 2021-01 起 900 个交易日：T=36 需要 756 天训练窗（再截 horizon）→ 不足
    assert bool(res.loc[36, "valid"]) is False
    assert res.loc[36, "n_train_days"] == 0
    assert res.loc[36, "oos_ic_mean"] is np.nan or pd.isna(res.loc[36, "oos_ic_mean"])
    for t in (3, 6, 12):
        assert bool(res.loc[t, "valid"]) is True


def test_train_window_slicing():
    comps, rets = _synthetic()
    res = _run(comps, rets)
    assert int(res.loc[3, "n_train_days"]) == 63        # 3 × 21
    assert int(res.loc[12, "n_train_days"]) == 252      # 12 × 21
    for t in (3, 6, 12):
        assert pd.Timestamp(res.loc[t, "train_end"]) < pd.Timestamp("2023-11-01")
        assert int(res.loc[t, "oos_n_days"]) > 100      # 测试段 ~ 200 日


def test_usable_train_dates_horizon_truncation():
    idx = pd.bdate_range("2024-01-01", periods=100)
    out = _usable_train_dates(idx, idx[90], horizon=5)
    assert len(out) == 85 and out[-1] == idx[84]        # 末端截掉标签实现期
    # horizon 大于可用历史 → 空
    assert len(_usable_train_dates(idx, idx[3], horizon=50)) == 0


def test_ic_ir_max_weights_long_only_and_parseable():
    comps, rets = _synthetic()
    res = _run(comps, rets)
    for t in (3, 6, 12):
        w = json.loads(res.loc[t, "weights"])
        assert set(w) == {"f1", "f2", "f3"}
        assert all(v >= -1e-12 for v in w.values())     # 研报 w≥0 截断口径
        assert np.isfinite(res.loc[t, "oos_ic_mean"])
        assert np.isfinite(res.loc[t, "oos_t_nw"])


def test_no_lookahead_test_segment_cannot_change_weights():
    """测试段收益行序打乱（日期错位）：任何一档 T 的权重都必须逐位不变。"""
    comps, rets = _synthetic()
    res = _run(comps, rets)
    rets2 = rets.copy()
    mask = rets2.index >= pd.Timestamp("2023-11-01")
    # 只打乱测试段"哪一天对应哪个截面"——合成权重不应受任何影响。
    # ⚠️ 必须转 numpy 再赋回：pandas 的 df 赋值按索引对齐，乱序 DataFrame
    # 赋回相同索引会静默还原成原样（第一次实现就栽在这里）。
    rets2.loc[mask] = rets2.loc[mask].sample(frac=1.0, axis=0,
                                             random_state=7).to_numpy()
    res2 = _run(comps, rets2)
    for t in (3, 6, 12):
        assert res.loc[t, "weights"] == res2.loc[t, "weights"]
    # 测试段评估数字则应当变化（错位生效的旁证；rank IC 对整体缩放免疫，
    # 所以旁证必须用错位而不是缩放）
    assert res.loc[3, "oos_ic_mean"] != pytest.approx(res2.loc[3, "oos_ic_mean"])


def test_ic_max_method_runs():
    comps, rets = _synthetic()
    res = _run(comps, rets, method="ic_max")
    for t in (3, 6, 12):
        assert bool(res.loc[t, "valid"]) is True
        w = json.loads(res.loc[t, "weights"])
        assert set(w) == {"f1", "f2", "f3"}             # ic_max 允许负权重


def test_empty_test_segment_raises():
    comps, rets = _synthetic()
    with pytest.raises(ValueError, match="测试段为空"):
        t_scan_composite(comps, rets, test_begin="2030-01-01")
