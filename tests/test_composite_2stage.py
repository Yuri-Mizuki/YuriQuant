"""`composite_2stage`（锚因子切 Q1 → Q1 内二级排序）的门控回归测试。

钉住两个**静默给出错数**的坑，都是实测踩过的：

1. **`cnt >= 2` 写死**：只有 1 个二级因子时 `cnt` 恒为 1，整片被置 NaN，
   随后 ``sec.where(sec.notna(), a.where(gate))`` 静默回退到锚因子本身
   ⇒ 该臂「跑了但等于没跑」。实测 9 个 ``gate+单因子`` 臂与 ``gate_only``
   逐位相同（毛收益都是 46.94%）才发现。
2. **量纲混用**：``sec`` 是 rank pct（0~1），回退值若用原始因子值（±1e-2），
   混在一张面板里排序会把「二级因子缺失的股票」系统性顶到最前/最后。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.evaluation.pit_intraday_probe import composite_2stage


def _panels(days: int = 40, n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-04", periods=days, freq="B")
    cols = [f"S{i:03d}" for i in range(n)]
    anchor = pd.DataFrame(rng.standard_normal((days, n)), index=idx, columns=cols)
    others = {
        "b": pd.DataFrame(rng.standard_normal((days, n)), index=idx, columns=cols),
        "c": pd.DataFrame(rng.standard_normal((days, n)), index=idx, columns=cols),
    }
    return anchor, others


class TestSingleSecondStage:
    def test_single_factor_changes_result(self):
        """🚨 核心回归：1 个二级因子必须真的改变排序。"""
        anchor, others = _panels()
        feats = {"a": anchor, **others}
        ics = {"a": 1.0, "b": 1.0, "c": 1.0}
        base = composite_2stage("a", [], feats, ics, q=0.1)
        one = composite_2stage("a", ["b"], feats, ics, q=0.1)
        assert not base.equals(one), (
            "只有 1 个二级因子时结果与 gate_only 相同 ⇒ cnt>=2 门控把二级排序"
            "整片丢了（静默回退到锚因子）")

    def test_single_vs_two_both_work(self):
        anchor, others = _panels()
        feats = {"a": anchor, **others}
        ics = {"a": 1.0, "b": 1.0, "c": 1.0}
        one = composite_2stage("a", ["b"], feats, ics, q=0.1)
        two = composite_2stage("a", ["b", "c"], feats, ics, q=0.1)
        assert one.notna().any().any()
        assert two.notna().any().any()


class TestGateSemantics:
    def test_gate_keeps_high_quantile(self):
        """定向后「越大越好」⇒ gate 必须保留高分位那端。

        🚨 按**秩集合**断言，不要用 ``quantile(0.9)`` 比值：``rank(pct=True) >= 0.9``
        与 ``quantile(0.9)`` 的线性插值口径不同，边界上会差一个样本（假失败）。
        """
        anchor, _ = _panels()
        out = composite_2stage("a", [], {"a": anchor}, {"a": 1.0}, q=0.1)
        for dt in anchor.index[:5]:
            kept = set(out.loc[dt].dropna().index)
            r = anchor.loc[dt].rank(pct=True)
            expect = set(r[r >= 1.0 - 0.1].index)
            assert kept == expect

    def test_negative_ic_flips_gate(self):
        """IC<0（反转因子）定向后，gate 保留的应是**原始值最小**的那端。

        ⚠️ 不追求与 ``rank <= 0.1`` 的集合精确相等：取负后 pct 恰有样本落在
        0.9 边界上（``54/60 == 0.9``），正反两侧会差 1 个，属浮点边界而非语义错。
        断言实质语义：保留池全在下半区、数量约为 q·n。
        """
        anchor, _ = _panels()
        n = anchor.shape[1]
        out = composite_2stage("a", [], {"a": anchor}, {"a": -1.0}, q=0.1)
        for dt in anchor.index[:5]:
            kept = list(out.loc[dt].dropna().index)
            med = float(anchor.loc[dt].median())
            assert all(v < med for v in anchor.loc[dt][kept]), "反转因子取到了高分位端"
            assert abs(len(kept) - 0.1 * n) <= 2


class TestScaleConsistency:
    def test_fallback_value_same_scale_as_rank(self):
        """二级因子缺失的股票，回退值必须与 rank pct 同量纲（都 ∈ (0,1]）。

        若回退用原始因子值（量级 ±1e-2），与 rank pct 混排会让缺失股票
        被系统性顶到最前/最后 —— 而 `run()` 只按值排序，不会察觉。
        """
        anchor, others = _panels()
        b = others["b"].copy()
        b.iloc[:, :30] = np.nan  # 一半股票没有二级因子
        feats = {"a": anchor, "b": b}
        out = composite_2stage("a", ["b"], feats, {"a": 1.0, "b": 1.0}, q=0.1)
        v = out.to_numpy()
        v = v[np.isfinite(v)]
        assert v.max() <= 1.0 + 1e-12 and v.min() > 0.0, (
            f"取值范围 [{v.min():.4g}, {v.max():.4g}] 不是 rank pct 量纲")


class TestEmptyPool:
    def test_no_second_stage_returns_gate(self):
        anchor, _ = _panels()
        out = composite_2stage("a", [], {"a": anchor}, {"a": 1.0}, q=0.2)
        expect_cnt = int(np.ceil(0.2 * anchor.shape[1]))
        got = out.notna().sum(axis=1)
        assert (got >= expect_cnt - 2).all() and (got <= expect_cnt + 2).all()