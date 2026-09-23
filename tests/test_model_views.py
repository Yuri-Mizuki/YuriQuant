"""model/views.py 预测层观点注入测试（AI43 等价实现）。

覆盖：
- 特征复制注入：k 语义（副本数）、缺因子报错、副本引用同一面板
- 两段模型：分组预测可达、观点分组与基线模型不同（注入非空转）、
  top_frac 校验、观点因子缺失报错
- 注入有效性：植入信号场景下两段模型 OOS 预测与单段模型不同
  （结构改变可观测），且自身 IC 为正（等价实现没把信号弄丢）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model.predictor import RidgePredictor, fit_predict_oos
from model.views import TwoStagePredictor, inject_by_feature_duplication


def _market(n_days: int = 300, n_codes: int = 30, seed: int = 7):
    """植入两个信号：sig（全因子信号）+ view（观点信号，独立来源）。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-06-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    def ar1():
        s = np.zeros((n_days, n_codes))
        s[0] = rng.normal(0, 1, n_codes)
        for t in range(1, n_days):
            s[t] = 0.9 * s[t - 1] + rng.normal(0, np.sqrt(1 - 0.81), n_codes)
        return pd.DataFrame(s, idx, codes)

    sig, view = ar1(), ar1()
    ret = 0.003 * sig + 0.002 * view + rng.normal(0, 0.015, (n_days, n_codes))
    close = pd.DataFrame(100.0 * np.cumprod(1 + ret, axis=0), idx, codes)
    return close, sig, view


@pytest.fixture(scope="module")
def market():
    close, sig, view = _market()
    from factor.preprocessing import standardize_zscore
    labels, _ = __import__("model.labels", fromlist=["build_labels"]).build_labels(
        close, horizon=5, mode="rank")
    feats = {
        "sig": standardize_zscore(sig),
        "noise": standardize_zscore(
            pd.DataFrame(np.random.default_rng(3).normal(
                0, 1, sig.shape), sig.index, sig.columns)),
        "view": standardize_zscore(view),
    }
    return feats, labels


# ===========================================================================
# 特征复制注入
# ===========================================================================
class TestFeatureDuplication:
    def test_k_semantics(self, market):
        feats, _ = market
        out1 = inject_by_feature_duplication(feats, ["view"], k=1)
        assert set(out1) == set(feats)           # k=1 无注入
        out2 = inject_by_feature_duplication(feats, ["view"], k=2)
        assert "view__dup2" in out2 and set(out2) == set(feats) | {"view__dup2"}
        out3 = inject_by_feature_duplication(feats, ["view"], k=3)
        assert {"view__dup2", "view__dup3"} <= set(out3)

    def test_missing_view_raises(self, market):
        feats, _ = market
        with pytest.raises(KeyError, match="不在特征集中"):
            inject_by_feature_duplication(feats, ["nope"], k=2)

    def test_invalid_k_raises(self, market):
        feats, _ = market
        with pytest.raises(ValueError, match="k 须"):
            inject_by_feature_duplication(feats, ["view"], k=0)

    def test_dup_is_same_object(self, market):
        """副本引用同一面板（不复制数据）——内存约定。"""
        feats, _ = market
        out = inject_by_feature_duplication(feats, ["view"], k=2)
        assert out["view__dup2"] is feats["view"]


# ===========================================================================
# 两段模型
# ===========================================================================
class TestTwoStage:
    def test_invalid_top_frac_raises(self, market):
        feats, labels = market
        with pytest.raises(ValueError, match="top_frac"):
            TwoStagePredictor(RidgePredictor, "view", top_frac=1.5)

    def test_missing_view_raises(self, market):
        feats, labels = market
        with pytest.raises(KeyError, match="不在特征集中"):
            TwoStagePredictor(RidgePredictor, "nope").fit(feats, labels)

    def test_two_stage_differs_from_single_stage(self, market):
        """注入有效性：同数据下两段模型 OOS 预测 ≠ 单段模型（结构改变可观测）。

        观点信号独立于全因子信号 → 分组后层内模型与混合模型的最优解不同，
        若两段实现退化为单段（空转），两组预测会逐位一致。
        """
        feats, labels = market
        cut = labels.index[len(labels) * 2 // 3]
        tr = labels.index[labels.index < cut]
        te = labels.index[labels.index >= cut]

        single = RidgePredictor(alpha=1.0).fit(
            {k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
        two = TwoStagePredictor(RidgePredictor, "view", top_frac=0.5,
                                alpha=1.0).fit(
            {k: v.loc[tr] for k, v in feats.items()}, labels.loc[tr])
        fte = {k: v.loc[te] for k, v in feats.items()}
        p1 = single.predict(fte)
        p2 = two.predict(fte)

        diff = (p1 - p2).abs()
        assert diff.stack().dropna().gt(1e-9).any()   # 存在实质差异
        # 两段模型自身 OOS IC 仍为正（等价实现没把可学信号弄丢）
        ok = p2.notna() & labels.loc[te].notna()
        ics = []
        for d in te:
            x, y = p2.loc[d], labels.loc[te].loc[d]
            m = ok.loc[d]
            if m.sum() >= 5:
                ics.append(x[m].corr(y[m], method="spearman"))
        assert float(np.nanmean(ics)) > 0.02

    def test_fit_predict_oos_integration(self, market):
        """与统一切分调度器协同：TwoStagePredictor 可作为 predictor_cls 传入。"""
        feats, labels = market
        out = fit_predict_oos(
            lambda **kw: TwoStagePredictor(RidgePredictor, "view",
                                           top_frac=0.5, alpha=1.0),
            feats, labels, n_splits=3, embargo_days=5, min_train_days=60)
        assert out.shape == labels.shape
        assert out.iloc[60:].notna().any().any()