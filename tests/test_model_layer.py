"""
模型层五组件单元测试（合成数据，不依赖 SDK）。

覆盖（对齐模型层蓝图 reports/yuriquant_model_layer_design）：
- ② LabelBuilder  : forward_returns 无未来函数 / rank-zscore-raw 三模式 / embargo
- ① FeatureStore  : 白黑名单 → 覆盖率 → 去冗余 → 截断 四级选择漏斗
- ③ Predictor     : ridge/gbdt fit-predict、截面标准化、fit_predict_oos 折纪律
- ④ Trainer       : train_predictor_model / train_and_register(kind=predictor)
- ⑤ Serving       : register_model_as_factor 血缘回写因子库
- 编排            : model.predictor.rolling_oos 的折边界 embargo 纪律
"""
import numpy as np
import pandas as pd
import pytest

from model.features import build_feature_set
from model.labels import build_labels, forward_returns
from model.predictor import (
    PREDICTORS,
    RidgePredictor,
    fit_predict_oos,
    rolling_oos,
)
from model.registry import ModelRegistry
from model.serving import register_model_as_factor
from model.training import train_and_register, train_predictor_model


def _lgbm_missing() -> bool:
    try:
        import lightgbm  # noqa: F401
        return False
    except ImportError:
        return True


# ===========================================================================
# 合成数据：AR(1) 信号植入日收益（特征可学习，5 日 horizon 标签有预测力）
# ===========================================================================
def _signal_market(n_days: int = 260, n_codes: int = 20, seed: int = 7,
                   phi: float = 0.9, beta: float = 0.004, noise: float = 0.015):
    """AR(1) 信号 + 信号驱动的日收益 → close 面板与信号面板。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]

    sig = np.zeros((n_days, n_codes))
    sig[0] = rng.normal(0, 1, n_codes)
    for t in range(1, n_days):
        sig[t] = phi * sig[t - 1] + rng.normal(0, np.sqrt(1 - phi**2), n_codes)
    signal = pd.DataFrame(sig, idx, codes)

    ret = 0.004 * sig + rng.normal(0, noise, (n_days, n_codes))
    close = pd.DataFrame(100.0 * np.cumprod(1 + ret, axis=0), idx, codes)
    return close, signal


@pytest.fixture(scope="module")
def market():
    return _signal_market()


@pytest.fixture(scope="module")
def feats(market):
    """信号特征 + 冗余特征 + 噪声特征（已截面标准化）。"""
    from factor.preprocessing import standardize_zscore

    close, signal = market
    rng = np.random.default_rng(11)
    raw = {
        "sig": signal,
        "sig_dup": signal * 2.0 + rng.normal(0, 0.05, signal.shape),  # 与 sig 相关 ~1
        "noise": pd.DataFrame(rng.normal(0, 1, signal.shape),
                              signal.index, signal.columns),
    }
    return {k: standardize_zscore(v) for k, v in raw.items()}


# ===========================================================================
# ② LabelBuilder
# ===========================================================================
class TestLabels:
    def test_forward_returns_hand_computed(self):
        close = pd.DataFrame(
            {"a": [100.0, 110.0, 121.0, 133.1], "b": [50.0, 50.0, 50.0, 50.0]},
            index=pd.date_range("2023-01-02", periods=4),
        )
        fwd = forward_returns(close, horizon=1)
        assert fwd.iloc[0]["a"] == pytest.approx(0.10)
        assert fwd.iloc[1]["a"] == pytest.approx(0.10)
        # 尾部 horizon 日无完整前瞻窗口 → NaN
        assert fwd.iloc[-1].isna().all()

    def test_forward_returns_horizon2(self):
        close = pd.DataFrame({"a": [100.0, 110.0, 121.0, 133.1]})
        fwd = forward_returns(close, horizon=2)
        assert fwd.iloc[0, 0] == pytest.approx(0.21)
        assert fwd.iloc[1, 0] == pytest.approx(0.21)
        assert np.isnan(fwd.iloc[2, 0]) and np.isnan(fwd.iloc[3, 0])

    def test_rank_mode_order_and_bounds(self, market):
        close, _ = market
        labels, embargo = build_labels(close, horizon=5, mode="rank")
        fwd = forward_returns(close, 5)
        # 秩标签保序：当日截面标签序 == 前瞻收益序（Spearman 等价性）
        d = labels.index[50]
        np.testing.assert_array_equal(np.argsort(labels.loc[d].values),
                                      np.argsort(fwd.loc[d].values))
        assert labels.notna().sum().sum() > 0
        # 值域 (-0.5, 0.5]
        vals = labels.values[~np.isnan(labels.values)]
        assert vals.min() > -0.5 and vals.max() <= 0.5
        assert embargo == 5

    def test_zscore_mode_cross_sectional_standardized(self, market):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="zscore")
        row = labels.iloc[60].dropna()
        assert row.mean() == pytest.approx(0.0, abs=1e-9)
        assert row.std() == pytest.approx(1.0, rel=1e-9)

    def test_raw_mode_equals_fwd(self, market):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="raw")
        fwd = forward_returns(close, 5)
        pd.testing.assert_frame_equal(labels, fwd)

    def test_from_fwd_panel(self, market):
        close, _ = market
        fwd = forward_returns(close, 5)
        labels, embargo = build_labels(fwd_returns_panel=fwd, horizon=5, mode="rank")
        ref, _ = build_labels(close, horizon=5, mode="rank")
        pd.testing.assert_frame_equal(labels, ref)
        assert embargo == 5

    def test_invalid_mode_raises(self, market):
        close, _ = market
        with pytest.raises(ValueError, match="未知 mode"):
            build_labels(close, horizon=5, mode="nope")

    def test_all_nan_raises(self):
        close = pd.DataFrame(np.nan, index=pd.date_range("2023-01-02", periods=3),
                             columns=["a"])
        with pytest.raises(ValueError, match="全为 NaN"):
            build_labels(close, horizon=5)


# ===========================================================================
# ① FeatureStore
# ===========================================================================
class TestFeatureSet:
    def test_include_exclude(self, feats):
        out = build_feature_set(feats, include=["sig", "noise"], dedup_corr=None)
        assert sorted(out) == ["noise", "sig"]
        out = build_feature_set(feats, exclude=["noise"], dedup_corr=None)
        assert "noise" not in out

    def test_filter_to_empty_raises(self, feats):
        with pytest.raises(ValueError, match="过滤后特征为空"):
            build_feature_set(feats, include=["nonexistent"])

    def test_min_coverage_drops_sparse(self, feats):
        sparse = feats["sig"].copy()
        sparse.iloc[: len(sparse) // 2] = np.nan   # 覆盖率 ~0.5 边界以下
        panels = dict(feats, sparse=sparse)
        out = build_feature_set(panels, min_coverage=0.55, dedup_corr=None)
        assert "sparse" not in out

    def test_dedup_corr_removes_redundant(self, feats):
        out = build_feature_set(feats, dedup_corr=0.7, min_coverage=0.0)
        # sig 与 sig_dup 高度相关 → 只留一个
        assert not {"sig", "sig_dup"} <= set(out)
        assert len(out) >= 2   # 保留一个信号 + noise

    def test_dedup_quality_keeps_better(self, feats):
        quality = pd.Series({"sig": 0.05, "sig_dup": 0.01, "noise": 0.0})
        out = build_feature_set(feats, dedup_corr=0.7, min_coverage=0.0,
                                quality=quality)
        assert "sig" in out and "sig_dup" not in out

    def test_max_features_truncation_by_quality(self, feats):
        quality = pd.Series({"sig": 0.05, "sig_dup": 0.04, "noise": 0.01})
        out = build_feature_set(feats, dedup_corr=None, min_coverage=0.0,
                                max_features=1, quality=quality)
        assert list(out) == ["sig"]

    def test_aligned_grid_intersection(self, feats):
        trimmed = {k: v.iloc[10:-10, :-5] for k, v in feats.items()}
        out = build_feature_set(trimmed, dedup_corr=None, min_coverage=0.0)
        for p in out.values():
            assert p.index.equals(feats["sig"].index[10:-10])
            assert list(p.columns) == list(feats["sig"].columns[:-5])

    def test_empty_panels_raises(self):
        with pytest.raises(ValueError):
            build_feature_set({})


# ===========================================================================
# ③ Predictor
# ===========================================================================
class TestPredictor:
    def _split(self, feats, labels, frac=0.6):
        n = len(labels.index)
        cut = labels.index[int(n * frac)]
        tr = labels.index[labels.index < cut]
        te = labels.index[labels.index >= cut]
        return ({k: v.loc[tr] for k, v in feats.items()},
                {k: v.loc[te] for k, v in feats.items()},
                labels.loc[tr], labels.loc[te])

    def test_ridge_fit_predict_shape_and_standardized(self, market, feats):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        ftr, fte, ltr, lte = self._split(feats, labels)
        p = RidgePredictor(alpha=1.0).fit(ftr, ltr)
        pred = p.predict(fte)
        assert pred.shape == lte.shape
        # 截面标准化：行均值 0、行标准差 1
        row = pred.iloc[0].dropna()
        assert row.mean() == pytest.approx(0.0, abs=1e-9)
        assert row.std() == pytest.approx(1.0, rel=1e-9)
        assert p.n_samples_ > 0

    def test_ridge_learns_planted_signal(self, market, feats):
        close, signal = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        ftr, fte, ltr, lte = self._split(feats, labels)
        pred = RidgePredictor(alpha=1.0).fit(ftr, ltr).predict(fte)
        # OOS Spearman IC 显著为正（信号已植入）
        ics = []
        for d in pred.index:
            x, y = pred.loc[d], lte.loc[d]
            ok = x.notna() & y.notna()
            if ok.sum() >= 5:
                ics.append(x[ok].corr(y[ok], method="spearman"))
        ic_mean = float(np.mean(ics))
        assert ic_mean > 0.05, f"植入信号未被学到: OOS IC={ic_mean:.4f}"

    def test_ridge_predict_missing_feature_raises(self, market, feats):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        ftr, fte, ltr, _ = self._split(feats, labels)
        p = RidgePredictor().fit(ftr, ltr)
        with pytest.raises(KeyError, match="缺少训练时的特征"):
            p.predict({"sig": fte["sig"]})

    @pytest.mark.skipif("gbdt" not in PREDICTORS or _lgbm_missing(),
                        reason="lightgbm 未安装")
    def test_gbdt_fit_predict(self, market, feats):
        from model.predictor import LGBMPredictor

        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        ftr, fte, ltr, lte = self._split(feats, labels)
        p = LGBMPredictor(n_estimators=60).fit(ftr, ltr)
        pred = p.predict(fte)
        assert pred.shape == lte.shape
        assert pred.notna().any().any()
        row = pred.iloc[-1].dropna()
        assert row.mean() == pytest.approx(0.0, abs=1e-9)

    def test_fit_predict_oos_coverage_and_discipline(self, market, feats):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        oos = fit_predict_oos(RidgePredictor, feats, labels,
                              n_splits=5, embargo_days=5, min_train_days=40)
        assert oos.shape == labels.shape
        # 首个训练段无预测（无未来函数：不可能预测训练自己的段）
        # _time_folds 口径：edges[1] = n_days // n_splits 之前的天不被任何折覆盖
        n_days = len(labels.index)
        assert oos.iloc[: n_days // 5].isna().all().all()
        # 之后各折测试段全部有预测
        assert oos.iloc[n_days // 5:].notna().any().all()

    def test_fit_predict_oos_insufficient_data_raises(self, market, feats):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        with pytest.raises(ValueError):
            fit_predict_oos(RidgePredictor, feats, labels,
                            n_splits=5, embargo_days=5, min_train_days=10**6)


# ===========================================================================
# ④ Trainer
# ===========================================================================
class TestTrainer:
    def test_train_predictor_model_stats(self, market, feats):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        result = train_predictor_model(feats, labels, method="ridge",
                                       n_splits=4, embargo_days=5,
                                       horizon=5, target_mode="rank")
        assert set(result) >= {"panel", "spec", "ic_mean", "ic_ir", "ic_t_nw"}
        assert result["panel"].shape == labels.shape
        assert np.isfinite(result["ic_mean"])
        spec = result["spec"]
        assert spec["method"] == "ridge" and spec["horizon"] == 5
        assert spec["embargo_days"] == 5
        assert sorted(spec["features"]) == sorted(feats)

    def test_train_and_register_predictor_kind(self, market, feats, tmp_path):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        reg = ModelRegistry(tmp_path / "mreg")
        mid, result = train_and_register(
            "ridge_h5", feats, labels, method="ridge", kind="predictor",
            horizon=5, target_mode="rank", n_splits=4, embargo_days=5,
            fingerprint="test:fp", registry=reg,
        )
        assert mid
        rec = reg.view(mid)
        assert rec["kind"] == "predictor" and rec["name"] == "ridge_h5"
        assert rec["fingerprint"] == "test:fp"
        assert rec["spec"]["horizon"] == 5
        assert "ic_mean" in rec["metrics"]
        # 血缘默认 = 特征名
        assert sorted(rec["parents"].split(",")) == sorted(feats)

    def test_train_and_register_predictor_requires_mapping(self, market, tmp_path):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        with pytest.raises(TypeError, match="特征字典"):
            train_and_register("bad", [close], labels, kind="predictor",
                               registry=ModelRegistry(tmp_path / "mreg"))


# ===========================================================================
# ⑤ Serving：模型预测回写因子库
# ===========================================================================
class TestServing:
    def _pred_panel(self, market):
        close, signal = market
        rng = np.random.default_rng(3)
        return signal + rng.normal(0, 0.3, signal.shape)

    def test_register_model_as_factor_lineage(self, market, tmp_path):
        from research.factor_library import FactorLibrary

        close, _ = market
        fwd = forward_returns(close, 5)
        pred = self._pred_panel(market)
        row = register_model_as_factor(
            name="model:ridge_h5", pred_panel=pred, returns_panel=fwd,
            parents=["sig", "noise"], dataset="mock",
            model_id="123456", horizon=5, oos=True,
            note="unit-test", root=tmp_path / "flib",
        )
        assert row["name"] == "model:ridge_h5"
        lib = FactorLibrary(root=tmp_path / "flib", dataset="mock")
        assert lib.has("model:ridge_h5")
        assert lib.lineage("model:ridge_h5") == ["sig", "noise"]
        # note 回写 model_id / horizon / OOS 三元信息（双向可追溯）
        assert "model_id=123456" in row["note"] and "horizon=5d" in row["note"]
        assert "OOS" in row["note"]
        assert row["maturity"] == "oos_verified"

    def test_register_model_as_factor_experimental(self, market, tmp_path):
        close, _ = market
        fwd = forward_returns(close, 5)
        row = register_model_as_factor(
            "model:full_fit_h5", self._pred_panel(market), fwd,
            dataset="mock", oos=False, root=tmp_path / "flib",
        )
        assert row["maturity"] == "experimental"


# ===========================================================================
# 编排：rolling_oos 折边界纪律（walk-forward 防泄漏核心）
# ===========================================================================
class TestRollingOOS:
    def test_embargo_and_fold_boundaries(self, market, feats):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        all_days = labels.index
        n = len(all_days)
        test_days = all_days[int(n * 0.8):]

        calls = []

        class SpyRidge(RidgePredictor):
            def fit(self, features, labels_):
                calls.append((labels_.index.min(), labels_.index.max()))
                return super().fit(features, labels_)

        embargo = 5
        pred = rolling_oos(SpyRidge, feats, labels, test_days, all_days,
                           n_folds=4, embargo_days=embargo, min_train_days=40)
        assert pred.shape == (len(test_days), labels.shape[1])

        # 每折训练段末尾必须剥掉 embargo 天：train_max == fold 前一日再往前 embargo 天
        fold_starts = [fd[0] for fd in np.array_split(test_days.to_numpy(), 4)]
        assert len(calls) == 4
        for (tr_min, tr_max), fold_start in zip(calls, fold_starts):
            pos = all_days.get_loc(pd.Timestamp(fold_start))
            assert tr_max == all_days[pos - 1 - embargo]
            assert tr_min == all_days[0]
            assert tr_max < pd.Timestamp(fold_start)

        # OOS 拼接：每天都被预测（无跳空、无交叉）
        assert pred.notna().all().all()

    def test_skip_short_folds(self, market, feats):
        close, _ = market
        labels, _ = build_labels(close, horizon=5, mode="rank")
        all_days = labels.index
        n = len(all_days)
        test_days = all_days[int(n * 0.8):]
        # min_train_days 超过全历史 → 全部折被跳过 → 报错
        with pytest.raises(RuntimeError, match="无任何 OOS 预测"):
            rolling_oos(RidgePredictor, feats, labels, test_days, all_days,
                        n_folds=2, embargo_days=5, min_train_days=10**6)
