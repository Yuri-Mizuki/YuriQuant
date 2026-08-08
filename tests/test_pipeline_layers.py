"""
模型层（model/）与优化层（optimize/）冒烟测试 —— 流程规范化（2026-08-07）。
mock 数据，不依赖 SDK。
"""
import numpy as np
import pandas as pd
import pytest

from model import (
    ModelRegistry,
    evaluate_model,
    train_and_register,
    train_stacking_model,
)
from optimize import monitor_report, optimize_weights, risk_attribution
from research.benchmarks import equal_weight_returns


def _mock_panel(n_days=120, n_codes=20, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    # 两个有预测力的因子（滞后收益）
    f1 = rets.shift(1)
    f2 = (-rets).shift(2) * 0.5
    f1 = f1.div(f1.abs().max(axis=1), axis=0)
    f2 = f2.div(f2.abs().max(axis=1), axis=0)
    return {"f1": f1, "f2": f2}, rets


# ===========================================================================
# model/ 模型层
# ===========================================================================
def test_model_registry_crud(tmp_path):
    reg = ModelRegistry(root=tmp_path / "models")
    mid = reg.register(
        name="stacking_ridge_v1", kind="ml_stacking",
        spec={"method": "ridge", "alpha": 1.0},
        metrics={"ic_mean": 0.03, "ic_ir": 1.1},
        parents=["f1", "f2"],
    )
    assert reg.list().shape[0] == 1
    # 同名再注册 = 迭代新版本
    reg.register(name="stacking_ridge_v1", spec={"method": "ridge", "alpha": 2.0},
                 metrics={"ic_mean": 0.04})
    assert reg.list(name="stacking_ridge_v1").shape[0] == 2
    assert reg.latest(name="stacking_ridge_v1")["metrics"] != ""  # JSON 原样保存
    # compare 按指标排名
    cmp = reg.compare(metric="ic_mean")
    assert cmp.iloc[0]["name"] == "stacking_ridge_v1"
    # view 解析 JSON
    v = reg.view(mid)
    assert v["spec"]["method"] == "ridge"
    # delete
    assert reg.delete(mid)
    assert not reg.delete(mid)


def test_train_and_register_e2e(tmp_path):
    panels, returns = _mock_panel()
    from factor.synthesis import CompositeInput

    comps = [
        CompositeInput(name="f1", panel=panels["f1"], ic=0.1),
        CompositeInput(name="f2", panel=panels["f2"], ic=0.1),
    ]
    reg = ModelRegistry(root=tmp_path / "models")
    mid, result = train_and_register(
        "smoke_ridge", comps, returns, method="ridge",
        fingerprint="abc123", train_begin=20230101, train_end=20230630,
        parents=["f1", "f2"], registry=reg,
    )
    panel = result["panel"]
    assert panel.shape == panels["f1"].shape
    assert result["ic_mean"] is not None
    # 注册表里有记录且指标已写入
    v = reg.view(mid)
    assert v["kind"] == "ml_stacking"
    assert v["fingerprint"] == "abc123"
    assert v["metrics"]["ic_mean"] == pytest.approx(result["ic_mean"], abs=1e-12)


def test_evaluate_model_outputs():
    panels, returns = _mock_panel()
    out = evaluate_model(panels["f1"], returns, max_lag=5, n_quantiles=5)
    assert "ic_mean" in out and "ic_ir" in out
    assert "ic_t_nw" in out and "ic_p_nw" in out
    assert len(out["ic_decay"]) == 5
    assert out["quantile_backtest"].shape[0] >= 5


# ===========================================================================
# optimize/ 优化层
# ===========================================================================
def test_optimize_weights_properties():
    panels, returns = _mock_panel()
    f = panels["f1"]
    # factor_weighted：非空行每行和≈1，非负
    w = optimize_weights(f, method="factor_weighted")
    sums = w.sum(axis=1)
    assert (sums[sums > 0] - 1.0).abs().max() < 1e-9
    assert (w.values >= 0).all()
    # equal_topk：每行恰好 k 个 1/k
    w2 = optimize_weights(f, method="equal_topk", k=5)
    sums2 = w2.sum(axis=1)
    assert (sums2[sums2 > 0] - 1.0).abs().max() < 1e-9
    row = w2.iloc[1]  # 非空行
    assert (row > 0).sum() == 5
    # max_weight 截断：上限真实成立；未触发裁剪的行和≈1，触发后 ≤1（现金仓位）
    w3 = optimize_weights(f, method="factor_weighted", max_weight=0.05)
    assert (w3.values <= 0.05 + 1e-9).all()
    sums3 = w3.sum(axis=1)
    assert (sums3[sums3 > 0] <= 1.0 + 1e-9).all()
    # 未知方法报错
    with pytest.raises(ValueError):
        optimize_weights(f, method="nope")


def test_optimize_constraints_industry_neutral():
    """行业中性：每行行业总权重投影到等权行业目标。"""
    panels, returns = _mock_panel()
    f = panels["f1"]
    codes = list(f.columns)
    industry_map = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    w = optimize_weights(f, method="factor_weighted", industry_map=industry_map)
    mask = w.sum(axis=1) > 0  # 排除全空仓行（mock 首行全 NaN）
    for ind in ("ind0", "ind1", "ind2"):
        cols = [c for c in codes if industry_map[c] == ind]
        assert np.allclose(w.loc[mask, cols].sum(axis=1).values, 1 / 3, atol=1e-6)
    # 未覆盖行业的 code 不参与投影 → 权重保持不变
    industry_map2 = {codes[0]: "only"}
    w2 = optimize_weights(f, method="factor_weighted", industry_map=industry_map2)
    w0 = optimize_weights(f, method="factor_weighted")
    assert np.allclose(w2[codes[1]].values, w0[codes[1]].values, atol=1e-9)


def test_optimize_constraints_turnover():
    """换手约束：从空仓起步，单边换手被压到上限内。"""
    panels, returns = _mock_panel()
    f = panels["f1"]
    codes = list(f.columns)
    prev = pd.Series(0.0, index=codes)
    w2 = optimize_weights(f, method="factor_weighted", prev_weights=prev, max_turnover=0.1)
    turn = (w2 - prev).abs().sum(axis=1) * 0.5
    assert (turn <= 0.1 + 1e-6).all()
    # 确实比无约束时换手低
    w0 = optimize_weights(f, method="factor_weighted")
    turn0 = (w0 - prev).abs().sum(axis=1) * 0.5
    assert turn.max() < turn0.max()


def test_optimize_constraints_limits_and_combo():
    """权重下限过滤、行业中性+上限并存。"""
    panels, returns = _mock_panel()
    f = panels["f1"]
    codes = list(f.columns)
    # min_weight 微仓过滤（stack 丢弃 NaN，返回 Series 断言无歧义）
    w3 = optimize_weights(f, method="factor_weighted", min_weight=0.05)
    nz = w3[w3 > 0].stack()
    assert (nz >= 0.05 - 1e-9).all()
    # 行业中性 + 上限并存
    industry_map = {c: f"ind{i % 3}" for i, c in enumerate(codes)}
    w4 = optimize_weights(f, method="factor_weighted",
                          industry_map=industry_map, max_weight=0.3)
    assert (w4.values <= 0.3 + 1e-9).all()
    # equal_topk 也走同一约束管线
    w5 = optimize_weights(f, method="equal_topk", k=5, max_weight=0.5)
    assert (w5.values <= 0.5 + 1e-9).all()


def test_risk_attribution_smoke():
    panels, returns = _mock_panel()
    port = panels["f1"].mean(axis=1).fillna(0.0)  # 伪组合收益
    bench = equal_weight_returns(1 + returns.cumsum())  # 等权基准
    out = risk_attribution(port, bench)
    assert "alpha_beta" in out and "benchmark" in out
    assert "excess_annual" in out["benchmark"]
    assert out["brinson"] is None  # 未提供权重/行业时跳过


def test_monitor_report_smoke():
    panels, returns = _mock_panel()
    out = monitor_report(panels["f1"], returns, window=40)
    assert "ic_mean_full" in out and "ic_mean_recent" in out
    assert "ic_drift" in out and "status" in out
    assert out["n_days"] > 0 and out["recent_n_days"] > 0
    assert len(out["ic_decay"]) == 10
