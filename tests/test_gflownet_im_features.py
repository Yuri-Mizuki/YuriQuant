"""GFlowNet 接分钟特征（国金22 × 国金24 交叉）测试。

覆盖三块：
1. ``--feat-source`` 取值解析（含非法取值必须抛错，不静默退化）；
2. ``attach_im_features`` 的对齐口径 —— 特征名顺序（raw 前缀不变）、mask 生效、
   覆盖区间交回调用方、全 NaN 面板剔除并抛错；
3. ``aligned_eval_masks`` 的窗口交集语义 —— 启用 im 时两类公式必须落在**同一
   评估段**（否则奖励/OOS IC 不可比）；``keep_window`` 时不裁。

另验证 ``FactorMDP`` 动作空间随 im 终端扩张、含 im 终端的公式能求值且 reward 有限。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor.gflownet.env import FactorMDP
from factor.gflownet.reward import make_reward_fn
from scripts.common.e2e_common import (
    FEAT_SOURCES,
    RAW_FEATURES,
    aligned_eval_masks,
    attach_im_features,
    resolve_feat_source,
)


def _panel(n_days: int = 160, n_codes: int = 10, start: str = "2021-06-01"):
    """合成面板：价量 + mask（前 40 日首只不在册，用来验证掩码生效）。"""
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    ret = pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)
    close = 10.0 * (1 + ret).cumprod()
    vol = pd.DataFrame(rng.lognormal(10, 0.4, (n_days, n_codes)), idx, codes)
    mask = pd.DataFrame(True, idx, codes)
    mask.loc[idx[:40], codes[0]] = False
    return {
        "close": close, "open": close, "high": close, "low": close,
        "volume": vol, "amount": vol * close, "mask": mask,
    }


def _im_loader(im_start: str = "2022-01-03", n_days: int = 100,
               codes: list[str] | None = None):
    """注入式 loader：im 特征只在 ``im_start`` 之后有值（模拟真实覆盖 2022+）。"""
    idx = pd.date_range(im_start, periods=n_days, freq="B")
    codes = codes or [f"{600000 + i:06d}.SH" for i in range(10)]

    def load(source: str, dataset: str):
        rng = np.random.default_rng(1)
        a = pd.DataFrame(rng.normal(0, 1, (n_days, len(codes))), idx, codes)
        b = pd.DataFrame(rng.normal(0, 1, (n_days, len(codes))), idx, codes)
        return {"im_beta": b, "im_alpha": a}, {"im_alpha": "doc"}

    return load


# ---------------------------------------------------------------------------
# 1. feat-source 解析
# ---------------------------------------------------------------------------
def test_feat_source_choices_resolve():
    """5 个取值都能解析，且 raw / im 开关符合定义。"""
    assert set(FEAT_SOURCES) == {
        "raw", "im_independent", "im_all", "raw+im_independent", "raw+im_all"}
    assert resolve_feat_source("raw") == (True, "")
    assert resolve_feat_source("im_all") == (False, "im_all")
    assert resolve_feat_source("raw+im_independent") == (True, "im_independent")
    # 单 raw 臂必须只有 6 个字段（改动前的行为）
    assert list(RAW_FEATURES) == ["open", "high", "low", "close", "volume", "amount"]


def test_feat_source_invalid_raises():
    """非法取值抛 ValueError（不静默退化成 raw —— 静默退化会让实验标签骗人）。"""
    with pytest.raises(ValueError, match="未知 feat_source"):
        resolve_feat_source("im_some")


# ---------------------------------------------------------------------------
# 2. attach_im_features
# ---------------------------------------------------------------------------
def test_raw_arm_is_passthrough():
    """raw 臂不触发 loader、不动面板、特征名等于 RAW_FEATURES。"""
    panel = _panel()
    called = []

    def boom(source, dataset):
        called.append(source)
        raise AssertionError("raw 臂不应调用 loader")

    out, feats, cov = attach_im_features(panel, "raw", loader=boom)
    assert called == []
    assert feats == list(RAW_FEATURES)
    assert cov is None
    assert list(out) == list(panel)
    assert out["close"] is panel["close"]


def test_im_features_appended_after_raw_and_sorted():
    """特征名 = raw 6 字段（顺序不变）+ im_* 升序；动作 id 前缀因此保持稳定。"""
    panel = _panel()
    out, feats, cov = attach_im_features(
        panel, "raw+im_all", loader=_im_loader())
    assert feats[:6] == list(RAW_FEATURES)
    assert feats[6:] == ["im_alpha", "im_beta"]   # 字典序，与 loader 返回顺序无关
    assert cov is not None and set(out) - set(panel) == {"im_alpha", "im_beta"}

    only_im, feats_im, _ = attach_im_features(panel, "im_all", loader=_im_loader())
    assert feats_im == ["im_alpha", "im_beta"]
    assert "close" in only_im and "close" not in feats_im


def test_im_panels_aligned_and_masked():
    """im 面板被 reindex 到 close 的网格，并按 panel['mask'] 掩码（同 close_m 口径）。"""
    panel = _panel()
    out, _feats, _cov = attach_im_features(
        panel, "im_all", loader=_im_loader(im_start="2021-06-01", n_days=160))
    q = out["im_alpha"]
    assert q.shape == panel["close"].shape
    assert q.index.equals(panel["close"].index)
    assert q.columns.equals(panel["close"].columns)
    first = panel["close"].columns[0]
    # 前 40 日首只不在册 -> 必须为 NaN
    assert q.loc[panel["close"].index[:40], first].isna().all()
    # 在册后应有值
    assert q.loc[panel["close"].index[45:], first].notna().all()


def test_im_coverage_is_intersection_with_panel():
    """im 覆盖区间 = im 有值日 ∩ 面板日期（不是盲取 im 自己的 index 范围）。"""
    panel = _panel(n_days=160, start="2021-06-01")     # ~2021-06 ~ 2022-01
    out, _f, cov = attach_im_features(
        panel, "im_all", loader=_im_loader(im_start="2021-01-04", n_days=400))
    assert cov[0] >= panel["close"].index[0]
    assert cov[-1] <= panel["close"].index[-1]
    assert len(cov) == len(panel["close"])
    assert "im_alpha" in out


def test_all_nan_im_panel_raises():
    """im 面板与面板网格零交集时必须抛错，而不是产出一个"全是空特征"的静默实验。"""
    panel = _panel(n_days=40, start="2019-01-02")      # 全在 im 覆盖之前
    with pytest.raises(RuntimeError, match="全部对齐失败"):
        attach_im_features(panel, "im_all", loader=_im_loader(im_start="2022-01-03"))


def test_dropped_panel_is_excluded():
    """部分面板全 NaN 时被剔除，其余保留（不整体失败）。"""
    panel = _panel(n_days=120, start="2022-01-03")

    def load(source, dataset):
        good = pd.DataFrame(1.0, panel["close"].index, panel["close"].columns)
        empty = pd.DataFrame(np.nan, pd.date_range("2000-01-03", periods=5),
                             panel["close"].columns)
        return {"im_good": good, "im_empty": empty}, {}

    out, feats, _cov = attach_im_features(panel, "im_all", loader=load)
    assert feats == ["im_good"]
    assert "im_empty" not in out


# ---------------------------------------------------------------------------
# 3. 窗口对齐语义
# ---------------------------------------------------------------------------
def test_masks_identical_to_legacy_when_no_im():
    """未启用 im 时 test_mask 必须等于旧写法 ``~train_mask``（向后兼容）。"""
    idx = pd.date_range("2019-01-02", periods=500, freq="B")
    tr, te = aligned_eval_masks(idx, 20250101, im_index=None)
    old_tr = np.asarray(idx < pd.Timestamp("20250101"))
    assert (tr == old_tr).all()
    assert (te == ~old_tr).all()
    assert tr.sum() + te.sum() == len(idx)


def test_masks_clipped_to_im_coverage():
    """启用 im 时训练段起点被抬到 im 首日，两类公式落在同一评估段。"""
    idx = pd.date_range("2019-01-02", periods=1500, freq="B")
    im_idx = idx[idx >= pd.Timestamp("2022-01-04")]
    tr, te = aligned_eval_masks(idx, 20250101, im_index=im_idx)
    assert idx[tr][0] >= im_idx[0]
    assert (idx[tr] < pd.Timestamp("20250101")).all()
    assert (idx[te] <= im_idx[-1]).all()
    # 裁掉的正是 im 覆盖之前的日子，且没有重复计算
    assert tr.sum() + te.sum() == len(im_idx)
    assert not (tr & te).any()


def test_keep_window_disables_clip():
    """keep_window=True 时不裁（保留"不可比"的逃生口，但仍须调用方告警）。"""
    idx = pd.date_range("2019-01-02", periods=1500, freq="B")
    im_idx = idx[idx >= pd.Timestamp("2022-01-04")]
    tr, te = aligned_eval_masks(idx, 20250101, im_index=im_idx, keep_window=True)
    assert idx[tr][0] == idx[0]
    assert tr.sum() + te.sum() == len(idx)


def test_eval_begin_clips_without_im():
    """eval_begin 单独即可裁训练段（raw 对照臂用它对齐窗口，不动面板构建）。"""
    idx = pd.date_range("2019-01-02", periods=1500, freq="B")
    tr, te = aligned_eval_masks(idx, 20250101, im_index=None, eval_begin=20220104)
    assert idx[tr][0] >= pd.Timestamp("2022-01-04")
    assert (idx[tr] < pd.Timestamp("20250101")).all()
    # 测试段不受 eval_begin 影响（它本来就晚于 eval_begin）
    assert te.sum() == (idx >= pd.Timestamp("20250101")).sum()


def test_eval_begin_and_im_take_later():
    """eval_begin 与 im 覆盖起点同时给出时取**较晚者**（取早会引入 im 无值日）。"""
    idx = pd.date_range("2019-01-02", periods=1500, freq="B")
    im_idx = idx[idx >= pd.Timestamp("2022-01-04")]
    # eval_begin 更早 -> 仍以 im 起点为准
    tr1, _ = aligned_eval_masks(idx, 20250101, im_index=im_idx, eval_begin=20200101)
    assert idx[tr1][0] >= pd.Timestamp("2022-01-04")
    # eval_begin 更晚 -> 以 eval_begin 为准
    tr2, _ = aligned_eval_masks(idx, 20250101, im_index=im_idx, eval_begin=20230101)
    assert idx[tr2][0] >= pd.Timestamp("2023-01-01")


def test_eval_begin_does_not_affect_panel_identity():
    """eval_begin 语义上只动掩码 —— 面板对象必须原样返回（列集不变的保证）。"""
    panel = _panel()
    out, _f, cov = attach_im_features(panel, "raw", loader=_im_loader())
    assert list(out) == list(panel)
    assert out["close"] is panel["close"]
    # 无 im 时 aligned_eval_masks 不返回任何面板，只返回掩码
    idx = pd.date_range("2019-01-02", periods=500, freq="B")
    tr, te = aligned_eval_masks(idx, 20240101, eval_begin=20220104)
    assert tr.dtype == bool and te.dtype == bool


def test_eval_end_clips_test_segment():
    """eval_end 裁测试段上界 —— 这是 raw 对照臂与 im 臂测试段对齐的最后一环。"""
    idx = pd.date_range("2019-01-02", periods=1500, freq="B")
    tr, te = aligned_eval_masks(idx, 20250101, im_index=None, eval_end=20260630)
    assert (idx[te] <= pd.Timestamp("2026-06-30")).all()
    assert tr.sum() == (idx < pd.Timestamp("20250101")).sum()   # 训练段不受影响


def test_eval_end_and_im_take_earlier():
    """eval_end 与 im 覆盖终点同时给出时取**较早者**（取晚会引入 im 无值日）。"""
    idx = pd.date_range("2019-01-02", periods=1500, freq="B")
    im_idx = idx[(idx >= pd.Timestamp("2022-01-04")) & (idx <= pd.Timestamp("2026-03-31"))]
    _tr, te1 = aligned_eval_masks(idx, 20250101, im_index=im_idx)
    assert (idx[te1] <= pd.Timestamp("2026-03-31")).all()
    # eval_end 更早 -> 以 eval_end 为准
    _tr, te2 = aligned_eval_masks(idx, 20250101, im_index=im_idx, eval_end=20250801)
    assert (idx[te2] <= pd.Timestamp("2025-08-01")).all()


def test_fair_protocol_produces_identical_window():
    """公平协议自证：im 臂与 raw 臂（--eval-begin/--eval-end）必须落在同一段。"""
    idx = pd.date_range("2019-01-02", periods=1900, freq="B")
    im_idx = idx[(idx >= pd.Timestamp("2022-01-04")) & (idx <= pd.Timestamp("2026-06-30"))]
    im_begin = int(im_idx[0].strftime("%Y%m%d"))
    im_end = int(im_idx[-1].strftime("%Y%m%d"))

    tr_im, te_im = aligned_eval_masks(idx, 20250101, im_index=im_idx)
    tr_raw, te_raw = aligned_eval_masks(idx, 20250101, im_index=None,
                                        eval_begin=im_begin, eval_end=im_end)
    assert (tr_im == tr_raw).all(), "训练段掩码不一致 → 公平协议失效"
    assert (te_im == te_raw).all(), "测试段掩码不一致 → 公平协议失效"


# ---------------------------------------------------------------------------
# 4. MDP / reward 端到端
# ---------------------------------------------------------------------------
def _make_builder(op: str, feat: str, win: int | None = None,
                  feat_idx: int = 0, max_depth: int = 3, max_nodes: int = 9):
    """按 ``op`` →（若有窗口）``win`` → ``feat`` 的真实动作顺序造一个 ExprBuilder。

    动作顺序由 :meth:`ExprBuilder.legal_kinds` 的 FIFO 槽位语义决定：带窗口的
    算子入栈后队首是 win 槽位，必须先选窗口再选特征（2026-09-16 实测确认）。
    """
    from factor.gflownet.expr import ExprBuilder
    from factor.operators import op_registry

    spec = op_registry()[op]
    b = ExprBuilder(max_depth=max_depth, max_nodes=max_nodes)
    assert b.step("op", spec, 0), f"放下 {op} 失败"
    if spec.n_window >= 1:
        assert win is not None, f"{op} 需要窗口"
        assert b.step("win", win, 0), "放窗口失败"
    assert b.step("feat", feat, feat_idx), f"放下特征 {feat} 失败"
    assert b.is_done(), "轨迹未终止"
    return b


def test_mdp_action_space_expands_with_im():
    """加 im 终端后 n_feat/n_actions 增加，且末位动作语义解析回 im 特征名。"""
    base = FactorMDP(["ts_mean", "abs"], (5, 10), list(RAW_FEATURES))
    aug = FactorMDP(["ts_mean", "abs"], (5, 10),
                    list(RAW_FEATURES) + ["im_alpha", "im_beta"])
    assert aug.n_feat == base.n_feat + 2
    assert aug.n_actions == base.n_actions + 2
    kind, value, idx = aug.action_semantics(aug.n_actions - 1)
    assert (kind, value, idx) == ("feat", "im_beta", aug.n_feat - 1)
    # 前 6 个 feat 动作语义不变（raw 前缀稳定性 —— 混合臂才与 raw 臂动作语义对齐）
    for i in range(6):
        assert aug.action_semantics(aug.n_actions - aug.n_feat + i)[1] == RAW_FEATURES[i]


def test_builders_from_im_terminals_canonicalize():
    """含 im 终端的表达式能构造完并 canonical 化成 formula_builder 可解析的串。"""
    from factor.gflownet.expr import canonical_formula

    assert canonical_formula(_make_builder("ts_mean", "im_sig", 5)) == "ts_mean_5(im_sig)"
    assert canonical_formula(_make_builder("cs_rank", "im_neg")) == "cs_rank(im_neg)"


def test_im_formula_evaluates_and_reward_finite():
    """含 im 终端的公式能求值；带真实信号的 im 终端 reward 显著优于噪声终端。

    这一条是"im 面板真的被消费了"的守门测试 —— 若 attach 只加了特征名却没接面板，
    ``formula_builder`` 求值会全 NaN、reward 退化为 eps，断言立刻失败。
    """
    from factor.formula import formula_builder

    n_days, n_codes = 240, 12
    idx = pd.date_range("2022-01-03", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    rng = np.random.default_rng(3)
    close = pd.DataFrame(
        10.0 * (1 + rng.normal(0, 0.02, (n_days, n_codes))).cumprod(axis=0), idx, codes)
    # im 特征带真实信号：t 日特征 = t→t+10 收益的截面百分位排名
    im_sig = close.pct_change(10).shift(-10).rank(axis=1, pct=True)
    im_neg = pd.DataFrame(rng.normal(0, 1, (n_days, n_codes)), idx, codes)

    panel = {"close": close, "amount": close * 1e6,
             "im_sig": im_sig, "im_neg": im_neg}
    features = ["close", "amount", "im_sig", "im_neg"]

    for formula in ("im_sig", "ts_mean_5(im_sig)", "cs_rank(im_neg)",
                    "add(im_sig, im_neg)", "sub(im_sig, ts_delay(im_neg, 5))"):
        val = formula_builder(formula, features=features)(panel)
        assert val.shape == close.shape, formula
        assert val.notna().any().any(), f"{formula} 求值全 NaN"

    reward_fn = make_reward_fn(panel, None, features, horizon=10,
                               market_cap=None, barra_mu=0.0, long_ir_lambda=0.0)
    r_sig = reward_fn(_make_builder("ts_mean", "im_sig", 5))
    r_neg = reward_fn(_make_builder("ts_mean", "im_neg", 5))
    assert np.isfinite(r_sig) and r_sig >= 0
    assert np.isfinite(r_neg) and r_neg >= 0
    assert r_sig > r_neg, f"信号终端 reward ({r_sig}) 未超过噪声终端 ({r_neg})"
