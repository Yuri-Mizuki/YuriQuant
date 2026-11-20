"""`research/dup_criteria.py` 的判据锁。

这些测试的作用不是"提高覆盖率"，而是**锁住几句曾经判反过的规则**：
每一条都对应 2026-11-12 实测踩到的坑，改口径时若把方向弄反，这里会红。
"""

from __future__ import annotations

import numpy as np
import pytest

from research.dup_criteria import (
    CORR_TRIGGER,
    REL_STD_DEGENERATE,
    REL_STD_TINY,
    RESID_IC_NEGLIGIBLE,
    V_DEGENERATE,
    V_INCREMENT,
    V_TINY,
    V_UNTESTED,
    Verdict,
    classify,
    explain,
    should_check,
    tier_of,
)


def _st(rel_std, ic=np.nan, t=np.nan, n_days=251, n_span=251, degenerate=None):
    if degenerate is None:
        degenerate = bool(np.isfinite(rel_std) and rel_std < REL_STD_DEGENERATE)
    return {"rel_std": rel_std, "ic": ic, "t": t, "n_days": n_days,
            "n_span": n_span, "degenerate": degenerate}


def test_degenerate_wins_over_zero_ic_days():
    """🔑 精确退化时 n_days=0（残差恒 0，IC 无从算），但必须判「等价」不是「无法检验」。

    这是 2026-11-12 的 34 个假保留事故：先查 n_days<20 就返回「无法检验 ⇒ 保留」，
    把精确等价/镜像判成「必须留」，方向完全反。
    """
    v = classify(_st(1e-15, ic=np.nan, t=np.nan, n_days=0, n_span=251,
                     degenerate=True))
    assert v.level == V_DEGENERATE
    assert v.is_equivalent is True
    assert v.removable is True
    assert v.tested is True          # 测过了，只是没有 IC 可算


def test_untested_is_never_removable():
    """截面不足 ⇒ 「无法检验」，绝不能当成「无增量」而判可删。"""
    v = classify(_st(0.10, ic=0.02, t=2.0, n_days=0, n_span=5))
    assert v.level == V_UNTESTED
    assert v.removable is False      # 关键：拿不准就留
    assert v.tested is False


def test_residual_without_ic_is_not_tiny():
    """残差存在但 IC 有效日不足 ⇒ 不判「微小增量」，宁可留。"""
    v = classify(_st(0.03, ic=np.nan, t=np.nan, n_days=3, n_span=251))
    assert v.level == V_UNTESTED
    assert v.removable is False


def test_tiny_needs_both_conditions():
    """「微小增量」必须 rel_std 与 |ic| **同时**达标，单条件不成立。"""
    assert classify(_st(0.02, ic=0.005, t=0.4)).level == V_TINY
    # rel_std 小但 IC 大 ⇒ 残差里有真信号，不能删
    assert classify(_st(0.02, ic=0.10, t=5.0)).level == V_INCREMENT
    # IC 小但残差大 ⇒ 残差本身有量，不能删
    assert classify(_st(0.30, ic=0.001, t=0.1)).level == V_INCREMENT


def test_high_corr_alone_is_not_equivalence():
    """🔑 |ρ|=0.96 的因子若秩空间还有残差，就不是「等价」——这正是 91 个误退的成因。"""
    v = classify(_st(0.104, ic=0.03, t=4.0))
    assert v.is_equivalent is False
    assert v.removable is False
    assert tier_of(0.96) == "T2近重复"      # 标注是「近重复」，判定是「有增量」


def test_tier_is_label_only():
    """tier 只是标注，阈值变化不影响可删性判定。"""
    assert tier_of(1.0) == "T1硬重复"
    assert tier_of(0.96) == "T2近重复"
    assert tier_of(0.90) == "T3同族"
    assert tier_of(0.50) == "独立"
    assert tier_of(float("nan")) == "未知"


def test_should_check_gate():
    """ρ 低于触发线不必做残差检验（8~13 s/因子），但这只是省时间不是判定。"""
    assert should_check(0.99) is True
    assert should_check(CORR_TRIGGER) is True
    assert should_check(0.50) is False
    assert should_check(float("nan")) is False


def test_verdict_dataclass_roundtrip():
    v = Verdict(V_INCREMENT, rel_std=0.1, resid_ic=0.03, resid_t=3.0,
                n_span=251, n_days=251, reason="x")
    d = v.as_dict()
    assert d["verdict"] == V_INCREMENT
    assert d["rel_std"] == pytest.approx(0.1)
    assert d["n_span"] == 251


def test_explain_mentions_that_rho_is_not_the_criterion():
    """提示语必须写明 ρ 不参与判定，否则调用方又会拿 ρ 当等价。"""
    s = explain()
    assert "不参与判定" in s
    assert f"{REL_STD_TINY:g}" in s and f"{RESID_IC_NEGLIGIBLE:g}" in s

def test_independent_is_a_conclusion_not_an_untested():
    """🔑「ρ 太低不相似」是**已完成的判断**，不是「没测成」。

    两者都不可删，但语义必须分开：把「不相似」写成「无法检验」会让日志出现
    「新因子 ρ=0.004 无法检验」这种自相矛盾的提示，也会掩盖真正的测不成。
    """
    from research.dup_criteria import V_INDEPENDENT, classify_independent

    v = classify_independent(0.42)
    assert v.level == V_INDEPENDENT
    assert v.removable is False
    assert v.tested is False          # 没做残差检验（不需要做）
    assert v.level != V_UNTESTED      # 但绝不等同于「没测成」
    assert "不相似" in v.reason


def test_high_corr_with_noise_is_not_removable():
    """加噪后 ρ 掉到 0.84 不到触发线 ⇒ 结论是「独立」，绝不判可删。"""
    from research.dup_criteria import classify_independent

    v = classify_independent(0.843)
    assert v.removable is False


# ---------------------------------------------------------------- 决策/执行层的 ρ 闸
# 2026-11-13：以下三条锁住的是「判据没有在每个档位上真正执行」的三道闸。
# 当时 `dup_criteria.CORR_TRIGGER=0.85` **从未生效**，判重实际只在 ρ≥0.95 区域运行。

def test_greedy_order_ranks_quality_then_coverage():
    """遍历顺序：质量分桶降序 → 桶内覆盖率降序。

    顺序本身不是「留谁」的判据（由覆盖交换兜底），但必须可复现且方向正确：
    高覆盖的因子不能仅因为 IC 被宽池稀释就排到最后。
    """
    import pandas as pd

    from scripts.factors.library_dedup_plan import greedy_order

    df = pd.DataFrame({
        "quality": [0.02, 0.02, 0.02, 0.50],
        "cover": [0.60, 0.90, 0.75, 0.10],
    })
    order = greedy_order(df)
    assert order[0] == 3, "质量最高者必须最先访问（桶最高）"
    low = [i for i in order if i != 3]
    assert low == [1, 2, 0], "同质量桶内按覆盖率降序"


def test_greedy_order_is_deterministic():
    """两次调用必须给出同一顺序——贪心结果依赖顺序，顺序不可抖。"""
    import pandas as pd

    from scripts.factors.library_dedup_plan import greedy_order

    df = pd.DataFrame({"quality": [0.01, 0.05, 0.03], "cover": [0.5, 0.5, 0.5]})
    assert greedy_order(df) == greedy_order(df)


def test_dedup_apply_trigger_equals_criteria_trigger():
    """🚨 执行层的覆盖门禁必须等于判据真源的触发线。

    历史事故：`library_dedup_apply.COVER_RHO_MIN=0.95` 与
    `dup_criteria.CORR_TRIGGER=0.85` 不一致 ⇒ 决策表给出 ρ=0.8975 的
    `safe_drop`（秩空间 rel_std=7.8e-16 的真等价）会被执行层以「ρ < 0.95」拒批。
    那是「判据只在 ρ≥0.95 生效」的第四道闸。
    """
    from scripts.factors import library_dedup_apply as app
    from research.dup_criteria import CORR_TRIGGER

    assert app.COVER_RHO_MIN == CORR_TRIGGER


def test_t3_band_is_below_trigger_line_by_cost_benefit():
    """🚨 T3 档（0.85~0.95）**整档不检验**，触发线 = 0.95。

    这里锁的是一个**被改回来的决定**，不是"更严格更好"。
    2026-11-13 上午曾把触发线压到 0.85 去测 T3，全量补测 156 个的结果是：
    **154 个「有独立增量」、仅 1 个真等价**，决策表 6 分钟 → 9 分钟。
    花 +55% 计算量换 1 个删除决定 ⇒ 整个档不值得测，改回 0.95。

    ⚠️ 若将来想重新压到 0.85，必须先证明该档的真等价率显著高于 1/156。
    """
    from research.dup_criteria import (
        CORR_TRIGGER, TIER_T2, TIER_T3, should_check)

    assert CORR_TRIGGER == TIER_T2, "触发线应等于 T2 下界：T3 整档不检验"
    assert should_check(0.8975) is False, "T3 档的 ρ 不应触发残差检验（产出比不抵）"
    assert should_check(CORR_TRIGGER) is True
    assert should_check(CORR_TRIGGER - 1e-6) is False
    assert TIER_T3 < CORR_TRIGGER


def test_t3_verdict_is_not_independent_and_not_tested():
    """T3 档不能报「独立」也不能报「测过了」。

    ρ=0.90 的两个因子**并不"不相似"**，只是整档不值得测。把它写成
    ``V_INDEPENDENT`` 会让日志输出「ρ=0.897 与库内不相似」这种自相矛盾的话；
    写成 ``tested=True`` 会让人以为有残差证据。
    """
    from research.dup_criteria import (
        V_BELOW_TRIGGER, V_INDEPENDENT, classify_independent)

    t3 = classify_independent(0.8975)
    assert t3.level == V_BELOW_TRIGGER
    assert t3.level != V_INDEPENDENT
    assert t3.tested is False        # 没测
    assert t3.removable is False     # 不可删（没证据）

    far = classify_independent(0.42)
    assert far.level == V_INDEPENDENT
    assert far.tested is False       # 也没测，但原因是"不相似"


def test_history_gate_defaults_to_false():
    """「曾进定版 selection」默认**不当硬闸**（2026-11-13 起）。

    核实过三件事：出榜链读静态 selection JSON（不读 registry）、重建链明确
    不重建 selection、当期四 horizon 与台账交集 0 ⇒ ``n_sel`` 不保护
    「已定方案」，只影响「主动重跑旧年份」。所以默认只护当期在用。
    """
    import inspect

    from scripts.factors.library_dedup_plan import build_decision

    sig = inspect.signature(build_decision)
    assert "history_is_gate" in sig.parameters
    assert sig.parameters["history_is_gate"].default is False
