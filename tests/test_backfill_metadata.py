"""registry 元数据回填（``scripts/factors/backfill_metadata.py``）的规则与安全性。

守三件事：
1. 规则产出的字符串**逐字**对齐仓库里既有的写法（alpha101 与 hs300 子库同款）；
2. 只补空缺——已有非空值哪怕与提案不同也不覆盖（冲突只报告）；
3. 不动任何统计列，且幂等（第二遍跑必须 0 变更）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.factors.backfill_metadata import (
    _alpha_formula,
    _e4_formula,
    build_plan,
    main,
    propose,
)


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------
def test_alpha101_uses_the_wording_hs300_already_ships():
    got = _alpha_formula("alpha101", "alpha101_001")
    assert got == "WorldQuant Alpha101 (Kakushadze 2016) #001"


def test_alpha191_keeps_the_three_digit_id():
    assert _alpha_formula("alpha191", "alpha191_166").endswith(" #166")


def test_alpha158_carries_the_feature_name_not_a_number():
    got = _alpha_formula("alpha158", "alpha158_BETA30")
    assert got.endswith("· BETA30")
    assert "Qlib Alpha158" in got


def test_alpha360_carries_the_field_and_lag():
    assert _alpha_formula("alpha360", "alpha360_CLOSE59").endswith("· CLOSE59")


def test_e4_splits_on_the_separator_because_buckets_contain_underscores():
    """``np_growth_yoy`` 自带下划线，只能按 ``_x_`` 切。"""
    got = _e4_formula("e4_np_growth_yoy_x_mom_20")
    assert got == "group_zscore(mom_20 | tercile(np_growth_yoy))"


def test_llm_reuses_the_label_which_is_the_raw_expression():
    row = pd.Series({"name": "llm_02", "set": "llmmcts",
                     "label": "div(sub(low,ts_mean_60(low)),ts_std_60(low))"})
    assert propose(row)["formula"].startswith("div(sub(low")


def test_llm_label_without_an_expression_is_not_used_as_formula():
    row = pd.Series({"name": "llm_02", "set": "llmmcts", "label": "某个候选"})
    assert propose(row)["formula"] == ""


def test_label_that_is_a_definition_is_reused_as_formula():
    row = pd.Series({"name": "rd_exp_ratio_ttm", "set": "fundamental",
                     "label": "研发强度=研发TTM/营收TTM"})
    assert propose(row)["formula"] == "研发强度=研发TTM/营收TTM"


def test_equals_sign_inside_prose_is_not_a_definition():
    row = pd.Series({"name": "report_delay", "set": "fundamental",
                     "label": "披露时滞相对偏移(晚披露=坏消息)"})
    assert propose(row)["formula"] == ""


def test_label_that_is_a_definition_is_reused_as_formula():
    row = pd.Series({"name": "rd_exp_ratio_ttm", "set": "fundamental",
                     "label": "研发强度=研发TTM/营收TTM"})
    assert propose(row)["formula"] == "研发强度=研发TTM/营收TTM"


def test_equals_sign_inside_prose_is_not_a_definition():
    row = pd.Series({"name": "report_delay", "set": "fundamental",
                     "label": "披露时滞相对偏移(晚披露=坏消息)"})
    assert propose(row)["formula"] == ""


@pytest.mark.parametrize("set_name,expected", [
    ("alpha101", "量价"), ("alpha158", "量价"),
    ("e4x", "合成"), ("llmmcts", "非线性组合"),
    ("不存在的集", ""),
])
def test_family_comes_from_the_canonical_table(set_name, expected):
    row = pd.Series({"name": "x_1", "set": set_name, "label": ""})
    assert propose(row)["family"] == expected


def test_set_can_be_recovered_from_the_name_prefix():
    """hs300 / zz1000 的 ``set`` 列整列为空——按前缀兜底，不猜别的。"""
    row = pd.Series({"name": "alpha191_007", "set": None, "label": ""})
    assert propose(row)["set"] == "alpha191"


# ---------------------------------------------------------------------------
# 计划：只补空缺 + 集级护栏
# ---------------------------------------------------------------------------
def _reg(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_plan_never_overwrites_a_non_empty_value():
    reg = _reg([{"name": "alpha101_001", "set": "alpha101", "family": "量价",
                 "formula": "手写的别的说法", "label": ""}])
    plan = build_plan(reg)
    assert not plan["fill_formula"].iloc[0]
    assert plan["conflict_formula"].iloc[0]
    assert plan["formula_new"].iloc[0] == "WorldQuant Alpha101 (Kakushadze 2016) #001"


def test_plan_skips_sets_whose_family_was_already_refined_by_hand():
    """fundamental 里商誉/质押被人工标成「股东」，按集名硬填会造出两套粒度。"""
    reg = _reg([
        {"name": "goodwill_ratio", "set": "fundamental", "family": "股东",
         "formula": None, "label": "商誉/净资产"},
        {"name": "brand_new_thing", "set": "fundamental", "family": None,
         "formula": None, "label": "新东西"},
    ])
    plan = build_plan(reg).set_index("name")
    assert not plan.loc["brand_new_thing", "fill_family"]
    assert plan.loc["brand_new_thing", "set_refined"]


def test_plan_fills_family_for_a_clean_set():
    reg = _reg([{"name": "alpha191_001", "set": "alpha191", "family": None,
                 "formula": None, "label": ""}])
    plan = build_plan(reg).iloc[0]
    assert plan["fill_family"] and plan["family_new"] == "量价"


# ---------------------------------------------------------------------------
# 端到端：写盘幂等、统计列不动
# ---------------------------------------------------------------------------
def _toy_lib(tmp_path):
    from research.factor_library import FactorLibrary

    lib = FactorLibrary(root=tmp_path, dataset="toy")
    pd.DataFrame([
        {"name": "alpha101_001", "set": "alpha101", "label": "",
         "formula": None, "family": None, "ic_mean_h1": 0.0375},
        {"name": "e4_bp_x_mom_20", "set": "e4x", "label": "E4 bp×mom_20",
         "formula": None, "family": None, "ic_mean_h1": -0.0125},
    ]).to_csv(lib.root / "registry.csv", index=False, encoding="utf-8-sig")
    return lib


def test_apply_fills_and_is_idempotent(tmp_path, capsys):
    lib = _toy_lib(tmp_path)
    argv = ["--dataset", "toy", "--root", str(tmp_path), "--apply"]
    main(argv)
    capsys.readouterr()

    reg = lib.list_all()
    assert reg["family"].notna().all()
    assert reg["formula"].notna().all()
    assert float(reg.set_index("name").loc["alpha101_001", "ic_mean_h1"]) == pytest.approx(0.0375)

    main(argv)                      # 第二遍不应有任何变更
    out = capsys.readouterr().out
    assert "待补 family 0 行" in out
    assert "待补 formula 0 行" in out


def test_dry_run_writes_nothing(tmp_path, capsys):
    lib = _toy_lib(tmp_path)
    main(["--dataset", "toy", "--root", str(tmp_path)])
    assert "[dry-run]" in capsys.readouterr().out
    reg = lib.list_all()
    assert reg["formula"].isna().all()
