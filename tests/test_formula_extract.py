"""``scripts/factors/formula_extract.py`` 的回归测试。

守的是三条**会静默失效**的东西：

1. **工厂 lambda 的闭包绑定**——``lambda w: lambda d: ts_mean(d.close, w)`` 里
   ``w`` 不代入就成了 ``ts_mean(close, w)``，字符串照样非空、看板照样显示，
   只是内容没用。这条错了两轮才被发现（2026-11-13）。
2. **实现名清洗**——``_sum`` / ``min_`` 是代码里的写法，公式里该是 ``sum`` / ``min``。
   上一版只处理了前缀式 ``_sum``，后缀式 ``min_`` 漏了 31 个因子。
3. **手写公式保护**——``pap_*`` 那 12 行是人写的，任何回填都不能覆盖；
   但同时不能把**上一轮自己填的**也当手写（否则源码改了公式永不更新）。
"""
from __future__ import annotations

import pytest

from scripts.factors.formula_extract import (
    extract_alpha_formula,
    formula_for,
    is_label,
    polish,
)


class TestRender:
    def test_plain_def_returns_expression(self):
        """普通 def 函数：返回可读表达式，``d.close`` 的 ``d.`` 被剥掉。"""
        import factor.alpha158 as m
        out = extract_alpha_formula("alpha158_KLEN", m.ALPHA158["alpha158_KLEN"])
        assert out == "safe_div(high - low, open)"

    def test_factory_lambda_binds_closure_window(self):
        """🚨 工厂 lambda 的窗口必须代入：出现 ``w`` 就是没绑定。"""
        import factor.alpha158 as m
        out = extract_alpha_formula("alpha158_ROC20", m.ALPHA158["alpha158_ROC20"])
        assert out is not None
        assert "w" not in out.replace("word", ""), f"窗口未绑定: {out}"
        assert "20" in out
        assert "close.shift(20)" in out

    def test_factory_lambda_binds_lag_for_alpha360(self):
        """alpha360 是两层转发（``_fn=_close_ref``），内层形参 ``i`` 也要代入。"""
        import factor.alpha158 as m
        out = extract_alpha_formula("alpha360_CLOSE3", m.ALPHA360["alpha360_CLOSE3"])
        assert out is not None
        assert "i > 0" not in out, f"条件未折叠: {out}"
        assert "close.shift(3)" in out

    def test_aliases_kept_when_used_twice(self):
        """被多次引用的中间量保留为 ``let``，不重复内联成巨型表达式。"""
        import factor.alpha191 as m
        out = extract_alpha_formula("alpha191_159", m.ALPHA191["alpha191_159"])
        assert out is not None
        assert out.startswith("let "), f"未保留别名: {out[:80]}"
        assert out.count("delay(close, 1)") <= 2, "别名被过度内联"


class TestPolish:
    @pytest.mark.parametrize("raw,want", [
        ("_sum(lo, 6)", "sum(lo, 6)"),
        ("_mean(volume, 30)", "mean(volume, 30)"),
        ("min_(low, dc)", "min(low, dc)"),
        ("max_(high, dc)", "max(high, dc)"),
    ])
    def test_impl_prefix_and_suffix_cleaned(self, raw, want):
        """🪤 仓库里 ``_sum``（前缀）与 ``min_``（后缀）两种写法并存，都要清洗。"""
        assert polish(raw) == want

    def test_does_not_touch_ordinary_identifiers(self):
        """带下划线的普通名（``sma_tdx`` / ``if_else``）不能被误伤。"""
        assert polish("sma_tdx(close, 15, 2)") == "sma_tdx(close, 15, 2)"
        assert polish("if_else(cond, 1.0, 0.0)") == "if_else(cond, 1.0, 0.0)"

    def test_const_condition_folded(self):
        """``i`` 代入后 ``if 3 > 0`` 是常量，应折叠成单支。"""
        assert polish("safe_div(close.shift(3), close) if 3 > 0 else close") \
            == "safe_div(close.shift(3), close)"

    def test_zero_lag_takes_else_branch(self):
        assert polish("safe_div(o, c) if 0 > 0 else safe_div(a, b)") \
            == "safe_div(a, b)"


class TestLabelHeuristic:
    @pytest.mark.parametrize("text,want", [
        ("资产负债率=总负债/总资产", True),
        ("大宗20日成交量", True),
        ("PEG", True),          # 纯缩写但够短
        ("ROE", True),
        ("block_amt_20d/block_vol_20d 但缺折价率的方向", True),
        ("", False),
    ])
    def test_label_detection(self, text, want):
        assert is_label(text) is want

    def test_long_ascii_is_not_a_label(self):
        """长纯英文串是正文，不是定义。"""
        assert is_label("block_amt_20d and block_vol_20d both exist") is False


class TestFormulaFor:
    def test_alpha_factor_carries_source_and_expression(self):
        """alpha 族：``<出处> #编号 = <表达式>``，前半是定义引用后半是公式。"""
        f, src = formula_for("alpha191_158")
        assert src == "ast"
        assert "Alpha191" in f
        assert "#158" in f
        assert "= " in f
        # 公式部分应体现 (high-a)-(low-a) 的抵消结构
        assert "sma_tdx" in f

    def test_non_alpha_factor_falls_back_to_label(self):
        """非 alpha 族没有注册表，只写中文标签，不伪造表达式。"""
        f, src = formula_for("debt_to_assets")
        assert src == "label"
        assert "资产负债率" in f

    def test_unknown_factor_returns_empty(self):
        f, src = formula_for("__no_such_factor__")
        assert (f, src) == ("", "none")


class TestBackfillBatch:
    """入库链的增量回填：只碰本批，且必须分清「手写」与「自动」。"""

    @staticmethod
    def _write(tmp_path, rows) -> None:
        """把一组 registry 行写成 CSV（供 ``backfill_batch`` 读）。"""
        import pandas as pd
        pd.DataFrame(rows).to_csv(tmp_path / "registry.csv", index=False,
                                  encoding="utf-8")

    def test_fills_empty_formula(self, tmp_path):
        from scripts.factors.formula_extract import backfill_batch
        self._write(tmp_path, [{"name": "alpha191_158", "formula": "",
                                    "formula_src": ""}])
        n = backfill_batch(tmp_path, ["alpha191_158"])
        assert n == 1
        import pandas as pd
        out = pd.read_csv(tmp_path / "registry.csv")
        assert "Alpha191" in out.loc[0, "formula"]
        assert out.loc[0, "formula_src"] == "ast"

    def test_does_not_touch_manual(self, tmp_path):
        from scripts.factors.formula_extract import backfill_batch
        self._write(tmp_path, [{"name": "pap_rev_5d",
                                    "formula": "手写=-(近5日收益)",
                                    "formula_src": "manual"}])
        assert backfill_batch(tmp_path, ["pap_rev_5d"]) == 0
        import pandas as pd
        assert (pd.read_csv(tmp_path / "registry.csv").loc[0, "formula"]
                == "手写=-(近5日收益)")

    def test_old_schema_blank_is_not_mistaken_for_manual(self, tmp_path):
        """🪤 空 formula 读回是 ``nan``，而 ``str(nan or '')`` 是 ``"nan"``（真值）
        ⇒ 若直接判非空，整批会被当手写静默跳过。这条守的就是那个坑。"""
        from scripts.factors.formula_extract import backfill_batch
        self._write(tmp_path, [{"name": "alpha191_158", "formula": ""}])
        assert backfill_batch(tmp_path, ["alpha191_158"]) == 1

    def test_old_schema_existing_value_is_kept(self, tmp_path):
        """无 ``formula_src`` 列的旧库：存量有值无法分辨来源 ⇒ 保守不动。"""
        from scripts.factors.formula_extract import backfill_batch
        self._write(tmp_path, [{"name": "alpha158_KLEN", "formula": "旧值"}])
        assert backfill_batch(tmp_path, ["alpha158_KLEN"]) == 0

    def test_auto_value_is_refreshable(self, tmp_path):
        """有来源列时 ``ast`` 值必须可刷新——否则源码改了公式永不更新。"""
        from scripts.factors.formula_extract import backfill_batch
        self._write(tmp_path, [{"name": "alpha158_KLEN", "formula": "旧值",
                                    "formula_src": "ast"}])
        assert backfill_batch(tmp_path, ["alpha158_KLEN"]) == 1
        import pandas as pd
        assert "KLEN" in pd.read_csv(tmp_path / "registry.csv").loc[0, "formula"]

    def test_ignores_names_outside_batch(self, tmp_path):
        from scripts.factors.formula_extract import backfill_batch
        self._write(tmp_path, [{"name": "alpha191_158", "formula": "",
                                    "formula_src": ""},
                                   {"name": "alpha101_001", "formula": "",
                                    "formula_src": ""}])
        assert backfill_batch(tmp_path, ["alpha191_158"]) == 1
        import pandas as pd
        out = pd.read_csv(tmp_path / "registry.csv")
        assert pd.isna(out.loc[1, "formula"])


class TestRegistryIntegrity:
    """对**真实库**的完整性检查：跑之前先看库在不在，缺库时跳过而不是假绿。"""

    @pytest.fixture
    def reg(self):
        from pathlib import Path

        import pandas as pd
        p = Path("E:/data/factor_library/all_a_2018_2026/registry.csv")
        if not p.exists():
            pytest.skip("因子库不在本机")
        return pd.read_csv(p)

    def test_every_factor_has_formula(self, reg):
        """formula 覆盖率必须是 100%：能反解析的写表达式，其余写中文定义。"""
        missing = reg.loc[reg["formula"].isna() | (reg["formula"] == ""), "name"].tolist()
        assert not missing, f"{len(missing)} 个因子无公式: {missing[:10]}"

    def test_no_impl_prefix_left(self, reg):
        """🚨 公式里不能残留实现名（``_sum(`` / ``min_(``），那是对外展示的脏值。"""
        bad = reg.loc[
            reg["formula"].astype(str).str.contains(
                r"(?<![\w.])_[a-z]+\(|(?<![\w.])(?:min_|max_|sum_|mean_)\(", regex=True),
            "name"].tolist()
        assert not bad, f"{len(bad)} 个因子公式残留实现名: {bad[:5]}"

    def test_formula_src_is_tracked(self, reg):
        """来源列必须齐：没有它，第二次回填会把自动填的值当手写保护起来。"""
        assert "formula_src" in reg.columns
        assert set(reg["formula_src"].dropna()) <= {"ast", "label", "manual"}
        assert (reg["formula_src"].isna()).sum() == 0

    def test_manual_formulas_untouched(self, reg):
        """``pap_*`` 手写行必须保持 manual 标记（防被自动覆盖）。"""
        pap = reg[reg["name"].str.startswith("pap_")]
        if pap.empty:
            pytest.skip("库内无 pap_* 因子")
        assert (pap["formula_src"] == "manual").all()