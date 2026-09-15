"""
因子库 schema 与来源规范测试（2026-09-12「因子库管理统一规范」）。

覆盖四块：
1. canonical 列对齐——**补出来的列不能被丢弃**（本次修掉的真 bug）；
2. 受控词表——family / frequency / maturity 越界须告警，SET_TO_FAMILY 取值须合法；
3. source 命名规范——前缀必须是因子集名（gp/model 是下游路由键）；
4. builder 通道（merge_outputs → upsert_rows）自动带来源与分类，且不覆盖人工标签。

mock 数据，不依赖 SDK。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from research.factor_library import (
    CANONICAL_COLUMNS,
    FAMILY_WHITELIST,
    FREQUENCY_WHITELIST,
    MATURITY_VALUES,
    SET_TO_FAMILY,
    FactorLibrary,
    _align_registry,
    family_for_set,
    normalize_tags,
    source_for,
)


class _Capture(logging.Handler):
    """直接挂在目标 logger 上收记录——不依赖 propagate / caplog 的全局状态。"""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _capture(logger_name: str):
    h = _Capture()
    lg = logging.getLogger(logger_name)
    lg.addHandler(h)
    return h, lg


# --------------------------------------------------------------------------
# 1. canonical 列对齐
# --------------------------------------------------------------------------

def test_canonical_columns_has_no_duplicates():
    """列清单有重复会让 _align_registry 的 head+tail 出重复列，直接坏表。"""
    assert len(CANONICAL_COLUMNS) == len(set(CANONICAL_COLUMNS))


def test_canonical_columns_cover_both_channels():
    """两条入库通道各自的字段都要在 canonical 清单里，否则跨数据集列集会漂移。"""
    for c in ("name", "source", "family", "subfamily", "kind", "frequency",
              "set", "label", "coverage", "panel_path", "eval_path"):
        assert c in CANONICAL_COLUMNS, c


def test_align_registry_adds_missing_columns_instead_of_dropping():
    """回归守卫：补列循环必须先于 head 计算。

    2026-09-12 之前的写法是 `head = [c for c in CANONICAL_COLUMNS if c in reg.columns]`
    先算、再补列、最后 `reg[head + tail]` 取列 —— 补出来的列既不在 head
    （当时还不存在）也不在 tail（它在 CANONICAL 里），于是被静默丢弃，
    表现为「新加的 canonical 列在旧库上永远落不了地，且毫无告警」。
    """
    df = pd.DataFrame({"name": ["a"], "kind": ["raw"]})
    out = _align_registry(df)
    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns]
    assert missing == []
    assert list(out.columns)[: len(CANONICAL_COLUMNS)] == CANONICAL_COLUMNS


def test_align_registry_keeps_unknown_columns_at_tail():
    df = pd.DataFrame({"name": ["a"], "未知列": [1]})
    out = _align_registry(df)
    assert "未知列" in out.columns
    assert list(out.columns)[-1] == "未知列"
    assert list(out.columns)[: len(CANONICAL_COLUMNS)] == CANONICAL_COLUMNS


def test_align_registry_fast_path_returns_same_object():
    """已对齐时不得重建 DataFrame——list_all / get_panel 走这条高频路径。"""
    df = pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    assert _align_registry(df) is df


def test_list_all_columns_align_across_datasets(tmp_path):
    """builder 形态（有 set/label/coverage）与 register 形态的库必须给出同一套列。"""
    a = FactorLibrary(root=tmp_path / "flib", dataset="has_set")
    b = FactorLibrary(root=tmp_path / "flib", dataset="no_set")
    (a.root / "registry.csv").write_text(
        "name,set,label,coverage\nf1,alpha101,x,0.9\n", encoding="utf-8")
    (b.root / "registry.csv").write_text(
        "name,kind,source\nf2,raw,q:x\n", encoding="utf-8")
    ca, cb = list(a.list_all().columns), list(b.list_all().columns)
    assert ca == cb
    assert ca[: len(CANONICAL_COLUMNS)] == CANONICAL_COLUMNS


# --------------------------------------------------------------------------
# 2. 受控词表
# --------------------------------------------------------------------------

def test_set_to_family_values_all_in_whitelist():
    bad = {k: v for k, v in SET_TO_FAMILY.items() if v not in FAMILY_WHITELIST}
    assert bad == {}, f"SET_TO_FAMILY 有白名单外的族: {bad}"


def test_family_for_set_known_and_unknown():
    assert family_for_set("alpha101") == "量价"
    assert family_for_set("evt") == "事件"
    assert family_for_set("moneyflow") == "资金流"
    assert family_for_set("margin") == "资金流"
    assert family_for_set("holder") == "股东"
    assert family_for_set("sentiment") == "情绪"
    # 未收录宁可留空，也不猜
    assert family_for_set("不存在的集") == ""
    assert family_for_set("") == ""


def test_normalize_tags_warns_on_out_of_whitelist():
    h, lg = _capture("factor_library")
    try:
        fam, freq, mat = normalize_tags(
            family="不存在的族", frequency="每纳秒", maturity="draft", where="unit")
    finally:
        lg.removeHandler(h)
    msgs = " | ".join(r.getMessage() for r in h.records)
    assert "FAMILY_WHITELIST" in msgs
    assert "FREQUENCY_WHITELIST" in msgs
    assert "MATURITY_VALUES" in msgs
    # 告警但不改写——静默改写会掩盖真实的口径漂移
    assert (fam, freq, mat) == ("不存在的族", "每纳秒", "draft")


def test_normalize_tags_silent_on_legal_values():
    h, lg = _capture("factor_library")
    try:
        normalize_tags(family="量价", frequency="日频", maturity="experimental",
                       where="unit")
    finally:
        lg.removeHandler(h)
    assert h.records == []


def test_whitelist_membership_smoke():
    assert {"量价", "事件", "资金流", "股东", "状态", "情绪"} <= FAMILY_WHITELIST
    assert {"日频", "月频", "日内"} <= FREQUENCY_WHITELIST
    assert "experimental" in MATURITY_VALUES


# --------------------------------------------------------------------------
# 3. source 命名规范
# --------------------------------------------------------------------------

def test_source_for_uses_set_name_as_prefix():
    """前缀必须是因子集名——``source.split(":")[0]`` 是下游路由键。

    ``extend_factor_library.py`` 用 ``src == "gp"`` / ``src == "model"``
    选出要重算的因子；统一写成 ``builders:*`` 会让这两批被静默跳过。
    """
    assert source_for("gp", scope="hs300_2025") == "gp:builders:hs300_2025"
    assert source_for("model", scope="ds") == "model:builders:ds"
    assert source_for("alpha101", producer="build_alla_alpha_panels",
                      scope="all_a_2018_2026") == (
        "alpha101:build_alla_alpha_panels:all_a_2018_2026")
    for s in ("gp", "model", "alpha101", "evt"):
        assert source_for(s, scope="ds").split(":")[0] == s


def test_source_for_empty_set_falls_back_to_producer():
    assert source_for("", scope="ds") == "builders:ds"
    assert source_for("") == "builders"


def test_every_set_to_family_key_is_a_known_source_prefix():
    """回填/builder 产出的 source 前缀会过 SOURCE_PREFIXES 校验，集名须在册。"""
    from research.factor_library import SOURCE_PREFIXES

    missing = sorted(k for k in SET_TO_FAMILY if k not in SOURCE_PREFIXES)
    assert missing == [], f"集名未登记为 source 前缀: {missing}"


# --------------------------------------------------------------------------
# 4. 轻量登记（builder 通道）
# --------------------------------------------------------------------------

def test_upsert_rows_fills_source_and_tags(tmp_path):
    lib = FactorLibrary(root=tmp_path / "flib", dataset="ds_meta")
    stat = lib.upsert_rows(
        [{"name": "evt_a", "ic_mean_h1": 0.02}, {"name": "evt_b"}],
        source="evt:builders:ds_meta", family="事件", kind="raw",
        maturity="experimental", frequency="日频", set_name="evt")
    assert stat == {"before": 0, "after": 2, "added": 2, "updated": 0}
    reg = lib.list_all().set_index("name")
    for col, want in (("source", "evt:builders:ds_meta"), ("family", "事件"),
                      ("kind", "raw"), ("maturity", "experimental"),
                      ("frequency", "日频"), ("set", "evt")):
        assert reg.loc["evt_a", col] == want, col


def test_upsert_rows_does_not_overwrite_human_tags(tmp_path):
    """builder 重跑不得把人工补过的标签冲掉（fill_missing_only 的语义）。"""
    lib = FactorLibrary(root=tmp_path / "flib", dataset="ds_keep")
    rows = [{"name": "f1"}]
    lib.upsert_rows(rows, source="alpha101:builders:ds_keep",
                    family="量价", kind="raw", set_name="alpha101")
    assert lib.set_tag("f1", family="动量") is True
    lib.upsert_rows(rows, source="alpha101:builders:ds_keep",
                    family="量价", kind="raw", set_name="alpha101")
    reg = lib.list_all().set_index("name")
    assert reg.loc["f1", "family"] == "动量"
    # 空字段仍然被补上
    assert reg.loc["f1", "source"] == "alpha101:builders:ds_keep"


def test_upsert_rows_warns_on_empty_source(tmp_path):
    """空来源必须告警——它会让这批因子进不了「延长数据集」的因子集路由。"""
    lib = FactorLibrary(root=tmp_path / "flib", dataset="ds_warn")
    h, lg = _capture("factor_library")
    try:
        lib.upsert_rows([{"name": "g1"}], family="量价", set_name="alpha101")
    finally:
        lg.removeHandler(h)
    assert any("source 为空" in r.getMessage() for r in h.records)


def test_merge_outputs_default_source_is_set_prefixed(tmp_path):
    """builder 收尾入口的默认来源：``<集名>:builders:<数据集>`` + 自动族标签。

    horizons=() 跳过 IC 融合（那部分要 ic_h 面板），只验证 registry 写入。
    """
    from scripts.builders.common import merge_outputs

    base = tmp_path / "flib"
    ds_dir = base / "ds1"
    ds_dir.mkdir(parents=True)
    (ds_dir / "registry.csv").write_text("name,source\n", encoding="utf-8")
    stats = {"name": "alpha101_001", "set": "alpha101", "label": "x",
             "coverage": 0.9, "ic_mean_h1": 0.03}
    (ds_dir / "factor_stats_alpha101.jsonl").write_text(
        json.dumps(stats) + "\n", encoding="utf-8")

    merge_outputs(ds_dir, "alpha101", horizons=())

    row = FactorLibrary(root=base, dataset="ds1").list_all().iloc[0]
    assert row["source"] == "alpha101:builders:ds1"
    assert row["source"].split(":")[0] == "alpha101"   # 路由键
    assert row["family"] == "量价"
    assert row["kind"] == "raw"
    assert row["frequency"] == "日频"
    assert row["set"] == "alpha101"


# --------------------------------------------------------------------------
# 5. 报告层回归守卫
# --------------------------------------------------------------------------

def test_report_layer_no_longer_overwrites_family():
    """报告层曾 ``reg["family"] = reg["source"].map(src_family)`` 覆盖库的六维标签。

    该脚本是 main() 形态、依赖真实库，不便直接单测，故此处在源码层面钉住。
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "reporting"
           / "factor_library_full_report.py").read_text(encoding="utf-8")
    # 只看非注释行：文件里保留了说明该 bug 的注释，注释里会引用旧写法
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    offenders = [ln for ln in code if 'reg["family"] =' in ln]
    assert offenders == [], f"报告层仍在覆盖 family: {offenders}"
    assert "source_group" in src
