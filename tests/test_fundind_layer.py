"""E5 覆盖层分层解析（``panels_neu`` 基准 + ``panels_neu_fundind`` 覆盖）。

2026-11-18：``panels_neu_fundind`` 从「全量副本目录」改为「覆盖层」——只存与
基准层内容不同的因子，其余回落 ``panels_neu``。这里钉住三条不变式：

1. 覆盖层有的因子 → 读覆盖层；否则读基准层；
2. 清单 = 基准层 ∪ 覆盖层（**不能**用目录 glob 代替，会漏掉回落的那批）；
3. 基本面/股东族在基准层有面板、覆盖层却没有 ⇒ 必须**响亮失败**，
   否则该因子会静默退化成「行业+市值全剥」，与 E5 口径不符。
"""
from pathlib import Path

import pandas as pd
import pytest

from scripts.common import alla_common as AC


def _mk_panel(path: Path) -> None:
    idx = pd.date_range("2024-01-02", periods=3, freq="B")
    pd.DataFrame({"000001.SZ": [1.0, 2.0, 3.0]}, index=idx).to_parquet(path)


def _fake_ds(root: Path, base: list[str], overrides: list[str]) -> Path:
    (root / AC.FUNDIND_BASE_SUBDIR).mkdir(parents=True)
    (root / AC.FUNDIND_SUBDIR).mkdir(parents=True)
    for n in base:
        _mk_panel(root / AC.FUNDIND_BASE_SUBDIR / f"{n}.parquet")
    for n in overrides:
        _mk_panel(root / AC.FUNDIND_SUBDIR / f"{n}.parquet")
    return root


def test_resolution_prefers_override_then_falls_back(tmp_path):
    ds = _fake_ds(tmp_path, ["alpha101_001", "bp"], ["bp"])
    base, name_dir = AC.fundind_dirs(ds)
    assert base == ds / AC.FUNDIND_BASE_SUBDIR
    assert set(name_dir) == {"bp"}

    assert AC.layer_panel_file("alpha101_001", AC.FUNDIND_SUBDIR, ds) == \
        ds / AC.FUNDIND_BASE_SUBDIR / "alpha101_001.parquet"
    assert AC.layer_panel_file("bp", AC.FUNDIND_SUBDIR, ds) == \
        ds / AC.FUNDIND_SUBDIR / "bp.parquet"


def test_names_cover_both_layers(tmp_path):
    ds = _fake_ds(tmp_path, ["alpha101_001", "bp"], ["bp", "pap_high52w"])
    assert AC.fundind_names(ds) == ["alpha101_001", "bp", "pap_high52w"]


def test_missing_fundamental_override_is_loud(tmp_path):
    # bp 属基本面族：基准层有、覆盖层没有 ⇒ 静默退化，必须拦
    ds = _fake_ds(tmp_path, ["bp", "alpha101_001"], ["alpha101_001"])
    with pytest.raises(SystemExit, match="覆盖层缺"):
        AC.fundind_dirs(ds)


def test_non_fundamental_without_override_is_fine(tmp_path):
    ds = _fake_ds(tmp_path, ["alpha101_001"], [])
    base, name_dir = AC.fundind_dirs(ds)
    assert name_dir == {}
    assert base == ds / AC.FUNDIND_BASE_SUBDIR


def test_store_reads_through_both_layers(tmp_path):
    ds = _fake_ds(tmp_path, ["alpha101_001"], ["bp"])
    _mk_panel(ds / AC.FUNDIND_BASE_SUBDIR / "bp.parquet")   # 基准层也有 bp
    base, name_dir = AC.fundind_dirs(ds)
    store = AC.FeatureStore(base, name_dir=name_dir)
    got = store.get_many(["alpha101_001", "bp"])
    assert set(got) == {"alpha101_001", "bp"}
    assert all(isinstance(v, pd.DataFrame) and not v.empty for v in got.values())


def test_writer_skips_redundant_override(tmp_path):
    """写者侧不变式：不该写覆盖层的因子就不能写。

    旧 ``build_alla_altfactor_neu`` 对 altf **无条件双写**（当时两变体确实逐字节
    相同）——正是 11-18 清掉的 863 个重复副本的来源。改成按 ``needs_override``
    判定后，月度重建才不会把省下的空间逐月涨回来。
    """
    from scripts.builders.build_alla_altfactor_neu import needs_override

    ds = _fake_ds(tmp_path, ["alpha101_001", "bp"], ["bp"])
    assert needs_override("alpha101_001", ds) is False   # 量价：两变体相同 ⇒ 只写基准
    assert needs_override("bp", ds) is True              # 基本面：变体真不同
    assert needs_override("pap_high52w", ds) is True     # 基准层没有 ⇒ 覆盖层独有
