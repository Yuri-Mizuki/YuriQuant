"""两条入库通道的库 API 兼容性测试（2026-09-12）。

因子库有两条入库通道：

- ``FactorLibrary.register()``：写全字段 + ``panel_path``（``panels/<md5>.parquet``）
  + ``eval_path``（``evals/<md5>.parquet``）。
- ``scripts/builders/common.py: merge_outputs()``：直写 registry.csv，只写
  ``name/set/label/coverage/ic_mean_h*``，面板落 ``panels/<name>.parquet``，
  **从不写 panel_path**。

此前库 API 直接 ``Path(hit.iloc[0]["panel_path"])``，在扁平行上拿到 NaN →
``Path(nan)`` 抛 ``TypeError``，导致 912 个批量因子在 ``get_panel`` /
``load_library_features`` / ``load_significant_features`` / ``monitor`` /
``select_diverse`` / ``delete`` 上整片不可用。因为库里同时有 26 个正规行
（能跑），所以表现为"部分可用"，长期没被发现。

本文件把"两条通道的面板都能读到、缺数据不抛异常、预检不静默跳过"钉死。
全部用 mock 数据落 tmp_path，不依赖 SDK 与真实因子库。
"""
import numpy as np
import pandas as pd

from research.factor_library import FactorLibrary

_NAMES = ("evt_insider", "mf_lhb_net_buy")


def _mock_panel(n_days=120, n_codes=20, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"{600000 + i:06d}.SH" for i in range(n_codes)]
    return pd.DataFrame(rng.normal(0, 0.02, (n_days, n_codes)), idx, codes)


def _builder_style_lib(tmp_path, names=_NAMES):
    """造一个 ``merge_outputs()`` 口径的库：registry 只有 builder 列 + 原名面板。

    刻意不写 ``panel_path`` / ``eval_path`` / ``source`` / ``significant``
    这些列——完全复刻全A 批量入库后的 registry 形态。
    """
    lib = FactorLibrary(root=tmp_path / "flib")
    panels = {}
    for i, n in enumerate(names):
        p = _mock_panel(seed=100 + i)
        panels[n] = p
        p.to_parquet(lib.panels_dir / f"{n}.parquet")
    reg = pd.DataFrame({
        "name": list(names),
        "set": ["evt", "moneyflow"][:len(names)],
        "label": ["内部人交易", "龙虎榜净买入"][:len(names)],
        "coverage": [0.9, 0.8][:len(names)],
    })
    reg.to_csv(lib.root / "registry.csv", index=False, encoding="utf-8-sig")
    return lib, panels


def _rewrite_registry(lib, **cols):
    reg = pd.read_csv(lib.root / "registry.csv")
    for k, v in cols.items():
        reg[k] = v
    reg.to_csv(lib.root / "registry.csv", index=False, encoding="utf-8-sig")


# ---- 面板路径解析 ----

def test_get_panel_falls_back_to_name_parquet(tmp_path):
    """扁平行（panel_path 列根本不存在）也能取到面板，不再抛 TypeError。"""
    lib, panels = _builder_style_lib(tmp_path)
    got = lib.get_panel("evt_insider")
    assert got is not None
    assert got.shape == panels["evt_insider"].shape
    assert lib.get_panel("不存在") is None


def test_get_panel_tolerates_nan_panel_path(tmp_path):
    """panel_path 列已存在但值为 NaN（混合库的典型形态）时也走回退。"""
    lib, _ = _builder_style_lib(tmp_path)
    _rewrite_registry(lib, panel_path=np.nan, eval_path=np.nan, source="")
    assert lib.get_panel("mf_lhb_net_buy") is not None


def test_stale_panel_path_falls_back(tmp_path):
    """panel_path 指向已不存在的文件时，回退原名面板。"""
    lib, _ = _builder_style_lib(tmp_path, names=("evt_insider",))
    _rewrite_registry(lib, panel_path="E:/nonexistent/deadbeef.parquet")
    assert lib.get_panel("evt_insider") is not None


def test_load_library_features_covers_builder_rows(tmp_path):
    lib, panels = _builder_style_lib(tmp_path)
    feats = lib.load_library_features()
    assert set(feats) == set(panels)


# ---- 显著因子加载（历史上整库抛异常的那条路径） ----

def test_load_significant_features_does_not_scan_whole_library(tmp_path):
    """混合库上不再抛异常，且只解析显著因子自己的面板。

    IC 用"同日收益"构造（mock 目的就是逼出 significant=True），
    注册后与两个扁平行共存，验证不会为它们付读盘成本。
    """
    lib, _ = _builder_style_lib(tmp_path)
    rets = _mock_panel(seed=7)
    noisy = rets + np.random.default_rng(9).normal(0, 1e-4, rets.shape)
    row = lib.register("sig_ok", noisy, rets, source="test:unit")
    assert bool(row["significant"]) is True
    feats = lib.load_significant_features(exclude_model=True)
    assert set(feats) == {"sig_ok"}
    assert set(lib.load_significant_features(correction="fdr")) == {"sig_ok"}


def test_missing_significant_column_is_tolerated(tmp_path):
    """纯批量库（连 significant / source 列都没有）不 KeyError，返回空特征集。"""
    lib, _ = _builder_style_lib(tmp_path)
    assert "significant" not in pd.read_csv(lib.root / "registry.csv").columns
    assert lib.load_significant_features(exclude_model=True) == {}
    assert lib.load_significant_features(correction="fdr") == {}


# ---- 冗余预检：覆盖面必须显式，不得静默跳过 ----

def test_dup_check_excludes_builder_rows_by_default(tmp_path):
    """默认不比批量入库因子 → 无可比对象时 dup_corr_max 为空（而不是静默假装比过）。"""
    lib, panels = _builder_style_lib(tmp_path, names=("evt_insider",))
    dup = panels["evt_insider"]
    row = lib.register("copy", dup, dup, check_dup=True)
    assert row["dup_corr_max"] == ""


def test_dup_include_builder_rows_expands_pool(tmp_path):
    """显式开启后批量入库因子参与比较——此前它们被静默跳过且 dup_checked 写 True。"""
    lib, panels = _builder_style_lib(tmp_path, names=("evt_insider",))
    dup = panels["evt_insider"]
    row = lib.register("copy", dup, dup, check_dup=True, dup_include_builder_rows=True)
    assert float(row["dup_corr_max"]) > 0.99
    assert row["dup_top"] == "evt_insider"


# ---- 标签回填（source 此前无任何回填入口） ----

def test_set_tag_writes_source(tmp_path):
    lib, _ = _builder_style_lib(tmp_path, names=("evt_insider",))
    assert lib.set_tag("evt_insider", source="build_alla_event_factors:20180701-20260716",
                       family="事件", maturity="experimental")
    r = lib.list_all()
    hit = r[r["name"] == "evt_insider"].iloc[0]
    assert hit["source"] == "build_alla_event_factors:20180701-20260716"
    assert hit["family"] == "事件"
    assert not lib.set_tag("不存在", source="x")


# ---- monitor / delete 对缺 eval 的行 ----

def test_monitor_tolerates_missing_evals(tmp_path):
    """扁平行没有 eval：monitor 返回空表而不是抛异常。"""
    lib, _ = _builder_style_lib(tmp_path)
    assert lib.monitor().empty


def test_delete_tolerates_builder_rows(tmp_path):
    """delete() 对扁平行不再 Path(nan) 抛异常，并真的删掉原名面板。"""
    lib, _ = _builder_style_lib(tmp_path, names=("evt_insider",))
    assert lib.delete("evt_insider") is True
    assert not (lib.panels_dir / "evt_insider.parquet").exists()
    assert not lib.has("evt_insider")
