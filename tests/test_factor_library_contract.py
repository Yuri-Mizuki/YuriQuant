"""因子库契约测试（2026-11-02，与两段式入库同时建立）。

锁三件事——它们此前只写在文档里，没有任何机制保证：

1. **状态机**：``ingest`` 只产生 ``ingested``（无 canonical 指标），
   ``evaluate`` 才产生 ``evaluated``（有指标）。不允许"标了已评估但指标是 NaN"，
   也不允许"只说登记却残留旧回测数字"。
2. **正式因子必须有数**：``maturity ∈ {oos_verified, active}`` ⇒
   评估过 + ``t_stat_nw`` 有限 + 查过冗余。
3. **多进程回填的正确性**：``apply_eval_many`` 一次落盘后所有行都到位。

真实库的集成检查用 skipif 守护：库不在本机/没有数据集时跳过，不让 CI 假红。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.factor_library import FactorLibrary, name_slug


def _panel_and_returns(n_dates: int = 260, n_codes: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_dates)
    cols = [f"{i:06d}.SZ" for i in range(n_codes)]
    panel = pd.DataFrame(rng.normal(size=(n_dates, n_codes)), index=idx, columns=cols)
    rets = pd.DataFrame(rng.normal(0, 0.01, size=(n_dates, n_codes)),
                        index=idx, columns=cols)
    return panel, rets


# ---------------------------------------------------------------------------
# 1. 状态机
# ---------------------------------------------------------------------------
def test_ingest_leaves_canonical_metrics_empty(tmp_path):
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()
    row = lib.ingest("f_ing", panel, rets, family="量价", frequency="日频",
                     evidence=False)

    assert row["eval_status"] == "ingested"
    assert str(row["evaluated_at"]).strip() == ""
    assert np.isnan(float(row.get("sharpe_ls_M", np.nan)))
    assert np.isfinite(row["ic_mean"])
    # 只登记也要有 eval 文件（里面至少要有 ic 列），否则时间段切片无从谈起
    ev = pd.read_parquet(tmp_path / "t" / "evals" / f"{name_slug(row['name'])}.parquet")
    assert "ic" in ev.columns
    assert "dret_ls_M" not in ev.columns


def test_evaluate_fills_metrics_and_flips_status(tmp_path):
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()
    lib.ingest("f_eval", panel, rets, evidence=False)
    upd = lib.evaluate("f_eval", rets)

    assert np.isfinite(upd["sharpe_ls_M"])
    row = lib.list_all().set_index("name").loc["f_eval"]
    assert row["eval_status"] == "evaluated"
    assert str(row["evaluated_at"]).strip() != ""
    ev = pd.read_parquet(tmp_path / "t" / "evals" / f"{name_slug('f_eval')}.parquet")
    assert {"ic", "dret_ls_M", "equity_ls_M"} <= set(ev.columns)


def test_evaluate_without_ingest_raises(tmp_path):
    lib = FactorLibrary(root=tmp_path, dataset="t")
    with pytest.raises(KeyError):
        lib.evaluate("nobody", _panel_and_returns()[1])


# ---------------------------------------------------------------------------
# 2. 正式因子必须有数
# ---------------------------------------------------------------------------
def test_formal_maturity_without_eval_is_violation(tmp_path):
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()
    # 先正常登记一个"正式"因子（合规）
    lib.register("f_ok", panel, rets, maturity="active", check_dup=True)
    assert lib.contract_violations() == []

    # 再造一个只有登记、却标成 active 的因子（违规）
    lib.ingest("f_bad", panel + 0.01, rets, maturity="active", evidence=False)
    rules = {v["rule"] for v in lib.contract_violations() if v["name"] == "f_bad"}
    assert "formal_without_eval" in rules
    assert "formal_without_dup_check" in rules


def test_ingested_row_with_stale_metrics_is_violation(tmp_path):
    """只登记却带着旧回测数字——"看起来评估过"，是最坏的一种半写态。"""
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()
    lib.register("f_x", panel, rets)
    reg = lib._load_registry()
    reg.loc[reg["name"] == "f_x", "eval_status"] = "ingested"
    lib._save_registry(reg)

    rules = {v["rule"] for v in lib.contract_violations()}
    assert "ingested_with_metrics" in rules


def test_evaluated_with_missing_eval_file_is_violation(tmp_path):
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()
    lib.register("f_y", panel, rets)
    (tmp_path / "t" / "evals" / f"{name_slug('f_y')}.parquet").unlink()

    rules = {v["rule"] for v in lib.contract_violations()}
    assert "eval_file_missing" in rules


# ---------------------------------------------------------------------------
# 3. 批量回填
# ---------------------------------------------------------------------------
def test_apply_eval_many_writes_all_rows_once(tmp_path):
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()
    names = [f"m{i}" for i in range(3)]
    for i, nm in enumerate(names):
        lib.ingest(nm, panel + i * 0.01, rets, evidence=False)

    mapping = {nm: lib.evaluate(nm, rets, persist=False) for nm in names}
    assert lib.apply_eval_many(mapping) == 3

    reg = lib._load_registry().set_index("name")
    for nm in names:
        assert reg.loc[nm, "eval_status"] == "evaluated"
        assert np.isfinite(float(reg.loc[nm, "sharpe_ls_M"]))
    assert lib.contract_violations() == []


# ---------------------------------------------------------------------------
# 4. 真实库集成（本机有库才跑）
# ---------------------------------------------------------------------------
def _real_lib(dataset: str = "all_a_2018_2026"):
    try:
        return FactorLibrary(dataset=dataset)
    except Exception:  # noqa: BLE001
        return None


def test_real_library_has_no_contract_violations():
    lib = _real_lib()
    if lib is None or not lib._registry_path.exists():
        pytest.skip("本机无该数据集")
    bad = lib.contract_violations()
    # 存量老库允许尚未回填的 ingested 行，但不能出现半写态或正式因子无 t 值
    fatal = [v for v in bad if v["rule"] in {
        "evaluated_but_nan", "eval_file_missing", "formal_without_eval",
        "formal_without_t", "ingested_with_metrics"}]
    assert not fatal, fatal[:10]


# 4. 口径：可交易掩码只能作用于回测，不许污染落盘面板
# ---------------------------------------------------------------------------
def test_evaluate_with_mask_does_not_touch_stored_panel(tmp_path):
    """套掩码后库里存的面板必须逐位不变。

    2026-11-04 把 ``evaluate`` 改成"回测前把不可交易格子置空"。实现里用的是
    ``panel.where(mask)``（返回新对象），但这属于**一句话就能写错**的地方——
    写成 ``panel[mask] = np.nan`` 就会把不可交易日的因子值永久抹掉，而
    ``dup_audit`` / 相关性 / 重新评估全都读这份面板，错误会静默扩散。
    """
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()
    before = panel.copy()
    lib.ingest("f_mask", panel, rets, evidence=False)
    lib.evaluate("f_mask", rets, use_mask=True)
    after = lib.get_panel("f_mask")

    assert after.shape == before.shape
    assert np.allclose(after.to_numpy(), before.to_numpy(), equal_nan=True)


def test_evaluate_records_basis_columns(tmp_path):
    """``ret_basis`` / ``mask_basis`` 必须如实登记——"没套上"不许看起来像"套上了"。"""
    lib = FactorLibrary(root=tmp_path, dataset="t")
    panel, rets = _panel_and_returns()

    lib.ingest("f_nomask", panel, rets, evidence=False)
    lib.evaluate("f_nomask", rets, use_mask=False)
    row = lib.list_all().set_index("name").loc["f_nomask"]
    # 调用方自带收益面板 ⇒ 口径只能标 caller_supplied，不能冒充库内复权口径
    assert row["ret_basis"] == "caller_supplied"
    assert str(row["mask_basis"]).strip() in ("", "nan")

    # 股票池与真实市场不重叠（显式用假代码）⇒ 不该套真实掩码，mask_basis 保持空。
    # 注意不能用 000001.SZ 这类真代码：它们会命中真实掩码，"回退"这条路径就测不到了。
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=120)
    codes = [f"MOCK{i:04d}.SZ" for i in range(40)]
    fake = pd.DataFrame(rng.normal(size=(120, 40)), index=idx, columns=codes)
    fake_r = pd.DataFrame(rng.normal(0, 0.01, size=(120, 40)), index=idx, columns=codes)
    lib.ingest("f_fake", fake, fake_r, evidence=False)
    lib.evaluate("f_fake", fake_r, mask_lag=0)
    row2 = lib.list_all().set_index("name").loc["f_fake"]
    assert str(row2["mask_basis"]).strip() in ("", "nan")


def test_returns_basis_survives_cache_hit(monkeypatch):
    """收益面板命中进程级缓存时，口径标记仍必须是复权。

    2026-11-04 实测踩到：``ret_basis`` 原本靠 ``self._adj_cache`` 是否已填来判断，
    而 ``default_returns_panel()`` 命中 ``_RP_CACHE`` 时**提前返回、不碰 adj**，
    于是"每个复用进程的第一个因子"被标成 ``close_pct_v1``（数据其实是复权的，
    只有标签错）。全库 965 个里没暴露，三个子集里冒出 11 行——正是这类
    "只在进程边界出现"的 bug 最难查。
    """
    import research.factor_library as fl
    from config import Config

    fake = pd.DataFrame(np.zeros((3, 3)))
    monkeypatch.setattr(Config, "cache", classmethod(lambda cls: {"root": "FAKE_ROOT"}))
    fl._RP_CACHE["FAKE_ROOT"] = fake
    fl._RP_BASIS["FAKE_ROOT"] = "adj_close_pct_v1"
    try:
        lib = FactorLibrary(root="/tmp/none", dataset="t")
        assert lib.default_returns_panel() is fake
        assert getattr(lib, "_ret_basis", None) == "adj_close_pct_v1"
    finally:
        fl._RP_CACHE.pop("FAKE_ROOT", None)
        fl._RP_BASIS.pop("FAKE_ROOT", None)
