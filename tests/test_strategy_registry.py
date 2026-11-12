"""`strategy.registry`（yqcore 族）的契约测试。

锁三件事：

1. **口径数字只有一个来源** —— 构造参数 / 信号键 / 所需面板全部从
   `config/baselines.yaml` 读，代码里不留第二份。故这里**不**断言「k 等于 15」
   这类字面量（那会变成第三个真源），而是用**注入的假登记表**证明「登记表改，
   注册表跟着变」，再断言产物落在合理区间。
2. **登记表与代码双向对得上** —— 登记了但代码没定义（缺标签/资金边界）、代码要有
   但登记表缺必填参数，都在 import 期报错（`assert_no_orphan_registration` /
   `_REQUIRED_PARAMS`）。这条防的是「代码改了名、登记没跟」，2026-11-03 的
   `cross_section` 漏进 git 就是同类事故。
3. **构造行为** —— 面板注入、overrides、每次 build 返回新实例（缓冲带有 `_prev`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.registry import (  # noqa: E402
    FAMILY,
    TIERS,
    assert_no_orphan_registration,
    build,
    describe,
    get_tier,
    tier_wiring,
)


def _registry() -> dict:
    """独立读一份登记表（**不**经 registry 的缓存，用于真源对拍）。"""
    return yaml.safe_load((ROOT / "config" / "baselines.yaml").read_text(encoding="utf-8"))


def _entry(reg: dict, baseline_id: str) -> dict:
    return next(b for b in reg["baselines"] if b["id"] == baseline_id)


def test_family_has_three_tiers():
    """族内三档 = 三种 alpha 形状 × 资金规模，不是版本更替。"""
    assert FAMILY == "yqcore"
    assert set(TIERS) == {"yqcore-breadth", "yqcore-focus", "yqcore-solo"}
    assert len({t.name for t in TIERS.values()}) == 3


def test_solo_tier_params():
    """solo = 行业去重档；param 与 `SectorCappedBufferedTopKLongOnly` 的签名对得上。"""
    t = get_tier("yqcore-solo")
    assert t.signal_key == "gbdt_mid13"
    assert t.strategy == "SectorCappedBufferedTopKLongOnly"
    # 值来自登记表（下面有专测「改登记表就跟着变」），此处钉**键集**：
    # 缺一个就会悄悄落回构造器缺省值（k 缺省 45、exit_mult 缺省 1.5）。
    assert set(t.params) == {"k", "exit_mult", "max_per_sector", "min_sectors",
                             "gross_exposure"}
    assert t.params["max_per_sector"] == 1 and t.params["min_sectors"] == 3
    assert (t.panel, t.panel_arg) == ("industry", "sectors")


def test_solo_requires_explicit_panel():
    """注册表不读数据：缺面板必须报错，且错误信息告诉调用方怎么传。"""
    with pytest.raises(TypeError, match="industry"):
        build("yqcore-solo")
    sig, st = build("yqcore-solo", industry=pd.DataFrame({"a": ["x"]}))
    assert sig == "gbdt_mid13"
    assert st.k == 8 and st.max_per_sector == 1
    # 面板名会被映射到构造器参数 sectors
    assert st.sectors.shape == (1, 1)


def test_focus_tier_params():
    t = get_tier("yqcore-focus")
    assert t.signal_key == "gbdt_mid13"
    assert t.strategy == "BufferedTopKLongOnly"
    assert set(t.params) == {"k", "exit_mult"}
    assert t.params["k"] >= 1 and t.params["exit_mult"] >= 1.0


def test_wide_tier_params():
    t = get_tier("yqcore-breadth")
    assert t.signal_key == "dingban_fundind"
    assert t.strategy == "BufferedTopFracLongOnly"
    assert set(t.params) == {"frac_entry", "frac_exit"}
    assert 0 < t.params["frac_entry"] <= t.params["frac_exit"] <= 1


def test_tier_params_are_read_from_registry():
    """注入假登记表：改数字 → 注册表跟着变。证明代码里没有第二份口径。

    这是「消双真源」的**反向验证**：若哪天有人把 k=15 抄回 `registry.py`，
    这里会立刻红。
    """
    reg = _registry()
    entry = _entry(reg, "yqcore-focus")
    entry["execution"]["k"] = 25
    entry["execution"]["exit_mult"] = 1.5
    entry["signal_key"] = "fake_signal_key"

    w = tier_wiring("yqcore-focus", reg)
    assert w["params"] == {"k": 25, "exit_mult": 1.5}
    assert w["signal_key"] == "fake_signal_key"
    # 真表不受影响（注入的是副本）
    assert get_tier("yqcore-focus").params["k"] != 25


def test_buffer_keys_map_to_constructor_kwargs():
    """登记表用 buffer_entry/buffer_exit，构造器用 frac_* —— 别名映射要双向都对。"""
    reg = _registry()
    entry = _entry(reg, "yqcore-breadth")
    entry["execution"]["buffer_entry"] = 0.20
    entry["execution"]["buffer_exit"] = 0.30
    assert tier_wiring("yqcore-breadth", reg)["params"] == {
        "frac_entry": 0.20, "frac_exit": 0.30}


def test_unknown_execution_param_raises():
    """登记表里写错参数名要报错，不能静默忽略（否则口径悄悄没生效）。"""
    reg = _registry()
    _entry(reg, "yqcore-focus")["execution"]["k_typo"] = 15
    with pytest.raises(ValueError, match="k_typo"):
        tier_wiring("yqcore-focus", reg)


def test_missing_required_param_raises():
    """必填参数被删要报错，不能悄悄落回构造器缺省值（TopK 缺省 k=45）。"""
    reg = _registry()
    del _entry(reg, "yqcore-focus")["execution"]["k"]
    with pytest.raises(ValueError, match="k"):
        tier_wiring("yqcore-focus", reg)


def test_orphan_registration_raises():
    """反向：登记了却没在代码里定义（缺标签/资金边界）必须报错。"""
    reg = _registry()
    reg["baselines"].append({"id": "yqcore-ghost", "family": "yqcore",
                             "signal_key": "x",
                             "execution": {"strategy": "BufferedTopKLongOnly"}})
    with pytest.raises(ValueError, match="yqcore-ghost"):
        assert_no_orphan_registration(reg)
    # 真实登记表不得有孤儿（import 时已跑过一次，这里显式再钉一次）
    assert_no_orphan_registration(_registry())


def test_build_returns_runnable_signal_and_fresh_instance():
    """返回的信号键必须是本机可读的那个；两次 build 不能共享状态（缓冲带有 _prev）。"""
    sig, s1 = build("yqcore-focus")
    assert sig == "gbdt_mid13"
    _, s2 = build("yqcore-focus")
    assert s1 is not s2
    assert s1._prev is not s2._prev


def test_wide_uses_local_proxy_signal():
    """wide 的登记 pred 本机缺失，`build` 必须给代理键，否则脚本读不到面板。"""
    sig, _ = build("yqcore-breadth")
    assert sig == "dingban_fundind_local"
    assert get_tier("yqcore-breadth").signal_key == "dingban_fundind"


def test_overrides_apply():
    base_k = get_tier("yqcore-focus").params["k"]      # 登记值，不在测试里写死
    sig, s = build("yqcore-focus", k=25)
    assert sig == "gbdt_mid13"
    assert s.k == 25
    assert get_tier("yqcore-focus").params["k"] == base_k   # 原件不被改


def test_unknown_tier_raises():
    with pytest.raises(KeyError):
        get_tier("yqcore-nonexistent")


def test_describe_mentions_signal():
    assert "gbdt_mid13" in describe("yqcore-focus")


def test_registry_entries_are_production_and_complete():
    """登记的档位口径必须完整：production + 口径三件套（成本/成交价/资金）。

    注：**参数值不再在这里对拍** —— 注册表直接从登记表取数，「两边相等」是构造出来的。
    值层面的真实风险（写错参数名、漏必填项）由
    ``test_unknown_execution_param_raises`` / ``test_missing_required_param_raises`` 覆盖。
    """
    reg = _registry()
    fam = [b for b in reg["baselines"] if b.get("family") == FAMILY]
    assert {b["id"] for b in fam} == set(TIERS), "登记表与 registry 的档位集合不一致"

    for b in fam:
        assert b["status"] == "production", f"{b['id']} 应为 production"
        assert b["signal_key"] == TIERS[b["id"]].signal_key
        assert b["caliber"]["profile"] == "production"
        assert b["caliber"]["source_script"], f"{b['id']} 缺报数脚本"
        if b.get("requires_panel"):
            assert b["requires_panel"] == TIERS[b["id"]].panel
        if b["id"] == "yqcore-solo":
            # 小资金档是 1e5 + 整手口径，必须显式登记（否则会被当成 1e6 读）
            assert b["caliber"]["lot_size"] == 100
            assert b["caliber"]["capital"] == 1.0e5


def test_metrics_present_and_plausible():
    """登记的数字必须在合理区间，且算术/几何 Sharpe 都不缺（跨策略比较用算术）。"""
    for b in _registry()["baselines"]:
        if b.get("family") != FAMILY:
            continue
        prof = b["metrics"]["profiles"]["production"]
        arm = next(iter(prof.values()))
        assert 0.0 < arm["ann"] < 1.0
        assert 0.0 < arm["sharpe"] < 3.0
        assert "sharpe_arith" in arm
        assert arm["sharpe_t_stat"] > 1.96, "定版档位的 NW-t 必须过 1.96"
        assert 0.0 < arm["max_dd"] < 1.0
