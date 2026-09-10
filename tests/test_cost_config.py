"""交易成本唯一真源的回归测试（2026-09-10 收敛）。

背景：此前费率分散在 `config/settings.yaml` 两处——`backtest` 段（佣金万1 / 滑点 5bp，
引擎缺省值）与 `model_portfolio` 段（万3 / 10bp，生产管线 default_costs()）。两套数字
差 2~3 倍，导致"网格实验选出的最优参数不能用引擎默认复跑"，跨脚本数字不可比。

收敛后唯一真源 = 顶层 `costs` 段；引擎缺省与生产管线都读它。本文件锁死这一点：
谁再把费率搬回去、或在两处各写一份，测试立刻红。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import Config

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

_RATE_KEYS = ("commission_rate", "commission_min", "stamp_duty", "slippage_bp")
_SHORT_KEYS = ("short_borrow_rate", "short_margin_ratio")


@pytest.fixture(scope="module")
def raw_settings() -> dict:
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_costs_section_exists_and_complete(raw_settings):
    """顶层 `costs` 段必须存在且含全部费率键。"""
    assert "costs" in raw_settings, "settings.yaml 缺少顶层 costs 段（交易成本唯一真源）"
    costs = raw_settings["costs"]
    for k in _RATE_KEYS + _SHORT_KEYS:
        assert k in costs, f"costs 段缺少 {k}"


def test_no_cost_keys_left_in_other_sections(raw_settings):
    """费率不得再出现在 backtest / model_portfolio 段（防止回到双源）。"""
    for section in ("backtest", "model_portfolio"):
        blk = raw_settings.get(section) or {}
        stray = [k for k in blk if "commission" in k or "slippage" in k
                 or "stamp" in k or "borrow" in k or "margin_ratio" in k]
        assert not stray, f"{section} 段仍残留费率键 {stray}；费率只应出现在顶层 costs 段"


def test_config_costs_accessor_matches_yaml(raw_settings):
    """Config.costs() 访问器（单一真源的类型化出口）与 yaml 原值一致。

    防的是"yaml 改了、访问器默认值没跟上"这类漂移——引擎与生产管线都走访问器，
    访问器与 yaml 不一致时两边会静默用不同费率。
    """
    acc = Config.costs()
    raw = raw_settings["costs"]
    for k in _RATE_KEYS + _SHORT_KEYS:
        assert k in acc, f"Config.costs() 缺少 {k}"
        assert acc[k] == pytest.approx(float(raw[k])), f"{k} 访问器与 yaml 不一致"


def test_engine_default_costs_match_config():
    """引擎未显式传 costs 时的缺省费率 == 配置里的唯一真源。"""
    from backtest import VectorBacktest
    from strategy.examples import build_strategy

    bt = VectorBacktest(strategy=build_strategy("topk_lo", 5), rebalance_freq="W")
    cfg = Config.get()["costs"]
    assert bt.costs.commission_rate == pytest.approx(cfg["commission_rate"])
    assert bt.costs.commission_min == pytest.approx(cfg["commission_min"])
    assert bt.costs.stamp_duty == pytest.approx(cfg["stamp_duty"])
    assert bt.costs.slippage_bp == pytest.approx(cfg["slippage_bp"])


def test_engine_default_short_costs_match_config():
    """空头腿成本同样读 costs 段（借券费/保证金比例）。"""
    from backtest import VectorBacktest
    from strategy.examples import build_strategy

    bt = VectorBacktest(strategy=build_strategy("topk_lo", 5), rebalance_freq="W")
    cfg = Config.get()["costs"]
    assert bt.short_costs.borrow_rate == pytest.approx(cfg["short_borrow_rate"])


def test_production_pipeline_and_engine_agree():
    """两条取费率的路（引擎缺省 / 生产管线 default_costs）必须给出同一组数字。

    这是收敛的核心诉求：网格实验（走 default_costs）选出的参数，
    必须能用引擎默认原样复跑。
    """
    from backtest import VectorBacktest
    from scripts.run_model_portfolio import default_costs
    from strategy.examples import build_strategy

    bt = VectorBacktest(strategy=build_strategy("topk_lo", 5), rebalance_freq="W")
    prod = default_costs()
    assert bt.costs.commission_rate == pytest.approx(prod.commission_rate)
    assert bt.costs.stamp_duty == pytest.approx(prod.stamp_duty)
    assert bt.costs.slippage_bp == pytest.approx(prod.slippage_bp)


def test_default_costs_zero_for_precost_comparison():
    """factor_cost=False 置零（无成本对照口径），不受 costs 段影响。"""
    from scripts.run_model_portfolio import default_costs

    z = default_costs(factor_cost=False)
    assert z.commission_rate == 0.0 and z.stamp_duty == 0.0 and z.slippage_bp == 0.0
