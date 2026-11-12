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
    from backtest.costs import default_costs
    from strategy.examples import build_strategy

    bt = VectorBacktest(strategy=build_strategy("topk_lo", 5), rebalance_freq="W")
    prod = default_costs()
    assert bt.costs.commission_rate == pytest.approx(prod.commission_rate)
    assert bt.costs.stamp_duty == pytest.approx(prod.stamp_duty)
    assert bt.costs.slippage_bp == pytest.approx(prod.slippage_bp)


def test_default_costs_zero_for_precost_comparison():
    """factor_cost=False 置零（无成本对照口径），不受 costs 段影响。"""
    from backtest.costs import default_costs

    z = default_costs(factor_cost=False)
    assert z.commission_rate == 0.0 and z.stamp_duty == 0.0 and z.slippage_bp == 0.0


def test_default_costs_source_is_the_backtest_layer():
    """成本工厂真源在 ``backtest.costs``；scripts 层不得再定义（2026-09-11 下沉）。

    该函数原先定义在 ``scripts/pipelines/run_model_portfolio.py``，被主实验
    （``rolling_grid_alla``）、生产（``alla_daily_rank``）与多个实验脚本
    **反向 import**——实验入口脚本成了生产链路的依赖库。成本是回测层概念，
    归 ``backtest/``；谁再把它搬回 scripts 层，本测试立刻红。
    """
    import re

    from backtest.costs import default_costs  # noqa: F401  （真源存在性）

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for f in (root / "scripts").rglob("*.py"):
        if {"oneoff", "archive"} & set(f.parts):
            continue
        if re.search(r"^def default_costs\s*\(", f.read_text(encoding="utf-8"), re.M):
            offenders.append(str(f.relative_to(root)))
    assert not offenders, (
        "scripts 层重新定义了 default_costs（应 import backtest.costs）:\n"
        + "\n".join(offenders))


# ---------------------------------------------------------------------------
# 时变印花税（2026-11-11 P0）
# ---------------------------------------------------------------------------
#
# 背景：印花税此前硬编码千1（`stamp_duty: 0.001`），但 2023-08-28 起已减半至万5
# （财政部 税务总局公告 2023 年第 39 号，仍卖出单边）。项目回测窗口 2019-01~2026-09
# **跨政策窗口**，用单一常数无论取哪一个都会错：
#   - 取千1 ⇒ 2023-08 之后每笔多收 5bp；
#   - 取万5 ⇒ 2019~2023 段少收 5bp（偏乐观）。
# 故印花税改为**按调仓日时变**（`backtest.costs.stamp_duty_on`），本组测试锁死它。


def test_stamp_duty_schedule_boundaries():
    """政策边界日：2023-08-28 当日即为万5，前一日仍千1。"""
    from backtest.costs import stamp_duty_on

    assert stamp_duty_on(20230825) == pytest.approx(0.001)
    assert stamp_duty_on(20230827) == pytest.approx(0.001)
    assert stamp_duty_on(20230828) == pytest.approx(0.0005)   # 生效当日
    assert stamp_duty_on(20260907) == pytest.approx(0.0005)


def test_stamp_duty_accepts_all_date_types():
    """PIT 日历是 int YYYYMMDD：int 必须先转 str，否则 pd.Timestamp(20230828) 落 1970。"""
    import numpy as np
    import pandas as pd

    from backtest.costs import stamp_duty_on

    for d in (20230828, np.int64(20230828), "2023-08-28", "20230828",
              pd.Timestamp("2023-08-28")):
        assert stamp_duty_on(d) == pytest.approx(0.0005), f"{d!r} 税率取错"


def test_stamp_duty_none_falls_back_to_current_rate():
    """不传日期时兜底现行税率（万5），不是历史千1。"""
    from backtest.costs import stamp_duty_on

    assert stamp_duty_on(None) == pytest.approx(0.0005)


def test_calc_applies_date_dependent_stamp_duty():
    """开启 stamp_duty_schedule 后 calc 按日期取税率；未开启则用固定值。"""
    import numpy as np

    from backtest.costs import TransactionCosts

    c = TransactionCosts(commission_rate=0.0001, stamp_duty=0.0005, slippage_bp=10.0,
                         stamp_duty_schedule=True)
    w0, w1, cap = np.zeros(2), np.array([0.5, 0.5]), 100000.0
    old = c.calc(w0, w1, 1.0, cap, date=20190102)    # 千1 ⇒ 往返 32bp
    new = c.calc(w0, w1, 1.0, cap, date=20240102)    # 万5 ⇒ 往返 27bp
    assert old == pytest.approx(320.0), f"2023 前应 32bp，实得 {old / cap * 1e4:.1f}bp"
    assert new == pytest.approx(270.0), f"2023 后应 27bp，实得 {new / cap * 1e4:.1f}bp"
    # 差额正好 = 印花税差 5bp（单边换手 1.0 ⇒ 卖出额 = capital）
    assert old - new == pytest.approx(cap * 0.0005)


def test_fixed_stamp_duty_not_overridden_by_date():
    """🚨 回归守卫：未开启时变时，date **不得**覆盖显式给定的 stamp_duty。

    2026-11-11 踩过：时变逻辑曾无条件生效，导致
    `default_costs(factor_cost=False)`（零成本对照）被重新收上印花税，
    `test_backtest_engine` 两个净值精确性测试因净收益被扣税而失败。
    零成本对照的「零」必须是真零。
    """
    import numpy as np

    from backtest.costs import TransactionCosts, default_costs

    zero = TransactionCosts(commission_rate=0, commission_min=0,
                            stamp_duty=0, slippage_bp=0)
    fee = zero.calc(np.zeros(2), np.array([0.5, 0.5]), 1.0, 100000.0, date=20240102)
    assert fee == 0.0, "零成本对象在传 date 后仍产生了印花税"
    # default_costs(factor_cost=False) 同样必须为零
    z = default_costs(factor_cost=False)
    assert z.calc(np.zeros(2), np.array([0.5, 0.5]), 1.0, 100000.0,
                  date=20240102) == 0.0


def test_default_costs_enables_schedule():
    """生产口径 `default_costs()` 必须开启时变（跨政策窗口的唯一正确做法）。"""
    from backtest.costs import default_costs

    assert default_costs().stamp_duty_schedule is True
    assert default_costs(factor_cost=False).stamp_duty_schedule is False


def test_config_stamp_duty_is_current_rate():
    """配置里的 stamp_duty 是**现行**税率（万5）；千1 是 2023-08-28 前的历史值。"""
    assert Config.costs()["stamp_duty"] == pytest.approx(0.0005)
    assert Config.costs()["commission_rate"] == pytest.approx(0.0001)
