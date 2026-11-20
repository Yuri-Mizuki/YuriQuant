"""策略目录（``strategy/catalog.py``）与参数真源的一致性。

2026-11-19 轻统一的边界：``CATALOG`` **只登记**「有哪些策略、入口在哪」，
实现与参数仍在 ``etf_contest_spec`` / ``strategy.registry``。本测试钉住：
新增/退役一档时，两边不许漂移（否则前端首页会漏卡或多卡）。
"""
from __future__ import annotations


def test_every_etf_profile_is_registered():
    from scripts.portfolio.etf_champ_signals import PROFILES
    from strategy.catalog import CATALOG

    etf = {e.profile for e in CATALOG.values() if e.kind == "etf"}
    assert etf == set(PROFILES), (
        f"ETF 档位漂移：catalog={sorted(map(str, etf))} vs PROFILES={sorted(PROFILES)}")
    for e in CATALOG.values():
        if e.kind == "etf":
            assert e.page and e.card, f"{e.key} 缺前端详情页或缺首页卡片"


def test_every_yqcore_tier_is_registered():
    from strategy.catalog import CATALOG
    from strategy.registry import DISPLAY_ORDER, TIERS

    alpha = {e.key for e in CATALOG.values() if e.kind == "alpha"}
    assert alpha == set(TIERS), (
        f"股票线档位漂移：catalog={sorted(alpha)} vs registry={sorted(TIERS)}")
    assert set(DISPLAY_ORDER) == set(TIERS), "DISPLAY_ORDER 漏登记档位"
    assert len(DISPLAY_ORDER) == len(TIERS), "DISPLAY_ORDER 有重复档位"


def test_catalog_entries_are_complete_and_ordered():
    from strategy.catalog import CATALOG, cards

    for e in CATALOG.values():
        assert e.key and e.label and e.runner and e.report_dir and e.href, (
            f"{e.key} 登记不完整")
    ordered = cards()
    assert [e.order for e in ordered] == sorted(e.order for e in ordered)
    assert len({e.key for e in ordered}) == len(ordered)


def test_event_driven_entry_and_rule_single_source():
    """事件驱动登记在案，且实盘出单器的事件口径从策略层导入（不许各写一份）。"""
    from strategy.catalog import CATALOG
    from strategy.event_driven import ACTIVE_REASONS, HOLD, MIN_RATIO

    e = CATALOG.get("inc_pure")
    assert e is not None and e.kind == "event" and e.card and e.page, (
        "事件驱动未登记为首页卡片 / 缺详情页")

    import scripts.live.inc_pure_live as live

    assert (live.HOLD, live.MIN_RATIO) == (HOLD, MIN_RATIO)
    assert tuple(live.ACTIVE) == tuple(ACTIVE_REASONS), "实盘成交方式与策略层漂移"
    assert live._sym is __import__(
        "strategy.event_driven", fromlist=["normalize_code"]).normalize_code, (
        "实盘代码归一化未复用策略层实现")
