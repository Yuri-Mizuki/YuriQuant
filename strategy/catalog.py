"""策略目录（全仓唯一的策略登记表）

**只登记「有哪些策略、入口在哪」，不登记实现。** 参数真源仍在各自模块：

- ETF：``scripts/portfolio/etf_contest_spec.py``（``PROFILES`` / ``RETIRED_CORE``）
- 股票线：``strategy/registry.py``（``TIERS``）+ ``config/baselines.yaml``
- 事件驱动：``strategy/event_driven.py``（事件定义 + 日频组合回测）

为什么要有这层：一条策略的名字/入口/报告目录此前散在代码、README、前端文案三处，
加一档或改名必漏一处。``scripts/reporting/strategy_home.py`` 的首页卡片按本表生成；
新增策略只在 ``CATALOG`` 加一条，前端不再手抄。

**不统一策略接口**：ETF 是端到端策略（选标 + 加权 + 波动目标都在类里），股票线是
纯排序壳（alpha 由 ``model/`` + ``factor/`` 产出）。两者共用 ``strategy.base.Strategy``
与 ``backtest/`` 引擎已足够，强行抽象同一套参数接口只会把两边都弄脏。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyEntry:
    """一条可生产/可展示的策略。"""

    key: str                  # 唯一键；ETF 也是前端详情页文件名（不含 .html）
    label: str                # 展示名（前端卡片标题）
    kind: str                 # "etf" | "alpha" | "event"
    runner: str               # 生产入口命令（信息性，给运维照抄）
    report_dir: str           # 产物目录，相对仓库根
    href: str                 # 入口页，相对 ``reports/``
    page: str | None = None   # 独立详情页（相对 ``reports/strategies/``），无则 None
    profile: str | None = None  # ETF：``etf_champ_signals.PROFILES`` 的键
    card: bool = False        # 是否在策略首页出卡片
    order: int = 100          # 卡片顺序


CATALOG: dict[str, StrategyEntry] = {
    "etf_monthly": StrategyEntry(
        key="etf_monthly", label="ETF 月频稳健", kind="etf", order=10, card=True,
        profile="monthly",
        runner="python -m scripts.portfolio.etf_champ_signals --profile monthly",
        report_dir="reports/etf_champ",
        href="strategies/etf_monthly.html", page="etf_monthly.html",
    ),
    "etf_balanced": StrategyEntry(
        key="etf_balanced", label="ETF 均衡轮动", kind="etf", order=20, card=True,
        profile="balanced",
        runner="python -m scripts.portfolio.etf_champ_signals --profile balanced",
        report_dir="reports/etf_contest5",
        href="strategies/etf_balanced.html", page="etf_balanced.html",
    ),
    "yqcore-focus-k8": StrategyEntry(
        key="yqcore-focus-k8", label="股票多因子 · K8", kind="alpha", order=30,
        card=True,
        runner="python -m scripts.live.yqcore --preset dingban",
        report_dir="reports/yq_core",
        href="strategies/yqcore-focus-k8.html",
        page="yqcore-focus-k8.html",
    ),
    "yqcore-focus-k15": StrategyEntry(
        key="yqcore-focus-k15", label="股票多因子 · K15", kind="alpha", order=31,
        card=True,
        runner="python -m scripts.live.yqcore --preset dingban",
        report_dir="reports/yq_core",
        href="strategies/yqcore-focus-k15.html",
        page="yqcore-focus-k15.html",
    ),
    "yqcore-breadth": StrategyEntry(
        key="yqcore-breadth", label="股票多因子 · 广度档", kind="alpha", order=32,
        card=True,
        runner="python -m scripts.live.yqcore --preset dingban",
        report_dir="reports/yq_core",
        href="strategies/yqcore-breadth.html",
        page="yqcore-breadth.html",
    ),
    "inc_pure": StrategyEntry(
        key="inc_pure", label="事件驱动 · 董监高增持", kind="event", order=40, card=True,
        runner="python -m scripts.live.inc_pure_live signal --asof <T> --capital 4000000",
        report_dir="reports/inc_pure_live",
        href="strategies/inc_pure.html", page="inc_pure.html",
    ),
}


def cards() -> list[StrategyEntry]:
    """首页要出卡片的策略（按 ``order``）。"""
    return sorted((e for e in CATALOG.values() if e.card), key=lambda e: e.order)


__all__ = ["CATALOG", "StrategyEntry", "cards"]
