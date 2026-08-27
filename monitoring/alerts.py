"""告警规则引擎 —— 阈值来自 config monitoring 段（Config.monitoring()）。

规则一览（level: warning / critical）：
- stale_data        面板落后数据源 > max_stale_days 交易日（warning）
- coverage_drop     近期窗口覆盖率 < min_coverage（warning）
- ic_decay          近期 IC 保留率不足：因子 vs 全期、模型 vs 上线基线
                    短窗（60d，灵敏）提前预警；长窗（252d）未确认时标记"观察中"。
                    （warning：retention < warn_ic_retention；critical：方向翻转）
- significance_loss 模型近期 NW-t < min_t_nw_recent（warning）
- monotonicity_break 分位单调性恶化：模型 signed < min_monotonicity，
                    因子 |signed| < min_monotonicity（warning）——
                    分位结构恶化先于 IC 归零（实验报告 8.3 退出预警）。
- style_rotation     raw IC 衰减但中性化 IC 稳定 → 风格轮动非 alpha 衰减
                    （warning，操作含义"重新中性化"而非"因子失效"）。
                    风格暴露占比过高（style_exposure_ratio > 0.5）时同发。
"""

from __future__ import annotations

from typing import Any

from monitoring.metrics import MonitorMetrics, _pick_baseline

_LEVELS = {"normal": 0, "warning": 1, "critical": 2}


def _ok(v: float) -> bool:
    return v == v  # 非 NaN


def evaluate_alerts(m: MonitorMetrics, cfg: dict[str, Any]) -> list[dict]:
    """对单个监控快照触发全部规则，返回 [{rule, level, message}]。"""
    alerts: list[dict] = []

    if m.stale_days > int(cfg.get("max_stale_days", 7)):
        alerts.append(
            {
                "rule": "stale_data",
                "level": "warning",
                "message": f"面板落后数据源 {m.stale_days} 个交易日（阈值 "
                f"{cfg.get('max_stale_days', 7)}），需重算因子",
            }
        )

    if _ok(m.coverage_recent) and m.coverage_recent < float(cfg.get("min_coverage", 0.5)):
        alerts.append(
            {
                "rule": "coverage_drop",
                "level": "warning",
                "message": f"近期覆盖率 {m.coverage_recent:.2f} 低于下限 "
                f"{cfg.get('min_coverage', 0.5)}",
            }
        )

    if _ok(m.ic_mean_recent) and _ok(m.ic_mean_full) and m.recent_n_days >= 20:
        baseline = _pick_baseline(m)
        if _ok(baseline) and abs(baseline) > 1e-12:
            retention = m.ic_mean_recent / baseline
            thr = float(cfg.get("warn_ic_retention", 0.5))
            # 长窗（252d）确认态：短窗已跌破预警、长窗仍稳健 → 观察中未确认
            retention_252 = (
                m.ic_mean_recent_252 / baseline if _ok(m.ic_mean_recent_252) else float("nan")
            )
            observing = _ok(retention_252) and retention_252 >= thr
            if m.ic_mean_recent * baseline < 0:
                alerts.append(
                    {
                        "rule": "ic_decay",
                        "level": "critical",
                        "message": (
                            f"近{m.suffix_short}d IC {m.ic_mean_recent:+.4f} 与基线 "
                            f"{baseline:+.4f} 方向翻转"
                        ),
                    }
                )
            elif retention < thr:
                alerts.append(
                    {
                        "rule": "ic_decay",
                        "level": "warning",
                        "message": (
                            f"近{m.suffix_short}d IC 保留率 {retention:.0%}"
                            f"（{m.ic_mean_recent:+.4f} / {baseline:+.4f}），低于下限 {thr:.0%}"
                            + (
                                f"；近{m.suffix_long}d 保留率 {retention_252:.0%} 仍稳健"
                                f" → 观察中，尚未长窗确认"
                                if observing
                                else f"；近{m.suffix_long}d 保留率 {retention_252:.0%} 同步衰减，予以确认"
                            )
                        ),
                    }
                )

    if m.category == "model" and _ok(m.ic_t_nw_recent):
        if m.ic_t_nw_recent < float(cfg.get("min_t_nw_recent", 1.0)):
            alerts.append(
                {
                    "rule": "significance_loss",
                    "level": "warning",
                    "message": f"近期 NW-t {m.ic_t_nw_recent:.2f} 低于下限 "
                    f"{cfg.get('min_t_nw_recent', 1.0)}，显著性丢失",
                }
            )

    if _ok(m.monotonicity_recent):
        threshold = float(cfg.get("min_monotonicity", 0.5))
        signed = m.monotonicity_recent
        if m.category == "model":
            broken = signed < threshold
            direction = "正向"
        else:
            broken = abs(signed) < threshold
            direction = "|方向无关|"
        if broken:
            alerts.append(
                {
                    "rule": "monotonicity_break",
                    "level": "warning",
                    "message": f"近期分位单调性 {direction} {signed:+.2f} 低于下限 {threshold}，"
                    f"分位结构恶化往往先于 IC 归零",
                }
            )

    # 风格轮动 vs alpha 衰减区分（中性化 IC 传入时才触发）
    if _ok(m.ic_neutral_mean_recent) and _ok(m.ic_mean_recent):
        # 风格暴露占比过高（raw IC 中 > 50% 来自风格）
        if _ok(m.style_exposure_ratio) and m.style_exposure_ratio > 0.5:
            alerts.append(
                {
                    "rule": "style_rotation",
                    "level": "warning",
                    "message": (
                        f"raw IC 中风格可解释比例 {m.style_exposure_ratio:.0%}"
                        f"（中性化 IC {m.ic_neutral_mean_recent:+.4f}"
                        f" vs raw IC {m.ic_mean_recent:+.4f}），"
                        f"因子主体为风格暴露而非纯 alpha"
                    ),
                }
            )
        # raw IC 衰减但中性化 IC 稳定 → 风格轮动而非因子失效
        elif (_ok(m.ic_neutral_drift) and _ok(m.ic_drift)
                and abs(m.ic_drift) > 0.01
                and abs(m.ic_neutral_drift) < 0.005):
            alerts.append(
                {
                    "rule": "style_rotation",
                    "level": "warning",
                    "message": (
                        f"raw IC 漂移 {m.ic_drift:+.4f} 但中性化 IC 漂移仅 "
                        f"{m.ic_neutral_drift:+.4f}，属于风格轮动"
                        f"（重新中性化可恢复，非因子失效）"
                    ),
                }
            )

    if m.category == "signal":
        _append_signal_alerts(m, cfg, alerts)

    if m.category == "crowd":
        _append_crowding_alerts(m, cfg, alerts)

    return alerts


def _append_crowding_alerts(m: MonitorMetrics, cfg: dict[str, Any], alerts: list[dict]) -> None:
    """库级拥挤度告警：因子 IC 高度同质化 → 分散度虚假。"""
    if _ok(m.corr_mean_full) and m.corr_mean_full > float(cfg.get("max_corr_mean", 0.5)):
        alerts.append({
            "rule": "factor_crowding",
            "level": "warning",
            "message": f"因子 IC 平均相关 {m.corr_mean_full:.2f} 超限"
            f"（阈值 {cfg.get('max_corr_mean', 0.5)}），{m.corr_n} 个因子高度同质化",
        })
    if _ok(m.pc1_share_full) and m.pc1_share_full > float(cfg.get("max_pc1_share", 0.4)):
        alerts.append({
            "rule": "factor_crowding",
            "level": "warning",
            "message": f"单一主成分解释 {m.pc1_share_full:.0%} 的因子 IC 波动"
            f"（阈值 {cfg.get('max_pc1_share', 0.4)}），等效押同一方向，分散度虚假",
        })


def _append_signal_alerts(m: MonitorMetrics, cfg: dict[str, Any], alerts: list[dict]) -> None:
    """组合/信号层告警：信号新鲜度、覆盖、集中度、净换手、受阻比例。"""
    if m.signal_freshness_days > int(cfg.get("max_freshness_days", 30)):
        alerts.append({
            "rule": "signal_stale",
            "level": "warning",
            "message": f"最新信号日已滞后 {m.signal_freshness_days} 个交易日"
            f"（阈值 {cfg.get('max_freshness_days', 30)}），信号流水线可能断产",
        })
    if m.n_signal_stocks and m.n_signal_stocks < int(cfg.get("min_signal_stocks", 20)):
        alerts.append({
            "rule": "signal_coverage",
            "level": "warning",
            "message": f"本期可交易股票仅 {m.n_signal_stocks} 只"
            f"（下限 {cfg.get('min_signal_stocks', 20)}）",
        })
    if _ok(m.hhi_recent) and m.hhi_recent > float(cfg.get("max_hhi", 0.3)):
        alerts.append({
            "rule": "signal_concentration",
            "level": "warning",
            "message": f"持仓集中度 HHI={m.hhi_recent:.3f} 过高"
            f"（上限 {cfg.get('max_hhi', 0.3)}），单票/少数票风险积聚",
        })
    if _ok(m.net_turnover) and m.net_turnover > float(cfg.get("max_net_turnover", 0.5)):
        alerts.append({
            "rule": "signal_turnover",
            "level": "warning",
            "message": f"本次净换手 {m.net_turnover:.0%} 过高"
            f"（上限 {cfg.get('max_net_turnover', 0.5)}），换手成本侵蚀",
        })
    if _ok(m.blocked_ratio) and m.blocked_ratio > float(cfg.get("max_blocked_ratio", 0.3)):
        alerts.append({
            "rule": "signal_blocked",
            "level": "warning",
            "message": f"受阻交易占比 {m.blocked_ratio:.0%} 过高"
            f"（上限 {cfg.get('max_blocked_ratio', 0.3)}），涨跌停/停牌导致目标无法建立",
        })


def rollup_status(alerts: list[dict]) -> str:
    """快照状态汇总：任一 critical → critical，任一 warning → warning。"""
    if any(a["level"] == "critical" for a in alerts):
        return "critical"
    if any(a["level"] == "warning" for a in alerts):
        return "warning"
    return "normal"


def attach_alerts(m: MonitorMetrics, cfg: dict[str, Any]) -> MonitorMetrics:
    """便捷入口：触发规则并写回快照（alerts + status）。"""
    m.alerts = evaluate_alerts(m, cfg)
    m.status = rollup_status(m.alerts)
    return m
