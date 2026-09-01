"""生产化监控层单测：指标 / 告警规则 / 账本幂等 / 调度时间 / 端到端。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoring.alerts import attach_alerts, evaluate_alerts, rollup_status
from monitoring.ledger import MonitoringLedger
from monitoring.metrics import (
    MonitorMetrics,
    compute_factor_metrics,
    quantile_monotonicity,
)
from monitoring.runner import generate_html_report, next_run_time, run_monitoring
from monitoring.signal_monitor import (
    build_signal_metrics,
    compute_signal_monitor,
)

CFG = {
    "window": 60,
    "window_long": 252,
    "max_stale_days": 7,
    "confirm_n": 1,  # 单测免去抖：单次告警即可确认
    "min_coverage": 0.5,
    "warn_ic_retention": 0.5,
    "min_monotonicity": 0.8,
    "min_t_nw_recent": 1.0,
    "ledger_root": "reports/monitoring",
}


# ---------------------------------------------------------------------------
# fixtures：合成面板
# ---------------------------------------------------------------------------
def _panel(days: int = 200, codes: int = 30, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    cols = [f"c{i:03d}" for i in range(codes)]
    close = pd.DataFrame(
        100 + rng.normal(0, 1, (days, codes)).cumsum(axis=0), index=dates, columns=cols
    )
    rets = close.pct_change(fill_method=None).shift(-1)
    return close, rets


def _ic_series(
    days: int, head_ic: float, tail_ic: float, tail: int = 60, seed: int = 1
) -> pd.Series:
    rng = np.random.default_rng(seed)
    tail = min(tail, days)
    vals = np.concatenate(
        [
            head_ic + rng.normal(0, 0.15, days - tail),
            tail_ic + rng.normal(0, 0.15, tail),
        ]
    )
    return pd.Series(vals, index=pd.bdate_range("2025-01-01", periods=days), name="ic")


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def test_quantile_monotonicity_directional():
    close, rets = _panel(seed=3)
    factor = rets.rank(pct=True, axis=1) - 0.5  # 完美预测（前瞻因子）
    mono, ls = quantile_monotonicity(factor, rets)
    assert mono > 0.9
    assert ls > 0
    mono_inv, ls_inv = quantile_monotonicity(-factor, rets)
    assert mono_inv < -0.9
    assert ls_inv < 0


def test_compute_factor_metrics_flip_and_model_baseline():
    close, rets = _panel(days=200)
    panel = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    ic = _ic_series(200, head_ic=0.06, tail_ic=-0.03)
    as_of = ic.index[-1]

    row = pd.Series({"kind": "raw", "ic_mean": 0.05, "note": "model_id=123"})
    m = compute_factor_metrics("gp_factor", row, panel, ic, rets, as_of, window=60)
    assert m.category == "factor"
    assert m.ic_mean_full > 0.01
    assert m.ic_mean_recent < 0
    assert m.ic_retention < 0  # 翻转 → 负保留率
    assert m.coverage_recent > 0.95  # 按当期可交易股票算（非 PIT 并集池）
    assert m.stale_days == 0

    mm = compute_factor_metrics("model:gbdt_h1", row, panel, ic, rets, as_of, window=60)
    assert mm.category == "model"
    assert mm.model_id == "123"
    assert mm.expected_ic == pytest.approx(0.05)


def test_frozen_baseline_uses_registered_ic():
    """① 冻结基线：显著注册基线优先于全期；近 0 基线回退全期。"""
    close, rets = _panel(days=200)
    panel = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    ic = _ic_series(200, head_ic=0.06, tail_ic=0.04)

    row = pd.Series({"kind": "raw", "ic_mean": 0.05, "note": ""})
    m = compute_factor_metrics("f1", row, panel, ic, rets, ic.index[-1], window=60)
    assert m.frozen_baseline == pytest.approx(0.05)          # 显著 → 冻结
    assert m.ic_retention == pytest.approx(m.ic_mean_recent / 0.05)

    row0 = pd.Series({"kind": "raw", "ic_mean": 0.002, "note": ""})
    m0 = compute_factor_metrics("f2", row0, panel, ic, rets, ic.index[-1], window=60)
    assert m0.expected_ic == pytest.approx(0.002)
    assert m0.frozen_baseline != m0.frozen_baseline          # 近 0 → 不冻结 -> nan
    assert m0.ic_retention == pytest.approx(m0.ic_mean_recent / m0.ic_mean_full)


def test_dual_window_ic_metrics():
    """双窗口（60/252）：60d 对近期反转灵敏，252d 因覆盖更长而更稳健。"""
    close, rets = _panel(days=200)
    panel = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    ic = _ic_series(200, head_ic=0.06, tail_ic=-0.03, tail=60)  # 后60d反转
    row = pd.Series({"kind": "raw", "ic_mean": 0.05, "note": ""})
    m = compute_factor_metrics("f1", row, panel, ic, rets, ic.index[-1],
                               window=60, window_long=252)

    assert m.recent_n_days == 60
    assert m.recent_n_days_252 >= 170            # 数据 170d < 长窗 → 回落全量(=169)
    assert m.ic_mean_recent < 0                # 短窗捕捉到反转
    assert m.ic_mean_recent_252 > 0            # 长窗被历史正段拉平
    assert m.ic_mean_recent < m.ic_mean_recent_252
    assert m.ic_retention_252 == pytest.approx(
        m.ic_mean_recent_252 / m.frozen_baseline   # 冻结基线（注册 ic_mean=0.05）
    )


# ---------------------------------------------------------------------------
# 告警规则
# ---------------------------------------------------------------------------
def _snap(**kw) -> MonitorMetrics:
    base = dict(
        name="x",
        ic_mean_full=0.05,
        ic_mean_recent=0.03,
        recent_n_days=60,
        coverage_recent=0.9,
        monotonicity_recent=0.9,
        ic_t_nw_recent=2.0,
        stale_days=0,
        expected_ic=float("nan"),
    )
    base.update(kw)
    return MonitorMetrics(**base)


def test_alert_rules_each():
    rules = {a["rule"]: a for a in evaluate_alerts(_snap(stale_days=10), CFG)}
    assert "stale_data" in rules

    rules = {a["rule"]: a for a in evaluate_alerts(_snap(coverage_recent=0.3), CFG)}
    assert "coverage_drop" in rules

    # 因子方向翻转 → critical
    rules = {a["rule"]: a for a in evaluate_alerts(_snap(ic_mean_recent=-0.02), CFG)}
    assert rules["ic_decay"]["level"] == "critical"

    # 保留率不足（0.01/0.05 = 20% < 50%）→ warning
    rules = {a["rule"]: a for a in evaluate_alerts(_snap(ic_mean_recent=0.01), CFG)}
    assert rules["ic_decay"]["level"] == "warning"

    # 模型基线：expected=0.04, recent=0.01 → warning；t_nw 低 → significance_loss
    m = _snap(
        category="model",
        expected_ic=0.04,
        ic_mean_recent=0.01,
        ic_t_nw_recent=0.5,
        monotonicity_recent=0.3,
    )
    rules = {a["rule"]: a for a in evaluate_alerts(m, CFG)}
    assert rules["ic_decay"]["level"] == "warning"
    assert "significance_loss" in rules
    assert "monotonicity_break" in rules

    # 因子单调性用 |signed|：-0.9（反向一致）不触发
    rules = [a["rule"] for a in evaluate_alerts(_snap(monotonicity_recent=-0.9), CFG)]
    assert "monotonicity_break" not in rules


def test_ic_decay_dual_window_confirmation():
    """双窗口确认：短窗(60d)已衰减而长窗(252d)仍稳健 → 观察中；长短同步衰减 → 予以确认。"""
    # 观察中：recent=0.01(20%) < 50%，recent_252=0.04(80%) >= 50%
    obs = {a["rule"]: a for a in evaluate_alerts(
        _snap(ic_mean_recent=0.01, ic_mean_recent_252=0.04, recent_n_days_252=252), CFG)}
    assert obs["ic_decay"]["level"] == "warning"
    assert "观察中，尚未长窗确认" in obs["ic_decay"]["message"]

    # 予以确认：recent_252=0.01(20%) < 50%，长短同步衰减
    conf = {a["rule"]: a for a in evaluate_alerts(
        _snap(ic_mean_recent=0.01, ic_mean_recent_252=0.01, recent_n_days_252=252), CFG)}
    assert "予以确认" in conf["ic_decay"]["message"]


def test_rollup_and_attach():
    m = attach_alerts(_snap(stale_days=10, ic_mean_recent=-0.02), CFG)
    assert m.status == "critical"  # critical > warning
    assert len(m.alerts) >= 2
    assert rollup_status([]) == "normal"
    assert rollup_status([{"level": "warning"}]) == "warning"


# ---------------------------------------------------------------------------
# 账本幂等
# ---------------------------------------------------------------------------
def test_ledger_idempotent(tmp_path: Path):
    led = MonitoringLedger(tmp_path)
    df = pd.DataFrame(
        [{"run_date": "2026-08-20 17:30", "as_of": "2026-08-19", "name": "f1", "status": "normal"}]
    )
    led.append_snapshots(df, "2026-08-20 17:30")
    led.append_snapshots(df, "2026-08-20 17:30")  # 同 run_date 重跑
    snap = led.load_snapshots()
    assert len(snap) == 1

    df2 = df.copy()
    df2["name"] = "f2"
    led.append_snapshots(df2, "2026-08-20 18:00")  # 新 run_date 追加
    assert len(led.load_snapshots()) == 2
    assert led.history("f1", "status").iloc[0]["status"] == "normal"

    alerts = pd.DataFrame(
        [
            {
                "run_date": "2026-08-20 17:30",
                "name": "f1",
                "rule": "stale_data",
                "level": "warning",
                "message": "x",
            }
        ]
    )
    led.append_alerts(alerts, "2026-08-20 17:30")
    led.append_alerts(alerts, "2026-08-20 17:30")
    assert len(led.load_alerts()) == 1


# ---------------------------------------------------------------------------
# 调度时间
# ---------------------------------------------------------------------------
def test_next_run_time():
    now = datetime(2026, 8, 20, 9, 0)
    assert next_run_time(now, "17:30") == datetime(2026, 8, 20, 17, 30)
    assert next_run_time(now, "08:00") == datetime(2026, 8, 21, 8, 0)
    assert next_run_time(datetime(2026, 8, 20, 17, 30, 0), "17:30") == datetime(
        2026, 8, 21, 17, 30
    )  # 恰好等于 → 明日
    assert next_run_time(datetime(2026, 8, 20, 23, 50), "00:10") == datetime(
        2026, 8, 21, 0, 10
    )  # 跨零点


# ---------------------------------------------------------------------------
# Windows 计划任务（每日自动化）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(sys.platform != "win32", reason="schtasks 计划任务注册仅适用于 Windows")
def test_task_scheduler_command():
    from scripts.monitor_performance import (
        SCHEDULED_TASK,
        SYSTEM_PY,
        task_scheduler_cmd,
    )

    cmd = task_scheduler_cmd("hs300_2022_2025", "C:/tools/py.exe")
    assert SCHEDULED_TASK in cmd
    assert "/SC DAILY /ST 17:30" in cmd
    assert "C:\\tools\\py.exe" in cmd
    assert "--dataset hs300_2022_2025" in cmd

    # 默认解析：系统 Python 3.12 存在时固定指向它（真实数据链路依赖其 SDK/凭证）。
    # Windows 路径大小写不敏感：resolve() 返回磁盘真实大小写，可能与 sys.executable
    # 字面大小写不同，故按小写比较。
    default = task_scheduler_cmd("ds")
    if SYSTEM_PY.exists():
        assert SYSTEM_PY.resolve().as_posix().lower() in default.replace("\\", "/").lower()


# ---------------------------------------------------------------------------
# 端到端：临时因子库 + 临时日线缓存
# ---------------------------------------------------------------------------
@pytest.fixture
def env(tmp_path: Path):
    from model.serving import register_model_as_factor
    from research.factor_library import FactorLibrary

    days, codes = 170, 25
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=days)
    cols = [f"c{i:03d}" for i in range(codes)]
    close = pd.DataFrame(
        100 + rng.normal(0, 1, (days, codes)).cumsum(axis=0), index=dates, columns=cols
    )
    rets = close.pct_change(fill_method=None).shift(-1)

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    long_close = close.stack().rename("close")
    long_close.index = long_close.index.set_names(["date", "code"])
    long_close.to_frame().to_parquet(cache_root / "daily_hs300.parquet")

    lib_root = tmp_path / "flib"
    lib = FactorLibrary(root=lib_root, dataset="testds")
    good = rets.rank(pct=True, axis=1) - 0.5  # 完美预测因子（测试用）
    sign = pd.Series(1.0, index=dates)
    sign.iloc[110:] = -1.0  # 后 60 日反转 → 衰减因子
    decay = good.mul(sign, axis=0)
    lib.register("good_factor", good, rets, kind="raw", formula="good")
    lib.register("decay_factor", decay, rets, kind="raw", formula="decay")
    register_model_as_factor(
        name="model:gbdt_h1",
        pred_panel=good,
        returns_panel=rets,
        parents=["good_factor"],
        dataset="testds",
        model_id="42",
        horizon=1,
        oos=True,
        note="test",
        root=lib_root,
    )

    return {
        "lib_root": lib_root,
        "cache_root": cache_root,
        "ledger_root": tmp_path / "ledger",
        "close": close,
        "rets": rets,
    }


def test_run_monitoring_e2e(env):
    summary = run_monitoring(
        dataset="testds",
        window=60,
        factor_root=env["lib_root"],
        cache_root=env["cache_root"],
        ledger_root=env["ledger_root"],
        cfg=CFG,
        record=False,
    )

    assert summary["n_factors"] == 3
    assert summary["n_models"] == 1
    snap = pd.read_csv(env["ledger_root"] / "snapshots.csv")
    assert set(snap["name"]) == {"good_factor", "decay_factor", "model:gbdt_h1"}
    by_name = snap.set_index("name")
    assert float(by_name.loc["model:gbdt_h1", "model_id"]) == 42.0
    assert by_name.loc["decay_factor", "ic_mean_recent"] < 0
    # 双窗口：decay_factor 后段反转，60d 捕捉为负、252d(接近全量)被历史正段拉平
    assert by_name.loc["decay_factor", "ic_mean_recent"] < by_name.loc[
        "decay_factor", "ic_mean_recent_252"
    ]
    assert by_name.loc["decay_factor", "recent_n_days_252"] >= 100

    alerts = pd.read_csv(env["ledger_root"] / "alerts.csv")
    assert len(alerts) > 0
    assert "ic_decay" in set(alerts["rule"])

    report = env["ledger_root"] / "monitor_report.html"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "model:gbdt_h1" in text and "svg" in text


def test_html_report_renders(tmp_path: Path):
    m = attach_alerts(
        MonitorMetrics(
            name="model:ridge_h1",
            category="model",
            ic_mean_recent=0.02,
            ic_mean_recent_252=0.03,
            recent_n_days_252=200,
            expected_ic=0.04,
            ic_retention=0.5,
            monotonicity_recent=0.5,
            ic_t_nw_recent=0.9,
        ),
        CFG,
    )
    out = tmp_path / "r.html"
    alert_rows = [{"name": m.name, "category": m.category, **a} for a in m.alerts]
    generate_html_report(
        [m],
        alert_rows,
        {"model:ridge_h1": _ic_series(50, 0.05, 0.02)},
        out,
        dataset="d",
        window=60,
        window_long=252,
        as_of=pd.Timestamp("2026-08-19"),
    )
    text = out.read_text(encoding="utf-8")
    assert "monotonicity_break" in text
    assert "significance_loss" in text
    assert "双窗口=60/252日" in text
    assert "近252日IC" in text
    assert ">严重<" in text or ">预警<" in text  # 状态/级别中文展示


# ---------------------------------------------------------------------------
# 组合/信号层监控（③）
# ---------------------------------------------------------------------------
def _signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2025-10-31"] * 3 + ["2025-11-28"] * 3,
            "code": ["A", "B", "C"] * 2,
            "target_weight": [0.5, 0.3, 0.2, 0.6, 0.4, 0.0],
            "status": ["OK", "OK", "OK", "OK", "OK", "BLOCKED_SELL"],
        }
    )


def test_signal_monitor_metrics():
    cal = pd.bdate_range("2025-10-01", "2025-12-15")
    m = compute_signal_monitor(_signal_frame(), cal, pd.Timestamp("2025-11-28"))
    assert m.category == "signal"
    assert m.signal_date == "2025-11-28"
    assert m.n_signal_stocks == 2
    assert m.hhi_recent == pytest.approx(0.6**2 + 0.4**2)   # 0.52
    assert m.net_turnover == pytest.approx(0.5 * (0.1 + 0.1 + 0.2))  # 0.20
    assert m.blocked_ratio == pytest.approx(1 / 3)
    assert m.signal_freshness_days == 0  # 基准日=信号日


def test_signal_monitor_alerts():
    cal = pd.bdate_range("2025-10-01", "2025-12-15")
    m = attach_alerts(compute_signal_monitor(_signal_frame(), cal, pd.Timestamp("2025-11-28")), CFG)
    rules = {a["rule"] for a in m.alerts}
    assert "signal_concentration" in rules   # HHI .52 > 0.3
    assert "signal_blocked" in rules         # 1/3 > 0.3
    assert m.status == "warning"
    # 信号滞后 → signal_stale
    m2 = build_signal_metrics(_signal_frame(), cal, pd.Timestamp(cal[-1]),
                              {**CFG, "max_freshness_days": 2})
    assert m2.signal_freshness_days > 2
    assert any(a["rule"] == "signal_stale" for a in m2.alerts)


def test_run_monitoring_e2e_signal(env, tmp_path: Path):
    sig_path = tmp_path / "sig.csv"
    _signal_frame().to_csv(sig_path, index=False)
    run_monitoring(
        dataset="testds",
        window=60,
        factor_root=env["lib_root"],
        cache_root=env["cache_root"],
        ledger_root=tmp_path / "ledger2",
        cfg=CFG,
        record=False,
        signal_path=sig_path,
    )
    snap = pd.read_csv(tmp_path / "ledger2" / "snapshots.csv")
    assert any(str(n).startswith("signal") for n in snap["name"])
    text = (tmp_path / "ledger2" / "monitor_report.html").read_text(encoding="utf-8")
    assert "组合 & 信号层" in text


# ---------------------------------------------------------------------------
# ④ 告警去抖 / 持续期确认
# ---------------------------------------------------------------------------
def test_state_confirm_rows_debounces(tmp_path: Path):
    from monitoring.state import confirm_rows

    def row(name, rule, run_date):
        return {"run_date": run_date, "name": name, "rule": rule, "level": "warning", "message": ""}

    # 历史上 f1 仅触发过 1 个不连续日期 → 加本次共 2 个日期的连续 < confirm_n=3 → 观察中
    prev = pd.DataFrame(
        [
            row("f1", "mem", "2026-08-18 17:30"),
            row("f2", "mem", "2026-08-18 17:30"),
            row("f2", "mem", "2026-08-19 17:30"),
        ]
    )
    current = [row("f1", "mem", "2026-08-20 17:30")]
    final, pending = confirm_rows(current, prev, confirm_n=3)
    assert final == [] and pending == 1          # 连续仅 f1@8-18+8-20，不足 3 期 → 观察中

    # 三个不连续日期都触发 → 连续 3 期 → 确认
    prev3 = pd.DataFrame(
        [
            row("f1", "mem", "2026-08-18 17:30"),
            row("f1", "mem", "2026-08-19 17:30"),
        ]
    )
    cur_ok = [row("f1", "mem", "2026-08-20 17:30")]
    final3, pending3 = confirm_rows(cur_ok, prev3, confirm_n=3)
    assert len(final3) == 1 and pending3 == 0

    # confirm_n<=1 不过滤
    final1, _ = confirm_rows(current, prev, confirm_n=1)
    assert len(final1) == 1


def test_state_confirm_e2e(tmp_path: Path, env):
    """连续 3 次触发才确认（confirmed=True）；首两次只落台账为观察中。"""
    from monitoring.ledger import MonitoringLedger
    from monitoring.runner import run_monitoring

    decay_cfg = {**CFG, "confirm_n": 3}
    for d in range(3):
        run_monitoring(
            dataset="testds",
            window=60,
            factor_root=env["lib_root"],
            cache_root=env["cache_root"],
            ledger_root=tmp_path / "state_ledger",
            cfg=decay_cfg,
            record=False,
            run_stamp=f"2026-08-{18 + d} 17:30:00",
        )
    alerts = MonitoringLedger(tmp_path / "state_ledger").load_alerts()
    assert not alerts.empty
    assert {"name", "rule", "confirmed"}.issubset(alerts.columns)
    # 第 3 次应形成连续确认
    assert bool(alerts["confirmed"].any())


# ---------------------------------------------------------------------------
# ⑤ 因子拥挤度 / 相关性
# ---------------------------------------------------------------------------
def test_crowding_metrics():
    from monitoring.crowding import compute_crowding

    idx = pd.bdate_range("2024-01-01", periods=120)
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, len(idx)) + np.arange(len(idx)) * 0.01, index=idx)
    # 三个高度同质因子 + 一个独立因子 → 平均相关 >0、主成分解释度高
    ic = {
        "f1": base,
        "f2": base * 1.05 + rng.normal(0, 0.02, len(idx)),
        "f3": base * 0.8 + rng.normal(0, 0.02, len(idx)),
        "indep": pd.Series(rng.normal(0, 1, len(idx)), index=idx),
    }
    m = compute_crowding(ic)
    assert m is not None and m.category == "crowd"
    assert m.corr_n == 4
    assert m.corr_mean_full > 0.3          # 同质主导 → 平均相关显著为正
    assert m.pc1_share_full > 0.4          # 首主成分解释度超阈值（等价于一票)
    m2 = compute_crowding({"a": ic["f1"]})  # 因子不足 → None
    assert m2 is None
