"""``scripts.common.power`` 的防待机上下文单测（不触真实电源状态）。"""

from __future__ import annotations

import pytest


def test_set_execution_state_noop_on_non_windows(monkeypatch):
    from scripts.common import power

    monkeypatch.setattr(power.sys, "platform", "darwin")
    assert power._set_execution_state(0x1) is False


def test_keep_system_awake_sets_and_restores(monkeypatch):
    from scripts.common import power

    calls: list[int] = []
    monkeypatch.setattr(power, "_set_execution_state",
                        lambda flags: (calls.append(flags), True)[1])

    with power.keep_system_awake() as active:
        assert active is True
        assert calls == [power._ES_CONTINUOUS | power._ES_SYSTEM_REQUIRED]

    # 退出时必须清掉 ES_SYSTEM_REQUIRED（只回写 ES_CONTINUOUS）
    assert calls[-1] == power._ES_CONTINUOUS
    assert len(calls) == 2


def test_keep_system_awake_noop_when_api_inactive(monkeypatch):
    from scripts.common import power

    calls: list[int] = []
    monkeypatch.setattr(power, "_set_execution_state",
                        lambda flags: (calls.append(flags), False)[1])

    with power.keep_system_awake() as active:
        assert active is False

    # 未生效时退出不再二次调用（避免无谓的 API 调用）
    assert calls == [power._ES_CONTINUOUS | power._ES_SYSTEM_REQUIRED]


def test_keep_system_awake_restores_on_exception(monkeypatch):
    from scripts.common import power

    calls: list[int] = []
    monkeypatch.setattr(power, "_set_execution_state",
                        lambda flags: (calls.append(flags), True)[1])

    with pytest.raises(RuntimeError):
        with power.keep_system_awake():
            raise RuntimeError("boom")

    assert calls[-1] == power._ES_CONTINUOUS
