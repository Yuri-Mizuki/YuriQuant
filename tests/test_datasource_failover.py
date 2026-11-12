"""AmazingData 双 VIP failover 的回归测试（2026-11-10 新增）。

**背景**：AmazingData 对外是**双 VIP**（主备/双活），SDK 自己的
``mdga.json`` 里就写着 ``"Vip": "101.230.159.234;140.206.44.234"``。
但 ``ad.login(host=...)`` 只接受**单个**地址，SDK 不会自动换 VIP。

2026-11-10 实测：两个 VIP 的 TCP 都通、``ad.login()`` 都成功、
``get_calendar()`` 都返回空 ⇒ 换 IP **解决不了**那次故障（是网关查询层
整体卡死）。但 IP 层面单点确实存在：任一个 VIP 挂掉，旧代码就直接失败。

本文件锁死两件事：
1. ``_candidate_hosts`` 的解析规则（去重、顺序、逗号/分号、缺省零回归）；
2. ``_login`` 在第一个 host 失败时会**继续尝试后续 host**，不是直接抛。
"""
from __future__ import annotations

import sys
import types

import pytest

from data.datasource import (
    AmazingDataSource,
    _call_with_timeout,
    _candidate_hosts,
)


# ---------------------------------------------------------------------------
# _call_with_timeout —— failover 能生效的前提
# ---------------------------------------------------------------------------
def test_call_with_timeout_returns_value():
    assert _call_with_timeout(lambda: 42, 5, "ok") == 42


def test_call_with_timeout_reraises_sdk_error():
    """SDK 抛的异常要原样冒出来（交给 failover 判断）。"""
    def boom():
        raise ConnectionError("sdk boom")

    with pytest.raises(ConnectionError, match="sdk boom"):
        _call_with_timeout(boom, 5, "boom")


def test_call_with_timeout_raises_on_hang():
    """hang 住（SDK 网关卡死的真实表现）必须变成 TimeoutError。

    没有这一层，`get_calendar()` 会让进程永远停在第一个 VIP 上，
    failover 根本执行不到。
    """
    import time

    def hang():
        time.sleep(30)

    t0 = time.time()
    with pytest.raises(TimeoutError, match="get_calendar"):
        _call_with_timeout(hang, 0.3, "get_calendar")
    assert time.time() - t0 < 5, "超时必须真的生效，而不是等满 30s"


# ---------------------------------------------------------------------------
# _candidate_hosts
# ---------------------------------------------------------------------------
def test_candidate_hosts_primary_only(monkeypatch):
    """无 ALT 时行为与旧版完全一致（零回归）。"""
    monkeypatch.delenv("AMAZINGDATA_HOST_ALT", raising=False)
    assert _candidate_hosts("101.230.159.234") == ["101.230.159.234"]


def test_candidate_hosts_adds_alt(monkeypatch):
    monkeypatch.setenv("AMAZINGDATA_HOST_ALT", "140.206.44.234")
    assert _candidate_hosts("101.230.159.234") == [
        "101.230.159.234", "140.206.44.234",
    ]


def test_candidate_hosts_dedupes_and_keeps_order(monkeypatch):
    """ALT 里重复主用地址时不得产生第二次尝试。"""
    monkeypatch.setenv("AMAZINGDATA_HOST_ALT", "140.206.44.234,101.230.159.234")
    assert _candidate_hosts("101.230.159.234") == [
        "101.230.159.234", "140.206.44.234",
    ]


def test_candidate_hosts_accepts_semicolon_and_spaces(monkeypatch):
    """SDK 的 mdga.json 用分号分隔 VIP，用户很可能直接照抄。"""
    monkeypatch.setenv("AMAZINGDATA_HOST_ALT", " 1.2.3.4 ; 5.6.7.8 ")
    assert _candidate_hosts("101.230.159.234") == [
        "101.230.159.234", "1.2.3.4", "5.6.7.8",
    ]


def test_candidate_hosts_ignores_empty_alt(monkeypatch):
    monkeypatch.setenv("AMAZINGDATA_HOST_ALT", "  ,  ")
    assert _candidate_hosts("101.230.159.234") == ["101.230.159.234"]


# ---------------------------------------------------------------------------
# _login failover
# ---------------------------------------------------------------------------
def _fake_ad(behaviour: dict[str, str]) -> types.SimpleNamespace:
    """构造一个假 SDK 模块。

    behaviour: {host: "ok" | "raise" | "empty_calendar"}
    """
    calls: list[str] = []

    def login(**kwargs):
        host = kwargs["host"]
        calls.append(host)
        if behaviour.get(host) == "raise":
            raise ConnectionError(f"boom {host}")

    class _BaseData:
        def __init__(self):
            pass

        def get_calendar(self):
            host = calls[-1]
            return None if behaviour.get(host) == "empty_calendar" else [20261110]

    return types.SimpleNamespace(
        login=login, BaseData=_BaseData, MarketData=lambda cal: object(),
        InfoData=lambda: object(), calls=calls,
    )


def _make_ds(cfg: dict, fake_ad) -> AmazingDataSource:
    """绕过 __init__ 直接测 _login。"""
    ds = AmazingDataSource.__new__(AmazingDataSource)
    ds._local_path = "."
    ds._ad = None
    ds._base = None
    ds._market = None
    ds._info = None
    ds._host_in_use = None

    real_import = __import__

    def fake_import(name, *a, **kw):
        if name == "AmazingData":
            return fake_ad
        return real_import(name, *a, **kw)

    return ds


def test_login_falls_back_to_alt_on_failure(monkeypatch):
    """主用抛异常 → 自动切备用成功，且记录实际使用的 host。"""
    monkeypatch.delenv("AMAZINGDATA_HOST_ALT", raising=False)
    monkeypatch.setattr("data.datasource._probe_login_failure",
                        lambda cfg: "")
    fake = _fake_ad({"101.230.159.234": "raise", "140.206.44.234": "ok"})
    monkeypatch.setitem(sys.modules, "AmazingData", fake)

    ds = _make_ds({}, fake)
    ds._login({
        "username": "u", "password": "p",
        "host": "101.230.159.234,140.206.44.234", "port": 8600,
    })
    assert fake.calls == ["101.230.159.234", "140.206.44.234"]
    assert ds._host_in_use == "140.206.44.234"


def test_login_retries_on_empty_calendar(monkeypatch):
    """登录成功但 get_calendar() 返回空 → 也算失败，换下一个 VIP。

    这正是 2026-11-10 的现场表现：token 拿到了，日历却是 None，
    随后在 SDK 内部炸成看不出根因的 TypeError。
    """
    monkeypatch.setattr("data.datasource._probe_login_failure",
                        lambda cfg: "")
    fake = _fake_ad({"101.230.159.234": "empty_calendar",
                     "140.206.44.234": "ok"})
    monkeypatch.setitem(sys.modules, "AmazingData", fake)

    ds = _make_ds({}, fake)
    ds._login({
        "username": "u", "password": "p",
        "host": "101.230.159.234,140.206.44.234", "port": 8600,
    })
    assert fake.calls == ["101.230.159.234", "140.206.44.234"]
    assert ds._host_in_use == "140.206.44.234"


def test_login_raises_when_all_hosts_fail(monkeypatch):
    """全部 VIP 都失败 → 抛 ConnectionError，且错误信息列出所有地址。"""
    monkeypatch.setattr("data.datasource._probe_login_failure",
                        lambda cfg: "")
    fake = _fake_ad({"101.230.159.234": "raise",
                     "140.206.44.234": "raise"})
    monkeypatch.setitem(sys.modules, "AmazingData", fake)

    ds = _make_ds({}, fake)
    with pytest.raises(ConnectionError) as ei:
        ds._login({
            "username": "u", "password": "p",
            "host": "101.230.159.234,140.206.44.234", "port": 8600,
        })
    msg = str(ei.value)
    assert "101.230.159.234" in msg and "140.206.44.234" in msg
    assert fake.calls == ["101.230.159.234", "140.206.44.234"]


def test_login_missing_credentials_still_fails_fast(monkeypatch):
    """缺凭证仍应立刻 ValueError，不要傻等着去连 VIP。"""
    fake = _fake_ad({})
    monkeypatch.setitem(sys.modules, "AmazingData", fake)
    ds = _make_ds({}, fake)
    with pytest.raises(ValueError, match="credentials are missing"):
        ds._login({"username": "u", "host": "1.2.3.4", "port": 8601})
    assert fake.calls == []


def test_login_falls_back_when_sdk_calls_exit(monkeypatch):
    """SDK 认证失败的真实形态是 ``exit(0)``（SystemExit），不是 raise。

    旧代码用 ``except SystemExit`` 包住 login —— 换 VIP 重试时必须同样接住，
    否则「第一个 VIP 把进程 exit 掉」就等于彻底失败。本测试锁死这一点。
    """
    monkeypatch.setattr("data.datasource._probe_login_failure",
                        lambda cfg: "")
    calls: list[str] = []

    def login(**kwargs):
        calls.append(kwargs["host"])
        if kwargs["host"] == "101.230.159.234":
            raise SystemExit(0)          # ← SDK 真实失败方式

    class _Base:
        def get_calendar(self):
            return [20261110]

    fake = types.SimpleNamespace(
        login=login, BaseData=_Base, MarketData=lambda c: object(),
        InfoData=lambda: object(),
    )
    monkeypatch.setitem(sys.modules, "AmazingData", fake)

    ds = _make_ds({}, fake)
    ds._login({
        "username": "u", "password": "p",
        "host": "101.230.159.234,140.206.44.234", "port": 8600,
    })
    assert calls == ["101.230.159.234", "140.206.44.234"]
    assert ds._host_in_use == "140.206.44.234"