"""统一入口 ``scripts.live.yqcore`` 的 CLI 装配测试。

只测**参数解析与分发**，不真跑出榜/持仓（那是子进程、要数据和模型）。
重点是防止两个功能被悄悄合成一条命令 —— 那是设计上明确拒绝的。
"""
from __future__ import annotations

import pytest

from scripts.live.yqcore import build_parser


def test_requires_subcommand():
    """必须显式给子命令，不允许「不带参数就默认做点什么」。"""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_rank_defaults_to_dingban_preset():
    """出榜候选必须是定版 dingban（族内 default），不是 legacy。"""
    a = build_parser().parse_args(["rank"])
    assert a.preset == "dingban"
    assert a.cmd == "rank"
    assert a.func.__name__ == "cmd_rank"


def test_rank_accepts_date_and_strict_status():
    a = build_parser().parse_args(
        ["rank", "--date", "20261109", "--strict-status", "--out-tag", "_solo"])
    assert a.date == "20261109"
    assert a.strict_status is True
    assert a.out_tag == "_solo"


def test_holdings_defaults_to_readonly():
    """🚨 holdings 默认必须只读（commit=False），防止随手一查就生成新快照。"""
    a = build_parser().parse_args(["holdings"])
    assert a.commit is False
    assert a.cmd == "holdings"
    assert a.func.__name__ == "cmd_holdings"


def test_holdings_commit_is_opt_in():
    a = build_parser().parse_args(["holdings", "--commit", "--date", "20261201"])
    assert a.commit is True
    assert a.date == "20261201"


def test_holdings_all_flag():
    assert build_parser().parse_args(["holdings", "--all"]).all is True


def test_rank_rejects_unknown_preset():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["rank", "--preset", "不存在"])


def test_two_subcommands_are_distinct():
    """两个子命令各自独立：holdings 没有 --preset，rank 没有 --all。"""
    r = build_parser().parse_args(["rank"])
    h = build_parser().parse_args(["holdings"])
    assert not hasattr(r, "all")
    assert not hasattr(h, "preset")
    assert r.func is not h.func