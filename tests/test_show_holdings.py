"""持仓查询脚本的回归测试（2026-11-10 新增）。

锁死两件事：

1. **只读**——查询绝不落盘，否则「当期持仓是哪一份」会因随手一查而变含糊；
2. **期次定位语义**——缺省取最新一期、`--date` 精确取、无期次时明确报错
   （不静默返回空表）。
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.live import show_holdings as sh


def _write_period(root, ds: str, codes: list[str]) -> None:
    """写一份最小持仓 + meta。"""
    rows = []
    for i, c in enumerate(codes):
        rows.append({
            "rank": i + 1, "name": "", "industry": "测试",
            "weight_target": 0.2, "price_ref": 10.0, "shares": 1000,
            "amount": 10000.0, "weight_actual": 0.2, "open_exec": 10.0,
            "cost": 10000.0, "last_close": 10.0, "mv": 10000.0,
            "ret_since_exec": 0.0, "pnl": 0.0,
        })
    df = pd.DataFrame(rows, index=pd.Index(codes, name="code"))
    df.to_csv(root / f"holdings_{ds}.csv", encoding="utf-8-sig")
    (root / f"meta_{ds}.json").write_text(
        '{"signal_date": "%s", "exec_date": "2026-11-03", "capital": 100000,'
        ' "k": 8, "max_per_sector": 1, "lot": 100, "portfolio_ret": 0.0021}'
        % ds, encoding="utf-8")


def test_list_periods_sorted_ascending(tmp_path):
    """期次按信号日升序，且只认 8 位日期后缀（防 seed 文件混入）。"""
    _write_period(tmp_path, "20260901", ["000001.SZ"])
    _write_period(tmp_path, "20261102", ["000001.SZ"])
    _write_period(tmp_path, "20261009", ["000001.SZ"])
    (tmp_path / "holdings_seed_backtest.csv").write_text("code\n000001.SZ\n",
                                                        encoding="utf-8")
    got = sh.list_periods(tmp_path)
    assert got == ["20260901", "20261009", "20261102"]


def test_latest_period_picks_max(tmp_path):
    _write_period(tmp_path, "20260901", ["000001.SZ"])
    _write_period(tmp_path, "20261102", ["000001.SZ"])
    assert sh.latest_period(tmp_path) == "20261102"


def test_latest_period_none_when_empty(tmp_path):
    assert sh.latest_period(tmp_path) is None
    assert sh.list_periods(tmp_path) == []


def test_load_holdings_codes_stay_string(tmp_path):
    """代码必须是字符串，不能被读成数字丢前导零（项目 PIT 铁律的同类坑）。"""
    _write_period(tmp_path, "20261102", ["000001.SZ", "600128.SH"])
    h = sh.load_holdings("20261102", tmp_path)
    assert list(h.index) == ["000001.SZ", "600128.SH"]
    assert all(isinstance(c, str) for c in h.index)


def test_missing_period_raises_not_empty(tmp_path):
    """查不存在的期次必须报错，不能返回空表让人误以为空仓。"""
    _write_period(tmp_path, "20261102", ["000001.SZ"])
    with pytest.raises(FileNotFoundError, match="20260901"):
        sh.load_holdings("20260901", tmp_path)


def test_value_position_uses_latest_close(tmp_path):
    """估值按最新收盘：成本 10 元、现价 11 元 ⇒ +10%，而非沿用落盘快照。"""
    _write_period(tmp_path, "20261102", ["000001.SZ"])
    h = sh.load_holdings("20261102", tmp_path)
    last = pd.Series({"000001.SZ": 11.0})
    v = sh.value_position(h, last, pd.Timestamp("2026-11-09"))
    assert v.loc["000001.SZ", "ret_total"] == pytest.approx(0.10)
    assert v.loc["000001.SZ", "mv_now"] == pytest.approx(11000.0)
    assert v.loc["000001.SZ", "pnl_now"] == pytest.approx(1000.0)


def test_show_period_does_not_write_anything(tmp_path, capsys):
    """🚨 只读保证：查询前后目录内容（含 mtime）完全不变。"""
    _write_period(tmp_path, "20261102", ["000001.SZ"])
    before = {p.name: p.stat().st_mtime for p in tmp_path.iterdir()}

    h = sh.load_holdings("20261102", tmp_path)
    last = pd.Series({"000001.SZ": 10.5})
    sh.value_position(h, last, pd.Timestamp("2026-11-09"))
    show = sh.value_position(h, last, pd.Timestamp("2026-11-09"))
    print(show.to_string())          # 模拟输出，不落盘
    capsys.readouterr()

    after = {p.name: p.stat().st_mtime for p in tmp_path.iterdir()}
    assert after == before, "查询不得新增或修改任何文件"


def test_meta_missing_returns_empty_dict(tmp_path):
    """meta 缺失不应阻断查询（只影响展示字段）。"""
    _write_period(tmp_path, "20261102", ["000001.SZ"])
    (tmp_path / "meta_20261102.json").unlink()
    assert sh.load_meta("20261102", tmp_path) == {}
    assert len(sh.load_holdings("20261102", tmp_path)) == 1