"""面板体检的判定逻辑（2026-11-07）。

这个文件守两件事：

1. **判据本身有效**：把一块面板人为做成「尾部冻结」/「全历史常量」/「尾部无覆盖」，
   :func:`audit_panel` 必须分别报出对应 reason，而干净面板一个都不许报。
   判据失效的代价很实在——`year_offset` 那个前视 bug 就是**覆盖率正常、值不动**，
   用覆盖率体检会全绿放行。
2. **静态污染清单与代码同步**：`TAINTED_BY_YEAR_OFFSET` 是人工维护的，它必须仍然
   指着真的调用 ``year_offset()`` 的 builder。清单漂移 = 下次重建漏掉某个因子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.factors.panel_leak_audit import (TAINTED_BY_YEAR_OFFSET, WINDOW,
                                              audit_panel)

N = 600          # 2 年多交易日，够 LOOKBACK+WINDOW
CODES = [f"{i:06d}.SZ" for i in range(60)]
DATELINE = pd.bdate_range("2024-01-01", periods=N)


def _write(tmp_path, values: np.ndarray, name: str):
    idx = DATELINE[:values.shape[0]]
    df = pd.DataFrame(values, index=idx, columns=CODES).astype("float32")
    p = tmp_path / f"{name}.parquet"
    df.to_parquet(p)
    return p


def _live(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N, len(CODES)))


def test_a_live_panel_is_not_flagged(tmp_path):
    r = audit_panel(_write(tmp_path, _live(), "live"))
    assert not r["frozen"]
    assert r["reason"] == ""
    assert r["const_all"] is False


def test_a_panel_frozen_in_the_tail_is_flagged(tmp_path):
    """尾部 250 日值不动、历史正常 —— 前视 bug 的签名。"""
    v = _live()
    v[-WINDOW:] = v[-WINDOW - 1]                 # 后一年整段冻在最后一个正常值
    r = audit_panel(_write(tmp_path, v, "tail_frozen"))
    assert r["frozen"]
    assert r["reason"] == "尾部新冻结"
    assert r["jump"] >= 60


def test_a_constant_panel_is_flagged_as_broadcast(tmp_path):
    """整块面板 = 首行复制 N 遍 —— 广播 bug，与"更新慢"不是一回事。"""
    v = np.tile(_live()[0], (N, 1))
    r = audit_panel(_write(tmp_path, v, "const"))
    assert r["frozen"]
    assert r["reason"] == "全历史常量"
    assert r["const_all"] is True


def test_a_panel_that_stopped_being_covered_is_flagged(tmp_path):
    """尾部几乎全 NaN：不是"值没变"，是根本没跟上。"""
    v = _live()
    v[-WINDOW:] = np.nan
    r = audit_panel(_write(tmp_path, v, "no_cover"))
    assert r["frozen"]
    assert r["reason"] == "尾部无覆盖"


def test_a_slow_factor_is_not_flagged_for_being_slow(tmp_path):
    """年报类因子本来一年动一次：停滞天数长，但 ``jump`` 小，不该报。"""
    v = np.tile(_live()[0], (N, 1))
    steps = np.arange(N) // 250                  # 每 250 日换一版
    v = v * steps[:, None]
    r = audit_panel(_write(tmp_path, v, "annual"))
    assert not r["frozen"]


def test_staleness_shrinks_when_the_panel_moves():
    from scripts.factors.panel_leak_audit import _staleness

    v = _live()
    stale_live, chg_live = _staleness(v, len(v) - 1)
    assert stale_live < 30 and chg_live == 1.0

    v2 = v.copy()
    v2[-WINDOW:] = np.nan
    stale_dead, chg_dead = _staleness(v2, len(v2) - 1)
    assert stale_dead >= 200 and chg_dead == 0.0


def test_tainted_list_points_at_builders_that_still_call_year_offset():
    """静态清单的锚：每个污染因子的名字要能在**某个真的调用 year_offset 的 builder** 里找到。"""
    from pathlib import Path

    builders = Path(__file__).resolve().parents[1] / "scripts" / "builders"
    users = {p.name for p in builders.glob("*.py")
             if "year_offset(" in p.read_text(encoding="utf-8")}
    assert users, "没有任何 builder 调用 year_offset —— 清单已失效，需要重新核对"

    for factor, owner in TAINTED_BY_YEAR_OFFSET.items():
        f = builders / f"{owner}.py"
        assert f.exists(), f"{factor} 的归属文件 {owner}.py 不存在"
        src = f.read_text(encoding="utf-8")
        assert "year_offset(" in src, f"{owner}.py 不再调用 year_offset，清单需复核"
        assert factor in src, f"{owner}.py 里找不到 {factor}，清单需复核"
