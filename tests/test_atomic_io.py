"""`scripts.common.atomic_io` 回归：原子写盘 + 降级路径 + 残渣清扫。

为什么值得锁死
--------------
本机（2026-10-16）实测文件守护拦删除：`unlink` 单次 ~6.7s、并发下 10s 超时，
而原子改名（`os.replace` 会删源）在批处理规模下**被持续拒绝**——退避重试 5 次
（~12s）也救不回来，整批中断。故 `write_parquet_atomic` 必须保留"降级就地覆写"
这条路径；它不直观（牺牲原子性换不中断），一旦被后来者"顺手删掉"就会重现
`WinError 5` 整批中断。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.atomic_io import sweep_stale_tmp, write_parquet_atomic  # noqa: E402


def _df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


def test_atomic_write_ok_and_no_tmp_left(tmp_path: Path) -> None:
    dst = tmp_path / "x.parquet"
    write_parquet_atomic(_df(), dst)
    assert dst.exists()
    assert list(tmp_path.glob("*.tmp")) == []      # 不残留
    pd.testing.assert_frame_equal(pd.read_parquet(dst), _df())


def test_overwrite_is_atomic_replacement(tmp_path: Path) -> None:
    dst = tmp_path / "x.parquet"
    write_parquet_atomic(_df(), dst)
    new = pd.DataFrame({"a": [9.0], "b": [9.0]})
    write_parquet_atomic(new, dst)
    pd.testing.assert_frame_equal(pd.read_parquet(dst), new)
    assert list(tmp_path.glob("*.tmp")) == []


def test_falls_back_to_direct_write_when_replace_always_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """改名被持续拒绝（WinError 5）时应降级就地覆写，而不是把整批打断。"""
    dst = tmp_path / "x.parquet"
    write_parquet_atomic(_df(), dst)               # 先有旧内容
    new = pd.DataFrame({"a": [42.0], "b": [43.0]})

    def _deny(*_a, **_k):
        raise PermissionError(5, "拒绝访问。")

    monkeypatch.setattr(os, "replace", _deny)
    # 退避基数调小，避免测试里真等 ~12s
    monkeypatch.setattr("scripts.common.atomic_io.BACKOFF", 0.001)
    write_parquet_atomic(new, dst)                 # 不抛异常
    pd.testing.assert_frame_equal(pd.read_parquet(dst), new)   # 内容已更新
    assert list(tmp_path.glob("*.tmp")) == []      # 降级路径也清掉自己的 tmp


def test_fallback_write_retries_transient_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """降级就地覆写也要扛瞬时占用。

    2026-11-05 现场：`panels_neu_fundind` 整批改名被拒后降级覆写，第一枪撞上
    文件仍被占用 ⇒ 裸抛 PermissionError ⇒ 重建链在 [18/19] 中止，40 分钟算力白烧。
    改名失败本身就是「有人正打开这个文件」的信号，覆写不重试等于放大失败率。
    """
    dst = tmp_path / "x.parquet"
    write_parquet_atomic(_df(), dst)
    new = pd.DataFrame({"a": [7.0], "b": [8.0]})

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(
        PermissionError(5, "拒绝访问。")))
    monkeypatch.setattr("scripts.common.atomic_io.BACKOFF", 0.001)
    monkeypatch.setattr("scripts.common.atomic_io.FALLBACK_BUDGET", 0.05)
    monkeypatch.setattr("scripts.common.atomic_io.FALLBACK_MAX_SLEEP", 0.01)

    import pandas as pd_mod
    real_to_parquet = pd_mod.DataFrame.to_parquet
    calls = {"n": 0}

    def _flaky(self, path, *a, **k):
        # 第一次就地覆写被拒，第二次成功（模拟瞬时占用释放）
        if ".tmp" not in str(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(13, "Permission denied")
        return real_to_parquet(self, path, *a, **k)

    monkeypatch.setattr(pd_mod.DataFrame, "to_parquet", _flaky)
    write_parquet_atomic(new, dst)
    assert calls["n"] >= 2                       # 确实重试过
    pd.testing.assert_frame_equal(pd.read_parquet(dst), new)


def test_fallback_write_raises_after_retries_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """持续被拒时仍然抛出（不能静默吞掉写入失败）。"""
    dst = tmp_path / "x.parquet"
    write_parquet_atomic(_df(), dst)

    monkeypatch.setattr(os, "replace", lambda *a, **k: (_ for _ in ()).throw(
        PermissionError(5, "拒绝访问。")))
    monkeypatch.setattr("scripts.common.atomic_io.BACKOFF", 0.001)
    monkeypatch.setattr("scripts.common.atomic_io.FALLBACK_BUDGET", 0.05)
    monkeypatch.setattr("scripts.common.atomic_io.FALLBACK_MAX_SLEEP", 0.01)

    import pandas as pd_mod
    real_to_parquet = pd_mod.DataFrame.to_parquet

    def _always_denied(self, path, *a, **k):
        if ".tmp" not in str(path):
            raise PermissionError(13, "Permission denied")
        return real_to_parquet(self, path, *a, **k)

    monkeypatch.setattr(pd_mod.DataFrame, "to_parquet", _always_denied)
    with pytest.raises(PermissionError, match="就地覆写"):
        write_parquet_atomic(pd.DataFrame({"a": [1.0]}), dst)


def test_sweep_stale_tmp_respects_min_age(tmp_path: Path) -> None:
    old = tmp_path / "old.parquet.tmp"
    fresh = tmp_path / "fresh.parquet.tmp"
    for p in (old, fresh):
        p.write_bytes(b"x")
    aged = time.time() - 3600
    os.utime(old, (aged, aged))

    assert sweep_stale_tmp(tmp_path, min_age_seconds=600) == 1   # 只清旧的那个
    assert not old.exists()
    assert fresh.exists()                                        # 保护"正在写"的
    assert sweep_stale_tmp(tmp_path, min_age_seconds=0) == 1
    assert not fresh.exists()
