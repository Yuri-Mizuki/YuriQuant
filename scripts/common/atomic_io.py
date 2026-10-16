"""原子写面板 + 抗瞬时文件占用（Windows 原生锁）。

为什么需要
----------
`Path.replace()` 在 Windows 上就是 ``MoveFileExW(MOVEFILE_REPLACE_EXISTING)`` ——
只要目标文件被**任何进程以不带 ``FILE_SHARE_DELETE`` 的方式打开**（杀软实时扫描刚写好的
几十 MB parquet、搜索索引器、或并发的第二个写入方），就会抛
``PermissionError: [WinError 5] 拒绝访问``。

2026-10-16 实测：`panels_neu` 中性化批处理跑到 736/1076 时整批中断，`panels_neu/` 留下
23 个 ``*.parquet.tmp``（内容其实是完整面板、只是没改名），而目标仍是 09-08 的旧版。
即 **一次瞬时锁 = 40 分钟算力白烧**。

对策三件套
----------
① 临时文件名带 pid —— 两个写入方不会互踩同名 tmp（并发的第二个写入方是锁的常见来源）；
② ``os.replace`` 失败按指数退避重试，扛过杀软/索引器的短暂占用窗口；
③ 重试仍失败则清掉自己的 tmp，不留垃圾文件。

（不做跨进程锁 —— 重建链由 `rebuild_panel_library` 单实例驱动；此处只负责单文件写入的
健壮性，不承担互斥职责。）
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

#: 重试次数与退避基数（秒）：0.4 / 0.8 / 1.6 / 3.2 ... 累计 ~6s，远超杀软扫描窗口。
RETRIES = 5
BACKOFF = 0.4


def write_parquet_atomic(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    """把 ``df`` 写到 ``path``：先写同目录 pid 专属 tmp，再原子改名。

    ``kwargs`` 透传给 ``DataFrame.to_parquet``。改名重试耗尽时抛最后一次
    ``PermissionError``（并清掉自己的 tmp）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, **kwargs)
    last_exc: PermissionError | None = None
    for i in range(RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # 瞬时占用 → 退避重试
            last_exc = exc
            time.sleep(BACKOFF * (2 ** i))
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    assert last_exc is not None
    raise last_exc


def sweep_stale_tmp(directory: Path | str, *, min_age_seconds: float = 600.0) -> int:
    """清掉 ``directory`` 下**无主**的 ``*.tmp``（上次被中断留下的残渣），返回清理数。

    ``min_age_seconds`` 是安全阈值：只删修改时间早于该秒数的 tmp，避免误伤**正在写**的
    文件（同名 pid tmp 由本进程独占，跨进程并发不属本函数职责）。
    """
    now = time.time()
    n = 0
    for f in Path(directory).glob("*.tmp"):
        try:
            if now - f.stat().st_mtime < min_age_seconds:
                continue
            f.unlink()
            n += 1
        except OSError:
            pass
    return n
