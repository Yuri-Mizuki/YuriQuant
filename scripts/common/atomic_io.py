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
import sys
import time
from pathlib import Path

import pandas as pd

#: 原子改名阶段的重试与退避基数（秒）：0.4 / 0.8 / 1.6 / 3.2 ...，累计 ~6s。
#: 这一阶段只为扛「瞬时」占用而设；改名被**持续**拒绝在本机是常态（文件守护拦删除），
#: 所以不必给更长预算——尽快降级才是不浪费算力的做法。
RETRIES = 5
BACKOFF = 0.4

#: 降级就地覆写阶段的重试预算（秒上限，不是次数）：每次退避不超过 ``FALLBACK_MAX_SLEEP``，
#: 总等待不超过 ``FALLBACK_BUDGET``。
#:
#: 🚨 2026-11-05 两连崩的根因：本机 Defender 实时防护开着，每个 57MB 的
#: ``panels_neu_fundind/*.parquet`` 写完都要被扫一遍；6 个 worker 并发写时扫描队列
#: 积压，文件占用时长**远超**原定的 ~6s 预算（实测 5 次重试后仍被拒，两次崩在
#: 不同的随机文件上：``alpha101_042`` / ``alpha101_005``）。改名被拒本身正是
#: 「有人正打开着这个文件」的信号，紧随其后的覆写几乎必然撞锁。
FALLBACK_BUDGET = 40.0
FALLBACK_MAX_SLEEP = 4.0


def _fallback(df: pd.DataFrame, path: Path, last_exc: Exception | None,
              **kwargs) -> None:
    """降级就地覆写，按时间预算重试（Defender 扫描窗口可长达数十秒）。"""
    waited = 0.0
    sleep_s = BACKOFF
    while waited < FALLBACK_BUDGET:
        try:
            df.to_parquet(path, **kwargs)
            return
        except PermissionError as exc:
            last_exc = exc
            time.sleep(sleep_s)
            waited += sleep_s
            sleep_s = min(sleep_s * 2, FALLBACK_MAX_SLEEP)
    raise PermissionError(
        f"就地覆写 {waited:.0f}s 内重试仍被拒：{path}"
        f"（最后一次：{last_exc}）—— 通常是杀软实时扫描占用，"
        f"可把 {path.parent} 加入扫描排除项") from last_exc


def write_parquet_atomic(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    """把 ``df`` 写到 ``path``：先写同目录 pid 专属 tmp，再原子改名。

    ``kwargs`` 透传给 ``DataFrame.to_parquet``。

    🚨 **降级路径**：若改名重试耗尽仍被拒（本环境实测：文件守护拦删除，批量下**几乎
    每次**都会被拒），则退回**就地覆写**（``to_parquet`` 直接写目标，不产生任何删除
    操作），并按**时间预算**重试（见 ``FALLBACK_BUDGET``，Defender 扫描 57MB
    parquet 的占用窗口可达数十秒）。代价 = 丢掉"进程被杀不留截断文件"的保证，换来
    "整批不因单点而中断"。仓库既有同款模式：`alla_daily_rank._atomic_or_direct`
    （`MEMORY-engineering.md` §B4：「写盘须原子优先 + 占用降级」）。
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
    print(f"[atomic_io] 原子改名重试 {RETRIES} 次仍被拒（{last_exc}）⇒ 降级就地覆写 "
          f"{path.name}", file=sys.stderr, flush=True)
    _fallback(df, path, last_exc, **kwargs)


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
