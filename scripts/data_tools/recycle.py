"""走 Windows 回收站删除（**不永久删除**），供维护类脚本复用。

仓库里的删除动作统一走这里，理由见 ``AGENTS.md`` / 用户约定：
删除可恢复、批量小步、先快照。Windows 侧用 ``SHFileOperationW``
（``FOF_ALLOWUNDO``）；非 Windows 平台直接报错，避免悄悄退化成 ``os.remove``。

命令行
------
    python -m scripts.data_tools.recycle <file1> <file2> ...   # 每批 ≤10
    python -m scripts.data_tools.recycle --list-file <manifest.txt>

库调用
------
    from scripts.data_tools.recycle import recycle
    rc = recycle([str(p) for p in paths])   # 0 = 全成功
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes

FO_DELETE = 3
FOF_ALLOWUNDO = 0x0040
FOF_NOCONFIRMATION = 0x0010
FOF_SILENT = 0x0004
FOF_NOERRORUI = 0x0400


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", ctypes.c_uint16),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


def recycle(paths: list[str]) -> int:
    """把 paths 送进回收站。返回非 0 表示有失败（此时不要当作已删除）。"""
    if sys.platform != "win32":
        raise RuntimeError("recycle() 只支持 Windows 回收站；其他平台请显式选择删除方式")

    # pFrom 必须是双 null 结尾的多字符串
    joined = "\0".join(os.path.abspath(p) for p in paths) + "\0\0"
    op = SHFILEOPSTRUCTW()
    op.hwnd = None
    op.wFunc = FO_DELETE
    op.pFrom = joined
    op.pTo = None
    op.fFlags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI
    res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    if res != 0:
        print(f"SHFileOperationW 返回 {res}", file=sys.stderr)
    return res


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--list-file":
        with open(args[1], encoding="utf-8") as fh:
            targets = [ln.strip() for ln in fh if ln.strip()]
    else:
        targets = args

    existing = [t for t in targets if os.path.exists(t)]
    missing = [t for t in targets if not os.path.exists(t)]
    print(f"待回收 {len(targets)} 个；存在 {len(existing)}，缺失 {len(missing)}")
    for m in missing:
        print(f"  MISSING {m}")
    if not existing:
        return 1

    # 分批，每批 <=10 个（遵守小批量原则）
    failed = []
    for i in range(0, len(existing), 10):
        batch = existing[i:i + 10]
        recycle(batch)
        for b in batch:
            ok = not os.path.exists(b)
            print(f"  {'OK  ' if ok else 'FAIL'} {b}")
            if not ok:
                failed.append(b)
    print(f"完成：成功 {len(existing) - len(failed)}，失败 {len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
