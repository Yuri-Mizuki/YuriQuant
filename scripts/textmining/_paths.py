# -*- coding: utf-8 -*-
"""textmining 输出路径约定。

统一由这里决定 reports/textmining 的分层布局：
    reports/textmining/{task}/{kind}/...
  task = fadt | sue   （研报文本 / 盈余事件文本）
  kind = samples | features | models | reports

脚本用法：把原来的 `OUT_DIR = TOOT / "reports" / "textmining"` 替换为
    from scripts.textmining._paths import Out
    OUT_DIR = Out("fadt")     # 或该任务对应的默认 task
之后 `OUT_DIR / f"xxx.png"` 即按文件名自动路由到 reports/textmining/{task}/{kind}。
读写共用同一路由，保证一致。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TM = ROOT / "reports" / "textmining"
KINDS = ("samples", "features", "models", "reports")


def task_root(task: str) -> Path:
    """返回某个研究的根目录并确保存在。"""
    d = TM / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def out(task: str, kind: str, *parts: str) -> Path:
    """返回 reports/textmining/{task}/{kind}/{parts...} ，确保目录存在。"""
    if kind not in KINDS:
        raise ValueError(f"unknown kind={kind!r}, expect one of {KINDS}")
    d = TM / task / kind
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*parts)


class Out:
    """路径代理：`Out(task) / filename` 按文件名 --> (task, kind) 自动分层。

    task 优先从文件名前缀推断（fadt/sue），以便一个脚本可同时处理两类文件；
    kind 由扩展名与关键字推断：joblib->models，txt/log/md->reports，
    *_samples*.parquet->samples，其余 parquet/无扩展->features。
    若文件名既非 fadt 也非 sue 前缀，则回退到构造时传入的默认 task。
    """

    def __init__(self, default_task: str):
        self._task = default_task

    def _target(self, name) -> Path:
        sn = str(name)
        low = sn.lower()
        if low.startswith("fadt"):
            task = "fadt"
        elif low.startswith("sue"):
            task = "sue"
        else:
            task = self._task
        if low.endswith(".joblib"):
            kind = "models"
        elif low.endswith((".txt", ".log", ".md")):
            kind = "reports"
        elif low.endswith(".parquet"):
            kind = "samples" if "_samples" in low else "features"
        elif low.endswith("_parts"):  # 临时分片目录
            kind = "features"
        else:
            kind = "features"
        d = TM / task / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    def __truediv__(self, name) -> Path:
        return self._target(name)