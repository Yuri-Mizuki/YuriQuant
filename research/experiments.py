"""
实验管理（Experiment Management）
================================

轻量实验记录：把每次挖掘 / 合成 / 回测 / 归因 / 数据更新跑完的**结果档案**
统一追加到 ``reports/experiments.csv``，回答三个问题：

1. 这个结果是什么时候、用什么命令/参数跑出来的？（可复现）
2. 它用的数据是哪一版？（数据指纹，与 cache 绑定）
3. 关键指标是多少、产物文件在哪？（可回溯）

设计原则（个人研究系统，不做外部服务）：
- 单 CSV 存储，追加写，无 DB / 无 UI / 无 MLflow。
- 复杂字段（params / metrics）用 JSON 字符串存列，查询时反序列化。
- ``record()`` 幂等追加；``list()`` / ``latest()`` / ``query()`` 只读查询。

字段：
    run_id / timestamp / kind / command / params(json) / data_fingerprint /
    result_path / metrics(json) / note
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

__all__ = ["Experiments", "default_experiments_path"]

DEFAULT_PATH = Path("reports") / "experiments.csv"

_COLUMNS = [
    "run_id", "timestamp", "kind", "command", "params",
    "data_fingerprint", "result_path", "metrics", "note",
]


def default_experiments_path() -> Path:
    return DEFAULT_PATH


class Experiments:
    """轻量实验记录（CSV 追加写）。"""

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else default_experiments_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 写入 ----
    def record(
        self,
        kind: str,
        command: str = "",
        params: dict | None = None,
        data_fingerprint: str = "",
        result_path: str = "",
        metrics: dict | None = None,
        note: str = "",
        run_id: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        """追加一条实验记录，返回 run_id。

        Args:
            kind: 实验类型（mining / gp / synthesis / backtest / attribution /
                  data_update ...），用于 list/query 过滤。
            command: 完整命令行（可复现）。
            params: 关键参数 dict（JSON 落盘）。
            data_fingerprint: 数据指纹（``DataCache.get_fingerprint()``）。
            result_path: 主要产物文件路径。
            metrics: 关键指标 dict（如 ic_mean/sharpe/...，JSON 落盘）。
            note: 备注。
            run_id: 缺省自动生成（时间戳 + 随机后缀）。
        """
        if run_id is None:
            run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "kind": kind,
            "command": command,
            "params": json.dumps(params, ensure_ascii=False) if params else "",
            "data_fingerprint": data_fingerprint,
            "result_path": result_path,
            "metrics": json.dumps(metrics, ensure_ascii=False) if metrics else "",
            "note": note,
        }
        df = self._load()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._save(df)
        return run_id

    # ---- 查询 ----
    def list(self, kind: str | None = None, limit: int | None = None) -> pd.DataFrame:
        """全部记录（可 kind 过滤，时间倒序，limit 截断）。metrics/params 解析为 dict 列。"""
        df = self._load()
        if df.empty:
            return df
        if kind is not None:
            df = df[df["kind"] == kind]
        # run_id 带纳秒后缀，与 timestamp 双键排序保证同一秒内后写入的在前
        df = df.sort_values(["timestamp", "run_id"], ascending=False).reset_index(drop=True)
        if limit is not None:
            df = df.head(limit)
        df["params"] = df["params"].apply(_try_json)
        df["metrics"] = df["metrics"].apply(_try_json)
        return df

    def latest(self, kind: str | None = None) -> pd.Series | None:
        """该类型最近一条记录（无则 None）。"""
        df = self.list(kind=kind, limit=1)
        if df.empty:
            return None
        return df.iloc[0]

    def query(self, kind: str | None = None, **filters: Any) -> pd.DataFrame:
        """按列值过滤（metrics 内的 key 支持 ``metrics__<key>`` 前缀匹配）。"""
        df = self.list(kind=kind)
        if df.empty:
            return df
        for key, val in filters.items():
            if key.startswith("metrics__"):
                k = key[len("metrics__"):]
                df = df[df["metrics"].apply(lambda m: (m or {}).get(k) == val)]
            else:
                df = df[df[key] == val]
        return df.reset_index(drop=True)

    # ---- 内部 ----
    def _load(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=_COLUMNS)
        df = pd.read_csv(self._path, dtype=str)
        for c in _COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df

    def _save(self, df: pd.DataFrame) -> None:
        df.to_csv(self._path, index=False, encoding="utf-8-sig")

    @property
    def path(self) -> Path:
        return self._path


def _try_json(s: Any) -> Any:
    if isinstance(s, str) and s:
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def record_experiment(**kwargs) -> str:
    """便捷入口：写默认 experiments.csv，返回 run_id。"""
    return Experiments().record(**kwargs)
