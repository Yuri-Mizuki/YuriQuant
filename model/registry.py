"""
模型注册表 —— 承接聚宽 AI 投研流程 02 模型层「模型设计 → 训练 → 评价 → 迭代」。

与因子库 FactorLibrary 同风格：模型元数据 + 血缘 + 迭代记录持久化。
每个模型记录：

- model_id      唯一 ID（time_ns，防同秒排序不稳，与 experiments.py 一致）
- created_at    注册时间
- kind          模型类型（ml_stacking / 预测 / 合成……）
- name          人类可读名称（同名再注册 = 新版本，即"模型迭代"）
- spec          设计规格 JSON（模型设计阶段的产物：方法、超参、特征）
- fingerprint   训练数据指纹（DataCache.get_fingerprint，绑定数据版本）
- train_begin / train_end  训练区间
- metrics       评价指标 JSON（模型评价阶段的产物）
- parents       血缘：参与构建该模型的因子/父模型（可追溯，供迭代复用）

文件：reports/models/registry.csv（原子写）+ 可选 spec 明细 JSON。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

__all__ = ["ModelRegistry", "default_model_root"]

MODEL_COLUMNS = [
    "model_id", "created_at", "kind", "name", "spec",
    "fingerprint", "train_begin", "train_end", "metrics", "parents", "note",
]


def default_model_root() -> Path:
    return Path("reports") / "models"


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class ModelRegistry:
    """模型注册表：模型设计/训练/评价/迭代的统一落点。

    Usage::

        reg = ModelRegistry()
        mid = reg.register(
            name="stacking_ridge_v2", kind="ml_stacking",
            spec={"method": "ridge", "alpha": 1.0, "components": ["a", "b"]},
            metrics={"ic_mean": 0.039, "ic_ir": 1.2},
            parents=["factor:a", "factor:b"],
        )
        reg.compare()   # 所有模型按指标排名
    """

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else default_model_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "registry.csv"

    # ------------------------------------------------------------------
    def _read(self) -> pd.DataFrame:
        if not self._path.exists():
            return pd.DataFrame(columns=MODEL_COLUMNS)
        return pd.read_csv(self._path, dtype={"model_id": str})

    def _write(self, df: pd.DataFrame) -> None:
        tmp = self._path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(self._path)  # 原子替换，避免并发写半文件

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        kind: str = "ml",
        spec: Mapping[str, Any] | None = None,
        fingerprint: str | None = None,
        train_begin: int | None = None,
        train_end: int | None = None,
        metrics: Mapping[str, Any] | None = None,
        parents: Sequence[str] | None = None,
        note: str = "",
        model_id: str | None = None,
    ) -> str:
        """注册一个模型（同名重复注册 = 新版本，天然支持迭代）。"""
        mid = model_id or f"{time.time_ns()}"
        row = {
            "model_id": mid,
            "created_at": _now_str(),
            "kind": kind,
            "name": name,
            "spec": json.dumps(spec, ensure_ascii=False, default=str) if spec else "",
            "fingerprint": fingerprint or "",
            "train_begin": train_begin,
            "train_end": train_end,
            "metrics": json.dumps(metrics, ensure_ascii=False, default=str) if metrics else "",
            "parents": ",".join(parents) if parents else "",
            "note": note,
        }
        df = self._read()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._write(df)
        return mid

    # ------------------------------------------------------------------
    def list(self, name: str | None = None, kind: str | None = None) -> pd.DataFrame:
        """列出模型（可过滤 name/kind），按注册时间排序。"""
        df = self._read()
        if df.empty:
            return df
        if name is not None:
            df = df[df["name"] == name]
        if kind is not None:
            df = df[df["kind"] == kind]
        return df.sort_values("created_at").reset_index(drop=True)

    def latest(self, name: str | None = None) -> pd.Series | None:
        """最新一个模型（迭代链的当前版本）。"""
        df = self.list(name=name)
        if df.empty:
            return None
        return df.iloc[-1]

    def view(self, model_id: str) -> dict:
        """查看单条模型记录（spec/metrics 解析回 dict）。"""
        df = self._read()
        hit = df[df["model_id"] == model_id]
        if hit.empty:
            raise KeyError(f"模型不存在: {model_id}")
        row = hit.iloc[0].to_dict()
        for key in ("spec", "metrics"):
            if isinstance(row.get(key), str) and row[key]:
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
        return row

    def compare(self, metric: str = "ic_mean", top: int | None = None) -> pd.DataFrame:
        """所有模型按指标排名（模型评价的横向对比）。"""
        df = self._read()
        if df.empty:
            return df
        rows = []
        for _, r in df.iterrows():
            m = {}
            if isinstance(r.get("metrics"), str) and r["metrics"]:
                try:
                    m = json.loads(r["metrics"])
                except json.JSONDecodeError:
                    m = {}
            row = {"model_id": r["model_id"], "name": r["name"],
                   "kind": r["kind"], "created_at": r["created_at"],
                   "parents": r.get("parents", "")}
            row.update(m)
            rows.append(row)
        out = pd.DataFrame(rows)
        if metric in out.columns:
            out = out.sort_values(metric, ascending=False, na_position="last")
        else:
            out = out.sort_values("created_at")
        if top is not None:
            out = out.head(top)
        return out.reset_index(drop=True)

    def delete(self, model_id: str) -> bool:
        """删除一条模型记录。"""
        df = self._read()
        if df.empty or (df["model_id"] == model_id).sum() == 0:
            return False
        self._write(df[df["model_id"] != model_id])
        return True
