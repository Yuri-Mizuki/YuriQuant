"""监控账本 —— 快照 / 告警 append-only 落盘（同 run_date 重跑幂等覆盖）。

文件（ledger_root 下）：
- snapshots.csv   每因子一行 × 每次运行（趋势可查：history() 取某因子指标时间序列）
- alerts.csv      每条告警一行 × 每次运行

幂等语义：同一 run_date 重复 append 时先剔除旧行再追加 —— 监控进程崩溃后
重跑不会产生重复记录（生产调度安全）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class MonitoringLedger:
    def __init__(self, root: str | Path = "reports/monitoring"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def snapshots_path(self) -> Path:
        return self.root / "snapshots.csv"

    @property
    def alerts_path(self) -> Path:
        return self.root / "alerts.csv"

    def _append(self, path: Path, df: pd.DataFrame, run_date: str) -> None:
        if df.empty:
            return
        old = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if not old.empty and "run_date" in old.columns:
            old = old[old["run_date"] != run_date]
        out = pd.concat([old, df], ignore_index=True)
        tmp = path.with_suffix(".csv.tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(path)

    def append_snapshots(self, df: pd.DataFrame, run_date: str) -> None:
        self._append(self.snapshots_path, df, run_date)

    def append_alerts(self, df: pd.DataFrame, run_date: str) -> None:
        self._append(self.alerts_path, df, run_date)

    def load_snapshots(self) -> pd.DataFrame:
        return pd.read_csv(self.snapshots_path) if self.snapshots_path.exists() else pd.DataFrame()

    def load_alerts(self) -> pd.DataFrame:
        return pd.read_csv(self.alerts_path) if self.alerts_path.exists() else pd.DataFrame()

    def history(self, name: str, metric: str = "ic_mean_recent") -> pd.DataFrame:
        """单因子跨运行趋势（监控的监控：IC 漂移是否持续恶化）。"""
        snap = self.load_snapshots()
        if snap.empty or "name" not in snap.columns:
            return pd.DataFrame()
        sub = snap[snap["name"] == name]
        if metric not in sub.columns:
            return pd.DataFrame()
        return sub[["run_date", metric]].dropna().reset_index(drop=True)
