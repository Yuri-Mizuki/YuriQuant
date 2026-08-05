"""
因子库（Factor Library / Factor Zoo）
====================================

把挖掘/合成产出的因子**持久化**为一个可查询、可对比、可迭代的因子库。

解决的问题（用户需求）：
1. 每个因子都能快速查看**某一段时间**的回测结果 —— 通过预存 IC 序列与多套
   canonical 回测的净值/日收益，``evaluate_period`` 按时间段切片、重算指标，秒级返回。
2. 所有因子可以**统一对比**绩效指标、挑选好因子 —— ``compare`` 按任意指标/配置排序。
3. 复合因子**入库并参与下一轮迭代** —— 复合因子记录 ``parents`` 血缘；``load_library_features``
   把它们作为新特征喂给下一轮挖掘（``mine_factors --use-library``）。

持久化结构（root 默认 e:/data/factor_library/）：
    registry.csv              # 所有因子的元数据 + 各配置绩效指标
    panels/<slug>.parquet     # 预计算因子面板 (date × code)，已截面标准化
    evals/<slug>.parquet      # ic 序列 + 各 canonical 配置的 equity/dret（按日期索引）

按数据集分库根（dataset）：
    传入 dataset=<name> 时，上述三件套落在 root/<dataset>/ 子目录下，互不串扰。
    FactorLibrary.list_datasets() 可列出全部数据集；CLI 用 --dataset 指定，
    mine/synthesize 用 --library-dataset（不传则自动推导：真实→<指数>_<年>，mock→mock）。

canonical 回测配置（注册时一次性算好，之后查看/对比都不重算）：
    ls_M  : TopK 多空, k=30, 月调仓
    lo_M  : TopK 纯多, k=30, 月调仓
    ls_W  : TopK 多空, k=30, 周调仓
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, VectorBacktest
from backtest.metrics import calc_all_metrics
from research.factor_analysis import calc_ic_series, calc_ir, calc_ic_decay, factor_autocorr
from strategy.examples import TopKLongOnly, TopKLongShort

log = logging.getLogger("factor_library")


# ===========================================================================
# canonical 回测配置
# ===========================================================================
class _Config:
    key: str          # 列后缀，如 "ls_M"
    label: str        # 人类可读
    strategy: type    # TopKLongShort / TopKLongOnly
    freq: str         # D / W / M
    k: int = 30

    def __init__(self, key, label, strategy, freq, k=30):
        self.key = key
        self.label = label
        self.strategy = strategy
        self.freq = freq
        self.k = k


CANONICAL_CONFIGS: list = [
    _Config("ls_M", "TopK多空·月", TopKLongShort, "M", 30),
    _Config("lo_M", "TopK纯多·月", TopKLongOnly, "M", 30),
    _Config("ls_W", "TopK多空·周", TopKLongShort, "W", 30),
]

_METRIC_COLS = ["annual_return", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate", "avg_turnover"]


def _slug(name: str) -> str:
    """把因子名（可能是长公式）映射为安全的文件名片段。"""
    return hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def _coerce_date(d) -> pd.Timestamp | None:
    if d is None:
        return None
    if isinstance(d, (int, np.integer)):
        s = str(int(d))
        return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
    return pd.Timestamp(str(d))


def _sanitize_dataset(ds: str) -> str:
    """把数据集名映射为安全的子目录名。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(ds).strip())


# ===========================================================================
# 因子库
# ===========================================================================
class FactorLibrary:
    """因子持久化仓库。"""

    def __init__(self, root: str | Path | None = None, dataset: str | None = None):
        if root is None:
            try:
                from config import Config
                root = Config.get().get("factor_library", {}).get("root", "e:/data/factor_library/")
            except Exception:
                root = "e:/data/factor_library/"
        self.base_root = Path(root)
        self.dataset = dataset
        # 按数据集分库根：dataset 给定时落到 base_root/<dataset>/，否则用 legacy 默认库（向后兼容）
        self.root = self.base_root / _sanitize_dataset(dataset) if dataset else self.base_root
        self.panels_dir = self.root / "panels"
        self.evals_dir = self.root / "evals"
        self.panels_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.root / "registry.csv"

    @classmethod
    def list_datasets(cls, root: str | Path | None = None) -> list[str]:
        """列出 base_root 下所有含 registry.csv 的数据集名（按数据集分库根）。"""
        if root is None:
            try:
                from config import Config
                root = Config.get().get("factor_library", {}).get("root", "e:/data/factor_library/")
            except Exception:
                root = "e:/data/factor_library/"
        base = Path(root)
        if not base.exists():
            return []
        return sorted(d.name for d in base.iterdir()
                      if d.is_dir() and (d / "registry.csv").exists())

    # ---- registry IO ----
    def _load_registry(self) -> pd.DataFrame:
        if self._registry_path.exists():
            return pd.read_csv(self._registry_path, dtype={"parents": str})
        cols = [
            "name", "kind", "formula", "source", "dataset", "parents", "created_at",
            "n_dates", "n_codes", "ic_mean", "ic_std", "ic_ir", "t_stat", "t_stat_nw",
            "ic_win_rate", "ic_decay5", "autocorr", "significant",
        ] + [f"{m}_{c.key}" for c in CANONICAL_CONFIGS for m in _METRIC_COLS] + [
            "best_sharpe", "best_config", "panel_path", "eval_path",
        ]
        return pd.DataFrame(columns=cols)

    def _save_registry(self, df: pd.DataFrame) -> None:
        # 写临时文件再原子替换，避免 read-modify-write 并发入库时损坏 CSV
        tmp = self._registry_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(self._registry_path)

    # ---- 注册 ----
    def register(
        self,
        name: str,
        panel: pd.DataFrame,
        returns_panel: pd.DataFrame,
        kind: str = "raw",
        formula: str | None = None,
        parents: list[str] | None = None,
        source: str = "",
        ic_method: str = "spearman",
    ) -> dict:
        """注册一个因子：预计算 IC 序列 + 各 canonical 回测，落盘面板/评估/registry。

        Args:
            name: 因子唯一名（建议用公式字符串）。
            panel: date×code 因子面板（建议已截面标准化）。
            returns_panel: date×code 未来一期收益面板。
            kind: 'raw' | 'composite'。
            formula: 人类可读公式（默认=name）。
            parents: 复合因子的父因子名列表（血缘）。
            source: 来源标注（如 "mining:factor_mining_xxx.csv" / "synthesis:ic_weighted"）。
            ic_method: IC 计算方式。
        Returns:
            该因子的 registry 行（dict）。
        """
        slug = _slug(name)
        panel_path = self.panels_dir / f"{slug}.parquet"
        eval_path = self.evals_dir / f"{slug}.parquet"

        # 1) IC 序列
        ic = calc_ic_series(panel, returns_panel, method=ic_method)
        ic_mean = float(ic.mean()) if ic.notna().any() else float("nan")
        ic_std = float(ic.std()) if ic.notna().any() else float("nan")
        ic_valid = ic.dropna()
        n = len(ic_valid)
        ic_ir = calc_ir(ic) if n >= 2 else 0.0
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
        # Newey-West 自相关稳健显著性（IC 序列强自相关时 OLS t 会虚高；
        # significant 判定基于 NW t，2026-08-03 修复，避免累积伪显著）
        from research.robust_stats import nw_tstat
        t_stat_nw, _se_nw, _lag = nw_tstat(ic_valid) if n >= 2 else (0.0, 0.0, 0)
        ic_win_rate = float((ic_valid > 0).mean()) if n else float("nan")
        significant = bool(abs(t_stat_nw) > 2.0)

        # 1b) IC 衰减 + 截面排名自相关（换手率代理）
        try:
            with np.errstate(all="ignore"):
                _decay = calc_ic_decay(panel, returns_panel, max_lag=10)
            ic_decay5 = float(_decay.get(5, float("nan")))
        except Exception:
            ic_decay5 = float("nan")
        try:
            autocorr = factor_autocorr(panel)
        except Exception:
            autocorr = float("nan")

        # 2) canonical 回测
        eval_cols: dict = {"ic": ic}
        metric_rows: dict = {}
        best_sharpe = -np.inf
        best_config = CANONICAL_CONFIGS[0].key
        for cfg in CANONICAL_CONFIGS:
            bt = VectorBacktest(cfg.strategy(k=cfg.k), rebalance_freq=cfg.freq)
            res = bt.run(panel, returns_panel)
            dret = res.daily_returns
            eval_cols[f"dret_{cfg.key}"] = dret
            eval_cols[f"equity_{cfg.key}"] = res.equity_curve
            m = res.metrics()
            for mc in _METRIC_COLS:
                if mc in m:
                    metric_rows[f"{mc}_{cfg.key}"] = m[mc]
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_config = cfg.key

        # 3) 落盘
        panel.to_parquet(panel_path, compression="snappy")
        eval_df = pd.DataFrame(eval_cols).sort_index()
        eval_df.to_parquet(eval_path, compression="snappy")

        # 4) registry 行
        row = {
            "name": name,
            "kind": kind,
            "formula": formula if formula is not None else name,
            "source": source,
            "dataset": self.dataset or "",
            "parents": "|".join(parents) if parents else "",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_dates": panel.shape[0],
            "n_codes": panel.shape[1],
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "t_stat": t_stat,
            "t_stat_nw": t_stat_nw,
            "ic_win_rate": ic_win_rate,
            "ic_decay5": ic_decay5,
            "autocorr": autocorr,
            "significant": significant,
            **metric_rows,
            "best_sharpe": best_sharpe,
            "best_config": best_config,
            "panel_path": str(panel_path),
            "eval_path": str(eval_path),
        }
        reg = self._load_registry()
        reg = reg[reg["name"] != name]  # 覆盖同名
        new_df = pd.DataFrame([row])
        reg = new_df if reg.empty else pd.concat([reg, new_df], ignore_index=True)
        self._save_registry(reg)
        log.info("因子入库: %s (kind=%s, IC=%.4f, best_sharpe=%.3f @%s)",
                 name, kind, ic_mean, best_sharpe, best_config)
        return row

    # ---- 查询 ----
    def has(self, name: str) -> bool:
        reg = self._load_registry()
        return name in set(reg["name"])

    def list_all(self, kind: str | None = None) -> pd.DataFrame:
        reg = self._load_registry()
        if kind is not None:
            reg = reg[reg["kind"] == kind]
        return reg

    def list_composites(self) -> pd.DataFrame:
        return self.list_all(kind="composite")

    def get_panel(self, name: str) -> pd.DataFrame | None:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return None
        p = Path(hit.iloc[0]["panel_path"])
        return pd.read_parquet(p) if p.exists() else None

    def _load_eval(self, name: str) -> pd.DataFrame | None:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return None
        p = Path(hit.iloc[0]["eval_path"])
        return pd.read_parquet(p) if p.exists() else None

    def load_library_features(self, kind: str | None = None) -> dict:
        """返回 {name: panel}，作为下一轮挖掘的特征集（迭代用）。"""
        reg = self.list_all(kind=kind)
        out: dict = {}
        for _, r in reg.iterrows():
            p = Path(r["panel_path"])
            if p.exists():
                out[r["name"]] = pd.read_parquet(p)
        return out

    # ---- 时间段回测查看 ----
    def evaluate_period(self, name: str, start=None, end=None, config: str = "ls_M") -> dict:
        """切片查看某因子在 [start, end] 的回测绩效（秒级，不重算）。

        Returns:
            dict: 含 metrics、ic 统计量、子区间 series、可直接喂报告的 BacktestResult。
        """
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        if config not in [c.key for c in CANONICAL_CONFIGS]:
            raise ValueError(f"未知 config: {config}，可选 {[c.key for c in CANONICAL_CONFIGS]}")
        s, e = _coerce_date(start), _coerce_date(end)
        sub = eval_df.loc[s:e]

        dret = sub[f"dret_{config}"].dropna()
        ic_sub = sub["ic"].dropna()
        metrics = calc_all_metrics(dret)
        ic_mean = float(ic_sub.mean()) if len(ic_sub) else float("nan")
        ic_std = float(ic_sub.std()) if len(ic_sub) else float("nan")
        ic_win = float((ic_sub > 0).mean()) if len(ic_sub) else float("nan")
        ic_ir = calc_ir(ic_sub) if len(ic_sub) >= 2 else 0.0

        bt = BacktestResult(
            daily_returns=dret,
            weights_history=pd.DataFrame(index=dret.index),
            equity_curve=(1 + dret).cumprod(),
            turnover_series=pd.Series(dtype=float),
            cost_series=pd.Series(dtype=float),
            config={"strategy": config, "period": f"{s} ~ {e}"},
        )
        return {
            "name": name, "config": config,
            "start": str(s) if s is not None else None,
            "end": str(e) if e is not None else None,
            "metrics": metrics,
            "ic_mean": ic_mean, "ic_std": ic_std, "ic_win_rate": ic_win, "ic_ir": ic_ir,
            "n_days": len(dret),
            "ic_series": ic_sub,
            "equity_curve": bt.equity_curve,
            "daily_returns": dret,
            "backtest_result": bt,
        }

    def reconstruct_backtest(self, name: str, config: str = "ls_M") -> BacktestResult:
        """从存储的序列重建完整 BacktestResult（用于报告/绘图）。"""
        eval_df = self._load_eval(name)
        if eval_df is None:
            raise KeyError(f"因子不存在: {name}")
        dret = eval_df[f"dret_{config}"].dropna()
        return BacktestResult(
            daily_returns=dret,
            weights_history=pd.DataFrame(index=dret.index),
            equity_curve=eval_df[f"equity_{config}"].reindex(dret.index),
            turnover_series=pd.Series(dtype=float),
            cost_series=pd.Series(dtype=float),
            config={"strategy": config},
        )

    # ---- 统一对比 ----
    def compare(
        self,
        metric: str = "ir",
        config: str | None = None,
        ascending: bool = False,
        kind: str | None = None,
        topn: int | None = None,
    ) -> pd.DataFrame:
        """所有因子按统一指标排序，方便挑好因子。

        默认按 **IR（ic_ir，业界统一主轴）** 排序；IR 尺度无关、跨因子可比。
        metric 也支持 ic_mean / sharpe / annual_return / max_drawdown / calmar /
        avg_turnover 等（sharpe/return/calmar/sortino 默认取 best_<metric>）。

        Args:
            metric: 排序指标。
            config: 指定配置列（如 "ls_M"）；为 None 对 sharpe 等用 best_<metric>。
            kind: 仅比较某类（raw/composite）。
            topn: 仅返回前 N 行。
        """
        reg = self.list_all(kind=kind).copy()
        if reg.empty:
            return reg
        if metric in ("ir", "ic_ir"):
            col = "ic_ir"
        elif config is not None:
            col = f"{metric}_{config}"
            if col not in reg.columns:
                raise ValueError(f"列不存在: {col}")
        elif metric in ("sharpe", "annual_return", "calmar", "sortino"):
            col = f"best_{metric}" if f"best_{metric}" in reg.columns else f"{metric}_{CANONICAL_CONFIGS[0].key}"
        else:
            col = f"{metric}_{CANONICAL_CONFIGS[0].key}"
        reg = reg.sort_values(col, ascending=ascending).reset_index(drop=True)
        reg["_sort_col"] = col
        if topn is not None:
            reg = reg.head(topn)
        return reg

    # ---- 删除 ----
    def delete(self, name: str) -> bool:
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return False
        r = hit.iloc[0]
        for p in (r.get("panel_path"), r.get("eval_path")):
            if p and Path(p).exists():
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    # 文件可能被占用/沙箱回收站不可用：保留文件，仅从 registry 移除
                    log.warning("因子文件删除失败（保留文件）: %s", p)
        reg = reg[reg["name"] != name]
        self._save_registry(reg)
        log.info("已删除因子: %s", name)
        return True

    # ---- 血缘 ----
    def lineage(self, name: str) -> list:
        """返回该因子的父因子链（复合因子才有）。"""
        reg = self._load_registry()
        hit = reg[reg["name"] == name]
        if hit.empty:
            return []
        parents = hit.iloc[0].get("parents", "")
        return [p for p in str(parents).split("|") if p]
